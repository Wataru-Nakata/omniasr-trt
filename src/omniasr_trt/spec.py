"""Greedy decoding sped up by drafts (speculative decoding with a draft text, e.g. a CTC model's transcript).

Each step feeds a row's newest token and the next ``draft_tokens`` proposed tokens, at the positions after it, and
keeps the proposed tokens the greedy search would have produced itself (each one equal to the argmax after its
predecessor) plus the argmax after the last kept one. A step therefore yields 1 to ``draft_tokens + 1`` tokens,
which are the greedy search's whatever was proposed: a wrong proposal only costs steps. The step reads the weights
once for all of a row's tokens and its keys once for all of its queries (``kernels.draft_attention``), so it costs
little more than a one-token step. In bf16 the logits round differently from a one-token step's (as at another
batch size).

The proposal is found again before every step, from two sources:

* the draft: the draft position whose preceding tokens match the newest generated ones best (up to ``NGRAM``),
  searched around where the previous step left off; the farther from there, the longer the match must be, so the
  draft may miss, add or change characters without the search jumping to a common character far away;
* the row's own tokens (prompt lookup): the latest earlier place where its newest tokens occurred, continued as
  they did there -- what the model does in a repetition loop until the repetition check stops it, which no draft
  foresees. Used when it matches more tokens (and at least ``SELF_MIN``) than the draft does.

The rules of the greedy search are kept: EOS or the length limit inside a step end the row there, and the
repetition check runs on the host on the same tokens.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from . import kernels
from .decode import LIMIT, GreedyDecoder
from .llm import EOS
from .modeling import LLMModel

DRAFT_CAPACITY = 1024  # draft tokens kept per row (a 15 s segment's CTC text is rarely longer than 400)
NGRAM = 8  # newest tokens matched against the draft / the row's own tokens
BEHIND, AHEAD = 64, 160  # draft positions searched before / after the expected one
# Matching tokens a draft position needs at each distance from the expected one (<= 1, <= 4, <= 12, <= 40, beyond).
REACH = ((1, 1), (4, 2), (12, 4), (40, 6))
SELF_WINDOW, SELF_MIN = 256, 3  # own positions searched back, matching tokens needed


class SpecDecoder(GreedyDecoder):
    def __init__(self, model: LLMModel, *, draft_tokens: int = 4, max_batch: int = 8, sync_every: int = 8,
                 **options):
        super().__init__(model, max_batch=max_batch, sync_every=sync_every, **options)
        if not self.fused:
            raise RuntimeError("speculative decoding needs the Triton kernels (CUDA)")
        self.draft_tokens = draft_tokens
        self.step_tokens = draft_tokens + 1
        b, dev = max_batch, self.device
        self.draft = torch.full((b, DRAFT_CAPACITY), -1, device=dev, dtype=torch.int32)
        self.draft_len = torch.zeros(b, device=dev, dtype=torch.long)
        self.next = torch.zeros(b, device=dev, dtype=torch.long)  # draft position expected after ``tok``
        self.first = torch.zeros(b, device=dev, dtype=torch.long)  # position of the segment's first token
        self.start = torch.zeros(b, device=dev, dtype=torch.long)  # draft position of a step's first proposal
        # The transcriber's: a stream's draft for its current context, (token ids, draft position of the first
        # generated token), or None.
        self.drafts = lambda stream: None

    def state(self) -> list[Tensor]:
        return super().state() + [self.next]

    def filled(self, slots: list[int], streams: list[int]) -> None:
        for s, j in zip(slots, streams, strict=True):
            draft = self.drafts(j)
            tokens, start = draft if draft is not None else ([], 0)
            tokens = tokens[:DRAFT_CAPACITY]
            self.draft[s] = -1
            if tokens:
                self.draft[s, : len(tokens)] = torch.as_tensor(tokens, dtype=torch.int32).to(self.device)
            self.draft_len[s] = len(tokens)
            self.next[s] = start + 1
            self.first[s] = self.last[s]

    def propose(self, b: int) -> tuple[Tensor, Tensor]:
        """Per row the next ``draft_tokens`` proposed tokens [B, k] and whether they came from the draft [B]."""
        return kernels.propose(self.out[:b], self.draft[:b], self.draft_len[:b], self.last[:b], self.first[:b],
                               self.next[:b], self.start[:b], self.draft_tokens, NGRAM, BEHIND, AHEAD, REACH,
                               SELF_WINDOW, SELF_MIN)

    def step(self, b: int, keys: int) -> None:
        m, k, dev = self.m, self.draft_tokens, self.device
        width = k + 1
        tok, last, done, out = self.tok[:b], self.last[:b], self.done[:b], self.out[:b]
        running = ~done
        p = last + running.long()  # input position of ``tok``; finished rows recompute their last slot
        j = torch.arange(width, device=dev)
        proposed, use_draft = self.propose(b)
        tokens = torch.cat([tok[:, None], proposed], 1)  # [B, k + 1]
        # Finished rows write all their tokens at their last position (nothing is kept); clamped positions are past
        # the length limit, where nothing is kept either.
        pos = torch.where(running[:, None], p[:, None] + j, last[:, None]).clamp(max=self.capacity - 1)
        slots = torch.arange(b, device=dev).repeat_interleave(width)
        lengths = torch.where(running, p + 1, 0)  # query i of a row sees lengths + i keys
        scale = self.head_dim**-0.5
        h = m.text_frontend.weight[tokens.view(-1)]  # [B * (k + 1), D]
        for i, layer in enumerate(m.llama_decoder.layers):
            qkv_w, out_w, gi_w, ffo_w = self.layers[i]
            qkv = layer.self_attn_layer_norm(h) @ qkv_w.t()
            q = kernels.rope_kv(qkv, m.rope, pos.view(-1), self.k[i], self.v[i], self.heads, slots)
            o = kernels.draft_attention(q, self.k[i], self.v[i], lengths, width, keys, scale).reshape(b * width, -1)
            h = torch.addmm(h, o, out_w.t())
            gate, inner = (layer.ffn_layer_norm(h) @ gi_w.t()).chunk(2, -1)
            h = torch.addmm(h, inner * F.silu(gate), ffo_w.t())
        logits = m.final_proj(m.llama_decoder.layer_norm(h))
        nxt = F.log_softmax(logits, dim=-1).argmax(-1).view(b, width)  # greedy token after each fed one
        kernels.accept(tokens, nxt, out, tok, last, done, self.next[:b], self.start[:b], use_draft, LIMIT, EOS)
