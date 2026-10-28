"""Batched beam search of the omniASR LLM decoder: omnilingual-asr's search at batch 1, for every input.

Per input (a "group" of ``beams`` rows), as ``LLMTranscriber.beam_reference``: at the first step only one beam is
live; every step each row offers ``score + log p`` (``length_norm``: the running mean of the log-probabilities,
chosen by the group's row 0), a finished row only EOS at its frozen score; the best ``beams`` offers of the group
survive and the rows follow their parents; the repetition stop freezes a row at the exact step it fires; the text
is row 0's (the best) when every row has finished or the position cap is reached.

Layout (why it is fast):

* the context (two segments of audio, the previous text, markers: ~1,750 tokens) is prefilled once per input into
  ``kc``/``vc`` [layers, slots, H, C, Dh], shared by its beams;
* the K/V of the token a row feeds at generated position j is written once at ``kg``/``vg`` [layers, rows, H, G, Dh]
  ``(row, j)``; ``ptr[r, j]`` names the row that wrote position j of r's history, so a reorder gathers int32
  pointers (and the token history ``hist``), never K/V;
* attention (``kernels.beam_attention``): the beams of an input read its context in one pass (one tile load for all
  of them), then each row its own generated positions through ``ptr``;
* one step (12 layers, LM head, per-group top-k over beams x vocabulary, the gathers) is one CUDA graph; the host
  looks every ``sync_every`` steps, or at the exact step a group's repetition check is due.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from . import kernels
from .decode import KEY_BUCKET, LIMIT, GreedyDecoder
from .llm import BOS, CHECK_EVERY, CHECK_RATIO, CHECK_WINDOW, EOS, MAX_GEN, PAD, compression_ratio
from .modeling import LLMModel, apply_rope


def bucket(n: int) -> int:
    return -(-max(n, 1) // KEY_BUCKET) * KEY_BUCKET


class BeamDecoder(GreedyDecoder):
    """Beam search with up to ``max_batch`` inputs in flight (``max_batch * beams`` rows)."""

    def __init__(self, model: LLMModel, *, beams: int, length_norm: bool = False, max_batch: int = 8,
                 sync_every: int = 16, graphs: bool = True, reserve: int = 512, fused: bool = True):
        super().__init__(model, max_batch=max_batch * beams, sync_every=sync_every, graphs=graphs, reserve=reserve,
                         fused=fused)
        self.beams = beams
        self.length_norm = length_norm
        self.slots = max_batch
        r, s, dev = max_batch * beams, max_batch, self.device
        # Scores in fp32 as the reference keeps them (its candidates are fp32 + log p: fp64 for an fp64 model).
        self.score = torch.zeros(r, device=dev, dtype=torch.promote_types(torch.float32, self.dtype))
        self.ended = torch.ones(r, device=dev, dtype=torch.bool)
        self.count = torch.zeros(r, device=dev, dtype=torch.long)  # tokens other than PAD / EOS (length_norm)
        self.group = torch.arange(r, device=dev) // beams
        self.k_step = torch.zeros(s, device=dev, dtype=torch.long)  # tokens generated in the group's segment
        self.ctx = torch.ones(s, device=dev, dtype=torch.long)  # its context length
        self.gdone = torch.ones(s, device=dev, dtype=torch.bool)
        self.ccap = self.gcap = 0
        self.stats["repetition_stops"] = 0

    # ------------------------------------------------------------------ buffers

    def ensure_beam(self, ctx_len: int, gen_len: int) -> None:
        """Context cache for ``ctx_len`` positions per input, generated cache / pointers / history for ``gen_len``."""
        ctx_len, gen_len = bucket(ctx_len), min(bucket(gen_len), bucket(MAX_GEN))
        if ctx_len <= self.ccap and gen_len <= self.gcap:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        layers, heads, dh, dt, dev = len(self.layers), self.heads, self.head_dim, self.dtype, self.device
        if ctx_len > self.ccap:
            kc = torch.zeros(layers, self.slots, heads, ctx_len, dh, device=dev, dtype=dt)
            vc = torch.zeros_like(kc)
            if self.ccap:
                kc[..., : self.ccap, :] = self.kc
                vc[..., : self.ccap, :] = self.vc
            else:
                self.stats["grows"] -= 1  # the first allocation is not a growth
            self.kc, self.vc, self.ccap = kc, vc, ctx_len
            self.stats["grows"] += 1
        if gen_len > self.gcap:
            rows = self.slots * self.beams
            kg = torch.zeros(layers, rows, heads, gen_len, dh, device=dev, dtype=dt)
            vg = torch.zeros_like(kg)
            ptr = torch.zeros(rows, gen_len, device=dev, dtype=torch.int32)
            hist = torch.full((rows, gen_len + 1), PAD, device=dev, dtype=torch.int32)
            if self.gcap:
                kg[..., : self.gcap, :] = self.kg
                vg[..., : self.gcap, :] = self.vg
                ptr[:, : self.gcap] = self.ptr
                hist[:, : self.gcap + 1] = self.hist
            else:
                self.stats["grows"] -= 1
            self.kg, self.vg, self.ptr, self.hist, self.gcap = kg, vg, ptr, hist, gen_len
            self.stats["grows"] += 1
        self.reset_graphs()

    def release(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.kc = self.vc = self.kg = self.vg = self.ptr = self.hist = None
        self.ccap = self.gcap = 0
        self.reset_graphs()

    # ------------------------------------------------------------------ one step

    def attention(self, i: int, q: Tensor, ctx_len: Tensor, gen_len: Tensor, s: int, ckeys: int, gkeys: int):
        scale = self.head_dim**-0.5
        r = s * self.beams
        if self.fused:
            return kernels.beam_attention(q, self.kc[i], self.vc[i], ctx_len, self.kg[i], self.vg[i], self.ptr[:r],
                                          gen_len, self.beams, ckeys, scale)
        kc = self.kc[i, self.group[:r], :, :ckeys]  # [R, H, C, Dh]
        vc = self.vc[i, self.group[:r], :, :ckeys]
        src = self.ptr[:r, :gkeys].long()  # [R, G]
        j = torch.arange(gkeys, device=self.device)
        kg = self.kg[i][src, :, j].transpose(1, 2)  # [R, G, H, Dh] -> [R, H, G, Dh]
        vg = self.vg[i][src, :, j].transpose(1, 2)
        k, v = torch.cat([kc, kg], 2), torch.cat([vc, vg], 2)
        allowed = torch.cat([torch.arange(ckeys, device=self.device)[None] < ctx_len[self.group[:r], None],
                             j[None] < gen_len[:, None]], 1)  # [R, C + G]
        scores = (q[:, :, None, :] @ k.transpose(-1, -2)).float() * scale  # [R, H, 1, C + G]
        scores = scores.masked_fill(~allowed[:, None, None], float("-inf"))
        weights = torch.softmax(scores, dim=-1).nan_to_num(0.0)  # rows of finished inputs attend to nothing
        return (weights.to(v.dtype) @ v)[:, :, 0]

    def step_beam(self, s: int, ckeys: int, gkeys: int) -> None:
        """In place on groups 0..s-1 (rows 0..s*N-1); finished groups are left as they are."""
        m, n = self.m, self.beams
        r = s * n
        tok, score, ended, count = self.tok[:r], self.score[:r], self.ended[:r], self.count[:r]
        k_step, ctx, gdone = self.k_step[:s], self.ctx[:s], self.gdone[:s]
        hist, ptr, group = self.hist[:r], self.ptr[:r], self.group[:r]
        rows = torch.arange(r, device=self.device)
        live = ~gdone[group]
        k = k_step[group]  # >= 1: the token in ``tok`` was chosen at step k - 1
        pos = ctx[group] - 1 + k  # its input position
        j = (k - 1).clamp(min=0)  # its generated cache position
        ctx_len = torch.where(gdone, 0, ctx)
        gen_len = torch.where(live, k, 0)
        # The token fed now is written at (row, j), and this step's attention already reads it through ``ptr``; the
        # rows then take their parents' pointers, which carry it on.
        ptr.scatter_(1, j[:, None], torch.where(live, rows, ptr.gather(1, j[:, None])[:, 0].long())[:, None]
                     .to(ptr.dtype))
        h = m.text_frontend.weight[tok]
        for i, layer in enumerate(m.llama_decoder.layers):
            qkv_w, out_w, gi_w, ffo_w = self.layers[i]
            qkv = layer.self_attn_layer_norm(h) @ qkv_w.t()
            if self.fused:
                q = kernels.rope_kv(qkv, m.rope, pos, self.kg[i], self.vg[i], self.heads, cache_pos=j)
            else:
                q, kk, vv = qkv.view(r, 3, self.heads, self.head_dim).unbind(1)
                cs = m.rope[pos][:, None]
                q = apply_rope(q, cs)
                self.kg[i, rows, :, j] = apply_rope(kk, cs)
                self.vg[i, rows, :, j] = vv
            o = self.attention(i, q, ctx_len, gen_len, s, ckeys, gkeys).reshape(r, -1)
            h = torch.addmm(h, o, out_w.t())
            gate, inner = (layer.ffn_layer_norm(h) @ gi_w.t()).chunk(2, -1)
            h = torch.addmm(h, inner * F.silu(gate), ffo_w.t())
        logp = F.log_softmax(m.final_proj(m.llama_decoder.layer_norm(h)), dim=-1)  # [R, V], model dtype
        vocab = logp.size(-1)
        if self.length_norm:
            first = count[group * n][:, None]  # the group's row 0 chooses, as the reference's row 0 does
            c = count[:, None]
            cand = torch.where(first > 0, (score[:, None] * c + logp) / (c + 1), score[:, None] + logp)
        else:
            cand = score[:, None] + logp
        cand = cand.masked_fill(ended[:, None], float("-inf"))
        cand[:, EOS] = torch.where(ended, score, cand[:, EOS])
        top, pick = cand.view(s, n * vocab).topk(n, dim=-1, sorted=True)  # [S, N]
        parent = (pick // vocab + torch.arange(s, device=self.device)[:, None] * n).view(-1)
        parent = torch.where(live, parent, rows)
        new = torch.where(live, (pick % vocab).view(-1), tok)
        ptr.copy_(ptr[parent])
        hist.copy_(hist[parent])
        p_ended, p_score, p_count = ended[parent], score[parent], count[parent]
        new = torch.where(p_ended, EOS, new)
        hist.scatter_(1, k[:, None], torch.where(live, new, hist.gather(1, k[:, None])[:, 0].long())[:, None]
                      .to(hist.dtype))
        score.copy_(torch.where(live & ~p_ended, top.view(-1), p_score))
        count.copy_(p_count + (live & ~p_ended & (new != PAD) & (new != EOS)).long())
        ended.copy_(torch.where(live, new == EOS, p_ended))
        tok.copy_(new)
        all_ended = ended.view(s, n).all(-1)
        k_step.copy_(k_step + (~gdone).long())
        gdone |= all_ended | (ctx - 1 + k_step - 1 >= LIMIT)

    def beam_runner(self, s: int, ckeys: int, gkeys: int):
        if not self.use_graphs:
            return lambda: self.step_beam(s, ckeys, gkeys)
        key = ("beam", s, ckeys, gkeys)
        if key not in self.graphs:
            torch.cuda.synchronize()
            state = (self.tok, self.score, self.ended, self.count, self.k_step, self.gdone, self.hist, self.ptr)
            saved = [t.clone() for t in state]
            stream = self.capture_stream
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self.step_beam(s, ckeys, gkeys)  # warm-up (the K/V it writes are rewritten identically)
                for t, v in zip(state, saved, strict=True):
                    t.copy_(v)
                graph = torch.cuda.CUDAGraph()
                graph.capture_begin(pool=self.pool)
                self.step_beam(s, ckeys, gkeys)
                graph.capture_end()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self.graphs[key] = graph
            self.stats["captures"] += 1
        return self.graphs[key].replay

    # ------------------------------------------------------------------ streams

    @torch.inference_mode()
    def decode_streams(self, n_streams: int, start, advance) -> None:
        """As ``GreedyDecoder.decode_streams``; ``advance(j, tokens)`` gets the best beam's tokens."""
        n, s = self.beams, min(self.slots, n_streams)
        dev = self.device
        stream = np.full(s, -1)
        ctx_host = np.zeros(s, dtype=np.int64)
        upcoming = iter(range(n_streams))
        normal = None
        self.gdone[:s] = True
        self.k_step[:s] = 1
        self.tok[: s * n] = 0

        def fill(pairs):  # [(slot, stream, context)]
            nonlocal normal
            k_live = self.k_step[:s].cpu().numpy()
            busy = [int(k_live[g]) for g in range(s) if stream[g] >= 0 and g not in {p[0] for p in pairs}]
            self.ensure_beam(max(self.context_floor, int(ctx_host.max()), max(c.size(0) for _, _, c in pairs)),
                             max([self.reserve] + [k + self.sync_every + 2 for k in busy]))
            normal = normal or (self.ccap, self.gcap)
            slots = [g for g, _, _ in pairs]
            with self.phase("prefill"):
                logp = self.prefill_into([c for _, _, c in pairs], slots, self.kc, self.vc)  # [b, V]
            top, first = logp.to(self.score.dtype).topk(n, dim=-1)  # the first step: only one live beam
            rows = (torch.tensor(slots, device=dev)[:, None] * n + torch.arange(n, device=dev)).view(-1)
            first = first.view(-1)
            self.tok[rows] = first
            self.score[rows] = top.view(-1)
            self.ended[rows] = first == EOS
            self.count[rows] = ((first != PAD) & (first != EOS)).long()
            self.hist[rows, 0] = first.to(torch.int32)
            idx = torch.tensor(slots, device=dev)
            lens = torch.tensor([c.size(0) for _, _, c in pairs], device=dev)
            self.k_step[idx] = 1
            self.ctx[idx] = lens
            self.gdone[idx] = self.ended[rows].view(-1, n).all(-1) | (lens - 1 >= LIMIT)
            for g, j, c in pairs:
                stream[g], ctx_host[g] = j, c.size(0)

        fill([(g, j, start(j)) for g, j in zip(range(s), upcoming)])
        while True:
            gdone = self.gdone[:s].cpu().numpy()
            finished = [g for g in range(s) if stream[g] >= 0 and gdone[g]]
            if finished:
                k_now = self.k_step[:s].cpu().numpy()
                hist = self.hist[: s * n].cpu().numpy()
                refill = []
                for g in finished:
                    tokens = [int(t) for t in hist[g * n, : k_now[g]] if t not in (PAD, BOS, EOS)]
                    j = int(stream[g])
                    ctx = advance(j, tokens)
                    if ctx is None:
                        j = next(upcoming, None)
                        ctx = None if j is None else start(j)
                    stream[g] = -1
                    if ctx is not None:
                        refill.append((g, j, ctx))
                if refill:
                    fill(refill)
                continue
            active = [g for g in range(s) if stream[g] >= 0]
            if not active:
                break
            k_now = self.k_step[:s].cpu().numpy()
            # Run up to the next step at which an active group's repetition check is due (position t = c - 1 + k
            # a multiple of 250, t > c + 100), so that it applies exactly there, as in the reference.
            due = {}
            for g in active:
                k0, c = int(k_now[g]), int(ctx_host[g])
                kc = max(k0, CHECK_WINDOW + 2)
                kc += (-(c - 1 + kc)) % CHECK_EVERY
                due[g] = kc
            steps = max(1, min(self.sync_every, min(due[g] - int(k_now[g]) + 1 for g in active)))
            need = max(int(k_now[g]) for g in active) + steps + 1
            if need + 1 > self.gcap:
                self.ensure_beam(self.ccap, need + self.reserve)
            ckeys = min(self.ccap, bucket(max(int(ctx_host[g]) for g in active)))
            gkeys = min(self.gcap, bucket(need))
            run = self.beam_runner(s, ckeys, gkeys)
            for _ in range(steps):
                run()
            self.stats["steps"] += steps
            self.stats["rows_steps"] += steps * n * len(active)
            k_after = self.k_step[:s].cpu().numpy()
            checks = [g for g in active if int(k_after[g]) - 1 == due[g]]
            if checks:
                hist = self.hist[: s * n].cpu().numpy()
                stop = []
                for g in checks:
                    kk = due[g]
                    for b in range(n):
                        if compression_ratio(hist[g * n + b, kk - CHECK_WINDOW : kk]) > CHECK_RATIO:
                            stop.append(g * n + b)
                if stop:
                    self.stats["repetition_stops"] += len(stop)
                    rows = torch.tensor(stop, device=dev)
                    self.ended[rows] = True
                    groups = torch.tensor(checks, device=dev)
                    self.gdone[groups] |= self.ended[: s * n].view(s, n)[groups].all(-1)
        if normal is not None and (self.ccap, self.gcap) != normal:
            self.release()


__all__ = ["BeamDecoder", "MAX_GEN"]
