"""omniASR LLM (Unlimited) streaming transcription with omnilingual-asr's greedy search.

omnilingual-asr 0.2.0 (``Wav2Vec2LlamaBeamSearchSeq2SeqGenerator`` with ``nbest=1``) cuts each input into
15 s segments and decodes segment i with segment i-1's audio and text as context:

    [lang] <lid> (audio_{i-1} <regular> <bos> text_{i-1} <eos>) audio_i <last|regular> <bos> -> text_i <eos>

Per input the result depends only on that input (the reference batches inputs in lock step, but a finished
row only repeats EOS), so the fast path is free to batch, prefill and schedule rows however it likes. The
rules kept exactly: greedy on log-softmax, stop at EOS, at position ``8192 - 4``, or when the last 100 tokens
compress more than 4x (checked when the absolute position is a multiple of 250).
"""

from __future__ import annotations

import json
import math
import time
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import resources

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .modeling import HOP, LLMModel, apply_rope, normalize

SEGMENT = 15 * 16000
BOS, PAD, EOS = 0, 1, 2
MAX_GEN = 8192
CHECK_EVERY, CHECK_WINDOW, CHECK_RATIO = 250, 100, 4.0


def lang_table() -> dict[str, int]:
    """omnilingual-asr's language id -> embedding row (lower-cased keys; 0 = no language)."""
    text = resources.files(__package__).joinpath("lang_ids.json").read_text()
    return {k.lower(): v for k, v in json.loads(text).items()}


def compression_ratio(tokens: np.ndarray) -> float:
    """The reference's repetition test: zlib ratio of ``np.array_str`` of the int32 token ids."""
    data = np.array_str(tokens.astype(np.int32)).replace("\n", "").encode("utf-8")
    return len(data) / len(zlib.compress(data))


def early_stop_index(out: np.ndarray, start: int, stop: int, ctx: int) -> int | None:
    """First checked position t in [start, stop] whose preceding 100 tokens trigger the repetition stop.

    ``out[t]`` is the token predicted from input position ``t``; the reference keeps it and emits EOS after.
    """
    first = -(-start // CHECK_EVERY) * CHECK_EVERY
    for t in range(first, stop + 1, CHECK_EVERY):
        if t > ctx + CHECK_WINDOW and compression_ratio(out[t - CHECK_WINDOW : t]) > CHECK_RATIO:
            return t
    return None


@dataclass
class Row:
    """One input's streaming state."""

    index: int
    wave: Tensor  # normalised, on the GPU, model dtype
    lang: int
    segments: int
    audio: list = field(default_factory=list)  # per segment: [F, D] embeddings
    tokens: list = field(default_factory=list)  # per segment: list[int]


class LLMTranscriber:
    def __init__(self, model: LLMModel, tokenizer, *, max_batch: int = 8, encoder_batch_seconds: float = 240.0,
                 reserve: int = 512, beams: int = 1, length_norm: bool = False):
        self.model = model
        self.tok = tokenizer
        self.langs = lang_table()
        self.dtype = next(model.parameters()).dtype
        self.device = next(model.parameters()).device
        self.max_batch = max_batch
        self.encoder_batch = int(encoder_batch_seconds * 16000)
        self.reserve = reserve  # decoder KV positions beyond the longest context (grows on demand)
        self.beams, self.length_norm = beams, length_norm  # search defaults (1: greedy)
        self._beam_decoders: dict = {}
        v = model.dec_cfg.vocab_size
        self.lid, self.last_seg, self.regular_seg = v, v + 1, v + 2
        self.profile = False  # synchronised wall time per phase in ``timings``
        self.timings: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str):
        if not self.profile:
            yield
            return
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        torch.cuda.synchronize()
        self.timings[name] = self.timings.get(name, 0.0) + time.perf_counter() - t0

    # ------------------------------------------------------------------ inputs

    def lang_id(self, lang: str | None) -> int:
        return 0 if lang is None else self.langs.get(lang.lower(), 0)

    def rows(self, waves, langs) -> list[Row]:
        out = []
        for i, (w, lang) in enumerate(zip(waves, langs, strict=True)):
            w = torch.as_tensor(np.asarray(w, dtype=np.float32))
            n = max(1, math.ceil(len(w) / SEGMENT))
            out.append(Row(i, normalize(w).to(self.device, self.dtype), self.lang_id(lang), n))
        return out

    @torch.inference_mode()
    def encode_segments(self, rows: list[Row]) -> None:
        """Encoder + projection of every 15 s segment; equal lengths share a batch without padding.

        (Queuing this on a side stream to overlap the decoder gained nothing: the encoder's GEMM tiles fill every
        SM, so the decoder's kernels only interleave with them.)"""
        pieces = [(r, s) for r in rows for s in range(r.segments)]
        for r in rows:
            r.audio = [None] * r.segments
        length = {(r.index, s): min(SEGMENT, len(r.wave) - s * SEGMENT) for r, s in pieces}
        pieces.sort(key=lambda p: -length[p[0].index, p[1]])
        while pieces:
            longest = length[pieces[0][0].index, pieces[0][1]]
            count = max(1, min(len(pieces), self.encoder_batch // longest))
            batch, pieces = pieces[:count], pieces[count:]
            lens = torch.tensor([length[r.index, s] for r, s in batch])
            x = torch.zeros(len(batch), longest, device=self.device, dtype=self.dtype)
            for j, (r, s) in enumerate(batch):
                x[j, : lens[j]] = r.wave[s * SEGMENT : s * SEGMENT + lens[j]]
            emb, frames = self.model.embed_audio(x, None if bool((lens == longest).all()) else lens)
            for j, (r, s) in enumerate(batch):
                r.audio[s] = emb[j, : frames[j]]

    def context(self, r: Row, i: int) -> Tensor:
        """Decoder input embeddings ``[C, D]`` for segment ``i`` of row ``r`` (ending with <bos>)."""
        m = self.model
        text = m.text_frontend.weight

        def tokens(ids):
            return text[torch.as_tensor(ids, dtype=torch.long, device=self.device)]

        parts = [m.lang_embeddings.weight[r.lang : r.lang + 1], tokens([self.lid])]
        if i > 0:
            parts += [r.audio[i - 1], tokens([self.regular_seg, BOS] + r.tokens[i - 1] + [EOS])]
        marker = self.last_seg if i == r.segments - 1 else self.regular_seg
        parts += [r.audio[i], tokens([marker, BOS])]
        return torch.cat(parts).to(self.dtype)

    def text(self, r: Row) -> str:
        ids = [t for seg in r.tokens for t in seg if t not in (PAD, BOS, EOS)]
        return self.tok.decode(ids)

    # ------------------------------------------------------------------ reference-order decoding

    @torch.inference_mode()
    def greedy_reference(self, ctx: Tensor) -> list[int]:
        """One row, computed in the reference's order: prefill all but the final <bos>, then one step at a time."""
        m = self.model
        c = ctx.size(0)
        cache = [None] * len(m.llama_decoder.layers)
        m.decoder_forward(ctx[None, : c - 1], torch.arange(c - 1, device=self.device), cache)
        out = np.full(MAX_GEN, PAD, dtype=np.int32)
        x = ctx[c - 1]
        t = c - 1
        while True:
            h = m.decoder_forward(x[None, None], torch.tensor([t], device=self.device), cache)
            logp = F.log_softmax(m.final_proj(h[0, 0]), dim=-1)
            token = int(logp.float().argmax())
            out[t] = token
            done = token == EOS
            if t % CHECK_EVERY == 0 and t > c + CHECK_WINDOW:
                done |= compression_ratio(out[t - CHECK_WINDOW : t]) > CHECK_RATIO
            if done or t == MAX_GEN - 4:
                break
            x = m.text_frontend.weight[token].to(self.dtype)
            t += 1
        return [int(v) for v in out[c - 1 : t + 1] if v not in (PAD, BOS, EOS)]

    @torch.inference_mode()
    def beam_reference(self, ctx: Tensor, beams: int, length_norm: bool = False) -> list[int]:
        """omnilingual-asr's beam search for one input (its batch-1 behaviour), in its order of computation.

        ``beams`` rows; at the first step only row 0 is live (the others score -1e6); every step each row offers
        ``score + log p`` (``length_norm``: the running mean, chosen by row 0's token count), a finished row only
        EOS at its frozen score; the best ``beams`` of all rows' offers survive (``topk``, sorted) and the cache
        follows their parents. The text is row 0's after the last step.
        """
        m = self.model
        c, n = ctx.size(0), beams
        cache = [None] * len(m.llama_decoder.layers)
        m.decoder_forward(ctx[None, : c - 1].expand(n, -1, -1), torch.arange(c - 1, device=self.device), cache)
        out = torch.full((n, MAX_GEN), PAD, dtype=torch.int32, device=self.device)
        scores = torch.full((n,), -1e6, dtype=torch.float32, device=self.device)
        scores[0] = 0.0
        ended = torch.zeros(n, dtype=torch.bool, device=self.device)
        x = ctx[c - 1].expand(n, -1)
        t = c - 1
        while True:
            h = m.decoder_forward(x[:, None], torch.tensor([t], device=self.device), cache)
            logp = F.log_softmax(m.final_proj(h[:, 0]), dim=-1)  # [n, V], model dtype
            if length_norm:
                count = ((out[:, :t] != PAD) & (out[:, :t] != EOS)).sum(1, keepdim=True)
                cand = (scores[:, None] * count + logp) / (count + 1) if count[0, 0] > 0 else scores[:, None] + logp
            else:
                cand = scores[:, None] + logp
            cand[ended] = float("-inf")
            cand[ended, EOS] = scores[ended].to(cand.dtype)
            top, pick = cand.view(1, -1).topk(n, sorted=True)
            parent, token = pick[0] // cand.size(1), pick[0] % cand.size(1)
            out, ended, scores = out[parent], ended[parent], scores[parent]
            for kv in cache:
                kv[0], kv[1] = kv[0].index_select(0, parent), kv[1].index_select(0, parent)
            scores = torch.where(ended, scores, top[0])
            out[:, t] = token.to(torch.int32)
            out[ended, t] = EOS
            ended = out[:, t] == EOS
            if t % CHECK_EVERY == 0 and t > c + CHECK_WINDOW:
                window = out[:, t - CHECK_WINDOW : t].cpu().numpy()
                stop = torch.tensor([compression_ratio(row) > CHECK_RATIO for row in window], device=self.device)
                ended |= stop
            if bool(ended.all()) or t == MAX_GEN - 4:
                break
            x = m.text_frontend.weight[out[:, t].long()].to(self.dtype)
            t += 1
        return [int(v) for v in out[0, c - 1 : t + 1].tolist() if v not in (PAD, BOS, EOS)]

    # ------------------------------------------------------------------ public

    @torch.inference_mode()
    def transcribe(self, waves, langs=None, *, mode: str = "fast", beams: int | None = None,
                   length_norm: bool | None = None) -> list[str]:
        """Texts of 16 kHz mono waves (any length; a last segment needs >= 400 samples, as the reference).

        ``beams`` > 1: beam search (default: the transcriber's), ``length_norm``: rank hypotheses by their mean
        log-probability (omnilingual-asr's option)."""
        langs = [None] * len(waves) if langs is None else list(langs)
        beams = self.beams if beams is None else beams
        length_norm = self.length_norm if length_norm is None else length_norm
        rows = self.rows(waves, langs)
        if mode == "reference":
            self.encode_segments(rows)
            for r in rows:
                for i in range(r.segments):
                    ctx = self.context(r, i)
                    r.tokens.append(self.greedy_reference(ctx) if beams == 1 and not length_norm else
                                    self.beam_reference(ctx, beams, length_norm))
            return [self.text(r) for r in rows]
        with self.phase("encoder"):
            self.encode_segments(rows)
        return self.transcribe_rows(rows, beams, length_norm)

    def decoder(self, beams: int, length_norm: bool):
        """The fast decoder for this search (one per setting; the others' caches are released)."""
        from .beam import BeamDecoder
        from .decode import GreedyDecoder

        if beams == 1 and not length_norm:
            decoder = getattr(self, "_decoder", None)
            if decoder is None:
                decoder = self._decoder = GreedyDecoder(self.model, max_batch=self.max_batch, reserve=self.reserve)
        else:
            decoder = self._beam_decoders.get((beams, length_norm))
            if decoder is None:
                decoder = self._beam_decoders[beams, length_norm] = BeamDecoder(
                    self.model, beams=beams, length_norm=length_norm, max_batch=self.max_batch, reserve=self.reserve)
        for other in [getattr(self, "_decoder", None), *self._beam_decoders.values()]:
            if other is not None and other is not decoder and (getattr(other, "capacity", 0)
                                                               or getattr(other, "ccap", 0)):
                other.release()
        decoder.phase = self.phase
        return decoder

    def transcribe_rows(self, rows: list[Row], beams: int = 1, length_norm: bool = False) -> list[str]:
        """The fast decoder on encoded rows."""
        decoder = self.decoder(beams, length_norm)

        def start(j):
            with self.phase("context"):
                return self.context(rows[j], 0)

        def advance(j, tokens):
            r = rows[j]
            r.tokens.append(tokens)
            if len(r.tokens) == r.segments:
                return None
            with self.phase("context"):
                return self.context(r, len(r.tokens))

        # Each input's segments one after another (segment i is the context of i + 1); inputs share the slots.
        with self.phase("decoder"):
            decoder.decode_streams(len(rows), start, advance)
        return [self.text(r) for r in rows]


__all__ = ["LLMTranscriber", "SEGMENT", "HOP", "apply_rope", "early_stop_index"]
