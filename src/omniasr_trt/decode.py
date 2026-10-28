"""Batched greedy decoding of the omniASR LLM decoder with a static KV cache and CUDA graphs.

Each row has its own position: contexts of different lengths are prefilled together (right padding, causal
attention, so no valid token sees a pad), and decode steps write K/V at each row's own position and mask the
keys after it. One decode step (12 layers, LM head, argmax) is one CUDA graph per (rows, key bucket); the host
looks at the tokens only every ``sync_every`` steps (EOS, length cap, the repetition check).

State per row, as in the reference's ``out_tokens``: ``out[r, t]`` is the token predicted from input position
``t``; ``last[r]`` the position of the newest token ``tok[r]``, which is fed at ``last + 1`` by the next step.
"""

from __future__ import annotations

import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from . import kernels
from .llm import BOS, EOS, MAX_GEN, PAD, early_stop_index
from .modeling import LLMModel, apply_rope

KEY_BUCKET = 256
LIMIT = MAX_GEN - 4  # the reference stops after generating from this position
# A streaming context holds two segments of audio (2 x 749 frames), the previous segment's text and 7 markers; the
# caches start that large so that the contexts after the first segment do not grow them (and the CUDA graphs captured
# on them) in every call: a 15 s segment's text is rarely longer than 400 characters.
CONTEXT_FLOOR = 2048


def fuse_projections(model: LLMModel) -> list[tuple[Tensor, Tensor, Tensor, Tensor]]:
    """Per layer: (qkv [3D, D], output [D, D], gate+inner [2F, D], ffn output [D, F]).

    q/k/v and gate/inner are concatenated once and the modules' weights re-pointed at slices of the fused
    tensors, so no weight is held twice.
    """
    fused = []
    for layer in model.llama_decoder.layers:
        a, f = layer.self_attn, layer.ffn
        if not hasattr(layer, "_qkv"):
            with torch.no_grad():
                qkv = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight])
                gi = torch.cat([f.gate_proj.weight, f.inner_proj.weight])
            d, n = a.q_proj.weight.size(0), f.gate_proj.weight.size(0)
            for j, p in enumerate((a.q_proj, a.k_proj, a.v_proj)):
                p.weight = torch.nn.Parameter(qkv[j * d : (j + 1) * d], requires_grad=False)
            f.gate_proj.weight = torch.nn.Parameter(gi[:n], requires_grad=False)
            f.inner_proj.weight = torch.nn.Parameter(gi[n:], requires_grad=False)
            layer._qkv, layer._gi = qkv, gi
        fused.append((layer._qkv, a.output_proj.weight, layer._gi, f.output_proj.weight))
    return fused


class GreedyDecoder:
    def __init__(self, model: LLMModel, *, max_batch: int = 16, sync_every: int = 16, graphs: bool = True,
                 reserve: int = 768, fused: bool = True):
        self.m = model
        cfg = model.dec_cfg
        self.layers = fuse_projections(model)
        self.heads = cfg.num_heads
        self.head_dim = cfg.model_dim // cfg.num_heads
        self.dtype = model.final_proj.weight.dtype
        self.device = model.final_proj.weight.device
        self.max_batch = max_batch
        self.sync_every = sync_every
        self.step_tokens = 1  # positions a row advances by at most in one step
        self.use_graphs = graphs and self.device.type == "cuda"
        self.fused = fused and kernels.available(self.device)  # Triton RoPE/KV write and flash-decode attention
        self.fused_prefill = self.fused and os.environ.get("OMNIASR_TRT_FUSED_PREFILL", "1") != "0"
        self.debug_sync = os.environ.get("OMNIASR_TRT_DEBUG_SYNC") == "1"  # synchronise after each operation
        if self.use_graphs:
            self.pool = torch.cuda.graph_pool_handle()
            self.capture_stream = torch.cuda.Stream(self.device)
        self.reserve = reserve  # generated positions beyond the longest context before the cache grows
        self.context_floor = CONTEXT_FLOOR
        self.capacity = 0
        self.graphs: dict = {}
        self.stats = {"steps": 0, "rows_steps": 0, "tokens": 0, "prefill_tokens": 0, "captures": 0, "grows": 0}
        # Per-slot state (graphs hold these addresses): newest token, its position, finished.
        self.tok = torch.zeros(max_batch, device=self.device, dtype=torch.long)
        self.last = torch.zeros(max_batch, device=self.device, dtype=torch.long)
        self.done = torch.ones(max_batch, device=self.device, dtype=torch.bool)
        self.phase = lambda name: nullcontext()  # the transcriber's timer when profiling

    # ------------------------------------------------------------------ persistent buffers

    def ensure_capacity(self, length: int) -> None:
        """KV cache and token buffer for ``length`` positions (graphs hold their addresses, so they persist)."""
        length = min(-(-length // KEY_BUCKET) * KEY_BUCKET, -(-MAX_GEN // KEY_BUCKET) * KEY_BUCKET)
        if length <= self.capacity:
            return
        b, dev = self.max_batch, self.device
        if dev.type == "cuda":
            torch.cuda.synchronize()  # no step in flight while buffers and graphs are replaced
        shape = (len(self.layers), b, self.heads, length, self.head_dim)
        k = torch.zeros(shape, device=dev, dtype=self.dtype)
        v = torch.zeros(shape, device=dev, dtype=self.dtype)
        out = torch.full((b, length), PAD, device=dev, dtype=torch.int32)
        if self.capacity:
            k[..., : self.capacity, :] = self.k
            v[..., : self.capacity, :] = self.v
            out[:, : self.capacity] = self.out
            self.stats["grows"] += 1
        self.k, self.v, self.out, self.capacity = k, v, out, length
        self.reset_graphs()  # captured against the old buffers

    def check(self, where: str) -> None:
        if self.debug_sync:
            try:
                torch.cuda.synchronize()
            except Exception as e:
                raise RuntimeError(f"CUDA error after {where} (capacity {self.capacity})") from e

    def reset_graphs(self) -> None:
        """Drop the captured graphs and start a new memory pool (a pool dies with the last graph that used it)."""
        self.graphs.clear()
        if self.use_graphs:
            self.pool = torch.cuda.graph_pool_handle()

    def release(self) -> None:
        """Drop the KV cache and the graphs captured on it (the next call allocates its normal size again)."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.k = self.v = self.out = None
        self.capacity = 0
        self.reset_graphs()

    # ------------------------------------------------------------------ prefill

    def prefill(self, ctx: list[Tensor], slots: list[int] | None = None) -> Tensor:
        """Fill cache rows ``slots`` (default 0..B-1) with the contexts; returns each one's first token [B]."""
        return self.prefill_into(ctx, slots, self.k, self.v).argmax(-1)

    def prefill_into(self, ctx: list[Tensor], slots, k_cache: Tensor, v_cache: Tensor) -> Tensor:
        """Contexts into rows ``slots`` of ``k_cache`` / ``v_cache`` ([layers, rows, H, S, Dh]); returns the
        log-softmax after each context's last token [B, V] (model dtype)."""
        m, b = self.m, len(ctx)
        lens = [c.size(0) for c in ctx]
        width = max(lens)
        rows = None if slots is None or slots == list(range(b)) else torch.tensor(slots, device=self.device)
        x = torch.zeros(b, width, ctx[0].size(1), device=self.device, dtype=self.dtype)
        for j, c in enumerate(ctx):
            x[j, : lens[j]] = c
        cs = m.rope[:width][:, None]
        h = x.view(b * width, -1)
        if self.fused_prefill:  # every token's position and cache row, for kernels.rope_kv
            pos = torch.arange(width, device=self.device).repeat(b)
            slot = (torch.arange(b, device=self.device) if rows is None else rows).repeat_interleave(width)
        for i, layer in enumerate(m.llama_decoder.layers):
            qkv_w, out_w, gi_w, ffo_w = self.layers[i]
            qkv = layer.self_attn_layer_norm(h) @ qkv_w.t()  # [B*W, 3*H*Dh]
            if self.fused_prefill:  # q rotated; k, v rotated / written straight into the cache rows
                q = kernels.rope_kv(qkv, m.rope, pos, k_cache[i], v_cache[i], self.heads, slot)
                q = q.view(b, width, self.heads, -1).transpose(1, 2)
                k = k_cache[i, :b, :, :width] if rows is None else k_cache[i, rows, :, :width]
                v = v_cache[i, :b, :, :width] if rows is None else v_cache[i, rows, :, :width]
            else:
                q, k, v = qkv.view(b, width, 3, self.heads, self.head_dim).unbind(2)
                q = apply_rope(q, cs).transpose(1, 2)
                k = apply_rope(k, cs).transpose(1, 2)
                v = v.transpose(1, 2)
                if rows is None:
                    k_cache[i, :b, :, :width] = k
                    v_cache[i, :b, :, :width] = v
                else:
                    k_cache[i, rows, :, :width] = k
                    v_cache[i, rows, :, :width] = v
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            h = torch.addmm(h, o.transpose(1, 2).reshape(b * width, -1), out_w.t())
            gate, inner = (layer.ffn_layer_norm(h) @ gi_w.t()).chunk(2, -1)
            h = torch.addmm(h, inner * F.silu(gate), ffo_w.t())
        h = h.view(b, width, -1)
        last = h[torch.arange(b, device=self.device), torch.tensor(lens, device=self.device) - 1]
        logits = m.final_proj(m.llama_decoder.layer_norm(last))
        self.stats["prefill_tokens"] += sum(lens)
        return F.log_softmax(logits, dim=-1)

    # ------------------------------------------------------------------ one decode step

    def step(self, b: int, keys: int) -> None:
        """In place on rows 0..b-1 of the persistent buffers; finished rows are left as they are."""
        m = self.m
        tok, last, done, out = self.tok[:b], self.last[:b], self.done[:b], self.out[:b]
        running = ~done
        p = last + running.long()  # input position of ``tok``; finished rows recompute their last slot
        h = m.text_frontend.weight[tok]  # [B, D]
        cs = None if self.fused else m.rope[p][:, None]  # [B, 1, Dh/2, 2]
        scale = self.head_dim**-0.5
        if self.fused:
            lengths = torch.where(running, p + 1, 0)  # finished rows read no keys (their outputs are not kept)
        else:
            rows = torch.arange(b, device=self.device)
            allowed = torch.arange(keys, device=self.device)[None, :] <= p[:, None]  # [B, S]
            bias = torch.zeros(b, 1, 1, keys, device=self.device, dtype=torch.float32)
            bias.masked_fill_(~allowed[:, None, None], float("-inf"))
        for i, layer in enumerate(m.llama_decoder.layers):
            qkv_w, out_w, gi_w, ffo_w = self.layers[i]
            qkv = layer.self_attn_layer_norm(h) @ qkv_w.t()
            if self.fused:
                q = kernels.rope_kv(qkv, m.rope, p, self.k[i], self.v[i], self.heads)
                o = kernels.decode_attention(q, self.k[i], self.v[i], lengths, keys, scale).reshape(b, -1)
            else:
                q, k, v = qkv.view(b, 3, self.heads, self.head_dim).unbind(1)
                q = apply_rope(q, cs)
                k = apply_rope(k, cs)
                self.k[i, rows, :, p] = k
                self.v[i, rows, :, p] = v
                kc = self.k[i, :b, :, :keys]  # [B, H, S, Dh]
                vc = self.v[i, :b, :, :keys]
                scores = (q[:, :, None, :] @ kc.transpose(-1, -2)).float() * scale + bias  # [B, H, 1, S]
                o = (torch.softmax(scores, dim=-1).to(self.dtype) @ vc).reshape(b, -1)
            h = torch.addmm(h, o, out_w.t())
            gate, inner = (layer.ffn_layer_norm(h) @ gi_w.t()).chunk(2, -1)
            h = torch.addmm(h, inner * F.silu(gate), ffo_w.t())
        logits = m.final_proj(m.llama_decoder.layer_norm(h))
        nxt = F.log_softmax(logits, dim=-1).argmax(-1)
        kept = out.gather(1, p[:, None])[:, 0]
        out.scatter_(1, p[:, None], torch.where(running, nxt.to(out.dtype), kept)[:, None])
        done |= running & ((nxt == EOS) | (p >= LIMIT))
        tok.copy_(torch.where(running, nxt, tok))
        last.copy_(p)

    def state(self) -> list[Tensor]:
        """The per-slot buffers a step changes (restored after the warm-up step of a capture)."""
        return [self.tok, self.last, self.done, self.out]

    def filled(self, slots: list[int], streams: list[int]) -> None:
        """Called after ``slots`` were prefilled with a context of ``streams`` (subclasses add their state)."""

    def runner(self, b: int, keys: int):
        if not self.use_graphs:
            return lambda: self.step(b, keys)
        key = (b, keys)
        if key not in self.graphs:
            # Captured by hand, not with ``torch.cuda.graph``: that calls ``empty_cache()`` on every capture, and with
            # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True a capture in the middle of decoding then ended in an
            # illegal memory access on some runs (none without expandable segments or with a synchronisation after
            # every operation). All graphs share one memory pool.
            torch.cuda.synchronize()
            saved = [t[:b].clone() for t in self.state()]
            stream = self.capture_stream
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self.step(b, keys)  # warm-up outside the capture (K/V it writes are rewritten identically)
                for t, v in zip(self.state(), saved, strict=True):
                    t[:b].copy_(v)
                graph = torch.cuda.CUDAGraph()
                graph.capture_begin(pool=self.pool)
                self.step(b, keys)
                graph.capture_end()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self.graphs[key] = graph
            self.stats["captures"] += 1
        return self.graphs[key].replay

    # ------------------------------------------------------------------ generate

    @torch.inference_mode()
    def generate(self, contexts: list[Tensor]) -> list[list[int]]:
        """Greedy text tokens of each context (``[C, D]`` embeddings ending with <bos>)."""
        results = [None] * len(contexts)

        def finish(j, tokens):
            results[j] = tokens
            return None

        self.decode_streams(len(contexts), contexts.__getitem__, finish)
        return results

    @torch.inference_mode()
    def decode_streams(self, n: int, start, advance) -> None:
        """Decode ``n`` independent streams of contexts with up to ``max_batch`` of them in flight.

        ``start(j)`` gives stream j's first context; ``advance(j, tokens)`` receives the greedy tokens of its current
        context and returns its next one (None: the stream is finished). A slot whose context is finished takes its
        stream's next context (or a new stream) at the next host check, so the batch stays full instead of waiting
        for its slowest row, as decoding all inputs' segment i in lock step would.
        """
        b = min(self.max_batch, n)
        dev = self.device
        stream = np.full(b, -1)  # stream decoded in each slot (-1: idle)
        lens = np.zeros(b, dtype=np.int64)  # its context length
        checked = np.zeros(b, dtype=np.int64)  # positions up to which the repetition check has run
        stopped = np.full(b, -1)  # position of a repetition stop
        upcoming = iter(range(n))
        normal = None

        def fill(pairs):  # [(slot, stream, context)]: prefill and reset those slots
            nonlocal normal
            longest = max(c.size(0) for _, _, c in pairs)
            in_flight = [int(self.last[s]) for s in range(b) if stream[s] >= 0 and s not in {p[0] for p in pairs}]
            self.ensure_capacity(max([max(longest, self.context_floor) + self.reserve] +
                                     [p + self.sync_every * self.step_tokens + 2 for p in in_flight]))
            normal = normal or self.capacity
            slots = [s for s, _, _ in pairs]
            with self.phase("prefill"):
                first = self.prefill([c for _, _, c in pairs], slots)
            self.check("prefill")
            idx = torch.tensor(slots, device=dev)
            pos = torch.tensor([c.size(0) - 1 for _, _, c in pairs], device=dev)
            self.out[idx] = PAD
            self.out[idx, pos] = first.to(torch.int32)
            self.tok[idx] = first
            self.last[idx] = pos
            self.done[idx] = (first == EOS) | (pos >= LIMIT)
            for s, j, c in pairs:
                stream[s], lens[s], checked[s], stopped[s] = j, c.size(0), c.size(0) - 1, -1
            self.filled(slots, [j for _, j, _ in pairs])

        # Idle slots still run in the step graph (reading no keys) and write at their ``last`` position: reset it, as
        # a previous call may have left positions beyond a cache released since.
        self.done[:b] = True
        self.last[:b] = 0
        self.tok[:b] = 0
        fill([(s, j, start(j)) for s, j in zip(range(b), upcoming)])
        while True:
            done = self.done[:b].cpu().numpy()
            finished = [s for s in range(b) if stream[s] >= 0 and done[s]]
            if finished:
                out = self.out[:b].cpu().numpy()
                last = self.last[:b].cpu().numpy()
                refill = []
                for s in finished:
                    end = int(stopped[s]) if stopped[s] >= 0 else int(last[s])
                    tokens = [int(t) for t in out[s, lens[s] - 1 : end + 1] if t not in (PAD, BOS, EOS)]
                    self.stats["tokens"] += end - int(lens[s]) + 2
                    j = int(stream[s])
                    ctx = advance(j, tokens)
                    if ctx is None:
                        j = next(upcoming, None)
                        ctx = None if j is None else start(j)
                    stream[s] = -1
                    if ctx is not None:
                        refill.append((s, j, ctx))
                if refill:
                    fill(refill)
                continue  # a refilled slot may be finished at once (first token EOS)
            if (stream < 0).all():
                break
            last = self.last[:b].cpu().numpy()
            need = int(last[stream >= 0].max()) + self.sync_every * self.step_tokens + 2
            if need > self.capacity:
                self.ensure_capacity(need + self.reserve)
                self.check("grow")
            keys = min(self.capacity, -(-need // KEY_BUCKET) * KEY_BUCKET)
            run = self.runner(b, keys)
            self.check("runner")
            for _ in range(self.sync_every):
                run()
                self.check("step")
            self.stats["steps"] += self.sync_every
            self.stats["rows_steps"] += self.sync_every * int(((stream >= 0) & ~done).sum())
            out = self.out[:b].cpu().numpy()
            last = self.last[:b].cpu().numpy()
            stop_now = []
            for s in range(b):
                if stream[s] < 0 or stopped[s] >= 0:
                    continue
                hit = early_stop_index(out[s], int(checked[s]) + 1, int(last[s]), int(lens[s]))
                checked[s] = last[s]
                if hit is not None:
                    stopped[s] = hit
                    stop_now.append(s)
            if stop_now:
                self.done[torch.tensor(stop_now, device=dev)] = True
        if normal is not None and self.capacity > normal:
            # A long generation grew the cache (up to 8192 positions for every row); a process that lives for hours
            # must not keep that, so the next call starts from its normal size again.
            self.release()
