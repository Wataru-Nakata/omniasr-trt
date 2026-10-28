"""Triton kernels for the LLM decode step (CUDA only; the decoder falls back to PyTorch).

* :func:`rope_kv` -- each token's q and k rotated at its own position (interleaved pairs in fp32, as
  fairseq2's RotaryEncoder) and k, v written into the layer's KV cache at that position; one launch for what
  takes ~20 element-wise kernels in PyTorch (decode: one token per row; prefill: whole contexts).
* :func:`decode_attention` -- one query per row over its first ``lengths[r]`` keys, split over the keys
  (flash-decoding: online softmax in fp32 per split, then one combine pass), so each step reads only the
  valid keys once and all SMs share the work even for a single row. A row of length 0 reads nothing and gets
  zeros (finished rows of a decode batch).
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # torch without Triton: the PyTorch code paths are used
    triton = None


def available(device: torch.device) -> bool:
    return triton is not None and device.type == "cuda"


if triton is not None:

    @triton.jit
    def _rope_kv(qkv, cs, pos, slot, q_out, k_cache, v_cache, s_kb, s_kh, s_ks, H: tl.constexpr, D: tl.constexpr,
                 SLOTS: tl.constexpr):
        b = tl.program_id(0)
        h = tl.program_id(1)
        p = tl.load(pos + b)
        cache_row = tl.load(slot + b) if SLOTS else b
        j = tl.arange(0, D // 2)
        row = qkv + b * (3 * H * D) + h * D
        c = tl.load(cs + p * D + 2 * j)
        s = tl.load(cs + p * D + 2 * j + 1)
        q_re = tl.load(row + 2 * j).to(tl.float32)
        q_im = tl.load(row + 2 * j + 1).to(tl.float32)
        qo = q_out + (b * H + h) * D
        tl.store(qo + 2 * j, (q_re * c - q_im * s).to(q_out.dtype.element_ty))
        tl.store(qo + 2 * j + 1, (q_re * s + q_im * c).to(q_out.dtype.element_ty))
        k_re = tl.load(row + H * D + 2 * j).to(tl.float32)
        k_im = tl.load(row + H * D + 2 * j + 1).to(tl.float32)
        ko = k_cache + cache_row * s_kb + h * s_kh + p * s_ks
        tl.store(ko + 2 * j, (k_re * c - k_im * s).to(k_cache.dtype.element_ty))
        tl.store(ko + 2 * j + 1, (k_re * s + k_im * c).to(k_cache.dtype.element_ty))
        d = tl.arange(0, D)
        tl.store(v_cache + cache_row * s_kb + h * s_kh + p * s_ks + d, tl.load(row + 2 * H * D + d))

    @triton.jit
    def _attn_split(q, k_cache, v_cache, lengths, part_o, part_ml, scale, s_kb, s_kh, s_ks, CHUNK,
                    H: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr, BLOCK_N: tl.constexpr):
        b = tl.program_id(0)
        h = tl.program_id(1)
        sp = tl.program_id(2)
        n = tl.load(lengths + b)
        start = sp * CHUNK
        stop = tl.minimum(start + CHUNK, n)
        d = tl.arange(0, D)
        qv = tl.load(q + (b * H + h) * D + d).to(tl.float32) * scale
        base_k = k_cache + b * s_kb + h * s_kh
        base_v = v_cache + b * s_kb + h * s_kh
        m = tl.full([], float("-inf"), tl.float32)
        l_sum = tl.full([], 0.0, tl.float32)
        acc = tl.zeros([D], tl.float32)
        for s0 in range(start, stop, BLOCK_N):
            s = s0 + tl.arange(0, BLOCK_N)
            valid = s < stop
            kb = tl.load(base_k + s[:, None] * s_ks + d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
            qk = tl.sum(kb * qv[None, :], axis=1)
            qk = tl.where(valid, qk, float("-inf"))
            m_new = tl.maximum(m, tl.max(qk, axis=0))
            alpha = tl.exp(m - m_new)
            pr = tl.exp(qk - m_new)
            vb = tl.load(base_v + s[:, None] * s_ks + d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
            acc = acc * alpha + tl.sum(pr[:, None] * vb, axis=0)
            l_sum = l_sum * alpha + tl.sum(pr, axis=0)
            m = m_new
        idx = (b * H + h) * SPLITS + sp
        tl.store(part_o + idx * D + d, acc)
        tl.store(part_ml + idx * 2, m)
        tl.store(part_ml + idx * 2 + 1, l_sum)

    @triton.jit
    def _attn_combine(part_o, part_ml, out, H: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr):
        b = tl.program_id(0)
        h = tl.program_id(1)
        base = (b * H + h) * SPLITS
        sp = tl.arange(0, SPLITS)
        m = tl.load(part_ml + (base + sp) * 2)
        l_sum = tl.load(part_ml + (base + sp) * 2 + 1)
        top = tl.max(m, axis=0)
        w = tl.where(m == float("-inf"), 0.0, tl.exp(m - top))  # empty splits contribute nothing
        d = tl.arange(0, D)
        acc = tl.load(part_o + (base + sp)[:, None] * D + d[None, :])
        denom = tl.sum(l_sum * w, axis=0)
        o = tl.sum(acc * w[:, None], axis=0) / tl.where(denom > 0, denom, 1.0)  # rows of length 0: zeros
        tl.store(out + (b * H + h) * D + d, o.to(out.dtype.element_ty))


def rope_kv(qkv: torch.Tensor, cs: torch.Tensor, pos: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
            heads: int, slots: torch.Tensor | None = None) -> torch.Tensor:
    """``qkv`` [N, 3*H*D] (q|k|v of N tokens), ``cs`` [P, D/2, 2] fp32, ``pos`` [N]; ``k_cache``/``v_cache``
    [rows, H, S, D] with unit stride in D. Writes token n's k, v at cache row ``slots[n]`` (default n), position
    ``pos[n]``; returns the rotated q [N, H, D]. Decode: one token per row; prefill: every token of a context."""
    n = qkv.size(0)
    dim = qkv.size(1) // (3 * heads)
    q = torch.empty(n, heads, dim, device=qkv.device, dtype=qkv.dtype)
    _rope_kv[(n, heads)](qkv, cs, pos, pos if slots is None else slots, q, k_cache, v_cache, k_cache.stride(0),
                         k_cache.stride(1), k_cache.stride(2), H=heads, D=dim, SLOTS=slots is not None)
    return q


def splits_for(rows: int, heads: int, keys: int, block: int = 64, programs: int = 264) -> tuple[int, int]:
    """Key splits (a power of two, about ``programs`` programs in all) and the keys each split covers."""
    want = max(1, programs // (rows * heads))
    splits = 1
    while splits * 2 <= want and splits * 2 * block <= keys:
        splits *= 2
    chunk = -(-keys // splits)
    return splits, -(-chunk // block) * block


ATTN = {"block": 64, "warps": 8, "stages": 2, "programs": 264}  # launch settings: scripts/tune_attn.py on an H200


def decode_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, lengths: torch.Tensor,
                     keys: int, scale: float, block: int | None = None, warps: int | None = None,
                     stages: int | None = None, programs: int | None = None) -> torch.Tensor:
    """``q`` [B, H, D]; ``k_cache``/``v_cache`` [>=B, H, S, D]; row r attends to keys [0, lengths[r]) (all
    within the first ``keys``, which fixes the launch grid). Returns [B, H, D] in q's dtype."""
    block = block or ATTN["block"]
    warps = warps or ATTN["warps"]
    stages = stages or ATTN["stages"]
    b, heads, dim = q.shape
    splits, chunk = splits_for(b, heads, keys, block, programs or ATTN["programs"])
    part_o = torch.empty(b, heads, splits, dim, device=q.device, dtype=torch.float32)
    part_ml = torch.empty(b, heads, splits, 2, device=q.device, dtype=torch.float32)
    _attn_split[(b, heads, splits)](q, k_cache, v_cache, lengths, part_o, part_ml, scale, k_cache.stride(0),
                                    k_cache.stride(1), k_cache.stride(2), chunk, H=heads, D=dim, SPLITS=splits,
                                    BLOCK_N=block, num_warps=warps, num_stages=stages)
    out = torch.empty_like(q)
    _attn_combine[(b, heads)](part_o, part_ml, out, H=heads, D=dim, SPLITS=splits, num_warps=4)
    return out
