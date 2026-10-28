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
    def _rope_kv(qkv, cs, pos, slot, cpos, q_out, k_cache, v_cache, s_kb, s_kh, s_ks, H: tl.constexpr,
                 D: tl.constexpr, SLOTS: tl.constexpr, CPOS: tl.constexpr):
        b = tl.program_id(0)
        h = tl.program_id(1)
        p = tl.load(pos + b)
        cache_row = tl.load(slot + b) if SLOTS else b
        cp = tl.load(cpos + b) if CPOS else p  # cache position (beam search: the generated position)
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
        ko = k_cache + cache_row * s_kb + h * s_kh + cp * s_ks
        tl.store(ko + 2 * j, (k_re * c - k_im * s).to(k_cache.dtype.element_ty))
        tl.store(ko + 2 * j + 1, (k_re * s + k_im * c).to(k_cache.dtype.element_ty))
        d = tl.arange(0, D)
        tl.store(v_cache + cache_row * s_kb + h * s_kh + cp * s_ks + d, tl.load(row + 2 * H * D + d))

    @triton.jit
    def _beam_ctx(q, k_cache, v_cache, ctx_len, part_o, part_ml, scale, s_kb, s_kh, s_ks, CHUNK, H: tl.constexpr,
                  D: tl.constexpr, N: tl.constexpr, NP: tl.constexpr, ST: tl.constexpr, BLOCK_N: tl.constexpr,
                  IEEE: tl.constexpr):
        """The N beam queries of input g over its shared context keys, one K/V tile load for all of them."""
        g = tl.program_id(0)
        h = tl.program_id(1)
        sp = tl.program_id(2)
        n = tl.load(ctx_len + g)
        start = sp * CHUNK
        stop = tl.minimum(start + CHUNK, n)
        beam = tl.arange(0, NP)
        live = beam < N
        rows = g * N + beam
        d = tl.arange(0, D)
        qt = tl.load(q + (rows[:, None] * H + h) * D + d[None, :], mask=live[:, None], other=0.0)  # [NP, D]
        base_k = k_cache + g * s_kb + h * s_kh
        base_v = v_cache + g * s_kb + h * s_kh
        m = tl.full([NP], float("-inf"), tl.float32)
        l_sum = tl.zeros([NP], tl.float32)
        acc = tl.zeros([NP, D], tl.float32)
        for s0 in range(start, stop, BLOCK_N):
            s = s0 + tl.arange(0, BLOCK_N)
            valid = s < stop
            kt = tl.load(base_k + s[:, None] * s_ks + d[None, :], mask=valid[:, None], other=0.0)  # [BLOCK_N, D]
            if IEEE:
                sc = tl.dot(qt, tl.trans(kt), input_precision="ieee")
            else:
                sc = tl.dot(qt, tl.trans(kt))
            sc = tl.where(valid[None, :], sc * scale, float("-inf"))  # [NP, BLOCK_N]
            m_new = tl.maximum(m, tl.max(sc, axis=1))
            alpha = tl.exp(m - m_new)
            pr = tl.exp(sc - m_new[:, None])
            vt = tl.load(base_v + s[:, None] * s_ks + d[None, :], mask=valid[:, None], other=0.0)
            if IEEE:
                pv = tl.dot(pr, vt.to(tl.float32), input_precision="ieee")
            else:
                pv = tl.dot(pr.to(vt.dtype), vt)
            acc = acc * alpha[:, None] + pv
            l_sum = l_sum * alpha + tl.sum(pr, axis=1)
            m = m_new
        idx = (rows * H + h) * ST + sp
        tl.store(part_o + idx[:, None] * D + d[None, :], acc, mask=live[:, None])
        tl.store(part_ml + idx * 2, m, mask=live)
        tl.store(part_ml + idx * 2 + 1, l_sum, mask=live)

    @triton.jit
    def _beam_gen(q, k_cache, v_cache, ptr, gen_len, part_o, part_ml, scale, s_kb, s_kh, s_ks, s_ptr,
                  H: tl.constexpr, D: tl.constexpr, ST: tl.constexpr, SP: tl.constexpr, BLOCK_N: tl.constexpr):
        """Row r's generated keys: position j was written by row ``ptr[r, j]`` (an ancestor of r's beam)."""
        r = tl.program_id(0)
        h = tl.program_id(1)
        n = tl.load(gen_len + r)
        d = tl.arange(0, D)
        qv = tl.load(q + (r * H + h) * D + d).to(tl.float32) * scale
        m = tl.full([], float("-inf"), tl.float32)
        l_sum = tl.full([], 0.0, tl.float32)
        acc = tl.zeros([D], tl.float32)
        for s0 in range(0, n, BLOCK_N):
            s = s0 + tl.arange(0, BLOCK_N)
            valid = s < n
            src = tl.load(ptr + r * s_ptr + s, mask=valid, other=0).to(tl.int64)
            off = src[:, None] * s_kb + h * s_kh + s[:, None] * s_ks + d[None, :]
            kb = tl.load(k_cache + off, mask=valid[:, None], other=0.0).to(tl.float32)
            qk = tl.where(valid, tl.sum(kb * qv[None, :], axis=1), float("-inf"))
            m_new = tl.maximum(m, tl.max(qk, axis=0))
            alpha = tl.exp(m - m_new)
            pr = tl.exp(qk - m_new)
            vb = tl.load(v_cache + off, mask=valid[:, None], other=0.0).to(tl.float32)
            acc = acc * alpha + tl.sum(pr[:, None] * vb, axis=0)
            l_sum = l_sum * alpha + tl.sum(pr, axis=0)
            m = m_new
        idx = (r * H + h) * ST + SP
        tl.store(part_o + idx * D + d, acc)
        tl.store(part_ml + idx * 2, m)
        tl.store(part_ml + idx * 2 + 1, l_sum)

    @triton.jit
    def _beam_combine(part_o, part_ml, out, H: tl.constexpr, D: tl.constexpr, ST: tl.constexpr, SPP: tl.constexpr):
        r = tl.program_id(0)
        h = tl.program_id(1)
        base = (r * H + h) * ST
        sp = tl.arange(0, SPP)
        ok = sp < ST
        m = tl.load(part_ml + (base + sp) * 2, mask=ok, other=float("-inf"))
        l_sum = tl.load(part_ml + (base + sp) * 2 + 1, mask=ok, other=0.0)
        top = tl.max(m, axis=0)
        w = tl.where(m == float("-inf"), 0.0, tl.exp(m - top))
        d = tl.arange(0, D)
        acc = tl.load(part_o + (base + sp)[:, None] * D + d[None, :], mask=ok[:, None], other=0.0)
        denom = tl.sum(l_sum * w, axis=0)
        o = tl.sum(acc * w[:, None], axis=0) / tl.where(denom > 0, denom, 1.0)
        tl.store(out + (r * H + h) * D + d, o.to(out.dtype.element_ty))

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
            heads: int, slots: torch.Tensor | None = None, cache_pos: torch.Tensor | None = None) -> torch.Tensor:
    """``qkv`` [N, 3*H*D] (q|k|v of N tokens), ``cs`` [P, D/2, 2] fp32, ``pos`` [N]; ``k_cache``/``v_cache``
    [rows, H, S, D] with unit stride in D. Rotates at ``pos[n]`` and writes token n's k, v at cache row ``slots[n]``
    (default n), position ``cache_pos[n]`` (default ``pos[n]``); returns the rotated q [N, H, D]. Decode: one token
    per row; prefill: every token of a context; beam search: generated positions apart from the context."""
    n = qkv.size(0)
    dim = qkv.size(1) // (3 * heads)
    q = torch.empty(n, heads, dim, device=qkv.device, dtype=qkv.dtype)
    _rope_kv[(n, heads)](qkv, cs, pos, pos if slots is None else slots, pos if cache_pos is None else cache_pos, q,
                         k_cache, v_cache, k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), H=heads, D=dim,
                         SLOTS=slots is not None, CPOS=cache_pos is not None)
    return q


def beam_attention(q: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor, ctx_len: torch.Tensor, kg: torch.Tensor,
                   vg: torch.Tensor, ptr: torch.Tensor, gen_len: torch.Tensor, beams: int, ctx_keys: int,
                   scale: float) -> torch.Tensor:
    """Beam rows ``q`` [S*N, H, D] (input g's beams are rows g*N .. g*N+N-1): attention over input g's context
    ``kc``/``vc`` [>=S, H, C, D] (first ``ctx_len[g]`` keys, all within ``ctx_keys``) and row r's generated keys
    ``kg``/``vg`` [>=S*N, H, G, D], position j taken from row ``ptr[r, j]`` (first ``gen_len[r]``)."""
    rows, heads, dim = q.shape
    groups = rows // beams
    splits, chunk = splits_for(groups, heads, ctx_keys, 32)
    st = splits + 1  # the context splits, then the generated part
    part_o = torch.empty(rows, heads, st, dim, device=q.device, dtype=torch.float32)
    part_ml = torch.empty(rows, heads, st, 2, device=q.device, dtype=torch.float32)
    ieee = q.dtype == torch.float32  # fp32 (agreement checks): exact products, smaller tiles for shared memory
    block, stages = (16, 1) if ieee else (32, 2)
    np2 = max(16, triton.next_power_of_2(beams))  # tl.dot needs 16 rows
    _beam_ctx[(groups, heads, splits)](q, kc, vc, ctx_len, part_o, part_ml, scale, kc.stride(0), kc.stride(1),
                                       kc.stride(2), chunk, H=heads, D=dim, N=beams, NP=np2, ST=st, BLOCK_N=block,
                                       IEEE=ieee, num_warps=8, num_stages=stages)
    _beam_gen[(rows, heads)](q, kg, vg, ptr, gen_len, part_o, part_ml, scale, kg.stride(0), kg.stride(1),
                             kg.stride(2), ptr.stride(0), H=heads, D=dim, ST=st, SP=splits, BLOCK_N=32, num_warps=4)
    out = torch.empty_like(q)
    _beam_combine[(rows, heads)](part_o, part_ml, out, H=heads, D=dim, ST=st, SPP=triton.next_power_of_2(st),
                                 num_warps=4)
    return out


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
