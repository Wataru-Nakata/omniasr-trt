"""Triton kernels for the LLM decode step (CUDA only; the decoder falls back to PyTorch).

* :func:`rope_kv` -- each token's q and k rotated at its own position (interleaved pairs in fp32, as
  fairseq2's RotaryEncoder) and k, v written into the layer's KV cache at that position; one launch for what
  takes ~20 element-wise kernels in PyTorch (decode: one token per row; prefill: whole contexts).
* :func:`decode_attention` -- one query per row over its first ``lengths[r]`` keys, split over the keys
  (flash-decoding: online softmax in fp32 per split, then one combine pass), so each step reads only the
  valid keys once and all SMs share the work even for a single row. A row of length 0 reads nothing and gets
  zeros (finished rows of a decode batch).
* :func:`beam_attention` -- beam search: a context's beams share one read of its keys, plus each beam's own
  generated keys.
* :func:`draft_attention` -- speculative decoding: a row's draft tokens share one read of its keys (causal);
  :func:`propose` -- its proposals, the search of ``spec.SpecDecoder`` in one launch.
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
                  IEEE: tl.constexpr, CAUSAL: tl.constexpr = False):
        """The N beam queries of input g over its shared context keys, one K/V tile load for all of them.

        CAUSAL (draft verification): query i of row g sees ``ctx_len[g] + i`` keys, its own and the i before."""
        g = tl.program_id(0)
        h = tl.program_id(1)
        sp = tl.program_id(2)
        n = tl.load(ctx_len + g)
        start = sp * CHUNK
        beam = tl.arange(0, NP)
        live = beam < N
        if CAUSAL:
            stop = tl.minimum(start + CHUNK, n + N - 1)
            own = tl.minimum(start + CHUNK, n + beam)  # [NP]
        else:
            stop = tl.minimum(start + CHUNK, n)
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
            if CAUSAL:  # a query may have no key in this tile (or split): keep its m at -inf, add nothing
                sc = tl.where(s[None, :] < own[:, None], sc * scale, float("-inf"))
                m_new = tl.maximum(m, tl.max(sc, axis=1))
                m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
                alpha = tl.exp(m - m_safe)
                pr = tl.exp(sc - m_safe[:, None])
            else:
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


    @triton.jit
    def _propose(out, draft, draft_len, last, first, nxt, prop, use_draft, start, s_out, s_draft,
                 K: tl.constexpr, KP: tl.constexpr, NG: tl.constexpr, BEHIND: tl.constexpr, AHEAD: tl.constexpr,
                 W: tl.constexpr, SW: tl.constexpr, SELF_MIN: tl.constexpr, R0: tl.constexpr, N0: tl.constexpr,
                 R1: tl.constexpr, N1: tl.constexpr, R2: tl.constexpr, N2: tl.constexpr, R3: tl.constexpr,
                 N3: tl.constexpr):
        """Row b's proposal (``spec.SpecDecoder``): the draft candidates around the expected position and the
        row's own earlier positions, each scored by how many of its newest NG tokens precede it."""
        b = tl.program_id(0)
        lst = tl.load(last + b)
        fst = tl.load(first + b)
        nx = tl.load(nxt + b)
        dl = tl.load(draft_len + b)
        orow = out + b * s_out
        drow = draft + b * s_draft
        i = tl.arange(0, W)
        off = i - BEHIND
        anchor = nx - 1 + off  # candidate draft positions of the newest token
        alive = (off <= AHEAD) & (anchor < dl)
        score = tl.zeros([W], dtype=tl.int32)
        back = lst - 1 - tl.arange(0, SW)  # candidate own positions, latest first
        salive = back >= fst
        sscore = tl.zeros([SW], dtype=tl.int32)
        for t in tl.static_range(NG):
            hp = lst - t
            hv = hp >= fst
            h = tl.load(orow + hp, mask=hv, other=-1)
            at = anchor - t
            alive = alive & (at >= 0) & hv
            alive = alive & (tl.load(drow + at, mask=alive, other=-2) == h)
            score += alive.to(tl.int32)
            at = back - t
            salive = salive & (at >= 0) & hv
            salive = salive & (tl.load(orow + at, mask=salive, other=-2) == h)
            sscore += salive.to(tl.int32)
        dist = tl.abs(off)
        need = tl.where(dist <= R0, N0, tl.where(dist <= R1, N1, tl.where(dist <= R2, N2,
                                                                          tl.where(dist <= R3, N3, NG))))
        rank = tl.where((score >= need) & (off <= AHEAD), score * 256 - dist, -1)
        best = tl.argmax(rank, 0, tie_break_left=True)
        found = tl.max(rank, 0) >= 0
        drafted = tl.where(found, tl.sum(tl.where(i == best, score, 0), 0), 0)
        st = tl.where(found, nx + best - BEHIND, nx)
        mine = tl.max(sscore, 0)
        j = lst - 1 - tl.argmax(sscore, 0, tie_break_left=True)
        period = tl.maximum(lst - j, 1)
        ud = (mine < SELF_MIN) | (mine <= drafted)
        kk = tl.arange(0, KP)
        kv = kk < K
        at = st + kk
        from_draft = tl.load(drow + at, mask=kv & (at < dl), other=0)
        from_self = tl.load(orow + j + 1 + kk % period, mask=kv, other=0)
        tl.store(prop + b * K + kk, tl.where(ud, from_draft, from_self).to(tl.int64), mask=kv)
        tl.store(use_draft + b, ud)
        tl.store(start + b, st)


    @triton.jit
    def _accept(tokens, nxt, out, tok, last, done, draft_next, start, use_draft, s_out, cap, limit, eos,
                T: tl.constexpr, TP: tl.constexpr):
        """Row b of a drafted step: keep the fed tokens equal to the argmax before them, end at EOS / the length
        limit, write the kept argmaxes into ``out`` and advance the row (finished rows are left as they are)."""
        b = tl.program_id(0)
        if tl.load(done + b) == 0:
            p = tl.load(last + b) + 1
            j = tl.arange(0, TP)
            g = tl.load(nxt + b * T + j, mask=j < T, other=-1)
            fed = tl.load(tokens + b * T + j + 1, mask=j + 1 < T, other=-2)  # the token fed after argmax j
            kept = tl.minimum(tl.min(tl.where(fed == g, TP, j), 0), T - 1)  # leading equal ones
            stop = ((g == eos) | (p + j >= limit)) & (j <= kept) & (j < T)
            ended = tl.max(stop.to(tl.int32), 0) > 0
            n = tl.where(ended, tl.min(tl.where(stop, j, TP), 0), kept)
            tl.store(out + b * s_out + tl.minimum(p + j, cap - 1), g.to(tl.int32), mask=(j <= n) & (j < T))
            tl.store(tok + b, tl.sum(tl.where(j == n, g, 0), 0))
            tl.store(last + b, p + n)
            tl.store(done + b, ended)
            base = tl.where(tl.load(use_draft + b), tl.load(start + b), tl.load(draft_next + b))
            tl.store(draft_next + b, base + n + 1)


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


DRAFT_ATTN = {"block": 32, "warps": 4, "stages": 2, "programs": 264}  # bf16 (scripts/profile_spec.py --tune, H200)


def draft_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, lengths: torch.Tensor,
                    tokens: int, keys: int, scale: float, **launch) -> torch.Tensor:
    """Draft verification: ``q`` [B*T, H, D], row b's T queries at rows b*T .. b*T+T-1; query i of row b attends to
    keys [0, lengths[b] + i) of cache row b (``k_cache``/``v_cache`` [>=B, H, S, D], all within the first ``keys``).
    One K/V tile load serves all T queries of a row. Returns [B*T, H, D] in q's dtype. ``launch``: overrides of
    ``DRAFT_ATTN`` (bf16)."""
    rows, heads, dim = q.shape
    groups = rows // tokens
    ieee = q.dtype == torch.float32
    cfg = {**DRAFT_ATTN, **launch} if not ieee else {"block": 16, "warps": 8, "stages": 1, "programs": 264}
    splits, chunk = splits_for(groups, heads, keys, cfg["block"], cfg["programs"])
    part_o = torch.empty(rows, heads, splits, dim, device=q.device, dtype=torch.float32)
    part_ml = torch.empty(rows, heads, splits, 2, device=q.device, dtype=torch.float32)
    _beam_ctx[(groups, heads, splits)](q, k_cache, v_cache, lengths, part_o, part_ml, scale, k_cache.stride(0),
                                       k_cache.stride(1), k_cache.stride(2), chunk, H=heads, D=dim, N=tokens,
                                       NP=max(16, triton.next_power_of_2(tokens)), ST=splits, BLOCK_N=cfg["block"],
                                       IEEE=ieee, CAUSAL=True, num_warps=cfg["warps"], num_stages=cfg["stages"])
    out = torch.empty_like(q)
    _beam_combine[(rows, heads)](part_o, part_ml, out, H=heads, D=dim, ST=splits,
                                 SPP=triton.next_power_of_2(splits), num_warps=4)
    return out


def propose(out: torch.Tensor, draft: torch.Tensor, draft_len: torch.Tensor, last: torch.Tensor,
            first: torch.Tensor, nxt: torch.Tensor, start: torch.Tensor, tokens: int, ngram: int, behind: int,
            ahead: int, reach: tuple, self_window: int, self_min: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Drafted decoding's proposals of rows 0..B-1 (B = ``last.numel()``): ``tokens`` ids each [B, k] (int64) and
    whether they come from the draft [B] (else from the row's own tokens); ``start`` [B] gets the draft position of
    the first one. ``out`` [>=B, S] int32 generated tokens by position, ``draft`` [>=B, C] int32; ``draft_len``,
    ``last`` (position of the newest token), ``first`` (of the segment's first), ``nxt`` (draft position expected
    after the newest) [B] int64. See ``spec`` for the rule."""
    b = last.numel()
    prop = torch.empty(b, tokens, device=out.device, dtype=torch.long)
    use_draft = torch.empty(b, device=out.device, dtype=torch.bool)
    (r0, n0), (r1, n1), (r2, n2), (r3, n3) = reach
    _propose[(b,)](out, draft, draft_len, last, first, nxt, prop, use_draft, start, out.stride(0), draft.stride(0),
                   K=tokens, KP=triton.next_power_of_2(tokens), NG=ngram, BEHIND=behind, AHEAD=ahead,
                   W=triton.next_power_of_2(behind + ahead + 1), SW=triton.next_power_of_2(self_window),
                   SELF_MIN=self_min, R0=r0, N0=n0, R1=r1, N1=n1, R2=r2, N2=n2, R3=r3, N3=n3, num_warps=4)
    return prop, use_draft


def accept(tokens: torch.Tensor, nxt: torch.Tensor, out: torch.Tensor, tok: torch.Tensor, last: torch.Tensor,
           done: torch.Tensor, draft_next: torch.Tensor, start: torch.Tensor, use_draft: torch.Tensor, limit: int,
           eos: int) -> None:
    """In place on rows 0..B-1 after a drafted step: ``tokens`` fed [B, T] (the newest token, then the proposal) and
    ``nxt`` the argmax after each [B, T] (int64); ``out`` [>=B, S] int32, ``tok``/``last``/``draft_next``/``start``
    [B] int64, ``done``/``use_draft`` [B] bool. A running row keeps argmax 0..n (n: its leading proposed tokens
    equal to the argmax before them, or the first EOS / position >= ``limit`` among those) at positions
    ``last + 1 + j``."""
    b, width = tokens.shape
    _accept[(b,)](tokens, nxt, out, tok, last, done, draft_next, start, use_draft, out.stride(0), out.size(1), limit,
                  eos, T=width, TP=triton.next_power_of_2(width), num_warps=1)


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
