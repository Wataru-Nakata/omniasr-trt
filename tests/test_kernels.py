"""Triton decode kernels against PyTorch (CUDA only)."""

import pytest
import torch

from omniasr_trt import kernels
from omniasr_trt.decode import GreedyDecoder
from omniasr_trt.modeling import DecoderConfig, apply_rope, rope_table

cuda = pytest.mark.skipif(not (torch.cuda.is_available() and kernels.available(torch.device("cuda"))),
                          reason="needs CUDA and Triton")


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rope_kv(dtype):
    torch.manual_seed(0)
    b, h, d, s = 5, 8, 512, 300
    cs = rope_table(DecoderConfig(), "cuda")
    qkv = torch.randn(b, 3 * h * d, device="cuda", dtype=dtype)
    pos = torch.tensor([0, 7, 299, 150, 31], device="cuda")
    k_cache = torch.zeros(b + 2, h, s, d, device="cuda", dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    q = kernels.rope_kv(qkv, cs, pos, k_cache, v_cache, h)
    rq, rk, rv = qkv.view(b, 3, h, d).unbind(1)
    c = cs[pos][:, None]
    tol = 1e-6 if dtype == torch.float32 else 0  # Triton contracts a*c - b*s into an FMA; bf16 rounds it away
    torch.testing.assert_close(q, apply_rope(rq, c), atol=tol, rtol=0)
    rows = torch.arange(b, device="cuda")
    torch.testing.assert_close(k_cache[rows, :, pos], apply_rope(rk, c), atol=tol, rtol=0)
    torch.testing.assert_close(v_cache[rows, :, pos], rv, atol=0, rtol=0)
    # explicit cache rows (prefill of scattered slots): token n goes to row slots[n]
    slots = torch.tensor([6, 0, 3, 6, 1], device="cuda")
    k2, v2 = torch.zeros_like(k_cache), torch.zeros_like(v_cache)
    q2 = kernels.rope_kv(qkv, cs, pos, k2, v2, h, slots)
    torch.testing.assert_close(q2, q, atol=0, rtol=0)
    torch.testing.assert_close(k2[slots, :, pos], k_cache[rows, :, pos], atol=0, rtol=0)
    torch.testing.assert_close(v2[slots, :, pos], rv, atol=0, rtol=0)


@cuda
@pytest.mark.parametrize("b,keys", [(1, 256), (3, 1024), (8, 2048), (16, 512)])
def test_decode_attention(b, keys):
    torch.manual_seed(1)
    h, d, cap = 8, 512, 2048
    k_cache = torch.randn(b + 1, h, cap, d, device="cuda")
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(b, h, d, device="cuda")
    lengths = torch.randint(1, keys + 1, (b,), device="cuda")
    lengths[0] = keys
    out = kernels.decode_attention(q, k_cache, v_cache, lengths, keys, d**-0.5)
    for r in range(b):
        n = int(lengths[r])
        att = torch.softmax((q[r, :, None] @ k_cache[r, :, :n].transpose(-1, -2)) * d**-0.5, -1)
        torch.testing.assert_close(out[r], (att @ v_cache[r, :, :n])[:, 0], atol=2e-5, rtol=2e-5)


@cuda
def test_fused_decoder_matches_torch_path():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_decode import small_model

    m = small_model(seed=2).float().cuda()
    m.rope = rope_table(m.dec_cfg, "cuda")
    torch.manual_seed(3)
    ctxs = [torch.randn(n, 32, device="cuda") for n in (5, 33, 17, 60)]
    a = GreedyDecoder(m, max_batch=4, sync_every=8, fused=False, graphs=False, reserve=64).generate(ctxs)
    b = GreedyDecoder(m, max_batch=4, sync_every=8, fused=True, graphs=True, reserve=64).generate(ctxs)
    assert a == b


@cuda
def test_fused_streams_across_calls():
    """Continuous batching on the GPU (Triton kernels, CUDA graphs): refills of scattered slots, cache growth and
    release, then smaller calls, each against the reference-order loop."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_decode import FakeTokenizer, small_model

    from omniasr_trt.llm import LLMTranscriber

    m = small_model(seed=7).float().cuda()
    m.rope = rope_table(m.dec_cfg, "cuda")
    t = LLMTranscriber(m, FakeTokenizer(), max_batch=3)
    dec = GreedyDecoder(m, max_batch=3, sync_every=4, fused=True, graphs=True, reserve=8)
    torch.manual_seed(8)
    for n, lengths in ((7, (250, 40, 5, 251, 90, 12, 248)), (2, (30, 8)), (5, (60, 252, 9, 33, 100)), (1, (17,))):
        ctxs = [torch.randn(k, 32, device="cuda") for k in lengths[:n]]
        assert dec.generate(ctxs) == [t.greedy_reference(c) for c in ctxs], (n, lengths)
        torch.cuda.synchronize()


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("beams", [1, 3, 5])
def test_beam_attention(dtype, beams):
    torch.manual_seed(9)
    s, h, d, cap, gcap = 4, 8, 512, 600, 300
    r = s * beams
    kc = torch.randn(s + 1, h, cap, d, device="cuda", dtype=dtype)
    vc = torch.randn_like(kc)
    kg = torch.randn(r + 2, h, gcap, d, device="cuda", dtype=dtype)
    vg = torch.randn_like(kg)
    q = torch.randn(r, h, d, device="cuda", dtype=dtype)
    ctx_len = torch.tensor([600, 333, 1, 0], device="cuda")  # the last input is finished: it reads nothing
    gen_len = torch.randint(0, gcap, (r,), device="cuda")
    gen_len[-beams:] = 0
    ptr = torch.randint(0, r, (r, gcap), device="cuda", dtype=torch.int32)
    out = kernels.beam_attention(q, kc, vc, ctx_len, kg, vg, ptr, gen_len, beams, 640, d**-0.5)
    for row in range(r):
        g = row // beams
        nc, ng = int(ctx_len[g]), int(gen_len[row])
        if nc + ng == 0:
            assert torch.all(out[row] == 0)
            continue
        j = torch.arange(ng, device="cuda")
        k = torch.cat([kc[g, :, :nc], kg[ptr[row, :ng].long(), :, j].transpose(0, 1)], 1).float()
        v = torch.cat([vc[g, :, :nc], vg[ptr[row, :ng].long(), :, j].transpose(0, 1)], 1).float()
        att = torch.softmax((q[row].float()[:, None] @ k.transpose(-1, -2)) * d**-0.5, -1)
        tol = 1e-4 if dtype == torch.float32 else 2e-2
        torch.testing.assert_close(out[row].float(), (att @ v)[:, 0], atol=tol, rtol=tol)


@cuda
@pytest.mark.parametrize("beams,length_norm", [(3, False), (5, True)])
def test_fused_beam_decoder(beams, length_norm):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_decode import FakeTokenizer, small_model

    from omniasr_trt.beam import BeamDecoder
    from omniasr_trt.llm import LLMTranscriber

    m = small_model(seed=16).float().cuda()
    m.rope = rope_table(m.dec_cfg, "cuda")
    t = LLMTranscriber(m, FakeTokenizer())
    dec = BeamDecoder(m, beams=beams, length_norm=length_norm, max_batch=3, sync_every=4, fused=True, graphs=True,
                      reserve=8)
    torch.manual_seed(17)
    for lengths in ((250, 40, 5, 251, 90), (30, 8), (60, 252, 9, 33)):
        ctxs = [torch.randn(k, 32, device="cuda") for k in lengths]
        assert dec.generate(ctxs) == [t.beam_reference(c, beams, length_norm) for c in ctxs], lengths
        torch.cuda.synchronize()
