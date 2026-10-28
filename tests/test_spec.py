"""Speculative (drafted) greedy decoding: the same tokens as the greedy search whatever the drafts (CUDA only)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_decode import FakeTokenizer, small_model
from torch import Tensor

from omniasr_trt import decode, kernels, llm, spec
from omniasr_trt.decode import GreedyDecoder
from omniasr_trt.llm import LLMTranscriber
from omniasr_trt.modeling import rope_table
from omniasr_trt.spec import AHEAD, BEHIND, DRAFT_CAPACITY, NGRAM, REACH, SELF_MIN, SELF_WINDOW, SpecDecoder

cuda = pytest.mark.skipif(not (torch.cuda.is_available() and kernels.available(torch.device("cuda"))),
                          reason="needs CUDA and Triton")


def cuda_model(seed):
    m = small_model(seed=seed).float().cuda()
    m.rope = rope_table(m.dec_cfg, "cuda")
    return m


def noisy(tokens, rng, vocab=12):
    """A draft with characters changed, dropped and added, and a tail of junk."""
    out = []
    for t in tokens:
        r = rng.random()
        if r < 0.1:
            out.append(int(rng.integers(3, vocab)))
        elif r < 0.15:
            continue
        elif r < 0.2:
            out += [t, int(rng.integers(3, vocab))]
        else:
            out.append(t)
    return out + [int(x) for x in rng.integers(3, vocab, 5)]


def reference_propose(self, b):
    """``SpecDecoder.propose`` in PyTorch (the kernel's reference): proposals [B, k], from the draft [B]."""
    dev, k = self.device, self.draft_tokens
    out, draft, last, first = self.out[:b], self.draft[:b], self.last[:b], self.first[:b]
    t = torch.arange(NGRAM, device=dev)
    hist_pos = last[:, None] - t  # [B, n]: tok, the token before, ...
    hist = out.gather(1, hist_pos.clamp(min=0))
    hist_ok = hist_pos >= first[:, None]

    def matching(source: Tensor, at: Tensor, ok: Tensor) -> Tensor:  # newest tokens equal before ``at`` [B, W]
        at = at[:, :, None] - t
        ok = ok[:, :, None] & (at >= 0) & hist_ok[:, None, :]
        same = source.gather(1, at.clamp(0, source.size(1) - 1).view(b, -1)).view_as(at) == hist[:, None, :]
        return (same & ok).long().cumprod(2).sum(2)

    # The draft: candidate positions of ``tok`` around the expected one.
    offset = torch.arange(-BEHIND, AHEAD + 1, device=dev)
    need = torch.full_like(offset, NGRAM)
    for reach, n in reversed(REACH):
        need = torch.where(offset.abs() <= reach, n, need)
    anchor = self.next[:b, None] - 1 + offset  # [B, W]
    score = matching(draft, anchor, anchor < self.draft_len[:b, None])
    rank = torch.where(score >= need, score * 256 - offset.abs(), -1)
    best = rank.argmax(1, keepdim=True)
    found = rank.gather(1, best)[:, 0] >= 0
    drafted = score.gather(1, best)[:, 0] * found
    start = torch.where(found, anchor.gather(1, best)[:, 0] + 1, self.next[:b])
    at = start[:, None] + torch.arange(k, device=dev)
    from_draft = torch.where(at < self.draft_len[:b, None], draft.gather(1, at.clamp(max=DRAFT_CAPACITY - 1)), 0)
    # The row's own tokens: the latest earlier position matching ``tok`` and the tokens before it.
    back = last[:, None] - 1 - torch.arange(SELF_WINDOW, device=dev)  # [B, Ws], latest first
    own = matching(out, back, back >= first[:, None])
    j = own.argmax(1, keepdim=True)  # the first maximum: the latest
    mine = own.gather(1, j)[:, 0]
    j = back.gather(1, j)[:, 0]
    period = (last - j).clamp(min=1)
    at = j[:, None] + 1 + torch.arange(k, device=dev) % period[:, None]
    from_self = out.gather(1, at.clamp(0, out.size(1) - 1))
    use_draft = (mine < SELF_MIN) | (mine <= drafted)
    start_out = start
    return torch.where(use_draft[:, None], from_draft.long(), from_self.long()), use_draft, start_out


def run(dec, ctxs, drafts):
    dec.drafts = lambda j: drafts[j]
    return dec.generate(ctxs)


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tokens", [1, 3, 5, 9])
def test_draft_attention(dtype, tokens):
    torch.manual_seed(2)
    b, h, d, cap = 4, 8, 512, 700
    k_cache = torch.randn(b + 1, h, cap, d, device="cuda", dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(b * tokens, h, d, device="cuda", dtype=dtype)
    lengths = torch.tensor([600, 257, 1, 0], device="cuda")  # 0: a finished row (its outputs are not used)
    out = kernels.draft_attention(q, k_cache, v_cache, lengths, tokens, 640, d**-0.5)
    for r in range(3):
        for i in range(tokens):
            n = int(lengths[r]) + i
            k, v = k_cache[r, :, :n].float(), v_cache[r, :, :n].float()
            att = torch.softmax((q[r * tokens + i].float()[:, None] @ k.transpose(-1, -2)) * d**-0.5, -1)
            tol = 1e-4 if dtype == torch.float32 else 2e-2
            torch.testing.assert_close(out[r * tokens + i].float(), (att @ v)[:, 0], atol=tol, rtol=tol)


@cuda
@pytest.mark.parametrize("draft_tokens", [1, 4, 8])
def test_same_tokens_as_greedy(draft_tokens):
    m = cuda_model(seed=21)
    t = LLMTranscriber(m, FakeTokenizer())
    torch.manual_seed(22)
    ctxs = [torch.randn(n, 32, device="cuda") for n in (40, 7, 251, 90, 13, 60)]
    ref = [t.greedy_reference(c) for c in ctxs]
    assert min(len(r) for r in ref) >= 3
    rng = np.random.default_rng(0)
    dec = SpecDecoder(m, draft_tokens=draft_tokens, max_batch=4, sync_every=3, reserve=8)
    exact = [(r + [5, 6, 7], 0) for r in ref]  # perfect drafts running on past EOS
    assert run(dec, ctxs, exact) == ref
    tokens_per_step = dec.stats["tokens"] / dec.stats["rows_steps"]
    assert run(dec, ctxs, [(noisy(r, rng), 0) for r in ref]) == ref
    assert run(dec, ctxs, [(noisy(r[3:], rng), 2) for r in ref]) == ref  # starting mid-draft
    assert run(dec, ctxs, [None, ([], 0), *[(list(rng.integers(3, 12, 30)), 0) for _ in ctxs[2:]]]) == ref
    if draft_tokens > 1:  # perfect drafts: several tokens per step
        assert tokens_per_step > 1.5


@cuda
def test_length_limit_and_eos_inside_a_step(monkeypatch):
    m = cuda_model(seed=23)
    with torch.no_grad():
        m.final_proj.weight[llm.EOS] *= 0.3  # long generations
    monkeypatch.setattr(decode, "LIMIT", 70)
    monkeypatch.setattr(spec, "LIMIT", 70)
    torch.manual_seed(24)
    ctxs = [torch.randn(n, 32, device="cuda") for n in (5, 30, 64, 69, 12)]
    ref = GreedyDecoder(m, max_batch=3, sync_every=4, reserve=8).generate(ctxs)
    dec = SpecDecoder(m, draft_tokens=6, max_batch=3, sync_every=4, reserve=8)
    assert run(dec, ctxs, [(r + [4] * 20, 0) for r in ref]) == ref


@cuda
def test_repetition_stop():
    m = cuda_model(seed=15)
    with torch.no_grad():
        m.final_proj.weight[llm.EOS] = 0.0  # no EOS: decoding ends by the repetition stop (or the cache grows)
    t = LLMTranscriber(m, FakeTokenizer())
    torch.manual_seed(25)
    ctxs = [torch.randn(n, 32, device="cuda") for n in (5, 60, 140, 151)]
    ref = [t.greedy_reference(c) for c in ctxs]
    rng = np.random.default_rng(3)
    dec = SpecDecoder(m, draft_tokens=4, max_batch=3, sync_every=5, reserve=8)
    dec.context_floor = 0
    assert run(dec, ctxs, [(noisy(r, rng), 0) for r in ref]) == ref
    assert dec.stats["grows"] >= 1


@cuda
def test_transcriber_with_frame_drafts():
    m = cuda_model(seed=3)
    t = LLMTranscriber(m, FakeTokenizer(), max_batch=3)
    rng = np.random.default_rng(0)
    old = llm.SEGMENT
    llm.SEGMENT = 2000
    try:
        waves = [rng.standard_normal(n).astype(np.float32) for n in (5000, 2000, 900)]
        langs = ["aae_Latn", None, "AAL_latn"]
        ref = t.transcribe(waves, langs, mode="reference")
        frames = [rng.integers(0, 12, n // 320) for n in (5000, 2000, 900)]  # random CTC argmax: any draft works
        assert t.transcribe(waves, langs, drafts=frames) == ref
        assert t._spec_decoder.stats["steps"] > 0
        assert t.transcribe(waves, langs, drafts=[None, frames[1], None]) == ref
        assert t.transcribe(waves, langs) == ref  # the default: own tokens only
        assert t.last_decoder is t._spec_decoder
        t.draft_tokens = 0  # one token per step
        assert t.transcribe(waves, langs) == ref
        assert t.last_decoder is t._decoder
    finally:
        llm.SEGMENT = old


@cuda
def test_own_tokens_without_draft():
    """No draft: a repetition loop is proposed from the row's own tokens (several tokens per step)."""
    m = cuda_model(seed=15)
    with torch.no_grad():
        m.final_proj.weight[llm.EOS] = 0.0
    t = LLMTranscriber(m, FakeTokenizer())
    torch.manual_seed(26)
    ctxs = [torch.randn(n, 32, device="cuda") for n in (5, 60, 99)]
    ref = [t.greedy_reference(c) for c in ctxs]
    dec = SpecDecoder(m, draft_tokens=4, max_batch=3, sync_every=5, reserve=8)
    assert run(dec, ctxs, [None] * len(ctxs)) == ref
    assert dec.stats["tokens"] / dec.stats["rows_steps"] > 1.5


@cuda
@pytest.mark.parametrize("k", [1, 4, 7])
def test_propose_kernel(k):
    g = torch.Generator(device="cuda").manual_seed(30 + k)
    b, cap = 8, 2048
    for trial in range(20):
        vocab = 4 + trial % 5  # few symbols: many partial matches, ties and repetitions
        out = torch.randint(3, vocab, (b, cap), device="cuda", dtype=torch.int32, generator=g)
        first = torch.randint(0, 600, (b,), device="cuda", generator=g)
        last = first + torch.randint(0, 1400, (b,), device="cuda", generator=g)
        last[0] = first[0]  # the segment's first token only
        out[torch.arange(cap, device="cuda")[None, :] < first[:, None]] = llm.PAD
        draft = torch.randint(3, vocab, (b, DRAFT_CAPACITY), device="cuda", dtype=torch.int32, generator=g)
        dlen = torch.randint(0, DRAFT_CAPACITY + 1, (b,), device="cuda", generator=g)
        dlen[1] = 0
        draft[torch.arange(DRAFT_CAPACITY, device="cuda")[None, :] >= dlen[:, None]] = -1
        nxt = (torch.rand(b, device="cuda", generator=g) * (dlen + 8)).long()
        st = SimpleNamespace(device=torch.device("cuda"), draft_tokens=k, out=out, draft=draft, draft_len=dlen,
                             last=last, first=first, next=nxt)
        want, want_draft, want_start = reference_propose(st, b)
        start = torch.empty(b, device="cuda", dtype=torch.long)
        got, got_draft = kernels.propose(out, draft, dlen, last, first, nxt, start, k, NGRAM, BEHIND, AHEAD, REACH,
                                         SELF_WINDOW, SELF_MIN)
        assert torch.equal(got_draft, want_draft), trial
        assert torch.equal(got, want), trial
        assert torch.equal(start, want_start), trial
