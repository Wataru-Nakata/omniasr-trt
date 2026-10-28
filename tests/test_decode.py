"""The batched decoder against the reference-order loop on a small random model (CPU, fp64)."""

import numpy as np
import torch

from omniasr_trt import llm
from omniasr_trt.decode import GreedyDecoder
from omniasr_trt.llm import LLMTranscriber, compression_ratio, early_stop_index
from omniasr_trt.modeling import DecoderConfig, EncoderConfig, LLMModel


class FakeTokenizer:
    def decode(self, ids):
        return " ".join(map(str, ids))


def small_model(seed=0, vocab=12):
    torch.manual_seed(seed)
    enc = EncoderConfig(model_dim=16, num_layers=2, num_heads=2, ffn_dim=32, feature_dim=8,
                        conv_layers=((8, 10, 5), (8, 3, 2), (8, 2, 2)), pos_kernel=4, pos_groups=2)
    dec = DecoderConfig(model_dim=32, num_layers=2, num_heads=2, ffn_dim=24, vocab_size=vocab, num_langs=5,
                        max_seq_len=8192)
    m = LLMModel(enc, dec).double().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.5)
        m.final_proj.weight.mul_(4)  # peaked distributions: EOS within tens of steps
    return m


def test_fast_matches_reference_order():
    m = small_model()
    t = LLMTranscriber(m, FakeTokenizer(), max_batch=4)
    dec = GreedyDecoder(m, max_batch=4, sync_every=5, graphs=False, reserve=32)
    torch.manual_seed(1)
    for trial in range(6):
        lens = [int(x) for x in torch.randint(3, 40, (3,))]
        ctxs = [torch.randn(n, 32, dtype=torch.float64) for n in lens]
        ref = [t.greedy_reference(c) for c in ctxs]
        fast = dec.generate(ctxs)
        assert fast == ref, (trial, lens)


def test_end_to_end_streaming_modes_agree():
    m = small_model(seed=3)
    t = LLMTranscriber(m, FakeTokenizer(), max_batch=3)
    rng = np.random.default_rng(0)
    # 2.5 segments, 1 segment and a short input (segment of 15 s scaled down by a smaller SEGMENT)
    old = llm.SEGMENT
    llm.SEGMENT = 2000
    try:
        waves = [rng.standard_normal(5000).astype(np.float32), rng.standard_normal(2000).astype(np.float32),
                 rng.standard_normal(900).astype(np.float32)]
        a = t.transcribe(waves, ["aae_Latn", None, "AAL_latn"], mode="reference")
        t._decoder = GreedyDecoder(m, max_batch=3, sync_every=7, graphs=False, reserve=16)
        b = t.transcribe(waves, ["aae_Latn", None, "AAL_latn"], mode="fast")
        # fewer slots than inputs: finished slots take the next segment or the next input
        t._decoder = GreedyDecoder(m, max_batch=2, sync_every=3, graphs=False, reserve=16)
        c = t.transcribe(waves, ["aae_Latn", None, "AAL_latn"], mode="fast")
    finally:
        llm.SEGMENT = old
    assert a == b == c


def test_early_stop_rule():
    out = np.full(1000, 1, dtype=np.int32)
    out[300:] = 7  # a repeated token compresses far beyond 4x
    assert compression_ratio(out[400:500]) > 4.0
    # checks at multiples of 250 only, and only past ctx + 100
    assert early_stop_index(out, 240, 600, ctx=300) == 500
    assert early_stop_index(out, 240, 499, ctx=300) is None
    assert early_stop_index(out, 240, 800, ctx=420) == 750


def test_cache_grows_then_shrinks():
    m = small_model(seed=5)
    t = LLMTranscriber(m, FakeTokenizer(), max_batch=3)
    dec = GreedyDecoder(m, max_batch=3, sync_every=5, graphs=False, reserve=1)
    torch.manual_seed(6)
    # Contexts just under the first allocation (one 256-key bucket): a few generated tokens make the cache grow.
    ctxs = [torch.randn(n, 32, dtype=torch.float64) for n in (250, 251, 252)]
    ref = [t.greedy_reference(c) for c in ctxs]
    assert max(len(r) for r in ref) > 8
    assert dec.generate(ctxs) == ref
    assert dec.stats["grows"] >= 1 and dec.capacity == 0  # released after growing
    # the next calls start from a smaller cache again
    assert dec.generate(ctxs[:1]) == ref[:1]
    assert dec.generate(ctxs) == ref
