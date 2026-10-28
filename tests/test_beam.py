"""Beam search: the batched decoder against the reference-order loop on small random models (CPU, fp64)."""

import numpy as np
import pytest
import torch
from test_decode import FakeTokenizer, small_model

from omniasr_trt import llm
from omniasr_trt.beam import BeamDecoder
from omniasr_trt.llm import LLMTranscriber


def contexts(seed, lengths, dim=32, **kw):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, dim, generator=g, dtype=torch.float64, **kw) for n in lengths]


def test_one_beam_is_greedy():
    m = small_model(seed=11)
    t = LLMTranscriber(m, FakeTokenizer())
    for c in contexts(1, (5, 30, 17)):
        assert t.beam_reference(c, 1) == t.greedy_reference(c)


@pytest.mark.parametrize("beams,length_norm", [(2, False), (5, False), (5, True), (3, True)])
def test_fast_matches_reference_order(beams, length_norm):
    m = small_model(seed=12)
    t = LLMTranscriber(m, FakeTokenizer())
    ctxs = contexts(2, (6, 40, 13, 25, 9))
    ref = [t.beam_reference(c, beams, length_norm) for c in ctxs]
    assert len({tuple(r) for r in ref}) > 1
    # fewer slots than inputs: finished groups take the next input
    dec = BeamDecoder(m, beams=beams, length_norm=length_norm, max_batch=2, sync_every=5, graphs=False, reserve=16)
    assert dec.generate(ctxs) == ref
    assert dec.generate(ctxs[:2]) == ref[:2]


def test_beam_differs_from_greedy_somewhere():
    m = small_model(seed=13)
    t = LLMTranscriber(m, FakeTokenizer())
    ctxs = contexts(3, (10, 20, 30, 40, 50, 15))
    assert any(t.beam_reference(c, 4) != t.greedy_reference(c) for c in ctxs)


def test_streaming_modes_agree():
    m = small_model(seed=14)
    t = LLMTranscriber(m, FakeTokenizer(), max_batch=2, beams=3)
    rng = np.random.default_rng(1)
    old = llm.SEGMENT
    llm.SEGMENT = 2000
    try:
        waves = [rng.standard_normal(n).astype(np.float32) for n in (5000, 2000, 900, 4100)]
        langs = ["aae_Latn", None, "aal_Latn", None]
        a = t.transcribe(waves, langs, mode="reference")
        t._beam_decoders[3, False] = BeamDecoder(m, beams=3, max_batch=2, sync_every=4, graphs=False, reserve=16)
        b = t.transcribe(waves, langs)
        assert t.transcribe(waves, langs, beams=1) == t.transcribe(waves, langs, mode="reference", beams=1)
    finally:
        llm.SEGMENT = old
    assert a == b


@pytest.mark.parametrize("beams,length_norm", [(1, False), (3, False), (4, True)])
def test_repetition_stop_lands_on_the_same_step(beams, length_norm):
    """EOS made (nearly) impossible: decoding ends by the repetition stop, which must fire at the step it fires in
    the reference loop (it changes the competition between beams afterwards)."""
    m = small_model(seed=15)
    with torch.no_grad():
        m.final_proj.weight[llm.EOS] = 0.0  # EOS logit 0 against peaked others: practically never the best
    t = LLMTranscriber(m, FakeTokenizer())
    ctxs = contexts(4, (5, 60, 140, 151, 30, 99))  # checks at t = 250, 500 relative to different context lengths
    ref = [t.beam_reference(c, beams, length_norm) for c in ctxs]
    dec = BeamDecoder(m, beams=beams, length_norm=length_norm, max_batch=3, sync_every=7, graphs=False, reserve=64)
    assert dec.generate(ctxs) == ref
    assert dec.stats["repetition_stops"] >= 1  # the stop fired, and the texts still agree
    assert dec.stats["grows"] >= 1 and dec.ccap == dec.gcap == 0  # > 256 generated positions: grown, then released
    assert dec.generate(ctxs[:1]) == ref[:1]
