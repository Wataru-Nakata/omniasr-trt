"""Padded encoder batches: equal-length attention runs, the key mask and one row at a time agree (CPU, fp64)."""

import torch

from omniasr_trt import modeling
from omniasr_trt.modeling import CTCModel, EncoderConfig, fuse_encoder, runs


def small_ctc():
    torch.manual_seed(0)
    cfg = EncoderConfig(model_dim=16, num_layers=3, num_heads=2, ffn_dim=32, feature_dim=8,
                        conv_layers=((8, 10, 5), (8, 3, 2), (8, 2, 2)), pos_kernel=4, pos_groups=2)
    m = CTCModel(cfg, vocab_size=11).double().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.3)
    return m


def test_runs():
    assert runs([5, 5, 3, 3, 2]) == [(0, 2, 5), (2, 4, 3), (4, 5, 2)]
    assert runs([5, 3, 5]) is None


def test_padded_batch_matches_single_rows():
    m = small_ctc()
    lens = torch.tensor([2000, 2000, 1700, 1700, 1100])
    x = torch.randn(len(lens), 2000, dtype=torch.float64)
    for i, n in enumerate(lens):
        x[i, n:] = 0
    with torch.no_grad():
        by_runs, frames = m(x, lens)
        original = modeling.runs
        modeling.runs = lambda lengths: None  # force the key-mask path
        try:
            by_mask, _ = m(x, lens)
        finally:
            modeling.runs = original
        fuse_encoder(m)
        fused, _ = m(x, lens)
        for i, n in enumerate(lens):
            alone, f = m(x[i : i + 1, :n])
            assert int(f[0]) == int(frames[i])
            torch.testing.assert_close(by_runs[i, : frames[i]], alone[0], rtol=1e-9, atol=1e-9)
            torch.testing.assert_close(by_mask[i, : frames[i]], alone[0], rtol=1e-9, atol=1e-9)
            torch.testing.assert_close(fused[i, : frames[i]], alone[0], rtol=1e-9, atol=1e-9)
