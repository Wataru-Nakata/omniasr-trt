"""omniASR CTC emissions of inputs of any length, for forced alignment.

Inputs up to ``max_single_seconds`` run in one pass; longer ones in ``window_seconds`` windows with
``context_seconds`` on both sides, whose centre frames are stitched into one ``[T, V]`` sequence. Each piece is
normalised on its own (zero mean, unit variance in fp32), pieces are batched longest first up to
``batch_seconds`` of padded audio.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .modeling import HOP, CTCModel


class CTCRunner:
    def __init__(self, model: CTCModel, *, window_seconds: float = 20, context_seconds: float = 5,
                 max_single_seconds: float = 30, batch_seconds: float = 320):
        self.model = model
        self.dtype = model.final_proj.weight.dtype
        self.device = model.final_proj.weight.device
        self.single = int(max_single_seconds * 16000)
        self.window = int(window_seconds * 16000)
        self.context = int(context_seconds * 16000)
        self.batch_samples = int(batch_seconds * 16000)

    def windows(self, samples: int) -> list[tuple[int, int, int, int | None]]:
        """Each window's sample span and the global frame range [lo, hi) it contributes."""
        if samples <= self.single:
            return [(0, samples, 0, None)]
        spans = []
        for lo in range(0, samples, self.window):
            hi = lo + self.window
            spans.append((max(0, lo - self.context), min(samples, hi + self.context),
                          lo // HOP, None if hi >= samples else hi // HOP))
        return spans

    @torch.inference_mode()
    def forward(self, pieces: list) -> list[Tensor]:
        """Logits ``[F, V]`` (model dtype, on the GPU) of each waveform, batched by padded samples."""
        outputs = [None] * len(pieces)
        order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i]))
        while order:
            longest = len(pieces[order[0]])
            count = max(1, min(len(order), self.batch_samples // max(longest, 1)))
            batch, order = order[:count], order[count:]
            lengths = torch.tensor([len(pieces[i]) for i in batch])
            seqs = torch.zeros(len(batch), longest, device=self.device, dtype=self.dtype)
            for row, i in enumerate(batch):
                wave = torch.from_numpy(np.ascontiguousarray(pieces[i], dtype=np.float32))
                seqs[row, : len(wave)] = torch.nn.functional.layer_norm(wave, wave.shape).to(self.device, self.dtype)
            padded = bool((lengths < longest).any())
            logits, frames = self.model(seqs, lengths if padded else None)
            for row, i in enumerate(batch):
                outputs[i] = logits[row, : frames[row]]
        return outputs

    def emissions(self, waves: list) -> tuple[list[Tensor], list[int]]:
        """Stitched logits ``[T, V]`` of each waveform and its window count."""
        spans = [self.windows(len(wave)) for wave in waves]
        pieces = [wave[start:end] for wave, ss in zip(waves, spans, strict=True) for start, end, _, _ in ss]
        logits = iter(self.forward(pieces))
        stitched = []
        for ss in spans:
            parts = []
            for (start, _, lo, hi), frames in zip(ss, logits):
                offset = start // HOP
                parts.append(frames[lo - offset : None if hi is None else hi - offset])
            stitched.append(parts[0] if len(parts) == 1 else torch.cat(parts))
        return stitched, [len(ss) for ss in spans]
