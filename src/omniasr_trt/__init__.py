"""Fast omniASR (Omnilingual ASR) inference without fairseq2."""

from __future__ import annotations

from pathlib import Path

import torch

TOKENIZER = "omniASR_tokenizer_written_v2.model"


def _tokenizer_path(checkpoint, tokenizer):
    if tokenizer is not None:
        return Path(tokenizer)
    return Path(checkpoint).with_name(TOKENIZER)


class OmniASRLLM:
    """omniASR LLM Unlimited (streaming, greedy) transcripts.

    >>> m = OmniASRLLM.from_pretrained("omniASR-LLM-Unlimited-7B-v2.pt")
    >>> m.transcribe([wave16k], langs=["hin_Deva"])
    """

    def __init__(self, transcriber):
        self.transcriber = transcriber

    @classmethod
    def from_pretrained(cls, checkpoint, tokenizer=None, *, dtype=torch.bfloat16, device="cuda", max_batch=8,
                        beam_size=1, length_norm=False, **options):
        from .llm import LLMTranscriber
        from .modeling import build, load_checkpoint
        from .tokenizer import Tokenizer

        model = load_checkpoint(build("llm"), checkpoint, dtype, device)
        return cls(LLMTranscriber(model, Tokenizer(_tokenizer_path(checkpoint, tokenizer)), max_batch=max_batch,
                                  beams=beam_size, length_norm=length_norm, **options))

    def transcribe(self, waves, langs=None, *, mode="fast", beam_size=None, length_norm=None):
        """``beam_size`` > 1: omnilingual-asr's beam search (default: ``from_pretrained(beam_size=...)``, else 1)."""
        return self.transcriber.transcribe(waves, langs, mode=mode, beams=beam_size, length_norm=length_norm)


class OmniASRCTC:
    """omniASR CTC emissions (for alignment) and greedy transcripts."""

    def __init__(self, runner, tokenizer):
        self.runner = runner
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, checkpoint, tokenizer=None, *, dtype=torch.bfloat16, device="cuda", **options):
        from .ctc import CTCRunner
        from .modeling import build, load_checkpoint
        from .tokenizer import Tokenizer

        model = load_checkpoint(build("ctc"), checkpoint, dtype, device)
        return cls(CTCRunner(model, **options), Tokenizer(_tokenizer_path(checkpoint, tokenizer)))

    def emissions(self, waves):
        return self.runner.emissions(waves)

    def forward(self, pieces):
        return self.runner.forward(pieces)

    def transcribe(self, waves):
        frames, _ = self.emissions(waves)
        return [self.tokenizer.decode([int(i) for i in torch.unique_consecutive(f.argmax(-1)) if int(i) != 0])
                for f in frames]


__all__ = ["OmniASRLLM", "OmniASRCTC"]
