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
                        beam_size=1, length_norm=False, drafter=None, draft_tokens=4, **options):
        """Greedy search proposes ``draft_tokens`` per decoder step (speculative decoding: the same tokens in fewer
        steps; 0: one token per step) from each row's own tokens and, with ``drafter`` (an omniASR CTC checkpoint,
        e.g. omniASR-CTC-300M-v2.pt, or an :class:`OmniASRCTC`), from its transcript."""
        from .llm import LLMTranscriber
        from .modeling import load_checkpoint
        from .tokenizer import Tokenizer

        model = load_checkpoint("llm", checkpoint, dtype, device)
        tok = Tokenizer(_tokenizer_path(checkpoint, tokenizer))
        if drafter is not None and not isinstance(drafter, OmniASRCTC):
            drafter = OmniASRCTC.from_pretrained(drafter, _tokenizer_path(checkpoint, tokenizer), dtype=dtype,
                                                 device=device)
        return cls(LLMTranscriber(model, tok, max_batch=max_batch, beams=beam_size, length_norm=length_norm,
                                  drafter=drafter, draft_tokens=draft_tokens, **options))

    def transcribe(self, waves, langs=None, *, mode="fast", beam_size=None, length_norm=None, drafts=None):
        """``beam_size`` > 1: omnilingual-asr's beam search (default: ``from_pretrained(beam_size=...)``, else 1).
        ``drafts``: per input the CTC argmax of each 20 ms frame (e.g. ``OmniASRCTC.frame_ids(waves)``), instead of
        the drafter's, for the greedy search."""
        return self.transcriber.transcribe(waves, langs, mode=mode, beams=beam_size, length_norm=length_norm,
                                           drafts=drafts)


class OmniASRCTC:
    """omniASR CTC emissions (for alignment) and greedy transcripts."""

    def __init__(self, runner, tokenizer):
        self.runner = runner
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, checkpoint, tokenizer=None, *, dtype=torch.bfloat16, device="cuda", **options):
        from .ctc import CTCRunner
        from .modeling import load_checkpoint
        from .tokenizer import Tokenizer

        model = load_checkpoint("ctc", checkpoint, dtype, device)
        return cls(CTCRunner(model, **options), Tokenizer(_tokenizer_path(checkpoint, tokenizer)))

    def emissions(self, waves):
        return self.runner.emissions(waves)

    def forward(self, pieces):
        return self.runner.forward(pieces)

    def frame_ids(self, waves) -> list:
        """Per input the argmax of each 20 ms frame (int64, CPU): drafts for :class:`OmniASRLLM`. A few minutes of
        audio at a time, so that the logits of long calls are never all held."""
        out, k = [], 0
        while k < len(waves):
            n, samples = 1, len(waves[k])
            while k + n < len(waves) and samples + len(waves[k + n]) <= self.runner.batch_samples:
                samples += len(waves[k + n])
                n += 1
            out += [f.argmax(-1).cpu() for f in self.emissions(waves[k : k + n])[0]]
            k += n
        return out

    def transcribe(self, waves):
        frames, _ = self.emissions(waves)
        return [self.tokenizer.decode([int(i) for i in torch.unique_consecutive(f.argmax(-1)) if int(i) != 0])
                for f in frames]


__all__ = ["OmniASRLLM", "OmniASRCTC"]
