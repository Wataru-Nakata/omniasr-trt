"""Drop-in pieces for code written against omnilingual-asr / fairseq2.

* :class:`Pipeline` -- omnilingual-asr's ``ASRInferencePipeline.transcribe(inp, lang=..., batch_size=...)`` for the
  LLM Unlimited model, with a ``streaming_config`` (``is_streaming``, ``segment_secs``) as callers read it and its
  ``beam_search_config`` (``nbest``, ``length_norm``). Mixed language ids and ``None`` may share one call
  (omnilingual-asr 0.2.0 needs two).
* :class:`CTC` -- ``forward(pieces)`` -> per piece logits ``[F, V]`` (bf16, GPU) of 16 kHz waveforms, each
  normalised on its own and batched by padded length, for code that windows and stitches emissions itself.
* :class:`TokenizerShim` -- ``create_encoder()`` / ``create_decoder(skip_special_tokens=...)`` / ``vocab_info``
  of the fairseq2 char tokenizer, on SentencePiece.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import OmniASRCTC, OmniASRLLM
from .tokenizer import Tokenizer


@dataclass(frozen=True)
class StreamingConfig:
    is_streaming: bool = True
    segment_secs: float = 15.0
    sample_rate: int = 16000


def search_settings(config) -> tuple[int, bool]:
    """``nbest`` / ``length_norm`` of an omnilingual-asr ``Wav2Vec2LlamaBeamSearchConfig`` (or a dict); None:
    omnilingual-asr's pipeline default, one hypothesis."""
    if config is None:
        return 1, False
    get = config.get if isinstance(config, dict) else lambda k, d: getattr(config, k, d)
    return int(get("nbest", 1)), bool(get("length_norm", False))


class Pipeline:
    def __init__(self, model: OmniASRLLM, beam_search_config=None):
        self.model = model
        self.streaming_config = StreamingConfig()
        self.beams, self.length_norm = search_settings(beam_search_config)

    @classmethod
    def from_pretrained(cls, checkpoint, tokenizer=None, beam_search_config=None, **options) -> "Pipeline":
        return cls(OmniASRLLM.from_pretrained(checkpoint, tokenizer, **options), beam_search_config)

    @torch.inference_mode()
    def transcribe(self, inp, *, lang=None, batch_size: int = 8) -> list[str]:
        waves = []
        for item in inp:
            if int(item["sample_rate"]) != 16000:
                raise ValueError("omniasr_trt.compat.Pipeline takes 16 kHz waveforms")
            wave = np.asarray(item["waveform"], dtype=np.float32)
            waves.append(wave.mean(axis=0) if wave.ndim == 2 and wave.shape[0] < wave.shape[1] else
                         wave.mean(axis=1) if wave.ndim == 2 else wave)
        langs = list(lang) if lang else [None] * len(waves)
        texts = []
        for k in range(0, len(waves), batch_size):  # omnilingual-asr's buckets, in the given order
            texts += self.model.transcribe(waves[k : k + batch_size], langs[k : k + batch_size],
                                           beam_size=self.beams, length_norm=self.length_norm)
        return texts


class TokenizerShim:
    class VocabInfo:
        def __init__(self, tok: Tokenizer):
            sp = tok.sp
            self.size = sp.get_piece_size()
            self.unk_idx = sp.unk_id()
            self.bos_idx, self.eos_idx, self.pad_idx = sp.bos_id(), sp.eos_id(), sp.pad_id()

    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self.vocab_info = TokenizerShim.VocabInfo(tok)

    def create_encoder(self, **_):
        return lambda text: torch.tensor(self.tok.encode(text), dtype=torch.int64)

    def create_decoder(self, *, skip_special_tokens: bool = False):
        return lambda ids: self.tok.decode([int(i) for i in (ids.tolist() if hasattr(ids, "tolist") else ids)])


class CTC:
    def __init__(self, model: OmniASRCTC):
        self.ctc = model
        self.model = None
        self.tokenizer = TokenizerShim(model.tokenizer)

    @classmethod
    def from_pretrained(cls, checkpoint, tokenizer=None, **options) -> "CTC":
        """``options``: ``window_seconds``, ``context_seconds``, ``max_single_seconds``, ``batch_seconds``."""
        return cls(OmniASRCTC.from_pretrained(checkpoint, tokenizer, **options))

    def forward(self, pieces: list) -> list:
        return self.ctc.forward(pieces)
