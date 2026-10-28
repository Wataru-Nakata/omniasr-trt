"""The omniASR character tokenizer (a SentencePiece model; fairseq2 ``RawSentencePieceTokenizer``)."""

from __future__ import annotations

import sentencepiece as spm


class Tokenizer:
    def __init__(self, path):
        self.sp = spm.SentencePieceProcessor(model_file=str(path))

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode([int(i) for i in ids])

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text)

    def __len__(self) -> int:
        return self.sp.get_piece_size()
