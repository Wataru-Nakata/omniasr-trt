"""Character edit distances between transcript sets (numpy only; transcripts of 10-minute inputs are ~8k characters)."""

import unicodedata

import numpy as np


def levenshtein(a: str, b: str) -> int:
    """Row-by-row DP; within a row cur[j] = min_k<=j (t[k] + j - k) is a running minimum, so each row is O(n) numpy."""
    if a == b:
        return 0
    if not a or not b:
        return len(a) + len(b)
    x = np.frombuffer(a.encode("utf-32-le"), dtype=np.uint32)
    y = np.frombuffer(b.encode("utf-32-le"), dtype=np.uint32)
    idx = np.arange(len(y) + 1)
    prev = idx.copy()
    for i, c in enumerate(x, 1):
        t = np.empty_like(prev)
        t[0] = i
        t[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (y != c))
        prev = np.minimum.accumulate(t - idx) + idx
    return int(prev[-1])


def normalize(text: str) -> str:
    """NFKC, lower case, no punctuation or symbols, single spaces (for CER against captions)."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = "".join(c if unicodedata.category(c)[0] not in "PS" else " " for c in text)
    return " ".join(text.split())


def compare(ref: list, hyp: list) -> dict:
    pairs = list(zip(ref, hyp, strict=True))
    edits = sum(levenshtein(a or "", b or "") for a, b in pairs)
    chars = sum(len(a or "") for a, _ in pairs)
    return {"identical": sum(a == b for a, b in pairs), "inputs": len(pairs), "character_edit_rate": edits / max(1, chars)}


def cer(captions: list, hyp: list) -> dict:
    pairs = [(normalize(c), normalize(h)) for c, h in zip(captions, hyp, strict=True) if c]
    edits = sum(levenshtein(c, h) for c, h in pairs)
    return {"inputs": len(pairs), "cer": edits / max(1, sum(len(c) for c, _ in pairs))}


if __name__ == "__main__":  # self-check against the plain DP
    import random

    def plain(a, b):
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    rng = random.Random(0)
    for _ in range(300):
        a = "".join(rng.choice("abcdé ") for _ in range(rng.randint(0, 30)))
        b = "".join(rng.choice("abcdé ") for _ in range(rng.randint(0, 30)))
        assert levenshtein(a, b) == plain(a, b), (a, b)
    print("ok")
