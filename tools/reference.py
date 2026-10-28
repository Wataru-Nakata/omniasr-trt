"""The same manifest through omnilingual-asr (the reference), for speed and agreement comparisons.

Needs omnilingual-asr and fairseq2 (not dependencies of omniasr-trt), e.g. ``pip install omnilingual-asr==0.2.0``.

    python tools/reference.py --card omniASR_LLM_Unlimited_7B_v2 --checkpoint omniASR-LLM-Unlimited-7B-v2.pt \\
        --tokenizer omniASR_tokenizer_written_v2.model --manifest data.jsonl --batch 8 --out ref-llm.json

Greedy search by default (``nbest=1``, as omnilingual-asr's pipeline; ``--nbest`` for its beam search), inputs
longest first in calls of ``--batch``, the tail padding of ``scripts/bench.py``; conditioned and unconditioned
inputs go to separate ``transcribe`` calls (0.2.0 cannot mix them). With ``--dtype float32`` this is the agreement
reference of the README. The CTC card takes inputs up to 40 s only (omnilingual-asr's limit; ``scripts/bench.py
ctc`` windows longer ones).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bench import read_manifest, tail_padding  # noqa: E402


def load(card: str, checkpoint: Path, tokenizer: Path, dtype):
    """The official model card's architecture with local files in place of the downloads."""
    from fairseq2.assets import AssetCard
    from fairseq2.assets.store import get_asset_store
    from fairseq2.data.tokenizers.hub import load_tokenizer
    from fairseq2.models.hub import load_model

    store = get_asset_store()
    official = store.retrieve_card(card)
    token_card = store.retrieve_card(official.field("tokenizer_ref").as_(str))
    model = load_model(AssetCard(f"{card}-local", {"checkpoint": str(checkpoint)}, base=official),
                       device=torch.device("cuda"), dtype=dtype, mmap=True, progress=False).eval()
    tok = load_tokenizer(AssetCard(f"{token_card.name}-local", {"tokenizer": str(tokenizer)}, base=token_card),
                         progress=False)
    return model, tok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--card", required=True, help="omniASR_LLM_Unlimited_7B_v2 or omniASR_CTC_7B_v2")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--nbest", type=int, default=1, help="beam size (1: greedy, the pipeline's default)")
    p.add_argument("--length-norm", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    from omnilingual_asr.models.wav2vec2_llama.config import Wav2Vec2LlamaBeamSearchConfig

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from omniasr_trt.cli import load_wave

    dtype = getattr(torch, args.dtype)
    items = read_manifest(args.manifest)
    waves = [load_wave(it["audio"]) for it in items]
    waves = [np.pad(w, (0, tail_padding(len(w)))) for w in waves]
    langs = [it.get("lang") for it in items]
    t0 = time.monotonic()
    model, tok = load(args.card, args.checkpoint, args.tokenizer, dtype)
    load_seconds = time.monotonic() - t0
    pipe = ASRInferencePipeline(None, model=model, tokenizer=tok, dtype=dtype,
                                beam_search_config=Wav2Vec2LlamaBeamSearchConfig(nbest=args.nbest,
                                                                                 length_norm=args.length_norm))
    ctc = "CTC" in args.card
    order = sorted(range(len(waves)), key=lambda i: -len(waves[i]))
    texts = [None] * len(waves)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    for k in range(0, len(order), args.batch):
        chosen = order[k : k + args.batch]
        for conditioned in ((None,) if ctc else (True, False)):
            sub = [i for i in chosen if conditioned is None or (langs[i] is not None) == conditioned]
            if not sub:
                continue
            outs = pipe.transcribe([{"waveform": waves[i], "sample_rate": 16000} for i in sub],
                                   lang=[langs[i] for i in sub] if conditioned else None, batch_size=args.batch)
            for i, text in zip(sub, outs, strict=True):
                texts[i] = text
    torch.cuda.synchronize()
    wall = time.monotonic() - started
    seconds = sum(len(w) for w in waves) / 16000
    result = {"impl": "omnilingual-asr", "card": args.card, "dtype": args.dtype, "batch": args.batch,
              "nbest": args.nbest, "length_norm": args.length_norm,
              "inputs": len(waves), "audio_seconds": seconds, "load_seconds": load_seconds,
              "passes": [{"wall_seconds": wall, "rtfx": seconds / wall,
                          "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}],
              "ids": [it.get("id", it["audio"]) for it in items], "texts": texts}
    args.out.write_text(json.dumps(result, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k not in ("ids", "texts")}))


if __name__ == "__main__":
    main()
