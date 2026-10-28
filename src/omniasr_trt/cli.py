"""omniasr-trt command line: weight conversion and transcription.

    omniasr-trt convert omniASR-LLM-Unlimited-7B-v2.pt llm-bf16.safetensors
    omniasr-trt transcribe --model llm-bf16.safetensors --tokenizer omniASR_tokenizer_written_v2.model \\
        --lang ben_Beng speech.wav
    omniasr-trt transcribe --ctc --model ctc-bf16.safetensors --tokenizer ... speech.wav
"""

import argparse
import json

import numpy as np


def load_wave(path: str) -> np.ndarray:
    """Mono 16 kHz float32 (scipy's polyphase resampler for other rates)."""
    import soundfile as sf

    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    if rate != 16000:
        try:
            from math import gcd

            from scipy.signal import resample_poly
        except ImportError as e:
            raise SystemExit(f"{path}: {rate} Hz; resample to 16 kHz first (or install scipy)") from e
        g = gcd(rate, 16000)
        wave = resample_poly(wave, 16000 // g, rate // g).astype(np.float32)
    return wave


def main():
    p = argparse.ArgumentParser(prog="omniasr-trt", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("convert", help="official .pt -> .safetensors (weight norm folded)")
    c.add_argument("src")
    c.add_argument("dst")
    c.add_argument("--dtype", default="bfloat16")
    t = sub.add_parser("transcribe")
    t.add_argument("audio", nargs="+")
    t.add_argument("--model", required=True, help=".safetensors from `convert`, or the official .pt")
    t.add_argument("--tokenizer", help="default: omniASR_tokenizer_written_v2.model beside the model")
    t.add_argument("--ctc", action="store_true", help="the CTC model (greedy); default: the LLM model")
    t.add_argument("--lang", help="omnilingual language id, e.g. eng_Latn (LLM only; default: unconditioned)")
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--dtype", default="bfloat16")
    t.add_argument("--json", action="store_true")
    args = p.parse_args()
    import torch

    if args.command == "convert":
        from .modeling import convert

        convert(args.src, args.dst, getattr(torch, args.dtype))
        return
    waves = [load_wave(a) for a in args.audio]
    dtype = getattr(torch, args.dtype)
    if args.ctc:
        from . import OmniASRCTC

        texts = OmniASRCTC.from_pretrained(args.model, args.tokenizer, dtype=dtype).transcribe(waves)
    else:
        from . import OmniASRLLM

        model = OmniASRLLM.from_pretrained(args.model, args.tokenizer, dtype=dtype, max_batch=args.batch_size)
        texts = []
        for k in range(0, len(waves), args.batch_size):
            chunk = waves[k : k + args.batch_size]
            texts += model.transcribe(chunk, [args.lang] * len(chunk))
    for path, text in zip(args.audio, texts):
        print(json.dumps({"audio": path, "text": text}, ensure_ascii=False) if args.json else f"{path}\t{text}")


if __name__ == "__main__":
    main()
