# omniasr-trt

Fast inference for Meta's [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) 7B models:
`omniASR_LLM_Unlimited_7B_v2` (transcripts of inputs of any length, greedy) and `omniASR_CTC_7B_v2` (CTC
emissions, e.g. for forced alignment, and greedy transcripts). Plain PyTorch with a few Triton kernels and CUDA
graphs; no fairseq2 or omnilingual-asr dependency. The official checkpoints load with `strict=True` (only the
positional convolution's weight norm is folded).

There is no TensorRT engine, despite the name: the gains come from the decoding loop, not from the GEMMs.

## Install

```bash
pip install git+https://github.com/Wataru-Nakata/omniasr-trt
```

Python >= 3.10, torch >= 2.8 (tested with 2.8 / Triton 3.4 and 2.14 / Triton 3.8), a CUDA GPU (developed on an H200;
the Triton kernels need CUDA, the rest also runs on the CPU).

The official files (as in omnilingual-asr's model cards):

```bash
wget https://dl.fbaipublicfiles.com/mms/omniASR-LLM-Unlimited-7B-v2.pt     # 31 GB, fp32
wget https://dl.fbaipublicfiles.com/mms/omniASR-CTC-7B-v2.pt               # 26 GB, fp32
wget https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer_written_v2.model
omniasr-trt convert omniASR-LLM-Unlimited-7B-v2.pt llm-bf16.safetensors    # once: 15.6 GB, loads in ~10-20 s
omniasr-trt convert omniASR-CTC-7B-v2.pt ctc-bf16.safetensors              # 13 GB
```

The official `.pt` files load directly too (cast at load). The tokenizer is looked up beside the checkpoint unless
given.

## Use

```python
from omniasr_trt import OmniASRCTC, OmniASRLLM

llm = OmniASRLLM.from_pretrained("llm-bf16.safetensors", "omniASR_tokenizer_written_v2.model")
llm.transcribe([wave_a, wave_b], langs=["ben_Beng", None])   # 16 kHz mono float32, any length; None: unconditioned

ctc = OmniASRCTC.from_pretrained("ctc-bf16.safetensors", "omniASR_tokenizer_written_v2.model")
frames, windows = ctc.emissions([wave])   # logits [T, 10288] per input (bf16, GPU), 20 ms frames
ctc.transcribe([wave])                    # greedy text
```

```bash
omniasr-trt transcribe --model llm-bf16.safetensors --lang ben_Beng speech.wav
omniasr-trt transcribe --ctc --model ctc-bf16.safetensors speech.wav
```

* LLM: inputs are cut into 15 s segments and segment i is decoded with segment i-1's audio and text as context, as
  omnilingual-asr's streaming pipeline does; language ids are omnilingual-asr's (`eng_Latn`, `ben_Beng`, ...).
  `max_batch` (default 8) is the number of inputs decoded at once.
* CTC: inputs up to `max_single_seconds` (30) in one pass, longer ones in `window_seconds` (20) windows with
  `context_seconds` (5) on both sides whose centre frames are stitched; each piece is normalised on its own.
* `omniasr_trt.compat` has drop-ins for code written against omnilingual-asr / fairseq2: `Pipeline`
  (`ASRInferencePipeline.transcribe(inp, lang=..., batch_size=...)`), `CTC` (`forward(pieces)` -> logits per
  piece) and a fairseq2-style tokenizer (`create_encoder`, `create_decoder`, `vocab_info`).

## Speed

Test setup:
* data: real-world speech segments from YouTube videos (not distributed): **llm** -- 160 segments in 11 languages
  (bn, te, ta, ml, yo, ne, hi, sw, km, kn, jv), 5 s to 10 min (median 40 s), 7.85 h; **ctc** -- the same plus 160
  segments in 31 other languages, 13.1 h in all. Mono 16 kHz.
* reference: omnilingual-asr 0.2.0 + fairseq2 0.6 (torch 2.8): `ASRInferencePipeline.transcribe` with greedy search
  (`nbest=1`), conditioned on the language, inputs longest first (`tools/reference.py`); ours: torch 2.14, bf16.
* one H200; RTFx = seconds of audio per second of wall time, first pass over the data.

omniASR-LLM (the llm set):

| implementation | batch | RTFx | peak allocated | model load |
|---|---:|---:|---:|---:|
| omnilingual-asr 0.2.0 | 4 | 16.2 | 18.6 GiB | ~450 s (fp32 `.pt` from network storage) |
| omnilingual-asr 0.2.0 | 8 | 18.5 | 22.6 GiB | |
| **omniasr-trt** | 8 | **152** | 23.7 GiB | 10-24 s (bf16 `.safetensors`) |

8.2x at batch 8 (151-155 over repeated runs; the second pass over the data is as fast as the first).

omniASR-CTC emissions (20 s windows with 5 s context, 320 s of audio per forward, 4 inputs per call):

| implementation | both sets (13.1 h) | inputs up to 90 s (1.4 h) | model load |
|---|---:|---:|---:|
| fairseq2 `Wav2Vec2AsrModel` | 345 | 251-260 | ~400 s |
| **omniasr-trt** | **376-387** | **321** | 7-15 s |

The CTC model is the 7B encoder only, and its GEMMs (65% of the time, ~640 TFLOPS) are the same cuBLAS calls as
fairseq2's, so the gains are small: attention without padding masks and the load time.

## How

* omnilingual-asr's decoding loop feeds the longer contexts of a batch one token per decoder call and re-indexes the
  whole KV cache at every step (also with one hypothesis). Here every context is prefilled at once and each row
  keeps its own position in a static KV cache.
* Continuous batching over the inputs of a call: a slot whose segment is finished takes its input's next segment
  (whose context is the text just decoded) or the next input, instead of waiting for the slowest row of a lock-step
  round. Finished rows read no keys.
* One decode step (12 layers, LM head, argmax) is one CUDA graph per row count and key-length bucket; the host looks
  at the tokens every 16 steps (EOS, the length cap, the repetition stop). The graphs are captured by hand into one
  memory pool: `torch.cuda.graph` empties the allocator cache on every capture, which under
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ended in illegal memory accesses.
* Triton kernels: RoPE + KV-cache write (decode and prefill) and split-K flash-decode attention (head dim 512, too
  large for flash-attention), each tuned on an H200 (`scripts/tune_attn.py`). A decode step of 8 rows takes 1.87 ms
  (1.31 ms for one), within 1.4x of reading the 2.5 GB of decoder weights and the KV cache once.
* Encoder: q|k|v as one GEMM; padded batches attend per run of equal-length rows with flash kernels (no mask);
  PyTorch's default cuDNN attention would build a new plan for every new sequence length (204 instead of 319 RTFx on
  a first pass) and is only used for unpadded shapes that recur; the feature extractor runs on 120 s at a time,
  which bounds the encoder's memory.

Tried without gain: running the encoder on a second stream to overlap the decoder (its GEMM tiles fill every SM; the
time and the transcripts did not change). Tried and dropped: FP8 (e4m3, per-channel weights, per-row activations)
for the encoder's linears changed the CTC emissions four times as much as bf16 batch changes do (mean |delta
log-prob| 0.29 vs 0.07, 3.6% of CTC word alignments moved by more than a frame).

## Agreement

The search rules are omnilingual-asr's: greedy on the log-softmax, EOS, the position cap (8192 - 4), the zlib
repetition stop (every 250 positions, ratio > 4 over 100 tokens). Per input the result does not depend on the other
inputs of a batch, so batching and prefilling may change only the rounding.

omniASR-LLM in fp32, the 92 llm inputs up to 90 s (67 of them in several segments), against omnilingual-asr in fp32
at batch 1:

| omniasr-trt, fp32 | identical texts | character edit rate |
|---|---:|---:|
| reference order (prefill all but `<bos>`, then one step at a time), batch 1 | 91/92 | 0.008% |
| fast path (prefill at once, CUDA graphs, Triton kernels), batch 1 | 91/92 | 0.008% |
| fast path, batch 8 | 81/92 | 0.63% |

The one input that differs is the same in every batch-1 run (a near tie in a 15 s segment). At batch 8 the other
differences come from batching itself (padding, other GEMM shapes): fp32 rounding at near ties of noisy inputs.

bf16 changes many transcripts of such data (low-resource languages, noisy audio, long inputs where one changed word
changes the next segment's context), for omnilingual-asr as much as for omniasr-trt. Against the fp32 reference on
the same 92 inputs:

| bf16 run | character edit rate |
|---|---:|
| omnilingual-asr, batch 8 / 4 / 2 / 1 | 8.96% / 9.12% / 9.14% / 9.27% |
| omniasr-trt, batch 8 | 9.04% |
| omniasr-trt, other batch sizes, kernels and builds (8 runs) | 5.8% to 10.4% (mean 8.6%) |

omniASR-CTC, 206 inputs up to 90 s: logits, and the word times of a CTC forced alignment of a transcript:

| comparison | frames with another argmax | mean abs. log-prob diff. | word times identical / within 20 ms |
|---|---:|---:|---:|
| fp32: fairseq2 vs omniasr-trt | 41 of 256,590 | 0.003 | 99.95% / 100% |
| bf16: fairseq2, 4 inputs per call vs 1 | 677 | 0.071 | 98.2% / 98.7% |
| bf16: fairseq2 vs omniasr-trt | 754 | 0.066 | 99.0% / 99.6% |

(log-prob differences over the entries either side puts above -10 nats.) The SentencePiece tokenizer encodes and
decodes 761/761 test texts as fairseq2's char tokenizer does.

To compare on your own data: `scripts/bench.py` (ours), `tools/reference.py` (omnilingual-asr, installed separately)
and `scripts/compare.py` on a JSONL manifest of `{"id", "audio", "lang"}`.

## Layout

* `src/omniasr_trt/modeling.py` -- encoder, CTC head, LLaMA decoder; checkpoint loading and conversion
* `src/omniasr_trt/llm.py` -- streaming syntax and the reference-order greedy loop; `decode.py` -- the fast decoder
* `src/omniasr_trt/kernels.py` -- Triton kernels; `ctc.py` -- windowed CTC emissions; `compat.py` -- drop-ins
* `scripts/` -- benchmarks and kernel tuning; `tools/reference.py` -- omnilingual-asr runner (never imported)
* `tests/` -- `pytest` (the decoder against the reference-order loop on small random models, kernels on CUDA)

## License

Apache-2.0. Not affiliated with Meta. `src/omniasr_trt/lang_ids.json` is omnilingual-asr's language table
(Apache-2.0, Copyright Meta Platforms, Inc. and affiliates); the model architecture and decoding rules follow
omnilingual-asr and fairseq2 (MIT). The model weights are Meta's and are not included.
