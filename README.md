# omniasr-trt

Fast inference for Meta's [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) 7B models:
`omniASR_LLM_Unlimited_7B_v2` (transcripts of inputs of any length, greedy or beam search) and `omniASR_CTC_7B_v2`
(CTC emissions, e.g. for forced alignment, and greedy transcripts; tested also with the 300M and 1B CTC models). Plain
PyTorch with a few Triton kernels and CUDA graphs; no fairseq2 or omnilingual-asr dependency. The official
checkpoints load with `strict=True` (only the positional convolution's weight norm is folded). The greedy search is
decoded speculatively (several tokens per decoder step, the same tokens).

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
wget https://dl.fbaipublicfiles.com/mms/omniASR-CTC-300M-v2.pt             # optional drafter: 1.3 GB
```

The official `.pt` files load directly too (cast at load). The tokenizer is looked up beside the checkpoint unless
given.

## Use

```python
from omniasr_trt import OmniASRCTC, OmniASRLLM

llm = OmniASRLLM.from_pretrained("llm-bf16.safetensors", "omniASR_tokenizer_written_v2.model")
llm.transcribe([wave_a, wave_b], langs=["ben_Beng", None])   # 16 kHz mono float32, any length; None: unconditioned
llm.transcribe([wave_a], langs=["ben_Beng"], beam_size=5)     # beam search (also: from_pretrained(beam_size=5))

# Optional: a CTC model's transcripts draft the greedy search (same texts, fewer decoder steps).
llm = OmniASRLLM.from_pretrained("llm-bf16.safetensors", "omniASR_tokenizer_written_v2.model",
                                 drafter="omniASR-CTC-300M-v2.pt")

ctc = OmniASRCTC.from_pretrained("ctc-bf16.safetensors", "omniASR_tokenizer_written_v2.model")
frames, windows = ctc.emissions([wave])   # logits [T, 10288] per input (bf16, GPU), 20 ms frames
ctc.transcribe([wave])                    # greedy text
```

```bash
omniasr-trt transcribe --model llm-bf16.safetensors --lang ben_Beng speech.wav
omniasr-trt transcribe --model llm-bf16.safetensors --lang ben_Beng --beam-size 5 speech.wav
omniasr-trt transcribe --model llm-bf16.safetensors --drafter omniASR-CTC-300M-v2.pt --lang ben_Beng speech.wav
omniasr-trt transcribe --ctc --model ctc-bf16.safetensors speech.wav
```

* LLM: inputs are cut into 15 s segments and segment i is decoded with segment i-1's audio and text as context, as
  omnilingual-asr's streaming pipeline does; language ids are omnilingual-asr's (`eng_Latn`, `ben_Beng`, ...).
  `max_batch` (default 8) is the number of inputs decoded at once; a call may hold more, a slot whose input is
  finished takes the next one. Greedy search proposes `draft_tokens` (4) tokens per decoder step
  ([Speculative decoding](#speculative-greedy-decoding); 0: one token per step), from each row's own tokens and,
  with `drafter` or `transcribe(..., drafts=...)`, a CTC transcript. `beam_size` > 1 runs omnilingual-asr's beam
  search (`length_norm=True`: rank hypotheses by their mean log-probability, its option), as it behaves for a
  single input; see [Beam search](#beam-search) for what batching does to it in omnilingual-asr.
* CTC: inputs up to `max_single_seconds` (30) in one pass, longer ones in `window_seconds` (20) windows with
  `context_seconds` (5) on both sides whose centre frames are stitched; each piece is normalised on its own.
* `omniasr_trt.compat` has drop-ins for code written against omnilingual-asr / fairseq2: `Pipeline`
  (`ASRInferencePipeline.transcribe(inp, lang=..., batch_size=...)`, with its `beam_search_config`; all inputs of a
  call, up to an hour of audio at a time, share the decoder slots rather than going `batch_size` at a time), `CTC`
  (`forward(pieces)` -> logits per piece) and a fairseq2-style tokenizer (`create_encoder`, `create_decoder`,
  `vocab_info`).

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
| omniasr-trt, one token per decoder step | 8 | 152 | 22.8 GiB | 10-24 s (bf16 `.safetensors`) |
| **omniasr-trt** (speculative, own tokens: the default) | 8 | **166** | 22.8 GiB | |
| **omniasr-trt** + omniASR-CTC-300M-v2 drafter | 8 | **173** | 23.4 GiB | +6 s |
| **omniasr-trt**, beam 5 | 8 | **129** | 25.8 GiB | |

Inputs longest first, in calls of 8 (as omnilingual-asr's `batch_size`). 9.0x omnilingual-asr at batch 8 (9.4x with
the drafter); the second pass over the data is as fast as the first. A call of all 160 inputs keeps the 8 decoder
slots full: 175 / 197 / 209 RTFx (see [Speculative decoding](#speculative-greedy-decoding)). Beam search with 5
hypotheses costs 16% more than one-token greedy steps here; omnilingual-asr's own beam search is compared below.

omniASR-CTC emissions (20 s windows with 5 s context, 320 s of audio per forward, 4 inputs per call):

| implementation | both sets (13.1 h) | inputs up to 90 s (1.4 h) | model load |
|---|---:|---:|---:|
| fairseq2 `Wav2Vec2AsrModel` | 345 | 251-260 | ~400 s |
| **omniasr-trt** | **376-387** | **321** | 7-15 s |

The CTC model is the 7B encoder only, and its GEMMs (65% of the time, ~640 TFLOPS) are the same cuBLAS calls as
fairseq2's, so the gains are small: attention without padding masks and the load time.

## Speculative greedy decoding

Every greedy decoder step feeds each row's newest token and 4 proposed tokens, and keeps the proposed tokens the
greedy search would have produced itself (each equal to the argmax after its predecessor) plus the argmax after the
last one kept: 1 to 5 tokens per step. The tokens are the greedy search's whatever was proposed; a wrong proposal
only costs a step. EOS, the position cap and the repetition stop apply at the token where they apply one token at
a time. The proposals come from

* the row's own tokens (prompt lookup): the latest earlier place where its newest tokens occurred, continued as it
  went on there. This is what the model does in a repetition loop until the repetition stop ends it, and costs
  nothing; it is the default;
* a CTC transcript of the same audio, optionally: `drafter=` an omniASR CTC checkpoint (the 300M is enough), or
  `transcribe(..., drafts=)` per input the argmax of each 20 ms CTC frame that you already have (e.g. from the 7B CTC
  model run for alignment; `OmniASRCTC.frame_ids`). Each 15 s segment gets its frames' text (and 0.5 / 1 s around
  it); the proposal continues the draft after the position whose preceding tokens match the newest generated ones,
  searched around where the previous step left off (the farther from there, the longer the match must be, so a
  misrecognised character does not send it to a frequent one far away). The own tokens win where they match longer.

A step of 5 tokens per row costs 1.16x a one-token step (8 rows, ~1,450 keys: 2.03 vs 1.75 ms): the weights are
read once for all of a row's tokens, one Triton kernel attends a row's 5 causal queries with a single read of its
keys, and the proposal search and the keeping of tokens are one Triton kernel each.

The llm set, 8 decoder slots, RTFx (bf16, two passes each):

| decoding | calls of 8 inputs | all 160 inputs in one call | decoder steps (calls of 8) |
|---|---:|---:|---:|
| one token per step (`draft_tokens=0`) | 152-154 | 175-176 | 63,440 |
| own tokens (the default) | 164-169 | 197-198 | 53,616 |
| + omniASR-CTC-300M-v2 drafter (its ~6% of the time included) | 173-177 | 209-214 | 45,376 |
| + omniASR-CTC-7B-v2 frame argmaxes, computed beforehand | 192 | | 42,620 |

Calls of 8 inputs leave slots idle once their input is finished (the segments of an input are decoded one after
another, each the context of the next); larger calls keep them full, which helps one token per step too.
`compat.Pipeline.transcribe` therefore decodes all inputs of a call together (up to an hour of audio at a time). 8
proposed tokens instead of 4 are no faster (4% fewer steps at a higher cost per step); the 1B CTC model drafted
only slightly better than the 300M in a simulation and takes 2.2x as long.

In fp32 the texts are exactly those of one token per step: 92/92 llm inputs up to 90 s, at batch 1 and at batch 8,
with the 300M drafts and with own tokens only. In bf16 a 5-token step rounds differently from a one-token step, as
another batch size does: against the fp32 reference 9.2-9.3% character edits (one token per step: 9.0%; bf16 runs in
general 5.8-10.4%, see [Agreement](#agreement)), caption CER 6.11-6.30 for all of them. Runs are reproducible:
12 of 13 processes of the default on this set gave identical texts (one differed on 70 of 160 inputs,
within that spread, and was not reproduced); as with one token per step, the first pass of a process can differ
from the later ones on an input (1 of 160 here).

## Beam search

`beam_size=5` is the setting of the omnilingual-asr paper (`Wav2Vec2LlamaBeamSearchConfig(nbest=5)`); the search
rules are omnilingual-asr's: every step each hypothesis offers `score + log p` (with `length_norm`, the running mean),
a finished one only EOS at its frozen score, the best 5 offers survive, the repetition stop freezes a hypothesis at
the step it fires, and the text is the best hypothesis's when all have finished. Here this is what happens for every
input, whatever the batch.

In omnilingual-asr 0.2.0 it is what happens at batch 1 only. A batch is prefilled up to its shortest context and the
longer contexts are fed one token per step; during that phase every hypothesis of those inputs has its score reset
to 0, so their 5 hypotheses start identical and stay identical: each segment of every input whose context is longer
than the batch's shortest is decoded greedily. Measured in fp32 on the 25 llm inputs up to 15 s (one segment each;
4 calls of up to 8): in every call exactly the shortest input got its batch-1 beam-search text, 15 of the other 21
got their greedy text and 6 texts on which beam search and greedy agree. Longer inputs mix beam-searched and greedy
segments.

The 92 llm inputs up to 90 s (39 min):

| implementation | batch | RTFx |
|---|---:|---:|
| omnilingual-asr 0.2.0, beam 5 | 1 | 7.85 |
| omnilingual-asr 0.2.0, beam 5 (mostly greedy, as above) | 8 | 13.3 |
| **omniasr-trt, beam 5** | 8 | **121** (132 on a second pass) |

| comparison (same 92 inputs) | identical texts | character edit rate |
|---|---:|---:|
| fp32, batch 1: omniasr-trt vs omnilingual-asr (fast path and reference order) | 92/92 | 0% |
| bf16: omniasr-trt batch 8 vs omnilingual-asr batch 1 | 65/92 | 3.6% |
| bf16: omnilingual-asr batch 8 vs omnilingual-asr batch 1 | 27/92 | 10.5% |
| bf16, omnilingual-asr batch 1: beam 5 vs greedy | 17/92 | 19.3% |

How: the context (~1,750 tokens) is prefilled once per input and its K/V shared by the 5 hypotheses; the K/V of
each generated token is written once and hypotheses follow parent pointers (a reorder gathers int32 indices, not
K/V); one Triton kernel reads an input's context once for all 5 queries, another each hypothesis's generated keys
through its pointers. One step, including the top-5 over 5 x 10288 candidates per input, is one CUDA graph; the host
steps in exactly at the step a repetition check is due.

## How

* omnilingual-asr's decoding loop feeds the longer contexts of a batch one token per decoder call and re-indexes the
  whole KV cache at every step (also with one hypothesis). Here every context is prefilled at once and each row
  keeps its own position in a static KV cache.
* Continuous batching over the inputs of a call: a slot whose segment is finished takes its input's next segment
  (whose context is the text just decoded) or the next input, instead of waiting for the slowest row of a lock-step
  round. Finished rows read no keys.
* One decode step (12 layers, LM head, argmax; speculative: 5 tokens per row, the proposal search and the kept
  tokens) is one CUDA graph per row count and key-length bucket; the host looks at the tokens every 16 steps (8
  speculative ones: EOS, the length cap, the repetition stop). The graphs are captured by hand into one
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
* `src/omniasr_trt/llm.py` -- streaming syntax and the reference-order greedy and beam loops; `decode.py` -- the
  fast greedy decoder; `spec.py` -- its speculative step; `beam.py` -- the fast beam search
* `src/omniasr_trt/kernels.py` -- Triton kernels; `ctc.py` -- windowed CTC emissions; `compat.py` -- drop-ins
* `scripts/` -- benchmarks and kernel tuning; `tools/reference.py` -- omnilingual-asr runner (never imported)
* `tests/` -- `pytest` (the decoder against the reference-order loop on small random models, kernels on CUDA)

## License

Apache-2.0. Not affiliated with Meta. `src/omniasr_trt/lang_ids.json` is omnilingual-asr's language table
(Apache-2.0, Copyright Meta Platforms, Inc. and affiliates); the model architecture and decoding rules follow
omnilingual-asr and fairseq2 (MIT). The model weights are Meta's and are not included.
