# Audio8 TTS LoRA Fine-Tuning

Train small PEFT adapters for Audio8 TTS Preview 0.6B on a single CUDA GPU or
Apple Silicon Mac, then reload the adapter for speech generation.

This repository is an Instavar-maintained derivative of
[`Audio8-AI/Audio8_TTS`](https://github.com/Audio8-AI/Audio8_TTS) at commit
`3346560df718d33096ac2fef7e5c7984ee5248e6`. It preserves the upstream Git
history and Apache 2.0 notices. Instavar added LoRA training, adapter-aware
inference, a portable single-device launcher and bounded CUDA and macOS
reproduction guidance.

The repository does not include model weights, voice recordings, prepared
datasets or trained adapters. Download the Apache 2.0 Audio8 model separately
and train only on recordings that you have the right to use.

## What this adds

- rank, alpha, dropout and target-module controls through Hugging Face PEFT;
- adapter-only checkpoints plus an optional merged export;
- adapter loading through `audio8_tts_infer.py --adapter`;
- one direct Python launcher that avoids DeepSpeed and distributed assumptions;
- CUDA and Apple Silicon smoke configurations;
- explicit checkpoint generation checks because lower validation loss did not
  reliably predict normal end-of-speech behavior in our experiment.

## Pinned starting point

| Component | Revision |
| --- | --- |
| Upstream code | `3346560df718d33096ac2fef7e5c7984ee5248e6` |
| Audio8 model | `1b17c91db5f4dccb6914aa4aa5cb0e56661a6c17` |
| Licence | Apache License 2.0 |

## Quick LoRA run

Install the appropriate dependencies:

```bash
# CUDA LoRA without DeepSpeed
pip install -r requirements-lora-cuda.txt

# Apple Silicon
pip install -r requirements-macos.txt
```

Prepare the codec targets using the upstream data format, then run:

```bash
TRAIN_JSONL=prepared_data/train.jsonl \
EVAL_JSONL=prepared_data/validation.jsonl \
MAX_STEPS=100 \
BF16=true \
bash audio8_tts_lora.sh
```

For the first 16 GB Mac smoke test, use the smaller bounded configuration:

```bash
TRAIN_JSONL=prepared_data/train.jsonl \
MAX_STEPS=1 \
MAX_LENGTH=256 \
BATCH_SIZE=1 \
BF16=false \
FP16=false \
bash audio8_tts_lora.sh
```

The Transformers trainer selects MPS automatically when it is available. MPS
does not support distributed training, which is why this launcher calls the
trainer directly.

Reload the saved adapter without merging it:

```bash
python audio8_tts_infer.py \
  --model model/audio8_tts_0_6B_preview \
  --adapter outputs/audio8_tts_lora \
  --text "A held-out sentence checks whether the adapter still speaks clearly." \
  --device auto \
  --greedy \
  --output outputs/adapter-check.wav
```

### Executable Instavar Voice lifecycle

[`instavar-voice-backend.json`](instavar-voice-backend.json) turns the Audio8
LoRA path into a five-stage executable recipe. It requires an explicit CUDA or
MPS device, hashes the complete local base-model tree, audits raw grouped
splits, writes training outputs under the unique lifecycle work directory,
selects one exact Trainer checkpoint, reloads the archived adapter in a fresh
process, evaluates the frozen prompt plan, and packages immutable evidence.

Validate the recipe with evaluator revision
`1a413952ae3f43aeda88fde5109e724771c12b0c`. Use an empty work directory
outside the checkout and set `SELECTED_ADAPTER_NAME` to a real checkpoint child
such as `checkpoint-20`. A CUDA pass proves only the selected CUDA environment;
an MPS pass proves only the selected MPS environment. Neither is a perceptual
quality or cross-runtime equivalence claim.

## Promotion gate

### Frozen multi-prompt adapter evaluation

Build an Instavar Voice generation plan, then run every Audio8 row through one
loaded base model and adapter:

```bash
python scripts/run_evaluation_suite.py \
  --generation-plan evaluation/generation-plan.json \
  --candidate-id audio8-adapter \
  --model /path/to/audio8_tts_0_6B_preview \
  --adapter /path/to/adapter \
  --output-dir evaluation/audio8-adapter \
  --runtime-id pytorch_cuda \
  --device cuda
```

Batch rows carry explicit seeds and are grouped by reference-conditioning mode
and seed before generation. `NO_EOS` remains an invalid observation even when
a WAV was written. The runner therefore cannot improve its apparent completion
rate by dropping capped generations.

For an exact CUDA-versus-MPS experiment, also pass `--artifact-set-id` and
`--artifact-set-sha256` together. The runner infers `pytorch_cuda` or
`pytorch_mps` from the device when `--runtime-id` is omitted and rejects partial
or malformed artifact bindings. ONNX or SGLang exports remain `derived`, not
exact.

A completed training command is not enough to select an adapter. At minimum,
compare the base model and every candidate checkpoint on held-out prompts for:

1. correct and intelligible words;
2. normal speech duration;
3. normal end-of-speech behavior;
4. voice similarity using a stated proxy;
5. blind human listening.

In Instavar's first 128-clip pilot, step 100 ended normally on all 16 held-out
prompts. Several later checkpoints reached a 500-frame generation cap even as
validation loss continued to improve. That result applies only to the pinned
experiment, but it is enough to make generation checks mandatory for this
workflow.

The [2026-08-12 CUDA runtime report](reports/runtime-smoke-2026-08-12.md)
records a frozen-prompt base-model smoke with output and model hashes, WER,
cold-start time, and peak GPU memory. No Audio8 adapter was available on that
host, so the report deliberately does not promote CUDA adapter support beyond
the existing bounded training evidence.

## Rights and attribution

Code and model weights are separate distribution surfaces, even though both
currently declare Apache 2.0. Dataset and voice rights are separate again.
Read [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before redistributing a
derivative or adapter.

## Upstream documentation

The remaining documentation below is retained from the Audio8 project and
describes the base model, data format, inference path and full SFT workflow.

---

<div align="center">

<img src="assets/20260729-124515.jpeg" alt="Audio8" width="100%">

# audio8_tts Preview

**A 0.6B-parameter multilingual text-to-speech model with zero-shot voice cloning.**

[![GitHub](https://img.shields.io/badge/GitHub-Audio8__TTS-black?style=for-the-badge&logo=github)](https://github.com/Audio8-AI/Audio8_TTS)
[![Demo](https://img.shields.io/badge/Demo-Audio%20Samples-brightgreen?style=for-the-badge&logo=githubpages)](https://audio8-ai.github.io/Audio8_TTS/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Audio8--TTS--Preview--0.6b-yellow?style=for-the-badge)](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

中文文档: [README_zh.md](README_zh.md)

</div>

This repository provides the audio8_tts Preview checkpoint, Hugging Face remote
code, inference tools, and an independent SFT pipeline for multilingual speech
generation and zero-shot voice cloning.

> **Preview status:** language coverage is intentionally limited in this
> release. Use the model primarily with the 11 recommended languages below.
> Multilingual coverage and Chinese dialect support will be expanded in later
> releases.

## Supported Languages

The Preview checkpoint performs best in the following languages:

| Language | Name |
|---|---|
| Cantonese | 粤语 |
| Chinese | 中文 |
| Dutch | 荷兰语 |
| English | 英语 |
| French | 法语 |
| German | 德语 |
| Italian | 意大利语 |
| Japanese | 日语 |
| Korean | 韩语 |
| Polish | 波兰语 |
| Spanish | 西班牙语 |

## Architecture

audio8_tts uses a DualAR architecture inspired by
[Fish Audio S2 Pro](https://github.com/fishaudio/fish-speech).

| Component | Configuration |
|---|---|
| Main model | 601,159,424 parameters, excluding the codec |
| Slow AR | 24 layers, width 896, 14 attention heads, 2 KV heads |
| Fast AR | 4 layers, width 896, 14 attention heads, 2 KV heads |
| Acoustic tokens | 10 codebooks, 4,096 entries per codebook |
| Codec | 44.1 kHz, 2,048 samples per model frame (~21.5 frames/s) |
| Context | Up to 2,048 packed text/audio positions |

The slow AR transformer predicts one semantic token for each audio frame. The
fast AR transformer then predicts the frame's codec codebooks, conditioned on
the slow hidden state and preceding codebooks. Static KV caches are used by
both branches during generation. The checkpoint also bundles its neural codec,
so reference encoding and waveform decoding require no separate model.

## Installation

Python 3.10 or newer and a CUDA-capable GPU are recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the checkpoint from
[Hugging Face](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b) and
place it in the repository's `model/` directory. The expected local checkpoint
path is `model/audio8_tts_0_6B_preview/`. All commands also accept a Hugging
Face model ID through `--model`.

## Inference

### Zero-shot voice cloning

The reference transcript should match the spoken content in the reference
audio.

```bash
python audio8_tts_infer.py \
  --text "Welcome to audio8_tts." \
  --reference-audio examples/reference.wav \
  --reference-text "Transcript of the reference recording." \
  --output outputs/clone.wav
```

### Generation without a reference

```bash
python audio8_tts_infer.py \
  --text "This utterance does not use a reference voice." \
  --output outputs/no_reference.wav
```

### Batch inference

Each line in the input manifest is an independent JSON object. Relative audio
paths are resolved from the manifest directory.

```json
{"id":"sample_001","text":"Target text","reference_audio":"audio/ref.wav","reference_text":"Reference transcript"}
{"id":"sample_002","text":"Text without a reference voice"}
```

```bash
python audio8_tts_infer.py \
  --input-jsonl data/prompts.jsonl \
  --output-dir outputs/batch \
  --batch-size 2
```

The batch command writes `manifest.jsonl` and `failures.jsonl`. Existing WAV
files are skipped unless `--overwrite` is passed. See
`python audio8_tts_infer.py --help` for sampling and code-saving options.

## Supervised Fine-tuning

Install the training dependencies first:

```bash
pip install -r requirements-train.txt
```

### 1. Create a raw manifest

The target `audio` field is required. `reference_audio` and `reference_text`
are optional, but must be provided together.

```json
{"id":"utt_001","text":"Target transcript","audio":"audio/target.wav","reference_audio":"audio/reference.wav","reference_text":"Reference transcript"}
{"id":"utt_002","text":"Another transcript","audio":"audio/another.wav"}
```

### 2. Precompute codec indices

```bash
python audio8_tts_prepare.py \
  --input-jsonl data/train.jsonl \
  --output-jsonl prepared_data/train.jsonl \
  --batch-size 4
```

The prepared manifest points to validated `[10, T]` NumPy arrays using paths
relative to the prepared manifest. Existing valid arrays are reused unless
`--overwrite` is passed.

### 3. Train

Single GPU:

```bash
TRAIN_JSONL=prepared_data/train.jsonl \
NPROC_PER_NODE=1 \
bash audio8_tts_sft.sh
```

Eight GPUs on one node:

```bash
TRAIN_JSONL=prepared_data/train.jsonl \
NPROC_PER_NODE=8 \
BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=8 \
bash audio8_tts_sft.sh
```

For multi-node training, set `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and
`MASTER_PORT` on each node. Common hyperparameters and output paths can be
overridden through the environment variables in `audio8_tts_sft.sh`; additional
Transformers arguments may be appended to the command.

SFT optimizes both the slow semantic/EOS objective and the fast codebook
teacher-forcing objective. Set `FREEZE_SLOW_AR=true` or `FREEZE_FAST_AR=true`
when adapting only one branch. The exported directory remains loadable with
standard `AutoModel` and `AutoProcessor` APIs using `trust_remote_code=True`.

## Evaluation

Audio8 TTS Preview is the smallest model in this comparison at just **0.6B
parameters**. Despite using only a fraction of the parameters of the other
systems, it delivers results in the first tier of industry-leading SOTA TTS
models on the benchmarks below. In particular, it achieves the best English
WER and competitive Chinese CER on Seed-TTS, while remaining competitive
across the CV3 multilingual evaluation.

Lower WER/CER is better; higher SIM is better. Seed-TTS similarity values are
shown as percentages.

### Seed-TTS

| Model | Parameters | EN WER / SIM | ZH CER / SIM | Hard ZH CER / SIM |
|---|---:|---:|---:|---:|
| **Audio8 TTS Preview** | **0.6B** | **1.506** / 63.2 | 0.950 / 73.1 | 11.510 / 68.7 |
| Fish S2 Pro | 4.6B | 1.607 / 64.6 | 1.038 / 73.8 | 10.149 / 70.1 |
| Higgs Audio v2 | 4.7B | 1.524 / 66.4 | **0.806** / 72.1 | 10.622 / 69.3 |
| CosyVoice3-1.5B | 1.5B | 2.22 / 72.0 | 1.12 / 78.1 | **5.83** / **75.8** |
| MOSS-TTS | 8.5B | 1.85 / 73.4 | 1.20 / 78.8 | - |
| VoxCPM2 | 2.3B | 1.84 / **75.3** | 0.97 / **79.5** | 8.13 / 75.3 |

### CV3 multilingual error rate

| Model | Parameters | zh | en | hard-zh | hard-en | ja | ko | de | es | fr | it | ru |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Audio8 TTS Preview** | **0.6B** | **3.205** | **3.128** | 10.535 | 5.997 | 7.205 | 4.223 | 3.447 | 3.641 | 8.790 | 4.790 | - |
| Fish S2 Pro | 4.6B | 3.600 | 3.493 | 10.588 | 7.349 | 5.139 | **4.111** | 3.605 | 2.972 | **8.600** | 4.229 | **4.702** |
| Higgs Audio v2 | 4.7B | 3.378 | 3.404 | 10.424 | **5.754** | **4.742** | 4.260 | **3.300** | **2.929** | 9.425 | **3.555** | 5.423 |
| CosyVoice3-1.5B | 1.5B | 3.91 | 4.99 | 9.77 | 10.55 | 7.57 | 5.69 | 6.43 | 4.47 | 11.8 | 10.5 | 6.64 |
| VoxCPM2 | 2.3B | 3.65 | 5.00 | **8.55** | 8.48 | 5.96 | 5.69 | 4.77 | 3.80 | 9.85 | 4.25 | 5.21 |

Parameter counts are calculated directly from the released weight tensors.
MOSS-TTS contains 8,489,841,664 parameters. VoxCPM2's main model contains
2,290,004,544 parameters; the separate AudioVAE is not included in the
parameter comparison.

Fish S2 Pro was reevaluated because its official evaluation uses its own
normalizer. Higgs Audio v2 was evaluated locally because concrete values were
unavailable. All other baseline values were collected from their official
reports through the [VoxCPM repository](https://github.com/OpenBMB/VoxCPM).

Different normalizers and evaluators make cross-project values reference
comparisons rather than a strictly matched ranking. Evaluation coverage does
not expand the Preview's supported-language claim beyond the 11 languages
listed above.

## Limitations and Responsible Use

- This is a Preview checkpoint with limited multilingual and dialect coverage.
- Very long, noisy, or inaccurate reference clips can reduce stability and
  speaker similarity.
- Generated speech can be misused for impersonation or misinformation. Obtain
  consent before cloning a voice and clearly disclose synthetic audio where
  appropriate.
- Test the model for accuracy, safety, and legal compliance before deployment.

## License and Acknowledgements

Code and model weights in this repository are released under the
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution details.

We thank the Fish Audio team for publishing the DualAR architecture used in
Fish S2 Pro.

## Instavar Voice conformance

[`instavar-voice-capabilities.json`](instavar-voice-capabilities.json) distinguishes the validated PyTorch adapter paths from upstream ONNX and SGLang surfaces whose adapter-aware exports remain unverified. It also records the frozen objective and blinded-listening gates required before promotion. CI validates the manifest against the pinned public [Instavar Voice evaluation contract](https://github.com/instavar/instavar-voice-evaluation).

The lifecycle fixes evaluation batch size at one so timing belongs to one
sample, preserves invalid generations as explicit rows, and uses evaluator
revision `1a413952ae3f43aeda88fde5109e724771c12b0c` to bind timing, duration,
and peak-memory fields to the frozen plan and live output audio. CUDA peak
allocation is measured by PyTorch. MPS timing explicitly synchronizes the
device, but non-CUDA paths omit peak memory because this runner has no equivalent
peak-allocation probe. A version 1.1 plan requiring peak memory therefore fails
closed instead of accepting a synthetic zero. Use the packaged
`objective-observations.json`, not the raw generation file, for a version 1.1
runtime comparison.

The pinned evaluator provides schema 1.3 frozen speaker-reference assignments,
the optional schema 1.4 SpeechBrain ECAPA execution path, and the optional
schema 1.5 local faster-whisper ASR path. Version 0.20 also distinguishes
generation-plan-bound ASR reference text from observation-declared strings.
Version 0.21 adds plan-bound category strata so pronunciation, local-context,
and long-form proxy regressions remain visible instead of disappearing into one
candidate mean.
Version 0.22 carries frozen lexical anchors and accepted ASR forms into the
generation plan, reports hit, miss, coverage, and matched deltas, and rejects
candidate-specific alias drift. Phrase hits remain recognition evidence, not
pronunciation or accent judgments.
Version 0.23 preregisters criterion-specific blind-listening assignments so
lexical pronunciation, cadence, fatigue, and emotion ratings only cover prompts
that can support those claims while preserving candidate-symmetric coverage.
Version 0.24 binds exact requested text, optional instructions, and lexical
target surfaces into each blind stimulus while excluding accepted ASR aliases
and candidate identity. Reviewers no longer need an uncontrolled prompt file.
Version 0.25 binds each listening criterion to a reviewer question, low and
high scale anchors, and an explicit score direction. Harm criteria remain raw
and separate instead of being silently inverted or folded into a composite.
This companion bundles neither model
weights nor optional extractor dependencies and runs neither learned metric
automatically. Run them explicitly after generation with trusted, content-addressed
models, frozen decoding, and a preregistered reference plan where applicable.
Runtime-bound observations, same-recording smoke scores, or human-recording ASR
alone are not TTS-quality evidence.
