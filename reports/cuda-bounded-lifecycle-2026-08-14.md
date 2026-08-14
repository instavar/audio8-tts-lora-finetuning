# Audio8 bounded CUDA LoRA lifecycle

Date: 2026-08-14, Asia/Singapore

## Scope

This qualification exercised the current public Audio8 LoRA path on an NVIDIA
RTX 3090 Ti. Lifecycle commit
`5f89ec01b94558de217d39bf059f829a0be48460` completed preflight, one BF16
training update, inference-only adapter selection, fresh-process reload, two
frozen held-out rows, every objective metric required by the generation plan,
and content-addressed package persistence. Restore-verifier commit
`622c781bebf9d6b21ea676afba2553d420fd74dd` then restored that exact package
twice in new directories.

This proves a bounded PyTorch CUDA engineering path. It does not establish
convergence, adaptation benefit, perceptual quality, Singapore English accent
fidelity, long-form stability, resume equivalence, MPS equivalence, ONNX or
SGLang adapter equivalence, distribution rights, or production readiness.

## Bound inputs

- upstream Audio8 code: `3346560df718d33096ac2fef7e5c7984ee5248e6`
- Audio8 model revision: `1b17c91db5f4dccb6914aa4aa5cb0e56661a6c17`
- local base-model tree SHA-256:
  `5a2b2e402ebb42aa645d22137f39be75cdf43b80a1853be70eb93d710d1055ce`
- evaluator revision: `2812e200233804fde685c35ea1da1cbf9fe8ef4b`
- faster-whisper revision: `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`
- SpeechBrain ECAPA revision: `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`
- raw train SHA-256:
  `790e0ff98267ad84d42677a05c5393110d1dde678d6d05b27464aebe7e14e444`
- raw validation SHA-256:
  `d1dba52a8caf4c1889c7c25b19028327aca753ef4c9b670aadc7ce2f960fa7be`
- raw test SHA-256:
  `ff7a9792b33b85d0cd6e44df85b6c8c90761e71872bb7379eae1e25ef56cb28c`
- prepared train manifest SHA-256:
  `48984cdd067999f3be4637252ff88e5fc592aab326d2689c8ba2edd25ea04b9a`
- prepared validation manifest SHA-256:
  `1f464c2079d975940acc52be190c15b5eb94e4e39812d3930a01d4f12c63c90e`

The qualification uses one distinct source group per raw split and one prepared
training row plus one prepared validation row. The audit reported zero errors
or warnings and the lineage receipt binds all raw files to both prepared codec
trees. This validates the small qualification path, not broad corpus quality or
linguistic generalization.

## Training and adapter

The bounded run used one update, batch size 1, rank 8, alpha 16, BF16, seed 42,
and one validation row.

- training loss: 11.8883
- training gradient norm: 7.8226
- validation loss: 11.8794
- adapter weights: 18,958,448 bytes
- adapter weights SHA-256:
  `4561ba83a1a45e301b26150e358befdd50365ab17421f20b0f0d6c2933e08be4`
- selected adapter archive SHA-256:
  `3c11d39068b12a12b9e8ba1bdc0866f6e175663c0289bb8d8910f057534b9fb4`

One update and two loss values do not establish useful adaptation. The selected
archive intentionally excludes optimizer and other training-only state.

## Frozen objective evaluation

Both planned rows were valid 44.1 kHz mono WAV files. Every required metric had
2 of 2 eligible observations.

| Prompt | WER | ECAPA similarity | Duration | RTF | Peak CUDA memory | Silence | Clipping |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `neutral-brief` | 0.0000 | 0.6488 | 9.38 s | 0.9895 | 2,502,739,456 B | 0.3800 | 0.0000 |
| `names-numbers` | 0.1333 | 0.7165 | 14.12 s | 0.7626 | 2,783,827,456 B | 0.3526 | 0.0000 |

Aggregate observations:

- invalid-output rate: 0.0
- mean WER: 0.0667
- mean ECAPA similarity: 0.6826
- mean duration: 11.75 seconds
- mean RTF: 0.8760
- mean peak CUDA memory: 2,643,283,456 bytes
- objective score SHA-256:
  `56d4f09f67b5b77290fa59312c51d43e6c63c5a4907900afd24a03328b165c21`
- evaluation bundle SHA-256:
  `56657be308b682e6f9f0a371e59a869390f34bf44e412baff9c6f66ed8f07fab`

The neutral row matched the requested transcript under faster-whisper. The
names-and-numbers row rendered `Sze Min` as `C-Min` and normalized punctuation
and time formatting. The row-level failure stays visible instead of being
hidden by the aggregate mean.

ECAPA similarity is an objective proxy. No blind listening was performed, and
there is no matched Base candidate. The run therefore makes no claim that the
adapter improved speaker identity, accent, cadence, naturalness, or fatigue.

## Package and restore

The retained package is 21,514,240 bytes with SHA-256
`3aea0ae24f55da6ea9fb6e43408b1fc8ad03ea3298c2af2f74736ba36d890329`.
It lives outside both the checkout and lifecycle work tree. The package contains
the selected adapter, evaluation bundle, experiment manifest, generation plan,
dataset lineage, preflight evidence, and smoke WAV.

The restore verifier requires the expected outer package digest, verifies every
listed package file, rejects unlisted files, checks the complete external base
tree, validates the packaged speaker catalog and assignment plan against live
reference bytes, reloads the archived adapter, and generates one frozen row.
Two new restore directories produced the exact same neutral WAV SHA-256 as the
original evaluation row:
`9dcf0f810e6934e77e9b6121eced8ed75fbb4df552e11a816d9fea95a469dbc7`.
A deliberately wrong expected package digest failed before creating its output
directory.

These are same-host restores from one local package copy. They do not prove an
independent backup, clean-host portability, disaster recovery, retention policy,
access control, or distribution permission.

## Failure that improved portability

The first live lifecycle at commit
`917983b9821f8a26b3ceb49c6a5184a04262ebbc` completed training but failed
before inference because Python 3.10 does not support the newer `filter=`
argument to `TarFile.extractall`. The repository already validated every member
against traversal, links, devices, and unexpected prefixes. Commit `5f89ec0`
retained those explicit safety checks and removed the Python 3.12-only argument.
The full rerun then passed all five stages on the supported Python 3.10 CUDA
environment.

## Remaining work

- Run a preregistered matched Base-versus-LoRA matrix with more prompts and
  seeds.
- Perform criterion-scoped blind listening for identity, accent, pronunciation,
  cadence, monotony, naturalness, artifacts, and fatigue.
- Reproduce an actual interrupted and uninterrupted training pair before
  claiming guarded-resume equivalence.
- Restore from an independent copy on a clean host.
- Qualify MPS against the exact same package and plan before making runtime
  equivalence claims.
- Treat ONNX and SGLang adapter-aware export as separate qualification paths.
