# Audio8 exact-package MPS restore

Date: 2026-08-14, Asia/Singapore

## Scope

This probe moved the exact package produced by the bounded CUDA lifecycle to a
16 GB Apple M2 Pro and restored the frozen neutral row twice plus the frozen
names-and-numbers row once through PyTorch MPS. It reused the packaged
candidate, prompts, seed, adapter archive, complete base-model tree identity,
reference audio, and reference transcript.

The executed restore revision was
`3b9630fe474c3756ad46a4f4720b531d6277125f`. Both hosted workflows passed at
that revision.

This validates one second-host package reload and same-runtime repeatability.
It does not establish CUDA-versus-MPS output equivalence, perceptual quality,
adaptation benefit, broad MPS stability, or backup durability.

## Bound artifacts

- package SHA-256:
  `3aea0ae24f55da6ea9fb6e43408b1fc8ad03ea3298c2af2f74736ba36d890329`
- selected adapter archive SHA-256:
  `3c11d39068b12a12b9e8ba1bdc0866f6e175663c0289bb8d8910f057534b9fb4`
- 49-file base-model tree SHA-256:
  `5a2b2e402ebb42aa645d22137f39be75cdf43b80a1853be70eb93d710d1055ce`
- reference WAV SHA-256:
  `2dc2a3d83dab1e5569d1adac7828c907acc78271cb495d80228b15ca6e460237`
- reference transcript SHA-256:
  `7b5f531abde272946e3638bbd35736923e1b3562779deff69aed968bf471ba1e`
- candidate: `audio8-lora-bounded-step1`
- prompt: `neutral-brief`
- seed: 42

The Mac already held the 16 model payload files at the same revision and byte
hashes in its Hugging Face cache. The probe dereferenced them into a new
symlink-free tree and copied the small remote download-metadata tree needed to
reconstruct the exact 49-file packaged base identity. It did not weaken the
package verifier or substitute a semantic model identifier for byte identity.

## MPS environment

- MacBook Pro with Apple M2 Pro and 16 GB unified memory
- macOS 26.5.2 build 25F84, arm64
- Python 3.11.15
- PyTorch 2.6.0 with MPS available
- Transformers 4.57.6
- PEFT 0.20.0
- dtype: float32
- `PYTORCH_ENABLE_MPS_FALLBACK=1`

System memory was 61 percent free before model loading. Both successful runs
completed with zero process swaps.

## OOD failure and correction

The first restore attempt failed before model loading because the README used
the capability label `audio8-selected-adapter`, while the immutable package
plan names the candidate `audio8-lora-bounded-step1`. The old exception only
said that one selected row was required, which did not expose the valid
candidate and prompt values.

Revision `3b9630f` corrects the command and makes a failed selection list every
available `candidate/prompt` pair. A dependency-free regression test covers the
message. This failure did not create audio or load model weights.

Generalisability: restore tools that require opaque identifiers should expose
valid immutable choices in errors or discovery output. A valid package should
not require the operator to guess internal plan identifiers.

## Repeatability result

Two fresh output directories completed successfully. Both receipts were
byte-identical with SHA-256
`15caa3abd1018152c1bfbbb7ca079dec6d1102b65f2acd65f3f9cefdd0af6505`.
Both generated WAV files were byte-identical with SHA-256
`7395d041909354ef24d97ebb1f1dfd9d81ddb2606e7a30d4d525069d081d1f37`.

| Run | Output duration | Item time | Total wall time | Maximum process RSS |
| --- | ---: | ---: | ---: | ---: |
| MPS restore 1 | 9.6131 s | 48.04 s | 54.75 s | 3,099,983,872 B |
| MPS restore 2 | 9.6131 s | 36.39 s | 42.15 s | 4,063,772,672 B |

The RSS values are process observations, not total unified-memory or MPS
allocation. The timing spread also prevents treating either single run as a
stable performance benchmark.

## CUDA and MPS diagnostics

The original CUDA neutral output and repeated MPS output match channel count,
44.1 kHz sample rate, and 16-bit sample width, but not waveform bytes.

| Diagnostic | CUDA BF16 | MPS float32 | MPS minus CUDA |
| --- | ---: | ---: | ---: |
| duration | 9.3809 s | 9.6131 s | +0.2322 s |
| peak amplitude | 0.6406 | 0.5002 | -0.1405 |
| RMS amplitude | 0.1014 | 0.0809 | -0.0205 |
| silence fraction | 0.3800 | 0.3678 | -0.0122 |
| clipping fraction | 0.0 | 0.0 | 0.0 |
| faster-whisper WER | 0.0 | 0.0 | 0.0 |
| SpeechBrain ECAPA similarity | 0.6488 | 0.6785 | +0.0298 |

The MPS WAV was copied back to the evaluator host for the same content-bound
faster-whisper and SpeechBrain extractors. The resulting objective-observation
artifact has SHA-256
`d4cca724f0da7bc353ccb97fec1875bdee1d1575c2d9467dfd206f77720a715e`.
Learned-extractor execution happened on Linux and CUDA, but each result binds
the MPS output audio hash.

The ECAPA difference is a one-row proxy and not evidence that MPS sounds more
like the target speaker. The waveform and deterministic audio differences mean
this probe does not establish cross-runtime equivalence. Two identical MPS runs
establish only exact repeatability in the tested MPS environment.

## Second frozen MPS row

The names-and-numbers row also restored successfully as a valid 44.1 kHz WAV:

- WAV SHA-256:
  `c26b88f39c87e04b230b1b4a6f895c7e769bf3004aa46f420db546bf3b01039b`
- duration: 13.7927 seconds
- item generation time: 53.09 seconds
- total wall time: 59.16 seconds
- maximum process RSS: 4,064,018,432 bytes
- process swaps: 0
- objective-observation SHA-256:
  `2ef8092a130a701f840d84af376b6ac4974e44b3ea99b88f27bec1775aa67a3b`

| Diagnostic | CUDA BF16 | MPS float32 | MPS minus CUDA |
| --- | ---: | ---: | ---: |
| duration | 14.1177 s | 13.7927 s | -0.3251 s |
| peak amplitude | 0.6406 | 0.6157 | -0.0250 |
| RMS amplitude | 0.1008 | 0.0833 | -0.0175 |
| silence fraction | 0.3526 | 0.4151 | +0.0625 |
| clipping fraction | 0.0 | 0.0 | 0.0 |
| faster-whisper WER | 0.1333 | 0.2333 | +0.1000 |
| SpeechBrain ECAPA similarity | 0.7165 | 0.7782 | +0.0617 |

The MPS ASR hypothesis rendered `Sze Min` as `seimin` and flattened punctuation
and currency formatting. WER worsened while ECAPA increased. These opposing
proxy movements do not identify a quality winner. They make the row-level
runtime interaction visible and reinforce the need for criterion-scoped blind
listening.

## Evidence retention and limits

The local probe root is
`/private/tmp/audio8-mps-cross-runtime-20260814.2RixYw`. It contains the exact
package, materialized model tree, two restore directories, receipts, WAV files,
and deterministic comparison artifacts. The content-bound learned-extractor
artifacts are retained at
`/mnt/work/chee-wei-jie/voice-models/instavar-audio8-mps-cross-runtime-20260814`.

The package was copied from the CUDA host to the Mac, but both copies remain
operator-local. This is second-host portability evidence, not an independent
backup or disaster-recovery drill. No blind listening was performed. Two
prompts and one seed ran on MPS, and the plan's CUDA peak-memory requirement is
not available from the MPS runner.

## Remaining work

- Define a plan-valid MPS memory metric or explicitly use a cross-runtime plan
  that does not require CUDA peak allocation.
- Run matched Base and LoRA candidates on both runtimes before making adaptation
  or runtime-interaction claims.
- Complete criterion-scoped blind listening.
- Restore from independently managed storage on a clean host.
