# Fresh-clone CUDA reproduction

Date: 2026-07-30

## Question

Can the previously tested Audio8 LoRA extension reproduce from a new upstream
clone and a new Python environment on the CUDA host, without relying on the old
working checkout?

## Isolated setup

- fresh clone of `Audio8-AI/Audio8_TTS` at
  `3346560df718d33096ac2fef7e5c7984ee5248e6`
- new Python 3.12 virtual environment
- PyTorch 2.6.0 and torchaudio 2.6.0 with CUDA 12.4
- Transformers 4.57.6, Accelerate 1.14.0 and PEFT 0.20.0
- NVIDIA RTX 3090 Ti with 24 GB VRAM
- pinned Audio8 model revision
  `1b17c91db5f4dccb6914aa4aa5cb0e56661a6c17`
- 128 authorised private training rows and 32 private validation rows
- LoRA rank 8, alpha 16, batch size 1, maximum length 512 and BF16
- 100 optimization steps with evaluation and saving every 20 steps

The private data, codec targets, model weights and resulting adapter remain
outside this repository.

## Result

The fresh-clone 100-step run passed. Trainer time was 113.19 seconds. Including
model loading from the busy data volume, end-of-run evaluation, saving and merge,
the process wall time was 11 minutes 13.58 seconds. Final evaluation loss was
10.2695. The run saved:

- an 18,958,448-byte PEFT adapter;
- a 1,202,342,528-byte merged model export.

The process reported a maximum resident set size of about 2.83 GB. This does not
include all CUDA allocations and should not be used as a VRAM estimate.

## Exact public-code cross-check

The public training, inference and launcher files were copied into a second
fresh upstream clone. SHA-256 hashes matched the local publication commit. The
exact public runner then passed one-step training, adapter reload and held-out
generation on Apple MPS, as recorded in `macos-mps-smoke-test.md`.

A second CUDA held-out generation was attempted after the 100-step run. It was
stopped before GPU execution because reads from `/mnt/work` were repeatedly
blocked in the Linux storage queue. At that point the volume was 97 percent full
and unrelated workloads were active. This is an incomplete CUDA inference check,
not a speech-generation failure. CUDA held-out generation should be rerun from a
less contended volume before making a fresh-clone CUDA end-to-end claim.

## Scope

The result reproduces CUDA training, evaluation, adapter saving and merge from a
clean environment. Together with the exact public-code MPS test, it gives good
evidence that the source-only repository is usable on both device families.

It does not establish adapter quality, long-run Mac stability, generality beyond
the tested model revision and private dataset, or CUDA held-out generation under
a clean storage environment.
