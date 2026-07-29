# Apple MPS LoRA smoke test

Date: 2026-07-30

## Question

Can the public LoRA runner complete one training step on a 16 GB Apple Silicon
Mac, save an adapter, reload it in a new process and generate held-out audio?

## Environment

- MacBook Pro with Apple M2 Pro and 16 GB unified memory
- macOS 26.5.2
- Python 3.11.15
- PyTorch 2.6.0 and torchaudio 2.6.0
- Audio8 model revision `1b17c91db5f4dccb6914aa4aa5cb0e56661a6c17`
- one authorised private training clip with a precomputed codec target
- LoRA rank 8, alpha 16, 4,730,880 trainable parameters
- batch size 1, maximum sequence length 256 and float32

The private recording, transcript-derived manifest, codec target and adapter are
not part of this repository.

## Result

The one-step run passed on MPS. The training step took 10.78 seconds and saved
an 18,958,448-byte adapter. A separate inference process loaded the base model
and adapter, then generated a 4.04-second, 44.1 kHz WAV for a sentence absent
from the training row. The waveform was finite and non-silent, with peak
amplitude 0.447 and RMS amplitude 0.0476.

macOS reported a maximum resident set size of about 2.58 GB for the training
process and 4.06 GB for the adapter inference process. Unified-memory and MPS
allocations are not completely represented by process RSS, so these figures
are operational observations rather than total memory requirements.

## What this proves

This proves the bounded software path on this machine: one-step MPS training,
adapter persistence, adapter reload and held-out waveform generation.

It does not prove that a one-step adapter improves voice similarity,
intelligibility or naturalness. It also does not prove that long training runs
fit comfortably in 16 GB. Those claims require longer runs, held-out speech
evaluation and monitoring for memory growth or thermal throttling.
