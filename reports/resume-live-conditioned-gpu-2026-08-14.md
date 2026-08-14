# Live-conditioned interrupted-resume GPU evidence

Date: 2026-08-14, Asia/Singapore

## Result

Audio8 companion revision
`c0b5bfe84672417c41bc5153cc8a3aaecc8d15e1` and Instavar Voice evaluator
revision `29c38cfd86b889abc8b79df063c817dd8f684903` completed a paired
uninterrupted and interrupted-resumed LoRA drill on an NVIDIA GeForce RTX 3090
Ti. Evaluator 0.45 reported:

- status `passed`;
- claim tier `byte_exact_live_conditioned_artifact_set`;
- exact resume artifact equivalence `true`;
- conditioning artifacts verified `true`; and
- independent final artifact storage verified `true`.

The evaluator correctly reports `proves_training_semantics: false` and
`proves_model_quality: false`. This is checkpoint evidence, not a quality or
adaptation-benefit result. No audio was generated.

## Live conditioning

Both fresh processes explicitly loaded the same serialized initial adapter.
The trainer content-bound all four files in that directory into each guarded
checkpoint contract. The initial adapter weights SHA-256 was:

`b71f132bdf9ae1e5d93d56cb9b0556314c00cbc57720ff36c0f534a690402064`

The evaluator independently rehashed:

- the exact local Base model tree;
- the prepared train and validation lineage receipt;
- the common two-update training controls; and
- the serialized initial adapter tree.

The runtime was Python 3.10.19, Torch 2.9.0 with CUDA 12.8, Transformers
4.57.1, PEFT 0.18.1, NVIDIA driver 580.173.02, and a 24,564 MiB RTX 3090 Ti.

## Interruption and continuation

The control process completed two updates. A separate process group published
the immutable `checkpoint-1` sidecar and was then sent `SIGTERM`. Its exit
status was `143`. After exit, `checkpoint-2` was absent and no partial sidecar
remained. A new process loaded the shared initial adapter, verified the exact
guarded checkpoint contract, deserialized the trusted continuation state, and
completed update 2.

The observed loss values were:

| Path | Update 1 | Update 2 |
| --- | ---: | ---: |
| uninterrupted | 11.8883 | 11.8911 |
| interrupted then resumed | 11.8883 before interruption | 11.8911 after resume |

All five final role files were byte-identical:

| Evaluator role | SHA-256 |
| --- | --- |
| model state | `8c46c2eccd98c00bb7a4793ea63d64807a2d97d12be1a5f0be3771e7a04bc5e8` |
| optimizer state | `fa47bec10297b12466cecf6c61ad56a6eb63c0ba90ef264c85aa138c8cbb82dd` |
| scheduler state | `568262fc4130d3849f1def1999b72aa339876c77107d63645315cbd97cc84919` |
| trainer state | `23e9b4c9aab6cfeb11923cd5af5d858246b90b4cbfd482b5bc762b82d213b953` |
| RNG state | `de69a2834426ff9ef8199d077e00892579278af31d8969d77f98235b5cfc010a` |

The evaluator report SHA-256 is
`d98d23c737d81cc95ca439f2522cfcd5957c095a7e9288fc919e16bdf1da0e1d`.
The run-summary SHA-256 is
`10887bb361cad555904ee46a9ade53620f2af182a2f822b8970e14a8fd6146a0`.

## OOD failures found first

The first initializer invocation named the parent download directory instead
of the nested Hugging Face snapshot. `AutoModel` failed before any adapter was
published. This confirmed the no-overwrite initializer failed safely.

A preserved Torch 2.5.1 attempt then found two real resume blockers:

1. Hugging Face generated a timestamped default `logging_dir`, which the
   guarded contract treated as a semantic training control. A new process
   therefore failed contract verification before deserialization.
2. Supplying the exact historical log path passed contract verification, but
   current Transformers rejected optimizer-state loading under Torch 2.5.1
   because it requires Torch 2.6 or newer after CVE-2025-32434.

Revision `c0b5bfe84672417c41bc5153cc8a3aaecc8d15e1` excludes the observational
log path and rejects guarded training under unsafe Torch versions before model
loading. The successful pair used that revision from a clean checkout.

## Scope and retention

The result applies only to two sample-batched updates, one prepared training
row, one seed, world size 1, zero data-loader workers, one Base tree, one
serialized initial adapter, and the named dependency and GPU stack. It does
not generalize to mid-epoch recovery, accumulated gradients, stochastic
workers, multiple ranks, another Torch release, MPS, long training, inference,
quality, voice identity, accent, cadence, or distribution rights.

Full checkpoints remain at:

`/mnt/work/chee-wei-jie/voice-models/instavar-audio8-resume-live-045-torch29-20260814`

A 22-file compact export was copied to:

- `/mnt/work/chee-wei-jie/voice-model-outputs/evaluation/audio8-resume-live-045-torch29-20260814`
- `/Users/CheeWeiJie/Downloads/desktop-tailscale-tts/audio8-resume-live-045-torch29-20260814`

Remote and local compact-export hashes matched file-for-file.
