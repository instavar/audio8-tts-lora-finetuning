# Audio8 bounded LoRA CUDA qualification

Date: 2026-08-14

## Purpose

Qualify the current public Audio8 LoRA path through one bounded CUDA update,
safe adapter save, fresh-process held-out generation, complete objective
evaluation, and a content-addressed local package. This is a lifecycle smoke,
not a quality or convergence experiment.

## Frozen controls

- upstream code: `3346560df718d33096ac2fef7e5c7984ee5248e6`
- base model revision: `1b17c91db5f4dccb6914aa4aa5cb0e56661a6c17`
- candidate: `audio8-lora-bounded-step1`
- prompts: `neutral-brief` and `names-numbers`
- seed: 42
- training: one update, batch size 1, rank 8, alpha 16, BF16
- runtime: PyTorch CUDA on the RTX 3090 Ti

The exact public companion revision, split and prepared-data hashes, evaluator
revision, learned-extractor revisions, selected adapter hash, and package hash
will be recorded in the result report after the run.

## Gates

1. The checkout, complete base-model tree, generation plan, reference audio,
   reference transcript, and raw-to-prepared lineage are bound before training.
2. Raw train, validation, and test splits pass the shared corpus audit.
3. One selected adapter checkpoint reloads from its inference-only archive in a
   new process.
4. Both frozen rows retain valid or invalid attempts and every objective metric
   required by the plan reaches complete row coverage before packaging.
5. The evaluation archive retains the frozen speaker catalog and assignment
   plan.
6. The final package is published to an operator-declared directory outside the
   checkout, lifecycle work tree, base model, and prepared-data roots.

## Explicit non-claims

Success does not establish convergence, adaptation benefit, speaker identity,
Singapore English fidelity, cadence, naturalness, blind-listening preference,
long-run stability, resume equivalence, MPS equivalence, ONNX or SGLang adapter
equivalence, production readiness, remote backup, or distribution permission.
