# Evaluator 0.45 resume readiness

Date: 2026-08-14, Asia/Singapore

## Finding

Audio8's guarded LoRA checkpoints already expose the five independent final
state roles required by Instavar Voice evaluator 0.45:

| Evaluator role | Audio8 checkpoint member |
| --- | --- |
| `model_state` | `adapter_model.safetensors` or `adapter_model.bin` |
| `optimizer_state` | `optimizer.pt` |
| `scheduler_state` | `scheduler.pt` |
| `trainer_state` | `trainer_state.json` |
| `rng_state` | the single `rng_state*.pth` member |

The checkpoint sidecar hashes every continuation member. Its validator requires
optimizer, scheduler, trainer, RNG, and model or adapter state before a sidecar
can become the completion marker. This makes a future LoRA checkpoint eligible
for evaluator role declaration without adding duplicate state files.

`evaluator_lora_artifact_paths(...)` provides the repository-specific mapping.
It rejects ambiguous adapter or RNG files and cross-role hardlinks. The helper
does not replace guarded sidecar authority or the evaluator's live rehashing.

Full-SFT checkpoints can use the evaluator only when their model state has an
unambiguous independently declared file or tree. This report does not collapse
sharded model weights and unrelated continuation state into one artifact role.

## OOD controls

Existing dependency-free tests cover:

- a complete adapter checkpoint;
- optimizer byte drift after sidecar publication;
- absent continuation files;
- sharded full-model sidecar eligibility;
- symlinked checkpoint members;
- ambiguous LoRA model or RNG role mapping;
- cross-role hardlinks;
- sidecar overwrite attempts;
- completed-target, trajectory-fork, and distributed-resume rejection; and
- retention behavior around unowned or best checkpoints.

The public contract workflow pins evaluator revision
`29c38cfd86b889abc8b79df063c817dd8f684903` and verifies the live-conditioning
receipt builder and resume artifact comparison APIs.

## Evidence boundary

No new model training was run for this adoption change. The mapping establishes
repository readiness and dependency-free contract coverage only. It does not
show byte equality between an uninterrupted and interrupted-resumed run.

A stronger comparison must start from preregistered live conditioning:

- the exact Base artifact;
- the dataset-lineage receipt;
- the effective training-controls artifact; and
- the initial model and runtime state.

Both independently stored final checkpoints must reach the same target update.
The resumed condition also needs an observed interruption receipt and a
checkpoint before the target. Evaluator 0.45 then rehashes the four conditioning
artifacts and compares the five final-state roles. Even a passing report does
not prove trainer semantics, numerical equivalence inside opaque serialization,
model quality, or perceptual improvement.

The 2026-08-14 CUDA and MPS lifecycle evidence predates schema 1.1 receipts and
is not retroactively upgraded.
