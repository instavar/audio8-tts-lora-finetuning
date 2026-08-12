# Instavar Voice conformance

This repository declares its model-specific adaptation and runtime surface in `instavar-voice-capabilities.json`. The manifest and executable [`instavar-voice-backend.json`](instavar-voice-backend.json) LoRA recipe use the public [Instavar Voice evaluation contract](https://github.com/instavar/instavar-voice-evaluation) pinned by CI to merge commit `e689ee121ee4a6ae07793ef1c49d70c48b0ad271`.

The executable recipe requires an explicit CUDA or MPS device and compatible dtype. It audits raw grouped splits, isolates trainer output, selects one exact checkpoint child, reloads it in a fresh process, runs the frozen evaluation plan, and packages the adapter with source and evaluation evidence. One passed runtime does not establish equivalence with the other runtime.

Capability schema 1.2 separates validated CUDA training from bounded MPS reload and inference evidence. It records the missing selected adapter as an artifact-retention blocker rather than a model failure.

A capability marked `supported` means the referenced repository evidence reaches the stated engineering boundary. It does not prove perceptual quality, accent fidelity, commercial suitability, or equivalence across untested runtimes. `unverified_for_adapter` keeps an upstream or community runtime visible without implying that this repository's adapted artifact works there.

The common evaluation pack separates deterministic audio diagnostics and objective proxies from blinded human listening. It intentionally defines no universal composite score.

For a reference and candidate runtime, generate the same frozen prompt with recorded settings and run `instavar-voice-eval compare-audio reference.wav candidate.wav`. The result exposes format and signal-level deltas while explicitly refusing to claim runtime equivalence. Establish intelligibility, speaker identity, accent, cadence, and naturalness separately through objective proxies and the blind listening pack.

Before training, use the contract's `audit-corpus` command with explicit train, validation, and test manifests. Supply a parent recording or source identifier through `--group-field` so the audit can reject leakage across splits. File presence and manifest integrity do not prove transcript accuracy or audio quality, which remain separate checks.

`audio8_tts_lora.sh` can enforce the audit before it starts training. Set `AUDIT_CORPUS=1`, the three `RAW_*_JSONL` paths, and `INSTAVAR_VOICE_EVAL_DIR`. This audits the raw audio manifests rather than the codec-index manifests, whose `target_codes` paths are training artifacts rather than source recordings.

Validate locally with a checkout of the pinned contract:

```bash
python /path/to/instavar-voice-evaluation/main.py validate-repository .
```
