#!/usr/bin/env python3
"""Execute Audio8 LoRA through the Instavar Voice lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).parents[1]
UPSTREAM_REVISION = "3346560df718d33096ac2fef7e5c7984ee5248e6"


def _path(name: str, *, directory: bool = False) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    unresolved = Path(value).expanduser()
    if unresolved.is_symlink():
        raise FileNotFoundError(f"{name} is a symlink: {unresolved}")
    path = unresolved.resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid or path.is_symlink():
        raise FileNotFoundError(f"{name} is missing or unsafe: {path}")
    return path


def _work() -> Path:
    return _path("INSTAVAR_VOICE_WORK_DIR", directory=True)


def _persistent_package_root(*, protect_inputs: bool = True) -> Path:
    root = _path("PERSISTED_PACKAGE_ROOT", directory=True)
    work = _work()
    repository = REPO_ROOT.resolve()
    if root == work or root.is_relative_to(work):
        raise ValueError("PERSISTED_PACKAGE_ROOT must be outside the lifecycle work directory")
    if root == repository or root.is_relative_to(repository):
        raise ValueError("PERSISTED_PACKAGE_ROOT must be outside the repository checkout")
    if protect_inputs:
        protected = {
            "BASE_MODEL_DIR": _path("BASE_MODEL_DIR", directory=True),
            "PREPARED_TRAIN_ROOT": _path("PREPARED_TRAIN_ROOT", directory=True),
            "PREPARED_VALIDATION_ROOT": _path(
                "PREPARED_VALIDATION_ROOT", directory=True
            ),
        }
        for name, input_root in protected.items():
            if root == input_root or root.is_relative_to(input_root):
                raise ValueError(f"PERSISTED_PACKAGE_ROOT must not mutate {name}")
    return root


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _probe_persistent_package_root(root: Path) -> dict[str, Any]:
    probe_path: Path | None = None
    linked_path: Path | None = None
    linked_created = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=root,
            prefix=".instavar-voice-persistence-probe.",
            suffix=".partial",
            delete=False,
        ) as probe:
            probe_path = Path(probe.name)
            probe.write(b"instavar-voice-persistence-probe-v1\n")
            probe.flush()
            os.fsync(probe.fileno())
        linked_path = probe_path.with_suffix(".linked")
        os.link(probe_path, linked_path)
        linked_created = True
        _fsync_directory(root)
        if linked_path.read_bytes() != probe_path.read_bytes():
            raise ValueError("persistent package root failed its atomic publication probe")
        identity = root.stat()
        return {
            "writable": True,
            "atomic_hard_link": True,
            "device": identity.st_dev,
            "inode": identity.st_ino,
        }
    except OSError as error:
        raise ValueError(
            f"PERSISTED_PACKAGE_ROOT cannot publish an atomic package: {error}"
        ) from error
    finally:
        if linked_path is not None and linked_created:
            linked_path.unlink(missing_ok=True)
        if probe_path is not None:
            probe_path.unlink(missing_ok=True)


def _locked_persistent_package_root(preflight: dict[str, Any]) -> Path:
    root = _persistent_package_root()
    recorded_root = preflight.get("persistent_package_root")
    recorded_device = preflight.get("persistence_probe", {}).get("device")
    recorded_inode = preflight.get("persistence_probe", {}).get("inode")
    identity = root.stat()
    if (
        recorded_root != str(root)
        or recorded_device != identity.st_dev
        or recorded_inode != identity.st_ino
    ):
        raise ValueError("PERSISTED_PACKAGE_ROOT changed after preflight")
    return root


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    path = Path(value)
    if not value or value in {".", ".."} or path.is_absolute() or len(path.parts) != 1:
        raise ValueError("SELECTED_ADAPTER_NAME must be one safe child directory")
    return value


def _runtime() -> str:
    device = os.environ["DEVICE"].strip().casefold()
    dtype = os.environ["DTYPE"].strip().casefold()
    if device == "mps":
        if dtype != "float32":
            raise ValueError("Audio8 MPS lifecycle requires DTYPE=float32")
        return "pytorch_mps"
    if device == "cuda":
        if dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("Audio8 CUDA DTYPE must be bfloat16, float16, or float32")
        return "pytorch_cuda"
    raise ValueError("DEVICE must explicitly name cuda or mps")


def _run(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
    capture: bool = False,
) -> str:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=capture,
        text=capture,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip() if capture else ""
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: {command[0]}: {detail}"
        )
    return (result.stdout or "").strip() if capture else ""


def _git_head() -> str:
    return _run(["git", "rev-parse", "HEAD"], capture=True)


def _git_clean() -> bool:
    return not _run(["git", "status", "--porcelain=v1", "--untracked-files=all"], capture=True)


def _tree_manifest(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"base model tree contains a symlink: {path}")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
        elif not path.is_dir():
            raise ValueError(f"base model tree contains an unsupported entry: {path}")
    if not files:
        raise ValueError("base model tree contains no files")
    digest = hashlib.sha256()
    for record in files:
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return {"sha256": digest.hexdigest(), "file_count": len(files), "files": files}


def _audit_prepared_manifest(path: Path) -> tuple[dict[str, Any], set[str]]:
    sample_ids: set[str] = set()
    rows = 0
    with path.open(encoding="utf-8") as source:
        for line_number, raw in enumerate(source, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            sample_id = str(row.get("id", "")).strip()
            if not sample_id or sample_id in sample_ids:
                raise ValueError(f"{path}:{line_number}: missing or duplicate id")
            for field in ("target_codes", "reference_codes"):
                value = str(row.get(field, "")).strip()
                if field == "reference_codes" and not value:
                    continue
                artifact = Path(value).expanduser()
                if not artifact.is_absolute():
                    artifact = path.parent / artifact
                if artifact.is_symlink():
                    raise ValueError(f"{path}:{line_number}: {field} is a symlink")
                artifact = artifact.resolve()
                if not artifact.is_file() or artifact.stat().st_size == 0:
                    raise ValueError(f"{path}:{line_number}: {field} is missing, empty, or unsafe")
            if bool(row.get("reference_codes")) != bool(str(row.get("reference_text", "")).strip()):
                raise ValueError(f"{path}:{line_number}: reference codes and text must be paired")
            sample_ids.add(sample_id)
            rows += 1
    if rows == 0:
        raise ValueError(f"prepared manifest contains no rows: {path}")
    return {"path": str(path), "sha256": _sha256(path), "rows": rows}, sample_ids


def _verify_prepared_root(manifest: Path, root: Path, *, label: str) -> None:
    if not manifest.is_relative_to(root):
        raise ValueError(f"{label} manifest must be inside its declared prepared-data root")
    with manifest.open(encoding="utf-8") as source:
        for line_number, raw in enumerate(source, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            for field in ("target_codes", "reference_codes", "reference_audio"):
                value = str(row.get(field, "")).strip()
                if not value:
                    continue
                artifact = Path(value).expanduser()
                if not artifact.is_absolute():
                    artifact = manifest.parent / artifact
                if not artifact.resolve().is_relative_to(root):
                    raise ValueError(
                        f"{label}:{line_number}: {field} escapes its declared "
                        "prepared-data root"
                    )


def _verify_dataset_lineage() -> dict[str, Any]:
    from instavar_voice_lab.lineage import verify_dataset_lineage

    train = _path("TRAIN_JSONL")
    validation = _path("EVAL_JSONL")
    train_root = _path("PREPARED_TRAIN_ROOT", directory=True)
    validation_root = _path("PREPARED_VALIDATION_ROOT", directory=True)
    _verify_prepared_root(train, train_root, label="train")
    _verify_prepared_root(validation, validation_root, label="validation")
    document = json.loads(_path("DATASET_LINEAGE").read_text(encoding="utf-8"))
    return verify_dataset_lineage(
        document,
        producer_revision=_git_head(),
        inputs={
            "raw_train": (_path("RAW_TRAIN_JSONL"), "file"),
            "raw_validation": (_path("RAW_VALIDATION_JSONL"), "file"),
            "raw_test": (_path("RAW_TEST_JSONL"), "file"),
        },
        outputs={
            "prepared_train": (train_root, "tree"),
            "prepared_validation": (validation_root, "tree"),
        },
    )


def _archive(source: Path, destination: Path, *, arcname: str) -> None:
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"archive source must be a non-symlink directory: {source}")
    count = 0
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"archive source contains a symlink: {path}")
        if path.is_file():
            count += 1
        elif not path.is_dir():
            raise ValueError(f"archive source contains an unsupported entry: {path}")
    if count == 0:
        raise ValueError("archive source contains no files")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w") as archive:
        archive.add(source, arcname=arcname, recursive=True)


def _verify_persisted_package(path: Path, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"persisted package is missing, empty, or unsafe: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"persisted package hash mismatch: expected {expected_sha256}, got {actual_sha256}"
        )


def _persist_package(source: Path, root: Path) -> dict[str, Any]:
    if source.is_symlink() or not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"package source is missing, empty, or unsafe: {source}")
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"persistent package root is missing or unsafe: {root}")
    package_sha256 = _sha256(source)
    destination = root / f"audio8-adapter-package-sha256-{package_sha256}.tar"
    reused_existing = destination.exists() or destination.is_symlink()
    if reused_existing:
        _verify_persisted_package(destination, package_sha256)
    else:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=root,
                prefix=f".{destination.name}.",
                suffix=".partial",
                delete=False,
            ) as target:
                temporary_path = Path(target.name)
                with source.open("rb") as package:
                    shutil.copyfileobj(package, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            _verify_persisted_package(temporary_path, package_sha256)
            try:
                os.link(temporary_path, destination)
            except FileExistsError:
                reused_existing = True
            else:
                _fsync_directory(root)
            _verify_persisted_package(destination, package_sha256)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return {
        "schema_version": "1.0.0",
        "package_sha256": package_sha256,
        "package_bytes": source.stat().st_size,
        "persisted_path": str(destination),
        "reused_existing": reused_existing,
    }


def _stage_adapter(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"selected adapter must be a non-symlink directory: {source}")
    required = ("adapter_config.json", "adapter_model.safetensors")
    destination.mkdir(parents=True, exist_ok=False)
    for name in required:
        artifact = source / name
        if artifact.is_symlink() or not artifact.is_file() or artifact.stat().st_size == 0:
            raise ValueError(f"selected adapter is missing safe inference artifact: {artifact}")
        shutil.copyfile(artifact, destination / name)
    try:
        config = json.loads((destination / "adapter_config.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("adapter_config.json is invalid JSON") from error
    if not isinstance(config, dict) or not config:
        raise ValueError("adapter_config.json must be a non-empty object")


def _extract(source: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(source, "r") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError("adapter archive is empty")
        for member in members:
            parts = PurePosixPath(member.name).parts
            target = (destination / member.name).resolve()
            if (
                not parts
                or parts[0] != "adapter"
                or ".." in parts
                or not target.is_relative_to(destination.resolve())
                or member.issym()
                or member.islnk()
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"unsafe adapter archive member: {member.name}")
        archive.extractall(destination, members=members, filter="data")
    adapter = destination / "adapter"
    if not adapter.is_dir() or not any(path.is_file() for path in adapter.rglob("*")):
        raise ValueError("adapter archive did not contain a non-empty adapter root")
    return adapter


def _training_settings() -> dict[str, str]:
    defaults = {
        "MAX_STEPS": "100",
        "MAX_LENGTH": "512",
        "BATCH_SIZE": "1",
        "EVAL_BATCH_SIZE": "1",
        "GRADIENT_ACCUMULATION_STEPS": "1",
        "LEARNING_RATE": "2e-5",
        "WARMUP_STEPS": "10",
        "SAVE_STEPS": "20",
        "SAVE_TOTAL_LIMIT": "5",
        "LORA_R": "8",
        "LORA_ALPHA": "16",
        "LORA_DROPOUT": "0",
        "LORA_TARGET_MODULES": "wqkv,wo,w1,w2,w3",
        "SEED": "42",
        "DATA_SEED": "42",
    }
    return {name: os.environ.get(name, value) for name, value in defaults.items()}


def _preflight() -> None:
    from instavar_voice_lab.corpus import audit_corpus

    lineage = _verify_dataset_lineage()
    experiment = json.loads(_path("INSTAVAR_VOICE_EXPERIMENT_MANIFEST").read_text(encoding="utf-8"))
    revision = _git_head()
    if not _git_clean():
        raise ValueError("repository must be clean; use a work directory outside the checkout")
    backend = experiment.get("backend", {})
    if backend.get("instavar_revision") != revision:
        raise ValueError("experiment backend.instavar_revision does not match the Audio8 checkout")
    if backend.get("upstream_revision") != UPSTREAM_REVISION:
        raise ValueError(
            "experiment backend.upstream_revision does not match the pinned Audio8 upstream base"
        )
    splits = {
        "train": _path("RAW_TRAIN_JSONL"),
        "validation": _path("RAW_VALIDATION_JSONL"),
        "test": _path("RAW_TEST_JSONL"),
    }
    audit = audit_corpus(splits, group_field=os.environ.get("CORPUS_GROUP_FIELD") or None)
    if audit["status"] != "passed":
        raise ValueError("corpus audit failed: " + "; ".join(audit["errors"]))
    train_prepared, train_ids = _audit_prepared_manifest(_path("TRAIN_JSONL"))
    validation_prepared, validation_ids = _audit_prepared_manifest(_path("EVAL_JSONL"))
    overlap = sorted(train_ids.intersection(validation_ids))
    if overlap:
        raise ValueError(
            "prepared train and validation manifests overlap by id: " + ", ".join(overlap[:10])
        )
    _path("REFERENCE_AUDIO")
    base = _path("BASE_MODEL_DIR", directory=True)
    plan = json.loads(_path("GENERATION_PLAN").read_text(encoding="utf-8"))
    rows = [
        row
        for row in plan.get("samples", [])
        if row.get("candidate_id") == os.environ["CANDIDATE_ID"]
    ]
    if plan.get("schema_version") not in {"1.0.0", "1.1.0"} or not rows:
        raise ValueError(
            "GENERATION_PLAN must be schema 1.0.0 or 1.1.0 and contain CANDIDATE_ID rows"
        )
    selected = _safe_name(os.environ["SELECTED_ADAPTER_NAME"])
    persistent_package_root = _persistent_package_root()
    persistence_probe = _probe_persistent_package_root(persistent_package_root)
    _write_json(
        _work() / "preflight" / "preflight.json",
        {
            "schema_version": "1.0.0",
            "status": "passed",
            "companion_revision": revision,
            "upstream_revision": UPSTREAM_REVISION,
            "runtime_id": _runtime(),
            "device": os.environ["DEVICE"],
            "dtype": os.environ["DTYPE"],
            "selected_adapter_name": selected,
            "persistent_package_root": str(persistent_package_root),
            "persistence_probe": persistence_probe,
            "base_model": _tree_manifest(base),
            "corpus_audit": audit,
            "prepared_manifests": {
                "train": train_prepared,
                "validation": validation_prepared,
            },
            "generation_rows": len(rows),
            "training_settings": _training_settings(),
            "dataset_lineage": lineage,
        },
    )


def _train() -> None:
    _verify_dataset_lineage()
    work = _work()
    output = work / "train" / "output"
    environment = os.environ.copy()
    environment.update(_training_settings())
    environment.update(
        {
            "PYTHON": sys.executable,
            "MODEL": str(_path("BASE_MODEL_DIR", directory=True)),
            "TRAIN_JSONL": str(_path("TRAIN_JSONL")),
            "EVAL_JSONL": str(_path("EVAL_JSONL")),
            "OUTPUT_DIR": str(output),
            "EXPORT_DIR": "",
            "AUDIT_CORPUS": "0",
            "OVERWRITE_OUTPUT_DIR": "false",
            "REPORT_TO": "none",
            "BF16": "true" if os.environ["DTYPE"].casefold() == "bfloat16" else "false",
            "FP16": "true" if os.environ["DTYPE"].casefold() == "float16" else "false",
        }
    )
    _run(["bash", "audio8_tts_lora.sh"], environment=environment)
    selected = output / _safe_name(os.environ["SELECTED_ADAPTER_NAME"])
    staged = work / "train" / "selected-adapter"
    _stage_adapter(selected, staged)
    _archive(staged, work / "train" / "selected-adapter.tar", arcname="adapter")


def _infer() -> None:
    work = _work()
    adapter = _extract(work / "train" / "selected-adapter.tar", work / "infer" / "reload")
    output = work / "infer" / "candidate.wav"
    command = [
        sys.executable,
        "audio8_tts_infer.py",
        "--model",
        str(_path("BASE_MODEL_DIR", directory=True)),
        "--adapter",
        str(adapter),
        "--reference-audio",
        str(_path("REFERENCE_AUDIO")),
        "--reference-text",
        os.environ["REFERENCE_TEXT"],
        "--text",
        os.environ.get("SMOKE_TEXT", "A held-out sentence verifies adapter reload."),
        "--device",
        os.environ["DEVICE"],
        "--dtype",
        os.environ["DTYPE"],
        "--greedy",
        "--overwrite",
        "--output",
        str(output),
    ]
    _run(command)
    if not output.is_file() or output.stat().st_size == 0:
        raise ValueError("fresh-process adapter inference did not produce audio")


def _evaluate() -> None:
    work = _work()
    adapter = _extract(work / "train" / "selected-adapter.tar", work / "evaluate" / "reload")
    output = work / "evaluate" / "output"
    command = [
        sys.executable,
        "scripts/run_evaluation_suite.py",
        "--generation-plan",
        os.environ["GENERATION_PLAN"],
        "--candidate-id",
        os.environ["CANDIDATE_ID"],
        "--model",
        str(_path("BASE_MODEL_DIR", directory=True)),
        "--adapter",
        str(adapter),
        "--reference-audio",
        str(_path("REFERENCE_AUDIO")),
        "--reference-text",
        os.environ["REFERENCE_TEXT"],
        "--output-dir",
        str(output),
        "--device",
        os.environ["DEVICE"],
        "--dtype",
        os.environ["DTYPE"],
        "--greedy",
        "--allow-invalid-output",
    ]
    _run(command)
    raw_observations = output / "generation-observations.json"
    receipt = output / "generation-attempt-receipt.json"
    bound_observations = output / "objective-observations.json"
    plan = _path("GENERATION_PLAN")
    producer_revision = _git_head()
    _run([
        sys.executable, "-m", "instavar_voice_lab.cli", "build-generation-attempt-receipt",
        str(raw_observations), "--plan", str(plan), "--audio-base-dir", str(output),
        "--producer-name", "audio8-evaluation-runner", "--producer-revision", producer_revision,
        "--output", str(receipt),
    ])
    _run([
        sys.executable, "-m", "instavar_voice_lab.cli", "apply-generation-attempt-receipt",
        str(raw_observations), str(receipt), "--plan", str(plan), "--audio-base-dir", str(output),
        "--output", str(bound_observations),
    ])
    _archive(output, work / "evaluate" / "evaluation-bundle.tar", arcname="evaluation")


def _package() -> None:
    work = _work()
    preflight = json.loads((work / "preflight" / "preflight.json").read_text(encoding="utf-8"))
    staging = work / "package" / "staging"
    staging.mkdir(parents=True, exist_ok=False)
    sources = {
        "selected-adapter.tar": work / "train" / "selected-adapter.tar",
        "evaluation-bundle.tar": work / "evaluate" / "evaluation-bundle.tar",
        "preflight.json": work / "preflight" / "preflight.json",
        "smoke-candidate.wav": work / "infer" / "candidate.wav",
        "experiment-manifest.json": _path("INSTAVAR_VOICE_EXPERIMENT_MANIFEST"),
        "generation-plan.json": _path("GENERATION_PLAN"),
        "dataset-lineage.json": _path("DATASET_LINEAGE"),
    }
    for name, source in sources.items():
        if source.is_symlink() or not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f"package source is missing, empty, or unsafe: {source}")
        shutil.copyfile(source, staging / name)
    files = [
        {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(staging.iterdir())
        if path.is_file()
    ]
    _write_json(
        staging / "package-manifest.json",
        {
            "schema_version": "1.0.0",
            "backend_id": "audio8-tts-lora-pytorch",
            "runtime_id": preflight["runtime_id"],
            "external_base_model_sha256": preflight["base_model"]["sha256"],
            "files": files,
            "evidence_boundary": (
                "The adapter and evidence completed one explicit runtime lifecycle. "
                "Perceptual quality, cross-runtime equivalence, and distribution rights "
                "remain separate gates."
            ),
        },
    )
    package = work / "package" / "adapter-package.tar"
    _archive(staging, package, arcname="package")
    receipt = _persist_package(package, _locked_persistent_package_root(preflight))
    _write_json(work / "package" / "persisted-package.json", receipt)


def run(stage: str) -> None:
    actions = {
        "preflight": _preflight,
        "train": _train,
        "infer": _infer,
        "evaluate": _evaluate,
        "package": _package,
    }
    if stage not in actions:
        raise ValueError(f"unknown lifecycle stage: {stage}")
    actions[stage]()
    if stage in {"preflight", "train"}:
        _verify_dataset_lineage()
    _write_json(
        Path(os.environ["INSTAVAR_VOICE_STAGE_RESULT"]),
        {"schema_version": "1.0.0", "stage": stage, "status": "passed"},
    )


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else argv
    if len(values) != 1:
        print("usage: instavar_voice_lifecycle.py STAGE", file=sys.stderr)
        return 2
    try:
        run(values[0])
    except (
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
        tarfile.TarError,
    ) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
