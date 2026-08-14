"""Dependency-free guarded checkpoint contract for Audio8 TTS training."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TextIO

SCHEMA_VERSION = "1.0.0"
SIDECAR_NAME = "instavar-resume-contract.json"
LOCK_NAME = ".instavar-training.lock"
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
_IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")


class ResumeContractError(ValueError):
    """Raised when continuation state is unsafe or does not match the current run."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): canonical_value(item) for key, item in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [canonical_value(item) for item in value]
    return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_without_symlink(path: str | Path, *, expected: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise ResumeContractError(f"{expected} must not be a symlink: {raw}")
    resolved = raw.resolve(strict=True)
    if expected == "file" and not resolved.is_file():
        raise ResumeContractError(f"Expected a regular file: {resolved}")
    if expected == "directory" and not resolved.is_dir():
        raise ResumeContractError(f"Expected a directory: {resolved}")
    return resolved


def file_identity(path: str | Path) -> dict[str, Any]:
    resolved = _resolved_without_symlink(path, expected="file")
    stat = resolved.stat()
    return {"path": str(resolved), "sha256": sha256_file(resolved), "size": stat.st_size}


def tree_identity(path: str | Path) -> dict[str, Any]:
    resolved = _resolved_without_symlink(path, expected="directory")
    root_stat = resolved.stat()
    files: list[dict[str, Any]] = []
    for item in sorted(resolved.rglob("*")):
        if item.is_symlink():
            raise ResumeContractError(f"Directory identity rejects symlinks: {item}")
        if item.is_file():
            identity = file_identity(item)
            identity["path"] = item.relative_to(resolved).as_posix()
            files.append(identity)
    return {
        "path": str(resolved),
        "device": root_stat.st_dev,
        "inode": root_stat.st_ino,
        "files": files,
    }


def initial_adapter_contract_files(value: str | Path) -> list[Path]:
    root = Path(value).expanduser()
    if root.is_symlink():
        raise ResumeContractError("Initial adapter directory must not be a symlink")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ResumeContractError("Initial adapter path must be a directory")
    files: list[Path] = []
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise ResumeContractError(f"Initial adapter tree rejects symlinks: {item}")
        if item.is_file():
            files.append(item)
    names = {path.relative_to(root).as_posix() for path in files}
    model_names = names & {"adapter_model.safetensors", "adapter_model.bin"}
    if "adapter_config.json" not in names or len(model_names) != 1:
        raise ResumeContractError(
            "Initial adapter needs adapter_config.json and exactly one adapter model file"
        )
    return files


def model_identity(model_name_or_path: str | Path, resolved_revision: str | None) -> dict[str, Any]:
    raw = Path(model_name_or_path).expanduser()
    if raw.exists() or raw.is_symlink():
        if raw.is_dir() and not raw.is_symlink():
            return {"kind": "local_tree", "tree": tree_identity(raw)}
        if raw.is_file() and not raw.is_symlink():
            return {"kind": "local_file", "file": file_identity(raw)}
        raise ResumeContractError(f"Base model path is unsafe: {raw}")
    revision = str(resolved_revision or "").strip()
    if not _IMMUTABLE_REVISION_RE.fullmatch(revision):
        raise ResumeContractError(
            "A remote base model needs the immutable resolved commit hash before guarded training"
        )
    return {
        "kind": "huggingface_revision",
        "model_id": str(model_name_or_path),
        "revision": revision.lower(),
    }


def output_identity(output_dir: str | Path) -> dict[str, Any]:
    output = _resolved_without_symlink(output_dir, expected="directory")
    stat = output.stat()
    return {"path": str(output), "device": stat.st_dev, "inode": stat.st_ino}


def build_contract(
    *,
    output_dir: str | Path,
    mode: str,
    base_model: str | Path,
    base_revision: str | None,
    input_files: Mapping[str, Iterable[str | Path]],
    source_files: Iterable[str | Path],
    training_config: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    inputs: dict[str, list[dict[str, Any]]] = {}
    for role, paths in sorted(input_files.items()):
        seen: set[str] = set()
        identities: list[dict[str, Any]] = []
        for path in paths:
            identity = file_identity(path)
            if identity["path"] not in seen:
                identities.append(identity)
                seen.add(identity["path"])
        inputs[role] = identities
    sources = [
        file_identity(path) for path in sorted((Path(path) for path in source_files), key=str)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": str(mode),
        "output_dir": output_identity(output_dir),
        "base_model": model_identity(base_model, base_revision),
        "inputs": inputs,
        "sources": sources,
        "training_config": canonical_value(training_config),
        "runtime": canonical_value(runtime),
    }


def contract_digest(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest()


def resolve_resume_request(
    resume_from: str | None,
    legacy_resume_mode: str | None,
) -> str | None:
    explicit = str(resume_from or "").strip()
    legacy = str(legacy_resume_mode or "none").strip()
    legacy_disabled = legacy.casefold() in {"", "none", "false", "no"}
    if explicit and not legacy_disabled:
        raise ResumeContractError("Use only --resume_from for guarded resume")
    selected = explicit or ("" if legacy_disabled else legacy)
    if not selected:
        return None
    if selected.casefold() in {"auto", "latest", "true", "yes"}:
        raise ResumeContractError(
            "Implicit latest checkpoint selection is not allowed; provide one exact checkpoint path"
        )
    return selected


def _checkpoint_step(name: str) -> int | None:
    match = _CHECKPOINT_RE.fullmatch(name)
    return int(match.group(1)) if match else None


def resolve_checkpoint(checkpoint: str | Path, output_dir: str | Path) -> Path:
    raw = Path(checkpoint).expanduser()
    if raw.is_symlink():
        raise ResumeContractError(f"Checkpoint symlinks are not allowed: {raw}")
    resolved = raw.resolve(strict=True)
    output = _resolved_without_symlink(output_dir, expected="directory")
    if resolved.parent != output:
        raise ResumeContractError("Resume checkpoint must be a direct child of output_dir")
    if not resolved.is_dir() or _checkpoint_step(resolved.name) is None:
        raise ResumeContractError("Resume checkpoint must be an exact checkpoint-N directory")
    return resolved


def checkpoint_children(output_dir: str | Path) -> list[Path]:
    output = _resolved_without_symlink(output_dir, expected="directory")
    checkpoints: list[Path] = []
    for item in output.iterdir():
        if _checkpoint_step(item.name) is None:
            continue
        if item.is_symlink() or not item.is_dir():
            raise ResumeContractError(f"Unsafe checkpoint child: {item}")
        checkpoints.append(item)
    return sorted(checkpoints, key=lambda path: _checkpoint_step(path.name) or -1)


def require_fresh_output(output_dir: str | Path) -> None:
    output = _resolved_without_symlink(output_dir, expected="directory")
    conflicts = sorted(item.name for item in output.iterdir() if item.name != LOCK_NAME)
    if conflicts:
        raise ResumeContractError(
            "Guarded fresh training needs an empty output directory; use a fresh directory or "
            f"explicit --resume_from ({', '.join(conflicts[:5])})"
        )


def checkpoint_manifest(checkpoint_dir: str | Path) -> list[dict[str, Any]]:
    checkpoint = _resolved_without_symlink(checkpoint_dir, expected="directory")
    files: list[dict[str, Any]] = []
    for item in sorted(checkpoint.rglob("*")):
        if item.is_symlink():
            raise ResumeContractError(f"Checkpoint manifests reject symlinks: {item}")
        if not item.is_file():
            continue
        relative = item.relative_to(checkpoint).as_posix()
        if relative == SIDECAR_NAME or relative.startswith(f".{SIDECAR_NAME}."):
            continue
        identity = file_identity(item)
        identity["path"] = relative
        files.append(identity)
    return files


def _read_trainer_step(checkpoint: Path) -> int:
    state_path = checkpoint / "trainer_state.json"
    if state_path.is_symlink() or not state_path.is_file():
        raise ResumeContractError(f"Checkpoint has no safe trainer_state.json: {checkpoint}")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ResumeContractError("trainer_state.json is invalid JSON") from error
    step = state.get("global_step")
    if not isinstance(step, int) or step < 1:
        raise ResumeContractError("trainer_state.json has an invalid completed global_step")
    if _checkpoint_step(checkpoint.name) != step:
        raise ResumeContractError("Checkpoint name and completed global_step disagree")
    return step


def _require_continuation_files(manifest: list[dict[str, Any]]) -> None:
    names = {item["path"] for item in manifest}
    required = {"optimizer.pt", "scheduler.pt", "trainer_state.json"}
    missing = sorted(required - names)
    if missing:
        raise ResumeContractError(f"Checkpoint omits continuation files: {', '.join(missing)}")
    if not any(Path(name).name.startswith("rng_state") and name.endswith(".pth") for name in names):
        raise ResumeContractError("Checkpoint omits RNG continuation state")
    weight_names = {
        "adapter_model.safetensors",
        "adapter_model.bin",
        "model.safetensors",
        "pytorch_model.bin",
    }
    basenames = {Path(name).name for name in names}
    has_sharded_safetensors = "model.safetensors.index.json" in basenames and any(
        name.startswith("model-") and name.endswith(".safetensors") for name in basenames
    )
    has_sharded_pytorch = "pytorch_model.bin.index.json" in basenames and any(
        name.startswith("pytorch_model-") and name.endswith(".bin") for name in basenames
    )
    if not (basenames & weight_names or has_sharded_safetensors or has_sharded_pytorch):
        raise ResumeContractError("Checkpoint omits adapter or model weights")


def evaluator_lora_artifact_paths(checkpoint_dir: str | Path) -> dict[str, Path]:
    """Return one independent checkpoint member for each evaluator 0.45 role."""
    checkpoint = _resolved_without_symlink(checkpoint_dir, expected="directory")
    manifest = checkpoint_manifest(checkpoint)
    _require_continuation_files(manifest)
    names = {item["path"] for item in manifest}

    model_candidates = sorted(
        name
        for name in names
        if name in {"adapter_model.safetensors", "adapter_model.bin"}
    )
    rng_candidates = sorted(
        name
        for name in names
        if Path(name).name.startswith("rng_state") and name.endswith(".pth")
    )
    if len(model_candidates) != 1:
        raise ResumeContractError(
            "Evaluator LoRA mapping needs exactly one adapter model state file"
        )
    if len(rng_candidates) != 1:
        raise ResumeContractError(
            "Evaluator LoRA mapping needs exactly one RNG state file"
        )

    relative_by_role = {
        "model_state": model_candidates[0],
        "optimizer_state": "optimizer.pt",
        "scheduler_state": "scheduler.pt",
        "trainer_state": "trainer_state.json",
        "rng_state": rng_candidates[0],
    }
    resolved = {role: checkpoint / relative for role, relative in relative_by_role.items()}
    identities = [(path.stat().st_dev, path.stat().st_ino) for path in resolved.values()]
    if len(identities) != len(set(identities)):
        raise ResumeContractError("Evaluator LoRA artifact roles must not share hardlinks")
    return resolved


def write_checkpoint_sidecar(
    checkpoint_dir: str | Path,
    *,
    output_dir: str | Path,
    contract: Mapping[str, Any],
) -> Path:
    checkpoint = resolve_checkpoint(checkpoint_dir, output_dir)
    target = checkpoint / SIDECAR_NAME
    if target.exists() or target.is_symlink():
        raise ResumeContractError(f"Refusing to overwrite checkpoint sidecar: {target}")
    completed_step = _read_trainer_step(checkpoint)
    manifest = checkpoint_manifest(checkpoint)
    _require_continuation_files(manifest)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_name": checkpoint.name,
        "completed_updates": completed_step,
        "contract_sha256": contract_digest(contract),
        "contract": canonical_value(contract),
        "files": manifest,
    }
    temporary = checkpoint / f".{SIDECAR_NAME}.{os.getpid()}.partial"
    created_temporary = False
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            created_temporary = True
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError as error:
            raise ResumeContractError(
                f"Checkpoint sidecar appeared concurrently: {target}"
            ) from error
        _fsync_directory(checkpoint)
    finally:
        if created_temporary:
            temporary.unlink(missing_ok=True)
    return target


def _load_sidecar(checkpoint: Path) -> dict[str, Any]:
    sidecar_path = checkpoint / SIDECAR_NAME
    if sidecar_path.is_symlink() or not sidecar_path.is_file():
        raise ResumeContractError(f"Checkpoint has no safe {SIDECAR_NAME}: {checkpoint}")
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ResumeContractError("Checkpoint sidecar is invalid JSON") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ResumeContractError("Unsupported checkpoint sidecar schema")
    return payload


def _validate_sidecar(
    checkpoint: Path,
    *,
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _load_sidecar(checkpoint)
    if payload.get("contract_sha256") != contract_digest(expected_contract):
        raise ResumeContractError("Checkpoint run contract drift detected")
    if payload.get("contract") != canonical_value(expected_contract):
        raise ResumeContractError("Checkpoint run contract payload does not match")
    if payload.get("checkpoint_name") != checkpoint.name:
        raise ResumeContractError("Checkpoint sidecar names a different directory")
    completed_step = _read_trainer_step(checkpoint)
    if payload.get("completed_updates") != completed_step:
        raise ResumeContractError("Checkpoint sidecar and trainer state disagree")
    expected_manifest = payload.get("files")
    if not isinstance(expected_manifest, list):
        raise ResumeContractError("Checkpoint sidecar has no file manifest")
    actual_manifest = checkpoint_manifest(checkpoint)
    if expected_manifest != actual_manifest:
        raise ResumeContractError("Checkpoint file identity drift detected")
    _require_continuation_files(actual_manifest)
    return payload


def validate_resume_checkpoint(
    checkpoint: str | Path,
    *,
    output_dir: str | Path,
    expected_contract: Mapping[str, Any],
    trust_resume_state: bool,
    world_size: int,
) -> Path:
    if not trust_resume_state:
        raise ResumeContractError(
            "Trainer resume loads pickle-capable optimizer and RNG state; set "
            "--trust_resume_state true only for state you trust"
        )
    if world_size != 1:
        raise ResumeContractError(
            "Guarded resume supports world_size=1 only because rank-local loader and RNG "
            "state are not represented"
        )
    resolved = resolve_checkpoint(checkpoint, output_dir)
    payload = _validate_sidecar(resolved, expected_contract=expected_contract)
    children = checkpoint_children(output_dir)
    for child in children:
        if child != resolved:
            _validate_sidecar(child, expected_contract=expected_contract)
    if not children or children[-1] != resolved:
        raise ResumeContractError(
            "Resume checkpoint must be the newest owned checkpoint to avoid trajectory forks"
        )
    max_steps = expected_contract.get("training_config", {}).get("max_steps")
    if isinstance(max_steps, int) and max_steps > 0 and payload["completed_updates"] >= max_steps:
        raise ResumeContractError("Checkpoint already reached the configured max_steps target")
    return resolved


def assert_save_destination_absent(output_dir: str | Path, completed_step: int) -> None:
    output = _resolved_without_symlink(output_dir, expected="directory")
    target = output / f"checkpoint-{int(completed_step)}"
    if target.exists() or target.is_symlink():
        raise ResumeContractError(
            f"Refusing to overwrite or adopt checkpoint destination: {target}"
        )


def _owned_checkpoint(
    checkpoint: Path,
    *,
    expected_contract: Mapping[str, Any],
) -> tuple[int, int, int]:
    if checkpoint.is_symlink() or checkpoint.parent != checkpoint.parent.resolve():
        raise ResumeContractError(f"Unsafe checkpoint retention candidate: {checkpoint}")
    stat = checkpoint.stat()
    _validate_sidecar(checkpoint, expected_contract=expected_contract)
    step = _checkpoint_step(checkpoint.name)
    assert step is not None
    return step, stat.st_dev, stat.st_ino


def prune_owned_checkpoints(
    output_dir: str | Path,
    *,
    keep_last: int | None,
    expected_contract: Mapping[str, Any],
    best_checkpoint: str | None,
) -> list[Path]:
    if keep_last is None:
        return []
    if keep_last < 1:
        raise ResumeContractError("save_total_limit must be at least 1")
    checkpoints = checkpoint_children(output_dir)
    identities = {
        path: _owned_checkpoint(path, expected_contract=expected_contract) for path in checkpoints
    }
    ordered = list(checkpoints)
    best: Path | None = None
    if best_checkpoint:
        candidate = Path(best_checkpoint).expanduser()
        if candidate.is_symlink():
            raise ResumeContractError("Best checkpoint path must not be a symlink")
        candidate = candidate.resolve(strict=True)
        if candidate in identities:
            best = candidate
            ordered.remove(candidate)
            ordered.insert(max(0, len(ordered) - 1), candidate)
    effective_limit = keep_last
    if keep_last == 1 and best is not None and ordered and ordered[-1] != best:
        effective_limit = 2
    victims = ordered[: max(0, len(ordered) - effective_limit)]
    for victim in victims:
        _, device, inode = identities[victim]
        current = victim.stat()
        if current.st_dev != device or current.st_ino != inode:
            raise ResumeContractError(f"Checkpoint changed identity before pruning: {victim}")
        shutil.rmtree(victim)
        _fsync_directory(victim.parent)
    return victims


def acquire_output_lock(output_dir: str | Path) -> TextIO:
    raw = Path(output_dir).expanduser()
    if raw.is_symlink():
        raise ResumeContractError(f"Output directory must not be a symlink: {raw}")
    raw.mkdir(parents=True, exist_ok=True)
    output = _resolved_without_symlink(raw, expected="directory")
    lock_path = output / LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise ResumeContractError(f"Could not open safe training lock: {lock_path}") from error
    handle = os.fdopen(descriptor, "r+", encoding="utf-8")
    lock_stat = os.fstat(handle.fileno())
    if (
        not stat.S_ISREG(lock_stat.st_mode)
        or lock_stat.st_nlink != 1
        or lock_stat.st_uid != os.geteuid()
    ):
        handle.close()
        raise ResumeContractError(f"Training lock has unsafe ownership or link count: {lock_path}")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise ResumeContractError(f"Another guarded writer holds output_dir: {output}") from error
    path_stat = lock_path.stat(follow_symlinks=False)
    if (path_stat.st_dev, path_stat.st_ino) != (lock_stat.st_dev, lock_stat.st_ino):
        handle.close()
        raise ResumeContractError(f"Training lock path changed during acquisition: {lock_path}")
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
