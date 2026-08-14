#!/usr/bin/env python3
"""Verify and reload one packaged Audio8 adapter in a clean directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from instavar_voice_lifecycle import _extract, _sha256, _tree_manifest


REPO_ROOT = Path(__file__).parents[1]


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_extract(source: Path, destination: Path, *, prefix: str) -> Path:
    if source.is_symlink() or not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"archive is missing, empty, or unsafe: {source}")
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(source, "r") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError("archive is empty")
        for member in members:
            parts = PurePosixPath(member.name).parts
            target = (destination / member.name).resolve()
            if (
                not parts
                or parts[0] != prefix
                or ".." in parts
                or not target.is_relative_to(destination.resolve())
                or member.issym()
                or member.islnk()
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"unsafe archive member: {member.name}")
        archive.extractall(destination, members=members)
    root = destination / prefix
    if not root.is_dir() or not any(path.is_file() for path in root.rglob("*")):
        raise ValueError(f"archive did not contain a non-empty {prefix} root")
    return root


def _verify_package_files(root: Path) -> dict[str, Any]:
    manifest_path = root / "package-manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("package-manifest.json is missing or unsafe")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0.0":
        raise ValueError("unsupported package manifest schema")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("package manifest has no files")
    expected_names = {"package-manifest.json"}
    for row in rows:
        name = row.get("path") if isinstance(row, dict) else None
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("package manifest contains an unsafe file name")
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"packaged file is missing or unsafe: {name}")
        if path.stat().st_size != row.get("bytes") or _sha256(path) != row.get("sha256"):
            raise ValueError(f"packaged file identity mismatch: {name}")
        expected_names.add(name)
    actual_names = {path.name for path in root.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise ValueError("package contains files outside package-manifest.json")
    return manifest


def _verify_reference_inputs(
    evaluation_root: Path,
    generation_plan: dict[str, Any],
    reference_audio: Path,
    reference_transcript: Path,
    reference_id: str,
) -> None:
    catalog = json.loads(
        (evaluation_root / "speaker-reference-catalog.json").read_text(encoding="utf-8")
    )
    catalog_payload = {key: value for key, value in catalog.items() if key != "catalog_sha256"}
    if catalog.get("catalog_sha256") != _canonical_sha256(catalog_payload):
        raise ValueError("speaker reference catalog hash mismatch")
    references = [
        row for row in catalog.get("references", []) if row.get("reference_id") == reference_id
    ]
    if len(references) != 1:
        raise ValueError("speaker reference catalog must contain one selected reference")
    selected = references[0]
    if (
        reference_audio.stat().st_size != selected["audio"]["bytes"]
        or _sha256(reference_audio) != selected["audio"]["sha256"]
    ):
        raise ValueError("reference audio does not match the packaged speaker catalog")
    if (
        reference_transcript.stat().st_size != selected["transcript"]["bytes"]
        or _sha256(reference_transcript) != selected["transcript"]["sha256"]
    ):
        raise ValueError("reference transcript does not match the packaged speaker catalog")
    assignment = json.loads(
        (evaluation_root / "speaker-reference-plan.json").read_text(encoding="utf-8")
    )
    assignment_payload = {
        key: value for key, value in assignment.items() if key != "assignment_plan_sha256"
    }
    if assignment.get("assignment_plan_sha256") != _canonical_sha256(assignment_payload):
        raise ValueError("speaker reference assignment plan hash mismatch")
    if assignment.get("generation_plan_sha256") != _canonical_sha256(generation_plan):
        raise ValueError("speaker reference plan does not bind the packaged generation plan")
    if assignment.get("reference_catalog_sha256") != catalog.get("catalog_sha256"):
        raise ValueError("speaker reference plan does not bind the packaged catalog")


def _select_row(plan: dict[str, Any], candidate_id: str, prompt_id: str) -> dict[str, Any]:
    samples = plan.get("samples", [])
    rows = [
        row
        for row in samples
        if row.get("candidate_id") == candidate_id and row.get("prompt_id") == prompt_id
    ]
    if len(rows) != 1:
        available = sorted(
            {
                (str(row.get("candidate_id")), str(row.get("prompt_id")))
                for row in samples
                if row.get("candidate_id") and row.get("prompt_id")
            }
        )
        choices = ", ".join(f"{candidate}/{prompt}" for candidate, prompt in available)
        raise ValueError(
            "generation plan must contain one selected restore row for "
            f"{candidate_id}/{prompt_id}; available candidate/prompt pairs: "
            f"{choices or 'none'}"
        )
    return rows[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--expected-package-sha256", required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--reference-transcript", type=Path, required=True)
    parser.add_argument("--reference-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "mps"), required=True)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), required=True
    )
    return parser.parse_args()


def main() -> int:
    import soundfile as sf

    args = parse_args()
    package = args.package.resolve()
    base_model = args.base_model.resolve()
    reference_audio = args.reference_audio.resolve()
    reference_transcript = args.reference_transcript.resolve()
    for name, path, directory in (
        ("package", package, False),
        ("base model", base_model, True),
        ("reference audio", reference_audio, False),
        ("reference transcript", reference_transcript, False),
    ):
        valid = path.is_dir() if directory else path.is_file()
        if path.is_symlink() or not valid:
            raise ValueError(f"{name} is missing or unsafe: {path}")
    if (
        len(args.expected_package_sha256) != 64
        or any(c not in "0123456789abcdef" for c in args.expected_package_sha256)
    ):
        raise ValueError("expected package SHA-256 must be one lowercase digest")
    package_sha256 = _sha256(package)
    if package_sha256 != args.expected_package_sha256:
        raise ValueError("package does not match the expected SHA-256")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("output directory must not already exist")
    package_root = _safe_extract(package, args.output_dir / "outer", prefix="package")
    manifest = _verify_package_files(package_root)
    base_identity = _tree_manifest(base_model)
    if base_identity["sha256"] != manifest.get("external_base_model_sha256"):
        raise ValueError("base model does not match the packaged external identity")
    plan_path = package_root / "generation-plan.json"
    generation_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    row = _select_row(generation_plan, args.candidate_id, args.prompt_id)
    evaluation_root = _safe_extract(
        package_root / "evaluation-bundle.tar",
        args.output_dir / "evaluation",
        prefix="evaluation",
    )
    _verify_reference_inputs(
        evaluation_root,
        generation_plan,
        reference_audio,
        reference_transcript,
        args.reference_id,
    )
    adapter = _extract(
        package_root / "selected-adapter.tar", args.output_dir / "adapter-reload"
    )
    output = args.output_dir / "restored.wav"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "audio8_tts_infer.py"),
            "--model",
            str(base_model),
            "--adapter",
            str(adapter),
            "--reference-audio",
            str(reference_audio),
            "--reference-text",
            reference_transcript.read_text(encoding="utf-8").strip(),
            "--text",
            row["text"],
            "--seed",
            str(row["seed"]),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--greedy",
            "--overwrite",
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    info = sf.info(output)
    if info.frames < 1 or info.channels != 1:
        raise ValueError("restored inference did not produce a valid mono WAV")
    receipt = {
        "schema_version": "1.0.0",
        "status": "passed",
        "package_sha256": package_sha256,
        "package_manifest_sha256": _sha256(package_root / "package-manifest.json"),
        "base_model_sha256": base_identity["sha256"],
        "selected_adapter_sha256": _sha256(package_root / "selected-adapter.tar"),
        "generation_plan_sha256": _sha256(plan_path),
        "candidate_id": args.candidate_id,
        "prompt_id": args.prompt_id,
        "seed": row["seed"],
        "runtime_id": f"pytorch_{args.device}",
        "output_audio_sha256": _sha256(output),
        "output_audio_bytes": output.stat().st_size,
        "sample_rate_hz": info.samplerate,
        "duration_seconds": info.duration,
        "evidence_boundary": (
            "This verifies one package, external base-model identity, frozen reference, "
            "adapter reload, and generated waveform. It does not establish perceptual "
            "quality, backup independence, cross-host portability, or runtime equivalence."
        ),
    }
    receipt_path = args.output_dir / "restore-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
