#!/usr/bin/env python3
"""Run an Instavar Voice generation plan through Audio8's single-load batch path."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-plan", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--reference-audio")
    parser.add_argument("--reference-text")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--retry-max-new-tokens", type=int, default=2000)
    parser.add_argument("--greedy", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    args = parse_args()
    plan = json.loads(args.generation_plan.read_text(encoding="utf-8"))
    rows = [row for row in plan.get("samples", []) if row.get("candidate_id") == args.candidate_id]
    if not rows:
        raise ValueError(f"generation plan has no rows for candidate {args.candidate_id!r}")
    if bool(args.reference_audio) != bool(args.reference_text):
        raise ValueError("reference audio and text must be provided together")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch_manifest = args.output_dir / "audio8-input.jsonl"
    output_root = args.output_dir / "audio"
    with batch_manifest.open("w", encoding="utf-8") as target:
        for row in rows:
            payload = {
                "id": row["sample_id"],
                "text": row["text"],
                "seed": row["seed"],
            }
            if args.reference_audio:
                payload.update(
                    {
                        "reference_audio": str(Path(args.reference_audio).resolve()),
                        "reference_text": args.reference_text,
                    }
                )
            target.write(json.dumps(payload, ensure_ascii=False) + "\n")

    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "audio8_tts_infer.py"),
        "--input-jsonl",
        str(batch_manifest),
        "--output-dir",
        str(output_root),
        "--manifest",
        str(args.output_dir / "audio8-manifest.jsonl"),
        "--failures",
        str(args.output_dir / "audio8-failures.jsonl"),
        "--model",
        args.model,
        "--adapter",
        args.adapter,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--batch-size",
        str(args.batch_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--retry-max-new-tokens",
        str(args.retry_max_new_tokens),
        "--overwrite",
    ]
    if args.greedy:
        command.append("--greedy")
    completed = subprocess.run(command, check=False)

    records = {row["id"]: row for row in read_jsonl(args.output_dir / "audio8-manifest.jsonl")}
    failures = {row["id"]: row for row in read_jsonl(args.output_dir / "audio8-failures.jsonl")}
    observations: list[dict] = []
    for row in rows:
        record = records.get(row["sample_id"])
        failure = failures.get(row["sample_id"])
        observation = {
            "sample_id": row["sample_id"],
            "candidate_id": row["candidate_id"],
            "prompt_id": row["prompt_id"],
            "category": row["category"],
            "seed": row["seed"],
            "requested_text": row["text"],
            "valid": False,
            "runtime": "audio8_pytorch_adapter",
            "instruction_applied": False,
        }
        if record and record.get("status") in {"OK", "NO_EOS"}:
            audio = Path(record["output_audio"])
            info = sf.info(audio)
            observation.update(
                {
                    "valid": record["status"] == "OK" and info.frames > 0,
                    "audio_path": str(audio),
                    "audio_sha256": sha256(audio),
                    "audio_duration_seconds": float(info.duration),
                    "generation_status": record["status"],
                }
            )
        if failure:
            observation.update(failure)
        if row.get("instruction"):
            observation["instruction_note"] = "Audio8 has no separate instruction input in this path."
        observations.append(observation)
    (args.output_dir / "generation-observations.json").write_text(
        json.dumps(observations, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if completed.returncode == 0 and all(row["valid"] for row in observations) else 1


if __name__ == "__main__":
    raise SystemExit(main())
