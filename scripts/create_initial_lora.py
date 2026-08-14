#!/usr/bin/env python3
"""Publish one immutable initial Audio8 LoRA adapter for paired training runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, set_seed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--producer-revision", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--target-modules", default="wqkv,wo,w1,w2,w3")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.producer_revision) != 40 or any(
        value not in "0123456789abcdef" for value in args.producer_revision
    ):
        raise ValueError("--producer-revision must be a lowercase 40-character Git commit")
    output = args.output.expanduser()
    if not output.is_absolute():
        raise ValueError("--output must be absolute")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite initial adapter: {output}")
    targets = [value.strip() for value in args.target_modules.split(",") if value.strip()]
    if not targets:
        raise ValueError("--target-modules must contain at least one module")

    set_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=dtype)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules=targets,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    parent = output.parent.resolve(strict=True)
    if output.parent != parent or parent.is_symlink():
        raise ValueError("--output parent must not traverse a symlink")
    temporary = parent / f".{output.name}.{os.getpid()}.partial"
    temporary.mkdir(exist_ok=False)
    model.save_pretrained(temporary, safe_serialization=True)
    files = []
    for path in sorted(temporary.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"initial adapter publication rejects symlinks: {path}")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(temporary).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    receipt = {
        "schema_version": "1.0.0",
        "producer_repository": "instavar/audio8-tts-lora-finetuning",
        "producer_revision": args.producer_revision,
        "base_model": args.model,
        "seed": args.seed,
        "dtype": args.dtype,
        "lora": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "target_modules": targets,
        },
        "files": files,
        "evidence_boundary": (
            "These are serialized initial adapter bytes. Training must load and content-bind "
            "this directory before it can serve as evaluator conditioning evidence."
        ),
    }
    receipt_path = temporary / "initial-adapter-receipt.json"
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(temporary)
    os.replace(temporary, output)
    fsync_directory(parent)


if __name__ == "__main__":
    main()
