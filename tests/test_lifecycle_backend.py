from __future__ import annotations

import importlib.util
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from instavar_voice_lab.lineage import build_dataset_lineage

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audio8_lifecycle", ROOT / "scripts" / "instavar_voice_lifecycle.py"
)
assert SPEC and SPEC.loader
LIFECYCLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIFECYCLE)


class LifecycleBackendTests(unittest.TestCase):
    def test_backend_binds_both_explicit_pytorch_runtimes(self) -> None:
        spec = json.loads((ROOT / "instavar-voice-backend.json").read_text())
        self.assertEqual(spec["schema_version"], "1.2.0")
        self.assertEqual(spec["capability_binding"]["adaptation"], "lora")
        self.assertEqual(spec["capability_binding"]["runtime_ids"], ["pytorch_cuda", "pytorch_mps"])
        for stage in ("preflight", "train", "infer", "evaluate", "package"):
            self.assertEqual(spec["commands"][stage][-1], stage)

    def test_runtime_rejects_ambiguous_or_incompatible_device(self) -> None:
        with patch.dict(os.environ, {"DEVICE": "cuda", "DTYPE": "bfloat16"}, clear=False):
            self.assertEqual(LIFECYCLE._runtime(), "pytorch_cuda")
        with patch.dict(os.environ, {"DEVICE": "mps", "DTYPE": "float32"}, clear=False):
            self.assertEqual(LIFECYCLE._runtime(), "pytorch_mps")
        invalid = (
            ("auto", "auto"),
            ("cpu", "float32"),
            ("cuda:1", "bfloat16"),
            ("mps", "float16"),
        )
        for device, dtype in invalid:
            with (
                patch.dict(os.environ, {"DEVICE": device, "DTYPE": dtype}, clear=False),
                self.assertRaises(ValueError),
            ):
                LIFECYCLE._runtime()

    def test_selected_adapter_is_one_safe_child(self) -> None:
        self.assertEqual(LIFECYCLE._safe_name("checkpoint-20"), "checkpoint-20")
        for unsafe in ("", ".", "..", "../checkpoint", "nested/checkpoint", "/checkpoint"):
            with self.assertRaises(ValueError):
                LIFECYCLE._safe_name(unsafe)

    def test_tree_manifest_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text("{}\n")
            first = LIFECYCLE._tree_manifest(root)
            self.assertEqual(first, LIFECYCLE._tree_manifest(root))
            (root / "linked.bin").symlink_to(root / "config.json")
            with self.assertRaises(ValueError):
                LIFECYCLE._tree_manifest(root)

    def test_prepared_manifest_rejects_duplicate_ids_and_missing_codes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            codes = root / "codes.npy"
            codes.write_bytes(b"fixture")
            manifest = root / "prepared.jsonl"
            row = {"id": "sample", "text": "text", "target_codes": codes.name}
            manifest.write_text(json.dumps(row) + "\n")
            report, ids = LIFECYCLE._audit_prepared_manifest(manifest)
            self.assertEqual(report["rows"], 1)
            self.assertEqual(ids, {"sample"})
            manifest.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                LIFECYCLE._audit_prepared_manifest(manifest)

    def test_dataset_lineage_binds_raw_splits_to_codec_trees(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw: dict[str, Path] = {}
            for split in ("train", "validation", "test"):
                audio = root / f"{split}.wav"
                audio.write_bytes(b"audio")
                manifest = root / f"raw-{split}.jsonl"
                manifest.write_text(json.dumps({"audio": str(audio), "text": split}) + "\n")
                raw[split] = manifest
            prepared: dict[str, Path] = {}
            manifests: dict[str, Path] = {}
            for split in ("train", "validation"):
                split_root = root / f"prepared-{split}"
                split_root.mkdir()
                codes = split_root / "codes.npy"
                codes.write_bytes(split.encode())
                manifest = split_root / "manifest.jsonl"
                manifest.write_text(
                    json.dumps({"id": split, "text": split, "target_codes": codes.name}) + "\n"
                )
                prepared[split] = split_root
                manifests[split] = manifest
            receipt = root / "dataset-lineage.json"
            receipt.write_text(
                json.dumps(
                    build_dataset_lineage(
                        lineage_id="audio8-fixture-v1",
                        producer_repository="instavar/audio8-tts-lora-finetuning",
                        producer_revision="a" * 40,
                        inputs={
                            "raw_train": (raw["train"], "file"),
                            "raw_validation": (raw["validation"], "file"),
                            "raw_test": (raw["test"], "file"),
                        },
                        outputs={
                            "prepared_train": (prepared["train"], "tree"),
                            "prepared_validation": (prepared["validation"], "tree"),
                        },
                    )
                )
            )
            environment = {
                "RAW_TRAIN_JSONL": str(raw["train"]),
                "RAW_VALIDATION_JSONL": str(raw["validation"]),
                "RAW_TEST_JSONL": str(raw["test"]),
                "TRAIN_JSONL": str(manifests["train"]),
                "EVAL_JSONL": str(manifests["validation"]),
                "PREPARED_TRAIN_ROOT": str(prepared["train"]),
                "PREPARED_VALIDATION_ROOT": str(prepared["validation"]),
                "DATASET_LINEAGE": str(receipt),
            }
            with (
                patch.dict(os.environ, environment, clear=False),
                patch.object(LIFECYCLE, "_git_head", return_value="a" * 40),
            ):
                report = LIFECYCLE._verify_dataset_lineage()
            self.assertEqual(report["lineage_id"], "audio8-fixture-v1")
            (prepared["validation"] / "codes.npy").write_bytes(b"changed")
            with (
                patch.dict(os.environ, environment, clear=False),
                patch.object(LIFECYCLE, "_git_head", return_value="a" * 40),
                self.assertRaisesRegex(ValueError, "prepared_validation"),
            ):
                LIFECYCLE._verify_dataset_lineage()

    def test_extract_rejects_empty_and_special_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = root / "empty.tar"
            with tarfile.open(empty, "w"):
                pass
            with self.assertRaises(ValueError):
                LIFECYCLE._extract(empty, root / "empty-output")

            special = root / "special.tar"
            with tarfile.open(special, "w") as archive:
                member = tarfile.TarInfo("adapter/device")
                member.type = tarfile.CHRTYPE
                archive.addfile(member, io.BytesIO())
            with self.assertRaises(ValueError):
                LIFECYCLE._extract(special, root / "special-output")

    def test_train_isolates_output_and_archives_only_selected_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            (work / "train").mkdir(parents=True)
            work = work.resolve()
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}\n")
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            train.write_text("{}\n")
            validation.write_text("{}\n")

            def fake_run(command, *, environment=None, capture=False):
                self.assertEqual(command, ["bash", "audio8_tts_lora.sh"])
                self.assertFalse(capture)
                self.assertEqual(environment["EXPORT_DIR"], "")
                output = Path(environment["OUTPUT_DIR"])
                self.assertTrue(output.is_relative_to(work))
                selected = output / "checkpoint-20"
                selected.mkdir(parents=True)
                (selected / "adapter_config.json").write_text('{"r": 8}\n')
                (selected / "adapter_model.safetensors").write_bytes(b"adapter")
                (selected / "optimizer.pt").write_bytes(b"unsafe-pickle-placeholder")
                (output / "checkpoint-40").mkdir()
                return ""

            environment = {
                "INSTAVAR_VOICE_WORK_DIR": str(work),
                "BASE_MODEL_DIR": str(model),
                "TRAIN_JSONL": str(train),
                "EVAL_JSONL": str(validation),
                "SELECTED_ADAPTER_NAME": "checkpoint-20",
                "DTYPE": "float32",
            }
            with (
                patch.dict(os.environ, environment, clear=False),
                patch.object(LIFECYCLE, "_run", side_effect=fake_run),
                patch.object(LIFECYCLE, "_verify_dataset_lineage", return_value={"status": "passed"}),
            ):
                LIFECYCLE._train()
            with tarfile.open(work / "train" / "selected-adapter.tar", "r") as archive:
                names = set(archive.getnames())
            self.assertIn("adapter/adapter_model.safetensors", names)
            self.assertIn("adapter/adapter_config.json", names)
            self.assertNotIn("adapter/optimizer.pt", names)
            self.assertFalse(any("checkpoint-40" in name for name in names))


if __name__ == "__main__":
    unittest.main()
