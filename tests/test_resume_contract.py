from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audio8_tts_resume import (
    LOCK_NAME,
    ResumeContractError,
    acquire_output_lock,
    assert_save_destination_absent,
    build_contract,
    checkpoint_manifest,
    model_identity,
    prune_owned_checkpoints,
    require_fresh_output,
    resolve_checkpoint,
    resolve_resume_request,
    validate_resume_checkpoint,
    write_checkpoint_sidecar,
)


class ResumeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.base = self.root / "base"
        self.base.mkdir()
        (self.base / "model.safetensors").write_bytes(b"base")
        self.manifest = self.root / "train.jsonl"
        self.manifest.write_text('{"id":"sample"}\n', encoding="utf-8")
        self.codes = self.root / "codes.npy"
        self.codes.write_bytes(b"codes")
        self.source = self.root / "trainer.py"
        self.source.write_text("print('fixture')\n", encoding="utf-8")
        self.contract = self._contract()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _contract(self, *, max_steps: int = 10) -> dict:
        return build_contract(
            output_dir=self.output,
            mode="lora",
            base_model=self.base,
            base_revision=None,
            input_files={"train": [self.manifest, self.codes], "evaluation": []},
            source_files=[self.source],
            training_config={"max_steps": max_steps, "seed": 42},
            runtime={"python": "fixture", "world_size": 1},
        )

    def _checkpoint(
        self,
        step: int,
        *,
        contract: dict | None = None,
        sidecar: bool = True,
    ) -> Path:
        checkpoint = self.output / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "adapter_model.safetensors").write_bytes(f"adapter-{step}".encode())
        (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (checkpoint / "optimizer.pt").write_bytes(f"optimizer-{step}".encode())
        (checkpoint / "scheduler.pt").write_bytes(f"scheduler-{step}".encode())
        (checkpoint / "rng_state.pth").write_bytes(f"rng-{step}".encode())
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": step}) + "\n",
            encoding="utf-8",
        )
        if sidecar:
            write_checkpoint_sidecar(
                checkpoint,
                output_dir=self.output,
                contract=contract or self.contract,
            )
        return checkpoint

    def test_exact_trusted_checkpoint_validates(self) -> None:
        checkpoint = self._checkpoint(2)
        resolved = validate_resume_checkpoint(
            checkpoint,
            output_dir=self.output,
            expected_contract=self.contract,
            trust_resume_state=True,
            world_size=1,
        )
        self.assertEqual(resolved, checkpoint.resolve())

    def test_resume_requires_explicit_trust(self) -> None:
        checkpoint = self._checkpoint(2)
        with self.assertRaisesRegex(ResumeContractError, "pickle-capable"):
            validate_resume_checkpoint(
                checkpoint,
                output_dir=self.output,
                expected_contract=self.contract,
                trust_resume_state=False,
                world_size=1,
            )

    def test_implicit_latest_and_conflicting_flags_are_rejected(self) -> None:
        for value in ("auto", "latest", "true", "yes"):
            with self.assertRaisesRegex(ResumeContractError, "Implicit latest"):
                resolve_resume_request(value, "none")
        with self.assertRaisesRegex(ResumeContractError, "only --resume_from"):
            resolve_resume_request("checkpoint-1", "checkpoint-2")

    def test_exact_legacy_path_can_enter_the_guarded_contract(self) -> None:
        self.assertEqual(resolve_resume_request(None, "/tmp/checkpoint-2"), "/tmp/checkpoint-2")
        self.assertIsNone(resolve_resume_request("", "none"))

    def test_checkpoint_must_be_a_direct_numeric_child(self) -> None:
        checkpoint = self._checkpoint(2)
        nested = self.output / "nested"
        nested.mkdir()
        moved = nested / checkpoint.name
        checkpoint.rename(moved)
        with self.assertRaisesRegex(ResumeContractError, "direct child"):
            resolve_checkpoint(moved, self.output)

    def test_checkpoint_symlink_is_rejected(self) -> None:
        checkpoint = self._checkpoint(2)
        link = self.output / "checkpoint-3"
        link.symlink_to(checkpoint, target_is_directory=True)
        with self.assertRaisesRegex(ResumeContractError, "symlinks"):
            resolve_checkpoint(link, self.output)

    def test_checkpoint_byte_drift_is_rejected(self) -> None:
        checkpoint = self._checkpoint(2)
        (checkpoint / "optimizer.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ResumeContractError, "file identity drift"):
            validate_resume_checkpoint(
                checkpoint,
                output_dir=self.output,
                expected_contract=self.contract,
                trust_resume_state=True,
                world_size=1,
            )

    def test_dataset_drift_changes_the_run_contract(self) -> None:
        checkpoint = self._checkpoint(2)
        self.codes.write_bytes(b"changed")
        changed = self._contract()
        with self.assertRaisesRegex(ResumeContractError, "contract drift"):
            validate_resume_checkpoint(
                checkpoint,
                output_dir=self.output,
                expected_contract=changed,
                trust_resume_state=True,
                world_size=1,
            )

    def test_completed_target_and_distributed_resume_are_rejected(self) -> None:
        checkpoint = self._checkpoint(2, contract=self._contract(max_steps=2))
        with self.assertRaisesRegex(ResumeContractError, "reached"):
            validate_resume_checkpoint(
                checkpoint,
                output_dir=self.output,
                expected_contract=self._contract(max_steps=2),
                trust_resume_state=True,
                world_size=1,
            )
        with self.assertRaisesRegex(ResumeContractError, "world_size=1"):
            validate_resume_checkpoint(
                checkpoint,
                output_dir=self.output,
                expected_contract=self._contract(max_steps=2),
                trust_resume_state=True,
                world_size=2,
            )

    def test_resume_requires_the_newest_owned_checkpoint(self) -> None:
        older = self._checkpoint(1)
        self._checkpoint(2)
        with self.assertRaisesRegex(ResumeContractError, "newest owned"):
            validate_resume_checkpoint(
                older,
                output_dir=self.output,
                expected_contract=self.contract,
                trust_resume_state=True,
                world_size=1,
            )

    def test_resume_rejects_an_unowned_sibling_before_training(self) -> None:
        self._checkpoint(1, sidecar=False)
        selected = self._checkpoint(2)
        with self.assertRaisesRegex(ResumeContractError, "no safe"):
            validate_resume_checkpoint(
                selected,
                output_dir=self.output,
                expected_contract=self.contract,
                trust_resume_state=True,
                world_size=1,
            )

    def test_sidecar_is_last_marker_and_cannot_be_overwritten(self) -> None:
        checkpoint = self._checkpoint(2)
        with self.assertRaisesRegex(ResumeContractError, "overwrite"):
            write_checkpoint_sidecar(
                checkpoint,
                output_dir=self.output,
                contract=self.contract,
            )
        self.assertFalse(any(path.name.endswith(".partial") for path in checkpoint.iterdir()))

    def test_incomplete_checkpoint_cannot_receive_a_sidecar(self) -> None:
        checkpoint = self.output / "checkpoint-2"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text('{"global_step":2}\n')
        with self.assertRaisesRegex(ResumeContractError, "continuation files"):
            write_checkpoint_sidecar(
                checkpoint,
                output_dir=self.output,
                contract=self.contract,
            )

    def test_sharded_full_model_checkpoint_can_receive_a_sidecar(self) -> None:
        checkpoint = self.output / "checkpoint-2"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors.index.json").write_text("{}\n")
        (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"first")
        (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"second")
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
        (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
        (checkpoint / "rng_state.pth").write_bytes(b"rng")
        (checkpoint / "trainer_state.json").write_text('{"global_step":2}\n')
        sidecar = write_checkpoint_sidecar(
            checkpoint,
            output_dir=self.output,
            contract=self.contract,
        )
        self.assertTrue(sidecar.is_file())

    def test_symlinked_checkpoint_member_is_rejected(self) -> None:
        checkpoint = self.output / "checkpoint-2"
        checkpoint.mkdir()
        target = self.root / "outside.bin"
        target.write_bytes(b"outside")
        (checkpoint / "optimizer.pt").symlink_to(target)
        with self.assertRaisesRegex(ResumeContractError, "reject symlinks"):
            checkpoint_manifest(checkpoint)

    def test_fresh_output_allows_only_the_owned_lock(self) -> None:
        (self.output / LOCK_NAME).write_text("pid=1\n")
        require_fresh_output(self.output)
        (self.output / "adapter_model.safetensors").write_bytes(b"old")
        with self.assertRaisesRegex(ResumeContractError, "empty output"):
            require_fresh_output(self.output)

    def test_save_destination_must_not_exist(self) -> None:
        assert_save_destination_absent(self.output, 2)
        self._checkpoint(2)
        with self.assertRaisesRegex(ResumeContractError, "overwrite or adopt"):
            assert_save_destination_absent(self.output, 2)

    def test_retention_deletes_only_owned_numeric_children(self) -> None:
        first = self._checkpoint(1)
        self._checkpoint(2)
        self._checkpoint(3)
        (self.output / "checkpoint-not-a-number").mkdir()
        victims = prune_owned_checkpoints(
            self.output,
            keep_last=2,
            expected_contract=self.contract,
            best_checkpoint=None,
        )
        self.assertEqual(victims, [first.resolve()])
        self.assertFalse(first.exists())
        self.assertTrue((self.output / "checkpoint-not-a-number").is_dir())

    def test_retention_fails_closed_on_sidecarless_numeric_child(self) -> None:
        self._checkpoint(1, sidecar=False)
        self._checkpoint(2)
        with self.assertRaisesRegex(ResumeContractError, "no safe"):
            prune_owned_checkpoints(
                self.output,
                keep_last=1,
                expected_contract=self.contract,
                best_checkpoint=None,
            )
        self.assertTrue((self.output / "checkpoint-1").is_dir())

    def test_retention_preserves_distinct_best_and_latest_at_limit_one(self) -> None:
        best = self._checkpoint(1)
        middle = self._checkpoint(2)
        latest = self._checkpoint(3)
        victims = prune_owned_checkpoints(
            self.output,
            keep_last=1,
            expected_contract=self.contract,
            best_checkpoint=str(best),
        )
        self.assertEqual(victims, [middle.resolve()])
        self.assertTrue(best.is_dir())
        self.assertTrue(latest.is_dir())

    def test_output_directory_lock_rejects_a_second_writer(self) -> None:
        first = acquire_output_lock(self.output)
        try:
            with self.assertRaisesRegex(ResumeContractError, "Another guarded writer"):
                acquire_output_lock(self.output)
        finally:
            first.close()

    def test_output_lock_rejects_a_hardlinked_file_before_truncation(self) -> None:
        protected = self.root / "protected.txt"
        protected.write_text("keep me\n", encoding="utf-8")
        (self.output / LOCK_NAME).hardlink_to(protected)
        with self.assertRaisesRegex(ResumeContractError, "unsafe ownership or link count"):
            acquire_output_lock(self.output)
        self.assertEqual(protected.read_text(encoding="utf-8"), "keep me\n")

    def test_local_model_tree_rejects_symlinks(self) -> None:
        target = self.root / "outside-model.bin"
        target.write_bytes(b"outside")
        (self.base / "linked.bin").symlink_to(target)
        with self.assertRaisesRegex(ResumeContractError, "rejects symlinks"):
            model_identity(self.base, None)

    def test_remote_model_requires_an_immutable_revision(self) -> None:
        with self.assertRaisesRegex(ResumeContractError, "immutable resolved commit"):
            model_identity("org/model", None)
        identity = model_identity("org/model", "a" * 40)
        self.assertEqual(identity["revision"], "a" * 40)


if __name__ == "__main__":
    unittest.main()
