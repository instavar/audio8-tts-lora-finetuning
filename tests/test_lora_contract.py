from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LoRAContractTests(unittest.TestCase):
    def test_trainer_exposes_lora_configuration(self) -> None:
        source = (ROOT / "audio8_tts_sft.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {
            node.target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        self.assertTrue(
            {"use_lora", "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules"}
            <= names
        )

    def test_inference_exposes_adapter_argument(self) -> None:
        source = (ROOT / "audio8_tts_infer.py").read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--adapter"', source)
        self.assertIn("PeftModel.from_pretrained", source)

    def test_portable_launcher_avoids_distributed_and_deepspeed(self) -> None:
        source = (ROOT / "audio8_tts_lora.sh").read_text(encoding="utf-8")
        self.assertNotIn("torch.distributed", source)
        self.assertNotIn("--deepspeed", source)
        self.assertIn("--use_lora true", source)

    def test_batch_inference_groups_frozen_seeds(self) -> None:
        source = (ROOT / "audio8_tts_infer.py").read_text(encoding="utf-8")
        self.assertIn("seed: int = 42", source)
        self.assertIn("a generation batch must use one frozen seed", source)
        self.assertIn("(item.reference_audio is not None, item.seed)", source)
        self.assertIn('"generation_seconds": elapsed', source)
        self.assertIn('"peak_memory_bytes": peak_memory_bytes', source)
        self.assertIn("torch.mps.synchronize()", source)

    def test_attempt_bound_suite_rejects_ambiguous_batch_timing(self) -> None:
        source = (ROOT / "scripts" / "run_evaluation_suite.py").read_text(encoding="utf-8")
        self.assertIn("attempt-bound evaluation requires --batch-size 1", source)
        self.assertNotIn('"peak_memory_bytes": record["peak_memory_bytes"],', source)

    def test_evaluation_suite_preserves_no_eos_as_invalid(self) -> None:
        source = (ROOT / "scripts" / "run_evaluation_suite.py").read_text(encoding="utf-8")
        self.assertIn('record["status"] == "OK"', source)
        self.assertIn("generation-observations.json", source)
        self.assertIn("allow-invalid-output", source)
        self.assertIn('not in {"1.0.0", "1.1.0"}', source)
        self.assertIn("generation_seconds", source)
        self.assertIn("artifact set id and sha256 must be provided together", source)
        self.assertIn('"runtime_id": runtime_id', source)
        self.assertIn('"artifact_set_sha256": args.artifact_set_sha256', source)
        self.assertIn('"observation_schema_version": "1.0.0"', source)

    def test_lifecycle_binds_runtime_attempt_evidence(self) -> None:
        source = (ROOT / "scripts" / "instavar_voice_lifecycle.py").read_text(encoding="utf-8")
        self.assertIn("build-generation-attempt-receipt", source)
        self.assertIn("apply-generation-attempt-receipt", source)
        self.assertIn("objective-observations.json", source)


if __name__ == "__main__":
    unittest.main()
