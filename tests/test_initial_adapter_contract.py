from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class InitialAdapterContractTests(unittest.TestCase):
    def test_trainer_seeds_before_loading_and_accepts_one_bound_adapter(self) -> None:
        source = (ROOT / "audio8_tts_sft.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {
            node.target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        self.assertIn("initial_adapter_dir", names)
        self.assertLess(
            source.index("set_seed(training_args.seed)"),
            source.index("AutoModel.from_pretrained"),
        )
        self.assertIn("PeftModel.from_pretrained", source)
        self.assertIn('"initial_adapter": initial_adapter_files', source)

    def test_launcher_forwards_initial_adapter(self) -> None:
        source = (ROOT / "audio8_tts_lora.sh").read_text(encoding="utf-8")
        self.assertIn('--initial_adapter_dir "${INITIAL_ADAPTER_DIR:-}"', source)

    def test_initializer_is_atomic_and_deterministic(self) -> None:
        source = (ROOT / "scripts" / "create_initial_lora.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index("set_seed(args.seed)"), source.index("AutoModel.from_pretrained")
        )
        self.assertIn("safe_serialization=True", source)
        self.assertIn("temporary.mkdir(exist_ok=False)", source)
        self.assertIn("os.replace(temporary, output)", source)
        self.assertIn("fsync_directory(parent)", source)
        self.assertIn("initial-adapter-receipt.json", source)


if __name__ == "__main__":
    unittest.main()
