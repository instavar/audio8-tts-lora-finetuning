from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "audio8_package_restore", ROOT / "scripts" / "verify_package_restore.py"
)
assert SPEC and SPEC.loader
RESTORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESTORE)


class PackageRestoreTests(unittest.TestCase):
    def test_restore_cli_requires_expected_package_identity(self) -> None:
        source = (ROOT / "scripts" / "verify_package_restore.py").read_text()
        self.assertIn('parser.add_argument("--expected-package-sha256", required=True)', source)
        self.assertIn("package does not match the expected SHA-256", source)

    def test_safe_extract_accepts_regular_members_and_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid.tar"
            with tarfile.open(valid, "w") as archive:
                member = tarfile.TarInfo("package/file.bin")
                member.size = 4
                archive.addfile(member, io.BytesIO(b"data"))
            extracted = RESTORE._safe_extract(valid, root / "valid-output", prefix="package")
            self.assertEqual((extracted / "file.bin").read_bytes(), b"data")

            unsafe = root / "unsafe.tar"
            with tarfile.open(unsafe, "w") as archive:
                member = tarfile.TarInfo("package/../escape.bin")
                member.size = 6
                archive.addfile(member, io.BytesIO(b"escape"))
            with self.assertRaisesRegex(ValueError, "unsafe archive member"):
                RESTORE._safe_extract(unsafe, root / "unsafe-output", prefix="package")

    def test_package_manifest_rejects_unlisted_files_and_byte_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "payload.bin"
            payload.write_bytes(b"payload")
            manifest = {
                "schema_version": "1.0.0",
                "files": [
                    {
                        "path": payload.name,
                        "bytes": payload.stat().st_size,
                        "sha256": RESTORE._sha256(payload),
                    }
                ],
            }
            (root / "package-manifest.json").write_text(json.dumps(manifest))
            RESTORE._verify_package_files(root)
            extra = root / "unlisted.bin"
            extra.write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "outside package-manifest"):
                RESTORE._verify_package_files(root)
            extra.unlink()
            payload.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                RESTORE._verify_package_files(root)

    def test_reference_inputs_are_bound_to_catalog_and_assignment_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "reference.wav"
            transcript = root / "reference.txt"
            audio.write_bytes(b"audio")
            transcript.write_bytes(b"transcript\n")
            generation_plan = {
                "schema_version": "1.1.0",
                "samples": [
                    {
                        "candidate_id": "candidate",
                        "prompt_id": "prompt",
                        "seed": 42,
                        "text": "Restore this row.",
                    }
                ],
            }
            catalog_payload = {
                "catalog_id": "catalog",
                "references": [
                    {
                        "reference_id": "speaker",
                        "audio": {
                            "bytes": audio.stat().st_size,
                            "sha256": RESTORE._sha256(audio),
                        },
                        "transcript": {
                            "bytes": transcript.stat().st_size,
                            "sha256": RESTORE._sha256(transcript),
                        },
                    }
                ],
            }
            catalog = {
                **catalog_payload,
                "catalog_sha256": RESTORE._canonical_sha256(catalog_payload),
            }
            assignment_payload = {
                "schema_version": "1.0.0",
                "plan_id": "plan",
                "generation_plan_sha256": RESTORE._canonical_sha256(generation_plan),
                "reference_catalog_sha256": catalog["catalog_sha256"],
                "reference_aggregation": "mean_cosine_similarity_v1",
                "selection_policy": {},
                "assignments": [],
            }
            assignment = {
                **assignment_payload,
                "assignment_plan_sha256": RESTORE._canonical_sha256(assignment_payload),
            }
            (root / "speaker-reference-catalog.json").write_text(json.dumps(catalog))
            (root / "speaker-reference-plan.json").write_text(json.dumps(assignment))
            RESTORE._verify_reference_inputs(
                root, generation_plan, audio, transcript, "speaker"
            )
            audio.write_bytes(b"other")
            with self.assertRaisesRegex(ValueError, "reference audio"):
                RESTORE._verify_reference_inputs(
                    root, generation_plan, audio, transcript, "speaker"
                )


if __name__ == "__main__":
    unittest.main()
