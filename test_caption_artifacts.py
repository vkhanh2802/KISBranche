import json
import tempfile
import unittest
from pathlib import Path

from artifact_io import atomic_write_json
from caption_schema import (
    load_frame_mapping,
    normalize_caption_record,
    rebuild_retrieval_text,
)
from migrate_caption_artifacts import migrate_triplet, plan_migration
from paths import (
    canonical_video_artifact_path,
    resolve_video_artifact,
    video_group,
)


class CaptionArtifactTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_canonical_path_and_legacy_resolution(self):
        artifact_root = self.root / "captions"
        legacy_path = artifact_root / "L26a" / "L26_V001.json"
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_text("[]", encoding="utf-8")

        self.assertEqual(video_group("L26_V001"), "L26")
        self.assertEqual(
            canonical_video_artifact_path(
                artifact_root,
                "L26_V001",
                ".json",
            ),
            artifact_root / "L26" / "L26_V001.json",
        )
        self.assertEqual(
            resolve_video_artifact(artifact_root, "L26_V001", ".json"),
            legacy_path.resolve(),
        )

    def test_duplicate_artifacts_are_rejected(self):
        artifact_root = self.root / "captions"
        for group in ("L26", "L26a"):
            path = artifact_root / group / "L26_V001.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("[]", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "Duplicate artifacts"):
            resolve_video_artifact(artifact_root, "L26_V001", ".json")

    def test_atomic_json_write_keeps_previous_file_on_failure(self):
        destination = self.root / "artifact.json"
        atomic_write_json(destination, {"state": "valid"})

        with self.assertRaises(TypeError):
            atomic_write_json(destination, {"invalid": object()})

        self.assertEqual(
            json.loads(destination.read_text(encoding="utf-8")),
            {"state": "valid"},
        )

    def test_normalizes_mapping_path_and_repairs_ocr_text(self):
        mapping_dir = self.root / "mapping"
        mapping_dir.mkdir()
        (mapping_dir / "L21_V001.csv").write_text(
            "n,frame_idx\n1,123\n",
            encoding="utf-8",
        )
        frame_mapping = load_frame_mapping("L21_V001", mapping_dir)
        record = {
            "video_id": "L21_V001",
            "frame_id": None,
            "keyframe_n": "1",
            "image_path": "data\\L21_V001\\001.jpg",
            "caption": "A sign",
            "details": "outside",
            "ocr_text": "OPEN",
            "retrieval_text": "stale",
        }

        normalized = normalize_caption_record(
            record,
            "L21_V001",
            frame_mapping,
        )

        self.assertEqual(normalized["frame_id"], 123)
        self.assertEqual(normalized["image_path"], "data/L21_V001/001.jpg")
        self.assertEqual(
            normalized["retrieval_text"],
            "A sign. outside. [OCR]: OPEN",
        )

    def test_empty_ocr_is_a_completed_result(self):
        record = {"caption": "A room", "details": "indoor", "ocr_text": ""}
        self.assertEqual(rebuild_retrieval_text(record), "A room. indoor")

    def test_migrates_complete_legacy_triplet(self):
        caption_root = self.root / "caption_generator"
        mapping_root = self.root / "caption_mapping"
        index_root = self.root / "index"
        specs = (
            ("caption", caption_root, ".json"),
            ("mapping", mapping_root, ".json"),
            ("index", index_root, ".index"),
        )
        for _, root, suffix in specs:
            path = root / "L26a" / f"L26_V001{suffix}"
            path.parent.mkdir(parents=True)
            path.write_bytes(f"content-{suffix}".encode())

        plan = plan_migration(specs)
        self.assertEqual(len(plan), 1)
        video_id, triplet = plan[0]
        migrate_triplet(video_id, triplet)

        for _, root, suffix in specs:
            canonical = root / "L26" / f"L26_V001{suffix}"
            self.assertTrue(canonical.is_file())
            self.assertFalse((root / "L26a" / f"L26_V001{suffix}").exists())


if __name__ == "__main__":
    unittest.main()
