import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import artifact_io
import migrate_caption_artifacts
from artifact_io import atomic_write_json, publish_staged_files
from caption_artifact_manifest import manifest_data, validate_caption_manifest
from caption_schema import (
    load_frame_mapping,
    normalize_caption_record,
    rebuild_retrieval_text,
    validate_unique_caption_records,
)
from migrate_caption_artifacts import migrate_triplet, plan_migration
from paths import (
    canonical_video_artifact_path,
    discover_video_image_dirs,
    resolve_video_artifact,
    resolve_video_image_dir,
    safe_filename_component,
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

    def test_staged_publish_restores_previous_file_on_failure(self):
        destination = self.root / "artifact.json"
        staged = self.root / "staged.json"
        destination.write_text("old", encoding="utf-8")
        staged.write_text("new", encoding="utf-8")
        real_replace = artifact_io.os.replace

        def fail_publish(source, target):
            if Path(source) == staged:
                raise OSError("simulated publish failure")
            return real_replace(source, target)

        with patch("artifact_io.os.replace", side_effect=fail_publish):
            with self.assertRaisesRegex(OSError, "simulated"):
                publish_staged_files([(destination, staged)])

        self.assertEqual(destination.read_text(encoding="utf-8"), "old")

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

    def test_existing_frame_id_must_match_mapping(self):
        record = {
            "video_id": "L21_V001",
            "frame_id": 999,
            "keyframe_n": 1,
            "image_path": "data/L21_V001/001.jpg",
        }
        with self.assertRaisesRegex(ValueError, "frame_id mismatch"):
            normalize_caption_record(record, "L21_V001", {1: 123})

    def test_duplicate_caption_identities_are_rejected(self):
        records = [
            {"keyframe_n": 1, "frame_id": 10},
            {"keyframe_n": 1, "frame_id": 11},
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate keyframe_n"):
            validate_unique_caption_records(records, "L21_V001")

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

    def test_failed_migration_removes_partial_destinations(self):
        specs = tuple(
            (kind, self.root / kind, suffix)
            for kind, suffix in (
                ("caption", ".json"),
                ("mapping", ".json"),
                ("index", ".index"),
            )
        )
        for _, root, suffix in specs:
            source = root / "L26a" / f"L26_V001{suffix}"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"valid")

        _, triplet = plan_migration(specs)[0]
        real_replace = migrate_caption_artifacts.os.replace
        calls = 0

        def fail_second_publish(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated migration failure")
            return real_replace(source, target)

        with patch(
            "migrate_caption_artifacts.os.replace",
            side_effect=fail_second_publish,
        ):
            with self.assertRaisesRegex(OSError, "simulated"):
                migrate_triplet("L26_V001", triplet)

        for _, source, destination in triplet:
            self.assertTrue(source.is_file())
            self.assertFalse(destination.exists())

    def test_migration_recovers_identical_canonical_duplicate(self):
        specs = tuple(
            (kind, self.root / kind, suffix)
            for kind, suffix in (
                ("caption", ".json"),
                ("mapping", ".json"),
                ("index", ".index"),
            )
        )
        for _, root, suffix in specs:
            legacy = root / "L26a" / f"L26_V001{suffix}"
            canonical = root / "L26" / f"L26_V001{suffix}"
            legacy.parent.mkdir(parents=True)
            canonical.parent.mkdir(parents=True)
            legacy.write_bytes(b"same")
            canonical.write_bytes(b"same")

        video_id, triplet = plan_migration(specs)[0]
        migrate_triplet(video_id, triplet)

        for _, root, suffix in specs:
            self.assertTrue((root / "L26" / f"L26_V001{suffix}").is_file())
            self.assertFalse((root / "L26a" / f"L26_V001{suffix}").exists())

    def test_cleanup_failure_keeps_published_destinations(self):
        specs = tuple(
            (kind, self.root / kind, suffix)
            for kind, suffix in (
                ("caption", ".json"),
                ("mapping", ".json"),
                ("index", ".index"),
            )
        )
        for _, root, suffix in specs:
            source = root / "L26a" / f"L26_V001{suffix}"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"valid")

        video_id, triplet = plan_migration(specs)[0]
        failed_source = triplet[0][1]
        real_unlink = Path.unlink

        def fail_source_cleanup(path, *args, **kwargs):
            if path == failed_source:
                raise OSError("simulated cleanup failure")
            return real_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", fail_source_cleanup):
            with self.assertRaisesRegex(OSError, "simulated"):
                migrate_triplet(video_id, triplet)

        for _, source, destination in triplet:
            self.assertTrue(destination.is_file())
        self.assertTrue(failed_source.is_file())

    def test_manifest_detects_tampered_caption(self):
        caption_path = self.root / "caption.json"
        mapping_path = self.root / "mapping.json"
        index_path = self.root / "caption.index"
        manifest_path = self.root / "caption.meta.json"
        caption_path.write_text("[]", encoding="utf-8")
        mapping_path.write_text("[]", encoding="utf-8")
        index_path.write_bytes(b"index")
        manifest = manifest_data(
            "L21_V001",
            caption_path,
            mapping_path,
            index_path,
            0,
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        index = SimpleNamespace(d=384)

        validate_caption_manifest(
            "L21_V001",
            caption_path,
            mapping_path,
            index_path,
            manifest_path,
            index,
            0,
        )
        caption_path.write_text("[{}]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            validate_caption_manifest(
                "L21_V001",
                caption_path,
                mapping_path,
                index_path,
                manifest_path,
                index,
                0,
            )

    def test_filename_component_is_sanitized(self):
        self.assertEqual(
            safe_filename_component("../q:01?"),
            "q_01",
        )

    def test_grouped_and_flat_image_layouts_are_resolved(self):
        data_root = self.root / "data"
        flat_video = data_root / "L21_V001"
        flat_video.mkdir(parents=True)
        (flat_video / "001.jpg").write_bytes(b"flat")
        grouped_video = data_root / "L27" / "L27_V001"
        grouped_video.mkdir(parents=True)
        (grouped_video / "001.jpg").write_bytes(b"grouped")

        self.assertEqual(
            resolve_video_image_dir("L21_V001", image_root=data_root),
            flat_video.resolve(),
        )
        self.assertEqual(
            resolve_video_image_dir("L27_V001", image_root=data_root),
            grouped_video.resolve(),
        )
        discovered = discover_video_image_dirs(data_root)
        self.assertEqual(
            discovered,
            {
                "L21_V001": flat_video,
                "L27_V001": grouped_video,
            },
        )

    def test_duplicate_image_layouts_are_rejected(self):
        data_root = self.root / "data"
        flat_video = data_root / "L21_V001"
        flat_video.mkdir(parents=True)
        grouped_video = data_root / "L21" / "L21_V001"
        grouped_video.mkdir(parents=True)

        with self.assertRaisesRegex(RuntimeError, "Duplicate image"):
            resolve_video_image_dir("L21_V001", image_root=data_root)
        with self.assertRaisesRegex(RuntimeError, "Duplicate image"):
            discover_video_image_dirs(data_root)


if __name__ == "__main__":
    unittest.main()
