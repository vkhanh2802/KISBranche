import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from artifact_io import atomic_write_json
from object_schema import OBJECT_FIELDS, provider_object_record, validate_object_schema
from prepare_video_data import OBJECT_MODEL, KeyframeConfig, VideoDatasetBuilder, discover_videos


class ProviderObjectSchemaTest(unittest.TestCase):
    def test_exports_the_same_fields_and_string_types_as_provider(self):
        detection = {"detection_class_entities": ["boat", "person"], "detection_scores": [0.9, 0.8],
                     "detection_boxes": [[0.1, 0.2, 0.8, 0.9], [0.0, 0.1, 0.9, 1.0]]}
        record = provider_object_record(detection)
        self.assertEqual(tuple(record), OBJECT_FIELDS)
        self.assertEqual(record["detection_class_names"], ["/m/019jd", "/m/01g317"])
        self.assertEqual(record["detection_class_labels"], ["43", "69"])
        self.assertEqual(record["detection_class_entities"], ["Boat", "Person"])
        self.assertEqual(record["detection_boxes"][0], ["0.1", "0.2", "0.8", "0.9"])
        self.assertIsInstance(record["detection_scores"][0], str)
        validate_object_schema(record)

    def test_accepts_original_open_images_schema(self):
        validate_object_schema({
            "detection_scores": ["0.758653"], "detection_class_names": ["/m/0dzct"],
            "detection_class_entities": ["Human face"],
            "detection_boxes": [["0.29001886", "0.572104", "0.5024236", "0.6660007"]],
            "detection_class_labels": ["502"],
        })

    def test_does_not_invent_mid_for_unmapped_coco_class(self):
        record = provider_object_record({"detection_class_entities": ["cup"], "detection_scores": [0.8],
                                         "detection_boxes": [[0.1, 0.2, 0.8, 0.9]]})
        self.assertTrue(all(values == [] for values in record.values()))
        validate_object_schema(record)

    def test_schema_rejects_metadata_float_scores_and_invalid_boxes(self):
        record = provider_object_record({"detection_class_entities": ["boat"], "detection_scores": [0.8],
                                         "detection_boxes": [[0.1, 0.2, 0.8, 0.9]]})
        for changed in ({**record, "video_id": "L20_V001"}, {**record, "detection_scores": [0.8]},
                        {**record, "detection_boxes": [["0.8", "0.2", "0.1", "0.9"]]}):
            with self.assertRaises(ValueError):
                validate_object_schema(changed)

    def test_existing_object_filter_reads_exported_strings(self):
        from apply_object_filter import apply_object_filter

        record = provider_object_record({"detection_class_entities": ["boat"], "detection_scores": [0.9],
                                         "detection_boxes": [[0.1, 0.2, 0.8, 0.9]]})

        class Lookup:
            def get(self, _key):
                return record

        result = apply_object_filter([{"video_id": "L20_V001", "frame_id": 0, "score": 0.5}],
                                     [{"type": "object", "value": "boat"}], Lookup())
        self.assertGreater(result[0]["score"], 0.5)


class VideoDiscoveryTest(unittest.TestCase):
    def test_rejects_duplicate_video_ids_and_invalid_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "L20_V001.mp4").touch()
            nested = root / "L20"
            nested.mkdir()
            (nested / "L20_V001.mkv").touch()
            with self.assertRaisesRegex(ValueError, "Duplicate video_id"):
                discover_videos(root)
            (nested / "L20_V001.mkv").unlink()
            (root / "arbitrary-name.mp4").touch()
            with self.assertRaisesRegex(ValueError, "Invalid video_id"):
                discover_videos(root)

    def test_selected_video_must_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "L20_V001.mp4").touch()
            with self.assertRaisesRegex(FileNotFoundError, "Missing videos"):
                discover_videos(directory, ["L20_V002"])

    def test_rejects_invalid_keyframe_settings(self):
        for settings in ({"interval_seconds": 0}, {"scene_threshold": 2},
                         {"max_duration_seconds": -1}, {"jpeg_quality": 101},
                         {"frame_gap_min": 0}, {"frame_gap_max": 79},
                         {"frame_gap_min": 80.5}, {"frame_gap_min": True},
                         {"random_seed": "42"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                KeyframeConfig(**settings)


@unittest.skipUnless(importlib.util.find_spec("av"), "PyAV is required for video ingestion tests")
class VideoIngestionTest(unittest.TestCase):
    def setUp(self):
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_dir.name)
        self.video_path = self.root / "video" / "L20_V001.mp4"
        self.video_path.parent.mkdir()
        self._write_video()
        self.builder = VideoDatasetBuilder(self.root)
        self.config = KeyframeConfig(frame_gap_min=None, scene_check_seconds=5.0)

    def _write_video(self, frame_count=30, fps=10):
        import av
        import numpy as np

        with av.open(str(self.video_path), "w") as container:
            stream = container.add_stream("mpeg4", rate=fps)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            for frame_idx in range(frame_count):
                pixels = np.full((48, 64, 3), 0 if frame_idx < 10 else 255, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    def tearDown(self):
        self.temporary_dir.cleanup()

    def test_mapping_uses_real_decoded_indices_and_timestamps(self):
        rows = self.builder.extract_keyframes(self.video_path, self.config)
        self.assertEqual([row["frame_idx"] for row in rows], [0, 20])
        self.assertEqual([row["pts_time"] for row in rows], [0.0, 2.0])
        self.assertEqual([row["n"] for row in rows], [1, 2])
        self.assertEqual(self.builder.mapping_rows("L20_V001"), rows)

    def test_scene_transition_is_kept_between_periodic_samples(self):
        rows = self.builder.extract_keyframes(self.video_path, KeyframeConfig(frame_gap_min=None, interval_seconds=10))
        self.assertEqual([row["frame_idx"] for row in rows], [0, 10])

    def test_resume_keeps_numbering_and_rejects_configuration_changes(self):
        rows = self.builder.extract_keyframes(self.video_path, self.config)
        self.assertEqual(self.builder.extract_keyframes(self.video_path, self.config), rows)
        with self.assertRaisesRegex(ValueError, "Source/config changed"):
            self.builder.extract_keyframes(self.video_path, KeyframeConfig(frame_gap_min=None, interval_seconds=5))
        self.assertEqual(self.builder.mapping_rows("L20_V001"), rows)

    def test_random_gaps_stay_in_range_and_ignore_scene_cuts(self):
        self._write_video(frame_count=1000, fps=30)
        rows = self.builder.extract_keyframes(self.video_path)
        frame_ids = [row["frame_idx"] for row in rows]
        gaps = [current - previous for previous, current in zip(frame_ids, frame_ids[1:])]
        self.assertEqual(frame_ids[0], 0)
        self.assertTrue(all(80 <= gap <= 160 for gap in gaps))
        self.assertGreater(len(set(gaps)), 1)
        self.assertLess(999 - frame_ids[-1], 160)
        self.assertNotIn(10, frame_ids)  # black/white scene cut must not add a sample
        for row in rows:
            self.assertAlmostEqual(row["pts_time"], row["frame_idx"] / 30)
        self.assertEqual(self.builder.mapping_rows("L20_V001"), rows)

    def test_random_seed_is_reproducible_and_changes_the_schedule(self):
        self._write_video(frame_count=500)
        schedules = []
        for name, seed in (("first", 42), ("repeat", 42), ("changed", 43)):
            builder = VideoDatasetBuilder(self.root / name)
            rows = builder.extract_keyframes(self.video_path, KeyframeConfig(random_seed=seed))
            schedules.append([row["frame_idx"] for row in rows])
        self.assertEqual(schedules[0], schedules[1])
        self.assertNotEqual(schedules[0], schedules[2])

    def test_frame_sampling_is_independent_of_fps(self):
        schedules = []
        for fps in (25, 30):
            self._write_video(frame_count=500, fps=fps)
            builder = VideoDatasetBuilder(self.root / f"fps_{fps}")
            rows = builder.extract_keyframes(self.video_path)
            schedules.append([row["frame_idx"] for row in rows])
            for row in rows:
                self.assertAlmostEqual(row["pts_time"], row["frame_idx"] / fps)
        self.assertEqual(schedules[0], schedules[1])

    def test_short_video_keeps_first_frame(self):
        rows = self.builder.extract_keyframes(self.video_path)
        self.assertEqual([row["frame_idx"] for row in rows], [0])

    def test_existing_dataset_is_not_replaced(self):
        image_dir = self.root / "data" / "L20_V001"
        image_dir.mkdir(parents=True)
        (image_dir / "001.jpg").write_bytes(b"existing user artifact")
        with self.assertRaisesRegex(FileExistsError, "Existing artifacts"):
            self.builder.extract_keyframes(self.video_path)
        self.assertEqual((image_dir / "001.jpg").read_bytes(), b"existing user artifact")

    def test_publish_failure_does_not_leave_misaligned_images(self):
        with patch("prepare_video_data.publish_staged_files", side_effect=OSError("publish failed")):
            with self.assertRaisesRegex(OSError, "publish failed"):
                self.builder.extract_keyframes(self.video_path)
        paths = self.builder.artifact_paths("L20_V001")
        self.assertFalse(paths["images"].exists())
        self.assertFalse(paths["mapping"].exists())

    def test_changed_keyframe_is_detected_before_inference(self):
        self.builder.extract_keyframes(self.video_path, self.config)
        image_path = self.builder.artifact_paths("L20_V001")["images"] / "001.jpg"
        image_path.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Keyframe was changed"):
            self.builder.mapping_rows("L20_V001")

    @unittest.skipUnless(importlib.util.find_spec("faiss"), "FAISS is required for the artifact integration test")
    def test_features_objects_and_caption_index_share_frame_identity(self):
        import numpy as np
        from build_caption_index import write_caption_artifacts

        rows = self.builder.extract_keyframes(self.video_path, self.config)

        class Encoder:
            calls = 0

            def encode_images(self, images):
                self.calls += 1
                vectors = np.zeros((len(images), 768), dtype=np.float32)
                for position, image in enumerate(images):
                    vectors[position, int(np.asarray(image).mean() > 128)] = 2
                return vectors

        class Detector:
            def detect(self, images, threshold):
                return [{"detection_class_entities": ["boat"], "detection_class_names": ["9"],
                         "detection_scores": [0.9], "detection_boxes": [[0.1, 0.2, 0.8, 0.9]]}
                        for _ in images]

        encoder = Encoder()
        self.builder.encode_visual_features(["L20_V001"], encoder=encoder)
        self.builder.encode_visual_features(["L20_V001"], encoder=encoder)
        self.assertEqual(encoder.calls, 1)
        self.builder.generate_objects(["L20_V001"], detector=Detector())
        paths = self.builder.artifact_paths("L20_V001")
        features = np.load(paths["features"])
        self.assertEqual(features.shape, (2, 768))
        self.assertEqual(np.argmax(features, axis=1).tolist(), [0, 1])
        objects = json.loads((paths["objects"] / "002.json").read_text())
        self.assertEqual(set(objects), set(OBJECT_FIELDS))
        self.assertEqual(objects["detection_class_labels"], ["43"])
        state = self.builder._state("L20_V001")
        self.assertEqual(state["object_records"]["002.json"]["frame_id"], rows[1]["frame_idx"])
        self.assertNotIn("\n", (paths["objects"] / "002.json").read_text())

        captions = [
            {"video_id": "L20_V001", "keyframe_n": row["n"], "frame_id": row["frame_idx"],
             "image_path": f"data/L20/L20_V001/{row['n']:03d}.jpg", "caption": "A boat", "retrieval_text": "A boat",
             "details": "", "vqa_answers": {field: "unknown" for field in (
                 "people", "clothing_color", "main_object", "object_color", "action", "setting", "location")}}
            for row in rows
        ]
        embeddings = np.eye(2, 384, dtype=np.float32)
        write_caption_artifacts("L20_V001", captions, embeddings, project_root=self.root)
        report = self.builder.validate_video("L20_V001")
        self.assertEqual(report["status"], "OK")
        self.assertEqual(report["caption_vectors"], 2)

        from audit_artifact_formats import audit_video_formats

        formats = audit_video_formats(self.root, "L20_V001")
        self.assertEqual(len(formats), 7)
        self.assertTrue(all(item["status"] == "OK" for item in formats), formats)

        (paths["caption_mapping"]).unlink()
        partial = audit_video_formats(self.root, "L20_V001", allow_incomplete=True)
        self.assertEqual(next(item for item in partial if item["folder"] == "caption_mapping")["status"], "PENDING")

        features[0, 0] = 0.5
        np.save(paths["features"], features)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.builder.validate_video("L20_V001")

    def test_migrates_own_legacy_objects_without_inference(self):
        rows = self.builder.extract_keyframes(self.video_path, self.config)
        paths = self.builder.artifact_paths("L20_V001")
        state = self.builder._state("L20_V001")
        producer = {**OBJECT_MODEL, "score_threshold": 0.2}
        for row in rows:
            n = row["n"]
            atomic_write_json(paths["objects"] / f"{n:03d}.json", {
                "video_id": "L20_V001", "keyframe_n": n, "frame_id": row["frame_idx"],
                "image_sha256": state["keyframes"]["images"][f"{n:03d}.jpg"], "detector": producer,
                "detection_class_entities": ["boat"], "detection_class_names": ["9"],
                "detection_scores": [0.9], "detection_boxes": [[0.1, 0.2, 0.8, 0.9]],
                "box_format": "ymin,xmin,ymax,xmax (normalized)",
            })
        state["objects"] = producer
        self.builder._save_state("L20_V001", state)
        with patch("prepare_video_data.CPUObjectDetector") as detector:
            self.builder.migrate_generated_objects(["L20_V001"])
            self.builder.migrate_generated_objects(["L20_V001"])
            detector.assert_not_called()
        raw = json.loads((paths["objects"] / "002.json").read_text())
        self.assertEqual(raw["detection_scores"], ["0.9"])
        self.assertEqual(raw["detection_class_names"], ["/m/019jd"])
        self.assertEqual(raw["detection_class_labels"], ["43"])
        validate_object_schema(raw)
        self.assertEqual(self.builder._state("L20_V001")["object_records"]["002.json"]["frame_id"], rows[1]["frame_idx"])


class PreparationNotebookTest(unittest.TestCase):
    def test_cpu_notebook_cells_compile(self):
        notebook_path = Path(__file__).with_name("prepare_videos_cpu.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        for cell in code_cells:
            compile("".join(cell["source"]), cell["id"], "exec")
        self.assertGreaterEqual(len(code_cells), 7)


if __name__ == "__main__":
    unittest.main()
