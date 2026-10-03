import math
import unittest
from unittest.mock import Mock

from apply_object_filter import apply_object_filter
from model_config import milvus_model_properties
from search_clip import validate_collection_model
from solve_kis import (
    console_safe_text,
    finalize_vlm_ranking,
    max_score_fusion,
    normalize_scores,
    static_weight_fusion_3_way,
)


class ObjectLookup:
    def __init__(self, confidence):
        self.confidence = confidence

    def get(self, key):
        return {
            "detection_class_entities": ["person"],
            "detection_scores": [self.confidence],
        }


class RankingValidationTest(unittest.TestCase):
    def test_non_finite_object_confidence_is_ignored(self):
        candidates = [{"video_id": "V1", "frame_id": 1, "score": 0.5}]
        entities = [{"type": "object", "value": "person"}]

        results = apply_object_filter(
            candidates,
            entities,
            ObjectLookup(float("nan")),
        )

        self.assertTrue(math.isfinite(results[0]["score"]))
        self.assertEqual(results[0]["object_score"], 0.0)

    def test_object_confidence_is_capped(self):
        candidates = [{"video_id": "V1", "frame_id": 1, "score": 0.5}]
        entities = [{"type": "object", "value": "person"}]

        results = apply_object_filter(candidates, entities, ObjectLookup(5.0))

        self.assertEqual(results[0]["object_score"], 1.0)
        self.assertAlmostEqual(results[0]["score"], 0.575)

    def test_normalization_rejects_non_finite_scores(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            normalize_scores([{"score": float("inf")}])

    def test_fusion_weights_must_sum_to_one(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            static_weight_fusion_3_way([], [], [], 0.5, 0.5, 0.5)

    def test_identical_vlm_scores_still_apply_bounded_boost(self):
        candidates = [
            {"video_id": "V1", "frame_id": 1, "score": 0.6054},
            {"video_id": "V2", "frame_id": 2, "score": 0.5456, "vlm_raw_score": 8.0},
            {"video_id": "V3", "frame_id": 3, "score": 0.49, "vlm_raw_score": 8.0},
        ]

        ranked, applied = finalize_vlm_ranking(
            candidates,
            top_k=3,
            vlm_weight=0.5,
        )

        self.assertTrue(applied)
        self.assertEqual([item["frame_id"] for item in ranked], [2, 3, 1])
        self.assertAlmostEqual(ranked[0]["final_score"], 0.68192, places=4)
        self.assertAlmostEqual(ranked[1]["final_score"], 0.643, places=4)
        self.assertEqual(ranked[2]["final_score"], 0.6054)

    def test_max_fusion_preserves_single_variant_spike(self):
        spike = {"video_id": "V1", "frame_id": 1, "score": 0.9}
        filler_a = {"video_id": "V2", "frame_id": 2, "score": 0.5}
        filler_b = {"video_id": "V2", "frame_id": 2, "score": 0.45}
        fused = max_score_fusion([[spike], [filler_a], [filler_b]])

        # RRF would rank V2 (two appearances) above the one-off V1 spike;
        # max keeps the spike on top with its raw score and metadata.
        self.assertEqual(
            [(item["video_id"], item["score"]) for item in fused],
            [("V1", 0.9), ("V2", 0.5)],
        )

    def test_max_fusion_empty_input(self):
        self.assertEqual(max_score_fusion([]), [])

    def test_single_vlm_score_applies_bounded_boost(self):
        candidates = [
            {"video_id": "V1", "frame_id": 1, "score": 0.7},
            {"video_id": "V2", "frame_id": 2, "score": 0.5, "vlm_raw_score": 10.0},
        ]

        ranked, applied = finalize_vlm_ranking(
            candidates,
            top_k=2,
            vlm_weight=0.5,
        )

        self.assertTrue(applied)
        self.assertEqual([item["frame_id"] for item in ranked], [2, 1])
        self.assertAlmostEqual(ranked[0]["final_score"], 0.75, places=4)
        self.assertEqual(ranked[1]["final_score"], 0.7)

    def test_milvus_model_metadata_must_match_manifest(self):
        client = Mock()
        client.describe_collection.return_value = {
            "description": "{}",
        }
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            validate_collection_model(client, "clip_keyframes")

        import json

        client.describe_collection.return_value = {
            "description": json.dumps(milvus_model_properties()),
        }
        validate_collection_model(client, "clip_keyframes")

    def test_console_logging_handles_vietnamese_on_legacy_windows_encoding(self):
        escaped = console_safe_text("mặt bàn", encoding="cp1252")
        self.assertIn("\\u1eb7", escaped)
        self.assertIn("bàn", escaped)


if __name__ == "__main__":
    unittest.main()
