import math
import unittest

from apply_object_filter import apply_object_filter
from solve_kis import normalize_scores, static_weight_fusion_3_way


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


if __name__ == "__main__":
    unittest.main()
