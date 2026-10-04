import math
from pathlib import Path
import unittest


try:
    from web_app import serialize_search_results
except ModuleNotFoundError:
    serialize_search_results = None


@unittest.skipIf(serialize_search_results is None, "web dependencies are not installed")
class WebResultSerializationTest(unittest.TestCase):
    def test_serializes_card_metadata_and_scores(self):
        candidate = {
            "video_id": "L27_V013",
            "frame_id": 8876,
            "final_score": 0.7178,
            "fusion_score": 0.701,
            "retrieval_text": "Two people on a boat.",
        }

        def image_resolver(item, _lookup):
            item["keyframe_n"] = 186
            return Path("186.jpg")

        results = serialize_search_results(
            [candidate],
            object_lookup=None,
            image_resolver=image_resolver,
        )

        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[0]["video_id"], "L27_V013")
        self.assertEqual(results[0]["image_number"], 186)
        self.assertEqual(results[0]["frame_id"], 8876)
        self.assertEqual(results[0]["image_url"], "/api/images/L27_V013/186")
        self.assertEqual(results[0]["caption"], "Two people on a boat.")

    def test_non_finite_scores_are_json_safe(self):
        candidate = {
            "video_id": "L27_V013",
            "frame_id": 1,
            "final_score": math.nan,
        }
        results = serialize_search_results(
            [candidate],
            object_lookup=None,
            image_resolver=lambda *_: None,
        )

        self.assertIsNone(results[0]["score"])
        self.assertIsNone(results[0]["image_url"])


if __name__ == "__main__":
    unittest.main()
