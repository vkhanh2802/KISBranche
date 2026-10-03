import importlib.util
import json
import unittest

from PIL import Image

from apply_object_filter import apply_object_filter
from caption_artifact_manifest import validate_caption_manifest
from caption_schema import load_frame_mapping, normalize_caption_record
from model_config import CAPTION_CONFIG
from object_filter import ObjectMetadataLookup
from paths import (
    KIS_ROOT,
    video_caption_index_path,
    video_caption_manifest_path,
    video_caption_mapping_path,
    video_caption_path,
    video_image_dir,
    video_object_dir,
)

SAMPLE_VIDEO_ID = "L21_V031"
SAMPLE_KEYFRAME = 1


class SampleDataTest(unittest.TestCase):
    def test_sample_image_mapping_caption_and_object_align(self):
        image_path = video_image_dir(SAMPLE_VIDEO_ID) / "001.jpg"
        self.assertTrue(image_path.is_file())
        with Image.open(image_path) as image:
            image.verify()
            self.assertGreater(image.width, 0)
            self.assertGreater(image.height, 0)

        frame_mapping = load_frame_mapping(SAMPLE_VIDEO_ID)
        self.assertIn(SAMPLE_KEYFRAME, frame_mapping)

        caption_path = video_caption_path(SAMPLE_VIDEO_ID)
        records = json.loads(caption_path.read_text(encoding="utf-8"))
        first_record = normalize_caption_record(
            records[0],
            SAMPLE_VIDEO_ID,
            frame_mapping,
        )
        self.assertEqual(first_record["keyframe_n"], SAMPLE_KEYFRAME)
        self.assertEqual(
            first_record["frame_id"],
            frame_mapping[SAMPLE_KEYFRAME],
        )
        self.assertTrue((KIS_ROOT / first_record["image_path"]).is_file())
        self.assertTrue(
            (video_object_dir(SAMPLE_VIDEO_ID) / "001.json").is_file()
        )

        candidates = [
            {
                "video_id": SAMPLE_VIDEO_ID,
                "frame_id": frame_mapping[SAMPLE_KEYFRAME],
                "score": 0.5,
            }
        ]
        scored = apply_object_filter(
            candidates,
            [{"type": "object", "value": "lantern"}],
            ObjectMetadataLookup(),
        )
        self.assertGreater(scored[0]["object_score"], 0.0)
        self.assertGreater(scored[0]["score"], 0.5)

    @unittest.skipUnless(
        importlib.util.find_spec("faiss") is not None,
        "faiss is not installed in this interpreter",
    )
    def test_sample_caption_index_and_mapping_counts_match(self):
        import faiss

        captions = json.loads(
            video_caption_path(SAMPLE_VIDEO_ID).read_text(encoding="utf-8")
        )
        mapping = json.loads(
            video_caption_mapping_path(SAMPLE_VIDEO_ID).read_text(
                encoding="utf-8"
            )
        )
        index = faiss.read_index(str(video_caption_index_path(SAMPLE_VIDEO_ID)))

        self.assertEqual(index.ntotal, len(captions))
        self.assertEqual(index.ntotal, len(mapping))
        self.assertEqual(index.d, int(CAPTION_CONFIG["dimension"]))
        validate_caption_manifest(
            SAMPLE_VIDEO_ID,
            video_caption_path(SAMPLE_VIDEO_ID),
            video_caption_mapping_path(SAMPLE_VIDEO_ID),
            video_caption_index_path(SAMPLE_VIDEO_ID),
            video_caption_manifest_path(SAMPLE_VIDEO_ID),
            index,
            len(captions),
        )


if __name__ == "__main__":
    unittest.main()
