import argparse
import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from artifact_io import atomic_write_json, temporary_path
from caption_schema import load_frame_mapping, normalize_caption_record
from model_config import CAPTION_CONFIG
from paths import (
    CAPTION_DIR,
    CAPTION_MAPPING_DIR,
    INDEX_DIR,
    video_caption_index_path,
    video_caption_mapping_path,
    video_caption_path,
)

CAPTION_MODEL = CAPTION_CONFIG["model"]


def collect_caption_files():
    caption_files = {}
    for path in sorted(CAPTION_DIR.rglob("L*_V*.json")):
        if not path.is_file():
            continue
        if path.stem in caption_files:
            raise RuntimeError(
                f"Duplicate caption files for {path.stem}: "
                f"{caption_files[path.stem]}, {path}"
            )
        caption_files[path.stem] = path
    return caption_files


def load_normalized_captions(video_id, caption_path):
    with caption_path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Caption file must be a non-empty array: {caption_path}")

    frame_mapping = load_frame_mapping(video_id)
    normalized = [
        normalize_caption_record(record, video_id, frame_mapping)
        for record in records
    ]
    texts = [
        item.get("retrieval_text") or item.get("caption", "")
        for item in normalized
    ]
    if not any(texts):
        return normalized, None
    if any(not text for text in texts):
        raise ValueError(f"Some caption records have no searchable text: {caption_path}")

    return normalized, texts


def write_caption_artifacts(video_id, records, embeddings):
    mapping = [
        {
            "video_id": item["video_id"],
            "frame_id": int(item["frame_id"]),
            "keyframe_n": int(item["keyframe_n"]),
            "image_path": item.get("image_path", ""),
        }
        for item in records
    ]

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_path = video_caption_index_path(video_id)
    mapping_path = video_caption_mapping_path(video_id)
    temp_index_path = temporary_path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        faiss.write_index(index, str(temp_index_path))
        atomic_write_json(mapping_path, mapping)
        os.replace(temp_index_path, index_path)
    finally:
        temp_index_path.unlink(missing_ok=True)

    return index_path, mapping_path


def build_caption_index(video_ids=None):
    caption_files = collect_caption_files()
    if video_ids is not None:
        requested = set(video_ids)
        missing = sorted(requested - set(caption_files))
        if missing:
            raise FileNotFoundError(f"Missing caption files: {', '.join(missing)}")
        caption_files = {
            video_id: caption_files[video_id]
            for video_id in sorted(requested)
        }

    if not caption_files:
        raise FileNotFoundError(f"No per-video captions found in {CAPTION_DIR}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CAPTION_MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Caption] Found {len(caption_files)} per-video caption files")
    print(f"[Caption] Loading {CAPTION_MODEL}...")
    model = SentenceTransformer(
        CAPTION_MODEL,
        revision=CAPTION_CONFIG["revision"],
    )

    built = 0
    skipped = []
    total_records = 0
    for video_id, caption_path in sorted(caption_files.items()):
        canonical_caption_path = video_caption_path(video_id)
        if caption_path.resolve() != canonical_caption_path.resolve():
            raise RuntimeError(
                f"Legacy caption path detected for {video_id}: {caption_path}. "
                "Run migrate_caption_artifacts.py --apply first."
            )
        records, texts = load_normalized_captions(video_id, caption_path)
        if texts is None:
            skipped.append(video_id)
            print(f"[{video_id}] skipped: no searchable caption text")
            continue

        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        faiss.normalize_L2(embeddings)

        if records != json.loads(caption_path.read_text(encoding="utf-8")):
            atomic_write_json(canonical_caption_path, records)

        index_path, mapping_path = write_caption_artifacts(
            video_id,
            records,
            embeddings,
        )
        built += 1
        total_records += len(records)
        print(
            f"[{video_id}] records={len(records)} "
            f"index={index_path} mapping={mapping_path}"
        )

    print(f"Built {built} videos ({total_records} records)")
    if skipped:
        print(f"Skipped {len(skipped)} videos without searchable text")

    return {"built": built, "records": total_records, "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(
        description="Build canonical per-video caption indexes."
    )
    parser.add_argument(
        "--video-id",
        action="append",
        dest="video_ids",
        help="build only this video; repeat for multiple videos",
    )
    args = parser.parse_args()
    build_caption_index(video_ids=args.video_ids)


if __name__ == "__main__":
    main()
