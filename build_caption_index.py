import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from artifact_io import publish_staged_files, temporary_path, write_json
from caption_artifact_manifest import write_caption_manifest
from caption_schema import (
    load_frame_mapping,
    normalize_caption_record,
    validate_unique_caption_records,
)
from model_config import CAPTION_CONFIG
from paths import (
    CAPTION_DIR,
    KIS_ROOT,
    MAPPING_DIR,
    canonical_video_artifact_path,
)

CAPTION_MODEL = CAPTION_CONFIG["model"]


def collect_caption_files(caption_dir=CAPTION_DIR):
    caption_files = {}
    for path in sorted(Path(caption_dir).rglob("L*_V*.json")):
        if not path.is_file():
            continue
        if path.stem in caption_files:
            raise RuntimeError(
                f"Duplicate caption files for {path.stem}: "
                f"{caption_files[path.stem]}, {path}"
            )
        caption_files[path.stem] = path
    return caption_files


def load_normalized_captions(video_id, caption_path, mapping_dir=MAPPING_DIR):
    with caption_path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Caption file must be a non-empty array: {caption_path}")

    frame_mapping = load_frame_mapping(video_id, mapping_dir)
    normalized = [
        normalize_caption_record(record, video_id, frame_mapping)
        for record in records
    ]
    validate_unique_caption_records(normalized, video_id)
    texts = [
        item.get("retrieval_text") or item.get("caption", "")
        for item in normalized
    ]
    if not any(texts):
        return normalized, None
    if any(not text for text in texts):
        raise ValueError(f"Some caption records have no searchable text: {caption_path}")

    return normalized, texts


def write_caption_artifacts(video_id, records, embeddings, project_root=KIS_ROOT):
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

    project_root = Path(project_root)
    index_path = canonical_video_artifact_path(project_root / "index", video_id, ".index")
    mapping_path = canonical_video_artifact_path(project_root / "caption_mapping", video_id, ".json")
    caption_path = canonical_video_artifact_path(project_root / "caption_generator", video_id, ".json")
    temp_index_path = temporary_path(index_path)
    temp_mapping_path = temporary_path(mapping_path)
    temp_caption_path = temporary_path(caption_path)

    try:
        write_json(temp_caption_path, records)
        write_json(temp_mapping_path, mapping)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(temp_index_path))
        publish_staged_files(
            (
                (caption_path, temp_caption_path),
                (mapping_path, temp_mapping_path),
                (index_path, temp_index_path),
            )
        )
    finally:
        temp_caption_path.unlink(missing_ok=True)
        temp_mapping_path.unlink(missing_ok=True)
        temp_index_path.unlink(missing_ok=True)

    manifest_path = write_caption_manifest(
        video_id,
        caption_path,
        mapping_path,
        index_path,
        len(records),
        manifest_path=index_path.with_suffix(".meta.json"),
    )
    return index_path, mapping_path, manifest_path


def build_caption_index(video_ids=None, project_root=KIS_ROOT, device=None, batch_size=32):
    project_root = Path(project_root)
    caption_dir = project_root / "caption_generator"
    caption_files = collect_caption_files(caption_dir)
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
        raise FileNotFoundError(f"No per-video captions found in {caption_dir}")

    (project_root / "index").mkdir(parents=True, exist_ok=True)
    (project_root / "caption_mapping").mkdir(parents=True, exist_ok=True)
    print(f"[Caption] Found {len(caption_files)} per-video caption files")
    print(f"[Caption] Loading {CAPTION_MODEL}...")
    model = SentenceTransformer(
        CAPTION_MODEL,
        revision=CAPTION_CONFIG["revision"],
        device=device,
    )

    built = 0
    skipped = []
    total_records = 0
    for video_id, caption_path in sorted(caption_files.items()):
        canonical_caption_path = canonical_video_artifact_path(caption_dir, video_id, ".json")
        if caption_path.resolve() != canonical_caption_path.resolve():
            raise RuntimeError(
                f"Legacy caption path detected for {video_id}: {caption_path}. "
                "Run migrate_caption_artifacts.py --apply first."
            )
        records, texts = load_normalized_captions(video_id, caption_path, project_root / "mapping")
        if texts is None:
            skipped.append(video_id)
            print(f"[{video_id}] skipped: no searchable caption text")
            continue

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        faiss.normalize_L2(embeddings)

        index_path, mapping_path, manifest_path = write_caption_artifacts(
            video_id,
            records,
            embeddings,
            project_root=project_root,
        )
        built += 1
        total_records += len(records)
        print(
            f"[{video_id}] records={len(records)} "
            f"index={index_path} mapping={mapping_path} manifest={manifest_path}"
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
