import argparse
import hashlib
import json
from pathlib import Path

from artifact_io import atomic_write_json
from model_config import CAPTION_CONFIG
from paths import (
    CAPTION_DIR,
    video_caption_index_path,
    video_caption_manifest_path,
    video_caption_mapping_path,
)

SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_data(video_id, caption_path, mapping_path, index_path, count):
    return {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "count": int(count),
        "files": {
            "caption_sha256": file_sha256(caption_path),
            "mapping_sha256": file_sha256(mapping_path),
            "index_sha256": file_sha256(index_path),
        },
        "embedding": {
            "model": CAPTION_CONFIG["model"],
            "revision": CAPTION_CONFIG["revision"],
            "dimension": int(CAPTION_CONFIG["dimension"]),
            "normalized": bool(CAPTION_CONFIG["normalized"]),
            "metric": CAPTION_CONFIG["metric"],
        },
    }


def write_caption_manifest(
    video_id,
    caption_path,
    mapping_path,
    index_path,
    count,
):
    manifest_path = video_caption_manifest_path(video_id)
    atomic_write_json(
        manifest_path,
        manifest_data(
            video_id,
            caption_path,
            mapping_path,
            index_path,
            count,
        ),
    )
    return manifest_path


def validate_caption_manifest(
    video_id,
    caption_path,
    mapping_path,
    index_path,
    manifest_path,
    index,
    count,
):
    with Path(manifest_path).open("r", encoding="utf-8") as file:
        actual = json.load(file)
    expected = manifest_data(
        video_id,
        caption_path,
        mapping_path,
        index_path,
        count,
    )
    if actual != expected:
        raise ValueError(
            f"Caption artifact manifest mismatch for {video_id}: {manifest_path}. "
            "Rebuild this video's caption index."
        )
    if index.d != int(CAPTION_CONFIG["dimension"]):
        raise ValueError(
            f"Caption index dimension mismatch for {video_id}: "
            f"{index.d} != {CAPTION_CONFIG['dimension']}"
        )


def backfill_caption_manifests(video_ids=None, only_missing=False):
    import faiss

    requested = set(video_ids) if video_ids is not None else None
    written = 0
    for caption_path in sorted(CAPTION_DIR.rglob("L*_V*.json")):
        video_id = caption_path.stem
        if requested is not None and video_id not in requested:
            continue
        manifest_path = video_caption_manifest_path(video_id)
        if only_missing and manifest_path.is_file():
            continue
        mapping_path = video_caption_mapping_path(video_id)
        index_path = video_caption_index_path(video_id)
        if not mapping_path.is_file() or not index_path.is_file():
            raise FileNotFoundError(f"Incomplete caption artifacts for {video_id}")
        captions = json.loads(caption_path.read_text(encoding="utf-8"))
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        index = faiss.read_index(str(index_path))
        if len(captions) != len(mapping) or len(captions) != index.ntotal:
            raise ValueError(f"Artifact count mismatch for {video_id}")
        if index.d != int(CAPTION_CONFIG["dimension"]):
            raise ValueError(f"Caption index dimension mismatch for {video_id}")
        for position, (caption, mapped) in enumerate(zip(captions, mapping)):
            caption_identity = (
                caption.get("video_id"),
                caption.get("frame_id"),
                caption.get("keyframe_n"),
            )
            mapping_identity = (
                mapped.get("video_id"),
                mapped.get("frame_id"),
                mapped.get("keyframe_n"),
            )
            if caption_identity != mapping_identity:
                raise ValueError(
                    f"Artifact identity mismatch for {video_id} at row {position}"
                )
        write_caption_manifest(
            video_id,
            caption_path,
            mapping_path,
            index_path,
            len(captions),
        )
        written += 1
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Backfill integrity manifests for canonical caption artifacts."
    )
    parser.add_argument(
        "--video-id",
        action="append",
        dest="video_ids",
        help="backfill only this video; repeat for multiple videos",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="do not replace existing manifests",
    )
    parser.add_argument(
        "--acknowledge-model-provenance",
        action="store_true",
        help="confirm the existing indexes use the pinned caption model",
    )
    args = parser.parse_args()
    if not args.acknowledge_model_provenance:
        parser.error("--acknowledge-model-provenance is required")
    written = backfill_caption_manifests(
        video_ids=args.video_ids,
        only_missing=args.only_missing,
    )
    print(f"Wrote {written} caption artifact manifests")


if __name__ == "__main__":
    main()
