"""Check each dataset folder against the formats used by the main KIS corpus."""

import argparse
import csv
import json
from pathlib import Path

from caption_artifact_manifest import validate_caption_manifest
from caption_schema import normalize_caption_record, validate_unique_caption_records
from model_config import CAPTION_CONFIG, VISUAL_CONFIG
from object_schema import validate_object_schema
from paths import KIS_ROOT, canonical_video_artifact_path, resolve_video_image_dir, video_group


CAPTION_FIELDS = {"video_id", "frame_id", "keyframe_n", "image_path", "caption", "vqa_answers", "details", "retrieval_text"}
CAPTION_MAPPING_FIELDS = {"video_id", "frame_id", "keyframe_n", "image_path"}
VQA_FIELDS = {"people", "clothing_color", "main_object", "object_color", "action", "setting", "location"}


class IncompleteArtifact(ValueError):
    pass


def audit_video_formats(project_root, video_id, allow_incomplete=False):
    import numpy as np

    root = Path(project_root)
    group = video_group(video_id)
    reports = []
    rows = []
    captions = []

    def inspect(folder, operation):
        try:
            detail = operation()
            status = "OK"
        except (FileNotFoundError, IncompleteArtifact) as exc:
            status = "PENDING" if allow_incomplete else "ERROR"
            detail = str(exc)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            status, detail = "ERROR", str(exc)
        reports.append({"video_id": video_id, "folder": folder, "status": status, "detail": detail})

    def mapping():
        with (root / "mapping" / f"{video_id}.csv").open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != ["n", "pts_time", "fps", "frame_idx"]:
                raise ValueError("CSV columns must be n,pts_time,fps,frame_idx in this order")
            for row in reader:
                rows.append({"n": int(row["n"]), "pts_time": float(row["pts_time"]),
                             "fps": float(row["fps"]), "frame_idx": int(row["frame_idx"])})
        if not rows:
            raise ValueError("Mapping is empty")
        if [row["n"] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError("Mapping n must be consecutive and start at 1")
        if any(row["frame_idx"] < 0 or not np.isfinite(row["pts_time"]) or row["pts_time"] < 0
               or not np.isfinite(row["fps"]) or row["fps"] <= 0 for row in rows):
            raise ValueError("Mapping contains invalid frames, timestamps or FPS")
        if any(current["frame_idx"] <= previous["frame_idx"] or current["pts_time"] < previous["pts_time"]
               for previous, current in zip(rows, rows[1:])):
            raise ValueError("Mapping frames/timestamps must be ordered")
        return f"{len(rows)} rows; n,pts_time,fps,frame_idx"

    inspect("mapping", mapping)
    frame_mapping = {row["n"]: row["frame_idx"] for row in rows}
    image_dir = resolve_video_image_dir(video_id, root / "data")

    def data():
        from PIL import Image

        if not rows:
            raise IncompleteArtifact("Waiting for mapping")
        expected = {f"{row['n']:03d}.jpg" for row in rows}
        actual = {path.name for path in image_dir.glob("*.jpg")}
        if actual != expected:
            raise IncompleteArtifact(f"Expected {len(expected)} JPEGs, found {len(actual)}")
        for name in sorted(expected):
            with Image.open(image_dir / name) as image:
                if image.format != "JPEG" or image.mode != "RGB" or min(image.size) <= 0:
                    raise ValueError(f"Invalid JPEG/RGB image: {name}")
                image.verify()
        return f"{len(expected)} JPEG RGB images; 001.jpg,002.jpg,..."

    inspect("data", data)

    def features():
        vectors = np.load(root / "clip-features-32" / f"{video_id}.npy", mmap_mode="r", allow_pickle=False)
        expected = (len(rows), int(VISUAL_CONFIG["dimension"]))
        if vectors.shape != expected or vectors.dtype != np.float32:
            raise ValueError(f"Expected float32 {expected}, got {vectors.dtype} {vectors.shape}")
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
            raise ValueError("Visual vectors must be finite and approximately L2 normalized")
        return f"float32 {vectors.shape}; 768 dimensions; mapping row order"

    inspect("clip-features-32", features)

    def objects():
        object_dir = root / "objects" / video_id
        files = sorted(object_dir.glob("*.json"))
        expected = {f"{row['n']:03d}.json" for row in rows}
        for path in files:
            if path.name not in expected:
                raise ValueError(f"Unexpected object file: {path.name}")
            validate_object_schema(json.loads(path.read_text(encoding="utf-8")))
        if not rows or {path.name for path in files} != expected:
            raise IncompleteArtifact(f"Objects: {len(files)}/{len(rows)} files")
        return f"{len(files)} JSONs; 5 fields; strings; Open Images MID/label; normalized yxyx boxes"

    inspect("objects", objects)

    def caption_generator():
        path = canonical_video_artifact_path(root / "caption_generator", video_id, ".json")
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("Caption JSON must be an array")
        for record in records:
            if not CAPTION_FIELDS.issubset(record) or set(record) - CAPTION_FIELDS - {"ocr_text"}:
                raise ValueError("Caption fields do not match the main corpus schema")
            if type(record["frame_id"]) is not int or type(record["keyframe_n"]) is not int:
                raise ValueError("Caption frame_id/keyframe_n must be integers")
            for field in ("video_id", "image_path", "caption", "details", "retrieval_text"):
                if not isinstance(record[field], str):
                    raise ValueError(f"Caption {field} must be a string")
            answers = record["vqa_answers"]
            if not isinstance(answers, dict) or set(answers) != VQA_FIELDS or not all(isinstance(v, str) for v in answers.values()):
                raise ValueError("Caption VQA must contain the seven string answer fields")
            normalized = normalize_caption_record(record, video_id, frame_mapping)
            if not (root / normalized["image_path"]).is_file() or not normalized["retrieval_text"].strip():
                raise ValueError("Caption image/text is missing")
            captions.append(record)
        validate_unique_caption_records(captions, video_id)
        if {record["keyframe_n"] for record in captions} != set(frame_mapping):
            raise IncompleteArtifact(f"Captions: {len(captions)}/{len(rows)} records; available records have valid schema")
        return f"{len(captions)} records; integer IDs; relative image_path; 7 VQA fields"

    inspect("caption_generator", caption_generator)

    def caption_mapping():
        path = canonical_video_artifact_path(root / "caption_mapping", video_id, ".json")
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list) or len(records) != len(rows):
            raise IncompleteArtifact("Caption mapping count is incomplete")
        for record, caption in zip(records, captions):
            if set(record) != CAPTION_MAPPING_FIELDS or any(record[field] != caption[field] for field in CAPTION_MAPPING_FIELDS):
                raise ValueError("Caption mapping must contain four matching identity fields")
        if len(records) != len(captions):
            raise IncompleteArtifact("Caption mapping does not have complete captions yet")
        return f"{len(records)} records; video_id,frame_id,keyframe_n,image_path"

    inspect("caption_mapping", caption_mapping)

    def index():
        import faiss

        index_path = canonical_video_artifact_path(root / "index", video_id, ".index")
        if not index_path.is_file():
            raise FileNotFoundError(f"Waiting for {index_path.name}")
        vectors = faiss.read_index(str(index_path))
        if not isinstance(vectors, faiss.IndexFlatIP) or vectors.d != int(CAPTION_CONFIG["dimension"]):
            raise ValueError("Caption index must be IndexFlatIP with 384 dimensions")
        if vectors.ntotal != len(rows) or len(captions) != len(rows):
            raise IncompleteArtifact("Caption index count is incomplete")
        validate_caption_manifest(
            video_id, canonical_video_artifact_path(root / "caption_generator", video_id, ".json"),
            canonical_video_artifact_path(root / "caption_mapping", video_id, ".json"),
            index_path, index_path.with_suffix(".meta.json"), vectors, len(captions),
        )
        return f"IndexFlatIP ({vectors.ntotal},384); SHA-256/model manifest OK"

    inspect("index", index)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=KIS_ROOT)
    parser.add_argument("--video-id", action="append", required=True, dest="video_ids")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    reports = [report for video_id in args.video_ids
               for report in audit_video_formats(args.project_root, video_id, args.allow_incomplete)]
    for report in reports:
        print(json.dumps(report, ensure_ascii=False))
    if any(report["status"] == "ERROR" for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
