import csv
from pathlib import Path
from pathlib import PurePosixPath

from paths import MAPPING_DIR, video_csv_path, video_group


def load_frame_mapping(video_id: str, mapping_dir=MAPPING_DIR):
    video_group(video_id)
    mapping_dir = Path(mapping_dir)
    csv_path = (
        video_csv_path(video_id)
        if mapping_dir == MAPPING_DIR
        else mapping_dir / f"{video_id}.csv"
    )
    mapping = {}

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"n", "frame_idx"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{csv_path} must contain columns: n, frame_idx")

        for row_number, row in enumerate(reader, start=2):
            try:
                keyframe_n = int(row["n"])
                frame_id = int(row["frame_idx"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid mapping at {csv_path}:{row_number}"
                ) from exc
            if keyframe_n in mapping:
                raise ValueError(
                    f"Duplicate keyframe {keyframe_n} in {csv_path}"
                )
            mapping[keyframe_n] = frame_id

    return mapping


def rebuild_retrieval_text(record):
    base_text = ". ".join(
        text.strip(". ")
        for text in (
            str(record.get("caption", "") or ""),
            str(record.get("details", "") or ""),
        )
        if text.strip()
    )
    ocr_text = str(record.get("ocr_text", "") or "").strip()

    if ocr_text and base_text:
        return f"{base_text}. [OCR]: {ocr_text}"
    if ocr_text:
        return f"[OCR]: {ocr_text}"
    return base_text


def normalize_caption_record(record, video_id, frame_mapping=None):
    if not isinstance(record, dict):
        raise ValueError(f"Caption record for {video_id} must be an object")

    normalized = dict(record)
    if normalized.get("video_id") != video_id:
        raise ValueError(
            f"Caption video_id mismatch: expected {video_id}, "
            f"got {normalized.get('video_id')!r}"
        )

    try:
        keyframe_n = int(normalized["keyframe_n"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid keyframe_n for {video_id}") from exc
    normalized["keyframe_n"] = keyframe_n

    frame_id = normalized.get("frame_id")
    if frame_id is None:
        if frame_mapping is None or keyframe_n not in frame_mapping:
            raise ValueError(
                f"Cannot resolve frame_id for {video_id} keyframe {keyframe_n}"
            )
        frame_id = frame_mapping[keyframe_n]
    try:
        normalized["frame_id"] = int(frame_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid frame_id for {video_id}: {frame_id!r}") from exc
    if frame_mapping is not None:
        expected_frame_id = frame_mapping.get(keyframe_n)
        if expected_frame_id is None:
            raise ValueError(
                f"Missing mapping for {video_id} keyframe {keyframe_n}"
            )
        if normalized["frame_id"] != expected_frame_id:
            raise ValueError(
                f"frame_id mismatch for {video_id} keyframe {keyframe_n}: "
                f"{normalized['frame_id']} != {expected_frame_id}"
            )

    image_path = str(normalized.get("image_path", "")).replace("\\", "/")
    parsed_path = PurePosixPath(image_path)
    if (
        parsed_path.is_absolute()
        or ".." in parsed_path.parts
        or parsed_path.parts[:2] != ("data", video_id)
    ):
        raise ValueError(f"Unsafe image_path for {video_id}: {image_path!r}")
    normalized["image_path"] = parsed_path.as_posix()

    if "ocr_text" in normalized or not normalized.get("retrieval_text"):
        normalized["retrieval_text"] = rebuild_retrieval_text(normalized)

    return normalized


def caption_identity(record):
    return (
        record["video_id"],
        int(record["frame_id"]),
        int(record["keyframe_n"]),
    )


def validate_unique_caption_records(records, video_id):
    keyframes = set()
    for record in records:
        keyframe_n = int(record["keyframe_n"])
        if keyframe_n in keyframes:
            raise ValueError(
                f"Duplicate keyframe_n for {video_id}: {keyframe_n}"
            )
        keyframes.add(keyframe_n)
