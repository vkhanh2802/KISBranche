from pathlib import Path
import re


KIS_ROOT = Path(__file__).resolve().parent
DATA_DIR = KIS_ROOT / "data"
OBJECTS_DIR = KIS_ROOT / "objects"
MAPPING_DIR = KIS_ROOT / "mapping"
CLIP_FEATURES_DIR = KIS_ROOT / "clip-features-32"
CAPTION_DIR = KIS_ROOT / "caption_generator"
CAPTION_MAPPING_DIR = KIS_ROOT / "caption_mapping"
INDEX_DIR = KIS_ROOT / "index"

CAPTION_JSON = CAPTION_DIR / "captions.json"
CAPTION_INDEX_PATH = INDEX_DIR / "caption.index"
CAPTION_INDEX_MAPPING_PATH = INDEX_DIR / "caption_mapping.json"

VIDEO_ID_PATTERN = re.compile(r"^L\d+_V\d+$")


def video_group(video_id: str) -> str:
    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError(f"Invalid video_id: {video_id!r}")
    return video_id.split("_", 1)[0]


def canonical_video_artifact_path(
    root: Path,
    video_id: str,
    suffix: str,
) -> Path:
    return Path(root) / video_group(video_id) / f"{video_id}{suffix}"


def resolve_video_artifact(
    root: Path,
    video_id: str,
    suffix: str,
) -> Path:
    root = Path(root)
    canonical = canonical_video_artifact_path(root, video_id, suffix)
    candidates = []

    for path in (canonical, root / f"{video_id}{suffix}"):
        if path.is_file():
            candidates.append(path)

    candidates.extend(
        path
        for path in root.rglob(f"{video_id}{suffix}")
        if path.is_file()
    )
    unique_candidates = sorted({path.resolve() for path in candidates})

    if len(unique_candidates) > 1:
        locations = ", ".join(str(path) for path in unique_candidates)
        raise RuntimeError(
            f"Duplicate artifacts for {video_id}{suffix}: {locations}"
        )
    if unique_candidates:
        return unique_candidates[0]

    return canonical


def video_image_dir(video_id: str) -> Path:
    video_group(video_id)
    return DATA_DIR / video_id


def video_csv_path(video_id: str) -> Path:
    video_group(video_id)
    return MAPPING_DIR / f"{video_id}.csv"


def video_clip_path(video_id: str) -> Path:
    video_group(video_id)
    return CLIP_FEATURES_DIR / f"{video_id}.npy"


def video_caption_path(video_id: str) -> Path:
    return canonical_video_artifact_path(CAPTION_DIR, video_id, ".json")


def video_caption_index_path(video_id: str) -> Path:
    return canonical_video_artifact_path(INDEX_DIR, video_id, ".index")


def video_caption_mapping_path(video_id: str) -> Path:
    return canonical_video_artifact_path(
        CAPTION_MAPPING_DIR,
        video_id,
        ".json",
    )


def video_object_dir(video_id: str) -> Path:
    video_group(video_id)
    return OBJECTS_DIR / video_id


def stored_path(path: Path) -> str:
    return path.resolve().relative_to(KIS_ROOT).as_posix()
