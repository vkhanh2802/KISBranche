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


def safe_filename_component(value, default="output") -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return sanitized or default


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


def resolve_video_image_dir(video_id: str, image_root=DATA_DIR) -> Path:
    group = video_group(video_id)
    image_root = Path(image_root)
    candidates = [
        path
        for path in (image_root / video_id, image_root / group / video_id)
        if path.is_dir()
    ]
    unique_candidates = sorted({path.resolve() for path in candidates})

    if len(unique_candidates) > 1:
        locations = ", ".join(str(path) for path in unique_candidates)
        raise RuntimeError(
            f"Duplicate image directories for {video_id}: {locations}"
        )
    if unique_candidates:
        return unique_candidates[0]

    return image_root / video_id


def discover_video_image_dirs(image_root=DATA_DIR):
    image_root = Path(image_root)
    discovered = {}

    def register(video_id, image_dir):
        existing = discovered.get(video_id)
        if existing is not None and existing.resolve() != image_dir.resolve():
            raise RuntimeError(
                f"Duplicate image directories for {video_id}: "
                f"{existing}, {image_dir}"
            )
        discovered.setdefault(video_id, image_dir)

    if not image_root.is_dir():
        return discovered

    for child in sorted(image_root.iterdir()):
        if child.is_dir() and VIDEO_ID_PATTERN.fullmatch(child.name):
            register(child.name, child)

    for child in sorted(image_root.iterdir()):
        if child.is_dir() and not VIDEO_ID_PATTERN.fullmatch(child.name):
            for subdir in sorted(child.iterdir()):
                if subdir.is_dir() and VIDEO_ID_PATTERN.fullmatch(subdir.name):
                    register(subdir.name, subdir)

    return discovered


def video_image_dir(video_id: str) -> Path:
    return resolve_video_image_dir(video_id)


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


def video_caption_manifest_path(video_id: str) -> Path:
    return canonical_video_artifact_path(INDEX_DIR, video_id, ".meta.json")


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
