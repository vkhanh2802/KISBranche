from pathlib import Path


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


def video_image_dir(video_id: str) -> Path:
    return DATA_DIR / video_id


def video_csv_path(video_id: str) -> Path:
    return MAPPING_DIR / f"{video_id}.csv"


def video_clip_path(video_id: str) -> Path:
    return CLIP_FEATURES_DIR / f"{video_id}.npy"


def video_caption_path(video_id: str) -> Path:
    return CAPTION_DIR / f"{video_id}.json"


def video_caption_index_path(video_id: str) -> Path:
    return INDEX_DIR / f"{video_id}.index"


def video_caption_mapping_path(video_id: str) -> Path:
    return CAPTION_MAPPING_DIR / f"{video_id}.json"


def video_object_dir(video_id: str) -> Path:
    return OBJECTS_DIR / video_id


def stored_path(path: Path) -> str:
    return path.resolve().relative_to(KIS_ROOT).as_posix()
