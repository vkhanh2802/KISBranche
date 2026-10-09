"""CPU video ingestion producing artifacts consumed by the KIS retrievers."""

import argparse
import csv
from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import tempfile

from artifact_io import atomic_write_json, publish_staged_files, temporary_path, write_json
from caption_artifact_manifest import file_sha256, validate_caption_manifest
from caption_schema import normalize_caption_record, validate_unique_caption_records
from model_config import CAPTION_GENERATION_CONFIG, VISUAL_CONFIG
from object_schema import LABEL_MAP_PATH, provider_object_record, validate_object_schema
from paths import KIS_ROOT, canonical_video_artifact_path, resolve_video_image_dir, video_group


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
OBJECT_MODEL = {
    "model": "torchvision/ssdlite320_mobilenet_v3_large",
    "weights": "SSDLite320_MobileNet_V3_Large_Weights.COCO_V1",
}


def configure_cpu(threads=8):
    """Bound CPU parallelism on a shared server and force CPU-only inference."""
    threads = int(threads)
    if threads < 1:
        raise ValueError("threads must be positive")
    threads = min(threads, os.cpu_count() or 1)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy  # Import before PyTorch for the shared Windows OpenMP runtime.
    import torch

    torch.set_num_threads(threads)
    return {
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "platform": sys.platform,
        "cpu_threads": torch.get_num_threads(),
        "device": "cpu",
    }


def discover_videos(video_dir, video_ids=None):
    video_dir = Path(video_dir)
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Missing video directory: {video_dir}")
    videos = {}
    for path in sorted(video_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        video_group(path.stem)
        if path.stem in videos:
            raise ValueError(f"Duplicate video_id {path.stem}: {videos[path.stem]}, {path}")
        videos[path.stem] = path
    if video_ids is not None:
        requested = set(video_ids)
        missing = requested - set(videos)
        if missing:
            raise FileNotFoundError(f"Missing videos: {', '.join(sorted(missing))}")
        videos = {video_id: videos[video_id] for video_id in sorted(requested)}
    if not videos:
        raise FileNotFoundError(f"No videos found in {video_dir}")
    return videos


@dataclass(frozen=True)
class KeyframeConfig:
    frame_gap_min: int | None = 80
    frame_gap_max: int = 160
    random_seed: int = 42
    interval_seconds: float = 2.0
    scene_check_seconds: float = 0.5
    scene_threshold: float = 0.25
    min_scene_gap_seconds: float = 0.75
    jpeg_quality: int = 92
    max_duration_seconds: float | None = None

    def __post_init__(self):
        if self.frame_gap_min is not None:
            for value in (self.frame_gap_min, self.frame_gap_max):
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError("Frame gaps must be positive integers")
            if self.frame_gap_min > self.frame_gap_max:
                raise ValueError("frame_gap_min must not exceed frame_gap_max")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer")
        for value in (self.interval_seconds, self.scene_check_seconds, self.min_scene_gap_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Keyframe time intervals must be finite and positive")
        if not 0 <= self.scene_threshold <= 1:
            raise ValueError("scene_threshold must be between 0 and 1")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if self.max_duration_seconds is not None:
            if not math.isfinite(self.max_duration_seconds) or self.max_duration_seconds <= 0:
                raise ValueError("max_duration_seconds must be positive or None")


class CPUObjectDetector:
    """COCO detector with a lightweight MobileNet backbone for CPU inference."""

    def __init__(self):
        import numpy
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights,
            ssdlite320_mobilenet_v3_large,
        )

        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        self.categories = weights.meta["categories"]
        self.model = ssdlite320_mobilenet_v3_large(weights=weights).to("cpu").eval()

    def detect(self, images, threshold=0.2):
        import torch
        from torchvision.transforms.functional import to_tensor

        with torch.inference_mode():
            predictions = self.model([to_tensor(image).to("cpu") for image in images])
        output = []
        for image, prediction in zip(images, predictions):
            selected = prediction["scores"] >= threshold
            labels = prediction["labels"][selected].tolist()
            scores = prediction["scores"][selected].tolist()
            boxes = prediction["boxes"][selected].tolist()
            width, height = image.size
            output.append(
                {
                    "detection_class_entities": [self.categories[label] for label in labels],
                    "detection_class_names": [str(label) for label in labels],
                    "detection_scores": scores,
                    "detection_boxes": [
                        [
                            max(0.0, min(1.0, y1 / height)),
                            max(0.0, min(1.0, x1 / width)),
                            max(0.0, min(1.0, y2 / height)),
                            max(0.0, min(1.0, x2 / width)),
                        ]
                        for x1, y1, x2, y2 in boxes
                    ],
                    "box_format": "ymin,xmin,ymax,xmax (normalized)",
                }
            )
        return output


def _validate_legacy_object_record(record, identity, detector_config, image_sha256):
    if any(record.get(field) != value for field, value in identity.items()):
        raise ValueError(f"Object identity mismatch: {identity}")
    if record.get("detector") != detector_config or record.get("image_sha256") != image_sha256:
        raise ValueError(f"Object provenance mismatch: {identity}")
    labels = record.get("detection_class_entities")
    scores = record.get("detection_scores")
    boxes = record.get("detection_boxes")
    if not all(isinstance(items, list) for items in (labels, scores, boxes)):
        raise ValueError(f"Invalid object arrays: {identity}")
    if len(labels) != len(scores) or len(labels) != len(boxes):
        raise ValueError(f"Object array count mismatch: {identity}")
    for label, score, box in zip(labels, scores, boxes):
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Invalid object label: {identity}")
        if not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
            raise ValueError(f"Invalid object confidence: {identity}")
        if len(box) != 4 or any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in box):
            raise ValueError(f"Invalid object box: {identity}")
        if box[0] > box[2] or box[1] > box[3]:
            raise ValueError(f"Inverted object box: {identity}")


class VideoDatasetBuilder:
    def __init__(self, project_root=KIS_ROOT):
        self.root = Path(project_root).resolve()

    def artifact_paths(self, video_id):
        group = video_group(video_id)
        return {
            "images": self.root / "data" / group / video_id,
            "mapping": self.root / "mapping" / f"{video_id}.csv",
            "features": self.root / "clip-features-32" / f"{video_id}.npy",
            "objects": self.root / "objects" / video_id,
            "captions": canonical_video_artifact_path(self.root / "caption_generator", video_id, ".json"),
            "caption_mapping": canonical_video_artifact_path(self.root / "caption_mapping", video_id, ".json"),
            "index": canonical_video_artifact_path(self.root / "index", video_id, ".index"),
            "manifest": canonical_video_artifact_path(self.root / "index", video_id, ".meta.json"),
            "preparation": canonical_video_artifact_path(self.root / "preparation", video_id, ".json"),
        }

    def _state(self, video_id):
        path = self.artifact_paths(video_id)["preparation"]
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def _save_state(self, video_id, state):
        atomic_write_json(self.artifact_paths(video_id)["preparation"], state)

    def mapping_rows(self, video_id):
        """Read mapping in row order and verify its source JPEGs before inference."""
        paths = self.artifact_paths(video_id)
        extraction = self._state(video_id).get("keyframes")
        if not extraction:
            raise ValueError(f"No ingestion checkpoint for {video_id}; run keyframe extraction first")
        if file_sha256(paths["mapping"]) != extraction["mapping_sha256"]:
            raise ValueError(f"Mapping was changed after extraction: {video_id}")
        with paths["mapping"].open(encoding="utf-8", newline="") as file:
            rows = [
                {"n": int(row["n"]), "pts_time": float(row["pts_time"]),
                 "fps": float(row["fps"]), "frame_idx": int(row["frame_idx"])}
                for row in csv.DictReader(file)
            ]
        if not rows or len(rows) != len(extraction["images"]):
            raise ValueError(f"Keyframe count mismatch: {video_id}")
        previous_frame = -1
        previous_time = -1.0
        for n, row in enumerate(rows, start=1):
            if row["n"] != n or row["frame_idx"] <= previous_frame:
                raise ValueError(f"Invalid mapping order: {video_id}")
            if not math.isfinite(row["pts_time"]) or row["pts_time"] < previous_time:
                raise ValueError(f"Invalid presentation timestamp: {video_id}")
            if not math.isfinite(row["fps"]) or row["fps"] <= 0:
                raise ValueError(f"Invalid FPS: {video_id}")
            image_path = paths["images"] / f"{n:03d}.jpg"
            if file_sha256(image_path) != extraction["images"][image_path.name]:
                raise ValueError(f"Keyframe was changed after extraction: {image_path}")
            previous_frame, previous_time = row["frame_idx"], row["pts_time"]
        return rows

    def extract_keyframes(self, video_path, config=None):
        """Sample reproducible random frame gaps; optional legacy time/scene mode."""
        import av
        import numpy as np

        video_path = Path(video_path)
        video_id = video_path.stem
        paths = self.artifact_paths(video_id)
        config = config or KeyframeConfig()
        source = {"name": video_path.name, "size": video_path.stat().st_size,
                  "sha256": file_sha256(video_path)}
        state = self._state(video_id)
        if "keyframes" in state:
            if state["source"] != source or state["keyframes"]["config"] != asdict(config):
                raise ValueError(
                    f"Source/config changed for {video_id}. Use a new video_id or a separate output root."
                )
            rows = self.mapping_rows(video_id)
            print(f"[{video_id}] reuse {len(rows)} keyframes")
            return rows

        existing_images = resolve_video_image_dir(video_id, self.root / "data")
        occupied = [path for kind, path in paths.items() if kind != "preparation" and path.exists()]
        if existing_images.exists() or occupied:
            raise FileExistsError(f"Existing artifacts for {video_id}; refusing to replace their frame numbering")

        staging_root = self.root / "preparation"
        staging_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{video_id}-", dir=staging_root) as temporary_dir:
            staged_dir = Path(temporary_dir)
            staged_images = staged_dir / "images"
            staged_images.mkdir()
            rows = []
            image_hashes = {}
            decoded_frames = 0
            with av.open(str(video_path)) as container:
                if not container.streams.video:
                    raise ValueError(f"No video stream: {video_path}")
                stream = container.streams.video[0]
                stream.codec_context.thread_count = 1
                fps = float(stream.average_rate or stream.guessed_rate or 0)
                if not math.isfinite(fps) or fps <= 0:
                    raise ValueError(f"Cannot determine FPS: {video_path}")
                first_pts = None
                last_selected_time = -math.inf
                last_checked_time = -math.inf
                previous_thumbnail = None
                rng = random.Random(f"{config.random_seed}:{video_id}")
                next_frame_idx = 0
                for frame_idx, frame in enumerate(container.decode(stream)):
                    if frame.pts is not None and frame.time_base is not None:
                        pts = float(frame.pts * frame.time_base)
                        if first_pts is None:
                            first_pts = pts - frame_idx / fps
                        pts_time = pts - first_pts
                    else:
                        pts_time = frame_idx / fps
                    if config.max_duration_seconds is not None and pts_time >= config.max_duration_seconds:
                        break
                    decoded_frames += 1
                    if config.frame_gap_min is not None:
                        if frame_idx != next_frame_idx:
                            continue
                    else:
                        scene_change = False
                        if pts_time - last_checked_time >= config.scene_check_seconds:
                            thumbnail = frame.reformat(width=64, height=36, format="gray").to_ndarray().astype(np.float32)
                            if previous_thumbnail is not None:
                                difference = float(np.abs(thumbnail - previous_thumbnail).mean() / 255.0)
                                scene_change = difference >= config.scene_threshold
                            previous_thumbnail = thumbnail
                            last_checked_time = pts_time
                        due = not rows or pts_time - last_selected_time >= config.interval_seconds
                        scene_change = scene_change and pts_time - last_selected_time >= config.min_scene_gap_seconds
                        if not due and not scene_change:
                            continue
                    n = len(rows) + 1
                    image_path = staged_images / f"{n:03d}.jpg"
                    with frame.to_image() as image:
                        image.save(image_path, format="JPEG", quality=config.jpeg_quality)
                    image_hashes[image_path.name] = file_sha256(image_path)
                    rows.append({"n": n, "pts_time": pts_time, "fps": fps, "frame_idx": frame_idx})
                    last_selected_time = pts_time
                    if config.frame_gap_min is not None:
                        next_frame_idx = frame_idx + rng.randint(config.frame_gap_min, config.frame_gap_max)
            if not rows:
                raise ValueError(f"No decodable frames: {video_path}")

            staged_mapping = staged_dir / "mapping.csv"
            with staged_mapping.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["n", "pts_time", "fps", "frame_idx"])
                writer.writeheader()
                writer.writerows(rows)
            state = {
                "schema_version": 1,
                "video_id": video_id,
                "source": source,
                "keyframes": {
                    "config": asdict(config), "decoded_frames": decoded_frames,
                    "mapping_sha256": file_sha256(staged_mapping), "images": image_hashes,
                },
            }
            staged_state = staged_dir / "state.json"
            write_json(staged_state, state)
            paths["images"].parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_images, paths["images"])
            try:
                publish_staged_files([(paths["mapping"], staged_mapping), (paths["preparation"], staged_state)])
            except Exception:
                import shutil

                shutil.rmtree(paths["images"])
                raise
        print(f"[{video_id}] decoded {decoded_frames} frames -> {len(rows)} keyframes")
        return rows

    def _image_batches(self, video_id, rows, batch_size):
        from PIL import Image

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        image_dir = self.artifact_paths(video_id)["images"]
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            images = []
            try:
                for row in batch_rows:
                    with Image.open(image_dir / f"{row['n']:03d}.jpg") as image:
                        images.append(image.convert("RGB"))
                yield batch_rows, images
            finally:
                for image in images:
                    image.close()

    def _validate_features(self, video_id, count):
        import numpy as np

        paths = self.artifact_paths(video_id)
        state = self._state(video_id)
        metadata = state.get("visual_features", {})
        if metadata.get("model") != VISUAL_CONFIG or metadata.get("sha256") != file_sha256(paths["features"]):
            raise ValueError(f"Visual feature provenance/checksum mismatch: {video_id}")
        if metadata.get("mapping_sha256") != state["keyframes"]["mapping_sha256"]:
            raise ValueError(f"Visual features use a different mapping: {video_id}")
        features = np.load(paths["features"], mmap_mode="r", allow_pickle=False)
        if features.shape != (count, int(VISUAL_CONFIG["dimension"])) or features.dtype != np.float32:
            raise ValueError(f"Invalid visual feature shape/dtype: {video_id}: {features.shape}, {features.dtype}")
        if not np.isfinite(features).all() or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-4):
            raise ValueError(f"Non-finite or non-normalized visual features: {video_id}")

    def encode_visual_features(self, video_ids, batch_size=2, encoder=None):
        import numpy as np

        for video_id in video_ids:
            rows = self.mapping_rows(video_id)
            paths = self.artifact_paths(video_id)
            if paths["features"].is_file():
                self._validate_features(video_id, len(rows))
                print(f"[{video_id}] reuse visual features")
                continue
            if encoder is None:
                from siglip_encoder import SiglipEncoder

                encoder = SiglipEncoder()
            batches = []
            for _, images in self._image_batches(video_id, rows, batch_size):
                vectors = np.asarray(encoder.encode_images(images), dtype=np.float32)
                if vectors.shape != (len(images), int(VISUAL_CONFIG["dimension"])):
                    raise ValueError(f"Encoder returned invalid visual shape: {vectors.shape}")
                if not np.isfinite(vectors).all():
                    raise ValueError("Encoder returned non-finite visual features")
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                if np.any(norms <= np.finfo(np.float32).eps):
                    raise ValueError("Encoder returned zero visual features")
                batches.append(vectors / norms)
            features = np.concatenate(batches)
            paths["features"].parent.mkdir(parents=True, exist_ok=True)
            from artifact_io import temporary_path

            staged_features = temporary_path(paths["features"])
            staged_state = temporary_path(paths["preparation"])
            try:
                with staged_features.open("wb") as file:
                    np.save(file, features, allow_pickle=False)
                state = self._state(video_id)
                state["visual_features"] = {
                    "model": dict(VISUAL_CONFIG), "sha256": file_sha256(staged_features),
                    "mapping_sha256": state["keyframes"]["mapping_sha256"],
                }
                write_json(staged_state, state)
                publish_staged_files([(paths["features"], staged_features), (paths["preparation"], staged_state)])
            finally:
                staged_features.unlink(missing_ok=True)
                staged_state.unlink(missing_ok=True)
            self._validate_features(video_id, len(rows))
            print(f"[{video_id}] visual features {features.shape} float32")

    def _object_config(self, threshold):
        return {
            **OBJECT_MODEL, "score_threshold": threshold,
            "object_format": "provider-openimages-v1", "max_detections": 100,
            "label_map_sha256": file_sha256(LABEL_MAP_PATH),
        }

    def _validate_object_checkpoint(self, video_id, row, record, state, detector_config):
        validate_object_schema(record)
        filename = f"{row['n']:03d}.json"
        metadata = state.get("object_records", {}).get(filename)
        if metadata is None:
            raise ValueError(f"Missing object checkpoint: {video_id}/{filename}")
        identity = {"video_id": video_id, "keyframe_n": row["n"], "frame_id": row["frame_idx"]}
        if any(metadata.get(field) != value for field, value in identity.items()):
            raise ValueError(f"Object identity mismatch: {identity}")
        if metadata.get("detector") != detector_config:
            raise ValueError(f"Object detector configuration changed: {identity}")
        if metadata.get("image_sha256") != state["keyframes"]["images"][f"{row['n']:03d}.jpg"]:
            raise ValueError(f"Object source image mismatch: {identity}")
        if metadata.get("sha256") != file_sha256(self.artifact_paths(video_id)["objects"] / filename):
            raise ValueError(f"Object checksum mismatch: {identity}")

    def _publish_object_record(self, video_id, row, record, state, detector_config):
        validate_object_schema(record)
        paths = self.artifact_paths(video_id)
        filename = f"{row['n']:03d}.json"
        destination = paths["objects"] / filename
        staged_object = temporary_path(destination)
        staged_state = temporary_path(paths["preparation"])
        try:
            write_json(staged_object, record, indent=None)
            metadata = {
                "video_id": video_id, "keyframe_n": row["n"], "frame_id": row["frame_idx"],
                "image_sha256": state["keyframes"]["images"][f"{row['n']:03d}.jpg"],
                "detector": detector_config, "sha256": file_sha256(staged_object),
            }
            state.setdefault("object_records", {})[filename] = metadata
            state["objects"] = detector_config
            write_json(staged_state, state)
            publish_staged_files([(destination, staged_object), (paths["preparation"], staged_state)])
        finally:
            staged_object.unlink(missing_ok=True)
            staged_state.unlink(missing_ok=True)

    def generate_objects(self, video_ids, batch_size=2, threshold=0.2, detector=None):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        detector_config = self._object_config(threshold)
        legacy_config = {**OBJECT_MODEL, "score_threshold": threshold}
        for video_id in video_ids:
            rows = self.mapping_rows(video_id)
            paths = self.artifact_paths(video_id)
            state = self._state(video_id)
            pending = []
            converted = 0
            for row in rows:
                n = row["n"]
                path = paths["objects"] / f"{n:03d}.json"
                if path.is_file():
                    record = json.loads(path.read_text(encoding="utf-8"))
                    if "detector" in record:
                        _validate_legacy_object_record(
                            record, {"video_id": video_id, "keyframe_n": n, "frame_id": row["frame_idx"]},
                            legacy_config, state["keyframes"]["images"][f"{n:03d}.jpg"],
                        )
                        record = provider_object_record(record)
                        self._publish_object_record(video_id, row, record, state, detector_config)
                        converted += 1
                    self._validate_object_checkpoint(video_id, row, record, state, detector_config)
                else:
                    pending.append(row)
            if pending and detector is None:
                detector = CPUObjectDetector()
            for batch_rows, images in self._image_batches(video_id, pending, batch_size):
                detections = detector.detect(images, threshold=threshold)
                if len(detections) != len(batch_rows):
                    raise ValueError("Detector output count does not match input count")
                for row, detection in zip(batch_rows, detections):
                    record = provider_object_record(detection)
                    self._publish_object_record(video_id, row, record, state, detector_config)
            state["objects"] = detector_config
            self._save_state(video_id, state)
            print(f"[{video_id}] objects complete for {len(rows)} keyframes ({len(pending)} new, {converted} format conversions)")

    def migrate_generated_objects(self, video_ids):
        """Convert only this pipeline's old object JSONs without running inference."""
        for video_id in video_ids:
            rows = self.mapping_rows(video_id)
            paths = self.artifact_paths(video_id)
            if any(not (paths["objects"] / f"{row['n']:03d}.json").is_file() for row in rows):
                raise FileNotFoundError(f"Objects are incomplete for {video_id}; run the objects stage first")
            producer = self._state(video_id).get("objects", {})
            if any(producer.get(field) != value for field, value in OBJECT_MODEL.items()):
                raise ValueError(f"Cannot migrate objects with unknown model provenance: {video_id}")
            self.generate_objects([video_id], batch_size=1, threshold=producer["score_threshold"])

    def _caption_records(self, video_id, require_complete=True):
        rows = self.mapping_rows(video_id)
        mapping = {row["n"]: row["frame_idx"] for row in rows}
        path = self.artifact_paths(video_id)["captions"]
        records = json.loads(path.read_text(encoding="utf-8"))
        records = [normalize_caption_record(record, video_id, mapping) for record in records]
        validate_unique_caption_records(records, video_id)
        if require_complete:
            if {record["keyframe_n"] for record in records} != set(mapping):
                raise ValueError(f"Caption coverage is incomplete: {video_id}")
            if any(not (record.get("retrieval_text") or record.get("caption")) for record in records):
                raise ValueError(f"Empty searchable captions: {video_id}")
        return records

    def generate_captions(self, video_ids, batch_size=1):
        from caption_generator import generate_captions_with_mapping

        video_ids = list(video_ids)
        for video_id in video_ids:
            self.mapping_rows(video_id)
            state = self._state(video_id)
            previous_config = state.get("caption_generation")
            if previous_config is not None and previous_config != CAPTION_GENERATION_CONFIG:
                raise ValueError(f"Caption model changed for {video_id}")
            if self.artifact_paths(video_id)["captions"].is_file():
                if previous_config is None:
                    raise ValueError(f"Cannot resume captions with unknown model provenance: {video_id}")
                self._caption_records(video_id, require_complete=False)
            state["caption_generation"] = dict(CAPTION_GENERATION_CONFIG)
            self._save_state(video_id, state)
        generate_captions_with_mapping(
            image_root=self.root / "data", csv_root=self.root / "mapping",
            output_dir=self.root / "caption_generator", video_ids=video_ids,
            batch_size=batch_size, num_workers=0, device="cpu", path_root=self.root,
        )
        for video_id in video_ids:
            self._caption_records(video_id)
        gc.collect()

    def build_caption_indexes(self, video_ids, batch_size=8):
        from build_caption_index import build_caption_index

        video_ids = list(video_ids)
        for video_id in video_ids:
            self._caption_records(video_id)
        result = build_caption_index(video_ids=video_ids, project_root=self.root, device="cpu", batch_size=batch_size)
        gc.collect()
        return result

    def validate_video(self, video_id):
        import faiss
        import numpy as np

        rows = self.mapping_rows(video_id)
        self._validate_features(video_id, len(rows))
        paths = self.artifact_paths(video_id)
        state = self._state(video_id)
        for row in rows:
            n = row["n"]
            record = json.loads((paths["objects"] / f"{n:03d}.json").read_text(encoding="utf-8"))
            self._validate_object_checkpoint(video_id, row, record, state, state["objects"])
        captions = self._caption_records(video_id)
        caption_mapping = json.loads(paths["caption_mapping"].read_text(encoding="utf-8"))
        index = faiss.read_index(str(paths["index"]))
        if len(captions) != len(caption_mapping) or len(captions) != index.ntotal:
            raise ValueError(f"Caption/index count mismatch: {video_id}")
        for caption, mapped in zip(captions, caption_mapping):
            if any(caption[field] != mapped[field] for field in ("video_id", "keyframe_n", "frame_id", "image_path")):
                raise ValueError(f"Caption/index identity mismatch: {video_id}")
        validate_caption_manifest(video_id, paths["captions"], paths["caption_mapping"], paths["index"],
                                  paths["manifest"], index, len(captions))
        vectors = index.reconstruct_n(0, index.ntotal)
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise ValueError(f"Caption index is not normalized: {video_id}")
        return {"video_id": video_id, "keyframes": len(rows), "objects": len(rows),
                "visual_shape": (len(rows), int(VISUAL_CONFIG["dimension"])), "captions": len(captions),
                "caption_vectors": index.ntotal, "status": "OK"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=KIS_ROOT / "video")
    parser.add_argument("--output-root", type=Path, default=KIS_ROOT)
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--frame-range", type=int, nargs=2, metavar=("MIN", "MAX"),
                          help="random inclusive frame-gap range (default: 80 160)")
    sampling.add_argument("--interval", type=float, help="use legacy time/scene sampling instead of random frame gaps")
    parser.add_argument("--seed", type=int, default=42, help="repeatable frame-sampling seed (default: 42)")
    parser.add_argument("--max-duration", type=float, help="limit decoding for an isolated smoke run")
    parser.add_argument("--stages", nargs="+", choices=["keyframes", "features", "objects", "object-format", "captions", "index", "validate"],
                        default=["keyframes", "features", "objects", "captions", "index", "validate"])
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_duration is not None and args.output_root.resolve() == KIS_ROOT.resolve():
        parser.error("--max-duration requires a separate --output-root to avoid publishing partial videos")
    print(configure_cpu(args.threads))
    videos = discover_videos(args.video_dir, args.video_ids)
    builder = VideoDatasetBuilder(args.output_root)
    if "keyframes" in args.stages:
        gap_min, gap_max = args.frame_range if args.frame_range is not None else (80, 160)
        config = KeyframeConfig(
            frame_gap_min=None if args.interval is not None else gap_min,
            frame_gap_max=gap_max, random_seed=args.seed,
            interval_seconds=args.interval if args.interval is not None else 2.0,
            max_duration_seconds=args.max_duration,
        )
        for path in videos.values():
            builder.extract_keyframes(path, config)
    if "features" in args.stages:
        builder.encode_visual_features(videos, batch_size=args.batch_size)
    if "objects" in args.stages:
        builder.generate_objects(videos, batch_size=args.batch_size)
    if "object-format" in args.stages:
        builder.migrate_generated_objects(videos)
    if "captions" in args.stages:
        builder.generate_captions(videos, batch_size=1)
    if "index" in args.stages:
        builder.build_caption_indexes(videos)
    if "validate" in args.stages:
        for video_id in videos:
            print(builder.validate_video(video_id))


if __name__ == "__main__":
    main()
