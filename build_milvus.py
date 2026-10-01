import argparse
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from pymilvus import MilvusClient, DataType

from paths import CLIP_FEATURES_DIR, MAPPING_DIR

CLIP_DIR = CLIP_FEATURES_DIR
CSV_DIR = MAPPING_DIR

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "clip_keyframes"
EMBEDDING_DIM = 768
VALID_MODES = ("build", "rebuild")


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def create_collection(client: MilvusClient, collection_name: str):
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="video_id", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="frame_id", datatype=DataType.INT64)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 128},
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    print(f"Created collection: {collection_name}")


def validate_video_input(npy_path: Path) -> int:
    video_id = npy_path.stem
    csv_path = CSV_DIR / f"{video_id}.csv"

    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    features = np.load(npy_path, mmap_mode="r", allow_pickle=False)
    if features.ndim != 2 or features.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"{video_id}: invalid feature shape {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"{video_id}: features contain NaN or infinity")

    norms = np.linalg.norm(features, axis=1)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise ValueError(f"{video_id}: features contain zero vectors")

    df = pd.read_csv(csv_path)
    if "frame_idx" not in df.columns:
        raise ValueError(f"{csv_path} is missing frame_idx")
    if len(features) != len(df):
        raise ValueError(
            f"{video_id}: {len(features)} vectors != {len(df)} frames"
        )

    frame_ids = pd.to_numeric(df["frame_idx"], errors="raise").to_numpy()
    if not np.isfinite(frame_ids).all() or not np.equal(
        frame_ids, np.floor(frame_ids)
    ).all():
        raise ValueError(f"{video_id}: frame_idx must contain finite integers")
    if len(np.unique(frame_ids)) != len(frame_ids):
        raise ValueError(f"{video_id}: frame_idx contains duplicates")

    return len(features)


def preflight_inputs():
    if not CLIP_DIR.is_dir():
        raise FileNotFoundError(f"Feature directory does not exist: {CLIP_DIR}")
    if not CSV_DIR.is_dir():
        raise FileNotFoundError(f"Mapping directory does not exist: {CSV_DIR}")

    npy_files = sorted(path for path in CLIP_DIR.glob("*.npy") if path.is_file())
    if not npy_files:
        raise RuntimeError(f"No .npy files found in {CLIP_DIR}")

    total = 0
    for npy_path in npy_files:
        total += validate_video_input(npy_path)

    print(f"Preflight complete: {len(npy_files)} videos, {total} vectors")
    return npy_files, total


def get_active_collection(client: MilvusClient):
    for collection_name in client.list_collections():
        aliases = client.list_aliases(collection_name=collection_name)
        if isinstance(aliases, dict):
            aliases = aliases.get("aliases", [])
        if COLLECTION_NAME in aliases:
            return collection_name, True

    if client.has_collection(COLLECTION_NAME):
        return COLLECTION_NAME, False

    return None, False


def publish_collection(
    client: MilvusClient,
    staging_name: str,
    active_name: str | None,
    active_uses_alias: bool,
):
    if active_uses_alias:
        client.alter_alias(collection_name=staging_name, alias=COLLECTION_NAME)
        return active_name

    if active_name is None:
        client.create_alias(collection_name=staging_name, alias=COLLECTION_NAME)
        return None

    backup_name = f"{COLLECTION_NAME}__backup_{uuid4().hex[:8]}"
    client.rename_collection(old_name=active_name, new_name=backup_name)
    try:
        client.create_alias(collection_name=backup_name, alias=COLLECTION_NAME)
    except Exception:
        client.rename_collection(old_name=backup_name, new_name=active_name)
        raise

    client.alter_alias(collection_name=staging_name, alias=COLLECTION_NAME)

    return backup_name


def insert_video(
    client: MilvusClient,
    collection_name: str,
    npy_path: Path,
) -> int:
    video_id = npy_path.stem
    csv_path = CSV_DIR / f"{video_id}.csv"
    features = np.load(npy_path, allow_pickle=False).astype(np.float32)
    df = pd.read_csv(csv_path)
    features = normalize(features)
    frame_ids = df["frame_idx"].astype(int).tolist()

    rows = [
        {
            "video_id": video_id,
            "frame_id": frame_ids[i],
            "embedding": features[i].tolist(),
        }
        for i in range(len(features))
    ]

    result = client.insert(collection_name=collection_name, data=rows)
    inserted = int(result["insert_count"])
    if inserted != len(rows):
        raise RuntimeError(
            f"{video_id}: Milvus inserted {inserted}/{len(rows)} vectors"
        )

    print(f"{video_id}: {inserted} vectors")
    return inserted


def run(mode: str):
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of: {', '.join(VALID_MODES)}")

    npy_files, expected_total = preflight_inputs()
    client = MilvusClient(uri=MILVUS_URI)
    active_name, active_uses_alias = get_active_collection(client)
    if mode == "build" and active_name is not None:
        raise RuntimeError(
            f"{COLLECTION_NAME} already exists; use rebuild to replace it safely"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    staging_name = (
        f"{COLLECTION_NAME}__staging_{timestamp}_{uuid4().hex[:8]}"
    )
    published = False

    try:
        create_collection(client, staging_name)
        total = sum(
            insert_video(client, staging_name, npy_path)
            for npy_path in npy_files
        )
        client.flush(staging_name)

        if total != expected_total:
            raise RuntimeError(
                f"Staging count mismatch: {total} != {expected_total}"
            )

        stats = client.get_collection_stats(collection_name=staging_name)
        row_count = int(stats.get("row_count", -1))
        if row_count != expected_total:
            raise RuntimeError(
                f"Milvus row count mismatch: {row_count} != {expected_total}"
            )

        previous_name = publish_collection(
            client,
            staging_name,
            active_name,
            active_uses_alias,
        )
        published = True

        if previous_name is not None:
            try:
                client.drop_collection(previous_name)
            except Exception as exc:
                print(
                    f"Warning: active collection was switched, but cleanup "
                    f"failed for {previous_name}: {exc}"
                )

    finally:
        if not published and client.has_collection(staging_name):
            try:
                client.drop_collection(staging_name)
            except Exception as exc:
                print(f"Warning: failed to clean staging collection: {exc}")

    print(f"\n{mode.capitalize()} complete")
    print(f"Published: {len(npy_files)} videos ({expected_total} vectors)")
    print(f"Active alias: {COLLECTION_NAME} -> {staging_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Build the SigLIP keyframe collection safely."
    )
    parser.add_argument(
        "mode",
        choices=VALID_MODES,
        help="build creates a new collection; rebuild safely replaces it",
    )
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
