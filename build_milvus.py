import os
import sys
import numpy as np
import pandas as pd
from pymilvus import MilvusClient, DataType
from paths import CLIP_FEATURES_DIR, MAPPING_DIR

CLIP_DIR = CLIP_FEATURES_DIR
CSV_DIR = MAPPING_DIR

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "clip_keyframes"
EMBEDDING_DIM = 768


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return vectors / norms


def create_collection(client: MilvusClient, fresh: bool = False):
    if client.has_collection(COLLECTION_NAME):
        if fresh:
            client.drop_collection(COLLECTION_NAME)
        else:
            return

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
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"Created collection: {COLLECTION_NAME}")

def get_existing_videos(client: MilvusClient) -> set:
    existing = set()
    iterator = client.query_iterator(
        collection_name=COLLECTION_NAME,
        filter="",
        output_fields=["video_id"],
        batch_size=1000,
    )
    while True:
        batch = iterator.next()
        if not batch:
            break
        existing.update(row["video_id"] for row in batch)
    iterator.close()
    return existing


def insert_video(client: MilvusClient, npy_file: str) -> int:
    video_id = os.path.splitext(npy_file)[0]
    npy_path = os.path.join(CLIP_DIR, npy_file)
    csv_path = os.path.join(CSV_DIR, video_id + ".csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy CSV: {csv_path}")

    features = np.load(npy_path).astype(np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if features.ndim != 2 or features.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"{video_id}: shape không hợp lệ {features.shape}")

    df = pd.read_csv(csv_path)
    if "frame_idx" not in df.columns:
        raise ValueError(f"{csv_path} thiếu cột frame_idx")
    if len(features) != len(df):
        raise ValueError(
            f"{video_id}: {len(features)} vectors != {len(df)} frames"
        )

    features = normalize(features)
    frame_ids = df["frame_idx"].astype(int).tolist()

    rows = [
        {"video_id": video_id, "frame_id": frame_ids[i], "embedding": features[i].tolist()}
        for i in range(len(features))
    ]

    result = client.insert(collection_name=COLLECTION_NAME, data=rows)
    print(f"{video_id}: {result['insert_count']} vectors")
    return result["insert_count"]


def run(mode: str):
    client = MilvusClient(uri=MILVUS_URI)
    create_collection(client, fresh=(mode == "build"))

    npy_files = sorted(f for f in os.listdir(CLIP_DIR) if f.endswith(".npy"))
    if not npy_files:
        raise RuntimeError(f"Không có file .npy trong {CLIP_DIR}")

    existing_videos = get_existing_videos(client) if mode == "rebuild" else set()

    total, added, skipped = 0, 0, 0
    for npy_file in npy_files:
        video_id = os.path.splitext(npy_file)[0]
        if video_id in existing_videos:
            print(f"Skip: {video_id}")
            skipped += 1
            continue
        total += insert_video(client, npy_file)
        added += 1

    client.flush(COLLECTION_NAME)

    print(f"\n{mode.capitalize()} complete")
    print(f"Added: {added} videos ({total} vectors)")
    if mode == "rebuild":
        print(f"Skipped: {skipped} videos")
