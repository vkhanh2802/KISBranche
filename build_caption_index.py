import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from paths import (
    CAPTION_DIR,
    CAPTION_MAPPING_DIR,
    INDEX_DIR,
    video_caption_index_path,
    video_caption_mapping_path,
)


def build_caption_index():

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    caption_files = sorted(
        path for path in CAPTION_DIR.glob("L*.json")
        if path.is_file()
    )
    data = []
    for caption_file in caption_files:
        with open(caption_file, "r", encoding="utf-8") as f:
            video_data = json.load(f)
        data.extend(video_data)

    print(f"[Caption] Found {len(data)} records in {len(caption_files)} files")

    if len(data) == 0:
        raise ValueError("captions.json is empty")

    print("[Caption] Loading SentenceTransformer...")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    texts = []

    for item in data:

        text = item.get("retrieval_text", "")

        if not text:
            text = item.get("caption", "")

        texts.append(text)

    print("[Caption] Example retrieval text:")
    print(texts[0])
    print("[Caption] Encoding retrieval_text...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype(np.float32)

    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]

    mapping = []

    for item in data:

        mapping.append({
            "video_id": item["video_id"],
            "frame_id": int(item["frame_id"]),
            "keyframe_n": int(item["keyframe_n"]),
            "image_path": item.get("image_path", "")
        })

    CAPTION_MAPPING_DIR.mkdir(parents=True, exist_ok=True)

    video_positions = {
        caption_file.stem: [position for position, item in enumerate(data)
                            if item["video_id"] == caption_file.stem]
        for caption_file in caption_files
    }

    for video_id, positions in sorted(video_positions.items()):
        video_index = faiss.IndexFlatIP(dim)
        video_index.add(embeddings[positions])
        faiss.write_index(video_index, str(video_caption_index_path(video_id)))

        video_mapping = [mapping[position] for position in positions]
        with open(video_caption_mapping_path(video_id), "w", encoding="utf-8") as f:
            json.dump(video_mapping, f, ensure_ascii=False, indent=2)

        print(
            f"[{video_id}] index={video_caption_index_path(video_id)} "
            f"mapping={video_caption_mapping_path(video_id)} "
            f"records={len(positions)}"
        )

    print(f"Records      : {len(data)}")
    print(f"Index dim    : {dim}")
    print(f"Index directory   : {INDEX_DIR}")
    print(f"Mapping directory : {CAPTION_MAPPING_DIR}")
    print("======================================")


if __name__ == "__main__":
    build_caption_index()

