import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from paths import (
    CAPTION_DIR,
    CAPTION_MAPPING_DIR,
    INDEX_DIR,
    video_caption_index_path,
    video_caption_mapping_path,
    video_caption_path,
)


class CaptionRetriever:

    @staticmethod
    def _artifact_for_video(base_dir: Path, suffix: str, video_id: str) -> Path:
        """Resolve artifact path for a video in either nested or flat layout."""
        group = video_id.split("_")[0]
        nested = base_dir / group / f"{video_id}{suffix}"
        if nested.exists():
            return nested

        flat = base_dir / f"{video_id}{suffix}"
        if flat.exists():
            return flat

        matches = list(base_dir.rglob(f"{video_id}{suffix}"))
        if matches:
            return matches[0]

        return nested

    @staticmethod
    def _collect_artifact_map(base_dir: Path, pattern: str):
        files = sorted(path for path in base_dir.rglob(pattern) if path.is_file())
        return {path.stem: path for path in files}

    def __init__(
        self,
        video_id=None,
        json_path=None,
        index_path=None,
        mapping_path=None
    ):
        if video_id is not None:
            # Support both nested layout (L21/L21_V001.json) and flat layout.
            json_path = self._artifact_for_video(CAPTION_DIR, ".json", video_id)
            index_path = self._artifact_for_video(INDEX_DIR, ".index", video_id)
            mapping_path = self._artifact_for_video(CAPTION_MAPPING_DIR, ".json", video_id)

        self.video_id = video_id

        print("[Caption] Loading SentenceTransformer...")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )

        if video_id is None and json_path is None:
            caption_by_video = self._collect_artifact_map(CAPTION_DIR, "L*_V*.json")
            index_by_video = self._collect_artifact_map(INDEX_DIR, "L*_V*.index")
            mapping_by_video = self._collect_artifact_map(CAPTION_MAPPING_DIR, "L*_V*.json")

            common_video_ids = sorted(
                set(caption_by_video) & set(index_by_video) & set(mapping_by_video)
            )

            caption_data = []
            indexes = []
            mapping = []

            for vid in common_video_ids:
                caption_file = caption_by_video[vid]
                index_file = index_by_video[vid]
                mapping_file = mapping_by_video[vid]
                with open(caption_file, "r", encoding="utf-8") as f:
                    caption_data.extend(json.load(f))
                indexes.append(faiss.read_index(str(index_file)))
                with open(mapping_file, "r", encoding="utf-8") as f:
                    mapping.extend(json.load(f))

            if not indexes:
                raise FileNotFoundError("Khong tim thay caption/index theo video")

            missing_caption = sorted(set(index_by_video) - set(caption_by_video))
            missing_index = sorted(set(caption_by_video) - set(index_by_video))
            missing_mapping = sorted(set(caption_by_video) - set(mapping_by_video))
            if missing_caption or missing_index or missing_mapping:
                print(
                    "[Warning] Mot so video bi thieu artifact: "
                    f"missing_caption={len(missing_caption)}, "
                    f"missing_index={len(missing_index)}, "
                    f"missing_mapping={len(missing_mapping)}"
                )

            self.index = faiss.IndexFlatIP(indexes[0].d)
            for index in indexes:
                vectors = index.reconstruct_n(0, index.ntotal)
                self.index.add(x=vectors)
            self.caption_data = caption_data
            self.mapping = mapping
            self.json_path = [caption_by_video[vid] for vid in common_video_ids]
            self.index_path = [index_by_video[vid] for vid in common_video_ids]
            self.mapping_path = [mapping_by_video[vid] for vid in common_video_ids]
        else:
            self.json_path = json_path
            self.index_path = index_path
            self.mapping_path = mapping_path
            if json_path is None or index_path is None or mapping_path is None:
                raise ValueError("Caption paths must be provided")
            with open(json_path, "r", encoding="utf-8") as f:
                self.caption_data = json.load(f)
            self.index = faiss.read_index(str(index_path))
            with open(mapping_path, "r", encoding="utf-8") as f:
                self.mapping = json.load(f)

        print(
            f"[Caption] Loaded {self.index.ntotal} caption vectors"
        )

        if self.index.ntotal != len(self.caption_data):
            print(
                "[Warning] Number of vectors and JSON records "
                "do not match!"
            )

    def encode_query(self, query):
        """
        Encode user query using the same
        SentenceTransformer used when building the index.
        """

        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        return vector

    def search(self, query, top_k=50):

        query_vector = self.encode_query(query)

        top_k = min(top_k, self.index.ntotal)

        scores, ids = self.index.search(x=query_vector, k=top_k)

        results = []

        for score, idx in zip(scores[0], ids[0]):

            if idx == -1:
                continue
            if isinstance(self.mapping, dict):
                item = self.mapping.get(str(idx))

                if item is None:
                    continue

            else:
                if idx >= len(self.mapping):
                    continue

                item = self.mapping[idx]

            result = {
                "video_id": item["video_id"],
                "frame_id": int(item["frame_id"]),
                "score": float(score)
            }
            for field in ("keyframe_n", "image_path"):
                if item.get(field) is not None:
                    result[field] = item[field]

            if idx < len(self.caption_data):
                caption_item = self.caption_data[idx]
                if (
                    caption_item.get("video_id") == result["video_id"]
                    and int(caption_item.get("frame_id", -1)) == result["frame_id"]
                ):
                    result["retrieval_text"] = caption_item.get(
                        "retrieval_text",
                        caption_item.get("caption", ""),
                    )

            results.append(result)

        return results

    def search_with_text(self, query, top_k=20):

        results = self.search(query, top_k)

        output = []

        for r in results:

            video_id = r["video_id"]
            frame_id = r["frame_id"]

            text = None

            for item in self.caption_data:

                if (
                    item["video_id"] == video_id
                    and int(item["frame_id"]) == frame_id
                ):
                    text = item.get(
                        "retrieval_text",
                        item.get("caption", "")
                    )
                    break

            output.append({
                "video_id": video_id,
                "frame_id": frame_id,
                "score": r["score"],
                "retrieval_text": text
            })

        return output

