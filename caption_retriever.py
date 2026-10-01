import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from model_config import CAPTION_CONFIG
from paths import (
    CAPTION_DIR,
    CAPTION_MAPPING_DIR,
    INDEX_DIR,
    resolve_video_artifact,
    video_caption_index_path,
    video_caption_mapping_path,
    video_caption_path,
)


class CaptionRetriever:

    @staticmethod
    def _artifact_for_video(base_dir: Path, suffix: str, video_id: str) -> Path:
        """Resolve canonical, flat, or legacy-sharded artifacts."""
        return resolve_video_artifact(base_dir, video_id, suffix)

    @staticmethod
    def _collect_artifact_map(base_dir: Path, pattern: str):
        files = sorted(path for path in base_dir.rglob(pattern) if path.is_file())
        artifacts = {}
        for path in files:
            if path.stem in artifacts:
                raise RuntimeError(
                    f"Duplicate artifacts for {path.stem}: "
                    f"{artifacts[path.stem]}, {path}"
                )
            artifacts[path.stem] = path
        return artifacts

    @staticmethod
    def _load_json_list(path: Path, artifact_name: str):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(f"{artifact_name} must be a JSON array: {path}")
        return data

    @staticmethod
    def _validate_triplet(video_id, captions, mapping, index, paths):
        caption_path, mapping_path, index_path = paths
        expected = len(captions)
        if expected == 0:
            raise ValueError(f"Caption file is empty: {caption_path}")
        if len(mapping) != expected or index.ntotal != expected:
            raise ValueError(
                f"Artifact count mismatch for {video_id}: "
                f"captions={expected}, mapping={len(mapping)}, "
                f"index={index.ntotal} ({index_path})"
            )

        for position, (caption, mapped) in enumerate(zip(captions, mapping)):
            caption_identity = (
                caption.get("video_id"),
                caption.get("frame_id"),
                caption.get("keyframe_n"),
            )
            mapping_identity = (
                mapped.get("video_id"),
                mapped.get("frame_id"),
                mapped.get("keyframe_n"),
            )
            if (
                caption_identity[0] != video_id
                or caption_identity != mapping_identity
            ):
                raise ValueError(
                    f"Artifact identity mismatch for {video_id} at row "
                    f"{position}: {caption_path}, {mapping_path}"
                )

        return any(
            item.get("retrieval_text") or item.get("caption")
            for item in captions
        )

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
            CAPTION_CONFIG["model"],
            revision=CAPTION_CONFIG["revision"],
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
            loaded_video_ids = []

            for vid in common_video_ids:
                caption_file = caption_by_video[vid]
                index_file = index_by_video[vid]
                mapping_file = mapping_by_video[vid]
                video_captions = self._load_json_list(
                    caption_file,
                    "Caption artifact",
                )
                video_mapping = self._load_json_list(
                    mapping_file,
                    "Caption mapping",
                )
                video_index = faiss.read_index(str(index_file))
                has_text = self._validate_triplet(
                    vid,
                    video_captions,
                    video_mapping,
                    video_index,
                    (caption_file, mapping_file, index_file),
                )
                if not has_text:
                    print(f"[Caption] Skipping {vid}: no searchable text")
                    continue

                caption_data.extend(video_captions)
                indexes.append(video_index)
                mapping.extend(video_mapping)
                loaded_video_ids.append(vid)

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
            self.json_path = [caption_by_video[vid] for vid in loaded_video_ids]
            self.index_path = [index_by_video[vid] for vid in loaded_video_ids]
            self.mapping_path = [mapping_by_video[vid] for vid in loaded_video_ids]
        else:
            self.json_path = json_path
            self.index_path = index_path
            self.mapping_path = mapping_path
            if json_path is None or index_path is None or mapping_path is None:
                raise ValueError("Caption paths must be provided")
            self.caption_data = self._load_json_list(
                json_path,
                "Caption artifact",
            )
            self.index = faiss.read_index(str(index_path))
            self.mapping = self._load_json_list(
                mapping_path,
                "Caption mapping",
            )
            if not self._validate_triplet(
                video_id or self.caption_data[0].get("video_id"),
                self.caption_data,
                self.mapping,
                self.index,
                (json_path, mapping_path, index_path),
            ):
                raise ValueError(f"Caption artifact has no searchable text: {json_path}")

        print(
            f"[Caption] Loaded {self.index.ntotal} caption vectors"
        )

        if self.index.ntotal != len(self.mapping):
            raise ValueError("Number of vectors and mapping records do not match")

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

