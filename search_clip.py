import numpy as np
from pymilvus import MilvusClient


MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "clip_keyframes"
EMBEDDING_DIM = 768

class SiglipRetriever:
    def __init__(self):
        self.client = MilvusClient(uri=MILVUS_URI)
        self.collection_name = COLLECTION_NAME
        print(f"Connected to Milvus: {COLLECTION_NAME}")

    def search(self, query_vector, top_k=100): 
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_vector = np.asarray(query_vector, dtype=np.float32)

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        if query_vector.ndim != 2 or query_vector.shape[0] != 1:
            raise ValueError("search expects exactly one query vector")

        # Trình chặn lỗi nếu vector khác 768 chiều
        if query_vector.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Query vector has {query_vector.shape[1]} dimensions; "
                f"expected {EMBEDDING_DIM}"
            )
        if not np.isfinite(query_vector).all():
            raise ValueError("Query vector contains NaN or infinity")

        # Chuẩn hóa L2 (Bắt buộc cho search Inner Product - IP)
        norms = np.linalg.norm(query_vector, axis=1, keepdims=True)
        if np.any(norms <= np.finfo(np.float32).eps):
            raise ValueError("Query vector must not be zero")
        query_vector = query_vector / norms

        results = self.client.search(
            collection_name=self.collection_name,
            data=query_vector.tolist(),
            anns_field="embedding",
            search_params={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["video_id", "frame_id"]
        )

        return [
            {
                "video_id": result["entity"]["video_id"],
                "frame_id": result["entity"]["frame_id"],
                "score": float(result["distance"])
            }
            for result in results[0]
        ]
