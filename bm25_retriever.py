from rank_bm25 import BM25Okapi
import numpy as np

class BM25Retriever:
    def __init__(self, caption_data):
        """
        Nhận vào danh sách caption_data đã được nạp từ các file JSON.
        Định dạng mỗi phần tử: {"video_id": "...", "frame_id": 123, "retrieval_text": "..."}
        """
        self.caption_data = caption_data
        print(f"[*] Initializing BM25 for {len(caption_data)} frames...")
        
        # Tách từ đơn giản (Tokenization) bằng khoảng trắng và chuyển chữ thường
        self.corpus_tokens = [
            str(item.get("retrieval_text", "")).lower().split() 
            for item in self.caption_data
        ]
        
        # Khởi tạo thuật toán BM25
        self.bm25 = BM25Okapi(self.corpus_tokens)
        print("[*] BM25 indexing complete.")

    def search_bm25(self, query, top_k=100):
        # Tách từ câu truy vấn
        query_tokens = query.lower().split()
        
        # Lấy điểm số BM25 cho toàn bộ tập dữ liệu
        scores = self.bm25.get_scores(query_tokens)
        
        # Lấy index của top_k điểm cao nhất
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0: # Chỉ lấy những frame có ít nhất 1 từ khóa trùng khớp
                item = self.caption_data[idx]
                result = {
                    "video_id": item["video_id"],
                    "frame_id": item["frame_id"],
                    "score": score,
                    "retrieval_text": item.get("retrieval_text", "")
                }
                for field in ("keyframe_n", "image_path"):
                    if item.get(field) is not None:
                        result[field] = item[field]
                results.append(result)
                
        return results
