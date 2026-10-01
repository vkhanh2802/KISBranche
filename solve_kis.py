from collections import defaultdict
from apply_object_filter import apply_object_filter 

def multi_query_search(query_variants, encoder, retriever, top_k=50):
    """Tìm kiếm vector FAISS (Visual/CLIP)"""
    print(f"[Visual Search] {len(query_variants)} query variants")
    all_results = []
    for query in query_variants:
        vector = encoder.encode_text(query)
        results = retriever.search(vector, top_k=top_k)
        all_results.append(results)
    return all_results

def reciprocal_rank_fusion(multi_results, k=60):
    """Gộp các biến thể truy vấn của CÙNG 1 luồng"""
    scores = defaultdict(float)
    metadata = {}
    for results in multi_results:
        for rank, item in enumerate(results):
            key = (item["video_id"], item["frame_id"])
            scores[key] += 1.0 / (k + rank + 1)
            metadata[key] = item

    fused = []
    for key, score in scores.items():
        item = metadata[key].copy()
        item["score"] = score
        fused.append(item)
    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused

def normalize_scores(results):
    """Bắt buộc chuẩn hóa điểm số về [0, 1] trước khi Fusion tĩnh"""
    if not results: return []
    scores = [x["score"] for x in results]
    min_score, max_score = min(scores), max(scores)
    for item in results:
        if max_score == min_score:
            item["norm_score"] = 1.0
        else:
            item["norm_score"] = (item["score"] - min_score) / (max_score - min_score)
    return results

def static_weight_fusion_3_way(visual_results, bm25_results, semantic_results, w_visual=0.5, w_bm25=0.2, w_semantic=0.3):
    """Dung hợp 3 luồng: Hình ảnh (CLIP/SigLIP), Từ khóa (BM25) và Ngữ nghĩa (Caption)"""
    print(f"[Fusion] Trọng số: Visual={w_visual}, BM25={w_bm25}, Semantic={w_semantic}")
    candidates = {}

    def add_scores(results, score_key):
        for item in results:
            key = (item["video_id"], item["frame_id"])
            if key not in candidates:
                candidates[key] = {
                    "visual_score": 0.0,
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                }

            candidate = candidates[key]
            for field, value in item.items():
                if field in {"score", "norm_score"}:
                    continue
                if field not in candidate or candidate[field] in (None, ""):
                    candidate[field] = value

            candidate[score_key] = item.get("norm_score", 0.0)

    # Nạp điểm từ 3 luồng
    add_scores(visual_results, "visual_score")
    add_scores(bm25_results, "bm25_score")
    add_scores(semantic_results, "semantic_score")

    # Tính điểm tổng (Fusion Score)
    results = []
    for item in candidates.values():
        item["fusion_score"] = (item["visual_score"] * w_visual) + \
                               (item["bm25_score"] * w_bm25) + \
                               (item["semantic_score"] * w_semantic)
        item["score"] = item["fusion_score"] # Cập nhật score chính
        results.append(item)

    results.sort(key=lambda x: x["fusion_score"], reverse=True)
    return results

def solve_kis(structured_query, visual_retriever, visual_encoder, text_bm25_retriever, text_semantic_retriever, object_lookup, vlm_reranker=None):
    # 1. Bóc tách dữ liệu
    query_variants = structured_query.get("query_variants") or [structured_query.get("raw_query", "")] 
    entities = structured_query.get("entities", []) 

    # 2. Luồng 1: Visual Search (SigLIP/CLIP)
    visual_multi = multi_query_search(query_variants, visual_encoder, visual_retriever, top_k=100)
    visual_results = normalize_scores(reciprocal_rank_fusion(visual_multi))

    # 3. Luồng 2: Text Search (BM25 - Khớp từ khóa cứng)
    bm25_multi = []
    for q in query_variants:
        print(f"[BM25 Search] Đang search: {q}")
        bm25_multi.append(text_bm25_retriever.search_bm25(q, top_k=100)) 
    bm25_results = normalize_scores(reciprocal_rank_fusion(bm25_multi))

    # 4. Luồng 3: Semantic Text Search (Tìm kiếm ngữ nghĩa Vector)
    semantic_multi = []
    for q in query_variants:
        print(f"[Semantic Search] Đang search: {q}")
        semantic_multi.append(text_semantic_retriever.search(q, top_k=100))
    semantic_results = normalize_scores(reciprocal_rank_fusion(semantic_multi))

    # 5. Dung hợp 3 luồng
    candidates = static_weight_fusion_3_way(
        visual_results, 
        bm25_results, 
        semantic_results, 
        w_visual=0.5, 
        w_bm25=0.2, 
        w_semantic=0.3
    )

    # 6. Bộ lọc vật thể cứng
    candidates = apply_object_filter(candidates, entities, object_lookup) 

    
    # Xuất kết quả
    results = [(x["video_id"], x["frame_id"], x["score"]) for x in candidates[:100]] 
    print(f"[KIS] Hoàn tất. Trả về {len(results)} kết quả.")
    return results
