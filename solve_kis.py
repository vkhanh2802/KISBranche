from collections import defaultdict
import math
from pathlib import Path
import sys

from apply_object_filter import apply_object_filter
from paths import KIS_ROOT, video_image_dir


def console_safe_text(value, encoding=None):
    encoding = encoding or getattr(sys.stdout, "encoding", None) or "utf-8"
    return str(value).encode(encoding, errors="backslashreplace").decode(encoding)


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
    scores = [float(x["score"]) for x in results]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("Retrieval results contain non-finite scores")
    min_score, max_score = min(scores), max(scores)
    for item in results:
        if max_score == min_score:
            item["norm_score"] = 1.0
        else:
            item["norm_score"] = (item["score"] - min_score) / (max_score - min_score)
    return results


def finalize_vlm_ranking(candidates, top_k=20, vlm_weight=0.5):
    """Combine available VLM scores with retrieval scores and rank globally."""
    if not 0.0 <= vlm_weight <= 1.0:
        raise ValueError("vlm_weight must be between 0 and 1")

    rerank_candidates = candidates[:top_k]
    scored_candidates = []

    for candidate in candidates:
        retrieval_score = float(candidate["score"])
        if not math.isfinite(retrieval_score):
            raise ValueError("Candidate contains a non-finite retrieval score")
        candidate["final_score"] = retrieval_score

    for candidate in rerank_candidates:
        raw_score = candidate.get("vlm_raw_score")
        if isinstance(raw_score, (int, float)) and math.isfinite(raw_score):
            scored_candidates.append((candidate, max(0.0, min(10.0, float(raw_score)))))

    vlm_scores = [score for _, score in scored_candidates]
    if len(vlm_scores) < 2 or max(vlm_scores) - min(vlm_scores) < 1e-6:
        return sorted(candidates, key=lambda x: x["final_score"], reverse=True), False

    for candidate, raw_score in scored_candidates:
        retrieval_score = max(0.0, min(1.0, float(candidate["score"])))
        vlm_delta = (raw_score / 10.0) - 0.5

        if vlm_delta >= 0.0:
            final_score = retrieval_score + (
                2.0 * vlm_weight * vlm_delta * (1.0 - retrieval_score)
            )
        else:
            final_score = retrieval_score + (
                2.0 * vlm_weight * vlm_delta * retrieval_score
            )

        candidate["final_score"] = final_score

    return sorted(candidates, key=lambda x: x["final_score"], reverse=True), True

def static_weight_fusion_3_way(visual_results, bm25_results, semantic_results, w_visual=0.5, w_bm25=0.2, w_semantic=0.3):
    """Dung hợp 3 luồng: Hình ảnh (CLIP/SigLIP), Từ khóa (BM25) và Ngữ nghĩa (Caption)"""
    weights = (w_visual, w_bm25, w_semantic)
    if not all(math.isfinite(weight) and weight >= 0.0 for weight in weights):
        raise ValueError("Fusion weights must be finite and non-negative")
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
        raise ValueError("Fusion weights must sum to 1")
    print(
        f"[Fusion] Weights: Visual={w_visual}, "
        f"BM25={w_bm25}, Semantic={w_semantic}"
    )
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


def _resolve_candidate_image_path(candidate, object_lookup):
    stored_image_path = candidate.get("image_path")
    if stored_image_path:
        image_path = Path(stored_image_path)
        if not image_path.is_absolute():
            image_path = KIS_ROOT / image_path
        if image_path.is_file():
            return image_path

    video_id = candidate["video_id"]
    frame_id = candidate["frame_id"]
    keyframe_n = candidate.get("keyframe_n")
    if keyframe_n is None:
        keyframe_n = object_lookup.get_keyframe_n(video_id, frame_id)

    image_dir = video_image_dir(video_id)
    if keyframe_n is None or not image_dir.is_dir():
        return None

    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            if int(image_path.stem) == int(keyframe_n):
                candidate["keyframe_n"] = int(keyframe_n)
                candidate["image_path"] = str(image_path)
                return image_path
        except ValueError:
            continue

    return None


def rerank_candidates_with_vlm(
    candidates,
    raw_query,
    vlm_reranker,
    object_lookup,
    top_k=20,
    vlm_weight=0.5,
):
    if vlm_reranker is None or top_k <= 0:
        ranked, _ = finalize_vlm_ranking(candidates, top_k=0, vlm_weight=vlm_weight)
        return ranked

    print(f"[VLM] Reranking top {min(top_k, len(candidates))} candidates")
    for candidate in candidates[:top_k]:
        candidate.pop("vlm_error", None)
        image_path = _resolve_candidate_image_path(candidate, object_lookup)
        if image_path is None:
            candidate["vlm_raw_score"] = None
            candidate["vlm_error"] = "image_not_found"
            continue

        try:
            vlm_score = vlm_reranker.score(
                image_path,
                raw_query,
                caption=candidate.get("retrieval_text", ""),
            )
        except Exception as exc:
            candidate["vlm_raw_score"] = None
            candidate["vlm_error"] = str(exc)
            continue

        candidate["vlm_raw_score"] = vlm_score
        if vlm_score is None:
            candidate["vlm_error"] = "invalid_score"

    ranked, vlm_applied = finalize_vlm_ranking(
        candidates,
        top_k=top_k,
        vlm_weight=vlm_weight,
    )
    if not vlm_applied:
        print("[VLM] Scores have insufficient variance; keeping retrieval ranking")
    return ranked


def solve_kis(
    structured_query,
    visual_retriever,
    visual_encoder,
    text_bm25_retriever,
    text_semantic_retriever,
    object_lookup,
    vlm_reranker=None,
    retrieval_top_k=100,
    rerank_top_k=20,
    result_top_k=100,
    vlm_weight=0.5,
):
    # 1. Bóc tách dữ liệu
    raw_query = structured_query.get("raw_query", "")
    query_variants = structured_query.get("query_variants") or [raw_query]
    entities = structured_query.get("entities", [])
    vlm_query = raw_query or query_variants[0]

    # 2. Luồng 1: Visual Search (SigLIP/CLIP)
    visual_multi = multi_query_search(
        query_variants,
        visual_encoder,
        visual_retriever,
        top_k=retrieval_top_k,
    )
    visual_results = normalize_scores(reciprocal_rank_fusion(visual_multi))

    # 3. Luồng 2: Text Search (BM25 - Khớp từ khóa cứng)
    bm25_multi = []
    for q in query_variants:
        print(f"[BM25 Search] Query: {console_safe_text(q)}")
        bm25_multi.append(
            text_bm25_retriever.search_bm25(q, top_k=retrieval_top_k)
        )
    bm25_results = normalize_scores(reciprocal_rank_fusion(bm25_multi))

    # 4. Luồng 3: Semantic Text Search (Tìm kiếm ngữ nghĩa Vector)
    semantic_multi = []
    for q in query_variants:
        print(f"[Semantic Search] Query: {console_safe_text(q)}")
        semantic_multi.append(
            text_semantic_retriever.search(q, top_k=retrieval_top_k)
        )
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

    # 6. Bổ sung object evidence
    candidates = apply_object_filter(candidates, entities, object_lookup)

    # 7. VLM reranking và xếp hạng cuối
    candidates = rerank_candidates_with_vlm(
        candidates,
        vlm_query,
        vlm_reranker,
        object_lookup,
        top_k=rerank_top_k,
        vlm_weight=vlm_weight,
    )

    results = candidates[:result_top_k]
    print(f"[KIS] Complete. Returning {len(results)} results.")
    return results
