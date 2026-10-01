def fusion_clip_caption(
    clip_results,
    caption_results,
    k=60,
    caption_weight=1.0
):
    print("[Fusion] CLIP + Caption RRF")

    candidates = {}


    for rank, item in enumerate(clip_results, start=1):

        key = (
            item["video_id"],
            item["frame_id"]
        )

        rrf_score = 1.0 / (k + rank)

        if key not in candidates:
            candidates[key] = {
                "video_id": item["video_id"],
                "frame_id": item["frame_id"],
                "clip_score": 0.0,
                "caption_score": 0.0,
                "fusion_score": 0.0
            }

        candidates[key]["clip_score"] = rrf_score


    for rank, item in enumerate(caption_results, start=1):

        key = (
            item["video_id"],
            item["frame_id"]
        )

        rrf_score = (
            caption_weight /
            (k + rank)
        )

        if key not in candidates:
            candidates[key] = {
                "video_id": item["video_id"],
                "frame_id": item["frame_id"],
                "clip_score": 0.0,
                "caption_score": 0.0,
                "fusion_score": 0.0
            }

        candidates[key]["caption_score"] = rrf_score

    results = []

    for item in candidates.values():

        item["fusion_score"] = (
            item["clip_score"]
            +
            item["caption_score"]
        )

        results.append(item)


    results.sort(
        key=lambda x: x["fusion_score"],
        reverse=True
    )
    for item in results:
        item["score"] = item["fusion_score"]

    print(
        f"[Fusion] {len(results)} unique candidates"
    )

    return results