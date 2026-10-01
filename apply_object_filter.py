def apply_object_filter(candidates, entities, object_lookup):
    object_entities = [e for e in entities if e.get("type") == "object"]

    if not object_entities:
        return candidates

    print(f"[Object Score] Checking {len(object_entities)} object entities")

    for candidate in candidates:
        key = (candidate["video_id"], candidate["frame_id"])
        meta = object_lookup.get(key)

        if meta is None:
            candidate["object_score"] = 0.0
            continue

        detected = [
            str(x).lower()
            for x in meta.get("detection_class_entities", [])
        ]

        score = 0.0
        total = 0.0

        for entity in object_entities:
            value = entity.get("value", "").lower()
            attributes = entity.get("attributes", {})

            if not value:
                continue

            total += 1

            count = detected.count(value)

            if count == 0:
                continue

            score += 1.0

            quantity = attributes.get("quantity")

            if quantity == "many":
                if count >= 2:
                    score += 0.5

        candidate["object_score"] = score / total if total else 0.0

    candidates.sort(
        key=lambda x: (
            x.get("object_score", 0.0),
            x["score"]
        ),
        reverse=True
    )

    return candidates