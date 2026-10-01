import re
import math


DEFAULT_MIN_CONFIDENCE = 0.2
DEFAULT_OBJECT_WEIGHT = 0.15

OBJECT_ALIASES = {
    "photo": {"photo", "photograph", "picture", "picture frame", "poster"},
    "painting": {"painting", "picture", "picture frame", "poster"},
    "bức ảnh": {"photo", "photograph", "picture", "picture frame"},
    "tranh": {"painting", "picture", "picture frame", "poster"},
    "artisan hand": {"hand", "human hand"},
    "bàn tay nghệ nhân": {"hand", "human hand"},
}


def _normalize_label(value):
    value = re.sub(r"[^\w\s]", " ", str(value).casefold())
    return " ".join(value.split())


def _detector_labels(entity):
    explicit_labels = entity.get("detector_labels", [])
    if isinstance(explicit_labels, str):
        explicit_labels = [explicit_labels]

    value_parts = re.split(
        r"\s*/\s*|\s+or\s+|\s+hoặc\s+",
        str(entity.get("value", "")),
        flags=re.IGNORECASE,
    )
    labels = {
        normalized
        for value in [*explicit_labels, *value_parts]
        if (normalized := _normalize_label(value))
    }

    for label in list(labels):
        labels.update(OBJECT_ALIASES.get(label, ()))

    return labels


def apply_object_filter(
    candidates,
    entities,
    object_lookup,
    min_confidence=DEFAULT_MIN_CONFIDENCE,
    object_weight=DEFAULT_OBJECT_WEIGHT,
):
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    if not 0.0 <= object_weight <= 1.0:
        raise ValueError("object_weight must be between 0 and 1")

    object_entities = [e for e in entities if e.get("type") == "object"]

    if not object_entities:
        return candidates

    print(f"[Object Score] Checking {len(object_entities)} object entities")

    for candidate in candidates:
        key = (candidate["video_id"], candidate["frame_id"])
        retrieval_score = float(
            candidate.setdefault("retrieval_score", candidate["score"])
        )
        if not math.isfinite(retrieval_score):
            raise ValueError(f"Non-finite retrieval score for {key}")
        meta = object_lookup.get(key)

        if meta is None:
            candidate["object_score"] = 0.0
            continue

        detected = {}
        for label, raw_confidence in zip(
            meta.get("detection_class_entities", []),
            meta.get("detection_scores", []),
        ):
            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError):
                continue

            if not math.isfinite(confidence) or confidence < min_confidence:
                continue
            confidence = min(1.0, max(0.0, confidence))

            label = _normalize_label(label)
            detected[label] = max(confidence, detected.get(label, 0.0))

        score = 0.0
        total = 0.0

        for entity in object_entities:
            detector_labels = _detector_labels(entity)
            if not detector_labels:
                continue

            total += 1
            score += max(
                (detected.get(label, 0.0) for label in detector_labels),
                default=0.0,
            )

        candidate["object_score"] = score / total if total else 0.0
        candidate["score"] = retrieval_score + (
            object_weight
            * candidate["object_score"]
            * max(0.0, 1.0 - retrieval_score)
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)

    return candidates
