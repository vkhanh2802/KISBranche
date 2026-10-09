"""Provider-compatible object JSON: five arrays, Open Images labels, strings."""

import json
import math
from pathlib import Path
import re


LABEL_MAP_PATH = Path(__file__).with_name("object_label_map.json")
LABEL_MAP = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))["labels"]
OBJECT_FIELDS = (
    "detection_scores", "detection_class_names", "detection_class_entities",
    "detection_boxes", "detection_class_labels",
)


def validate_object_schema(record):
    if not isinstance(record, dict) or set(record) != set(OBJECT_FIELDS):
        raise ValueError("Object JSON must contain exactly the five provider fields")
    if any(not isinstance(record[field], list) for field in OBJECT_FIELDS):
        raise ValueError("All object fields must be arrays")
    count = len(record["detection_scores"])
    if any(len(record[field]) != count for field in OBJECT_FIELDS):
        raise ValueError("Object arrays must have equal lengths")
    for score, mid, entity, box, label in zip(*(record[field] for field in OBJECT_FIELDS)):
        if not all(isinstance(value, str) for value in (score, mid, entity, label)):
            raise ValueError("Object scores, class names, entities and labels must be strings")
        if not re.fullmatch(r"/m/[A-Za-z0-9_]+", mid) or not entity.strip():
            raise ValueError("Invalid Open Images class name/entity")
        if not label.isdigit() or int(label) < 1:
            raise ValueError("Object class labels must be positive integer strings")
        if not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
            raise ValueError("Invalid object confidence")
        if not isinstance(box, list) or len(box) != 4 or not all(isinstance(value, str) for value in box):
            raise ValueError("Object boxes must contain four normalized coordinate strings")
        coordinates = [float(value) for value in box]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in coordinates):
            raise ValueError("Object box coordinates must be finite and in [0,1]")
        if coordinates[0] > coordinates[2] or coordinates[1] > coordinates[3]:
            raise ValueError("Object boxes must use ymin,xmin,ymax,xmax order")


def provider_object_record(detection, max_detections=100):
    """Translate genuine COCO predictions into the provider's Open Images schema."""
    labels = detection.get("detection_class_entities", [])
    scores = detection.get("detection_scores", [])
    boxes = detection.get("detection_boxes", [])
    if not all(isinstance(values, list) for values in (labels, scores, boxes)):
        raise ValueError("Detection fields must be arrays")
    if len(labels) != len(scores) or len(labels) != len(boxes):
        raise ValueError("Detection arrays must have equal lengths")
    if max_detections < 1:
        raise ValueError("max_detections must be positive")
    entries = []
    for label, score, box in zip(labels, scores, boxes):
        mapped = LABEL_MAP.get(str(label).casefold())
        if mapped is None:
            continue
        entries.append((float(score), mapped, box))
    entries.sort(key=lambda entry: entry[0], reverse=True)
    output = {field: [] for field in OBJECT_FIELDS}
    for score, mapped, box in entries[:max_detections]:
        output["detection_scores"].append(str(score))
        output["detection_class_names"].append(mapped["mid"])
        output["detection_class_entities"].append(mapped["entity"])
        output["detection_boxes"].append([str(float(value)) for value in box])
        output["detection_class_labels"].append(mapped["label"])
    validate_object_schema(output)
    return output
