import json
from pathlib import Path

MODEL_MANIFEST_PATH = Path(__file__).with_name("model_manifest.json")
MODEL_MANIFEST = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))

VISUAL_CONFIG = MODEL_MANIFEST["visual_embedding"]
CAPTION_CONFIG = MODEL_MANIFEST["caption_embedding"]
VLM_CONFIG = MODEL_MANIFEST["vlm_reranker"]
CAPTION_GENERATION_CONFIG = MODEL_MANIFEST["caption_generation"]
QUERY_STRUCTURING_CONFIG = MODEL_MANIFEST["query_structuring"]


def milvus_model_properties():
    return {
        "kis.model": VISUAL_CONFIG["model"],
        "kis.pretrained": VISUAL_CONFIG["pretrained"],
        "kis.hf_repo": VISUAL_CONFIG["hf_repo"],
        "kis.hf_revision": VISUAL_CONFIG["hf_revision"],
        "kis.dimension": str(VISUAL_CONFIG["dimension"]),
        "kis.normalized": str(VISUAL_CONFIG["normalized"]).lower(),
        "kis.metric": VISUAL_CONFIG["metric"],
    }
