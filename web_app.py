import math
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from paths import KIS_ROOT, VIDEO_ID_PATTERN, video_image_dir
from solve_kis import _resolve_candidate_image_path, solve_kis
from structure_query import MAX_QUERY_CHARS, create_structured_query


WEB_DIR = KIS_ROOT / "web"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def serialize_search_results(results, object_lookup, image_resolver=None):
    image_resolver = image_resolver or _resolve_candidate_image_path
    serialized = []

    for rank, candidate in enumerate(results, start=1):
        image_path = None
        try:
            image_path = image_resolver(candidate, object_lookup)
        except (OSError, RuntimeError, ValueError):
            pass

        keyframe_n = candidate.get("keyframe_n")
        if keyframe_n is None and image_path is not None:
            try:
                keyframe_n = int(Path(image_path).stem)
            except ValueError:
                pass

        video_id = str(candidate["video_id"])
        frame_id = int(candidate["frame_id"])
        keyframe_n = int(keyframe_n) if keyframe_n is not None else None
        image_url = (
            f"/api/images/{video_id}/{keyframe_n}"
            if image_path is not None and keyframe_n is not None
            else None
        )

        serialized.append(
            {
                "rank": rank,
                "video_id": video_id,
                "image_number": keyframe_n,
                "frame_id": frame_id,
                "image_url": image_url,
                "caption": str(candidate.get("retrieval_text") or ""),
                "score": _finite_float(candidate.get("final_score")),
                "fusion_score": _finite_float(candidate.get("fusion_score")),
                "visual_score": _finite_float(candidate.get("visual_score")),
                "bm25_score": _finite_float(candidate.get("bm25_score")),
                "semantic_score": _finite_float(candidate.get("semantic_score")),
                "object_score": _finite_float(candidate.get("object_score")),
                "vlm_score": _finite_float(candidate.get("vlm_raw_score")),
                "vlm_error": candidate.get("vlm_error"),
            }
        )

    return serialized


@lru_cache(maxsize=4096)
def find_keyframe_image(video_id, image_number):
    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        return None
    if image_number < 0:
        return None

    image_dir = video_image_dir(video_id)
    if not image_dir.is_dir():
        return None

    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            if int(image_path.stem) == image_number:
                return image_path
        except ValueError:
            continue
    return None


class SearchService:
    def __init__(self):
        self._operation_lock = Lock()
        self.state = "cold"
        self.last_error = None
        self.vlm_enabled = os.environ.get("KIS_DISABLE_VLM", "").lower() not in {
            "1",
            "true",
            "yes",
        }
        self.query_structurer = None
        self.visual_encoder = None
        self.visual_retriever = None
        self.caption_retriever = None
        self.bm25_retriever = None
        self.object_lookup = None
        self.vlm_reranker = None

    def status(self):
        return {
            "state": self.state,
            "vlm_enabled": self.vlm_enabled,
            "last_error": self.last_error,
        }

    def _load(self):
        if self.state == "ready":
            return

        self.state = "loading"
        self.last_error = None
        try:
            from bm25_retriever import BM25Retriever
            from caption_retriever import CaptionRetriever
            from object_filter import ObjectMetadataLookup
            from paths import MAPPING_DIR, OBJECTS_DIR
            from search_clip import SiglipRetriever
            from siglip_encoder import SiglipEncoder
            from structure_query import GeminiQueryStructurer

            query_structurer = GeminiQueryStructurer()
            visual_encoder = SiglipEncoder()
            visual_retriever = SiglipRetriever()
            caption_retriever = CaptionRetriever()
            bm25_retriever = BM25Retriever(caption_retriever.caption_data)
            object_lookup = ObjectMetadataLookup(
                mapping_dir=MAPPING_DIR,
                objects_dir=OBJECTS_DIR,
            )

            vlm_reranker = None
            if self.vlm_enabled:
                from vlm_reranker import VLMReranker

                vlm_reranker = VLMReranker()

            self.query_structurer = query_structurer
            self.visual_encoder = visual_encoder
            self.visual_retriever = visual_retriever
            self.caption_retriever = caption_retriever
            self.bm25_retriever = bm25_retriever
            self.object_lookup = object_lookup
            self.vlm_reranker = vlm_reranker
            self.state = "ready"
        except Exception as exc:
            self.state = "failed"
            self.last_error = str(exc)
            raise

    def search(self, raw_query):
        started_at = perf_counter()
        with self._operation_lock:
            self._load()
            structured_query = create_structured_query(
                raw_query,
                output_path=None,
                generator=self.query_structurer,
            )
            results = solve_kis(
                structured_query=structured_query,
                visual_retriever=self.visual_retriever,
                visual_encoder=self.visual_encoder,
                text_bm25_retriever=self.bm25_retriever,
                text_semantic_retriever=self.caption_retriever,
                object_lookup=self.object_lookup,
                vlm_reranker=self.vlm_reranker,
                retrieval_top_k=100,
                rerank_top_k=20,
                result_top_k=20,
                vlm_weight=0.5,
            )
            serialized = serialize_search_results(results, self.object_lookup)

        return {
            "query_id": structured_query["query_id"],
            "raw_query": structured_query["raw_query"],
            "query_variants": structured_query["query_variants"],
            "elapsed_seconds": round(perf_counter() - started_at, 2),
            "vlm_enabled": self.vlm_enabled,
            "results": serialized,
        }


service = SearchService()
app = FastAPI(title="KIS Framefinder", version="1.0.0")
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/status")
def api_status():
    return service.status()


@app.post("/api/search")
async def api_search(request: SearchRequest):
    try:
        return await run_in_threadpool(service.search, request.query)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {exc}",
        ) from exc


@app.get("/api/images/{video_id}/{image_number}", include_in_schema=False)
def api_image(video_id: str, image_number: int):
    image_path = find_keyframe_image(video_id, image_number)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(
        image_path,
        headers={"Cache-Control": "public, max-age=86400"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, workers=1)
