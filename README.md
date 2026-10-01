## AIC 2026 KIS Branch
# Setup

1. Install the pinned direct dependencies from `requirements.txt`. Install the
   PyTorch CUDA wheel appropriate for the host before loading the VLM.
2. Place provider artifacts in `data/<video_id>/`, `mapping/`, `objects/`, and
   `clip-features-32/`.
3. Review `model_manifest.json`. The supplied visual vectors are expected to be
   normalized `ViT-B-16-SigLIP/webli` embeddings with 768 dimensions.
4. Migrate legacy caption shards with
   `python migrate_caption_artifacts.py --apply`. Run without `--apply` for a
   dry run.
5. Generate missing captions with `python caption_generator.py`.
6. Build per-video caption indexes with `python build_caption_index.py`. Use
   repeated `--video-id L21_V001` arguments to rebuild selected videos.
7. Start Milvus at `http://localhost:19530`, then create the initial visual
   collection with `python build_milvus.py build`. Use
   `python build_milvus.py rebuild` for a staged replacement.
8. Set `GEMINI_API_KEY` before running `gemini_ocr.py`. Images are uploaded to
   Gemini by that command.
9. Run the retrieval pipeline from `runbranch.ipynb`.

Canonical caption artifacts are stored as:

```text
caption_generator/<group>/<video_id>.json
caption_mapping/<group>/<video_id>.json
index/<group>/<video_id>.index
```
                             USER QUERY
                             │
                             ▼
              ┌─────────────────────────────┐
              │        1. RAW QUERY         │

              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │       2. AGENT CORE         │
              │     Query Understanding     │
              │                             │
              │ LLM phân tích query         │
              │            ↓                │
              │ query_variants              │
              │ visual_description          │
              │ entities                    │
              │ temporal_constraints        │
              │ needs_ocr / needs_asr       │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │     3. STRUCTURED QUERY     │
              │                             │
              │ {                           │
              │   raw_query,                │
              │   query_variants,           │
              │   visual_description,       │
              │   entities                  │
              │ }                           │
              └──────────────┬──────────────┘
                             │
               ┌─────────────┼─────────────┐
               │             │             │
               ▼             ▼             ▼
        ┌────────────┐ ┌────────────┐ ┌────────────┐
        │    CLIP    │ │  CAPTION   │ │   OBJECT   │
        │ RETRIEVER  │ │ RETRIEVER  │ │  FILTER /  │
        │            │ │            │ │  RETRIEVER │
        └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
              │              │              │
              ▼              ▼              ▼
        CLIP Index       Caption Index    Object DB
        FAISS/Milvus        FAISS          Metadata
              │              │              │
              ▼              ▼              ▼
           Top-K           Top-K          Object
          candidates      candidates      candidates
              │              │              │
              └──────────────┼──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │        4. FUSION            │
              │                             │
              │   CLIP Results              │
              │        +                    │
              │   Caption Results           │
              │        +                    │
              │   Object Evidence           │
              │                             │
              │        ↓                    │
              │   RRF / Score Fusion        │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │     TOP-N CANDIDATES        │
              │                             │
              │ video_id                    │
              │ frame_id                    │
              │ image_path                  │
              │ fusion_score                │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │      5. VLM RERANKER        │
              │                             │
              │ Candidate Image             │
              │        +                    │
              │ Original Query              │
              │        ↓                    │
              │ Vision-Language Model       │
              │        ↓                    │
              │ Relevance Score             │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │       6. FINAL RANKING      │
              │                             │
              │ VLM Score + Retrieval Score │
              │            ↓                │
              │        Sort Descending      │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │        FINAL TOP-K          │
              │                             │
              │ video_id                    │
              │ frame_id                    │
              │ score                       │
              │ image_path                  │
              └─────────────────────────────┘
