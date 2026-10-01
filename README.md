## AIC 2026 KIS Branch 
# Setup
Data:
    - List of keyframes from video
    - Object detection score
    - Mapping between keyframes and frame_index
    - Clip-features
    All of those data are provided by AIC
Operation:
    - Setup a Milvus instance at http://localhost:19530
    - Create the initial collection with `python build_milvus.py build`
    - Replace an existing collection safely with `python build_milvus.py rebuild`
    - Generate caption for each keyframe by caption_generator.py
    - Generate caption index and mapping by build_caption_index.py
    - Run all branch in notebook runbranch.py
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
