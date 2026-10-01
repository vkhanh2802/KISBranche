import json
from pathlib import Path
import unittest

from solve_kis import solve_kis


class FakeEncoder:
    def encode_text(self, query):
        return query


class FakeVisualRetriever:
    def search(self, vector, top_k):
        return [
            {"video_id": "V1", "frame_id": 1, "score": 0.9},
            {"video_id": "V2", "frame_id": 2, "score": 0.8},
        ][:top_k]


class FakeBm25Retriever:
    def search_bm25(self, query, top_k):
        return [
            {
                "video_id": "V2",
                "frame_id": 2,
                "score": 3.0,
                "retrieval_text": "second",
                "image_path": "solve_kis.py",
            },
            {
                "video_id": "V1",
                "frame_id": 1,
                "score": 2.0,
                "retrieval_text": "first",
                "image_path": "README.md",
            },
        ][:top_k]


class FakeSemanticRetriever:
    def search(self, query, top_k):
        return [
            {"video_id": "V1", "frame_id": 1, "score": 0.8},
            {"video_id": "V2", "frame_id": 2, "score": 0.7},
        ][:top_k]


class FakeObjectLookup:
    def get_keyframe_n(self, video_id, frame_id):
        return None


class FakeVlmReranker:
    def score(self, image_path, query, caption=None):
        return 10 if image_path.name == "solve_kis.py" else 0


class SolveKisIntegrationTest(unittest.TestCase):
    def run_pipeline(self, vlm_reranker=None):
        return solve_kis(
            structured_query={
                "raw_query": "test query",
                "query_variants": ["test variant"],
                "entities": [],
            },
            visual_retriever=FakeVisualRetriever(),
            visual_encoder=FakeEncoder(),
            text_bm25_retriever=FakeBm25Retriever(),
            text_semantic_retriever=FakeSemanticRetriever(),
            object_lookup=FakeObjectLookup(),
            vlm_reranker=vlm_reranker,
            retrieval_top_k=2,
            rerank_top_k=2,
            result_top_k=2,
        )

    def test_retrieval_ranking_without_vlm(self):
        results = self.run_pipeline()

        self.assertEqual([item["frame_id"] for item in results], [1, 2])
        self.assertTrue(
            all(item["final_score"] == item["score"] for item in results)
        )

    def test_vlm_reranks_and_preserves_metadata(self):
        results = self.run_pipeline(FakeVlmReranker())

        self.assertEqual([item["frame_id"] for item in results], [2, 1])
        self.assertEqual(results[0]["retrieval_text"], "second")
        self.assertTrue(all("final_score" in item for item in results))


class NotebookTest(unittest.TestCase):
    def test_code_cells_compile(self):
        notebook_path = Path(__file__).with_name("runbranch.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

        code_cells = [
            cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
        ]
        for index, cell in enumerate(code_cells):
            compile("".join(cell.get("source", [])), f"cell-{index}", "exec")

        self.assertGreater(len(code_cells), 0)


if __name__ == "__main__":
    unittest.main()
