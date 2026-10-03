import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from structure_query import (
    GeminiQueryStructurer,
    create_structured_query,
    default_query_id,
    normalize_structured_query,
)


def valid_generated_query():
    return {
        "query_variants": [
            "an artisan draws a portrait",
            "artisan hand drawing with a special pen",
        ],
        "visual_description": "An artisan draws a portrait at a desk.",
        "entities": [
            {
                "type": "object",
                "value": "special pen",
                "attributes": {"action": "drawing a portrait"},
            },
            {
                "type": "scene",
                "value": "workspace desk",
                "attributes": {},
            },
        ],
        "needs_ocr": False,
        "needs_asr": False,
        "question": None,
        "events": [
            {"action": "the artisan draws a portrait"},
        ],
        "temporal_constraints": [],
    }


class FakeGenerator:
    def __init__(self, response):
        self.response = response
        self.queries = []

    def generate(self, query):
        self.queries.append(query)
        return self.response


class FakeModels:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.text)


class StructureQueryTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = Path(self.temp_dir.name) / "structured.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_creates_test_json_shape_and_preserves_raw_query(self):
        query = "Nghệ nhân đang vẽ chân dung bằng bút đặc biệt"
        generated = valid_generated_query()
        generated["raw_query"] = "model tried to replace the query"
        generated["query_id"] = "model_id"
        generator = FakeGenerator(generated)

        result = create_structured_query(
            query,
            query_id="q_058",
            output_path=self.output_path,
            generator=generator,
        )

        self.assertEqual(generator.queries, [query])
        self.assertEqual(result["query_id"], "q_058")
        self.assertEqual(result["raw_query"], query)
        self.assertEqual(result["query_variants"][0], query)
        self.assertEqual(
            list(result),
            [
                "query_id",
                "raw_query",
                "query_variants",
                "visual_description",
                "entities",
                "needs_ocr",
                "needs_asr",
                "question",
                "events",
                "temporal_constraints",
            ],
        )
        self.assertEqual(
            json.loads(self.output_path.read_text(encoding="utf-8")),
            result,
        )

    def test_default_query_id_is_stable(self):
        query = "a person holding a red umbrella"
        self.assertEqual(default_query_id(query), default_query_id(query))
        self.assertRegex(default_query_id(query), r"^q_[0-9a-f]{8}$")

    def test_invalid_model_output_does_not_replace_existing_file(self):
        self.output_path.write_text('{"state": "old"}', encoding="utf-8")
        generated = valid_generated_query()
        generated["needs_ocr"] = "false"

        with self.assertRaisesRegex(ValueError, "needs_ocr"):
            create_structured_query(
                "query",
                output_path=self.output_path,
                generator=FakeGenerator(generated),
            )

        self.assertEqual(
            json.loads(self.output_path.read_text(encoding="utf-8")),
            {"state": "old"},
        )

    def test_rejects_invalid_entity_and_sequence(self):
        generated = valid_generated_query()
        generated["entities"][0]["type"] = "unknown"
        with self.assertRaisesRegex(ValueError, "must be one of"):
            normalize_structured_query("query", "q_001", generated)

        generated = valid_generated_query()
        generated["temporal_constraints"] = [
            {"type": "sequence", "order": ["one event"]}
        ]
        with self.assertRaisesRegex(ValueError, "at least two"):
            normalize_structured_query("query", "q_001", generated)

    def test_query_variants_are_deduplicated_and_bounded(self):
        generated = valid_generated_query()
        generated["query_variants"] = [
            "raw query",
            "variant 1",
            "variant 2",
            "variant 3",
            "variant 4",
            "variant 5",
        ]
        result = normalize_structured_query("raw query", "q_001", generated)
        self.assertEqual(len(result["query_variants"]), 5)
        self.assertEqual(result["query_variants"].count("raw query"), 1)

    def test_gemini_generator_requests_json_schema(self):
        models = FakeModels(json.dumps(valid_generated_query()))
        client = SimpleNamespace(models=models)
        generator = GeminiQueryStructurer(client=client, model="test-model")

        result = generator.generate("find a red lantern")

        self.assertEqual(result, valid_generated_query())
        call = models.calls[0]
        self.assertEqual(call["model"], "test-model")
        self.assertEqual(call["config"].response_mime_type, "application/json")
        self.assertIsNotNone(call["config"].response_json_schema)

    def test_gemini_generator_rejects_invalid_json(self):
        client = SimpleNamespace(models=FakeModels("not json"))
        generator = GeminiQueryStructurer(client=client)
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            generator.generate("query")


if __name__ == "__main__":
    unittest.main()
