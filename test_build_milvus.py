import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

try:
    import pymilvus  # noqa: F401
except ModuleNotFoundError:
    pymilvus = types.ModuleType("pymilvus")
    pymilvus.MilvusClient = object
    pymilvus.DataType = types.SimpleNamespace(
        INT64="INT64",
        VARCHAR="VARCHAR",
        FLOAT_VECTOR="FLOAT_VECTOR",
    )
    sys.modules["pymilvus"] = pymilvus

import build_milvus


class FakeSchema:
    def add_field(self, **kwargs):
        pass


class FakeIndexParams:
    def add_index(self, **kwargs):
        pass


class FakeMilvusClient:
    def __init__(
        self,
        collections=None,
        aliases=None,
        fail_insert=False,
        fail_alter_alias=False,
        aliases_as_dict=False,
    ):
        self.collections = set(collections or [])
        self.aliases = dict(aliases or {})
        self.row_counts = {name: 0 for name in self.collections}
        self.fail_insert = fail_insert
        self.fail_alter_alias = fail_alter_alias
        self.aliases_as_dict = aliases_as_dict
        self.alter_alias_calls = []

    def create_schema(self, **kwargs):
        return FakeSchema()

    def prepare_index_params(self):
        return FakeIndexParams()

    def create_collection(self, collection_name, **kwargs):
        self.collections.add(collection_name)
        self.row_counts[collection_name] = 0

    def list_collections(self):
        return sorted(self.collections)

    def list_aliases(self, collection_name):
        aliases = [
            alias
            for alias, target in self.aliases.items()
            if target == collection_name
        ]
        if self.aliases_as_dict:
            return {"aliases": aliases}
        return aliases

    def has_collection(self, collection_name):
        return collection_name in self.collections

    def insert(self, collection_name, data):
        if self.fail_insert:
            raise RuntimeError("insert failed")
        self.row_counts[collection_name] += len(data)
        return {"insert_count": len(data)}

    def flush(self, collection_name):
        pass

    def get_collection_stats(self, collection_name):
        return {"row_count": self.row_counts[collection_name]}

    def create_alias(self, collection_name, alias):
        if alias in self.aliases or alias in self.collections:
            raise RuntimeError("alias is already in use")
        self.aliases[alias] = collection_name

    def alter_alias(self, collection_name, alias):
        if self.fail_alter_alias:
            raise RuntimeError("alias switch failed")
        self.aliases[alias] = collection_name
        self.alter_alias_calls.append((collection_name, alias))

    def rename_collection(self, old_name, new_name):
        self.collections.remove(old_name)
        self.collections.add(new_name)
        self.row_counts[new_name] = self.row_counts.pop(old_name)
        for alias, target in list(self.aliases.items()):
            if target == old_name:
                self.aliases[alias] = new_name

    def drop_collection(self, collection_name):
        self.collections.remove(collection_name)
        self.row_counts.pop(collection_name, None)


class BuildMilvusTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.clip_dir = root / "features"
        self.csv_dir = root / "mapping"
        self.clip_dir.mkdir()
        self.csv_dir.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_video(self, dimensions=build_milvus.EMBEDDING_DIM):
        features = np.ones((2, dimensions), dtype=np.float32)
        np.save(self.clip_dir / "L01_V001.npy", features)
        pd.DataFrame({"frame_idx": [10, 20]}).to_csv(
            self.csv_dir / "L01_V001.csv",
            index=False,
        )

    def run_with_client(self, mode, client):
        with (
            patch.object(build_milvus, "CLIP_DIR", self.clip_dir),
            patch.object(build_milvus, "CSV_DIR", self.csv_dir),
            patch.object(build_milvus, "MilvusClient", return_value=client),
        ):
            build_milvus.run(mode)

    def test_preflight_failure_does_not_connect_to_milvus(self):
        self.write_video(dimensions=3)
        constructor = Mock()

        with (
            patch.object(build_milvus, "CLIP_DIR", self.clip_dir),
            patch.object(build_milvus, "CSV_DIR", self.csv_dir),
            patch.object(build_milvus, "MilvusClient", constructor),
        ):
            with self.assertRaisesRegex(ValueError, "invalid feature shape"):
                build_milvus.run("build")

        constructor.assert_not_called()

    def test_rebuild_migrates_legacy_collection_after_staging_succeeds(self):
        self.write_video()
        client = FakeMilvusClient(collections={build_milvus.COLLECTION_NAME})

        self.run_with_client("rebuild", client)

        target = client.aliases[build_milvus.COLLECTION_NAME]
        self.assertTrue(target.startswith("clip_keyframes__staging_"))
        self.assertEqual(client.row_counts[target], 2)
        self.assertNotIn(build_milvus.COLLECTION_NAME, client.collections)
        self.assertFalse(any("__backup_" in name for name in client.collections))

    def test_build_publishes_new_collection(self):
        self.write_video()
        client = FakeMilvusClient()

        self.run_with_client("build", client)

        target = client.aliases[build_milvus.COLLECTION_NAME]
        self.assertIn(target, client.collections)
        self.assertEqual(client.row_counts[target], 2)

    def test_rebuild_switches_existing_alias(self):
        self.write_video()
        old_target = "clip_keyframes__staging_old"
        client = FakeMilvusClient(
            collections={old_target},
            aliases={build_milvus.COLLECTION_NAME: old_target},
        )

        self.run_with_client("rebuild", client)

        new_target = client.aliases[build_milvus.COLLECTION_NAME]
        self.assertNotEqual(new_target, old_target)
        self.assertEqual(
            client.alter_alias_calls,
            [(new_target, build_milvus.COLLECTION_NAME)],
        )
        self.assertNotIn(old_target, client.collections)

    def test_detects_alias_from_pymilvus_2_response(self):
        target = "clip_keyframes__staging_old"
        client = FakeMilvusClient(
            collections={target},
            aliases={build_milvus.COLLECTION_NAME: target},
            aliases_as_dict=True,
        )

        self.assertEqual(
            build_milvus.get_active_collection(client),
            (target, True),
        )

    def test_insert_failure_keeps_legacy_collection(self):
        self.write_video()
        client = FakeMilvusClient(
            collections={build_milvus.COLLECTION_NAME},
            fail_insert=True,
        )

        with self.assertRaisesRegex(RuntimeError, "insert failed"):
            self.run_with_client("rebuild", client)

        self.assertEqual(client.collections, {build_milvus.COLLECTION_NAME})
        self.assertEqual(client.aliases, {})

    def test_alias_switch_failure_keeps_previous_collection_active(self):
        self.write_video()
        client = FakeMilvusClient(
            collections={build_milvus.COLLECTION_NAME},
            fail_alter_alias=True,
        )

        with self.assertRaisesRegex(RuntimeError, "alias switch failed"):
            self.run_with_client("rebuild", client)

        target = client.aliases[build_milvus.COLLECTION_NAME]
        self.assertIn(target, client.collections)
        self.assertIn("__backup_", target)
        self.assertEqual(client.collections, {target})

    def test_build_refuses_to_replace_existing_collection(self):
        self.write_video()
        client = FakeMilvusClient(collections={build_milvus.COLLECTION_NAME})

        with self.assertRaisesRegex(RuntimeError, "use rebuild"):
            self.run_with_client("build", client)

        self.assertEqual(client.collections, {build_milvus.COLLECTION_NAME})


if __name__ == "__main__":
    unittest.main()
