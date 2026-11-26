"""CPU checks using fake parquet IO and the unchanged loader as a fresh-run oracle.

Run: python -m unittest discover -s tests -p test_dataloader_exact.py -v
No dataset download, tokenizer installation, pyarrow, or GPU is needed. The real
packing code from both modules runs against the same controlled row groups.
"""

import copy
import importlib.util
import io
from pathlib import Path
import random
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

from nanochat.training_state_exact import (
    capture_rank_training_state, capture_rng_state,
    restore_rank_training_state, restore_rng_state,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def get_bos_token_id(self):
        return 0

    def encode(self, documents, prepend, num_threads):
        self.calls.append(list(documents))
        return [[prepend] + [int(value) for value in text.split()] for text in documents]


class FakeRowGroup:
    def __init__(self, texts):
        self.texts = texts

    def column(self, name):
        assert name == "text"
        return self

    def to_pylist(self):
        return list(self.texts)


class FakeParquetFile:
    def __init__(self, path, dataset, reads):
        self.path = str(path)
        self.groups = dataset[self.path]
        self.num_row_groups = len(self.groups)
        self.reads = reads

    def read_row_group(self, index):
        self.reads.append((self.path, index))
        return FakeRowGroup(self.groups[index])


def load_source(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "nanochat" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def serialized_roundtrip(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=True)


class ExactLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = [str(Path(self.temp.name) / f"shard_{i}.parquet") for i in range(3)]
        self.dataset, self.reads = {}, []
        # Uneven files, uneven row groups, empty groups/documents, short/long
        # documents, and tokenizer chunks that do not divide row-group length.
        for file_index, group_lengths in enumerate(([7, 5, 0, 6, 3], [4, 7, 5], [5, 6, 4, 7])):
            groups = []
            for group_index, length in enumerate(group_lengths):
                documents = []
                for row_index in range(length):
                    token_count = (row_index * 7 + group_index * 3 + file_index) % 19
                    base = 1 + file_index * 10000 + group_index * 1000 + row_index * 30
                    documents.append(" ".join(str(base + i) for i in range(token_count)))
                groups.append(documents)
            self.dataset[self.paths[file_index]] = groups
            Path(self.paths[file_index]).write_text(repr(groups))
        self.oracle_rank, self.oracle_world = 0, 2
        parquet = types.ModuleType("pyarrow.parquet")
        parquet.ParquetFile = lambda path: FakeParquetFile(path, self.dataset, self.reads)
        arrow = types.ModuleType("pyarrow")
        arrow.parquet = parquet
        dataset_module = types.ModuleType("nanochat.dataset")
        dataset_module.list_parquet_files = lambda **kwargs: list(self.paths)
        common = types.ModuleType("nanochat.common")
        common.get_dist_info = lambda: (True, self.oracle_rank, self.oracle_rank, self.oracle_world)
        with mock.patch.dict("sys.modules", {
            "pyarrow": arrow, "pyarrow.parquet": parquet,
            "nanochat.dataset": dataset_module, "nanochat.common": common,
        }):
            self.exact = load_source("loader_exact_test", "dataloader_exact.py")
            self.original = load_source("loader_original_test", "dataloader.py")

    def make_loader(self, rank=0, world_size=2, **kwargs):
        defaults = dict(B=2, T=11, split="train", tokenizer_batch_size=3,
                        buffer_size=5, device="cpu")
        defaults.update(kwargs)
        return self.exact.ExactBOSBestFitDataLoader(
            FakeTokenizer(), rank=rank, world_size=world_size, **defaults,
        )

    def assert_batch_equal(self, expected, actual):
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertTrue(torch.equal(expected[1], actual[1]))

    def test_fresh_batches_equal_unchanged_loader_for_both_ranks_and_splits(self):
        for rank in (0, 1):
            for split in ("train", "val"):
                for batch_size, buffer_size in ((1, 1), (3, 5), (8, 9)):
                    with self.subTest(rank=rank, split=split, batch=batch_size, buffer=buffer_size):
                        self.oracle_rank = rank
                        new = self.make_loader(rank, split=split, tokenizer_batch_size=batch_size,
                                               buffer_size=buffer_size)
                        old = self.original.tokenizing_distributed_data_loader_with_state_bos_bestfit(
                            FakeTokenizer(), 2, 11, split, tokenizer_batch_size=batch_size,
                            buffer_size=buffer_size, device="cpu",
                        )
                        for _ in range(35):
                            expected_x, expected_y, old_progress = next(old)
                            actual = next(new)
                            self.assert_batch_equal((expected_x, expected_y), actual)
                            self.assertEqual(old_progress, {
                                key: new.progress[key] for key in ("pq_idx", "rg_idx", "epoch")
                            })
                            self.assertTrue(torch.equal(actual[0][:, 0], torch.zeros(2, dtype=torch.long)))
                        self.assertGreater(new.progress["epoch"], 1)

    def test_resume_at_every_batch_across_refill_row_group_file_and_epoch(self):
        for rank in (0, 1):
            uninterrupted = self.make_loader(rank)
            snapshots, expected = [], []
            for _ in range(45):
                snapshots.append(serialized_roundtrip(uninterrupted.state_dict()))
                expected.append(tuple(t.clone() for t in next(uninterrupted)))
            seen = [s["next_document"] for s in snapshots]
            self.assertTrue(any(c["row_offset"] > 0 for c in seen))
            self.assertTrue(any(c["pq_idx"] > 0 for c in seen))
            self.assertTrue(any(c["epoch"] > 1 for c in seen))
            for cut in range(40):
                with self.subTest(rank=rank, cut=cut):
                    restored = self.make_loader(rank)
                    restored.load_state_dict(snapshots[cut])
                    for step in range(cut, cut + 5):
                        self.assert_batch_equal(expected[step], next(restored))
                    if cut + 5 < len(snapshots):
                        self.assertEqual(snapshots[cut + 5], restored.state_dict())

    def test_restored_buffer_refill_keeps_explicit_rank(self):
        original = self.make_loader(rank=1, buffer_size=15)
        for _ in range(4):
            next(original)
        state = original.state_dict()
        self.assertGreater(len(state["doc_buffer"]), 0)
        resumed = self.make_loader(rank=1, buffer_size=15)
        resumed.load_state_dict(state)
        # This reproduces the old emulated-rank failure mode: globals describe
        # rank zero when the restored token buffer eventually needs new data.
        self.oracle_rank, self.oracle_world = 0, 1
        self.reads.clear()
        for _ in range(30):
            self.assert_batch_equal(next(original), next(resumed))
        self.assertGreater(len(self.reads), 0)
        self.assertTrue(all(index % 2 == 1 for _, index in self.reads))

    def test_batches_and_snapshots_own_their_storage(self):
        loader = self.make_loader()
        first = next(loader)
        saved_batch = tuple(t.clone() for t in first)
        state = loader.state_dict()
        saved_state = copy.deepcopy(state)
        for _ in range(7):
            next(loader)
        self.assert_batch_equal(saved_batch, first)
        self.assertEqual(saved_state, state)
        restored = self.make_loader()
        restored.load_state_dict(state)
        next(restored)
        self.assertEqual(saved_state, state)

    def test_reject_changed_topology_packing_and_dataset(self):
        loader = self.make_loader()
        next(loader)
        state = loader.state_dict()
        for kwargs in ({"rank": 1}, {"world_size": 1}, {"T": 12},
                       {"buffer_size": 6}, {"tokenizer_batch_size": 2}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                    self.make_loader(**kwargs).load_state_dict(state)
        Path(self.paths[0]).write_text("changed corpus")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            self.make_loader().load_state_dict(state)

    def test_empty_rank_errors_instead_of_spinning(self):
        loader = self.make_loader(rank=7, world_size=8)
        with self.assertRaisesRegex(ValueError, "No nonempty document row groups"):
            next(loader)

    def test_rank_rng_and_optimizer_split_run_matches_uninterrupted(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        loader = self.make_loader()
        model = torch.nn.Sequential(torch.nn.Linear(11, 7), torch.nn.Dropout(0.3),
                                    torch.nn.Linear(7, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        def step(model, optimizer, loader):
            x, y = next(loader)
            # All three CPU RNG families affect the trajectory. Dropout checks
            # Torch RNG restoration, including RNG consumed by new model init.
            scale = (random.random() + float(np.random.random())) * 1e-4
            loss = (model(x.float() * scale) - y.float().mean(1, keepdim=True) * 1e-4).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            return loss.detach().clone()

        for _ in range(4):
            step(model, optimizer, loader)
        checkpoint = serialized_roundtrip({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "rank_state": capture_rank_training_state(loader, rank=0, world_size=2, device="cpu"),
        })
        expected_losses = [step(model, optimizer, loader) for _ in range(8)]
        expected_parameters = {name: p.clone() for name, p in model.state_dict().items()}
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        restored_model = torch.nn.Sequential(torch.nn.Linear(11, 7), torch.nn.Dropout(0.3),
                                             torch.nn.Linear(7, 1))
        restored_model.load_state_dict(checkpoint["model"])
        restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.001)
        restored_optimizer.load_state_dict(checkpoint["optimizer"])
        restored_loader = self.make_loader()
        restore_rank_training_state(checkpoint["rank_state"], restored_loader,
                                    rank=0, world_size=2, device="cpu")
        actual_losses = [step(restored_model, restored_optimizer, restored_loader) for _ in range(8)]
        self.assertTrue(torch.equal(torch.stack(expected_losses), torch.stack(actual_losses)))
        for name, parameter in restored_model.state_dict().items():
            self.assertTrue(torch.equal(expected_parameters[name], parameter), name)
        for index, values in restored_optimizer.state_dict()["state"].items():
            for key, value in values.items():
                expected = expected_optimizer["state"][index][key]
                self.assertTrue(torch.equal(expected, value) if torch.is_tensor(value) else expected == value)
        self.assertEqual(loader.state_dict(), restored_loader.state_dict())
        with self.assertRaisesRegex(ValueError, "rank/world_size mismatch"):
            restore_rank_training_state(checkpoint["rank_state"], restored_loader,
                                        rank=1, world_size=2, device="cpu")


class RngStateTests(unittest.TestCase):
    def test_weights_only_roundtrip_includes_cached_gaussian(self):
        random.seed(123)
        random.gauss(0, 1)
        np.random.seed(123)
        np.random.normal()
        torch.manual_seed(123)
        state = serialized_roundtrip(capture_rng_state("cpu"))
        expected = (random.gauss(0, 1), np.random.normal(size=4), torch.rand(4))
        restore_rng_state(state, "cpu")
        actual = (random.gauss(0, 1), np.random.normal(size=4), torch.rand(4))
        self.assertEqual(expected[0], actual[0])
        self.assertTrue(np.array_equal(expected[1], actual[1]))
        self.assertTrue(torch.equal(expected[2], actual[2]))


if __name__ == "__main__":
    unittest.main()
