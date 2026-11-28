"""Exactly resumable version of the original BOS best-fit pretraining loader.

Use ``next(loader)`` only for batches that will be consumed before checkpointing.
There is no hidden prefetch or pending batch. A checkpoint describes the next
batch to return, including tokenized documents not yet packed into a row.

Fresh runs use the same ordering, refill thresholds, tie breaks, and cropping as
``nanochat.dataloader``. Rank and world size are captured explicitly, never read
from mutable distributed globals while an iterator is being advanced. Dataset
files and tokenizer contents must remain immutable across a resume.
"""

from copy import deepcopy
from pathlib import Path

import pyarrow.parquet as pq
import torch

from nanochat.dataset import list_parquet_files


class ExactBOSBestFitDataLoader:
    """Synchronous iterable returning independently owned ``(inputs, targets)``.

    ``state_dict`` is a CPU-only, ``torch.load(weights_only=True)`` compatible
    snapshot. Restore only at a completed training-step boundary. The caller is
    responsible for saving model, optimizer, step/schedule, and per-rank RNG.
    """

    FORMAT_VERSION = 1

    def __init__(
        self, tokenizer, B, T, split, *, rank=0, world_size=1,
        tokenizer_threads=4, tokenizer_batch_size=128, device="cuda",
        buffer_size=1000,
    ):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        for name, value in (("B", B), ("T", T), ("world_size", world_size),
                            ("tokenizer_threads", tokenizer_threads),
                            ("tokenizer_batch_size", tokenizer_batch_size),
                            ("buffer_size", buffer_size)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world_size:
            raise ValueError("rank must be an integer in [0, world_size)")
        self.tokenizer = tokenizer
        self.B, self.T = B, T
        self.rank, self.world_size = rank, world_size
        self.tokenizer_threads = tokenizer_threads
        self.tokenizer_batch_size = tokenizer_batch_size
        self.buffer_size = buffer_size
        self.device = torch.device(device)
        self.bos_token = int(tokenizer.get_bos_token_id())
        paths = list_parquet_files(warn_on_legacy=rank == 0 and split == "train")
        self.parquet_paths = list(paths[:-1] if split == "train" else paths[-1:])
        if not self.parquet_paths:
            raise ValueError(f"No parquet files available for the {split!r} split")
        # Metadata catches common accidental dataset replacements without hashing
        # an entire pretraining corpus. Immutable contents remain a prerequisite.
        self._manifest = []
        for path in self.parquet_paths:
            stat = Path(path).stat()
            self._manifest.append({"path": str(path), "size": stat.st_size,
                                   "mtime_ns": stat.st_mtime_ns})
        self._config = {
            "B": B, "T": T, "split": split, "rank": rank,
            "world_size": world_size, "tokenizer_threads": tokenizer_threads,
            "tokenizer_batch_size": tokenizer_batch_size,
            "buffer_size": buffer_size, "bos_token_id": self.bos_token,
        }
        self._cursor = {"pq_idx": 0, "rg_idx": rank, "row_offset": 0, "epoch": 1}
        self._last_document_batch = None
        self._doc_buffer = []
        self._counters = {
            "batches_emitted": 0, "tokenizer_batches_read": 0,
            "raw_documents_read": 0, "row_groups_completed": 0,
            "epoch_batches_read": 0,
        }
        self._row_buffer = torch.empty((B, T + 1), dtype=torch.long)
        self._clear_read_cache()

    def _clear_read_cache(self):
        self._parquet_file = None
        self._cached_pq_idx = None
        self._cached_rg_idx = None
        self._row_group_texts = None

    def __iter__(self):
        return self

    @property
    def progress(self):
        """Small logging metadata; this is *not* an exact resume checkpoint."""
        last = self._last_document_batch or {
            "pq_idx": 0, "rg_idx": self.rank, "row_offset": 0, "epoch": 1,
        }
        return {**last, **self._counters, "buffered_documents": len(self._doc_buffer)}

    def _read_document_batch(self):
        while True:
            cursor = self._cursor
            if cursor["pq_idx"] == len(self.parquet_paths):
                if self._counters["epoch_batches_read"] == 0:
                    raise ValueError(f"No nonempty document row groups assigned to rank {self.rank}")
                cursor.update(pq_idx=0, rg_idx=self.rank, row_offset=0,
                              epoch=cursor["epoch"] + 1)
                self._counters["epoch_batches_read"] = 0
                self._clear_read_cache()
            if self._cached_pq_idx != cursor["pq_idx"]:
                self._parquet_file = pq.ParquetFile(self.parquet_paths[cursor["pq_idx"]])
                self._cached_pq_idx = cursor["pq_idx"]
                self._cached_rg_idx = None
                self._row_group_texts = None
            if cursor["rg_idx"] >= self._parquet_file.num_row_groups:
                cursor.update(pq_idx=cursor["pq_idx"] + 1,
                              rg_idx=self.rank, row_offset=0)
                self._clear_read_cache()
                continue
            if self._cached_rg_idx != cursor["rg_idx"]:
                row_group = self._parquet_file.read_row_group(cursor["rg_idx"])
                self._row_group_texts = row_group.column("text").to_pylist()
                self._cached_rg_idx = cursor["rg_idx"]
            texts = self._row_group_texts
            start = cursor["row_offset"]
            if start > len(texts):
                raise ValueError("Saved document offset exceeds its parquet row group")
            if start == len(texts):
                cursor.update(rg_idx=cursor["rg_idx"] + self.world_size, row_offset=0)
                self._counters["row_groups_completed"] += 1
                continue
            stop = min(start + self.tokenizer_batch_size, len(texts))
            batch = texts[start:stop]
            self._last_document_batch = dict(cursor)
            cursor["row_offset"] = stop
            self._counters["raw_documents_read"] += len(batch)
            self._counters["tokenizer_batches_read"] += 1
            self._counters["epoch_batches_read"] += 1
            # Advance to the next rank-owned row group after the final chunk.
            # A file/epoch boundary is normalized on the following read; the
            # cursor is valid even when rg_idx is just beyond this file's end.
            if stop == len(texts):
                cursor.update(rg_idx=cursor["rg_idx"] + self.world_size, row_offset=0)
                self._counters["row_groups_completed"] += 1
            return batch

    def _refill_buffer(self):
        documents = self._read_document_batch()
        encoded = self.tokenizer.encode(
            documents, prepend=self.bos_token, num_threads=self.tokenizer_threads,
        )
        if len(encoded) != len(documents):
            raise ValueError("Tokenizer returned a different number of documents")
        for tokens in encoded:
            doc = [int(token) for token in tokens]
            if not doc or doc[0] != self.bos_token:
                raise ValueError("Each encoded document must begin with BOS")
            self._doc_buffer.append(doc)

    def __next__(self):
        row_capacity = self.T + 1
        for row_idx in range(self.B):
            pos = 0
            while pos < row_capacity:
                while len(self._doc_buffer) < self.buffer_size:
                    self._refill_buffer()
                remaining = row_capacity - pos
                best_idx, best_len = -1, 0
                for i, doc in enumerate(self._doc_buffer):
                    if len(doc) <= remaining and len(doc) > best_len:
                        best_idx, best_len = i, len(doc)
                if best_idx >= 0:
                    doc = self._doc_buffer.pop(best_idx)
                else:
                    shortest_idx = min(range(len(self._doc_buffer)),
                                       key=lambda i: len(self._doc_buffer[i]))
                    doc = self._doc_buffer.pop(shortest_idx)[:remaining]
                self._row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                pos += len(doc)

        # Each batch owns both staging and destination storage. Reusing a pinned
        # staging tensor can race an asynchronous HtoD copy; reusing destination
        # views also silently modifies a previously returned batch.
        use_cuda = self.device.type == "cuda"
        staging = torch.empty(2 * self.B * self.T, dtype=torch.long, pin_memory=use_cuda)
        staging[:self.B * self.T].view(self.B, self.T).copy_(self._row_buffer[:, :-1])
        staging[self.B * self.T:].view(self.B, self.T).copy_(self._row_buffer[:, 1:])
        batch = staging.to(self.device, non_blocking=use_cuda)
        self._counters["batches_emitted"] += 1
        return (batch[:self.B * self.T].view(self.B, self.T),
                batch[self.B * self.T:].view(self.B, self.T))

    def state_dict(self):
        """Copy the entire iterator state without advancing or prefetching data."""
        return {
            "format_version": self.FORMAT_VERSION,
            "config": dict(self._config), "manifest": deepcopy(self._manifest),
            "next_document": dict(self._cursor),
            "last_document_batch": deepcopy(self._last_document_batch),
            "doc_buffer": [list(doc) for doc in self._doc_buffer],
            "counters": dict(self._counters),
        }

    def load_state_dict(self, state):
        """Restore a snapshot from this loader configuration and dataset."""
        if state.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("Unsupported exact loader checkpoint format")
        if state.get("config") != self._config:
            raise ValueError("Exact loader configuration mismatch (including rank/world_size)")
        if state.get("manifest") != self._manifest:
            raise ValueError("Exact loader dataset manifest changed")
        cursor = dict(state["next_document"])
        for name in ("pq_idx", "rg_idx", "row_offset", "epoch"):
            if type(cursor.get(name)) is not int:
                raise ValueError(f"Invalid saved cursor field: {name}")
        if not (0 <= cursor["pq_idx"] <= len(self.parquet_paths)
                and cursor["rg_idx"] >= self.rank
                and (cursor["rg_idx"] - self.rank) % self.world_size == 0
                and cursor["row_offset"] >= 0 and cursor["epoch"] >= 1):
            raise ValueError("Invalid saved document cursor")
        documents = deepcopy(state["doc_buffer"])
        if any(not isinstance(doc, list) or not doc or doc[0] != self.bos_token
               or any(type(token) is not int for token in doc) for doc in documents):
            raise ValueError("Invalid saved tokenized document buffer")
        counters = dict(state["counters"])
        if counters.keys() != self._counters.keys() or any(
                type(value) is not int or value < 0 for value in counters.values()):
            raise ValueError("Invalid saved loader counters")
        self._cursor = cursor
        self._last_document_batch = deepcopy(state["last_document_batch"])
        self._doc_buffer = documents
        self._counters = counters
        self._clear_read_cache()
