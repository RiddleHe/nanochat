import torch
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nanochat import dataloader


class Tokenizer:
    def get_bos_token_id(self):
        return 0

    def encode(self, docs, prepend, **kwargs):
        return [[prepend] + [int(d) + 1] * (3 + int(d) % 19) for d in docs]


@pytest.mark.parametrize("rank,world", [(0, 1), (0, 2), (1, 2), (0, 4), (1, 4), (2, 4), (3, 4)])
def test_exact_resume_across_documents_groups_files_epochs(tmp_path, monkeypatch, rank, world):
    paths = []
    for i in range(3):
        path = tmp_path / f"{i}.parquet"
        pq.write_table(pa.table({"text": [str(j + i * 100) for j in range(30)]}), path, row_group_size=9)
        paths.append(str(path))
    monkeypatch.setattr(dataloader, "list_parquet_files", lambda **kw: paths)
    monkeypatch.setattr(dataloader, "get_dist_info", lambda: (world > 1, rank, rank, world))
    def build(state=None):
        return dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(
            Tokenizer(), 2, 12, "train", tokenizer_batch_size=4, buffer_size=5,
            device="cpu", exact_resume=True, resume_state_dict=state)
    loader = build()
    for i in range(70):
        _, _, state = next(loader)
        if i % 7 == 0:
            resumed = build(state)
            for _ in range(4):
                x, y, _ = next(loader)
                rx, ry, _ = next(resumed)
                torch.testing.assert_close(x, rx, rtol=0, atol=0)
                torch.testing.assert_close(y, ry, rtol=0, atol=0)
