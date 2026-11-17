"""Finite lossless tokenization with document offsets into the global token stream."""
import argparse
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from nanochat.tokenizer import get_tokenizer
from .common import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    run = parser.parse_args().run_dir
    config = read_json(run / "inputs/config.json")
    tokenizer = get_tokenizer()
    bos = tokenizer.get_bos_token_id()
    parquet = pq.ParquetFile(run / "inputs/validation.parquet")
    offsets = [0]
    frequencies = np.zeros(tokenizer.get_vocab_size(), dtype=np.int64)
    with (run / "inputs/token_ids.u32").open("wb") as stream:
        for batch_id, batch in enumerate(parquet.iter_batches(batch_size=256, columns=["text"])):
            documents = tokenizer.encode(batch.column(0).to_pylist(), prepend=bos, num_threads=8)
            for document in documents:
                assert document[0] == bos
                ids = np.asarray(document, dtype="<u4")
                assert ids.max() < len(frequencies)
                ids.tofile(stream)
                frequencies += np.bincount(ids.astype(np.int64), minlength=len(frequencies))
                offsets.append(offsets[-1] + len(ids))
            if batch_id % 32 == 0:
                print(f"Tokenized {len(offsets)-1}/{parquet.metadata.num_rows} documents; {offsets[-1]:,} tokens", flush=True)
    assert len(offsets) - 1 == parquet.metadata.num_rows
    assert offsets[-1] == int(frequencies.sum())
    assert (run / "inputs/token_ids.u32").stat().st_size == offsets[-1] * 4
    np.save(run / "inputs/document_offsets.npy", np.array(offsets, dtype=np.int64))
    np.save(run / "inputs/token_frequencies.npy", frequencies)
    length = config["sequence_length"]
    write_json(run / "inputs/tokenization.json", dict(tokens=offsets[-1], documents=len(offsets)-1,
        corpus_tokens_excluding_inserted_bos=offsets[-1]-(len(offsets)-1), inserted_bos=len(offsets)-1,
        bos_token_id=bos, vocabulary_size=len(frequencies), full_sequences=offsets[-1]//length,
        final_sequence_length=offsets[-1]%length, dtype="little-endian uint32",
        policy=config["token_policy"], context_reset="Every 2048 tokens, position resets to zero; no KV cache; causal attention can span document boundaries within each context."))


if __name__ == "__main__":
    main()
