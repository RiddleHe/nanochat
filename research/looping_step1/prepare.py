"""Create disjoint, deterministic packed data, then freeze all source and inputs."""
import json
import os
from pathlib import Path
import hashlib
import numpy as np
import pyarrow.parquet as pq
from common import BASE, BASE_MANIFEST, CHECKPOINT, ROOT, SOURCE, SETTINGS, now, sha, verify_source, write_json
from nanochat.tokenizer import get_tokenizer


def main():
    baseline = verify_source()
    dest = ROOT / "data"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "manifest.json").exists():
        raise RuntimeError("Data already prepared; refusing to overwrite")
    tokenizer = get_tokenizer()
    assert tokenizer.get_vocab_size() <= 65536
    source_files = {}
    expected = {f["name"]: f["sha256"] for f in baseline["dataset_files"]}
    seen = set()

    def documents(indices):
        for index in indices:
            path = BASE / "base_data_climbmix" / f"shard_{index:05d}.parquet"
            digest = sha(path)
            assert digest == expected[path.name]
            source_files[path.name] = digest
            row = 0
            for record in pq.ParquetFile(path).iter_batches(batch_size=128, columns=["text"], use_threads=False):
                texts = record.column(0).to_pylist()
                ids = tokenizer.encode(texts, prepend=tokenizer.get_bos_token_id(), num_threads=4)
                for text, encoded in zip(texts, ids):
                    text_hash = hashlib.sha256(text.encode()).hexdigest()
                    source = dict(file=path.name, row=row, text_sha256=text_hash)
                    row += 1
                    if text_hash in seen or len(encoded) < 2:
                        continue
                    seen.add(text_hash)
                    yield encoded, source

    def pack(name, n, iterator):
        path = dest / f"{name}.npy"
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite {path}")
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=(n, 2049))
        flat = arr.reshape(-1)
        cursor = 0
        used = 0
        with open(dest / f"{name}_documents.jsonl", "x") as log:
            while cursor < flat.size:
                ids, source = next(iterator)
                take = min(len(ids), flat.size - cursor)
                flat[cursor:cursor + take] = ids[:take]
                source.update(token_start=cursor, tokens_used=take, document_tokens=len(ids))
                log.write(json.dumps(source) + "\n")
                cursor += take
                used += 1
        arr.flush()
        del flat, arr
        print(json.dumps(dict(split=name, sequences=n, documents=used, time=now())), flush=True)
        return dict(sequences=n, prediction_tokens=n * 2048, stored_tokens=n * 2049,
                    documents=used, sha256=sha(path),
                    documents_sha256=sha(dest / f"{name}_documents.jsonl"))

    diagnostic_docs = documents([300])
    validation_docs = documents([6542])
    splits = {}
    for name, n, iterator in [
        ("gradient", 64, diagnostic_docs),
        ("sensitivity", 128, diagnostic_docs),
        ("development", 2048, validation_docs),
        ("confirmation", 2048, validation_docs),
        ("train", 51200, documents(range(301, 320))),
    ]:
        splits[name] = pack(name, n, iterator)
    write_json(dest / "manifest.json", dict(
        created_utc=now(), splits=splits, source_files=source_files,
        packing="BOS-delimited document stream, consecutive nonoverlapping 2049-token records; x=first2048,y=last2048",
        selection="Ordered source rows; exact document text hashes deduplicated across all experiment splits. Remainder of last document is discarded at each split boundary.",
        limitations="Not a near-duplicate audit. Training uses original ClimbMix training partition, not an independent pretraining corpus.",
        train_permutation_seed=42,
    ))
    print("DATA_READY", flush=True)


def freeze():
    assert not (ROOT / "experiment_manifest.json").exists()
    sources = {p.name: sha(p) for p in sorted(SOURCE.glob("*.py"))}
    for test in ("tests_cpu.json", "tests_gpu.json"):
        report = json.loads((ROOT / test).read_text())
        assert report["passed"], test
        assert report["source_hashes"] == sources, f"Code changed since {test}"
    write_json(ROOT / "experiment_manifest.json", dict(
        created_utc=now(), settings=SETTINGS, source_hashes=sources,
        baseline_manifest_sha256=BASE_MANIFEST,
        checkpoint=str(CHECKPOINT / "model_019074.pt"),
        checkpoint_sha256=sha(CHECKPOINT / "model_019074.pt"),
        data_manifest_sha256=sha(ROOT / "data" / "manifest.json"),
        runtime=json.loads((ROOT / "tests_gpu.json").read_text())["runtime"],
    ))
    print("DATA_AND_SOURCE_FROZEN", flush=True)


if __name__ == "__main__":
    import sys
    freeze() if "--freeze" in sys.argv else main()
