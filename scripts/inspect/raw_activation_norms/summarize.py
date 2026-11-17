"""Validate every norm chunk and derive population moments in FP64 on CPU."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import read_json, sha256, verify_run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    run = parser.parse_args().run_dir
    manifest = verify_run(run)
    manifest_hash = sha256(run / "manifest.json")
    count, mean, m2 = 0, np.zeros(49), np.zeros(49)
    frequencies = np.zeros(manifest["tokenization"]["vocabulary_size"], dtype=np.int64)
    ids = np.memmap(run / "inputs/token_ids.u32", mode="r", dtype="<u4")
    collection = read_json(run / "collection.json")
    assert len(list((run / "tokens").glob("chunk_*.json"))) == collection["chunks"]
    for i in range(collection["chunks"]):
        path = run / "tokens" / f"chunk_{i:06d}.npz"
        record = read_json(path.with_suffix(".json"))
        assert record["sha256"] == sha256(path) and record["manifest_sha256"] == manifest_hash
        assert record["tokens_sha256"] == manifest["inputs_hashes"]["token_ids.u32"]
        assert record["start"] == count
        with np.load(path) as chunk:
            x = chunk["norms"].astype(np.float64)
        b = len(x)
        assert x.shape == (record["stop"]-count,49)
        assert np.isfinite(x).all() and (x >= 0).all()
        batch_mean = x.mean(axis=0)
        batch_m2 = np.square(x-batch_mean).sum(axis=0)
        delta = batch_mean-mean
        m2 += batch_m2 + delta*delta*(count*b/(count+b))
        mean += delta*(b/(count+b))
        frequencies += np.bincount(np.asarray(ids[count:count+b], dtype=np.int64), minlength=len(frequencies))
        count += b
    assert count == len(ids) == collection["tokens"]
    np.testing.assert_array_equal(frequencies, np.load(run / "inputs/token_frequencies.npy"))
    offsets = np.load(run / "inputs/document_offsets.npy")
    assert offsets[0] == 0 and offsets[-1] == count and np.all(np.diff(offsets) > 0)
    assert np.all(ids[offsets[:-1]] == manifest["tokenization"]["bos_token_id"])
    assert len(offsets)-1 == manifest["tokenization"]["documents"]
    variance = m2/count
    std = np.sqrt(variance)
    # Independent frequency-weighted CPU FP64 embedding check.
    state = torch.load(run / f"inputs/checkpoint/model_{manifest['config']['checkpoint_step']:06d}.pt", map_location="cpu", weights_only=True, mmap=True)
    table_norm = state["transformer.wte.weight"].to(torch.bfloat16).double().norm(dim=-1).numpy()
    embedding_mean = float(frequencies @ table_norm/count)
    embedding_var = float(frequencies @ np.square(table_norm-embedding_mean)/count)
    np.testing.assert_allclose(mean[0], embedding_mean, rtol=2e-6)
    np.testing.assert_allclose(variance[0], embedding_var, rtol=2e-5)
    sources = [dict(index=0, family="embedding", layer=-1, label="Emb", boundary=manifest["config"]["embedding_boundary"])]
    for layer in range(24):
        for family in ("attention", "mlp"):
            sources.append(dict(index=len(sources),family=family,layer=layer,
                                label=f"{family[0].upper()}{layer}",boundary=manifest["config"][family+"_boundary"]))
    np.savez(run / "derived/statistics.npz", mean=mean, variance=variance, std=std, count=np.full(49,count,dtype=np.int64))
    write_json(run / "derived/schema.json", dict(sources=sources, axis="source", dimensions=1536,
        mean_units="activation units", variance_units="activation units squared", variance_ddof=0,
        norm_definition="full-vector L2 norm per token, no centering; aggregate over token occurrences"))
    forward = read_json(run / "validation/forward.json")
    assert forward["passed"]
    validation = dict(passed=True, all_chunk_hashes_verified=True, gapless_single_pass=True,
        token_frequency_counts_match=True, all_document_bos_present=True, tokens=count,
        documents=len(offsets)-1, forward=forward, embedding_reference_mean=embedding_mean,
        embedding_reference_variance=embedding_var, embedding_mean_abs_error=abs(mean[0]-embedding_mean),
        embedding_variance_abs_error=abs(variance[0]-embedding_var))
    write_json(run / "validation/summary.json", validation)
    write_json(run / "results.json", dict(state="complete", tokens=count, documents=len(offsets)-1,
        statistics_sha256=sha256(run / "derived/statistics.npz"), schema_sha256=sha256(run / "derived/schema.json"),
        collection=collection, embedding=dict(mean=mean[0],std=std[0],variance=variance[0]),
        validation_sha256=sha256(run / "validation/summary.json")))
    write_json(run / "status.json", dict(state="complete", tokens=count))
    print(f"COMPLETE {count:,} tokens; Emb mean={mean[0]:.6f}, std={std[0]:.6f}", flush=True)


if __name__ == "__main__":
    main()
