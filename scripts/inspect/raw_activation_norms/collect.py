"""Collect exact per-token source norms through read-only forward hooks."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from nanochat.checkpoint_manager import build_model
from nanochat.flash_attention import USE_FA3
from .common import now, read_json, sha256, verify_run, write_json


class NormCapture:
    def __init__(self, model):
        self.model = model
        self.values = [None] * (1 + 2 * len(model.transformer.h))
        self.handles = [model.transformer.wte.register_forward_hook(self.hook(0))]
        for layer, block in enumerate(model.transformer.h):
            self.handles.append(block.attn.register_forward_hook(self.hook(1+2*layer, is_attention=True)))
            self.handles.append(block.mlp.register_forward_hook(self.hook(2+2*layer)))

    def hook(self, index, is_attention=False):
        def capture(module, inputs, output):
            value = output[0] if is_attention and isinstance(output, tuple) else output
            assert self.values[index] is None, 'Source module called more than once'
            assert value.dtype == torch.bfloat16
            self.values[index] = torch.linalg.vector_norm(value.detach().float(), dim=-1)
        return capture

    def stack(self):
        assert all(v is not None for v in self.values)
        result = torch.stack(self.values, dim=-1)
        self.values = [None] * len(self.values)
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def collect(run):
    manifest = verify_run(run)
    config = manifest["config"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == manifest["gpu_uuid"]
    assert torch.cuda.device_count() == 1 and USE_FA3
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, _, _ = build_model(str(run / "inputs/checkpoint"), config["checkpoint_step"], torch.device("cuda:0"), "eval")
    model.requires_grad_(False)
    write_json(run / "environment.json", dict(python=sys.version, torch=torch.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), gpu_uuid=manifest["gpu_uuid"],
        use_fa3=USE_FA3, numpy=np.__version__, tf32=False))
    ids = np.memmap(run / "inputs/token_ids.u32", mode="r", dtype="<u4")
    n = len(ids)
    length, batch_size = config["sequence_length"], config["batch_size"]
    assert n == manifest["tokenization"]["tokens"]
    # Validate two full contexts with and without hooks, plus embedding table identity.
    checks = []
    for start in (0, (n//length-1)*length):
        token_ids = torch.from_numpy(np.array(ids[start:start+length], dtype=np.int64)).unsqueeze(0).cuda()
        reference = model(token_ids)
        capture = NormCapture(model)
        actual = model(token_ids)
        norms = capture.stack()
        capture.close()
        error = float((actual-reference).abs().max())
        assert torch.equal(actual, reference), error
        table_norm = model.transformer.wte.weight.to(torch.bfloat16).float().norm(dim=-1)[token_ids]
        assert torch.equal(norms[:,:,0], table_norm)
        checks.append(dict(start=start, tokens=token_ids.numel(), logits_max_abs_error=error,
                           logits_bitwise_equal=True, embedding_norms_equal_direct_table_lookup=True))
        del actual, reference, norms
    write_json(run / "validation/forward.json", dict(passed=True, checks=checks))
    capture = NormCapture(model)
    initial = time.monotonic()
    manifest_hash = sha256(run / "manifest.json")
    chunk_id, start = 0, 0
    while start < n:
        full = min(batch_size, (n-start)//length)
        batch_shape = (full, length) if full else (1, n-start)
        stop = start + batch_shape[0]*batch_shape[1]
        path = run / "tokens" / f"chunk_{chunk_id:06d}.npz"
        record_path = path.with_suffix(".json")
        if record_path.exists():
            record = read_json(record_path)
            assert record["manifest_sha256"] == manifest_hash and record["sha256"] == sha256(path)
            assert (record["start"], record["stop"]) == (start, stop)
        else:
            token_ids = torch.from_numpy(np.array(ids[start:stop], dtype=np.int64)).reshape(batch_shape).cuda()
            logits = model(token_ids)
            norms = capture.stack().reshape(-1, 49).cpu().numpy()
            assert np.isfinite(norms).all() and (norms >= 0).all()
            assert torch.isfinite(logits).all().item()
            del logits
            temp = path.with_suffix(".npz.tmp")
            with temp.open("wb") as stream:
                np.savez(stream, norms=norms)
            temp.replace(path)
            write_json(record_path, dict(start=start, stop=stop, shape=list(batch_shape),
                sha256=sha256(path), manifest_sha256=manifest_hash, tokens_sha256=manifest["inputs_hashes"]["token_ids.u32"]))
        start = stop
        chunk_id += 1
        elapsed = time.monotonic()-initial
        write_json(run / "status.json", dict(state="collecting", completed_tokens=start, total_tokens=n,
            completed_chunks=chunk_id, elapsed_seconds=elapsed, updated_utc=now()))
        if chunk_id % 10 == 0 or start == n:
            print(f"COLLECT {start:,}/{n:,} tokens ({100*start/n:.2f}%), {elapsed:.1f}s, {start/max(elapsed,1):.0f} tokens/s", flush=True)
    capture.close()
    write_json(run / "collection.json", dict(tokens=n, chunks=chunk_id, seconds=time.monotonic()-initial,
        gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30, finished_utc=now()))
    write_json(run / "status.json", dict(state="collected", completed_tokens=n, completed_chunks=chunk_id))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    run = parser.parse_args().run_dir
    try:
        collect(run)
    except BaseException as error:
        write_json(run / "status.json", dict(state="failed", error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
