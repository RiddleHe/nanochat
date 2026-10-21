"""Read-only baseline loading and shared, explicitly frozen experiment settings."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

SOURCE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("LOOPING_STEP1_DIR", SOURCE / "runs/local")).expanduser().resolve()
BASE = Path(os.environ.get("LOOPING_BASELINE_DIR", SOURCE / "baseline")).expanduser().resolve()
BASE_MANIFEST = "49219d913e3530d392305f8e535bee4275e77fba694e8c2deb3a2a1b23ff7d9b"
CHECKPOINT = BASE / "base_checkpoints/gpt_loop_b24_r4_e96_10b_seed42"
os.environ["NANOCHAT_BASE_DIR"] = str(BASE)
os.environ.setdefault("NANOCHAT_DTYPE", "bfloat16")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
# CPU algebra tests use this checkout; real runs require the verified frozen snapshot.
sys.path.insert(0, str(BASE / "code" if (BASE / "code").is_dir() else SOURCE.parents[1]))

SETTINGS = dict(
    baseline_step=19074, physical_blocks=24, loops=4, width=1536,
    sequence_length=2048, microbatch=2, gradient_batches=32,
    sensitivity_sequences=128, eval_sequences=2048,
    train_sequences=51200, updates=200, accumulation=128,
    tokens_per_update=524288, train_tokens=104857600, seed=42, data_seed=42,
    learning_rate=1e-4, final_learning_rate=1e-5, warmup=20,
    adam_betas=[0.9, 0.95], adam_eps=1e-8, weight_decay=0.0,
    gradient_clip=None, lora_dropout=0.0, alpha_over_rank=1.0,
    eval_steps=[0, 50, 100, 200], save_steps=[0, 50, 100, 200],
    precision="bfloat16", execution="eager_FA3", deterministic_backward=True, gpus=[0, 1, 2],
    rank_small={"qk": 20, "vo": 20, "mlp": 8},
    intervention_scales=[0.9, 1.1], initial_seeds=1,
    scope="exploratory final-checkpoint study, not an early-prediction or matched-total-FLOPs claim",
)
TARGETS = {
    "q": "attn.c_q", "k": "attn.c_k", "v": "attn.c_v", "o": "attn.c_proj",
    "up": "mlp.c_fc", "down": "mlp.c_proj",
}
GROUPS = {"qk": ("q", "k"), "vo": ("v", "o"), "mlp": ("up", "down")}
VARIANTS = ("shared_small", "independent", "shared_large")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def append_json(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, allow_nan=False) + "\n")
        f.flush()


def verify_source():
    if not (BASE / "manifest.json").is_file():
        raise FileNotFoundError("Set LOOPING_BASELINE_DIR to the original frozen baseline directory")
    assert sha(BASE / "manifest.json") == BASE_MANIFEST
    manifest = json.loads((BASE / "manifest.json").read_text())
    for relative, expected in manifest["source_hashes"].items():
        assert sha(BASE / "code" / relative) == expected, relative
    for relative, expected in manifest["tokenizer_sha256"].items():
        assert sha(BASE / "tokenizer" / relative) == expected, relative
    own = ROOT / "experiment_manifest.json"
    if own.exists():
        frozen = json.loads(own.read_text())
        assert frozen["settings"] == SETTINGS
        for name, digest in frozen["source_hashes"].items():
            assert sha(SOURCE / name) == digest, name
    return manifest


def runtime_record():
    import torch
    import nanochat.flash_attention as attention
    module = Path(attention._fa3.__file__)
    return dict(torch=torch.__version__, cuda=torch.version.cuda, attention_files={
        str(p): sha(p) for p in module.parent.rglob("*")
        if p.is_file() and p.suffix in (".py", ".so", ".json") and "__pycache__" not in p.parts
    })


def load_model(device="cuda"):
    import torch
    from nanochat.checkpoint_manager import build_model
    import nanochat.flash_attention as attention
    torch.set_num_threads(4)
    torch.manual_seed(SETTINGS["seed"])
    torch.set_float32_matmul_precision("highest")
    if device == "cuda":
        assert attention.USE_FA3, "Do not silently switch attention implementation"
        torch.use_deterministic_algorithms(True)

        def deterministic_attention(q, k, v, causal=False, window_size=(-1, -1)):
            return attention._fa3.flash_attn_func(
                q, k, v, causal=causal, window_size=window_size, deterministic=True)

        attention.flash_attn.flash_attn_func = deterministic_attention
        if (ROOT / "experiment_manifest.json").exists():
            frozen = json.loads((ROOT / "experiment_manifest.json").read_text())
            assert runtime_record() == frozen["runtime"], "Runtime/kernel drift"
    model, tokenizer, meta = build_model(str(CHECKPOINT), 19074, torch.device(device), "eval")
    assert len(model.transformer.h) == 24 and model.config.num_loops == 4
    assert model.config.n_embd == 1536 and model.config.n_kv_head == 12
    assert sum(p.numel() for p in model.parameters()) == 780140569
    return model, tokenizer, meta


def matrices(model):
    for layer, block in enumerate(model.transformer.h):
        for short, path in TARGETS.items():
            module = block.get_submodule(path)
            yield f"L{layer + 1:02d}.{short}", module


def parameter_digest(parameters):
    """Bitwise evidence that interventions never change frozen model tensors."""
    import torch
    h = hashlib.sha256()
    with torch.no_grad():
        for name, p in parameters:
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def data(name):
    import numpy as np
    record = json.loads((ROOT / "data" / "manifest.json").read_text())["splits"][name]
    path = ROOT / "data" / f"{name}.npy"
    assert sha(path) == record["sha256"], name
    return np.load(path, mmap_mode="r")


def batch(array, indices, device="cuda"):
    import numpy as np
    import torch
    t = torch.from_numpy(np.array(array[indices], dtype=np.int64, copy=True)).to(device)
    return t[:, :-1].contiguous(), t[:, 1:].contiguous()


def token_bytes(device="cuda"):
    import torch
    return torch.load(BASE / "tokenizer/token_bytes.pt", map_location=device, weights_only=True)


def summarize_rows(rows):
    nats = sum(r["nats"] for r in rows)
    byte_nats = sum(r["byte_nats"] for r in rows)
    nbytes = sum(r["bytes"] for r in rows)
    ntokens = sum(r["tokens"] for r in rows)
    return dict(ce=nats / ntokens, bpb=byte_nats / (math.log(2) * nbytes),
                tokens=ntokens, bytes=nbytes, sequences=len(rows))


def evaluate(model, array, reset=None, max_sequences=None):
    import torch
    byte_map = token_bytes()
    rows = []
    model.eval()
    count = len(array) if max_sequences is None else min(len(array), max_sequences)
    with torch.inference_mode():
        for start in range(0, count, 2):
            x, y = batch(array, slice(start, min(start + 2, count)))
            if reset:
                reset()
            losses = model(x, y, loss_reduction="none").reshape_as(y).float()
            if not torch.isfinite(losses).all():
                raise RuntimeError("Nonfinite evaluation loss")
            sizes = byte_map[y].long()
            values = torch.stack([
                losses.sum(1), (losses * (sizes > 0)).sum(1), sizes.sum(1).double()
            ], dim=1).cpu().tolist()
            for j, (nats, bnats, nbytes) in enumerate(values):
                rows.append(dict(sequence=start + j, nats=nats, byte_nats=bnats,
                                 bytes=int(nbytes), tokens=y.shape[1]))
    return {"summary": summarize_rows(rows), "rows": rows}
