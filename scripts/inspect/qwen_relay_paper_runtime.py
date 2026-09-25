"""Only the numerical/runtime helpers needed by the paper relay.

Function bodies are copied from the executed experiment sources. No recipient-x
scan or exploratory K/V probe is imported. See qwen_relay_paper.md for provenance.
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import torch


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def outside_layers(n_layers, s, t):
    if not 0 <= s < t < n_layers:
        raise ValueError(f"invalid output-layer interval {s},{t}/{n_layers}")
    return tuple(i for i in range(n_layers) if not s < i <= t)


def clean(text):
    return re.sub(r'<\|[^|]+\|>', '', text).strip()


def hashes(*paths):
    return {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def consistent_cache_prefill(_module, args, kwargs):
    """Match fresh extraction forwards to generation's cache-enabled prefill.

    Never reuse a cache across passes. The caller owns and discards each
    returned cache; generation still owns its own independent decode cache.
    """
    kwargs = dict(kwargs)
    kwargs['use_cache'] = True
    return args, kwargs


def configure_model(model):
    model.float()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    return model.register_forward_pre_hook(consistent_cache_prefill, with_kwargs=True)
