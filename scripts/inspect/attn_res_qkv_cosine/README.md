# AttnResQKV source cosine

Measures cosine between each forward-cast Q/K/V depth-routing query and each
raw source in the bank (normalized Emb, raw attention outputs, raw MLP outputs).
The model forward remains BF16; analysis uses FP32 and token aggregation FP64.
Queries here are depth-routing vectors, not token queries projected by W_Q.

`prepare.py` freezes the checkpoint reference, metadata, identical saved tokens,
training model code, collector, dependencies, environment, and selected GPU UUID.
It requires the validated routing reference in `config.json`. Collection checks
exact reference weights, exact mixture replay, and unchanged complete logits for
every sequence. Query/source zero norms yield NaN cosine and zero valid counts.

Each token chunk stores `cosine_LL`, `key_cosine_LL`, and actual BF16 `scores_LL`
(losslessly stored in FP32), each shaped `[tokens,3,2*L+1]`; `source_norm_LL` and
`key_norm_LL` shaped `[tokens,2*L+1]`; and `query_norm_LL` shaped `[3]`.
`key_cosine` uses the actual BF16 RMS-normalized key, while `cosine` uses the raw
source. The exact score identity with actual keys is checked in FP32, separately
from BF16 score rounding. The ideal RMS key norm includes dtype-dependent epsilon.

From `/hdd/mh3897/nanochat-bank-of-values`, prepare and collect on an idle GPU:

```bash
PYTHONDONTWRITEBYTECODE=1 /hdd/mh3897/nanochat/.venv/bin/python \
  -m scripts.inspect.attn_res_qkv_cosine.prepare --gpu 0 --launch
```

Omit `--launch` to freeze only. `collect --run-dir /absolute/run/path` routes to
that run's frozen collector/environment. `summarize --run-dir /absolute/run/path`
rebuilds CPU summaries from verified chunks. Plot separately:

```bash
/hdd/mh3897/nanochat/.venv/bin/python \
  -m scripts.inspect.attn_res_qkv_cosine.plot_with_embedding \
  --run-dir /absolute/run/path --output /absolute/thread/path/source-cosine.html
```

The light-mode inline view shows Q/K/V simultaneously with receiving layers on Y
and Emb + attention sources on X. It uses a shared signed linear [-1,1] scale,
with white at zero, hatching for unavailable sources and gray for undefined
cosines. Hover/click retains token std, valid count and query norm. MLP remains in
collected records but is omitted from this view. Plot code, data, settings, and
hashes are archived in `visualization/attention_embedding_cosine/` independently
of collection code. Browser checks cover desktop/mobile layout and hover/pinning.

For MLP sources only, use `scripts.inspect.attn_res_qkv_cosine.plot --family mlp`
with the same `--run-dir` and a separate `--output`. This reads saved summaries,
uses the same [-1,1] scale, and archives under `visualization/mlp_cosine/`.
