#!/bin/bash
# Exp 1 (gradient compression per entropy bucket) + Exp 2 (correct-vs-incorrect
# direction at top-20% entropy tokens).
#
# Uses 4 GPUs: two shards per model, each shard = 64 prompts × 8 rollouts.
# Shards for the same model use distinct seeds so they sample disjoint prompts.
# After all shards finish, merge into one dir per model and run analyses.
#
set -euo pipefail

cd /hdd/mh3897/cc/nanochat
export PYTHONPATH=/hdd/mh3897/cc/nanochat:${PYTHONPATH:-}
VENV_PY=/hdd/mh3897/cc/nanochat_new/.venv/bin/python

BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TRAINED_MODEL="/hdd/mh3897/cc/nanochat/.nanochat/rl/rl_new_code_ge2_train_split_500/checkpoints/Qwen_Qwen2.5-1.5B-Instruct_dapo"

OUT_ROOT=/hdd/mh3897/cc/nanochat/.nanochat/probe
OUT_BASE_S0="$OUT_ROOT/exp12_base_s0"
OUT_BASE_S1="$OUT_ROOT/exp12_base_s1"
OUT_TRAINED_S0="$OUT_ROOT/exp12_trained_s0"
OUT_TRAINED_S1="$OUT_ROOT/exp12_trained_s1"
OUT_BASE="$OUT_ROOT/exp12_base"
OUT_TRAINED="$OUT_ROOT/exp12_trained"

mkdir -p "$OUT_BASE_S0" "$OUT_BASE_S1" "$OUT_TRAINED_S0" "$OUT_TRAINED_S1" "$OUT_BASE" "$OUT_TRAINED"

run_probe() {
    local gpu=$1 model=$2 seed=$3 outdir=$4
    echo "[run] GPU $gpu seed=$seed -> $outdir"
    CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $VENV_PY nanorl/scripts/probe_grad_vectors.py \
        --model-path "$model" \
        --output-dir "$outdir" \
        --num-prompts 64 --num-samples 8 \
        --layer 14 --param-suffix mlp.down_proj.weight --row-idx 0 \
        --seed $seed \
        > "$outdir/probe.log" 2>&1
}

# Launch 4 probes in parallel on GPUs 4-7
run_probe 4 "$BASE_MODEL"    0 "$OUT_BASE_S0"    &
PID_A=$!
run_probe 5 "$BASE_MODEL"    1 "$OUT_BASE_S1"    &
PID_B=$!
run_probe 6 "$TRAINED_MODEL" 0 "$OUT_TRAINED_S0" &
PID_C=$!
run_probe 7 "$TRAINED_MODEL" 1 "$OUT_TRAINED_S1" &
PID_D=$!

echo "[run] launched PIDs: $PID_A (GPU4 base-s0) $PID_B (GPU5 base-s1) $PID_C (GPU6 trained-s0) $PID_D (GPU7 trained-s1)"
wait $PID_A; echo "[run] base-s0 done"
wait $PID_B; echo "[run] base-s1 done"
wait $PID_C; echo "[run] trained-s0 done"
wait $PID_D; echo "[run] trained-s1 done"

echo "[run] merging shards..."
$VENV_PY nanorl/scripts/merge_grad_vector_shards.py "$OUT_BASE"    "$OUT_BASE_S0"    "$OUT_BASE_S1"
$VENV_PY nanorl/scripts/merge_grad_vector_shards.py "$OUT_TRAINED" "$OUT_TRAINED_S0" "$OUT_TRAINED_S1"

mkdir -p "$OUT_ROOT/exp12_figures"
echo "[run] running Exp 1 and Exp 2 analyses..."
$VENV_PY nanorl/scripts/analyze_exp1_compression.py \
    base="$OUT_BASE" \
    trained="$OUT_TRAINED" 2>&1 | tee "$OUT_ROOT/exp12_figures/exp1.txt" || true

$VENV_PY nanorl/scripts/analyze_exp2_direction.py "$OUT_TRAINED" \
    2>&1 | tee "$OUT_ROOT/exp12_figures/exp2.txt" || true

$VENV_PY nanorl/scripts/plot_exp12_intuitive.py 2>&1 | tee "$OUT_ROOT/exp12_figures/plot.txt" || true

echo "[run] ALL DONE. figures:"
ls -la "$OUT_ROOT/exp12_figures/"
