"""Train the 12-block x 4-loop model with per-loop Muon on ONE GPU, matched to the baseline.

Baseline: /hdd/mh3897/cc/looping-runs/b12_r4_10b_20261019 (2 GPUs, DDP). Everything that
defines the run is copied from it: init (model_000000.pt), data and data order, batch,
19,074 steps, LR / momentum / weight-decay schedules, evaluation. The only change is how
Muon combines the 4 loop gradients (--mode).

One GPU reproduces the 2-GPU data order: two loaders see the data exactly as rank 0 and
rank 1 did; each step runs rank 0's 8 microbatches then rank 1's 8, and divides by 16 (what
DDP's gradient average did). Validation also uses both rank views (64 batches each), so
val BPB is directly comparable with the baseline's train.log.

  python train_perloop.py --mode v1 --out /path/to/run
  python train_perloop.py --mode v1 --out /path/to/run --resume 5000
"""
import argparse
import gc
import json
import math
import os
import sys
import time

import torch

from nanochat.common import compute_init, get_peak_flops, COMPUTE_DTYPE
from nanochat.tokenizer import get_tokenizer, get_token_bytes
import nanochat.dataloader as dl
from nanochat.model.gpt_loop import GPTLoop, GPTLoopConfig
from nanochat.model.gpt_base import GPTBase, GPTBaseConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perloop import MODES, PerLoopMuonAdamW, build_twin, sync  # noqa: E402

BASELINE = "/hdd/mh3897/cc/looping-runs/b12_r4_10b_20261019"
BASELINE_CKPT = f"{BASELINE}/base_checkpoints/gpt_loop_b12_r4_e48_10b_seed42"

# Fixed by the baseline (meta_000000.json / train.log). Not command-line options on purpose.
WORLD = 2                  # emulated DDP ranks
DEVICE_BATCH = 16
SEQ_LEN = 2048
TOTAL_BATCH = 524_288
NUM_ITERATIONS = 19_074    # schedule horizon; --stop-after only stops early
ACCUM = TOTAL_BATCH // (WORLD * DEVICE_BATCH * SEQ_LEN)  # 8 microbatches per rank
EMBEDDING_LR, UNEMBEDDING_LR, SCALAR_LR, MATRIX_LR = 0.3, 0.008, 0.5, 0.02
WEIGHT_DECAY, TARGET_PARAM_DATA_RATIO = 0.28, 10.5
WARMUP, WARMDOWN_RATIO, FINAL_LR_FRAC = 40, 0.65, 0.05
EVAL_TOKENS = 4_194_304
EXPECTED_WD = 0.084483     # printed by the baseline: "Scaling weight decay from 0.280000 to 0.084483"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--resume", type=int, default=-1)
    p.add_argument("--stop-after", type=int, default=-1, help="stop after this step (schedules unchanged)")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--save-steps", type=int, nargs="*", default=[10, 50, 100, 250])
    p.add_argument("--stats-every", type=int, default=50, help="per-matrix ratio/cos dump interval")
    p.add_argument("--no-compile", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data: one process, two rank views

def rank_view(rank, make):
    """Run the loader factory as if this process were DDP rank `rank` of WORLD.

    The loader asks get_dist_info() once, when its first batch is requested, so the patch
    only needs to be active for that first next().
    """
    gen = make()
    original = dl.get_dist_info
    dl.get_dist_info = lambda: (True, rank, rank, WORLD)
    try:
        first = next(gen)
    finally:
        dl.get_dist_info = original

    def chained():
        yield first
        yield from gen
    return chained()


def train_loaders(tokenizer, device, states=None):
    return [rank_view(r, lambda r=r: dl.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, DEVICE_BATCH, SEQ_LEN, split="train", device=device,
        resume_state_dict=None if states is None else states[r], exact_resume=True)) for r in range(WORLD)]


@torch.no_grad()
def evaluate_bpb(model, tokenizer, token_bytes, device):
    """nanochat's evaluate_bpb, summed over both rank views exactly as the all-reduce did."""
    steps = EVAL_TOKENS // (DEVICE_BATCH * SEQ_LEN * WORLD)
    nats = torch.zeros((), dtype=torch.float32, device=device)
    nbytes = torch.zeros((), dtype=torch.int64, device=device)
    for r in range(WORLD):
        it = rank_view(r, lambda: dl.tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, DEVICE_BATCH, SEQ_LEN, split="val", device=device))
        for _ in range(steps):
            x, y = next(it)
            loss = model(x, y, loss_reduction="none").view(-1)
            b = token_bytes[y.view(-1)]
            nats += (loss * (b > 0)).sum()
            nbytes += b.sum()
    return nats.item() / (math.log(2) * nbytes.item())


# ---------------------------------------------------------------------------
# Schedules (verbatim from scripts/base_train.py)

def lr_multiplier(it):
    warmdown = round(WARMDOWN_RATIO * NUM_ITERATIONS)
    if it < WARMUP:
        return (it + 1) / WARMUP
    if it <= NUM_ITERATIONS - warmdown:
        return 1.0
    progress = (NUM_ITERATIONS - it) / warmdown
    return progress * 1.0 + (1 - progress) * FINAL_LR_FRAC


def muon_momentum(it):
    warmdown = round(WARMDOWN_RATIO * NUM_ITERATIONS)
    start = NUM_ITERATIONS - warmdown
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    if it >= start:
        progress = (it - start) / warmdown
        return 0.97 * (1 - progress) + 0.90 * progress
    return 0.97


def scaled_weight_decay(master):
    """Same derivation as base_train.py (batch == reference batch, so only the D_ref/D factor)."""
    counts = master.num_scaling_params()
    visits = master.config.n_layer // len(master.transformer.h)
    target_tokens = TARGET_PARAM_DATA_RATIO * (visits * counts["transformer_matrices"] + counts["lm_head"])
    ref_cfg = GPTBaseConfig(sequence_len=SEQ_LEN, vocab_size=master.config.vocab_size, n_layer=12,
                            n_head=master.config.n_head, n_kv_head=master.config.n_kv_head,
                            n_embd=master.config.n_embd, window_pattern="L")
    with torch.device("meta"):
        ref = GPTBase(ref_cfg).num_scaling_params()
    d_ref = TARGET_PARAM_DATA_RATIO * (ref["transformer_matrices"] + ref["lm_head"])
    return WEIGHT_DECAY * (d_ref / target_tokens)


# ---------------------------------------------------------------------------

def load_master(device):
    meta = json.load(open(f"{BASELINE_CKPT}/meta_000000.json"))
    assert meta["world_size"] == WORLD and meta["device_batch_size"] == DEVICE_BATCH
    assert meta["total_batch_size"] == TOTAL_BATCH and meta["user_config"]["num_iterations"] == NUM_ITERATIONS
    with torch.device("meta"):
        master = GPTLoop(GPTLoopConfig(**meta["model_config"]))
    master.to_empty(device=device)
    master.init_weights()  # fills the non-persistent rotary buffers; weights are overwritten below
    return master, meta


def save(out, step, master, optimizer, meta):
    ck = os.path.join(out, "checkpoints")
    os.makedirs(ck, exist_ok=True)
    loaders = meta.pop("_loaders")
    torch.save(master.state_dict(), f"{ck}/model_{step:06d}.pt")
    json.dump(meta, open(f"{ck}/meta_{step:06d}.json", "w"), indent=2)
    # Optimizer + data state only for the latest checkpoint (per-loop state is ~2 GB).
    torch.save(dict(optimizer=optimizer.state_dict(), loaders=loaders, step=step),
               f"{ck}/state_{step:06d}.pt.tmp")
    os.replace(f"{ck}/state_{step:06d}.pt.tmp", f"{ck}/state_{step:06d}.pt")
    for name in os.listdir(ck):
        if name.startswith("state_") and name != f"state_{step:06d}.pt":
            os.remove(os.path.join(ck, name))


def main():
    args = parse()
    assert COMPUTE_DTYPE == torch.bfloat16, "Baseline ran in bf16: set NANOCHAT_DTYPE=bfloat16"
    os.makedirs(args.out, exist_ok=True)
    _, _, _, _, device = compute_init("cuda")
    torch.manual_seed(42)
    peak = get_peak_flops(torch.cuda.get_device_name(0))
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)

    master, base_meta = load_master(device)
    resuming = args.resume >= 0
    src = f"{args.out}/checkpoints/model_{args.resume:06d}.pt" if resuming else f"{BASELINE_CKPT}/model_000000.pt"
    master.load_state_dict(torch.load(src, map_location=device), strict=True, assign=True)
    twin, loops = build_twin(master)
    model = twin if args.no_compile else torch.compile(twin, dynamic=False)
    flops_per_token = master.estimate_flops()
    assert flops_per_token == twin.estimate_flops(), "twin must do the same computation as GPTLoop"

    wd = scaled_weight_decay(master)
    assert abs(wd - EXPECTED_WD) < 1e-6, wd
    base_opt = master.setup_optimizer(unembedding_lr=UNEMBEDDING_LR, embedding_lr=EMBEDDING_LR,
                                      scalar_lr=SCALAR_LR, matrix_lr=MATRIX_LR, weight_decay=wd)
    optimizer = PerLoopMuonAdamW([dict(g) for g in base_opt.param_groups], args.mode, loops)
    del base_opt
    names = {p: n for n, p in master.named_parameters()}

    loader_states = None
    step = 0
    log_state = dict(min_val_bpb=float("inf"), smooth=0.0, train_time=0.0)
    val_bpb = None
    if resuming:
        st = torch.load(f"{args.out}/checkpoints/state_{args.resume:06d}.pt", map_location=device,
                        weights_only=False)
        optimizer.load_state_dict(st["optimizer"])
        loader_states = st["loaders"]
        step = st["step"]
        m = json.load(open(f"{args.out}/checkpoints/meta_{args.resume:06d}.json"))
        assert m["mode"] == args.mode
        log_state, val_bpb = m["log_state"], m["val_bpb"]
    loaders = train_loaders(tokenizer, device, loader_states)
    states = loader_states or [None, None]

    config = dict(mode=args.mode, init=src, baseline=BASELINE, world_emulated=WORLD, device_batch=DEVICE_BATCH,
                  total_batch=TOTAL_BATCH, num_iterations=NUM_ITERATIONS, weight_decay_scaled=wd,
                  flops_per_token=flops_per_token, total_flops=flops_per_token * TOTAL_BATCH * NUM_ITERATIONS,
                  args=vars(args))
    json.dump(config, open(f"{args.out}/config.json", "w"), indent=2)
    print(json.dumps(config, indent=2), flush=True)
    metrics = open(f"{args.out}/metrics.jsonl", "a")
    stats_file = open(f"{args.out}/matrix_stats.jsonl", "a")
    stop = NUM_ITERATIONS if args.stop_after < 0 else args.stop_after
    first = True

    while True:
        last = step == stop
        if args.eval_every > 0 and (last or step % args.eval_every == 0):
            model.eval()
            val_bpb = evaluate_bpb(model, tokenizer, token_bytes, device)
            model.train()
            log_state["min_val_bpb"] = min(log_state["min_val_bpb"], val_bpb)
            print(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}", flush=True)
            metrics.write(json.dumps(dict(step=step, val_bpb=val_bpb)) + "\n")
            metrics.flush()
        if step != max(args.resume, 0) and (last or step in args.save_steps or step % args.save_every == 0):
            meta = dict(base_meta, step=step, val_bpb=val_bpb, mode=args.mode, log_state=log_state,
                        perloop_config=config, _loaders=states)
            meta["dataloader_state_dict"] = {k: v for k, v in states[0].items() if k != "doc_buffer"}
            save(args.out, step, master, optimizer, meta)
        if last:
            break

        torch.cuda.synchronize()
        t0 = time.time()
        lrm, mom = lr_multiplier(step), muon_momentum(step)
        wd_t = wd * 0.5 * (1 + math.cos(math.pi * step / NUM_ITERATIONS))
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"], group["weight_decay"] = mom, wd_t
        train_loss = torch.zeros((), device=device)
        for r in range(WORLD):  # rank 0's microbatches, then rank 1's: DDP's average over 16
            for _ in range(ACCUM):
                x, y, states[r] = next(loaders[r])
                loss = model(x, y)
                train_loss += loss.detach() / (WORLD * ACCUM)
                (loss / (WORLD * ACCUM)).backward()
        optimizer.step()
        sync(loops)
        model.zero_grad(set_to_none=True)
        assert all(p.grad is None for p in master.transformer.h.parameters())
        ok = torch.stack(optimizer.finite).all() & torch.isfinite(train_loss)
        if optimizer.stats:
            ratio = torch.cat([r for _, r, _ in optimizer.stats])
            cos = torch.cat([c for _, _, c in optimizer.stats])
            summary = torch.stack([ratio.mean(), ratio.min(), ratio.max(), cos.mean(), cos.min(), cos.max()])
            values = torch.cat([train_loss[None], ok[None].float(), summary]).tolist()
        else:
            values = torch.stack([train_loss, ok.float()]).tolist()
        loss_f, ok_f = values[0], values[1]
        if ok_f != 1.0:
            raise FloatingPointError(f"Non-finite loss/gradient/update at step {step}")
        torch.cuda.synchronize()
        dt = time.time() - t0

        row = dict(step=step, loss=loss_f, lrm=lrm, momentum=mom, wd=wd_t, dt=dt)
        if optimizer.stats:
            row.update(zip(["ratio_mean", "ratio_min", "ratio_max", "cos_mean", "cos_min", "cos_max"], values[2:]))
            if step % args.stats_every == 0 or step < 10:
                per = {}
                for params, r_, c_ in optimizer.stats:
                    for p, rv, cv in zip(params, r_.tolist(), c_.tolist()):
                        per[names[p]] = [rv, cv]
                stats_file.write(json.dumps(dict(step=step, ratio_cos=per)) + "\n")
                stats_file.flush()
        metrics.write(json.dumps(row) + "\n")
        metrics.flush()

        if step > 10:
            log_state["train_time"] += dt
        beta = 0.9
        log_state["smooth"] = beta * log_state["smooth"] + (1 - beta) * loss_f
        smooth = log_state["smooth"] / (1 - beta ** (step + 1))
        mfu = 100 * flops_per_token * TOTAL_BATCH / dt / peak
        extra = f" | B/A: {row['ratio_mean']:.3f} | cos(A,B): {row['cos_mean']:.4f}" if optimizer.stats else ""
        print(f"step {step:05d}/{NUM_ITERATIONS:05d} | loss: {smooth:.6f} | raw_loss: {loss_f:.6f} | lrm: {lrm:.2f} "
              f"| dt: {dt * 1000:.0f}ms | tok/sec: {int(TOTAL_BATCH / dt):,} | mfu: {mfu:.2f}{extra} "
              f"| rg: {states[0]['rg_idx']}/{states[1]['rg_idx']} | time: {log_state['train_time'] / 60:.1f}m",
              flush=True)
        step += 1
        if first:
            gc.collect()
            gc.freeze()
            gc.disable()
            first = False
        elif step % 5000 == 0:
            gc.collect()

    json.dump(dict(final_step=step, val_bpb=val_bpb, **log_state), open(f"{args.out}/done.json", "w"), indent=2)


if __name__ == "__main__":
    main()
