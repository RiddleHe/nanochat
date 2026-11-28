"""Pinned two-arm looping experiment. Explicit data/RNG resume; ordinary MuonAdamW.

Run with torchrun --standalone --nproc_per_node=2 -m scripts.train_loop_select
--config <frozen.json> --output <new-run-dir>. --stop-after is for pilots only;
it does not shorten the optimizer's scheduled training horizon.
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
import torch.distributed as dist
from dataclasses import asdict

from nanochat.common import compute_init, compute_cleanup, COMPUTE_DTYPE
from nanochat.model.gpt_base import GPTBaseConfig
from nanochat.model.gpt_base_no_embed_norm import GPTBaseNoEmbedNorm
from nanochat.model.gpt_loop_select import GPTLoopSelect, GPTLoopSelectConfig
from nanochat.dataloader_exact import ExactBOSBestFitDataLoader
from nanochat.training_state_exact import capture_rank_training_state, restore_rank_training_state
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.flash_attention import USE_FA3


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def digest_tensors(state):
    sha = hashlib.sha256()
    for name, t in sorted(state.items()):
        sha.update(name.encode())
        sha.update(str(t.dtype).encode())
        sha.update(str(tuple(t.shape)).encode())
        sha.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return sha.hexdigest()


def main(model_class=GPTLoopSelect, config_class=GPTLoopSelectConfig):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--resume-checkpoint")
    ap.add_argument("--stop-after", type=int)
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")
    cfg = json.loads(Path(args.config).read_text())
    if hasattr(model_class, "validate_training_config"):
        model_class.validate_training_config(cfg)
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    _, rank, local_rank, world, device = compute_init(cfg.get("device_type", "cuda"))
    assert world == cfg["world_size"], (world, cfg["world_size"])
    assert str(COMPUTE_DTYPE).split(".")[-1] == cfg["dtype"]
    if cfg.get("require_fa3", False) and not USE_FA3:
        raise RuntimeError("Frozen CUDA experiment requires FlashAttention3")
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    if device.type == "cuda": torch.cuda.manual_seed(cfg["seed"])
    out = Path(args.output)
    if rank == 0:
        if out.exists() and (out / "config.json").exists() and not args.resume_checkpoint:
            raise RuntimeError("Run already exists; refusing overwrite or implicit restart")
        out.mkdir(parents=True, exist_ok=True)
        if (out / "config.json").exists():
            assert json.loads((out / "config.json").read_text()) == cfg
        else: atomic_json(out / "config.json", cfg)
    if world > 1: dist.barrier()
    def event(kind, **fields):
        if rank == 0:
            row = dict(event=kind, time=time.time(), **fields)
            with (out / "events.jsonl").open("a") as f: f.write(json.dumps(row, allow_nan=False)+"\n")
            print(json.dumps(row, allow_nan=False), flush=True)
    tokenizer = get_tokenizer()
    mc = config_class(**cfg["model"], vocab_size=tokenizer.get_vocab_size())
    with torch.device("meta"): model = model_class(mc)
    model.to_empty(device=device)
    model.init_weights()
    init_sha = digest_tensors({k:v for k,v in model.state_dict().items() if not k.startswith("selector_")})
    if world > 1:
        init_hashes = [None] * world
        dist.all_gather_object(init_hashes, init_sha)
        assert len(set(init_hashes)) == 1, "Backbone initialization differs across ranks"
    counts = model.num_scaling_params()
    fpt = model.estimate_flops()
    batch = cfg["total_batch_size"]
    horizon = round(cfg["target_flops"] / (fpt*batch))
    if "num_iterations" in cfg: horizon = cfg["num_iterations"]
    assert horizon > 0
    B, T = cfg["device_batch_size"], mc.sequence_len
    assert batch % (B*T*world) == 0
    accum = batch // (B*T*world)
    # Preserve Muyu base_train's depth/batch scaling, independent of repeated uses.
    with torch.device("meta"):
        ref = GPTBaseNoEmbedNorm(GPTBaseConfig(n_layer=12,n_embd=768,n_head=6,n_kv_head=6,vocab_size=mc.vocab_size))
    ref_counts = ref.num_scaling_params()
    ref_scaling = ref_counts["transformer_matrices"] + ref_counts["lm_head"]
    scaling = counts["transformer_matrices"] + counts["lm_head"]
    lr_scale = math.sqrt(batch / (2**19))
    wd = cfg["weight_decay"] * lr_scale * ref_scaling / scaling
    optimizer = model.setup_optimizer(embedding_lr=cfg["embedding_lr"]*lr_scale,
        unembedding_lr=cfg["unembedding_lr"]*lr_scale, scalar_lr=cfg["scalar_lr"]*lr_scale,
        matrix_lr=cfg["matrix_lr"]*lr_scale, weight_decay=wd)
    train = ExactBOSBestFitDataLoader(tokenizer,B,T,"train",rank=rank,world_size=world,device=device)
    token_bytes = get_token_bytes(device=device)
    step = 0; training_seconds = 0.; evaluation_seconds = 0.; optimizer_seconds = 0.
    if args.resume_checkpoint:
        ck = Path(args.resume_checkpoint)
        meta = json.loads((ck / "complete.json").read_text())
        assert meta["config_sha256"] == config_hash and meta["world_size"] == world
        model.load_state_dict(torch.load(ck / "model.pt",map_location=device,weights_only=True), strict=True)
        state = torch.load(ck/f"rank{rank}.pt",map_location="cpu",weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        for param, values in optimizer.state.items():
            for key, value in values.items():
                if isinstance(value,torch.Tensor) and key != "step":
                    assert value.device == param.device, (key,value.device,param.device)
        restore_rank_training_state(state["training"],train,rank=rank,world_size=world,device=device)
        step = meta["step"]
        training_seconds = meta["training_seconds"]
        evaluation_seconds = meta["evaluation_seconds"]
        optimizer_seconds = meta["optimizer_seconds"]
    if cfg.get("compile", "block") == "block":
        # Compile shared blocks once; avoid a very large 96-visit unrolled graph.
        for block in model.transformer.h: block.forward = torch.compile(block.forward,dynamic=False)
    elif cfg.get("compile") != "none": raise ValueError("Unknown compile policy")
    groups = [{k:v for k,v in g.items() if k != "params"} | {"parameter_count":sum(p.numel() for p in g["params"])} for g in optimizer.param_groups]
    event("initialized",rank_count=world,model_config=asdict(mc),config_sha256=config_hash,
        init_backbone_sha256=init_sha,parameters=counts,flops_per_token=fpt,horizon=horizon,
        planned_tokens=horizon*batch,planned_model_flops=horizon*batch*fpt,gradient_accumulation=accum,
        weight_decay_scaled=wd,optimizer_groups=groups,torch_version=torch.__version__,dtype=str(COMPUTE_DTYPE),flash_attention_3=USE_FA3)
    def sync():
        if device.type == "cuda": torch.cuda.synchronize(device)
    def checkpoint(reason):
        nonlocal step
        ck = out/"checkpoints"/f"step_{step:06d}"
        ck.mkdir(parents=True,exist_ok=True)
        if (ck/"complete.json").exists(): return
        if world > 1: dist.barrier()
        if rank == 0:
            torch.save(model.state_dict(),ck/"model.pt.tmp")
            os.replace(ck/"model.pt.tmp",ck/"model.pt")
        state = {"optimizer":optimizer.state_dict(),"training":capture_rank_training_state(train,rank=rank,world_size=world,device=device)}
        torch.save(state,ck/f"rank{rank}.pt.tmp")
        os.replace(ck/f"rank{rank}.pt.tmp",ck/f"rank{rank}.pt")
        if world > 1: dist.barrier()
        if rank == 0:
            meta = dict(step=step,reason=reason,config_sha256=config_hash,world_size=world,
                training_seconds=training_seconds,evaluation_seconds=evaluation_seconds,optimizer_seconds=optimizer_seconds)
            atomic_json(ck/"complete.json",meta)
            atomic_json(out/"latest.json",dict(path=str(ck),**meta))
            # Only checkpoints created inside this newly owned run are eligible.
            complete = sorted(p for p in (out/"checkpoints").glob("step_*") if (p/"complete.json").exists())
            for old in complete[:-2]:
                n = int(old.name[5:])
                if n != 0 and n % 500 != 0: shutil.rmtree(old)
            event("checkpoint",step=step,path=str(ck),reason=reason)
        if world > 1: dist.barrier()
    def evaluate():
        nonlocal evaluation_seconds
        sync(); t0=time.monotonic()
        loader = ExactBOSBestFitDataLoader(tokenizer,B,T,"val",rank=rank,world_size=world,device=device)
        n = cfg["eval_tokens"]//(B*T*world)
        assert n > 0 and n*B*T*world == cfg["eval_tokens"]
        nats=torch.zeros((),dtype=torch.float64,device=device)
        num_bytes=torch.zeros((),dtype=torch.int64,device=device)
        panel_hash=hashlib.sha256()
        model.eval()
        with torch.no_grad():
            for _ in range(n):
                x,y=next(loader)
                panel_hash.update(y.cpu().numpy().tobytes())
                loss=model(x,y,loss_reduction="none").reshape(-1)
                lengths=token_bytes[y.reshape(-1)]
                nats+=(loss.double()*(lengths>0)).sum(); num_bytes+=lengths.sum()
        if world>1:
            dist.all_reduce(nats);dist.all_reduce(num_bytes)
        bpb=nats.item()/math.log(2)/num_bytes.item()
        assert math.isfinite(bpb)
        sync(); dt=time.monotonic()-t0; evaluation_seconds+=dt
        with (out/f"evaluation_rank{rank}.jsonl").open("a") as f:
            f.write(json.dumps(dict(step=step,token_sha256=panel_hash.hexdigest(),tokens_per_rank=n*B*T))+"\n")
        event("validation",step=step,bpb=bpb,nats=nats.item(),bytes=num_bytes.item(),seconds=dt,tokens=cfg["eval_tokens"],model_flops=step*batch*fpt)
        model.train()
    # Initial evaluation is deliberately full sized in production.
    if step == 0 and cfg.get("initial_eval",True): evaluate()
    last_val = -1
    while step < horizon:
        sync();t0=time.monotonic();loss_sum=torch.zeros((),device=device)
        aux_sum=torch.zeros((),device=device)
        selection_counts = gate_mass = None
        log_selection = step == 0 or (cfg["eval_every"] > 0 and (step + 1) % cfg["eval_every"] == 0) or step + 1 == horizon
        for micro in range(accum):
            x,y=next(train)
            if hasattr(model, "training_loss"):
                ce, aux, stats = model.training_loss(x, y, return_stats=True)
                if log_selection:
                    counts, mass = stats["selected_counts"], stats["gate_mass"]
                    selection_counts = counts if selection_counts is None else selection_counts + counts
                    gate_mass = mass if gate_mass is None else gate_mass + mass
            else:
                ce, aux = model(x, y), torch.zeros((), device=device)
            loss = ce + aux
            loss_sum += ce.detach()
            aux_sum += aux.detach()
            (loss/accum).backward()
        finite=torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
        finite=finite & torch.isfinite(loss_sum) & torch.isfinite(aux_sum)
        if world>1: dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not finite.item(): raise FloatingPointError(f"Nonfinite loss/gradient at step{step}")
        missing=[name for name,p in model.named_parameters() if p.grad is None]
        if missing: raise RuntimeError(f"Missing gradients:{missing}")
        scorer_grad = None
        if model.selector_weight is not None:
            scorer_grad = torch.stack([model.selector_weight.grad.float().norm(),model.selector_bias.grad.float().norm()])
            if world>1:dist.all_reduce(scorer_grad,op=dist.ReduceOp.AVG)
            scorer_grad = scorer_grad.tolist()
        warmdown=round(cfg["warmdown_ratio"]*horizon)
        if step<cfg["warmup_steps"]: lrm=(step+1)/cfg["warmup_steps"]
        elif step<=horizon-warmdown:lrm=1.
        else:
            progress=(horizon-step)/warmdown
            lrm=progress+(1-progress)*cfg["final_lr_frac"]
        if step<400: momentum=.85+(.97-.85)*step/400
        elif step>=horizon-warmdown: momentum=.97+(.90-.97)*(step-(horizon-warmdown))/warmdown
        else:momentum=.97
        for group in optimizer.param_groups:
            group["lr"]=group["initial_lr"]*lrm
            if group["kind"]=="muon":
                group["momentum"]=momentum
                group["weight_decay"]=wd*.5*(1+math.cos(math.pi*step/horizon))
        sync();ot=time.monotonic();optimizer.step();sync();odt=time.monotonic()-ot
        optimizer_seconds+=odt
        model.zero_grad(set_to_none=True)
        loss_sum/=accum
        aux_sum/=accum
        if world>1:
            dist.all_reduce(loss_sum,op=dist.ReduceOp.AVG)
            dist.all_reduce(aux_sum,op=dist.ReduceOp.AVG)
        sync();dt=time.monotonic()-t0;training_seconds+=dt;step+=1
        if selection_counts is not None:
            if world > 1:
                dist.all_reduce(selection_counts)
                dist.all_reduce(gate_mass)
            event("selection", step=step, token_positions=batch,
                  selected_counts=selection_counts.tolist(), gate_mass=gate_mass.tolist())
        event("train",step=step,horizon=horizon,loss=loss_sum.item(),weighted_aux_loss=aux_sum.item(),total_loss=(loss_sum+aux_sum).item(),seconds=dt,optimizer_seconds=odt,
            lr_multiplier=lrm,muon_momentum=momentum,tokens=step*batch,model_flops=step*batch*fpt,
            selector_local_grad_norm_mean=scorer_grad,
            peak_memory_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=="cuda" else 0)
        if rank==0:atomic_json(out/"status.json",dict(state="running",step=step,horizon=horizon,updated=time.time()))
        if cfg["eval_every"]>0 and (step%cfg["eval_every"]==0 or step==horizon):evaluate();last_val=step
        stop_requested=torch.tensor(int((out/"STOP").exists()),device=device)
        if world>1:dist.all_reduce(stop_requested,op=dist.ReduceOp.MAX)
        should_stop=bool(stop_requested.item()) or (args.stop_after is not None and step>=args.stop_after)
        if step%cfg["save_every"]==0 or step==horizon or should_stop:checkpoint("complete" if step==horizon else "pause" if should_stop else "periodic")
        if should_stop:break
        if step==1:gc.collect();gc.freeze();gc.disable()
    state="complete" if step==horizon else "paused"
    event(state,step=step,horizon=horizon,training_seconds=training_seconds,evaluation_seconds=evaluation_seconds,optimizer_seconds=optimizer_seconds)
    if rank==0:atomic_json(out/"status.json",dict(state=state,step=step,horizon=horizon,updated=time.time()))
    compute_cleanup()


if __name__ == "__main__":
    try: main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
