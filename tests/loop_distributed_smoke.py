"""Run with PYTHONPATH=. torchrun --standalone --nproc_per_node=2; no dataset needed."""
import argparse
import tempfile
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup
from nanochat.model.gpt_loop import GPTLoop, GPTLoopConfig
from nanochat.training_state import check_finite_gradients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-depth", type=int, default=1)
    args = parser.parse_args()
    _, rank, _, world, device = compute_init("cuda")
    assert world >= 2
    with torch.device("meta"):
        model = GPTLoop(GPTLoopConfig.from_block_depth(args.physical_depth, num_loops=4, n_embd=64,
                                     n_head=1, n_kv_head=1, vocab_size=128, sequence_len=16))
    model.to_empty(device=device)
    model.init_weights()
    optimizer = model.setup_optimizer()
    # With one physical block some Muon groups have fewer matrices than ranks,
    # exercising distributed padding as well as shared-gradient synchronization.
    rng = torch.Generator(device=device).manual_seed(100 + rank)
    for _ in range(3):
        x = torch.randint(128, (2, 16), device=device, generator=rng)
        y = torch.randint(128, (2, 16), device=device, generator=rng)
        loss = model(x, y)
        loss.backward()
        check_finite_gradients(model, loss)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for p in model.parameters():
            reference = p.detach().clone()
            dist.broadcast(reference, src=0)
            assert torch.equal(p, reference), "Parameters diverged between ranks"
    # Real serialization round-trip of each rank's optimizer shard must preserve
    # the next update exactly, including 24 physical blocks and four-rank runs.
    with tempfile.TemporaryDirectory() as directory:
        path = directory + "/state.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
        model(x, y).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
        saved = torch.load(path, weights_only=True, map_location=device)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        model(x, y).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for key, value in model.state_dict().items():
            assert torch.equal(value, expected[key]), f"Resume update differs: {key}"
    # A failure on just one rank must stop *all* ranks before optimizer collectives.
    model(x, y).backward()
    if rank == world - 1:
        next(model.parameters()).grad.flatten()[0] = float("inf")
    try:
        check_finite_gradients(model, loss)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("Distributed non-finite check did not stop every rank")
    if rank == 0:
        print("PASS: shared updates agree; checkpoint resume is exact; rank-local Inf stops all ranks")
    compute_cleanup()


if __name__ == "__main__":
    main()
