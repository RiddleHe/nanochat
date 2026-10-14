"""Run with PYTHONPATH=. torchrun --standalone --nproc_per_node=2; no dataset needed."""
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup
from nanochat.model.gpt_loop import GPTLoop, GPTLoopConfig
from nanochat.training_state import check_finite_gradients


def main():
    _, rank, _, world, device = compute_init("cuda")
    assert world >= 2
    with torch.device("meta"):
        model = GPTLoop(GPTLoopConfig(n_layer=4, num_loops=4, n_embd=64,
                                     n_head=1, n_kv_head=1, vocab_size=128, sequence_len=16))
    model.to_empty(device=device)
    model.init_weights()
    optimizer = model.setup_optimizer()
    # One physical block: some Muon groups have fewer matrices than ranks,
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
        print("PASS: shared updates agree across ranks; rank-local Inf stops all ranks")
    compute_cleanup()


if __name__ == "__main__":
    main()
