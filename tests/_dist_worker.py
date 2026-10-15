"""Worker for test_train_conditional's real two-process all-reduce test (gloo, CPU)."""

import os
import sys

import torch
import torch.distributed as dist


def run(rank: int, world: int, port: str, out_dir: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from wrf_pinn.models.conditional import ConditionalModel, ConditionalModelConfig
    from wrf_pinn.training.train_conditional import _average_across_processes

    torch.manual_seed(0)  # same weights on every rank
    params = list(ConditionalModel(5, ConditionalModelConfig(hidden_layers=2, hidden_width=8)).parameters())
    gen = torch.Generator().manual_seed(100 + rank)  # different gradients per rank
    for p in params:
        p.grad = torch.randn(p.shape, generator=gen)
    losses = torch.arange(3, dtype=torch.float32) + rank
    expected = [p.grad.clone() for p in params] + [losses.clone()]
    for t in expected:  # the per-tensor reduce the packed one replaces
        dist.all_reduce(t)
        t.div_(world)
    got = _average_across_processes(params, losses, world)
    ok = all(torch.allclose(p.grad, e) for p, e in zip(params, expected[:-1])) \
        and torch.allclose(got, expected[-1])
    open(os.path.join(out_dir, f"rank{rank}.txt"), "w").write("ok" if ok else "mismatch")
    dist.destroy_process_group()


if __name__ == "__main__":
    run(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4])
