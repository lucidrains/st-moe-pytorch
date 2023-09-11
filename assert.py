import os
from copy import deepcopy

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from st_moe_pytorch.st_moe_pytorch import Experts, Expert
from st_moe_pytorch.distributed import all_gather_variable_dim

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank = rank, world_size = world_size)

def cleanup():
    dist.destroy_process_group()

def start(
    rank,
    world_size,
    batch_size,
    batch_size_var_len,
    num_experts,
    tokens_per_expert,
    dim,
    use_cuda
):
    setup(rank, world_size)

    net = Experts([Expert(dim) for _ in range(num_experts)])

    if batch_size_var_len:
        batch_size = batch_size + rank

    seq = torch.randn(batch_size, num_experts, tokens_per_expert, dim)

    # locally

    local_net = deepcopy(net)

    local_inputs, _ = all_gather_variable_dim(seq)

    local_out = local_net(
        local_inputs,
        is_distributed = False
    )

    local_out.mean().backward()

    # distributed

    model = DDP(net)
    ddp_inputs = seq

    if use_cuda:
        model.cuda(rank)
        ddp_inputs = seq.cuda(rank)

    out = model(ddp_inputs)
    out.mean().backward()

    ddp_all_out, _ = all_gather_variable_dim(out)

    if rank == 0:
        # validate output is the same for local vs distributed

        model.cpu()
        ddp_all_out.cpu()

        assert torch.allclose(local_out, ddp_all_out.cpu(), atol = 1e-3), 'output is not the same'

        # validate gradients of first expert is the same for local vs distributed

        get_first_expert_grad = lambda t: t.experts[0].net[0].weight.grad

        assert torch.allclose(
            get_first_expert_grad(net).cpu(),
            get_first_expert_grad(local_net),
            atol = 1e-2
        ), 'grad is not the same'

        print('✅ outputs and gradients are same between local and ddp')

    cleanup()

if __name__ == '__main__':
    world_size = 8
    num_experts = 3
    batch_size = 2
    batch_size_var_len = True
    use_cuda = False

    assert not use_cuda or torch.cuda.device_count() <= world_size

    seq_len = 32
    dim = 8

    mp.spawn(
        start,
        args = (
            world_size,
            batch_size,
            batch_size_var_len,
            num_experts,
            seq_len,
            dim,
            use_cuda
        ),
        nprocs = world_size,
        join = True
    )
