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
):
    setup(rank, world_size)

    net = Experts([Expert(dim) for _ in range(num_experts)])

    if batch_size_var_len:
        batch_size = batch_size + rank

    seq = torch.randn(batch_size, num_experts, tokens_per_expert, dim)

    # distributed

    model = DDP(net)
    out = model(seq)
    out.mean().backward()

    ddp_all_out, _ = all_gather_variable_dim(out)

    # on single device

    all_inputs, _ = all_gather_variable_dim(seq)
    copied_net = deepcopy(net)

    single_out = copied_net(
        all_inputs,
        is_distributed = False
    )

    single_out.mean().backward()

    if rank == 0:
        # validate output is the same
        # if done on 1 vs multiple machines

        assert torch.allclose(single_out, ddp_all_out), 'output is not the same'

        # validate backwards and grad

        get_first_expert_grad = lambda t: t.experts[0].net[0].weight.grad

        assert torch.allclose(
            get_first_expert_grad(net),
            get_first_expert_grad(copied_net),
            atol = 1e-2
        ), 'grad is not the same'

        print('✅')

    cleanup()

if __name__ == '__main__':
    world_size = 5
    num_experts = 8
    batch_size = 2
    batch_size_var_len = False

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
            dim
        ),
        nprocs = world_size,
        join = True
    )
