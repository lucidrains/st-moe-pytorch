from functools import partial
from collections import namedtuple
from typing import Optional, Tuple, Union

import torch
from torch.nn import Module, ModuleList
from torch import nn, einsum
import torch.nn.functional as F

from beartype import beartype

from einops import rearrange, repeat, reduce, pack, unpack

from colt5_attention import topk as maybe_differentiable_topk

import torch.distributed as dist

from st_moe_pytorch.distributed import (
    AllGather,
    split_by_rank,
    gather_sizes,
    has_only_one_value
)

# constants

MIN_EXPERT_CAPACITY = 4

MixtureOfExpertsReturn = namedtuple('MixtureOfExpertsReturn', [
    'outputs',
    'total_aux_loss',
    'balance_loss',
    'router_z_loss'
])

# helper functions

def exists(val):
    return val is not None

def default(val, default):
    if exists(val):
        return val

    return default() if callable(default) else default

def divisible_by(num, den):
    return (num % den) == 0

def chunk_num(num, chunks):
    num_per_chunk, remainder = divmod(num, chunks)

    out = []
    for i in range(chunks):
        n = num_per_chunk
        out.append(n + int(i < remainder))

    return out

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def cast_tuple(el, len = 1):
    return el if isinstance(el, tuple) else ((el,) * len)

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# tensor related helper functions

def cumsum_exclusive(t, dim = -3):
    assert dim < 0
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    return F.pad(t, (*pre_padding, 1, -1)).cumsum(dim = dim)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# pytorch one hot throws an error if there are out of bound indices.
# tensorflow, in contrast, does not throw an error

def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    one_hot_classes = max(max_index + 1, max_length)
    return F.one_hot(indexes, one_hot_classes)[..., :max_length]

# rms normalization

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

# expert class
# best performing was ff geglu with multiplicative bias (just after gating)

class GEGLU(Module):
    def __init__(
        self,
        dim,
        mult_bias = True
    ):
        super().__init__()
        self.mult_bias = nn.Parameter(torch.ones(dim)) if mult_bias else 1.

    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x * self.mult_bias

class Expert(Module):
    def __init__(
        self,
        dim,
        hidden_mult = 4,
        mult_bias = True,
        prenorm = False
    ):
        super().__init__()
        dim_hidden = int(dim * hidden_mult * 2 / 3)

        self.net = Sequential(
            RMSNorm(dim) if prenorm else None,
            nn.Linear(dim, dim_hidden * 2),
            GEGLU(dim_hidden, mult_bias = mult_bias),
            nn.Linear(dim_hidden, dim)
        )

        self.apply(self.init_)

    def init_(self, module):
        if isinstance(module, nn.Linear):
            dim = module.weight.shape[0]
            std = dim ** -0.5

            module.weight.data.uniform_(-std, std)
            module.bias.data.uniform_(-std, std)

    def forward(self, x):
        return self.net(x)

class Experts(nn.Module):
    def __init__(
        self,
        experts,
        is_distributed = None,
        allow_var_seq_len = False # whether to handle variable sequence length
    ):
        super().__init__()
        self.num_experts = len(experts)
        self.experts = nn.ModuleList(experts)

        # distributed related settings

        self.is_distributed = is_distributed
        if not exists(self.is_distributed):
            self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

        self.all_gather = AllGather()

        self.allow_var_seq_len = allow_var_seq_len

        # device tracker, since need to manually move experts not in use to CPU in distributed

        self.register_buffer('dummy', torch.ones(1), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    def all_experts_to_cpu_besides(self, selection):
        if isinstance(selection, int):
            experts = [self.experts[selection]]
        if isinstance(selection, slice):
            experts = self.experts[selection]
        else:
            experts = selection

        experts_set = set(experts)

        for expert in self.experts:
            device = self.device if expert in experts_set else 'cpu'
            expert.to(device)

    def forward(
        self,
        x,
        is_distributed = None
    ):
        """
        einops notation:
        b - batch
        r - rank (device / machines)
        e - experts
        n - sequence (number of tokens per expert)
        d - feature dimension
        """

        # declare some variables

        is_distributed = default(is_distributed, self.is_distributed)
        shape, num_experts = x.shape, self.num_experts
        seq_len = shape[-2]

        # for now naively all gather across batch dimension if distributed, optimize later

        world_size = 1
        rank = 0

        if is_distributed:
            seq_sizes = gather_sizes(x, dim = -2)
            var_seq_len = not has_only_one_value(seq_sizes)

            assert self.allow_var_seq_len or not var_seq_len, 'number of tokens per expert must be the same - if you want the framework to handle it, set `allow_var_seq_len = True` on `Experts`'

            # if variable sequence length, pad

            if var_seq_len:
                max_seq_size = seq_sizes.amax().item()
                x = pad_dim_to(x, max_seq_size, dim = -2)

            # gather and concat across batches, accounting for variable batch sizes

            x, batch_sizes = self.all_gather(x)
            total_batch_size = batch_sizes.sum().item()

            world_size = dist.get_world_size()
            rank = dist.get_rank()

        # the experts in use on the rank

        num_experts_per_rank = num_experts
        expert_slice = slice(0, num_experts)

        if is_distributed:
            if world_size <= num_experts:
                num_experts_across_ranks = chunk_num(num_experts, world_size)
                start_indices = cumsum_exclusive(torch.tensor(num_experts_across_ranks), dim = -1)

                num_experts_per_rank = num_experts_across_ranks[rank]
                num_experts_batches_across_ranks = tuple(i * total_batch_size for i in num_experts_across_ranks)

                expert_start_index = start_indices[rank].item()
            else:
                num_batch_chunks = world_size // num_experts
                total_ranks_in_use = num_batch_chunks * num_experts

                expert_start_index = rank // num_batch_chunks

                batch_splits = chunk_num(total_batch_size, num_batch_chunks)
                num_experts_batches_across_ranks = batch_splits * num_experts

                # for now, remaining machines just process nothing

                remain_ranks = world_size % num_experts
                num_experts_batches_across_ranks += (0,) * remain_ranks

                num_experts_per_rank = int(rank < total_ranks_in_use)

            assert len(num_experts_batches_across_ranks) == world_size

            expert_slice = slice(expert_start_index, expert_start_index + num_experts_per_rank)

        # if distributed, each machine only handles subset of experts and batch

        x = rearrange(x, 'b e n d -> e b n d')

        if is_distributed:
            x, expert_batch_packed_shape = pack_one(x, '* n d')

            x = x.split(num_experts_batches_across_ranks, dim = 0)
            x, experts_per_rank_sizes = split_by_rank(x)

            if num_experts_per_rank > 0:
                x = rearrange(x, '(e b) n d -> e b n d', e = num_experts_per_rank)
            else:
                x = x.reshape(num_experts, *x.shape)

        # get the experts in use

        self.all_experts_to_cpu_besides(expert_slice)

        experts = self.experts[expert_slice]

        # route tokens to appropriate experts

        outs = []

        for expert, expert_input in zip(experts, x):
            out = expert(expert_input)
            outs.append(out)

        if len(outs) > 0:
            outs = torch.stack(outs)
        else:
            outs = torch.empty_like(x, requires_grad = self.training)

        # all gather across merged expert batches dimensions
        # then split the batch dimension back

        if is_distributed:
            outs = rearrange(outs, 'e b n d -> (e b) n d')
            outs, _ = self.all_gather(outs, sizes = experts_per_rank_sizes)
            outs = unpack_one(outs, expert_batch_packed_shape, '* n d')

        outs = rearrange(outs, 'e b n d -> b e n d')

        if is_distributed:
            outs = outs.split(batch_sizes.tolist())
            outs, _ = split_by_rank(outs)

            # account for padded sequence length
            outs = outs[..., :seq_len, :]

        assert outs.shape == shape
        return outs

# the below code is almost all transcribed from the official tensorflow version, from which the papers are written
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/moe.py

# gating network

class TopNGating(Module):

    @beartype
    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        top_n = 2,
        threshold_train: Union[float, Tuple[float, ...]] = 0.2,
        threshold_eval: Union[float, Tuple[float, ...]] = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        straight_through_dispatch_tensor = True,
        differentiable_topk = False,
        differentiable_topk_fused = True
    ):
        super().__init__()
        self.eps = eps
        self.num_gates = num_gates
        self.to_gates = nn.Linear(dim, num_gates, bias = False)

        self.differentiable_topk = differentiable_topk

        self.topk = partial(
            maybe_differentiable_topk,
            non_differentiable = not differentiable_topk,
            fused = differentiable_topk_fused # use triton fused coordinate descent if possible by default
        )

        assert top_n >= 2, 'must be 2 or more experts'
        self.top_n = top_n
        top_n_minus_1 = top_n - 1

        threshold_train = cast_tuple(threshold_train, top_n_minus_1)
        threshold_eval = cast_tuple(threshold_eval, top_n_minus_1)

        assert len(threshold_train) == len(threshold_eval) == top_n_minus_1

        self.register_buffer('threshold_train', torch.tensor([eps, *threshold_train]))
        self.register_buffer('threshold_eval', torch.tensor([eps, *threshold_eval]))

        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval        

        self.straight_through_dispatch_tensor = straight_through_dispatch_tensor
        self.register_buffer('zero', torch.zeros((1,)), persistent = False)

    def forward(
        self,
        x,
        noise_gates = False,
        noise_mult = 1.
    ):
        """
        einstein notation:

        b - batch
        n - sequence
        e - experts
        k - top-n experts
        """

        *_, b, group_size, dim, dtype, top_n, num_gates, eps = *x.shape, x.dtype, self.top_n, self.num_gates, self.eps

        # threshold, capacity depending on training or eval

        suffix = 'train' if self.training else 'eval'

        threshold = getattr(self, f'threshold_{suffix}')
        capacity_factor = getattr(self, f'capacity_factor_{suffix}')

        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes

        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        # gate logits and gates

        gate_logits = self.to_gates(x)

        maybe_noised_gate_logits = gate_logits

        if noise_gates:
            noise = gumbel_noise(maybe_noised_gate_logits)
            maybe_noised_gate_logits = maybe_noised_gate_logits + noise * noise_mult

        raw_gates = maybe_noised_gate_logits.softmax(dim = -1)

        # find top N experts per position

        topk_return = self.topk(raw_gates, k = top_n)

        gate_indices = topk_return.indices

        if self.differentiable_topk:
            # allow for differentiable topk using coordinate descent
            # used successfully for routing from CoLT5 paper https://github.com/lucidrains/CoLT5-attention

            gates = topk_return.coor_descent_values
        else:
            gates = topk_return.values

        # move the top-n dimension to be first

        gates = rearrange(gates, '... k -> k ...')
        gate_indices = rearrange(gate_indices, '... k -> k ...')

        # masks

        one_hot_gate_indices = F.one_hot(gate_indices, num_gates)
        mask = one_hot_gate_indices.float()

        mask_1 = mask[0] # needed for balancing loss

        # normalize top-n gate scores

        denom = reduce(gates, 'k ... -> 1 ...', 'sum').clamp(min = eps)
        gates = gates / denom

        # best performing policy was to route to the second expert, with probability of min(1., score / threshold), where score = gate2 / (gate1 + gate2)
        # optimal threshold was ~ 0.2
        # generalized to more than 2 experts

        probs = torch.zeros_like(gates).uniform_(0., 1.)

        threshold = rearrange(threshold, 'k -> k 1 1')
        should_route = probs < (gates / threshold.clamp(min = eps))

        # tokens should always be routed to first expert
        # threshold for first expert already set to very small number, but just in case

        should_route[0, ...] = True

        mask *= rearrange(should_route.float(), '... -> ... 1')

        mask_cumsum = cumsum_exclusive(mask, dim = -2) # along sequence dimension

        # compute assignment to experts - (batch, seq, experts)

        # This is the position within the expert's mini-batch for this sequence

        positions = []
        prev_expert_count = 0.

        for n in range(self.top_n):
            position_in_expert = (mask_cumsum[n] + prev_expert_count) * mask[n]

            # Remove the elements that don't fit. (batch, sequence, experts)
            mask[n] *= (position_in_expert < expert_capacity_f).float()

            # How many examples in this sequence go to this expert - needed for the next iteration as offset
            prev_expert_count = reduce(mask[n], '... n e -> ... 1 e', 'sum')

            # (batch, sequence)
            position_in_expert = reduce(position_in_expert, '... n e -> ... n', 'sum')
            positions.append(position_in_expert)

        positions = torch.stack(positions)

        # (k, batch, sequence) - mostly ones, but zeros where something didn't fit
        mask_flat = reduce(mask, '... n e -> ... n', 'sum')

        # (k, batch, sequence) - weighted assignment
        # following https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py#L1903
        gates = gates * mask_flat

        # (batch, sequence, experts, expert_capacity)

        N = None

        gates = gates[..., N, N]
        mask_flat = mask_flat[..., N, N]
        one_hot_gate_indices = one_hot_gate_indices[..., N]
        safe_one_hot_gates = safe_one_hot(positions.long(), expert_capacity)[..., N, :]

        combine_tensor = reduce(
            gates
            * mask_flat
            * one_hot_gate_indices
            * safe_one_hot_gates
        , 'k ... -> ...', 'sum')

        # dispatch tensor

        dispatch_tensor = combine_tensor.bool().type(dtype)

        if self.straight_through_dispatch_tensor:
            dispatch_tensor = dispatch_tensor + combine_tensor - combine_tensor.detach()

        # balance losses - (batch, experts)
        # We want to equalize the fraction of the batch assigned to each expert

        if self.training:
            density_1 = reduce(mask_1, '... n e -> ... e', 'mean')
            density_1_proxy = reduce(raw_gates, '... n e -> ... e', 'mean') # Something continuous that is correlated with what we want to equalize.

            balance_loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)
        else:
            balance_loss = self.zero

        # calculate the router z-loss proposed in paper

        if self.training:
            router_z_loss = torch.logsumexp(gate_logits, dim = -1)
            router_z_loss = torch.square(router_z_loss)            
            router_z_loss = router_z_loss.mean()
        else:
            router_z_loss = self.zero

        return dispatch_tensor, combine_tensor, balance_loss, router_z_loss

# plain mixture of experts

class MoE(Module):

    @beartype
    def __init__(self,
        dim,
        num_experts = 16,
        expert_hidden_mult = 4,
        threshold_train = 0.2,
        threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        gating_top_n = 2,
        balance_loss_coef = 1e-2,
        router_z_loss_coef = 1e-3,
        experts: Optional[Module] = None,
        straight_through_dispatch_tensor = True,
        differentiable_topk = False,
        differentiable_topk_fused = True,
        is_distributed = None,
        allow_var_seq_len = False
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts

        self.gate = TopNGating(
            dim,
            top_n = gating_top_n,
            num_gates = num_experts,
            straight_through_dispatch_tensor = straight_through_dispatch_tensor,
            differentiable_topk = differentiable_topk,
            threshold_train = threshold_train,
            threshold_eval = threshold_eval,
            capacity_factor_train = capacity_factor_train,
            capacity_factor_eval = capacity_factor_eval
        )

        experts = default(experts, lambda: [Expert(dim = dim, hidden_mult = expert_hidden_mult) for _ in range(num_experts)])

        self.experts = Experts(
            experts,
            is_distributed = is_distributed,
            allow_var_seq_len = allow_var_seq_len
        )

        self.balance_loss_coef = balance_loss_coef
        self.router_z_loss_coef = router_z_loss_coef

    def forward(
        self,
        x,
        noise_gates = False,
        noise_mult = 1.
    ):
        dispatch_tensor, combine_tensor, balance_loss, router_z_loss = self.gate(x, noise_gates = noise_gates, noise_mult = noise_mult)

        # dispatch

        expert_inputs = einsum('b n d, b n e c -> b e c d', x, dispatch_tensor)

        # feed the expert inputs through the experts.

        expert_outputs = self.experts(expert_inputs)

        # combine

        output = einsum('b e c d, b n e c -> b n d', expert_outputs, combine_tensor)

        # losses

        weighted_balance_loss = balance_loss * self.balance_loss_coef
        weighted_router_z_loss = router_z_loss * self.router_z_loss_coef

        # combine the losses

        total_aux_loss = weighted_balance_loss + weighted_router_z_loss

        return MixtureOfExpertsReturn(output, total_aux_loss, balance_loss, router_z_loss)

# sparse moe block
# in particular, they found that adding a feedforward before or after greatly stabilized the training and improved results

class SparseMoEBlock(Module):

    @beartype
    def __init__(
        self,
        moe: MoE,
        *,
        add_ff_before = False,
        add_ff_after = True
    ):
        super().__init__()
        dim = moe.dim

        self.moe = moe
        self.moe_prenorm = RMSNorm(dim)

        self.ff_before = Expert(dim, prenorm = True) if add_ff_before else None
        self.ff_after = Expert(dim, prenorm = True) if add_ff_after else None

    def forward(
        self,
        x,
        noise_gates = False,
        noise_mult = 1.
    ):

        # feedforward before

        if exists(self.ff_before):
            x = self.ff_before(x) + x

        # mixture of experts layer

        residual = x

        moe_out, total_aux_loss, balance_loss, router_z_loss = self.moe(self.moe_prenorm(x), noise_gates = noise_gates, noise_mult = noise_mult)

        x = moe_out + residual

        # feedforward after

        if exists(self.ff_after):
            x = self.ff_after(x) + x

        return MixtureOfExpertsReturn(x, total_aux_loss, balance_loss, router_z_loss)
