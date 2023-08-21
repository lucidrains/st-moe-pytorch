import math
from functools import partial
from inspect import isfunction
from collections import namedtuple
from typing import Tuple, Union, Optional

import torch
from torch.nn import Module, ModuleList
from torch import nn, einsum
import torch.nn.functional as F

from beartype import beartype

from einops import rearrange, repeat, reduce, pack, unpack

from colt5_attention import topk as differentiable_topk

# constants

MIN_EXPERT_CAPACITY = 4

MixtureOfExpertsReturn = namedtuple('MixtureOfExpertsReturn', [
    'outputs',
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

def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# tensor related helper functions

def top1(t):
    topk_return = t.topk(k = 1)
    values = rearrange(topk_return.values, '... 1 -> ...')
    indices = rearrange(topk_return.indices, '... 1 -> ...')
    return values, indices

def cumsum_exclusive(t, dim = -2):
    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    return F.pad(t, (*pre_padding, 1, -1)).cumsum(dim = dim)

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

class Experts(Module):
    def __init__(
        self,
        dim,
        num_experts = 16,
        hidden_mult = 4
    ):
        super().__init__()
        self.experts = ModuleList([Expert(dim = dim, hidden_mult = hidden_mult) for _ in range(num_experts)])

    def forward(self, x):
        outputs = []

        for tokens, expert in zip(x, self.experts):
            outputs.append(expert(tokens))

        return torch.stack(outputs)

# the below code is almost all transcribed from the official tensorflow version, from which the papers are written
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/moe.py

# gating network

class Top2Gating(Module):
    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        outer_expert_dims: Tuple[int, ...] = tuple(),
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        detached_dispatch_tensor = True
    ):
        super().__init__()
        self.eps = eps
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))

        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval        

        self.detached_dispatch_tensor = detached_dispatch_tensor
        self.register_buffer('zero', torch.zeros((1,)), persistent = False)

    def forward(self, x):
        *_, b, group_size, dim, dtype, num_gates = *x.shape, x.dtype, self.num_gates

        # threshold, capacity depending on training or eval

        suffix = 'train' if self.training else 'eval'

        threshold = getattr(self, f'second_threshold_{suffix}')
        capacity_factor = getattr(self, f'capacity_factor_{suffix}')

        # logits and gates

        gate_logits = einsum('... b n d, ... d e -> ... b n e', x, self.w_gating)
        raw_gates = gate_logits.softmax(dim = -1)

        # FIND TOP 2 EXPERTS PER POSITON
        # Find the top expert for each position. shape=[batch, group]

        density_1_proxy = raw_gates

        topk_raw_gates_values, topk_raw_gates_indices = raw_gates.topk(k = 2, dim = -1)

        gate_1, gate_2 = topk_raw_gates_values.unbind(dim = -1)
        index_1, index_2 = topk_raw_gates_indices.unbind(dim = -1)

        mask_1 = F.one_hot(index_1, num_gates).float()
        mask_2 = F.one_hot(index_2, num_gates).float()

        # normalize top2 gate scores

        denom = gate_1 + gate_2 + self.eps
        gate_1 = gate_1 / denom
        gate_2 = gate_2 / denom

        # BALANCING LOSSES
        # shape = [batch, experts]
        # We want to equalize the fraction of the batch assigned to each expert

        density_1 = reduce(mask_1, '... n e -> ... e', 'mean')
        # Something continuous that is correlated with what we want to equalize.
        density_1_proxy = reduce(density_1_proxy, '... n e -> ... e', 'mean')

        if self.training:
            balance_loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)
        else:
            balance_loss = self.zero

        # Best performing policy was to route to the second expert, with probability of min(1., score / threshold), where score = gate2 / (gate1 + gate2)
        # optimal threshold was ~ 0.2

        probs = torch.zeros_like(gate_2).uniform_(0., 1.)
        route_second_expert = probs < (gate_2 / max(threshold, self.eps))
        mask_2 *= rearrange(route_second_expert.float(), '... -> ... 1')

        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes

        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        # COMPUTE ASSIGNMENT TO EXPERTS
        # [batch, group, experts]
        # This is the position within the expert's mini-batch for this sequence

        position_in_expert_1 = cumsum_exclusive(mask_1) * mask_1
        # Remove the elements that don't fit. [batch, group, experts]
        mask_1 *= (position_in_expert_1 < expert_capacity_f).float()
        # [batch, experts]
        # How many examples in this sequence go to this expert
        mask_1_count = reduce(mask_1, '... n e -> ... 1 e', 'mean')
        # [batch, group] - mostly ones, but zeros where something didn't fit
        mask_1_flat = reduce(mask_1, '... n e -> ... n', 'sum')
        # [batch, group]
        position_in_expert_1 = reduce(position_in_expert_1, '... n e -> ... n', 'sum')
        # Weight assigned to first expert.  [batch, group]
        gate_1 *= mask_1_flat

        position_in_expert_2 = cumsum_exclusive(mask_2) + mask_1_count
        position_in_expert_2 *= mask_2
        mask_2 *= (position_in_expert_2 < expert_capacity_f).float()
        mask_2_flat = reduce(mask_2, '... n e -> ... n', 'sum')

        position_in_expert_2 = reduce(position_in_expert_2, '... n e -> ... n', 'sum')
        gate_2 *= mask_2_flat
        
        # [batch, group, experts, expert_capacity]

        N = None

        combine_tensor = (
            gate_1[..., N, N]
            * mask_1_flat[..., N, N]
            * F.one_hot(index_1, num_gates)[..., N]
            * safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., N, :] +
            gate_2[..., N, N]
            * mask_2_flat[..., N, N]
            * F.one_hot(index_2, num_gates)[..., N]
            * safe_one_hot(position_in_expert_2.long(), expert_capacity)[..., N, :]
        )

        # dispatch tensor

        dispatch_tensor = combine_tensor.bool().type(dtype)

        if not self.detached_dispatch_tensor:
            dispatch_tensor = dispatch_tensor + combine_tensor - combine_tensor.detach()

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
    def __init__(self,
        dim,
        num_experts = 16,
        expert_hidden_mult = 4,
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        router_z_loss_coef = 1e-3,
        experts: Optional[Module] = None,
        detached_dispatch_tensor = True
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts

        gating_kwargs = dict(
            second_threshold_train = second_threshold_train,
            second_threshold_eval = second_threshold_eval,
            capacity_factor_train = capacity_factor_train,
            capacity_factor_eval = capacity_factor_eval
        )

        self.gate = Top2Gating(
            dim,
            num_gates = num_experts,
            detached_dispatch_tensor = detached_dispatch_tensor,
            **gating_kwargs
        )

        self.experts = default(experts, lambda: Experts(dim, num_experts = num_experts, hidden_mult = expert_hidden_mult))

        self.loss_coef = loss_coef
        self.router_z_loss_coef = router_z_loss_coef

    def forward(self, inputs, **kwargs):
        dispatch_tensor, combine_tensor, loss, router_z_loss = self.gate(inputs)

        # dispatch

        expert_inputs = einsum('b n d, b n e c -> e b c d', inputs, dispatch_tensor)

        # feed the expert inputs through the experts.

        expert_inputs, ps = pack_one(expert_inputs, 'e * d')
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = unpack_one(expert_outputs, ps, 'e * d')

        # combine

        output = einsum('e b c d, b n e c -> b n d', expert_outputs, combine_tensor)

        # losses

        balance_loss = loss * self.loss_coef
        router_z_loss = router_z_loss * self.router_z_loss_coef

        return MixtureOfExpertsReturn(output, balance_loss, router_z_loss)

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

    def forward(self, x):

        # feedforward before

        if exists(self.ff_before):
            x = self.ff_before(x) + x

        # mixture of experts layer

        residual = x

        moe_out, balance_loss, router_z_loss = self.moe(self.moe_prenorm(x))

        x = moe_out + residual

        # feedforward after

        if exists(self.ff_after):
            x = self.ff_after(x) + x

        return MixtureOfExpertsReturn(x, balance_loss, router_z_loss)
