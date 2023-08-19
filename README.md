## ST-MoE - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2202.08906">ST-MoE</a>, the latest incarnation of mixture of experts after years of research at Brain, in Pytorch. Will be largely a transcription of the <a href="https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py">official Mesh Tensorflow implementation</a>. If you have any papers you think should be added, while I have my attention on mixture of experts, please open an issue.

## Install

```bash
$ pip install st-moe-pytorch
```

## Appreciation

- <a href="https://stability.ai/">StabilityAI</a> for the generous sponsorship, as well as my other sponsors, for affording me the independence to open source artificial intelligence.

## Usage

```python
import torch
from torch import nn
from st_moe_pytorch.st_moe_pytorch import MoE

moe = MoE(
    dim = 512,
    num_experts = 16,               # increase the experts (# parameters) of your model without increasing computation
    second_policy_train = 'random', # in top_2 gating, policy for whether to use a second-place expert
    second_policy_eval = 'random',  # all (always) | none (never) | threshold (if gate value > the given threshold) | random (if gate value > threshold * random_uniform(0, 1))
    second_threshold_train = 0.2,
    second_threshold_eval = 0.2,
    capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
    capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
    loss_coef = 1e-2                # multiplier on the auxiliary expert balancing auxiliary loss
)

inputs = torch.randn(4, 1024, 512)
out, aux_loss, router_z_loss = moe(inputs) # (4, 1024, 512), (1,), (1,)
```

Hierarchical Mixture of Experts

```python
import torch
from st_moe_pytorch import HeirarchicalMoE

moe = HeirarchicalMoE(
    dim = 512,
    num_experts = (4, 4),       # 4 gates on the first layer, then 4 experts on the second, equaling 16 experts
)

inputs = torch.randn(4, 1024, 512)
out, aux_loss, router_z_loss = moe(inputs) # (4, 1024, 512), (1,), (1,)
```

## Todo

- [x] add the router z-loss proposed in paper
- [x] add the geglu expert with multiplicative gating

- [ ] double check equation for router z-loss for experts inner in hierarchical moe. also look at the top-n generalization
- [ ] add an entire sparse moe block, complete with rmsnorm + residual as well as the ability to specify a feedforward before or after for stability
- [ ] offer an option to use differentiable topk (using coordinate descent algorithm, applied successfully in CoLT5)
- [ ] use coordinate descent for top2 routing

## Citations

```bibtex
@inproceedings{Zoph2022STMoEDS,
    title   = {ST-MoE: Designing Stable and Transferable Sparse Expert Models},
    author  = {Barret Zoph and Irwan Bello and Sameer Kumar and Nan Du and Yanping Huang and Jeff Dean and Noam M. Shazeer and William Fedus},
    year    = {2022}
}
```
