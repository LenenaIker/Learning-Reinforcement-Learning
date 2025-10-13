
from __future__ import annotations
import math
import torch
import torch.nn as nn


def fanin_init(layer: nn.Linear):
    bound = 1.0 / math.sqrt(layer.weight.data.size()[0])
    nn.init.uniform_(layer.weight.data, -bound, bound)
    nn.init.uniform_(layer.bias.data, -bound, bound)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden = (256, 256), act = nn.ReLU, out_act = None):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            lin = nn.Linear(last, h)
            fanin_init(lin)
            layers += [lin, act()]
            last = h
        out = nn.Linear(last, out_dim)
        nn.init.uniform_(out.weight, -3e-3, 3e-3)
        nn.init.uniform_(out.bias, -3e-3, 3e-3)
        if out_act:
            layers += [out, out_act()]
        else:
            layers += [out]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.body = MLP(obs_dim, act_dim, hidden = (256, 256))

    def forward(self, obs):
        # Salida no acotada; la acotamos con tanh fuera y re-escalamos a [low, high]
        return self.body(obs)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q = MLP(obs_dim + act_dim, 1, hidden = (256, 256))

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim = -1)
        return self.q(x)


