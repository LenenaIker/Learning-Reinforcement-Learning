import math
import torch
import torch.nn as nn



def fanin_init(layer: nn.Linear):
    bound = 1.0 / math.sqrt(layer.weight.data.size()[1]) # in_features
    nn.init.uniform_(layer.weight.data, -bound, bound)
    nn.init.uniform_(layer.bias.data, -bound, bound)

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden = (256, 256), activation_fn = nn.ReLU, out_act = None):
        super().__init__()
        layers = []
        
        last = input_dim
        for h in hidden:
            lin = nn.Linear(last, h)
            fanin_init(lin)
            layers += [lin, activation_fn()]
            last = h
        
        out = nn.Linear(last, output_dim)

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
        self.body = MLP(
            obs_dim, # Input shape == Obserbations shape
            act_dim, # Output shape == Actions
            hidden = (256, 256)
        )

    def forward(self, obs):
        # Salida no acotada; la acotamos con tanh fuera y re-escalamos a [low, high]
        return self.body(obs)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q = MLP(
            obs_dim + act_dim, # Input shape == Obserbations + Actions shape
            1, # Output shape == Value
            hidden = (256, 256)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim = -1)
        return self.q(x)


