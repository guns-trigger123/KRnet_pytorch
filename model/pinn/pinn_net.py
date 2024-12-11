import torch
import torch.nn as nn


class SINActivation(nn.Module):
    def forward(self, x):
        return torch.sin(30 * x)


class PINN_FCN(nn.Module):
    def __init__(self, data_dim, output_dim, num_nerons=64):
        super().__init__()
        self.num_nerons = num_nerons
        self.fcn = nn.Sequential(
            nn.Linear(data_dim, num_nerons),
            SINActivation(),
            nn.Linear(num_nerons, num_nerons),
            nn.Tanh(),
            nn.Linear(num_nerons, num_nerons),
            nn.Tanh(),
            nn.Linear(num_nerons, output_dim),
        )

    def forward(self, x):
        for layer in self.fcn:
            x = layer(x)
        return x
