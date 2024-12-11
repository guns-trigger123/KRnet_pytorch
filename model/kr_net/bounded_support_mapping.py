import math
import torch.nn as nn


class BoundedSupportMapping(nn.Module):
    def __init__(self, input_dim: int, lb=-1.0, hb=1.0, delta=0.005, s=2):
        super(BoundedSupportMapping, self).__init__()
        self.input_dim = input_dim
        self.logistic_layer = _LogisticMapping(input_dim, delta, s)
        self.affine_linear_layer = _AffineLinearMapping(input_dim, lb, hb)

    def forward(self, x, pdj=None, sldj=None, reverse=True):
        if reverse:
            # [-inf, inf] -> [lb, hb]
            x, pdj, sldj = self.logistic_layer(x, pdj, sldj)
            x, pdj, sldj = self.affine_linear_layer(x, pdj, sldj)
        else:
            # [lb, hb] -> [-inf, inf]
            x, pdj, sldj = self.affine_linear_layer(x, pdj, sldj, reverse=False)
            x, pdj, sldj = self.logistic_layer(x, pdj, sldj, reverse=False)
        return x, pdj, sldj


class _AffineLinearMapping(nn.Module):
    def __init__(self, input_dim: int, lb=-1.0, hb=1.0):
        super(_AffineLinearMapping, self).__init__()
        self.input_dim = input_dim
        self.lb, self.hb = lb, hb

    def forward(self, x, pdj=None, sldj=None, reverse=True):
        if reverse:
            # [0, 1] -> [lb, hb]
            x = x + self.lb / (self.hb - self.lb)
            x = x * (self.hb - self.lb)
        else:
            # [lb, hb] -> [0, 1]
            x = x / (self.hb - self.lb)
            x = x - self.lb / (self.hb - self.lb)
            pdj = pdj * 1.0 / ((self.hb - self.lb) ** self.input_dim)
            sldj = sldj + (-math.log(self.hb - self.lb)) * self.input_dim

        return x, pdj, sldj


class _LogisticMapping(nn.Module):
    """
    Logistic mapping, (-inf, inf) --> (0, 1):
    y = (tanh(x/2) + 1) / 2 = e^x/(1 + e^x)
    derivative: dy/dx = y* (1-y)
    inverse: x = log(y) - log(1-y)

    For PDE, data to prior direction: [a,b] ---> (-inf, inf)
    So we need to use an affine linear mapping first and then use logistic mapping
    """

    def __init__(self, input_dim: int, delta, s):
        super(_LogisticMapping, self).__init__()
        self.input_dim = input_dim
        self.delta = delta
        self.s = s

        self.tanh = nn.Tanh()

    # The direction of this mapping is not related to the flow
    # Direction between the data and the prior
    def forward(self, x, pdj=None, sldj=None, reverse=True):
        if reverse:
            # [-inf, inf] -> [0, 1]
            # x = (self.tanh(x / 2.0) + 1.0) / 2.0
            x = (x * 2 / self.s).exp()
            x1 = (x * (1 + self.delta) - self.delta) / (1 + x)
        else:
            # [0, 1] -> [-inf, inf]
            # x = torch.clamp(x, min=1.0e-6, max=1.0 - 1.0e-6)
            x1 = (x + self.delta).log() - (1 + self.delta - x).log()
            x1 = x1 * self.s / 2
            s = (1.0 / (x + self.delta) + 1.0 / (1.0 - x + self.delta)).abs()
            s = s * self.s / 2
            pdj = pdj * s.prod(-1, keepdim=True)
            sldj = sldj + s.log().sum(-1, keepdims=True)

        return x1, pdj, sldj
