import math
import torch
import torch.nn as nn

from model.kr_net.bounded_support_mapping import BoundedSupportMapping
from model.kr_net.coupling_layer import AffineCouplingLayer
from model.kr_net.other_layers import ScaleBiasLayer, RotationLayer, NonLinearLayer


class KR_net(nn.Module):
    def __init__(self, num_scales=2, num_cf=8, input_dim=2, lb=-1.0, hb=1.0, delta=0.005, s=2):
        super(KR_net, self).__init__()
        self.lb, self.hb = lb, hb
        self.bounded_support_mapping = BoundedSupportMapping(input_dim, lb, hb, delta, s)
        self.flows = _KR_net(0, num_scales, num_cf, input_dim)

    def forward(self, x, reverse=True):
        if reverse:
            pdj = None
            sldj = None
            x, pdj, sldj = self.flows(x, pdj, sldj, reverse=reverse)
            x, pdj, sldj = self.bounded_support_mapping(x, pdj, sldj, reverse=reverse)
        else:
            pdj = torch.ones([x.shape[0], 1], device=x.device, dtype=x.dtype)
            sldj = torch.zeros([x.shape[0], 1], device=x.device, dtype=x.dtype)
            x, pdj, sldj = self.bounded_support_mapping(x, pdj, sldj, reverse=reverse)
            x, pdj, sldj = self.flows(x, pdj, sldj, reverse=reverse)

        return x, pdj, sldj


class _KR_net(nn.Module):
    def __init__(self, scale_idx: int, num_scales: int, num_cf: int, input_dim: int):
        super(_KR_net, self).__init__()
        self.input_dim = input_dim
        self.isLastBlock = scale_idx == num_scales - 1

        if self.isLastBlock:
            self.non_linear_layer = NonLinearLayer(input_dim, 30)
        else:
            self.rotation_layer = RotationLayer(input_dim)
            self.scale_bias_layers = nn.ModuleList([
                ScaleBiasLayer(input_dim) for _ in range(num_cf)
            ])
            self.coupling_layers = nn.ModuleList([
                AffineCouplingLayer(input_dim, alternating=True if i % 2 == 0 else False) for i in range(num_cf)
            ])
            self.next_block = _KR_net(scale_idx + 1, num_scales, num_cf, input_dim - 1)

    def forward(self, x, pdj=None, sldj=None, reverse=True):
        if reverse:
            if not self.isLastBlock:
                x, x_split = x.split(self.input_dim - 1, dim=1)
                x, pdj, sldj = self.next_block(x, pdj, sldj, reverse=reverse)
                x_split, pdj, sldj = self._transform_x_split(x, x_split, pdj, sldj, reverse=reverse)
                x = torch.cat((x, x_split), dim=1)

                for (scale_bias_layer, coupling_layer) in zip(reversed(self.scale_bias_layers),
                                                              reversed(self.coupling_layers)):
                    x, pdj, sldj = coupling_layer(x, pdj, sldj, reverse=reverse)
                    x, pdj, sldj = scale_bias_layer(x, pdj, sldj, reverse=reverse)
                    pass

                x, pdj, sldj = self.rotation_layer(x, pdj, sldj, reverse=reverse)
            else:
                x, pdj, sldj = self.non_linear_layer(x, pdj, sldj, reverse=reverse)
                # x, pdj, sldj = x, pdj, sldj

        else:
            if not self.isLastBlock:
                x, pdj, sldj = self.rotation_layer(x, pdj, sldj, reverse=reverse)

                for (scale_bias_layer, coupling_layer) in zip(self.scale_bias_layers, self.coupling_layers):
                    x, pdj, sldj = scale_bias_layer(x, pdj, sldj, reverse=reverse)
                    x, pdj, sldj = coupling_layer(x, pdj, sldj, reverse=reverse)
                    pass

                x, x_split = x.split(self.input_dim - 1, dim=1)
                x_split, pdj, sldj = self._transform_x_split(x, x_split, pdj, sldj, reverse=reverse)
                x, pdj, sldj = self.next_block(x, pdj, sldj, reverse=reverse)
                x = torch.cat((x, x_split), dim=1)
            else:
                x, pdj, sldj = self.non_linear_layer(x, pdj, sldj, reverse=reverse)
                # x, pdj, sldj = x, pdj, sldj

        return x, pdj, sldj

    def _transform_x_split(self, x, x_split, pdj=None, sldj=None, reverse=True):
        """
        Multiscale architecture.
        If reverse == False, compute scale-and-biased x_split and corresponding pdj.
        If reverse == True, compute inv-scale-and-biased x_split for earlier layer input.
        :param x: Reference for mean (bias) and std (scale)
        :param x_split: To be (inv-)scale-and-biased
        :param pdj: Due to scale-and-bias transform
        :param sldj: Due to scale-and-bias transform
        :param reverse: Reverse or not
        :return:
        """
        # Compute mean and std of x
        if x.shape[-1] == 1:
            x_mean = 0.0
            x_std = 1.0
        else:
            x_mean = torch.mean(x, dim=1, keepdim=True)
            x_std = torch.std(x, dim=1, keepdim=True)
        # Transform x_split
        if reverse:
            x_split = x_std * x_split + x_mean
        else:
            x_split = (x_split - x_mean) / x_std
            pdj = pdj * ((1.0 / x_std) ** x_split.shape[-1])
            sldj = sldj + math.log(1.0 / x_std) * x_split.shape[-1]

        return x_split, pdj, sldj


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    REVERSE = True
    net = KR_net().cuda()

    if not REVERSE:
        x0 = torch.rand([5000, 1]).cuda().requires_grad_() * 2 - 1
        x1 = torch.rand([5000, 1]).cuda().requires_grad_() * 2 - 1
        x = torch.cat([x0, x1], dim=1)

        plt.scatter(x0.clone().reshape(-1).cpu().detach().numpy(), x1.clone().reshape(-1).cpu().detach().numpy(), s=1)
        plt.show()

        w, _, _ = net(x, reverse=False)

    if REVERSE:
        net.eval()
        z = torch.normal(0.0, 1.0, [5000, 2]).cuda()

        plt.scatter(z[:, 0:1].clone().reshape(-1).cpu().detach().numpy(),
                    z[:, 1:2].clone().reshape(-1).cpu().detach().numpy(), s=1)
        plt.show()

        w, _, _ = net(z, reverse=True)

    plt.scatter(w[:, 0:1].clone().reshape(-1).cpu().detach().numpy(),
                w[:, 1:2].clone().reshape(-1).cpu().detach().numpy(), s=1)
    plt.show()
