import torch
import torch.nn as nn
import numpy as np

from scipy import linalg as la
from utils import logabs


class ScaleBiasLayer(nn.Module):
    """
    Same with ActNorm in Glow.
    """

    def __init__(self, input_dim: int):
        super(ScaleBiasLayer, self).__init__()
        self.isInitialize = False

        self.scale = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x, pdj=None, sldj=None, reverse=True):
        if not self.isInitialize:
            # self._initialize(x)
            self.isInitialize = True

        if reverse:
            x = x / self.scale - self.bias
        else:
            x = self.scale * (x + self.bias)

            # Time abs-determinant of the Jacobian
            pdj = pdj * self.scale.abs().prod(-1, keepdims=True)

            sldj = sldj + self.scale.abs().log().sum(-1, keepdims=True)

        return x, pdj, sldj

    def _initialize(self, x):
        with torch.no_grad():
            mean = torch.mean(x, dim=0)
            std = torch.std(x, dim=0)
            self.bias.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))


class RotationLayer(nn.Module):
    """
    Same with Invertible 1 × 1 convolution in Glow.
    """

    def __init__(self, input_dim: int):
        super(RotationLayer, self).__init__()

        # weights = torch.normal(0, 0.1, [input_dim, input_dim])
        weights = torch.eye(input_dim, input_dim)

        self.w_l = nn.Parameter(torch.tril(weights, diagonal=-1))
        self.w_u = nn.Parameter(torch.triu(weights, diagonal=1))
        self.register_buffer("l_mask", torch.tril(torch.ones(input_dim, input_dim), diagonal=-1))
        self.register_buffer("u_mask", torch.triu(torch.ones(input_dim, input_dim), diagonal=1))
        self.register_buffer("eye", torch.eye(input_dim, input_dim))

    def forward(self, x, pdj=None, sldj=None, reverse=True):
        if reverse:
            x = x @ ((self.w_u * self.u_mask) + self.eye).inverse() @ ((self.w_l * self.l_mask) + self.eye).inverse()
            # x = x
        else:
            x = x @ ((self.w_l * self.l_mask) + self.eye) @ ((self.w_u * self.u_mask) + self.eye)
            # x, pdj = x, pdj
        return x, pdj, sldj


    def _calc_weight(self):
        weight = (
                self.w_p
                @ (self.w_l * self.l_mask + self.l_eye)
                @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight


class NonLinearLayer(nn.Module):
    """
    Inspired by Cumulative Distribution Function (CDF).
    """

    def __init__(self, input_dim: int, num_elmt: int, mesh_ratio=1.2, bound=50.0):
        super(NonLinearLayer, self).__init__()

        assert num_elmt % 2 == 0, f'num_elmt should be even while input is {input_dim}'

        self.input_dim = input_dim
        self.num_elmt = num_elmt
        self.mesh_ratio = mesh_ratio
        self.bound = bound

        # Generate a nonuniform mesh symmetric to zero and increasing mesh step by ratio r away from zero.
        # Remark that mesh stay fixed while training
        one_step = bound * (mesh_ratio - 1.0) / (torch.tensor(mesh_ratio).pow(num_elmt / 2) - 1.0)
        one_step_norm = one_step / 2.0 / bound
        mesh_index = torch.arange(-num_elmt / 2, num_elmt / 2 + 1)
        mesh_index_sign = torch.sign(mesh_index)
        mesh = (torch.tensor(mesh_ratio).pow(abs(mesh_index)) - 1.0) / (mesh_ratio - 1.0) * one_step * mesh_index_sign
        mesh_norm = (mesh + bound) / 2.0 / bound
        # Eliminate numerical error of the first and last entry in mesh_norm after above steps
        mesh_norm = torch.cat([torch.tensor([0.0]), mesh_norm[1:-1], torch.tensor([1.0])])
        elmt_size = mesh_norm[1:] - mesh_norm[:-1]

        self.register_buffer("one_step_norm", one_step_norm.reshape(-1, 1))
        self.register_buffer("mesh_norm", mesh_norm.reshape(-1, 1))
        self.register_buffer("elmt_size", elmt_size.reshape(-1, 1))

        self.log_p = nn.Parameter(torch.zeros(num_elmt - 1, input_dim))

    def forward(self, x, pdj=None, sldj=None, reverse=True):
        # Normalize the PDF
        pdf_norm, F_ref = self._normalize_pdf()

        # Rescale such points in [-bound, bound] will be mapped to [0,1]
        x = (x + self.bound) / 2.0 / self.bound

        # cdf / cdf_inverse mapping
        if reverse:
            x = self._compute_cdf_inverse(pdf_norm, F_ref, x)
        else:
            x, pdj, sldj = self._compute_cdf(pdf_norm, F_ref, x, pdj, sldj)

        # Maps [0,1] back to [-bound, bound]
        x = x * 2.0 * self.bound - self.bound

        return x, pdj, sldj

    def _normalize_pdf(self):
        """
        Normalize self.log_p to get Probability Distribution Function (PDF) at all mesh-points, (pdf_norm),
        and Cumulative Distribution Function (CDF) at lower-points of all elements for reference, (F_ref).
        :return:
        """
        # Set probability at both mesh-ends as 1 (by default) which stay fixed while training
        p0 = torch.ones(self.input_dim, device=self.log_p.device).reshape(1, -1)
        # Normalize probability at inner mesh-points
        px = torch.exp(self.log_p) * (self.elmt_size[:-1] + self.elmt_size[1:]) / 2.0
        px = torch.exp(self.log_p) * (1 - self.elmt_size[0]) / torch.sum(px, dim=0, keepdims=True)
        # Concat normalized probability (a.k.a PDF) at all mesh-points
        pdf_norm = torch.cat([p0, px, p0], 0)

        # Compute cumulative-probability in each element
        cell = (pdf_norm[:-1, :] + pdf_norm[1:, :]) / 2.0 * self.elmt_size
        # Compute CDF at lower-points of all elements, contribution from previous elements.
        F_ref = torch.zeros(self.input_dim, device=cell.device).reshape(1, -1)
        for i in range(1, self.num_elmt):
            tp = torch.sum(cell[:i, :], dim=0, keepdims=True)
            F_ref = torch.cat([F_ref, tp], 0)
        return pdf_norm, F_ref

    def _compute_cdf(self, pdf_norm, F_ref, x, pdj=None, sldj=None):
        # Get the index of interval x_i belongs to, (k_ind),
        # and create cover for x outside [-self.bound, self.bound], (cover)
        with torch.no_grad():
            k_ind = torch.searchsorted(self.mesh_norm.reshape(-1), x)
            k_ind, cover = self._process_index(k_ind)

        # Remark for x of specifically one dimension and k_ind = i, cdf formula is as followed
        # cdf = F_ref[i-1] + (x - mesh[i-1]) ** 2 / 2.0 * (pdf[i] - pdf[i-1]) / (mesh[i] - mesh[i-1])
        #       + (x - mesh[i-1]) * pdf[i]

        # Compute each item in the cdf formula:
        # v1: pdf[i-1], which contribute grad to self.log_p
        v1 = torch.gather(pdf_norm, 0, k_ind - 1)
        # v2: pdf[i], which contribute grad to self.log_p
        v2 = torch.gather(pdf_norm, 0, k_ind)
        # xmodi: x - mesh[i-1]
        xmodi = x - torch.gather(self.mesh_norm.repeat(1, self.input_dim), 0, k_ind - 1)
        # h_list: mesh[i] - mesh[i-1]
        h_list = torch.gather(self.elmt_size.repeat(1, self.input_dim), 0, k_ind - 1)
        # F_pre: F_ref[i-1]
        F_pre = torch.gather(F_ref, 0, k_ind - 1)

        # Compute cdf(x) with cdf(x) = x when x is outside [-self.bound, self.bound]
        y = torch.where(cover > 0, F_pre + xmodi ** 2 / 2.0 * (v2 - v1) / h_list + xmodi * v1, x)

        # Compute abs-determinant of the Jacobian
        if pdj is not None:
            abs_det = torch.where(cover > 0, xmodi * (v2 - v1) / h_list + v1, 1.0)
            pdj = pdj * abs_det.prod(-1, keepdims=True)
            sldj = sldj + abs_det.log().sum(-1, keepdims=True)

        return y, pdj, sldj

    def _compute_cdf_inverse(self, pdf_norm, F_ref, y):
        # Do not worry about grad actually
        with torch.no_grad():
            # Set cdf 0 at the start and 1 at the end of the mesh-points
            p0 = torch.zeros(self.input_dim, device=self.log_p.device).reshape(1, -1)
            p1 = torch.ones(self.input_dim, device=self.log_p.device).reshape(1, -1)
            # Concat to get corrected cdf(x) where x are mesh-points
            yr = torch.cat([p0, F_ref[1:, :], p1], 0)

            k_ind = torch.searchsorted(yr.t(), y.t())
            k_ind = k_ind.t()
            k_ind, cover = self._process_index(k_ind)

        # Compute each item in the cdf_inverse formula:
        v1 = torch.where(cover > 0, torch.gather(pdf_norm, 0, k_ind - 1), -1.0)
        v2 = torch.where(cover > 0, torch.gather(pdf_norm, 0, k_ind), -2.0)
        ys = y - torch.gather(yr, 0, k_ind - 1)
        xs = torch.gather(self.mesh_norm.repeat(1, self.input_dim), 0, k_ind - 1)
        h_list = torch.gather(self.elmt_size.repeat(1, self.input_dim), 0, k_ind - 1)

        # Compute cdf_inverse(y) with cdf(y) = y when x is outside [0, 1]
        tp = 2.0 * ys * h_list * (v2 - v1)
        tp += v1 * v1 * h_list * h_list
        tp = torch.sqrt(tp) - v1 * h_list
        tp = torch.where(torch.abs(v1 - v2) < 1.0e-6, ys / v1, tp / (v2 - v1))
        tp += xs
        x = torch.where(cover > 0, tp, y)

        return x

    def _process_index(self, k_ind):
        k_ind = k_ind.type(torch.int64)
        # Create cover for x outside [-self.bound, self.bound]
        cover = torch.where((k_ind != 0) & (k_ind != (self.num_elmt + 1)), 1.0, 0.0)
        # Change index without subsequent definition of computation into adjacent one for the purpose of simplification
        k_ind = torch.where(k_ind == 0, k_ind + 1, k_ind)
        k_ind = torch.where(k_ind == (self.num_elmt + 1), k_ind - 1, k_ind)
        return k_ind, cover
