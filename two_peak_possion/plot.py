import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from possion_problem import TwoPeakPossion
from model.pinn.pinn_net import PINN_FCN
from model.kr_net.my_distribution import NormalDistribution
from model.kr_net.kr_net import KR_net


def plot(model, bvp, name, epoch, iterations, isSave, isShow):
    # test input
    x, y = torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256)
    input_x, input_y = torch.meshgrid(x, y, indexing='ij')
    input_x, input_y = input_x.reshape(-1, 1), input_y.reshape(-1, 1)
    input = torch.cat([input_x, input_y], dim=1).requires_grad_()

    # test output
    output = model(input)
    u_appr = output.clone().detach().numpy()
    laplace_u = -laplace(output, input).detach().numpy()

    # real output
    u_real = bvp.real_solution(input).detach().numpy()
    source = bvp.s(input).detach().numpy()

    # error
    u_error = np.abs(u_appr - u_real)
    average_l2_error = np.sum(u_error ** 2) / (256 * 256)
    loss_pde = np.abs(laplace_u - source)

    # plot
    subplots_dict = {'u_real': [u_real, (0, 0)],
                     'u_appr': [u_appr, (0, 1)],
                     'u_error': [u_error, (0, 2)],
                     'laplace_u_real': [source, (1, 0)],
                     'laplace_u_appr': [laplace_u, (1, 1)],
                     'laplace_u_error': [loss_pde, (1, 2)]}
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, layout="constrained")
    for key, value in subplots_dict.items():
        axs[value[1]].set_title(key)
        ax = axs[value[1]].scatter(input_x.reshape(-1).detach().numpy(),
                                   input_y.reshape(-1).detach().numpy(),
                                   c=value[0],
                                   s=10,
                                   cmap='rainbow')
        fig.colorbar(ax, ax=axs[value[1]])
    u_error[np.isnan(u_error)] = 0
    error_max = np.max(np.abs(u_error))
    plt.suptitle(f"model_epoch: {load_model_epoch} model_iterations: {load_model_iter} "
                 f"\n error_max: {error_max} \n average_l2_error: {average_l2_error}")

    if isShow:
        plt.show()
    if isSave:
        PLOT_PATH = f'../plots/two_peak_possion/{name}/'
        fig.savefig(PLOT_PATH + f'{name}_{epoch}_{iterations}.png')


def plot_samples(kr_net_model, reference_distribution,
                 name, epoch, iterations, isSave, isShow):
    z = reference_distribution.sample((2000, 2))
    S_domain_kr, _, _ = kr_net_model(z, reverse=True)
    S_domain_kr = S_domain_kr.detach()

    plt.scatter(S_domain_kr[:, 0:1].clone().reshape(-1).cpu().detach().numpy(),
                S_domain_kr[:, 1:2].clone().reshape(-1).cpu().detach().numpy(),
                s=1)

    if isShow:
        plt.show()
    if isSave:
        PLOT_PATH = f'../plots/two_peak_possion/{name}/'
        plt.savefig(PLOT_PATH+f'krsamples_{epoch}_{iterations}.png')


if __name__ == '__main__':
    two_peak_possion = TwoPeakPossion()
    model = PINN_FCN(2, 1)
    # kr_model = KR_net()
    # distribution = NormalDistribution()

    # plot
    load_model_epoch, load_model_iter = 49, 500
    # MODEL_NAME = "aas_pinn"
    # MODEL_NAME = "das_pinn"
    MODEL_NAME = "vanilla_pinn"
    MODEL_PATH = f'../saved_models/two_peak_possion/{MODEL_NAME}/pinn_{load_model_epoch}_{load_model_iter}.pt'
    model.load_state_dict(torch.load(MODEL_PATH))
    plot(model, two_peak_possion, MODEL_NAME, load_model_epoch, load_model_iter, True, True)
