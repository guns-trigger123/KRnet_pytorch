import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from ns_problem import LidDrive2D
from model.pinn.pinn_net import PINN_FCN
from model.kr_net.my_distribution import NormalDistribution
from model.kr_net.kr_net import KR_net


def plot(model, bvp, name, epoch, iter, isSave, isShow):
    # real
    velocity_data = bvp.velocity_norm_real()
    X = velocity_data.iloc[:, 0:2].values
    velocity_norm_real = velocity_data.iloc[:, 2].values
    p_real = bvp.p_velocity_real().iloc[:, 2].values
    # pred
    X_tensor = torch.tensor(X, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_tensor)
    velocity_norm_pred = torch.norm(y_pred_tensor[:, 0:2], dim=1).numpy()
    p_pred = y_pred_tensor[:, 2].numpy()
    # error
    velocity_norm_error = np.abs(velocity_norm_real - velocity_norm_pred)
    velocity_norm_average_l2_error = np.sum(velocity_norm_error ** 2) / (130 * 130)
    p_error = np.abs(p_real - p_pred)
    p_average_l2_error = np.sum(p_error ** 2) / (130 * 130)

    subplots_dict = {
        'velocity_norm_real': [velocity_norm_real, (0, 0)],
        'velocity_norm_pred': [velocity_norm_pred, (0, 1)],
        'velocity_norm_error': [velocity_norm_error, (0, 2)],
        'p_real': [p_real, (1, 0)],
        'p_pred': [p_pred, (1, 1)],
        'p_error': [p_error, (1, 2)],
    }
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, layout="constrained")
    for key, value in subplots_dict.items():
        axs[value[1]].set_title(key)
        ax = axs[value[1]].scatter(X[:, 0],
                                   X[:, 1],
                                   c=value[0],
                                   marker='s',
                                   s=1,
                                   cmap='rainbow')
        fig.colorbar(ax, ax=axs[value[1]])
    plt.suptitle(f"model_epoch: {epoch} model_iterations: {iter} "
                 f"\n velocity norm average_l2_error: {velocity_norm_average_l2_error}"
                 f"\n p average_l2_error: {p_average_l2_error}")

    if isShow:
        plt.show()
    if isSave:
        PLOT_PATH = f'../plots/lid_driven_cavity/Re{bvp.Re}/{name}/'
        fig.savefig(PLOT_PATH + f'{name}_{epoch}_{iter}.png')


def plot_samples(kr_net_model, reference_distribution, Re,
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
        PLOT_PATH = f'../plots/lid_driven_cavity/Re{Re}/{name}/'
        plt.savefig(PLOT_PATH + f'krsamples_{epoch}_{iterations}.png')


if __name__ == '__main__':
    Re = 400
    lid_driven_cavity = LidDrive2D(Re=Re)
    model = PINN_FCN(2, 3)
    # kr_model = KR_net()
    # distribution = NormalDistribution()

    load_model_epoch, load_model_iter = 499, 500
    # MODEL_NAME = "aas_pinn"
    # MODEL_NAME = "das_pinn"
    MODEL_NAME = "vanilla_pinn"
    MODEL_PATH = f'../saved_models/lid_driven_cavity/Re{Re}/{MODEL_NAME}/pinn_{load_model_epoch}_{load_model_iter}.pt'
    model.load_state_dict(torch.load(MODEL_PATH))
    plot(model, lid_driven_cavity, MODEL_NAME, load_model_epoch, load_model_iter, True, True)
