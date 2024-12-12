import os
import yaml
import time
import logging
import torch.optim as optim

from utils import *
from model.pinn.pinn_net import PINN_FCN
from ns_problem import LidDrive2D


def train_pinn(model, criterion, opt, sch, bvp, bou_weight, SAVED_MODEL_PATH,
               samples_domain, samples_boundary_lid, samples_boundary_wall, ITERS, epoch):
    domain_zeros = torch.zeros_like(samples_domain[:, 0:1], device=samples_domain.device)
    boundary_lid_zeros = torch.zeros_like(samples_boundary_lid[:, 0:1], device=samples_boundary_lid.device)
    boundary_wall_zeros = torch.zeros_like(samples_boundary_wall[:, 0:1], device=samples_boundary_wall.device)
    for iter in range(ITERS):
        # compute output
        output_domain = model(samples_domain)
        output_boundary_lid = model(samples_boundary_lid)
        output_boundary_wall = model(samples_boundary_wall)
        # compute residual
        u, v, p = output_domain[:, 0:1], output_domain[:, 1:2], output_domain[:, 2:3]
        grad_u, grad_v, grad_p = gradient(u, samples_domain), gradient(v, samples_domain), gradient(p, samples_domain)
        lap_u, lap_v = laplace(u, samples_domain), laplace(v, samples_domain)
        res_continuity = grad_u[:, 0:1] + grad_v[:, 1:2]
        res_momentum_x = u * grad_u[:, 0:1] + v * grad_u[:, 1:2] + grad_p[:, 0:1] - 1.0 / bvp.Re * lap_u
        res_momentum_y = u * grad_v[:, 0:1] + v * grad_v[:, 1:2] + grad_p[:, 1:2] - 1.0 / bvp.Re * lap_v
        res_u_boundary_lid, res_u_boundary_wall = output_boundary_lid[:, 0:1] - 1, output_boundary_wall[:, 0:1]
        res_v_boundary_lid, res_v_boundary_wall = output_boundary_lid[:, 1:2], output_boundary_wall[:, 1:2]
        # compute loss
        loss_continuity = criterion(res_continuity, domain_zeros)
        loss_momentum_x = criterion(res_momentum_x, domain_zeros)
        loss_momentum_y = criterion(res_momentum_y, domain_zeros)
        loss_u_boundary_lid = criterion(res_u_boundary_lid, boundary_lid_zeros)
        loss_u_boundary_wall = criterion(res_u_boundary_wall, boundary_wall_zeros)
        loss_v_boundary_lid = criterion(res_v_boundary_lid, boundary_lid_zeros)
        loss_v_boundary_wall = criterion(res_v_boundary_wall, boundary_wall_zeros)
        loss = (
                loss_continuity + loss_momentum_x + loss_momentum_y +
                bou_weight * loss_u_boundary_lid + bou_weight * loss_u_boundary_wall + bou_weight * loss_v_boundary_lid + bou_weight * loss_v_boundary_wall
        )
        # optimizer update
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
        samples_domain.grad, samples_boundary_lid.grad, samples_boundary_wall.grad = None, None, None
        # log loss
        if (iter + 1) % 100 == 0:
            logging.info(
                f"epoch: {epoch} iter: {iter} loss " +
                f"continuity: {loss_continuity} momentum_x: {loss_momentum_x} momentum_y: {loss_momentum_y} " +
                f"u_lid: {loss_u_boundary_lid} u_wall: {loss_u_boundary_wall} v_lid: {loss_v_boundary_lid} v_wall: {loss_v_boundary_wall}"
            )
        # save model
        if (iter + 1) % ITERS == 0:
            save_path = os.path.join(SAVED_MODEL_PATH, f'pinn_{epoch}_{iter + 1}.pt')
            torch.save(model.state_dict(), save_path)


def uni_sample(NUM_DOMAIN, NUM_BOUNDARY, device):
    samples_domain = torch.rand(size=(NUM_DOMAIN, 2)).requires_grad_()
    samples_boundary_lid = torch.cat([torch.rand(NUM_BOUNDARY, 1), torch.ones(NUM_BOUNDARY, 1)], 1)
    samples_boundary_wall = torch.cat(
        [
            torch.cat([torch.ones(NUM_BOUNDARY, 1), torch.rand(NUM_BOUNDARY, 1)], 1),
            torch.cat([torch.zeros(NUM_BOUNDARY, 1), torch.rand(NUM_BOUNDARY, 1)], 1),
            torch.cat([torch.rand(NUM_BOUNDARY, 1), torch.zeros(NUM_BOUNDARY, 1)], 1)
        ]
    )
    return samples_domain.to(device), samples_boundary_lid.to(device), samples_boundary_wall.to(device)


if __name__ == '__main__':
    # configuration
    device = torch.device("cuda:0")
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    experiment_params, logging_params = config["experiment_params"], config["logging_params"]
    # init logging
    SAVED_MODEL_PATH, LOG_PATH = f"../saved_models/lid_driven_cavity/Re{logging_params['Re']}/{logging_params['sample_method']}/", f"../logs/lid_driven_cavity/Re{logging_params['Re']}/"
    logging.basicConfig(
        filename=f"{LOG_PATH}{logging_params['sample_method']}_loss.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s)"
    )
    with open(LOG_PATH + f"{logging_params['sample_method']}_config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    # init model
    model_pinn = PINN_FCN(2, 3).to(device)
    MODEL_PATH = f'../saved_models/lid_driven_cavity/Re400/vanilla_pinn/pinn_99_500.pt'
    model_pinn.load_state_dict(torch.load(MODEL_PATH))
    # init training preparation
    criterion_pinn = torch.nn.MSELoss()
    opt_pinn = optim.Adam(model_pinn.parameters(), lr=experiment_params["pinn_lr"])
    sch_pinn = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_pinn, T_0=experiment_params["sch_T_0"], T_mult=experiment_params["sch_T_mult"])
    # init BVP
    lid_driven_cavity = LidDrive2D(Re=logging_params['Re'])

    # training
    model_pinn.train()
    for epoch in range(experiment_params["max_epochs"]):
        # uniform resample
        samples_domain, samples_boundary_lid, samples_boundary_wall = uni_sample(experiment_params["num_dom"], experiment_params["num_bou"], device)
        # train pinn model
        t1 = time.time()
        train_pinn(model_pinn, criterion_pinn, opt_pinn, sch_pinn, lid_driven_cavity, experiment_params["pinn_bou_weight"], SAVED_MODEL_PATH,
                   samples_domain, samples_boundary_lid, samples_boundary_wall, experiment_params["max_iters"], epoch)
        t2 = time.time()
        print(f"epoch: {epoch} training pinn time: {t2 - t1} s")
