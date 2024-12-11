import os
import yaml
import time
import logging
import torch.optim as optim

from utils import *
from model.pinn.pinn_net import PINN_FCN
from possion_problem import TwoPeakPossion


def train_pinn(model, criterion, opt, sch, bvp, bou_weight, SAVED_MODEL_PATH,
               samples_domain, samples_boundary, ITERS, epoch):
    domain_zeros, boundary_zeros = (torch.zeros_like(samples[:, 0:1], device=samples.device) for samples in (samples_domain, samples_boundary))
    for iter in range(ITERS):
        # compute residual & loss
        u = model(samples_domain)
        res_domain = -laplace(u, samples_domain) - bvp.s(samples_domain).detach()
        res_boundary = model(samples_boundary) - bvp.real_solution(samples_boundary).detach()
        loss_domain, loss_boundary = criterion(res_domain, domain_zeros), criterion(res_boundary, boundary_zeros)
        loss = loss_domain + bou_weight * loss_boundary
        # optimizer update
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
        samples_domain.grad, samples_boundary.grad = None, None
        # log loss
        if (iter + 1) % 100 == 0:
            logging.info(f"epoch: {epoch} iter: {iter} " +
                         f"loss_domain: {loss_domain} loss_boundary: {loss_boundary} loss_total: {loss} boundary_weight: {bou_weight}")
        # save model
        if (iter + 1) % ITERS == 0:
            save_path = os.path.join(SAVED_MODEL_PATH, f'pinn_{epoch}_{iter + 1}.pt')
            torch.save(model.state_dict(), save_path)


def uni_sample(NUM_DOMAIN, NUM_BOUNDARY, device):
    samples_domain = (torch.rand(size=(NUM_DOMAIN, 2)) * 2 - 1).requires_grad_()
    samples_boundary = torch.cat([torch.cat([torch.ones(NUM_BOUNDARY, 1), torch.rand(NUM_BOUNDARY, 1) * 2 - 1], 1),
                                  torch.cat([-torch.ones(NUM_BOUNDARY, 1), torch.rand(NUM_BOUNDARY, 1) * 2 - 1], 1),
                                  torch.cat([torch.rand(NUM_BOUNDARY, 1) * 2 - 1, torch.ones(NUM_BOUNDARY, 1)], 1),
                                  torch.cat([torch.rand(NUM_BOUNDARY, 1) * 2 - 1, -torch.ones(NUM_BOUNDARY, 1)], 1)])
    return samples_domain.to(device), samples_boundary.to(device)


if __name__ == '__main__':
    # configuration
    device = torch.device("cuda:0")
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    experiment_params, logging_params = config["experiment_params"], config["logging_params"]
    # init logging
    SAVED_MODEL_PATH, LOG_PATH = f"../saved_models/two_peak_possion/{logging_params['sample_method']}/", f"../logs/two_peak_possion/"
    logging.basicConfig(
        filename=f"{LOG_PATH}{logging_params['sample_method']}_loss.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s)"
    )
    with open(LOG_PATH + f"{logging_params['sample_method']}_config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    # init model
    model_pinn = PINN_FCN(2, 1).to(device)
    # init training preparation
    criterion_pinn = torch.nn.MSELoss()
    opt_pinn = optim.Adam(model_pinn.parameters(), lr=experiment_params["pinn_lr"])
    sch_pinn = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_pinn, T_0=experiment_params["sch_T_0"], T_mult=experiment_params["sch_T_mult"])
    # init BVP
    two_peak_possion = TwoPeakPossion()

    # training
    model_pinn.train()
    for epoch in range(experiment_params["max_epochs"]):
        # uniform resample
        samples_domain, samples_boundary = uni_sample(experiment_params["num_dom"], experiment_params["num_bou"], device)
        # train pinn model
        t1 = time.time()
        train_pinn(model_pinn, criterion_pinn, opt_pinn, sch_pinn, two_peak_possion, experiment_params["pinn_bou_weight"], SAVED_MODEL_PATH,
                   samples_domain, samples_boundary, experiment_params["max_iters"], epoch)
        t2 = time.time()
        print(f"epoch: {epoch} training pinn time: {t2 - t1} s")
