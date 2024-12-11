import os
import time
import yaml
import logging
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import *
from model.pinn.pinn_net import PINN_FCN
from model.kr_net.kr_net import KR_net
from model.kr_net.my_distribution import NormalDistribution
from possion_problem import TwoPeakPossion
from vanilla_pinn import uni_sample


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


def train_nf(model, criterion, opt, sch, dist, SAVED_MODEL_PATH,
             samples, res, ITERS, epoch):
    for iter in range(ITERS):
        # compute loss
        y, pdj, sldj = model(samples, reverse=False)
        log_p = dist.log_prob(y) + sldj
        loss = - log_p * (res ** 2)
        loss = loss.mean()
        # optimizer update
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
        # log loss
        if (iter + 1) % 100 == 0:
            logging.info(f"nf epoch: {epoch} iter: {iter} loss:{loss}")
        # save model
        if (iter + 1) % ITERS == 0:
            save_path = os.path.join(SAVED_MODEL_PATH, f'nf_{epoch}_{iter + 1}.pt')
            torch.save(model.state_dict(), save_path)


def uni_sample_nf(model, two_peak_possion, device, NUM_DOMAIN):
    model.eval()
    samples = (torch.rand(size=(NUM_DOMAIN, 2)) * 2 - 1).to(device).requires_grad_()
    u = model(samples)
    res = -laplace(u, samples) - two_peak_possion.s(samples).detach()
    return samples.clone().detach(), res.clone().detach()


def nf_sample(NUM_DOMAIN, NUM_BOUNDARY, model_nf, dist, device):
    model_nf.eval()
    z = dist.sample((NUM_DOMAIN, 2)).to(device)
    samples_domain_temp, _, _ = model_nf(z, reverse=True)
    samples_domain = torch.clamp(samples_domain_temp, min=-1, max=1).detach().requires_grad_()
    samples_boundary = torch.cat([torch.cat([torch.ones(NUM_BOUNDARY, 1), torch.rand(NUM_BOUNDARY, 1) * 2 - 1], 1),
                                  torch.cat([-torch.ones(NUM_BOUNDARY, 1), torch.rand(NUM_BOUNDARY, 1) * 2 - 1], 1),
                                  torch.cat([torch.rand(NUM_BOUNDARY, 1) * 2 - 1, torch.ones(NUM_BOUNDARY, 1)], 1),
                                  torch.cat([torch.rand(NUM_BOUNDARY, 1) * 2 - 1, -torch.ones(NUM_BOUNDARY, 1)], 1)])
    return samples_domain.to(device), samples_boundary.to(device)


def plot_nf_sample(samples, PLOT_PATH, epoch):
    plt.figure(figsize=(10, 10))
    plt.scatter(samples[:, 0:1].clone().cpu().detach(),
                samples[:, 1:2].clone().cpu().detach(),
                s=0.6)
    plt.xlim(-1.0, 1.0), plt.ylim(-1.0, 1.0)
    # plt.show()
    plt.savefig(f"{PLOT_PATH}nf_samples_{epoch}.png")


if __name__ == '__main__':
    # configuration
    device = torch.device("cuda:0")
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    experiment_params, logging_params = config["experiment_params"], config["logging_params"]
    # init logging
    SAVED_MODEL_PATH, LOG_PATH = f"../saved_models/two_peak_possion/{logging_params['sample_method']}/", f"../logs/two_peak_possion/"
    PLOT_PATH = f"../plots/two_peak_possion/{logging_params['sample_method']}/"
    logging.basicConfig(
        filename=f"{LOG_PATH}{logging_params['sample_method']}_loss.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s)"
    )
    with open(LOG_PATH + f"{logging_params['sample_method']}_config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    # init model
    model_pinn, model_nf = PINN_FCN(2, 1).to(device), KR_net().to(device)
    # init training preparation
    criterion_pinn, criterion_nf = torch.nn.MSELoss(), None
    opt_pinn = optim.Adam(model_pinn.parameters(), lr=experiment_params["pinn_lr"])
    opt_nf = optim.Adam(model_nf.parameters(), lr=experiment_params["nf_lr"])
    sch_pinn = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_pinn, T_0=experiment_params["sch_T_0"], T_mult=experiment_params["sch_T_mult"])
    sch_nf = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_nf, T_0=experiment_params["sch_T_0"], T_mult=experiment_params["sch_T_mult"])
    ref_dist = NormalDistribution()
    # init BVP
    two_peak_possion = TwoPeakPossion()

    # training
    for epoch in range(experiment_params["max_epochs"]):
        # train nf model
        samples_nf, res_nf = uni_sample_nf(model_pinn, two_peak_possion, device, experiment_params["num_dom"])
        model_nf.train()
        t1 = time.time()
        train_nf(model_nf, criterion_nf, opt_nf, sch_nf, ref_dist, SAVED_MODEL_PATH,
                 samples_nf, res_nf, experiment_params["max_iters"], epoch)
        t2 = time.time()
        print(f"epoch: {epoch} training nf time: {t2 - t1} s")
        # nf resample
        samples_domain, samples_boundary = nf_sample(experiment_params["num_dom"], experiment_params["num_bou"], model_nf, ref_dist, device)
        # plot nf sample
        plot_nf_sample(samples_domain, PLOT_PATH, epoch)
        # train pinn model
        model_pinn.train()
        t3 = time.time()
        train_pinn(model_pinn, criterion_pinn, opt_pinn, sch_pinn, two_peak_possion, experiment_params["pinn_bou_weight"], SAVED_MODEL_PATH,
                   samples_domain, samples_boundary, experiment_params["max_iters"], epoch)
        t4 = time.time()
        print(f"epoch: {epoch} training pinn time: {t4 - t3} s")
