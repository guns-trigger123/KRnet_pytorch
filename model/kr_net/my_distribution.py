import torch
import math


class NormalDistribution(object):
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
        super(NormalDistribution, self).__init__()

    def prob(self, x):
        return (-0.5 * ((x / self.std) ** 2).sum(-1, keepdims=True)).exp() / (2 * (self.std ** 2) * torch.pi)

    def log_prob(self, x):
        return - math.log(2 * (self.std ** 2) * torch.pi) - 0.5 * ((x / self.std) ** 2).sum(-1, keepdims=True)

    def sample(self, size):
        return torch.normal(self.mean, self.std, size)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    a = NormalDistribution()
    samples = a.sample((1000, 2)).numpy()
    plt.scatter(samples[:, 0:1], samples[:, 1:2])
    plt.show()
