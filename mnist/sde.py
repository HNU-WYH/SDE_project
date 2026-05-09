import math
import torch


class VESDE:
    def __init__(self, sigma_min=0.01, sigma_max=25.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def perturb(self, x0, t):
        sigma = self.sigma(t).view(-1, 1, 1, 1)
        z = torch.randn_like(x0)
        return x0 + sigma * z, z, sigma.view(-1)

    def prior_sample(self, shape, device):
        return torch.randn(shape, device=device) * self.sigma_max

    def discretize(self, num_steps, device):
        ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        return self.sigma(ts)
