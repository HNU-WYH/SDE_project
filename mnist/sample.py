import argparse
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sde import VESDE
from model import ScoreNet, Classifier
from viz import save_grid


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def conditional_score(score_net, classifier, x, sigma, y, scale):
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        log_probs = F.log_softmax(classifier(x_in, sigma), dim=-1)
        sel = log_probs.gather(1, y.unsqueeze(1)).sum()
        cls_grad = torch.autograd.grad(sel, x_in)[0]
    with torch.no_grad():
        s = score_net(x, sigma)
    return s + scale * cls_grad


def sample(score_net, classifier, sde, y, num_steps, M, r, scale, device):
    B = y.size(0)
    score_net.eval()
    classifier.eval()
    sigmas = sde.discretize(num_steps, device)
    x = sde.prior_sample((B, 1, 28, 28), device)
    d = x[0].numel()

    for i in tqdm(reversed(range(num_steps))):
        sigma_hi = sigmas[i + 1].repeat(B)
        sigma_lo = sigmas[i].repeat(B)

        s = conditional_score(score_net, classifier, x, sigma_hi, y, scale)
        diff_sq = sigmas[i + 1] ** 2 - sigmas[i] ** 2
        x = x + diff_sq * s + diff_sq.sqrt() * torch.randn_like(x)

        for _ in range(M):
            g = conditional_score(score_net, classifier, x, sigma_lo, y, scale)
            grad_norm = g.flatten(1).norm(dim=1).mean()
            eps = 2 * (r * math.sqrt(d) / grad_norm) ** 2
            x = x + eps * g + (2 * eps).sqrt() * torch.randn_like(x)

    s = conditional_score(score_net, classifier, x, sigmas[0].repeat(B), y, scale)
    return x + sigmas[0] ** 2 * s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-per-class", type=int, default=8)
    p.add_argument("--show-per-class", type=int, default=3)
    p.add_argument("--num-steps", type=int, default=2000)
    p.add_argument("--corrector-steps", type=int, default=1)
    p.add_argument("--snr", type=float, default=0.16)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--score-ckpt", default="ckpt/score.pt")
    p.add_argument("--classifier-ckpt", default="ckpt/classifier.pt")
    p.add_argument("--out", default="samples.png")
    args = p.parse_args()

    device = get_device()
    sde = VESDE()
    score_net = ScoreNet().to(device)
    classifier = Classifier().to(device)
    score_net.load_state_dict(torch.load(args.score_ckpt, map_location=device))
    classifier.load_state_dict(torch.load(args.classifier_ckpt, map_location=device))

    n = args.num_per_class
    y = torch.arange(10, device=device).repeat_interleave(n)
    x = sample(score_net, classifier, sde, y,
               num_steps=args.num_steps, M=args.corrector_steps,
               r=args.snr, scale=args.scale, device=device)

    sigma_eval = torch.full((y.size(0),), sde.sigma_min, device=device)
    with torch.no_grad():
        log_probs = F.log_softmax(classifier(x, sigma_eval), dim=-1)
    conf = log_probs.gather(1, y.unsqueeze(1)).squeeze(1).view(10, n)
    order = conf.argsort(dim=1, descending=True)
    x = x.view(10, n, 1, 28, 28)
    x = torch.gather(x, 1, order[..., None, None, None].expand(-1, -1, 1, 28, 28))
    x = x.reshape(10 * n, 1, 28, 28)

    save_grid(x, args.out, show_per_class=args.show_per_class)


if __name__ == "__main__":
    main()
