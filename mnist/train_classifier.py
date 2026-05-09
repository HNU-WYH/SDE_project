import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sde import VESDE
from model import Classifier


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ckpt", default="ckpt/classifier.pt")
    args = p.parse_args()

    device = get_device()
    sde = VESDE()
    model = Classifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = datasets.MNIST("data", train=True, download=True, transform=tf)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)
    test_x = torch.stack([test_ds[i][0] for i in range(2000)]).to(device)
    test_y = torch.tensor([test_ds[i][1] for i in range(2000)], device=device)
    sigma_grid = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 25.0]

    @torch.no_grad()
    def per_sigma_accuracy():
        model.eval()
        accs = []
        for s in sigma_grid:
            sigma = torch.full((test_x.size(0),), s, device=device)
            x_t = test_x + s * torch.randn_like(test_x)
            preds = model(x_t, sigma).argmax(-1)
            accs.append((preds == test_y).float().mean().item())
        model.train()
        return accs

    eps_t = 1e-5
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    model.train()
    for epoch in range(args.epochs):
        total = correct = n = 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            t = torch.rand(x.size(0), device=device) * (1 - eps_t) + eps_t
            x_t, _, sigma = sde.perturb(x, t)
            logits = model(x_t, sigma)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            correct += (logits.argmax(-1) == y).sum().item()
            n += y.size(0)
        accs = per_sigma_accuracy()
        per_sigma = " ".join(f"σ={s}:{a:.2f}" for s, a in zip(sigma_grid, accs))
        print(f"epoch {epoch + 1}/{args.epochs} loss={total / len(dl):.4f} acc={correct / n:.3f} | {per_sigma}")
        torch.save(model.state_dict(), args.ckpt)


if __name__ == "__main__":
    main()
