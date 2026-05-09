import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sde import VESDE
from model import ScoreNet


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ckpt", default="ckpt/score.pt")
    args = p.parse_args()

    device = get_device()
    sde = VESDE()
    model = ScoreNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = datasets.MNIST("data", train=True, download=True, transform=tf)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    eps_t = 1e-5
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for x, _ in dl:
            x = x.to(device)
            t = torch.rand(x.size(0), device=device) * (1 - eps_t) + eps_t
            x_t, z, sigma = sde.perturb(x, t)
            score = model(x_t, sigma)
            loss = ((sigma.view(-1, 1, 1, 1) * score + z) ** 2).sum(dim=[1, 2, 3]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs} loss={total / len(dl):.4f}")
        torch.save(model.state_dict(), args.ckpt)


if __name__ == "__main__":
    main()
