import os
import sys
import yaml
import numpy as np

import torch
import torch.optim as optim

# allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.infer import build_gmm
from models.score import ScoreNet
from utils.forward_vp import forward_diffuse


# ---- Train the ScoreNet by minimizing deviation of condition scores ---
def train(cfg_path: str, save_path: str = None):
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    beta_min = cfg["noise_scheduler"]["beta_min"]
    beta_max = cfg["noise_scheduler"]["beta_max"]
    T        = cfg["noise_scheduler"]["T"]
    train_cfg = cfg["training"]

    gmm   = build_gmm(cfg)
    model = ScoreNet(
        data_dim    = 2,
        hidden_dim  = train_cfg["hidden_dim"],
        n_layers    = train_cfg["n_layers"],
        time_emb_dim= train_cfg["time_emb_dim"],
        min_freq    = train_cfg["min_freq"],
        max_freq    = train_cfg["max_freq"],
    )
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"])

    batch_size = train_cfg["batch_size"]
    n_steps    = train_cfg["n_steps"]
    log_every  = train_cfg["log_interval"]
    t_eps      = train_cfg["t_eps"]

    model.train()
    for step in range(1, n_steps + 1):
        # sample clean data and random times
        x0 = gmm.sample(batch_size)                                  # (N, 2)
        t  = np.random.uniform(t_eps, T, size=batch_size)            # (N,)

        # closed-form forward noising, output noised data & conditional score
        x_t, target = forward_diffuse(x0, t, beta_min, beta_max)      # (N, 2), (N, 2)

        # convert to torch.tensor
        x_t_th  = torch.tensor(x_t,    dtype=torch.float32)
        t_th    = torch.tensor(t,      dtype=torch.float32)
        tgt_th  = torch.tensor(target, dtype=torch.float32)

        pred = model(x_t_th, t_th)
        loss = ((pred - tgt_th) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            print(f"step {step:6d}/{n_steps}  loss {loss.item():.5f}")

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to '{save_path}'.")

    return model


if __name__ == "__main__":
    cfg_path  = os.path.join(os.path.dirname(__file__), "..", "config", "vp_config.yaml")
    save_path = os.path.join(os.path.dirname(__file__), "..", "models", "../backup/score_net_ep20000.pt")
    train(cfg_path, save_path)
