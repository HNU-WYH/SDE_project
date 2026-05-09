import numpy as np
import torch
import torch.nn as nn


class FourierTimeEmbedding(nn.Module):
    def __init__(self, dim: int, min_freq: float = 0.5, max_freq: float = 64.0):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2

        freqs = torch.pow(10, torch.linspace(np.log10(min_freq), np.log10(max_freq), half))
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = 2.0 * np.pi * t[:, None] * self.freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ScoreNet(nn.Module):
    """
    MLP score network:  (x, t) -> score = ∇_x log p_t(x)

    Architecture: sinusoidal time embedding concatenated with x,
    followed by a stack of Linear + SiLU layers.
    """

    def __init__(self,
                 data_dim:    int = 2,
                 hidden_dim:  int = 256,
                 n_layers:    int = 4,
                 time_emb_dim: int = 64,
                 min_freq:    float = 0.5,
                 max_freq:    float = 64.0):

        super().__init__()
        self.time_emb = FourierTimeEmbedding(time_emb_dim, min_freq, max_freq)

        dims = [data_dim + time_emb_dim] + [hidden_dim] * n_layers + [data_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (N, data_dim)  noisy samples
            t : (N,)           diffusion times

        Returns:
            score : (N, data_dim)
        """
        return self.net(torch.cat([x, self.time_emb(t)], dim=-1))

    def score_fn(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Numpy wrapper matching the solver interface score_fn(x, t).

        Args:
            x : (N, D)   noisy samples
            t : scalar   diffusion time (broadcast to all samples)

        Returns:
            score : (N, D)
        """
        self.eval()
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32)
            t_t = torch.full((len(x),), t, dtype=torch.float32)
            return self.forward(x_t, t_t).numpy()
