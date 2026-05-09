import os
import numpy as np

from utils.noise_scheduler import alpha, sigma2


class gaussian_mix:
    """
    N-dimensional Gaussian Mixture Model with anisotropic covariance.

    Args:
        mus (K, D): Mean vector of each component.
        sigmas scalar, (K), (K, D): standard deviation of each component.
        weights (K,): weights, uniform by default.
    """

    def __init__(self,
                 mus: np.ndarray[float],
                 sigmas: np.ndarray[float] = 1.0,
                 weights: np.ndarray[float] = None):

        assert mus.ndim == 2, "mus must be 2-D"

        self.mus = mus
        self.K, self.D = mus.shape

        if weights is None:
            self.weights = np.ones(self.K) / self.K
        else:
            weights = weights.flatten()
            assert len(weights) == self.K
            self.weights = weights / weights.sum()

        if np.isscalar(sigmas):
            sigmas = np.full(shape=(self.K, self.D), fill_value=sigmas, dtype=float)
        elif sigmas.ndim == 1:
            if len(sigmas) == self.K:
                sigmas = np.repeat(sigmas[:, np.newaxis], axis=1, repeats=self.D)
            elif len(sigmas) == self.D:
                sigmas = np.repeat(sigmas[np.newaxis, :], axis=0, repeats=self.K)
            else:
                raise ValueError(f"sigmas shape is incompatible with mus, which matches neither K={self.K} nor D={self.D}")
        elif sigmas.ndim == 2:
            if sigmas.shape != (self.K, self.D):
                raise ValueError(f"sigmas shape {sigmas.shape} must be ({self.K}, {self.D})")
        else:
            raise ValueError("sigmas must be scalar, 1-D, or 2-D")

        assert (sigmas > 0).all(), "all sigmas must be positive"
        self.sigmas = sigmas  # (K, D)

    def sample(self, N: int) -> np.ndarray:
        """
        Random sampling N i.i.d. samples from the mixed distribution.

        Args:
            N (int): number of samples.

        Returns:
            (N, D): i.i.d. samples from the mixture.
        """
        components = np.random.choice(self.K, size=N, p=self.weights)
        eps = np.random.standard_normal(size=(N, self.D))
        return self.mus[components] + eps * self.sigmas[components]

    def save(self, path: str, N: int) -> None:
        """
        Sample N points and save as .npz at path.
        Skips silently if the file already exists.

        Args:
            path : output file path (e.g. "data/gmm.npz")
            N    : number of samples to draw
        """
        if os.path.exists(path):
            print(f"Dataset already exists at '{path}', skipping.")
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        samples = self.sample(N)
        np.savez(path, samples=samples)
        print(f"Saved {N} samples to '{path}'.")

    def density(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the probability density at given points (t=0).

        Args:
            x : array-like, shape (N, D)

        Returns:
            np.ndarray, shape (N,)
        """
        x = np.asarray(x, dtype=float)
        assert x.ndim == 2
        assert x.shape[-1] == self.D, f"last dim of x must be D={self.D}"

        out = np.zeros(x.shape[0])
        for k in range(self.K):
            det_sqrt = np.prod(self.sigmas[k])
            norm = 1.0 / (det_sqrt * (2 * np.pi) ** (self.D / 2))
            exp  = np.exp(-0.5 * np.sum(((x - self.mus[k]) / self.sigmas[k]) ** 2, axis=-1))
            out += self.weights[k] * norm * exp
        return out

    def vp_density(self, x: np.ndarray, t: float,
                   beta_min: float, beta_max: float) -> np.ndarray:
        """
        Evaluate the VP-SDE perturbed density p_t(x).

        Under the VP-SDE forward process, p_t is still a Gaussian mixture:
            mu_k(t)    = alpha(t) * mu_k
            sigma_k(t) = sqrt(alpha(t)^2 * sigma_k^2 + sigma2(t))   (per dimension)

        Args:
            x        : array-like, shape (N, D)
            t        : diffusion time in [0, T]
            beta_min : VP-SDE schedule parameter
            beta_max : VP-SDE schedule parameter

        Returns:
            np.ndarray, shape (N,)
        """
        x = np.asarray(x, dtype=float)
        assert x.ndim == 2
        assert x.shape[-1] == self.D, f"last dim of x must be D={self.D}"

        a  = alpha(t, beta_min, beta_max)
        s2 = sigma2(t, beta_min, beta_max)

        out = np.zeros(x.shape[0])
        for k in range(self.K):
            mu_t    = a * self.mus[k]
            sigma_t = np.sqrt(a**2 * self.sigmas[k]**2 + s2)
            det_sqrt = np.prod(sigma_t)
            norm     = 1.0 / (det_sqrt * (2 * np.pi) ** (self.D / 2))
            exp      = np.exp(-0.5 * np.sum(((x - mu_t) / sigma_t) ** 2, axis=-1))
            out += self.weights[k] * norm * exp
        return out

    def exact_score(self, x: np.ndarray, t: float,
                    beta_min: float, beta_max: float) -> np.ndarray:
        """
        Closed-form score ∇_x log p_t(x) for the VP-SDE perturbed GMM.

        p_t is a GMM with perturbed parameters:
            mu_k(t)       = alpha(t) * mu_k
            sigma_k(t)^2  = alpha(t)^2 * sigma_k^2 + sigma2(t)   (per dimension)

        Score = sum_k w_k(x,t) * [ -Sigma_k(t)^{-1} (x - mu_k(t)) ]
        where w_k are posterior mixture weights (computed via log-sum-exp).

        Args:
            x        : (N, D)
            t        : diffusion time in [0, T]
            beta_min, beta_max : VP-SDE schedule parameters

        Returns:
            score : (N, D)  ∇_x log p_t(x)
        """
        x = np.asarray(x, dtype=float)
        assert x.ndim == 2 and x.shape[-1] == self.D

        a  = alpha(t, beta_min, beta_max)
        s2 = sigma2(t, beta_min, beta_max)

        mu_t       = a * self.mus                      # (K, D)
        sigma_t_sq = a**2 * self.sigmas**2 + s2        # (K, D)  variance per dim

        # diff[n, k, d] = x[n, d] - mu_t[k, d]
        diff = x[:, np.newaxis, :] - mu_t[np.newaxis, :, :]   # (N, K, D)

        # Log-gaussian of each component at each point
        log_norm  = -0.5 * np.sum(np.log(2 * np.pi * sigma_t_sq), axis=-1)  # (K,)
        log_exp   = -0.5 * np.sum(diff**2 / sigma_t_sq[np.newaxis], axis=-1) # (N, K)
        log_gauss = log_norm[np.newaxis] + log_exp                            # (N, K)

        # Posterior weights via log-sum-exp for numerical stability
        log_w     = np.log(self.weights)[np.newaxis] + log_gauss  # (N, K)
        log_w    -= log_w.max(axis=1, keepdims=True)
        w         = np.exp(log_w)
        w        /= w.sum(axis=1, keepdims=True)                  # (N, K)

        # Per-component score: -Sigma_k(t)^{-1} (x - mu_k(t)), diagonal case
        comp_score = -diff / sigma_t_sq[np.newaxis]               # (N, K, D)

        return np.einsum('nk,nkd->nd', w, comp_score)             # (N, D)

if __name__ == "__main__":
    import yaml
    import matplotlib.colors as mcolors
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch

    cfg_path = os.path.join(os.path.dirname(__file__), "", "../config", "vp_config.yaml")
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # data generation config
    K      = cfg["data"]["gmm"]["K"]
    R      = cfg["data"]["gmm"]["R"]
    sigma0 = cfg["data"]["gmm"]["sigma"]
    N      = cfg["data"]["N"]
    path   = cfg["data"]["path"]

    # noise scheduler config
    beta_min = cfg["noise_scheduler"]["beta_min"]
    beta_max = cfg["noise_scheduler"]["beta_max"]
    T        = cfg["noise_scheduler"]["T"]

    # plot config
    n        = cfg["plot"]["vis_n"]
    xy_lim   = cfg["plot"]["xy_lim"]
    n_levels = cfg["plot"]["n_levels"]
    times    = [t_frac * T for t_frac in cfg["plot"]["times"]]

    # build GMM and save dataset
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    mus    = np.stack([R * np.cos(angles), R * np.sin(angles)], axis=1)
    sigmas = np.full((K, 2), sigma0)
    gen    = gaussian_mix(mus, sigmas)

    # 1. dat generation & save
    gen.save(path, N)

    # 2. plot the contour
    x = np.linspace(-xy_lim, xy_lim, n)
    y = np.linspace(-xy_lim, xy_lim, n)
    X, Y = np.meshgrid(x, y)
    pts  = np.stack([X.ravel(), Y.ravel()], axis=1)

    labels = [r"$p_0$", r"$p_{{T}/{3}}$", r"$p_T$"]
    colors = ["steelblue", "darkorange", "firebrick"]
    extent = [-xy_lim, xy_lim, -xy_lim, xy_lim]
    Z_list = [gen.vp_density(pts, t=t, beta_min=beta_min, beta_max=beta_max).reshape(n, n)
              for t in times]

    # 3. from transparent (low density) to opaque (high density).
    fig, ax = plt.subplots(figsize=(4, 4))
    for Z, color in zip(Z_list, colors):
        z_ceil = Z.max() * 1.001
        levels = np.linspace(Z.max() * 0.05, Z.max(), n_levels)
        alphas = np.linspace(0.04, 1.00, n_levels)

        for idx in range(len(levels) - 1):
            z_lower, z_upper, alpha_val = levels[idx], levels[idx + 1], alphas[idx]
            ax.contourf(X, Y, Z, levels=[z_lower, z_upper], colors=[color], alpha=alpha_val)
        ax.contourf(X, Y, Z, levels=[levels[-1], z_ceil], colors=[color], alpha=alphas[-1])


    # 4. other configurations
    ax.set_xlim([-xy_lim, xy_lim])
    ax.set_ylim([-xy_lim, xy_lim])
    # ax.set_xlabel(r"$x_1$")
    # ax.set_ylabel(r"$x_2$")
    # ax.set_title(r"Overlaid density plots of $p_0$, $p_{\frac{T}{3}}$, and $p_T$")

    legend_patches = [
        Patch(facecolor=c, label=lbl)
        for c, lbl in zip(colors, labels)
    ]
    ax.legend(handles=legend_patches, loc=[0.15, -0.2], ncol=3)
    plt.tight_layout()
    plt.show()
