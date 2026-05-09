import numpy as np

from utils.noise_scheduler import sigma2, alpha


def forward_diffuse(x0: np.ndarray, t, beta_min: float, beta_max: float):
    """
    Sample from the forward VP SDE such that:
        dx = -0.5 beta(t) x dt + \sqrt{beta(t)} dw

    The closed form of x_{t} from x_{0} is:
        x_t = alpha(t) * x_0 + sigma(t) * eps,   eps ~ N(0, I)

    where alpha(t) = exp^{\int_0^t -0.5 beta(s) ds}, sigma^2(t) + alpha^2(t) = 1

    The condition score for x_t ~N(alpha(t)x_0, sigma(t)^2 I) is:
        ∇x_t log p(x_t|x_0) = - [x_t - alpha(t)x_0] / sigma(t)^2
                            = - sigma(t) * eps / sigma(t)^2
                            = - eps / sigma(t)

    Args:
        x0       : (N, D) clean samples
        t        : scalar or (N,) per-sample diffusion times
        beta_min, beta_max : linear noise schedule parameters in VP-SDE

    Returns:
        x_t : (N, D)  noisy samples
        score : (N, D)  conditional score — neural network target
    """
    a_t   = alpha(t, beta_min, beta_max)
    sigma_t   = np.sqrt(sigma2(t, beta_min, beta_max))
    eps = np.random.randn(*x0.shape)

    # broadcast (N,) -> (N, 1) when t is per-sample
    if np.ndim(t) > 0:
        a_t = a_t[:, np.newaxis]
        sigma_t = sigma_t[:, np.newaxis]

    # compute the corrupted data
    xt = a_t * x0 + sigma_t * eps

    # compute the conditional score s(x_t|x_0)
    score = - eps / sigma_t

    return xt, score