"""

p(xt|x0) = N(exp(-0.25 t^2(β_max-β_min)-0.5tβ_min)x0, [1 - exp(-0.5 t^2(β_max-β_min)-tβ_min)] I)

"""

import numpy as np


def beta(t, beta_min, beta_max):
    """
    Linear noise schedule: beta(t) = beta_min + t*(beta_max - beta_min)

    Args:
        t : scalar or (N,) per-sample diffusion times
        beta_min, beta_max : scalar linear noise schedule parameters in VP-SDE

    Returns:
        beta: scalar or (N,)
    """
    return beta_min + t * (beta_max - beta_min)


def int_beta(t, beta_min, beta_max):
    """
    Integral of the linear noise scheduler beta(t) from 0 to t .

    \int_0^t beta(t) dt = 0.5 t^2(β_max-β_min) + tβ_min

    Args:
        t : scalar or (N,) per-sample diffusion times
        beta_min, beta_max : scalar linear noise schedule parameters in VP-SDE

    Returns:
        beta: scalar or (N,)

    """
    return beta_min * t + 0.5 * (beta_max - beta_min) * t**2


def alpha(t, beta_min, beta_max):
    """
    Single step noise factor in the closed form of VP-SDE:
        x_t = alpha(t) * x_0 + sigma(t) * eps,   eps ~ N(0, I)
    where alpha(t) = exp(-0.5 * int_beta(t))

    Args:
        t : scalar or (N,) per-sample diffusion times
        beta_min, beta_max : scalar linear noise schedule parameters in VP-SDE
    Returns:
        alpha: scalar or (N,)
    """

    return np.exp(-0.5 * int_beta(t, beta_min, beta_max))


def sigma2(t, beta_min, beta_max):
    """
    Single step noise factor in the closed form of VP-SDE:
        x_t = alpha(t) * x_0 + sigma(t) * eps,   eps ~ N(0, I)
    where the noise variance satisfies sigma^2(t) = 1 - alpha(t)^2

    Args:
        t : scalar or (N,) per-sample diffusion times
        beta_min, beta_max : scalar linear noise schedule parameters in VP-SDE
    Returns:
        alpha: scalar or (N,)
    """
    return 1.0 - alpha(t, beta_min, beta_max) ** 2

