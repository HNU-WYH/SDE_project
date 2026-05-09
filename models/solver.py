import numpy as np
from utils.noise_scheduler import beta


def euler_maruyama(score_model, x_T, ts, beta_min, beta_max):
    """
    Solving the reversed VP-SDE via Euler-Maruyama method:

        dx = [-1/2 * beta(t) * x - beta(t) * score(x, t)] dt + sqrt(beta(t)) dW

    Numerically solve (from T -> T_ε ≈ 0):

        x_{t-dt} = x_t + [1/2 * beta(t)*x_t + beta(t)*s(x_t,t)] * dt
                        + sqrt(beta(t) * dt) * z,   z ~ N(0, I)

    Args:
        score_model  : (x: (N,D), t: float) -> (N,D)
        x_T          : (N, D)  initial samples ~ N(0, I)
        ts           : 1-D array, decreasing from T to ≈0
        beta_min, beta_max : VP-SDE schedule parameters

    Returns:
        trajectory : list of (N, D) arrays, length = len(ts)
    """
    x = x_T.copy()
    trajectory = [x.copy()]
    assert np.all(ts[:-1] - ts[1:] >= 0), "The time sequence should decrease from T to ≈0"

    # Start from T to T_eps
    for i in range(len(ts) - 1):
        t  = ts[i]
        dt = ts[i] - ts[i + 1]          # positive

        beta_t    = beta(t, beta_min, beta_max)
        score     = score_model(x, t)
        drift     = 0.5 * beta_t * x + beta_t * score
        diffusion = np.sqrt(beta_t * dt)

        z = np.random.standard_normal(size=x.shape)
        x = x + drift * dt + diffusion * z
        trajectory.append(x.copy())

    return trajectory


def probability_flow_ode(score_model, x_T, ts, beta_min, beta_max):
    """
    Solve deterministic Probability Flow ODE from VP-SDE, which shares the same marginals as the reverse SDE
    but without diffusion term:

        dx = [ -0.5 * beta(t) * x - 0.5 * beta(t) * score(x, t)] dt

    Solving the Probability Flow ODE by the Euler Method (step from t -> t - dt, dt > 0):
        x_{t-dt} = x_t + 1/2 * beta(t) * [x_t + s(x_t, t)] * dt

    Args:
        score_model : (x: (N,D), t: float) -> score (N,D)
        x_T         : (N, D)  initial samples ~ N(0, I)
        ts          : 1-D array, decreasing from T to ≈0
        beta_min, beta_max : VP-SDE schedule parameters

    Returns:
        trajectory : list of (N, D) arrays, length = len(ts)
    """
    x = x_T.copy()
    trajectory = [x.copy()]
    assert np.all(ts[:-1] - ts[1:] >= 0), "The time sequence should decrease from T to ≈0"

    for i in range(len(ts) - 1):
        t  = ts[i]
        dt = ts[i] - ts[i + 1]          # positive

        beta_t = beta(t, beta_min, beta_max)
        score  = score_model(x, t)
        x = x + 0.5 * beta_t * (x + score) * dt
        trajectory.append(x.copy())

    return trajectory

def predictor_corrector(score_model, x_T, ts, beta_min, beta_max,
                        n_corrector=1, snr=0.01):
    """
    Solve the reverse VP-SDE via Predictor-Corrector (PC) follow Algorithm 3 in Song et al.

    Each step:
      Predictor  — one Euler-Maruyama step from t_curr -> t_next
      Corrector  — n_corrector steps of Langevin dynamics at t_next

    Args:
        score_model        : (x: (N,D), t: float) -> (N,D)
        x_T                : (N, D)  initial samples ~ N(0, I)
        ts                 : 1-D array, decreasing from T to ≈0
        beta_min, beta_max : VP-SDE schedule parameters
        n_corrector        : Langevin steps per predictor step
        snr                : signal-to-noise ratio for Langevin step size

    Returns:
        trajectory : list of (N, D) arrays, length = len(ts)
    """
    x = x_T.copy()
    trajectory = [x.copy()]
    assert np.all(ts[:-1] - ts[1:] >= 0), "The time sequence should decrease from T to ≈0"

    for i in range(len(ts) - 1):
        t_curr = ts[i]
        t_next = ts[i + 1]
        dt     = t_curr - t_next

        # --- Predictor: Euler-Maruyama ---
        beta_t    = beta(t_curr, beta_min, beta_max)
        score     = score_model(x, t_curr)
        drift     = 0.5 * beta_t * x + beta_t * score
        diffusion = np.sqrt(beta_t * dt)

        z = np.random.standard_normal(size=x.shape)
        x = x + drift * dt + diffusion * z

        # --- Corrector: Annealed Langevin at t_next ---
        for _ in range(n_corrector):
            x = _langevin_step(score_model, x, t_next, snr)

        trajectory.append(x.copy())

    return trajectory


def _langevin_step(score_model, x, t,
                   snr: float = 0.01):
    """
    The step size of Langevin Dynamics is chosen based on the signal-noise-ratio.

    Follow Algoithm 4 of "Score-based model", we have:
        Step size:  eps = 2 (snr * ||noise|| / ||score||)^2
        Update:     x <- x + eps * score + sqrt(2*eps) * noise
    """
    score = score_model(x, t)
    noise = np.random.standard_normal(size=x.shape)

    # compute the norm, and average over the batch size √N or dimension √D
    score_norm = np.sqrt(np.mean(np.sum(score**2, axis=-1)))
    noise_norm = np.sqrt(np.mean(np.sum(noise**2, axis=-1)))
    eps = 2.0 * (snr * noise_norm / (score_norm + 1e-8)) ** 2

    return x + eps * score + np.sqrt(2.0 * eps) * noise


