import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from data.gaussian_mixture import gaussian_mix
from models.score import ScoreNet
from models.solver import euler_maruyama, probability_flow_ode, predictor_corrector


def build_gmm(cfg: dict) -> gaussian_mix:
    """Construct the GMM defined in cfg['data']['gmm']."""
    K      = cfg['data']['gmm']['K']
    R      = cfg['data']['gmm']['R']
    sigma0 = cfg['data']['gmm']['sigma']
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    mus    = np.stack([R * np.cos(angles), R * np.sin(angles)], axis=1)
    return gaussian_mix(mus, np.full((K, 2), sigma0))


def build_score_fn(cfg: dict, gmm: gaussian_mix,
                   use_exact: bool = True,
                   weights_path: str = None):
    """
    Build the score function callable  score_fn(x: ndarray, t: float) -> ndarray.

    Args:
        cfg          : full config dict.
        gmm          : gaussian_mix — needed for the exact score closure.
        use_exact    : True  → analytic score from GMM.
                       False → trained ScoreNet loaded from weights_path.
        weights_path : path to ScoreNet .pt file (only used when use_exact=False).
                       Defaults to models/score_net.pt relative to this file.

    Returns:
        score_fn : callable  (x, t) -> score
        label    : str, short name for plot titles / legends.
    """
    beta_min = cfg['noise_scheduler']['beta_min']
    beta_max = cfg['noise_scheduler']['beta_max']

    if use_exact:
        score_fn = lambda x, t: gmm.exact_score(x, t, beta_min, beta_max)
        label    = 'Exact Score'
    else:
        if weights_path is None:
            weights_path = os.path.join(os.path.dirname(__file__), 'score_net.pt')

        train_cfg = cfg['training']
        model = ScoreNet(
            data_dim     = 2,
            hidden_dim   = train_cfg['hidden_dim'],
            n_layers     = train_cfg['n_layers'],
            time_emb_dim = train_cfg['time_emb_dim'],
            min_freq     = train_cfg['min_freq'],
            max_freq     = train_cfg['max_freq'],
        )
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        model.eval()
        score_fn = model.score_fn
        label    = 'Fitted Score (MLP)'

    return score_fn, label


def run_samplers(cfg: dict, score_fn,
                 n_samples: int = None,
                 save_path: str = None) -> dict:
    """
    Run all three reverse samplers from x_T ~ N(0, I).

    Args:
        cfg        : full config dict.
        score_fn   : callable (x: ndarray, t: float) -> score ndarray.
        n_samples  : number of samples to generate; overrides cfg['inference']['n_samples'].
        save_path  : if given, save trajectories to this .npz path.
                     Each trajectory is stored as an array of shape (n_steps+1, N, D).

    Returns:
        dict with keys:
            'em'  : list of (N, D) arrays — Euler-Maruyama trajectory
            'ode' : list of (N, D) arrays — Probability Flow ODE trajectory
            'pc'  : list of (N, D) arrays — Predictor-Corrector trajectory
            'ts'  : 1-D array — time grid used (T -> t_eps)
    """
    beta_min    = cfg['noise_scheduler']['beta_min']
    beta_max    = cfg['noise_scheduler']['beta_max']
    T           = cfg['noise_scheduler']['T']
    t_eps       = cfg['inference']['t_eps']
    n_steps     = cfg['inference']['n_steps']
    n_corrector = cfg['inference']['n_corrector']
    snr         = cfg['inference']['snr']

    if n_samples is None:
        n_samples = cfg['inference']['n_samples']

    ts  = np.linspace(T, t_eps, n_steps + 1)
    x_T = np.random.randn(n_samples, 2)

    print('Running Euler-Maruyama...')
    traj_em  = euler_maruyama(score_fn, x_T, ts, beta_min, beta_max)

    print('Running Probability Flow ODE...')
    traj_ode = probability_flow_ode(score_fn, x_T, ts, beta_min, beta_max)

    print('Running Predictor-Corrector...')
    traj_pc  = predictor_corrector(score_fn, x_T, ts, beta_min, beta_max,
                                   n_corrector=n_corrector, snr=snr)
    print('Done.')

    results = {'em': traj_em, 'ode': traj_ode, 'pc': traj_pc, 'ts': ts}

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        np.savez(save_path,
                 em  = np.array(traj_em),    # (n_steps+1, N, D)
                 ode = np.array(traj_ode),
                 pc  = np.array(traj_pc),
                 ts  = ts)
        print(f'Trajectories saved to {save_path!r}.')

    return results


def load_results(path: str) -> dict:
    """
    Load trajectories saved by run_samplers.

    Returns the same dict format as run_samplers:
        'em', 'ode', 'pc' as lists of (N, D) arrays, 'ts' as a 1-D array.
    """
    data = np.load(path)
    return {
        'em':  list(data['em']),
        'ode': list(data['ode']),
        'pc':  list(data['pc']),
        'ts':  data['ts'],
    }
