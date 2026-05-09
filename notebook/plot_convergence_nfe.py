#!/usr/bin/env python3
"""Visual 2 — Convergence Analysis

Sweeps over NFE (Number of Function Evaluations) and measures the Sliced
Wasserstein Distance between generated samples and true GMM samples.

Three curves are plotted:
1. Exact Score + P Sampler     — baseline stochastic reverse SDE
2. Exact Score + PC Sampler    — corrector reduces discretization error
3. Fitted Score (MLP) + PC Sampler — approximation error floor from MLP
"""
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch

from data.gaussian_mixture import gaussian_mix
from models.solver import euler_maruyama, predictor_corrector
from models.score_model import ScoreNet
from utils.wasserstein import sliced_wasserstein


def run_sweep(score_fn, sampler, nfe_list, n_corrector, snr,
              T, t_eps, n_samples, beta_min, beta_max, x_ref,
              N_SLICES, SW_SEED, nc=1):
    """Run sampler at each NFE budget and return sliced Wasserstein distances."""
    swd_list = []
    for nfe in nfe_list:
        if sampler == 'p':
            steps = nfe
        else:  # pc: each step costs 1 (predictor) + nc (corrector)
            steps = max(1, nfe // (1 + nc))

        ts_sweep = np.linspace(T, t_eps, steps + 1)
        x_init = np.random.randn(n_samples, 2)

        if sampler == 'p':
            traj = euler_maruyama(score_fn, x_init, ts_sweep,
                                  beta_min, beta_max)
        else:
            traj = predictor_corrector(score_fn, x_init, ts_sweep,
                                       beta_min, beta_max,
                                       n_corrector=nc, snr=snr)
        swd = sliced_wasserstein(traj[-1], x_ref,
                                 n_slices=N_SLICES, seed=SW_SEED)
        swd_list.append(swd)
        print(f'  NFE={nfe:5d}  steps={steps:5d}  SWD={swd:.4f}')
    return swd_list


def main():
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    with open('../config/vp_config.yaml', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg['seed'])

    beta_min = cfg['noise_scheduler']['beta_min']
    beta_max = cfg['noise_scheduler']['beta_max']
    T = cfg['noise_scheduler']['T']

    t_eps = cfg['inference']['t_eps']
    n_samples = cfg['inference']['n_samples']
    n_corrector = cfg['inference']['n_corrector']
    snr = cfg['inference']['snr']

    # build GMM
    K = cfg['data']['gmm']['K']
    R = cfg['data']['gmm']['R']
    sigma0 = cfg['data']['gmm']['sigma']
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    mus = np.stack([R * np.cos(angles), R * np.sin(angles)], axis=1)
    sigmas = np.full((K, 2), sigma0)
    gmm = gaussian_mix(mus, sigmas)

    # ------------------------------------------------------------------
    # Reference samples & score functions
    # ------------------------------------------------------------------
    REF_N = 5000
    x_ref = gmm.sample(REF_N)

    NFE_LIST = [10, 25, 50, 100, 250, 500, 1000]
    N_SLICES = 200
    SW_SEED = 42

    exact_score_fn = lambda x, t: gmm.exact_score(x, t, beta_min, beta_max)

    train_cfg = cfg['training']
    _fitted_model = ScoreNet(
        data_dim=2,
        hidden_dim=train_cfg['hidden_dim'],
        n_layers=train_cfg['n_layers'],
        time_emb_dim=train_cfg['time_emb_dim'],
        min_freq=train_cfg['min_freq'],
        max_freq=train_cfg['max_freq'],
    )
    _fitted_model.load_state_dict(
        torch.load('../backup/score_net_ep20000.pt', map_location='cpu')
    )
    _fitted_model.eval()
    fitted_score_fn = _fitted_model.score_fn

    print('Reference and score functions ready.')

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    print('=== Exact Score + P Sampler ===')
    swd_p_exact = run_sweep(
        exact_score_fn, 'p', NFE_LIST, n_corrector, snr,
        T, t_eps, n_samples, beta_min, beta_max, x_ref,
        N_SLICES, SW_SEED
    )

    print('\n=== Exact Score + PC Sampler ===')
    swd_pc_exact = run_sweep(
        exact_score_fn, 'pc', NFE_LIST, n_corrector, snr,
        T, t_eps, n_samples, beta_min, beta_max, x_ref,
        N_SLICES, SW_SEED, nc=n_corrector
    )

    print('\n=== Fitted Score + PC Sampler ===')
    swd_pc_fitted = run_sweep(
        fitted_score_fn, 'pc', NFE_LIST, n_corrector, snr,
        T, t_eps, n_samples, beta_min, beta_max, x_ref,
        N_SLICES, SW_SEED, nc=n_corrector
    )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(NFE_LIST, swd_p_exact, 'o-',
            label='Exact Score + P Sampler', color='steelblue')
    ax.plot(NFE_LIST, swd_pc_exact, 's-',
            label='Exact Score + PC Sampler', color='darkorange')
    ax.plot(NFE_LIST, swd_pc_fitted, '^--',
            label='Fitted Score (MLP) + PC Sampler', color='firebrick')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Function Evaluations (NFE)', fontsize=11)
    ax.set_ylabel('Sliced Wasserstein Distance', fontsize=11)
    ax.set_title('Convergence Analysis: SWD vs NFE', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    out_path = 'output/visual2_convergence.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
