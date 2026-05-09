#!/usr/bin/env python3
"""Visual 1 — Sampling Dynamics

Generates scatter plots of samples from the three reverse samplers
(Euler-Maruyama, Probability Flow ODE, Predictor-Corrector) with the
ground-truth p_0 density in the background.

Toggle USE_EXACT_SCORE below to switch between the analytic score and the
trained MLP.
"""
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch

from data.gaussian_mixture import gaussian_mix
from models.solver import euler_maruyama, probability_flow_ode, predictor_corrector
from models.score_model import ScoreNet


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
    n_steps = cfg['inference']['n_steps']
    n_samples = cfg['inference']['n_samples']
    n_corrector = cfg['inference']['n_corrector']
    snr = cfg['inference']['snr']

    vis_n = cfg['plot']['vis_n']
    xy_lim = cfg['plot']['xy_lim']
    n_levels = cfg['plot']['n_levels']

    # build GMM
    K = cfg['data']['gmm']['K']
    R = cfg['data']['gmm']['R']
    sigma0 = cfg['data']['gmm']['sigma']
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    mus = np.stack([R * np.cos(angles), R * np.sin(angles)], axis=1)
    sigmas = np.full((K, 2), sigma0)
    gmm = gaussian_mix(mus, sigmas)

    # time grid for reverse sampling (T -> t_eps)
    ts = np.linspace(T, t_eps, n_steps + 1)

    print(f'GMM: {K} components, R={R}, sigma={sigma0}')
    print(f'Reverse steps: {n_steps},  t in [{t_eps}, {T}]')

    # ------------------------------------------------------------------
    # Score function
    # ------------------------------------------------------------------
    USE_EXACT_SCORE = True  # <-- toggle here

    if USE_EXACT_SCORE:
        score_fn = lambda x, t: gmm.exact_score(x, t, beta_min, beta_max)
        score_label = 'Exact Score'
    else:
        train_cfg = cfg['training']
        model = ScoreNet(
            data_dim=2,
            hidden_dim=train_cfg['hidden_dim'],
            n_layers=train_cfg['n_layers'],
            time_emb_dim=train_cfg['time_emb_dim'],
            min_freq=train_cfg['min_freq'],
            max_freq=train_cfg['max_freq'],
        )
        model.load_state_dict(
            torch.load('../backup/score_net_ep20000.pt', map_location='cpu')
        )
        model.eval()
        score_fn = model.score_fn
        score_label = 'Fitted Score (MLP)'

    print(f'Using: {score_label}')

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    x_T = np.random.randn(n_samples, 2)

    print('Running Euler-Maruyama...')
    traj_em = euler_maruyama(score_fn, x_T, ts, beta_min, beta_max)
    samples_em = traj_em[-1]

    print('Running Probability Flow ODE...')
    traj_ode = probability_flow_ode(score_fn, x_T, ts, beta_min, beta_max)
    samples_ode = traj_ode[-1]

    print('Running Predictor-Corrector...')
    traj_pc = predictor_corrector(
        score_fn, x_T, ts, beta_min, beta_max,
        n_corrector=n_corrector, snr=snr
    )
    samples_pc = traj_pc[-1]

    print('Done.')

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    x_grid = np.linspace(-xy_lim, xy_lim, vis_n)
    y_grid = np.linspace(-xy_lim, xy_lim, vis_n)
    X, Y = np.meshgrid(x_grid, y_grid)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z0 = gmm.density(pts).reshape(vis_n, vis_n)

    def plot_density_bg(ax, Z, color='steelblue', n_lev=n_levels):
        """Draw smooth filled density contours from transparent to opaque."""
        z_ceil = Z.max() * 1.001
        levels = np.linspace(Z.max() * 0.05, Z.max(), n_lev)
        alphas = np.linspace(0.04, 1.0, n_lev)
        for i in range(len(levels) - 1):
            ax.contourf(
                X, Y, Z, levels=[levels[i], levels[i + 1]],
                colors=[color], alpha=alphas[i]
            )
        ax.contourf(
            X, Y, Z, levels=[levels[-1], z_ceil],
            colors=[color], alpha=alphas[-1]
        )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    configs = [
        (samples_em, 'P Sampler\n(Euler-Maruyama)'),
        (samples_pc, 'PC Sampler\n(Predictor-Corrector)'),
        (samples_ode, 'ODE Sampler\n(Probability Flow)'),
    ]

    for ax, (samples, title) in zip(axes, configs):
        plot_density_bg(ax, Z0, color='steelblue')
        ax.scatter(
            samples[:, 0], samples[:, 1],
            s=4, alpha=0.4, color='white', linewidths=0
        )
        ax.set_xlim(-xy_lim, xy_lim)
        ax.set_ylim(-xy_lim, xy_lim)
        ax.set_title(f'{title}\n({score_label})', fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')

    plt.tight_layout()
    out_path = 'output/visual1_dynamics.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
