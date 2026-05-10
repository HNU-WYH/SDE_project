#!/usr/bin/env python3
"""Visual 1 — Sampling Dynamics

Runs three reverse samplers and plots particle trajectories (T -> 0)
with p_0 and p_T density backgrounds overlaid.

Tweak the parameters in "Experiment Setup" below before running.
"""
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
import yaml

from models.infer import build_gmm, build_score_fn, run_samplers, load_results
from utils.visualize import plot_trajectories

# =============================================================================
# Experiment Setup
# =============================================================================
USE_EXACT_SCORE = False    # False → load MLP from models/score_net.pt
SAVE_DATA       = False   # True  → save trajectories to output/traj_data.npz
LOAD_DATA       = False   # True  → skip sampling, reload from output/traj_data.npz
DATA_PATH       = 'output/traj_data.npz'

# inference overrides (None → use value from vp_config.yaml)
N_SAMPLES   = 1000   # number of particles to generate
N_STEPS     = 500    # reverse discretisation steps
N_CORRECTOR = 1      # Langevin corrector steps per predictor step (PC only)
SNR         = 0.10   # signal-to-noise ratio for Langevin step size  (PC only)

# trajectory visualisation
N_TRAJ      = 300    # number of particle paths to draw
TRAJ_STEPS  = 500     # subsample trajectory to this many time snapshots
# =============================================================================


def main():
    with open('../config/vp_config.yaml', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg['seed'])

    beta_min = cfg['noise_scheduler']['beta_min']
    beta_max = cfg['noise_scheduler']['beta_max']
    T        = cfg['noise_scheduler']['T']
    vis_n    = cfg['plot']['vis_n']
    xy_lim   = cfg['plot']['xy_lim']

    # apply overrides
    if N_STEPS     is not None: cfg['inference']['n_steps']     = N_STEPS
    if N_SAMPLES   is not None: cfg['inference']['n_samples']   = N_SAMPLES
    if N_CORRECTOR is not None: cfg['inference']['n_corrector'] = N_CORRECTOR
    if SNR         is not None: cfg['inference']['snr']         = SNR

    # ------------------------------------------------------------------
    # GMM + score function
    # ------------------------------------------------------------------
    gmm = build_gmm(cfg)
    score_fn, score_label = build_score_fn(cfg, gmm, use_exact=USE_EXACT_SCORE)
    print(f'Score  : {score_label}')
    print(f'Steps  : {cfg["inference"]["n_steps"]}   '
          f'Samples: {cfg["inference"]["n_samples"]}   '
          f'SNR: {cfg["inference"]["snr"]}   '
          f'n_corrector: {cfg["inference"]["n_corrector"]}')

    # ------------------------------------------------------------------
    # Sampling (or load from disk)
    # ------------------------------------------------------------------
    if LOAD_DATA and os.path.exists(DATA_PATH):
        print(f'Loading from {DATA_PATH!r} ...')
        results = load_results(DATA_PATH)
    else:
        results = run_samplers(cfg, score_fn,
                               save_path=DATA_PATH if SAVE_DATA else None)

    traj_em  = results['em']
    traj_ode = results['ode']
    traj_pc  = results['pc']

    # ------------------------------------------------------------------
    # Figure: 1 row × 3 cols — trajectory plots
    # ------------------------------------------------------------------
    sampler_cfgs = [
        (traj_em,  'EM Solver (Euler-Maruyama)'),
        (traj_pc,  'PC Solver (EM & Langevin)'),
        (traj_ode, 'ODE Solver (Explicit Euler)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax, (traj, name) in zip(axes, sampler_cfgs):
        plot_trajectories(
            ax, traj, gmm,
            xy_lim=xy_lim, vis_n=vis_n,
            n_traj=N_TRAJ, traj_steps=TRAJ_STEPS,
            beta_min=beta_min, beta_max=beta_max, T=T,
        )
        ax.set_title(f'{name}\n with {score_label}', fontsize=10)
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
