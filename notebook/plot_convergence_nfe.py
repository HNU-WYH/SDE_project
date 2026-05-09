#!/usr/bin/env python3
"""Visual 2 — Convergence Analysis

Sweeps over NFE and measures Sliced Wasserstein Distance between
generated samples and true GMM samples.  Six curves (2 score × 3 samplers):

    Score \\ Sampler |  P (EM)   |  PC       |  ODE
    ----------------+-----------+-----------+-----------
    Exact           |  blue o-  |  blue s-- |  blue ^:
    Fitted (MLP)    |  red  o-  |  red  s-- |  red  ^:

NFE accounting:
    P / ODE : NFE = n_steps
    PC      : NFE = n_steps * (1 + N_CORRECTOR)

Tweak the parameters in "Experiment Setup" below before running.
"""
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
import yaml

from models.infer import build_gmm, build_score_fn
from models.solver import euler_maruyama, probability_flow_ode, predictor_corrector
from utils.wasserstein import sliced_wasserstein

# =============================================================================
# Experiment Setup
# =============================================================================
# NFE grid to sweep
NFE_LIST    = [10, 25, 50, 100, 250, 500, 1000]

# Sliced Wasserstein settings
N_SLICES    = 500    # number of random projections (more → less variance)
SW_SEED     = 42     # projection seed for reproducibility

# Reference distribution
REF_N       = 5000   # samples drawn from true GMM as ground truth

# Inference overrides (None → use value from vp_config.yaml)
N_SAMPLES   = 1000   # generated samples per NFE point
N_CORRECTOR = 1      # Langevin corrector steps per predictor step (PC only)
SNR         = 0.10   # signal-to-noise ratio for Langevin step size  (PC only)
T_EPS       = None   # stop time for reverse sampler; None → config value
# =============================================================================


def run_sweep(score_fn, sampler: str, nfe_list,
              T, t_eps, n_samples, beta_min, beta_max,
              n_corrector, snr, x_ref):
    """
    Run one sampler across all NFE budgets; return list of SWD values.

    sampler : 'p' | 'pc' | 'ode'
    """
    swd_list = []
    for nfe in nfe_list:
        steps = nfe if sampler in ('p', 'ode') else max(1, nfe // (1 + n_corrector))
        ts     = np.linspace(T, t_eps, steps + 1)
        x_init = np.random.randn(n_samples, 2)

        if sampler == 'p':
            traj = euler_maruyama(score_fn, x_init, ts, beta_min, beta_max)
        elif sampler == 'ode':
            traj = probability_flow_ode(score_fn, x_init, ts, beta_min, beta_max)
        else:
            traj = predictor_corrector(score_fn, x_init, ts, beta_min, beta_max,
                                       n_corrector=n_corrector, snr=snr)

        swd = sliced_wasserstein(traj[-1], x_ref, n_slices=N_SLICES, seed=SW_SEED)
        swd_list.append(swd)
        print(f'  NFE={nfe:5d}  steps={steps:5d}  SWD={swd:.4f}')
    return swd_list


def main():
    with open('../config/vp_config.yaml', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg['seed'])

    beta_min = cfg['noise_scheduler']['beta_min']
    beta_max = cfg['noise_scheduler']['beta_max']
    T        = cfg['noise_scheduler']['T']
    t_eps    = T_EPS if T_EPS is not None else cfg['inference']['t_eps']
    n_corrector = N_CORRECTOR if N_CORRECTOR is not None else cfg['inference']['n_corrector']
    snr         = SNR         if SNR         is not None else cfg['inference']['snr']
    n_samples   = N_SAMPLES   if N_SAMPLES   is not None else cfg['inference']['n_samples']

    print(f'NFE list    : {NFE_LIST}')
    print(f'N_samples   : {n_samples}   N_corrector: {n_corrector}   SNR: {snr}')
    print(f'N_slices    : {N_SLICES}    SW_seed: {SW_SEED}   Ref_N: {REF_N}\n')

    gmm   = build_gmm(cfg)
    x_ref = gmm.sample(REF_N)

    exact_fn,  _ = build_score_fn(cfg, gmm, use_exact=True)
    fitted_fn, _ = build_score_fn(cfg, gmm, use_exact=False)

    # ------------------------------------------------------------------
    # Sweep all 6 combinations
    # ------------------------------------------------------------------
    sweep_kw = dict(T=T, t_eps=t_eps, n_samples=n_samples,
                    beta_min=beta_min, beta_max=beta_max,
                    n_corrector=n_corrector, snr=snr, x_ref=x_ref)

    curves = {}
    for score_label, score_fn in [('Exact', exact_fn), ('Fitted', fitted_fn)]:
        for sampler in ('p', 'pc', 'ode'):
            print(f'=== {score_label} Score + {sampler.upper()} Sampler ===')
            curves[(score_label, sampler)] = run_sweep(
                score_fn, sampler, NFE_LIST, **sweep_kw)
            print()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    COLORS = {'Exact': 'steelblue', 'Fitted': 'firebrick'}
    STYLES = {
        'p':   ('o', '-',  'Euler-Maruyama (VP-SDE)'),
        'pc':  ('s', '--', 'Predictor-Corrector (VP-SDE)'),
        'ode': ('^', ':',  'Explicit Euler (ODE)'),
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for (score_label, sampler), swd_list in curves.items():
        marker, linestyle, sampler_name = STYLES[sampler]
        ax.plot(NFE_LIST, swd_list,
                marker=marker, linestyle=linestyle,
                color=COLORS[score_label], markersize=5,
                label=f'{score_label} Score + {sampler_name}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Function Evaluations (NFE)', fontsize=11)
    ax.set_ylabel('Sliced Wasserstein Distance', fontsize=11)
    ax.set_title('Convergence Analysis: SWD vs NFE', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    out_path = 'output/visual2_convergence.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
