"""
Entry point — run all experiments and generate figures.

Usage
-----
    python run.py              # full run  (~2 min)
    python run.py --quick      # fast test (~20 s)
"""

import sys, os
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from experiments import (run_phenotype_experiment,
                         run_granularity_experiment,
                         run_parameter_sweep,
                         run_emotion_validation)
from plotting import (plot_phenotypes, plot_joffily, plot_granularity,
                      plot_parameter_space, plot_phase_portrait,
                      plot_circumplex, plot_emotion_validation)


def main():
    quick = '--quick' in sys.argv
    T     = 100 if quick else 300
    n_r   = 3   if quick else 5
    n_g   = 6   if quick else 10

    out = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out, exist_ok=True)

    sep = "=" * 60
    print(sep)
    print("  Counterfactual Temporal Framing — Simulation Suite")
    print(sep)

    # ── 1. Phenotype comparison ────────────────────────────
    print("\n[1/4] Phenotype experiment …")
    pheno = run_phenotype_experiment(T=T, seed=42)
    plot_phenotypes(pheno,    save_path=f'{out}/fig1_phenotypes.png')
    plot_joffily(pheno,       save_path=f'{out}/fig2_joffily.png')
    plot_phase_portrait(pheno, save_path=f'{out}/fig5_phase_portrait.png')
    plot_circumplex(pheno,     save_path=f'{out}/fig6_circumplex.png')
    print("       -> fig1, fig2, fig5, fig6 saved")

    # ── 2. Granularity sweep ───────────────────────────────
    print("\n[2/4] Granularity sweep …")
    gran = run_granularity_experiment(T=T, n_runs=n_r, base_seed=42)
    plot_granularity(gran, save_path=f'{out}/fig3_granularity.png')
    print("       -> fig3 saved")

    # ── 3. Parameter landscape ─────────────────────────────
    print("\n[3/4] Parameter sweep …")
    sweep = run_parameter_sweep(T=T, seed=42, n_pi=n_g, n_omega=n_g)
    plot_parameter_space(sweep, save_path=f'{out}/fig4_parameter_space.png')
    print("       -> fig4 saved")

    # ── 4. Emotion validation ─────────────────────────────
    print("\n[4/4] Emotion validation …")
    emo = run_emotion_validation(T=T, seed=42)
    plot_emotion_validation(emo, save_path=f'{out}/fig7_emotion_validation.png')
    print("       -> fig7, fig7_projections saved")

    print(f"\n{sep}")
    print(f"  All figures saved to {out}/")
    print(sep)


if __name__ == '__main__':
    main()
