"""
Entry point — run all experiments and generate figures.

Usage
-----
    python run.py              # full run  (~3 min)
    python run.py --quick      # fast test (~30 s)
"""

import sys, os
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from experiments import (run_phenotype_experiment,
                         run_granularity_experiment,
                         run_parameter_sweep,
                         run_emotion_validation,
                         run_feedback_reliance_experiment,
                         run_chronic_stress_experiment,
                         run_stress_decay_experiment,
                         run_psychosis_experiment)
from plotting import (plot_phenotypes, plot_joffily, plot_granularity,
                      plot_parameter_space, plot_phase_portrait,
                      plot_circumplex, plot_emotion_validation,
                      plot_temporal_aiming, plot_temporal_summary,
                      plot_feedback_reliance, plot_framing_dynamics,
                      plot_chronic_stress, plot_pi_pos_dynamics,
                      plot_stress_decay, plot_psychosis)


def main():
    quick = '--quick' in sys.argv
    T     = 100 if quick else 300
    T_sd  = 300 if quick else 3000   # stress decay needs longer horizon
    n_r   = 3   if quick else 5
    n_g   = 6   if quick else 10

    out = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out, exist_ok=True)

    sep = "=" * 60
    print(sep)
    print("  Counterfactual Temporal Framing — Simulation Suite")
    print(sep)
    print(f"  Mode: {'quick' if quick else 'full'}  T={T}")
    print(sep)

    # ── 1. Phenotype comparison ────────────────────────────
    print("\n[1/11] Phenotype experiment …")
    pheno = run_phenotype_experiment(T=T, seed=42)
    plot_phenotypes(pheno,    save_path=f'{out}/fig1_phenotypes.png')
    plot_joffily(pheno,       save_path=f'{out}/fig2_joffily.png')
    plot_phase_portrait(pheno, save_path=f'{out}/fig5_phase_portrait.png')
    plot_circumplex(pheno,     save_path=f'{out}/fig6_circumplex.png')
    plot_temporal_aiming(pheno, save_path=f'{out}/fig8_temporal_aiming.png')
    print("       -> fig1, fig2, fig5, fig6, fig8 saved")

    # ── 2. Granularity sweep ───────────────────────────────
    print("\n[2/11] Granularity sweep …")
    gran = run_granularity_experiment(T=T, n_runs=n_r, base_seed=42)
    plot_granularity(gran, save_path=f'{out}/fig3_granularity.png')
    print("       -> fig3 saved (3x2: includes FUTURATE effectiveness + elaboration/nuance)")

    # ── 3. Parameter landscape ─────────────────────────────
    print("\n[3/11] Parameter sweep …")
    sweep = run_parameter_sweep(T=T, seed=42, n_pi=n_g, n_omega=n_g)
    plot_parameter_space(sweep, save_path=f'{out}/fig4_parameter_space.png')
    print("       -> fig4 saved")

    # ── 4. Emotion validation ─────────────────────────────
    print("\n[4/11] Emotion validation …")
    emo = run_emotion_validation(T=T, seed=42)
    plot_emotion_validation(emo, save_path=f'{out}/fig7_emotion_validation.png')
    print("       -> fig7, fig7_projections saved")

    # ── 5. Temporal aiming across all emotions ──────────────
    print("\n[5/11] Temporal aiming (all emotions) …")
    plot_temporal_summary(emo, save_path=f'{out}/fig9_temporal_summary.png')
    print("       -> fig9 saved")

    # ── 6. Feedback reliance (Gap 2) ────────────────────────
    print("\n[6/11] Feedback reliance experiment …")
    fb = run_feedback_reliance_experiment(T=T, seed=42)
    plot_feedback_reliance(fb, save_path=f'{out}/fig10_feedback_reliance.png')
    print("       -> fig10 saved")

    # ── 7. Framing dynamics (Gaps A, B, C) ──────────────────
    print("\n[7/11] Framing dynamics …")
    plot_framing_dynamics(pheno, save_path=f'{out}/fig11_framing_dynamics.png')
    print("       -> fig11 saved")

    # ── 8. Chronic stress (Gap D) ──────────────────────────
    print("\n[8/11] Chronic stress experiment …")
    stress = run_chronic_stress_experiment(T=T, seed=42)
    plot_chronic_stress(stress, save_path=f'{out}/fig12_chronic_stress.png')
    print("       -> fig12 saved")

    # ── 9. pi_pos dynamics (M5 mood layer) ─────────────────
    print("\n[9/11] pi_pos dynamics …")
    plot_pi_pos_dynamics(pheno, save_path=f'{out}/fig13_pi_pos_dynamics.png')
    print("       -> fig13 saved")

    # ── 10. Stress decay (emergent depression) ─────────────
    print("\n[10/11] Stress decay experiment …")
    sd = run_stress_decay_experiment(T=T_sd, seed=42)
    plot_stress_decay(sd, save_path=f'{out}/fig14_stress_decay.png')
    print("       -> fig14 saved")

    # ── 11. Psychotic decompensation (Experiment 8) ──────────
    print("\n[11/11] Psychosis experiment …")
    psycho = run_psychosis_experiment(T=T, seed=42)
    plot_psychosis(psycho, save_path=f'{out}/fig15_psychosis.png')
    print("       -> fig15 saved")

    print(f"\n{sep}")
    print(f"  All figures saved to {out}/")
    print(sep)


if __name__ == '__main__':
    main()
