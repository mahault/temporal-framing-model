"""Regenerate the two simulation figures used in the supplement (recall impairment, future fixation)
with publication titles. Same experiments, T and seed as run.py (full mode).
Run: python regen_sim_figs.py
"""
from experiments import run_feedback_reliance_experiment, run_chronic_stress_experiment
from plotting import plot_feedback_reliance, plot_chronic_stress

fb = run_feedback_reliance_experiment(T=300, seed=42)
plot_feedback_reliance(fb, save_path="figures/fig10_feedback_reliance.png")
st = run_chronic_stress_experiment(T=300, seed=42)
plot_chronic_stress(st, save_path="figures/fig12_chronic_stress.png")
print("done")
