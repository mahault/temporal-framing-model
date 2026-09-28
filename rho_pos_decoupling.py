"""Which of rho_pos's three roles carries each result? (round 2, 2026-09-27)

rho_pos (pi_pos in the code) does three jobs: it sets the identity prior D, it
sets the RECALL / FUTURATE / ABSTRACT valence targets through
alpha = sigma(rho_pos - theta), and it is the state the mood layer updates and
writes back into the targets. The overrides in generative_model
(TARGET_ALPHA_OVERRIDE, D_PI_POS_OVERRIDE) cut one pathway at a time.

Experiment A, RECALL collapse (healthy rho_pos 5 vs impaired 0.2, T = 300):
  coupled       impairment enters D and the targets
  D only        targets fixed at the healthy alpha; impairment enters D only
  targets only  D fixed at the healthy prior; impairment enters the targets only
The environment's recall effectiveness follows rho_pos in every condition.

Experiment B, diathesis-stress (vulnerable rho_pos 1 vs healthy 5, stress
volatility 0.9, T = 3000): the mood layer rewrites rho_pos every 50 steps.
  coupled        the updated rho_pos rewrites the targets (paper model)
  mood cut       targets fixed at their initial alpha; the mood posterior still
                 moves but changes nothing downstream (tests the feedback loop)
  D healthy      vulnerable targets, healthy identity prior (tests the D role)
Four seeds per cell. Writes reviews/rho_pos_decoupling.md.
Run:  python rho_pos_decoupling.py --workers 22
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
THETA = 2.0


def _alpha(p):
    return 1.0 / (1.0 + np.exp(-(p - THETA)))


def _job(a):
    exp, cond, profile, seed = a
    import generative_model as gm
    from experiments import run_trial, FEEDBACK_PROFILES, STRESS_DECAY_PROFILES
    from generative_model import RECALL
    gm.TARGET_ALPHA_OVERRIDE = None
    gm.D_PI_POS_OVERRIDE = None
    if exp == "A":
        prof = FEEDBACK_PROFILES[profile]
        if cond == "D only":
            gm.TARGET_ALPHA_OVERRIDE = _alpha(5.0)
        elif cond == "targets only":
            gm.D_PI_POS_OVERRIDE = 5.0
        h = run_trial(**prof, T=300, seed=seed)
        return exp, cond, profile, seed, float(np.mean(h["action"] == RECALL))
    prof = STRESS_DECAY_PROFILES[profile]
    if cond == "mood cut":
        gm.TARGET_ALPHA_OVERRIDE = _alpha(prof["pi_pos"])
    elif cond == "D healthy":
        gm.D_PI_POS_OVERRIDE = 5.0
    h = run_trial(**prof, T=3000, seed=seed)
    return exp, cond, profile, seed, float(h["pi_pos"][-200:].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    seeds = list(range(42, 42 + args.seeds))
    jobs = [("A", c, p, s) for c, p, s in itertools.product(
        ("coupled", "D only", "targets only"), ("healthy", "recall_impaired"), seeds)]
    jobs += [("B", c, p, s) for c, p, s in itertools.product(
        ("coupled", "mood cut", "D healthy"), ("healthy_stress", "vulnerable_stress"), seeds)]
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(_job, jobs))
    else:
        res = [_job(j) for j in jobs]
    L = ["# rho_pos role decoupling (2026-09-27)", "", "Script: `rho_pos_decoupling.py`, "
         f"{args.seeds} seeds per cell.", "", "## A. RECALL collapse (healthy minus impaired RECALL proportion, T=300)", "",
         "| condition | healthy RECALL | impaired RECALL | collapse (mean, SD over seeds) |", "|---|---:|---:|---:|"]
    for c in ("coupled", "D only", "targets only"):
        h = np.array([r[4] for r in res if r[0] == "A" and r[1] == c and r[2] == "healthy"])
        i = np.array([r[4] for r in res if r[0] == "A" and r[1] == c and r[2] == "recall_impaired"])
        L.append(f"| {c} | {h.mean():.3f} | {i.mean():.3f} | {(h - i).mean():.3f} ({(h - i).std():.3f}) |")
    L += ["", "## B. Diathesis-stress (final rho_pos, T=3000; knee theta = 2)", "",
          "| condition | healthy+stress final rho_pos | vulnerable+stress final rho_pos | vulnerable below knee | holds |",
          "|---|---:|---:|---:|---:|"]
    for c in ("coupled", "mood cut", "D healthy"):
        h = np.array([r[4] for r in res if r[0] == "B" and r[1] == c and r[2] == "healthy_stress"])
        v = np.array([r[4] for r in res if r[0] == "B" and r[1] == c and r[2] == "vulnerable_stress"])
        holds = np.mean((v < THETA) & (h > THETA))
        L.append(f"| {c} | {h.mean():.2f} ({h.std():.2f}) | {v.mean():.2f} ({v.std():.2f}) | "
                 f"{(v < THETA).sum()}/{len(v)} | {holds:.2f} |")
    L.append("")
    (ROOT / "reviews" / "rho_pos_decoupling.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
