"""One-factor-at-a-time decomposition of the stressed phenotype (round 2,
2026-09-27). The stressed profile of the paper changes four parameters at once
relative to the healthy one: rho_pos 5 -> 2.5, omega_e 5 -> 0.5, c_scale 1 -> 2,
volatility 0.3 -> 0.9. Each is applied alone, then all four together, four
seeds, T = 300. Writes reviews/stress_one_factor.md.
Run:  python stress_one_factor.py --workers 22
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = dict(K=8, M=8, pi_pos=5.0, omega_e=5.0, gamma=16.0, c_scale=1.0, volatility=0.3)
CHANGES = {"healthy": {}, "rho_pos 2.5": dict(pi_pos=2.5), "omega_e 0.5": dict(omega_e=0.5),
           "c_scale 2.0": dict(c_scale=2.0), "volatility 0.9": dict(volatility=0.9),
           "all four (stressed)": dict(pi_pos=2.5, omega_e=0.5, c_scale=2.0, volatility=0.9)}


def _job(a):
    name, seed = a
    from experiments import run_trial
    from generative_model import ABSTRACT, FUTURATE, RECALL
    p = dict(BASE)
    p.update(CHANGES[name])
    h = run_trial(**p, T=300, seed=seed)
    fb = h["frame_belief"].mean(axis=0)
    return name, seed, dict(q_past=float(fb[0]), q_present=float(fb[1]), q_future=float(fb[2]),
                            abstract=float(np.mean(h["action"] == ABSTRACT)),
                            futurate=float(np.mean(h["action"] == FUTURATE)),
                            recall=float(np.mean(h["action"] == RECALL)),
                            v_reward=float(h["v_reward"].mean()), valence=float(h["valence"].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    jobs = list(itertools.product(CHANGES, range(42, 42 + args.seeds)))
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(_job, jobs))
    else:
        res = [_job(j) for j in jobs]
    keys = ("q_past", "q_present", "q_future", "abstract", "futurate", "recall", "v_reward", "valence")
    L = ["# Stressed phenotype, one factor at a time (2026-09-27)", "",
         f"Script: `stress_one_factor.py`, {args.seeds} seeds, T=300. Mean over seeds (SD).", "",
         "| profile | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
    for name in CHANGES:
        rows = [r[2] for r in res if r[0] == name]
        L.append(f"| {name} | " + " | ".join(
            f"{np.mean([r[k] for r in rows]):.3f} ({np.std([r[k] for r in rows]):.3f})" for k in keys) + " |")
    L.append("")
    (ROOT / "reviews" / "stress_one_factor.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
