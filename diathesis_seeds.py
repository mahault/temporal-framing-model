"""Eight-seed check of the 2x2 diathesis-stress result (Experiment 7) under the
frame-gated model (revision 2026-09-27). Prints final E[pi_pos] per condition.
Run: python diathesis_seeds.py [--workers 8]"""
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from experiments import run_trial, STRESS_DECAY_PROFILES

THETA = 2.0


def job(a):
    name, seed = a
    h = run_trial(**STRESS_DECAY_PROFILES[name], T=3000, seed=seed)
    return name, seed, float(h['pi_pos'][-200:].mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    jobs = [(n, 42 + s) for n in STRESS_DECAY_PROFILES for s in range(8)]
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(job, jobs))
    for n in STRESS_DECAY_PROFILES:
        v = np.array([r[2] for r in res if r[0] == n])
        print(f"{n:18s} final pi_pos = {v.mean():.2f} +/- {v.std():.2f}  "
              f"below knee ({THETA}): {(v < THETA).sum()}/8  values={np.round(v, 2)}")
