"""Participant bootstrap of the BDI covariate model (analysis/gamble_bdi/analyze_bdi.py).

The published intervals came from a sandwich estimator computed with the person random effects
held at their MAP values. That ignores the uncertainty in the random effects, so the interval for
the baseline (which shares its variance with the random intercept) is far too narrow, and the
Wald and likelihood-ratio results disagree. Here each bootstrap replicate reweights participants
with multinomial counts (the weighted form of resampling participants with replacement) and refits
the full covariate model, betas and random effects together, from the converged full-data fit.

Usage: python analysis/gamble_r5/bdi_boot.py WORKER NWORKERS NBOOT [--iters 700]
Writes out/boot_bdi_w{WORKER}.json with the beta matrix of every replicate.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis/gamble_bdi"))
import analyze_bdi as ab  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("worker", type=int); ap.add_argument("nworkers", type=int); ap.add_argument("nboot", type=int)
    ap.add_argument("--iters", type=int, default=700); ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    fitd = torch.load(ROOT / "analysis/gamble_bdi/out/cov_fit.pt")
    init = (fitd["beta"], fitd["rand"])
    out = []
    path = HERE / f"out/boot_bdi_w{a.worker}.json"
    if path.exists():
        out = json.loads(path.read_text())
    for r in range(a.worker, a.nboot, a.nworkers):
        if any(o["rep"] == r for o in out):
            continue
        rs = np.random.RandomState(1000 + r)
        w = np.bincount(rs.randint(0, ab.N, ab.N), minlength=ab.N).astype(float)
        beta, rand, loss = ab.fit_cov(iters=a.iters, lr=0.005, mask=w, init=init)
        out.append({"rep": r, "beta": beta.numpy().tolist(), "loss": loss})
        path.write_text(json.dumps(out))
        print("rep", r, "done", flush=True)
    print("worker done", flush=True)


if __name__ == "__main__":
    main()
