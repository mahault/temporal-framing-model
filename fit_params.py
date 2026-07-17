"""Honest parameter fit for the temporal-framing model on Geschwind ESM.

Protocol (no leakage):
  - Split participants 70/30 (fixed seed) into FIT and TEST.
  - Grid-search global params on the FIT participants, scored by pooled
    h=1 and h=2 model RMSE (affine calibration fit on FIT only).
  - Evaluate the single chosen parameter set on the untouched TEST
    participants; report skill vs AR(1) (AR(1) also fit on FIT).

Speed: predictions use the policy-averaged transition with the
counterfactual rollout OFF (horizon=1). We verified (diagnose_mechanisms.py)
that this yields valence predictions essentially identical to the adaptive
rollout (r > 0.998), so it is a safe, faster surrogate for parameter search.
"""

from __future__ import annotations

import itertools
import math
import numpy as np

from agent import Agent
from generative_model import EPS, N_FRAMES, build_model
from empirical_rebuild import load_participants, _bin_e, _bin_v, HORIZONS


def _fit_linear(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _rmse(pred, y):
    pred = np.asarray(pred, float); y = np.asarray(y, float)
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def _r2(pred, y):
    pred = np.asarray(pred, float); y = np.asarray(y, float)
    sst = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if sst < EPS else 1.0 - float(np.sum((y - pred) ** 2)) / sst


def drive(seq, params, seed):
    K, M = 8, 8
    model = build_model(K=K, M=M, pi_pos=params["pi_pos"], omega_e=params["omega_e"],
                        gamma=16.0, c_pos=params.get("c_pos", 1.0),
                        c_neg=params.get("c_neg", 1.0),
                        neg_val_precision=1.0,
                        valence_inertia=params["valence_inertia"])
    agent = Agent(model, gamma=16.0, pi_pos=params["pi_pos"],
                  omega_e=params["omega_e"], c_pos=params.get("c_pos", 1.0),
                  c_neg=params.get("c_neg", 1.0), neg_val_precision=1.0,
                  valence_inertia=params["valence_inertia"],
                  counterfactual_horizon=1,
                  adaptive_counterfactual_horizon=False, seed=seed)
    v_axis = np.arange(K)
    preds = {h: [] for h in HORIZONS}
    for beep in seq:
        obs = [_bin_e(beep["e"]), 1, _bin_v(beep["v"], K)]
        _, info = agent.step(obs)
        pi = info["pi"]
        B = sum(pi[a] * model.B[a] for a in range(len(pi)))
        q = info["beliefs"].copy()
        for h in range(1, max(HORIZONS) + 1):
            q = B @ q; q = np.maximum(q, EPS); q /= q.sum()
            if h in preds:
                vm = q.reshape(K, M, N_FRAMES).sum(axis=(1, 2))
                preds[h].append(float(vm @ v_axis / max(K - 1, 1)))
    return preds


def build_pairs(parts, pids, params, seed0):
    """Return list of dicts: v_t, y1, y2, mpred[h]."""
    recs = []
    for i, pid in enumerate(pids):
        seq = parts[pid]
        preds = drive(seq, params, seed=seed0 + i)
        n = len(seq)
        for idx in range(n):
            r = {"v_t": seq[idx]["v"]}
            for h in HORIZONS:
                r[f"y{h}"] = seq[idx + h]["v"] if idx + h < n else None
                r[f"m{h}"] = preds[h][idx]
            recs.append(r)
    return recs


def score(recs_fit, recs_test):
    """Fit affine on fit-set, return TEST skill vs AR(1) at h=1,2."""
    out = {}
    for h in (1, 2):
        fit = [r for r in recs_fit if r[f"y{h}"] is not None]
        tst = [r for r in recs_test if r[f"y{h}"] is not None]
        yf = np.array([r[f"y{h}"] for r in fit]); yt = np.array([r[f"y{h}"] for r in tst])
        # model
        c = _fit_linear([[1, r[f"m{h}"]] for r in fit], yf)
        pm = c[0] + c[1] * np.array([r[f"m{h}"] for r in tst])
        # AR(1) iterated
        c1 = _fit_linear([[1, r["v_t"]] for r in fit], np.array([r["y1"] for r in fit if r["y1"] is not None])) \
            if h == 1 else None
        ca = _fit_linear([[1, r["v_t"]] for r in [x for x in recs_fit if x["y1"] is not None]],
                         np.array([r["y1"] for r in recs_fit if r["y1"] is not None]))
        pa = np.array([r["v_t"] for r in tst], float)
        for _ in range(h):
            pa = ca[0] + ca[1] * pa
        rm_m, rm_a = _rmse(pm, yt), _rmse(pa, yt)
        out[h] = dict(model_rmse=rm_m, ar1_rmse=rm_a,
                      skill=100 * (1 - rm_m / rm_a) if rm_a else float("nan"),
                      model_r2=_r2(pm, yt), ar1_r2=_r2(pa, yt))
    return out


def fit_metric(recs):
    """Pooled h1+h2 model RMSE on the fit set itself (selection objective)."""
    tot = 0.0
    for h in (1, 2):
        rr = [r for r in recs if r[f"y{h}"] is not None]
        y = np.array([r[f"y{h}"] for r in rr])
        c = _fit_linear([[1, r[f"m{h}"]] for r in rr], y)
        p = c[0] + c[1] * np.array([r[f"m{h}"] for r in rr])
        tot += _rmse(p, y)
    return tot / 2.0


def main():
    parts = load_participants()
    pids = sorted(parts)
    rng = np.random.RandomState(0)
    rng.shuffle(pids)
    cut = int(len(pids) * 0.7)
    fit_pids, test_pids = pids[:cut], pids[cut:]
    print(f"{len(fit_pids)} fit / {len(test_pids)} test participants")

    grid = {
        "pi_pos": [2.0, 3.0, 4.0],
        "valence_inertia": [0.2, 0.35, 0.5],
        "omega_e": [3.0, 5.0],
    }
    combos = [dict(zip(grid, vals)) for vals in itertools.product(*grid.values())]
    print(f"Searching {len(combos)} parameter combinations on fit set ...")

    best, best_m = None, math.inf
    for j, params in enumerate(combos):
        recs_fit = build_pairs(parts, fit_pids, params, seed0=1000 + j * 500)
        m = fit_metric(recs_fit)
        tag = f"pi_pos={params['pi_pos']} inertia={params['valence_inertia']} omega_e={params['omega_e']}"
        print(f"  [{j+1}/{len(combos)}] {tag}  fit_rmse={m:.4f}")
        if m < best_m:
            best_m, best = m, params

    print(f"\nBest params on fit set: {best}  (fit_rmse={best_m:.4f})")
    print("Evaluating on HELD-OUT test participants ...")
    recs_fit = build_pairs(parts, fit_pids, best, seed0=99000)
    recs_test = build_pairs(parts, test_pids, best, seed0=99500)
    res = score(recs_fit, recs_test)
    print("\n=== Held-out test performance (fitted params) ===")
    for h in (1, 2):
        d = res[h]
        print(f"  h={h}: model RMSE {d['model_rmse']:.4f} (R2 {d['model_r2']:.3f}) | "
              f"AR(1) RMSE {d['ar1_rmse']:.4f} (R2 {d['ar1_r2']:.3f}) | "
              f"skill {d['skill']:+.1f}%")


if __name__ == "__main__":
    main()
