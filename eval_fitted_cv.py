"""Robust 5-fold check of the FITTED parameters.

The single 70/30 split in fit_params.py reported +16% at h=2, but single
splits are high-variance. This re-evaluates the SAME fitted parameters under
5-fold participant cross-validation and reports mean +/- sd skill, so we know
whether the improvement is real or a lucky split.
"""
from __future__ import annotations

import numpy as np
from agent import Agent
from generative_model import EPS, N_FRAMES, build_model
from empirical_rebuild import load_participants, _bin_e, _bin_v, HORIZONS

FITTED = dict(pi_pos=2.0, valence_inertia=0.5, omega_e=5.0, c_pos=1.0, c_neg=1.0)


def _lin(X, y):
    coef, *_ = np.linalg.lstsq(np.asarray(X, float), np.asarray(y, float), rcond=None)
    return coef


def _rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def _r2(p, y):
    y = np.asarray(y, float); sst = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if sst < EPS else 1 - float(np.sum((y - np.asarray(p, float)) ** 2)) / sst


def drive(seq, seed):
    K = M = 8
    model = build_model(K=K, M=M, pi_pos=FITTED["pi_pos"], omega_e=FITTED["omega_e"],
                        gamma=16.0, c_pos=FITTED["c_pos"], c_neg=FITTED["c_neg"],
                        neg_val_precision=1.0, valence_inertia=FITTED["valence_inertia"])
    agent = Agent(model, gamma=16.0, pi_pos=FITTED["pi_pos"], omega_e=FITTED["omega_e"],
                  c_pos=FITTED["c_pos"], c_neg=FITTED["c_neg"], neg_val_precision=1.0,
                  valence_inertia=FITTED["valence_inertia"],
                  counterfactual_horizon=1, adaptive_counterfactual_horizon=False, seed=seed)
    v_axis = np.arange(K); preds = {h: [] for h in HORIZONS}
    for beep in seq:
        _, info = agent.step([_bin_e(beep["e"]), 1, _bin_v(beep["v"], K)])
        pi = info["pi"]; B = sum(pi[a] * model.B[a] for a in range(len(pi)))
        q = info["beliefs"].copy()
        for h in range(1, max(HORIZONS) + 1):
            q = B @ q; q = np.maximum(q, EPS); q /= q.sum()
            if h in preds:
                vm = q.reshape(K, M, N_FRAMES).sum(axis=(1, 2))
                preds[h].append(float(vm @ v_axis / max(K - 1, 1)))
    return preds


def main():
    parts = load_participants(); pids = sorted(parts)
    recs = []
    for i, pid in enumerate(pids):
        seq = parts[pid]; preds = drive(seq, 500 + i); n = len(seq)
        for idx in range(n):
            r = {"pid": pid, "v_t": seq[idx]["v"],
                 "e_t": seq[idx]["e"] if seq[idx]["e"] is not None else 0.0}
            for h in HORIZONS:
                r[f"y{h}"] = seq[idx + h]["v"] if idx + h < n else None
                r[f"m{h}"] = preds[h][idx]
            recs.append(r)

    rng = np.random.RandomState(0); order = list(pids); rng.shuffle(order)
    fold = {p: k % 5 for k, p in enumerate(order)}
    names = ["ar1_iter", "direct_v", "direct_ve", "model"]
    rmse = {nm: {h: [] for h in (1, 2, 3)} for nm in names}
    r2 = {nm: {h: [] for h in (1, 2, 3)} for nm in names}

    for k in range(5):
        tr = [r for r in recs if fold[r["pid"]] != k]
        te = [r for r in recs if fold[r["pid"]] == k]
        ca = _lin([[1, r["v_t"]] for r in tr if r["y1"] is not None],
                  [r["y1"] for r in tr if r["y1"] is not None])
        for h in (1, 2, 3):
            trh = [r for r in tr if r[f"y{h}"] is not None]
            teh = [r for r in te if r[f"y{h}"] is not None]
            yte = np.array([r[f"y{h}"] for r in teh])
            v_te = np.array([r["v_t"] for r in teh], float)
            e_te = np.array([r["e_t"] for r in teh], float)

            pa = v_te.copy()                       # iterated 1-step AR(1)
            for _ in range(h):
                pa = ca[0] + ca[1] * pa
            cdv = _lin([[1, r["v_t"]] for r in trh], [r[f"y{h}"] for r in trh])
            pdv = cdv[0] + cdv[1] * v_te           # direct h-step on valence
            cde = _lin([[1, r["v_t"], r["e_t"]] for r in trh], [r[f"y{h}"] for r in trh])
            pde = cde[0] + cde[1] * v_te + cde[2] * e_te   # direct h-step on valence+event
            cm = _lin([[1, r[f"m{h}"]] for r in trh], [r[f"y{h}"] for r in trh])
            pm = cm[0] + cm[1] * np.array([r[f"m{h}"] for r in teh])   # model

            for nm, p in (("ar1_iter", pa), ("direct_v", pdv),
                          ("direct_ve", pde), ("model", pm)):
                rmse[nm][h].append(_rmse(p, yte)); r2[nm][h].append(_r2(p, yte))

    print(f"Fitted params: {FITTED}")
    print(f"5-fold participant CV over {len(pids)} participants, {len(recs)} records\n")
    print("| h | predictor | RMSE | R2 |")
    print("|---|---|---:|---:|")
    for h in (1, 2, 3):
        for nm in names:
            print(f"| {h} | {nm} | {np.nanmean(rmse[nm][h]):.4f} | {np.nanmean(r2[nm][h]):.3f} |")
    print("\n=== Model skill vs the STRONGEST simple baseline per fold ===")
    print("| h | model RMSE | best-baseline RMSE | skill (mean +/- sd) |")
    print("|---|---:|---:|---:|")
    for h in (1, 2, 3):
        sk = []
        for k in range(5):
            best = min(rmse["ar1_iter"][h][k], rmse["direct_v"][h][k], rmse["direct_ve"][h][k])
            sk.append(100 * (1 - rmse["model"][h][k] / best))
        sk = np.array(sk)
        best_mean = min(np.nanmean(rmse[nm][h]) for nm in ("ar1_iter", "direct_v", "direct_ve"))
        print(f"| {h} | {np.nanmean(rmse['model'][h]):.4f} | {best_mean:.4f} | "
              f"{sk.mean():+.1f}% +/- {sk.std():.1f} |")


if __name__ == "__main__":
    main()
