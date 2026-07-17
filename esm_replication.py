"""Second-ESM replication of the affect-dynamics prediction (reviewer W1).

Replicates the Geschwind-Bringmann finding (full temporal-framing model predicts
next-step within-person affect ~2x better than linear/AR baselines, held out across
participants) on an INDEPENDENT ESM sample: osf.io/83cfk (n=91, Dutch emotion ESM).

This sample has momentary affect but no event-pleasantness or worry item, so the
model is driven on valence with a neutral event input, a strictly harder test than
Geschwind (the event channel is unavailable). Same fitted global parameters, same
5-fold participant CV, same baselines as eval_fitted_cv.py.

Also runs the inertia ablation (valence_inertia=0) to show the gain is not just
persistence.
"""
from __future__ import annotations
import csv
import numpy as np
from collections import defaultdict
from agent import Agent
from generative_model import EPS, N_FRAMES, build_model
from empirical_rebuild import _bin_e, _bin_v, HORIZONS

FITTED = dict(pi_pos=2.0, valence_inertia=0.5, omega_e=5.0, c_pos=1.0, c_neg=1.0)
DATA = "data_raw/osf_83cfk_emotions_data.csv"
POS = [3, 9, 13, 15, 21, 25]   # Rustig, Ontspannen, Blij, Tevreden, Opgewekt, Enthousiast
NEG = [5, 7, 11, 17, 19, 23]   # Angstig, Neerslachtig, Verveeld, Gestresseerd, Gefrustreerd, Droevig


def _f(x):
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def load_participants():
    rows = defaultdict(list)
    with open(DATA, encoding="utf-8") as fh:
        r = csv.reader(fh); next(r)
        for row in r:
            pid, ts = row[0], _f(row[1])
            pos = [_f(row[i]) for i in POS if _f(row[i]) is not None]
            neg = [_f(row[i]) for i in NEG if _f(row[i]) is not None]
            if not pos or not neg or ts is None:
                continue
            v = np.mean(pos) - np.mean(neg)
            rows[pid].append((ts, float(np.clip((v + 100) / 200, 0.0, 1.0))))
    parts = {}
    for pid, seq in rows.items():
        seq.sort(key=lambda t: t[0])
        parts[pid] = [dict(v=vn, e=None) for _, vn in seq]
    return parts


def _lin(X, y):
    coef, *_ = np.linalg.lstsq(np.asarray(X, float), np.asarray(y, float), rcond=None)
    return coef


def _rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def _r2(p, y):
    y = np.asarray(y, float); sst = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if sst < EPS else 1 - float(np.sum((y - np.asarray(p, float)) ** 2)) / sst


def drive(seq, seed, inertia):
    K = M = 8
    model = build_model(K=K, M=M, pi_pos=FITTED["pi_pos"], omega_e=FITTED["omega_e"],
                        gamma=16.0, c_pos=FITTED["c_pos"], c_neg=FITTED["c_neg"],
                        neg_val_precision=1.0, valence_inertia=inertia)
    agent = Agent(model, gamma=16.0, pi_pos=FITTED["pi_pos"], omega_e=FITTED["omega_e"],
                  c_pos=FITTED["c_pos"], c_neg=FITTED["c_neg"], neg_val_precision=1.0,
                  valence_inertia=inertia, counterfactual_horizon=1,
                  adaptive_counterfactual_horizon=False, seed=seed)
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


def run(inertia, label):
    parts = load_participants(); pids = sorted(parts)
    recs = []
    for i, pid in enumerate(pids):
        seq = parts[pid]; preds = drive(seq, 500 + i, inertia); n = len(seq)
        for idx in range(n):
            r = {"pid": pid, "v_t": seq[idx]["v"]}
            for h in HORIZONS:
                r[f"y{h}"] = seq[idx + h]["v"] if idx + h < n else None
                r[f"m{h}"] = preds[h][idx]
            recs.append(r)
    rng = np.random.RandomState(0); order = list(pids); rng.shuffle(order)
    fold = {p: k % 5 for k, p in enumerate(order)}
    names = ["ar1_iter", "direct_v", "model"]
    r2 = {nm: {h: [] for h in HORIZONS} for nm in names}
    for k in range(5):
        tr = [r for r in recs if fold[r["pid"]] != k]
        te = [r for r in recs if fold[r["pid"]] == k]
        ca = _lin([[1, r["v_t"]] for r in tr if r["y1"] is not None],
                  [r["y1"] for r in tr if r["y1"] is not None])
        for h in HORIZONS:
            trh = [r for r in tr if r[f"y{h}"] is not None]
            teh = [r for r in te if r[f"y{h}"] is not None]
            yte = np.array([r[f"y{h}"] for r in teh]); v_te = np.array([r["v_t"] for r in teh], float)
            pa = v_te.copy()
            for _ in range(h):
                pa = ca[0] + ca[1] * pa
            cdv = _lin([[1, r["v_t"]] for r in trh], [r[f"y{h}"] for r in trh])
            pdv = cdv[0] + cdv[1] * v_te
            cm = _lin([[1, r[f"m{h}"]] for r in trh], [r[f"y{h}"] for r in trh])
            pm = cm[0] + cm[1] * np.array([r[f"m{h}"] for r in teh])
            for nm, p in (("ar1_iter", pa), ("direct_v", pdv), ("model", pm)):
                r2[nm][h].append(_r2(p, yte))
    print(f"\n=== {label} (n={len(pids)} participants, {len(recs)} records) ===")
    print("| h | ar1_iter | direct_v | model |")
    print("|---|---:|---:|---:|")
    for h in HORIZONS:
        print(f"| {h} | {np.nanmean(r2['ar1_iter'][h]):.3f} | "
              f"{np.nanmean(r2['direct_v'][h]):.3f} | {np.nanmean(r2['model'][h]):.3f} |")
    return {h: np.nanmean(r2["model"][h]) for h in HORIZONS}, \
           {h: max(np.nanmean(r2["ar1_iter"][h]), np.nanmean(r2["direct_v"][h])) for h in HORIZONS}


if __name__ == "__main__":
    m, b = run(FITTED["valence_inertia"], "FULL MODEL (osf_83cfk replication)")
    run(0.0, "ABLATION: valence_inertia=0 (is it just persistence?)")
    print("\n=== model vs best baseline (full model) ===")
    for h in HORIZONS:
        ratio = m[h] / b[h] if b[h] > 0 else float("nan")
        print(f"h={h}: model R2={m[h]:.3f}  best-baseline R2={b[h]:.3f}  ratio={ratio:.2f}x")
