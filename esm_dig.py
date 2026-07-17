"""Dig-deeper decomposition of the ESM affect-dynamics advantage (reviewer W1, Q1).

Question: is the model's edge the event channel, the persistence-like valence
inertia, or the temporal-frame machinery? And does it hold on two samples?

For each (dataset x config) we run the SAME 5-fold participant CV and report model
held-out R^2 vs the strongest baseline on matched inputs, at h=1/2/3.

Configs:
  full        : event input (where available) + inertia=0.5   (paper's setting)
  val_only    : neutral event + inertia=0.5                    (isolates event channel)
  val_only_i0 : neutral event + inertia=0                      (isolates persistence)
"""
from __future__ import annotations
import numpy as np
from agent import Agent
from generative_model import EPS, N_FRAMES, build_model
from empirical_rebuild import _bin_e, _bin_v, HORIZONS, load_participants as load_gesch
from esm_replication import load_participants as load_osf

FITTED = dict(pi_pos=2.0, omega_e=5.0, c_pos=1.0, c_neg=1.0)


def _lin(X, y):
    coef, *_ = np.linalg.lstsq(np.asarray(X, float), np.asarray(y, float), rcond=None)
    return coef


def _r2(p, y):
    y = np.asarray(y, float); sst = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if sst < EPS else 1 - float(np.sum((y - np.asarray(p, float)) ** 2)) / sst


def drive(seq, seed, inertia, use_event):
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
        e_obs = _bin_e(beep["e"]) if use_event else 1
        _, info = agent.step([e_obs, 1, _bin_v(beep["v"], K)])
        pi = info["pi"]; B = sum(pi[a] * model.B[a] for a in range(len(pi)))
        q = info["beliefs"].copy()
        for h in range(1, max(HORIZONS) + 1):
            q = B @ q; q = np.maximum(q, EPS); q /= q.sum()
            if h in preds:
                vm = q.reshape(K, M, N_FRAMES).sum(axis=(1, 2))
                preds[h].append(float(vm @ v_axis / max(K - 1, 1)))
    return preds


def run(parts, inertia, use_event, has_event):
    pids = sorted(parts); recs = []
    for i, pid in enumerate(pids):
        seq = parts[pid]; preds = drive(seq, 500 + i, inertia, use_event); n = len(seq)
        for idx in range(n):
            r = {"pid": pid, "v_t": seq[idx]["v"],
                 "e_t": (seq[idx]["e"] if seq[idx]["e"] is not None else 0.0)}
            for h in HORIZONS:
                r[f"y{h}"] = seq[idx + h]["v"] if idx + h < n else None
                r[f"m{h}"] = preds[h][idx]
            recs.append(r)
    rng = np.random.RandomState(0); order = list(pids); rng.shuffle(order)
    fold = {p: k % 5 for k, p in enumerate(order)}
    mR = {h: [] for h in HORIZONS}; bR = {h: [] for h in HORIZONS}
    for k in range(5):
        tr = [r for r in recs if fold[r["pid"]] != k]; te = [r for r in recs if fold[r["pid"]] == k]
        ca = _lin([[1, r["v_t"]] for r in tr if r["y1"] is not None],
                  [r["y1"] for r in tr if r["y1"] is not None])
        for h in HORIZONS:
            trh = [r for r in tr if r[f"y{h}"] is not None]; teh = [r for r in te if r[f"y{h}"] is not None]
            yte = np.array([r[f"y{h}"] for r in teh]); v_te = np.array([r["v_t"] for r in teh], float)
            pa = v_te.copy()
            for _ in range(h):
                pa = ca[0] + ca[1] * pa
            cdv = _lin([[1, r["v_t"]] for r in trh], [r[f"y{h}"] for r in trh])
            pdv = cdv[0] + cdv[1] * v_te
            cands = [_r2(pa, yte), _r2(pdv, yte)]
            if has_event:                                   # event-aware baseline too
                e_te = np.array([r["e_t"] for r in teh], float)
                cde = _lin([[1, r["v_t"], r["e_t"]] for r in trh], [r[f"y{h}"] for r in trh])
                pde = cde[0] + cde[1] * v_te + cde[2] * e_te
                cands.append(_r2(pde, yte))
            cm = _lin([[1, r[f"m{h}"]] for r in trh], [r[f"y{h}"] for r in trh])
            pm = cm[0] + cm[1] * np.array([r[f"m{h}"] for r in teh])
            mR[h].append(_r2(pm, yte)); bR[h].append(max(cands))
    return {h: np.nanmean(mR[h]) for h in HORIZONS}, {h: np.nanmean(bR[h]) for h in HORIZONS}


def report(name, parts, has_event):
    print(f"\n########## {name} (n={len(parts)}) ##########")
    configs = [("full", 0.5, has_event), ("val_only", 0.5, False), ("val_only_i0", 0.0, False)]
    print("| config | h1 model/base | h2 model/base | h3 model/base |")
    print("|---|---|---|---|")
    for cname, inertia, use_ev in configs:
        m, b = run(parts, inertia, use_ev, has_event and use_ev)
        cells = " | ".join(f"{m[h]:.3f}/{b[h]:.3f}" for h in HORIZONS)
        print(f"| {cname} | {cells} |")


if __name__ == "__main__":
    report("GESCHWIND-BRINGMANN (remitted depression, has event)", load_gesch(), True)
    report("OSF_83CFK (reliability sample, no event)", load_osf(), False)
