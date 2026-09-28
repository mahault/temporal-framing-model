"""Diathesis-stress test on the Geschwind ESM with the inference stated (round 2,
2026-09-27). Same quantities as diathesis_stress_test.py, now with:
  * the cross-level event x neuroticism interaction from a pooled within-person
    regression (valence and event person-centred) with participant-clustered
    sandwich standard errors;
  * a two-stage estimate: per-participant event->valence slope regressed on
    neuroticism (OLS, HC SE);
  * participant-bootstrap 95 percent CIs for the between-person correlations.
Uses the corrected loader (period-aware; irrelevant here since only concurrent
pairs are used) and the same valence composite as the model input. Writes
reviews/diathesis_stats_v2.md.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import empirical_rebuild as er

ROOT = Path(__file__).resolve().parent
N_BOOT = 2000


def cluster_ols(X, y, groups):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    resid = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    G = 0
    for g in np.unique(groups):
        m = groups == g
        u = X[m].T @ resid[m]
        meat += np.outer(u, u)
        G += 1
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1))
    se = np.sqrt(np.diag(V))
    s2 = resid @ resid / (len(y) - X.shape[1])
    se_naive = np.sqrt(np.diag(s2 * XtX_inv))
    return b, se, se_naive, G


def boot_r(x, y, seed=0):
    rng = np.random.RandomState(seed)
    n = len(x)
    r = np.corrcoef(x, y)[0, 1]
    b = []
    for _ in range(N_BOOT):
        i = rng.randint(0, n, n)
        b.append(np.corrcoef(x[i], y[i])[0, 1])
    return float(r), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    parts = er.load_participants()
    per = []
    X, Y, Gr = [], [], []
    n_beeps = 0
    for pid, seq in parts.items():
        neur = seq[0].get("n")
        rows = [(b["e"], b["v"]) for b in seq if b["e"] is not None]
        if neur is None or len(rows) < 10:
            continue
        A = np.array(rows, float)
        x = A[:, 0] - A[:, 0].mean()
        y = A[:, 1] - A[:, 1].mean()
        if x.std() < 1e-6:
            continue
        slope = np.polyfit(x, y, 1)[0]
        per.append((neur, slope, A[:, 1].mean()))
        for k in range(len(A)):
            X.append([x[k], neur])
            Y.append(y[k])
            Gr.append(pid)
        n_beeps += len(A)
    per = np.array(per)
    neur, slope, mv = per[:, 0], per[:, 1], per[:, 2]
    gm = neur.mean()
    Xa = np.array([[xc, xc * (nn - gm), 1.0] for xc, nn in X])
    Ya = np.array(Y)
    Gr = np.array(Gr)
    b, se, se_naive, G = cluster_ols(Xa, Ya, Gr)
    # two-stage
    Z = np.column_stack([np.ones(len(neur)), neur - gm])
    bb, *_ = np.linalg.lstsq(Z, slope, rcond=None)
    res = slope - Z @ bb
    ZtZ = np.linalg.inv(Z.T @ Z)
    meat = (Z * res[:, None]).T @ (Z * res[:, None])
    Vhc = ZtZ @ meat @ ZtZ * len(neur) / (len(neur) - 2)
    r_mv = boot_r(neur, mv)
    r_sl = boot_r(neur, slope, 1)
    L = ["# Diathesis-stress test with stated inference (2026-09-27)", "",
         "Script: `diathesis_stats_v2.py`. Geschwind ESM, valence = mean(cheerful, relaxed) minus "
         "mean(worried, fearful, sad) on [0,1], event = pleasantness of the most recent event, "
         "neuroticism = baseline trait score (person constant, never used in model fitting).", "",
         f"- participants with neuroticism and >= 10 event-valence pairs: {len(per)}; beeps: {n_beeps}",
         f"- corr(neuroticism, mean valence) = {r_mv[0]:+.3f} [{r_mv[1]:+.3f}, {r_mv[2]:+.3f}] (participant bootstrap)",
         f"- corr(neuroticism, event->valence slope) = {r_sl[0]:+.3f} [{r_sl[1]:+.3f}, {r_sl[2]:+.3f}]",
         f"- two-stage: slope on neuroticism b = {bb[1]:+.4f}, HC1 SE {math.sqrt(Vhc[1, 1]):.4f}, "
         f"t = {bb[1] / math.sqrt(Vhc[1, 1]):+.2f} (n = {len(neur)})",
         f"- pooled within-person regression valence_c ~ event_c + event_c x neuroticism_c: "
         f"interaction b = {b[1]:+.4f}, cluster-robust SE {se[1]:.4f} (z = {b[1] / se[1]:+.2f}, "
         f"{G} clusters); naive SE {se_naive[1]:.4f} (t = {b[1] / se_naive[1]:+.2f})",
         f"- event main effect b = {b[0]:+.4f}, cluster-robust SE {se[0]:.4f}", ""]
    (ROOT / "reviews" / "diathesis_stats_v2.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
