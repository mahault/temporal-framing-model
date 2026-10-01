"""Hartung-Knapp-Sidik-Jonkman and fixed-effect intervals for the pooled vulnerability associations
(k = 2 or 3 samples), from the per-sample estimates and 95% intervals in reviews/DIATHESIS_DATA_TEST.md
(standard errors recovered as interval width / 3.92). Run: python hksj.py > out/hksj.txt
"""
import numpy as np
from scipy import stats

rows = {
    "(a) level, model baseline": [(-0.12, -0.27, 0.05), (-0.15, -0.27, -0.03), (-0.12, -0.23, -0.01)],
    "(a) level, model mean mood": [(-0.53, -0.67, -0.40), (-0.47, -0.58, -0.36), (-0.47, -0.58, -0.37)],
    "(a) level, person mean": [(-0.53, -0.66, -0.40), (-0.48, -0.59, -0.37), (-0.47, -0.58, -0.36)],
    "(b) inertia, model timescale": [(0.20, 0.03, 0.36), (0.05, -0.11, 0.20), (0.11, -0.02, 0.23)],
    "(b) inertia, AR(1)": [(0.13, -0.04, 0.30), (-0.14, -0.27, -0.02), (0.12, 0.00, 0.24)],
    "(c) reactivity, model": [(-0.21, -0.37, -0.06), (-0.11, -0.20, -0.03)],
    "(c) reactivity, event slope": [(0.14, -0.01, 0.33), (0.08, -0.04, 0.19)],
    "(d) thought persistence": [(0.14, -0.02, 0.30), (-0.08, -0.19, 0.03), (0.14, 0.04, 0.25)],
    "(d) thought level": [(0.46, 0.31, 0.59), (0.24, 0.13, 0.35), (0.27, 0.14, 0.40)],
}
for k, v in rows.items():
    y = np.array([a for a, _, _ in v]); se = np.array([(h - l) / 3.92 for _, l, h in v]); w = 1 / se ** 2
    yf = (w * y).sum() / w.sum(); Q = (w * (y - yf) ** 2).sum(); n = len(y)
    tau2 = max(0.0, (Q - (n - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (se ** 2 + tau2); mu = (ws * y).sum() / ws.sum()
    var_hk = (ws * (y - mu) ** 2).sum() / ((n - 1) * ws.sum()); t = stats.t.ppf(0.975, n - 1)
    print(f"{k:30s} DL {mu:+.2f} [{mu - 1.96 / np.sqrt(ws.sum()):+.2f}, {mu + 1.96 / np.sqrt(ws.sum()):+.2f}]"
          f"  HKSJ [{mu - t * np.sqrt(var_hk):+.2f}, {mu + t * np.sqrt(var_hk):+.2f}]"
          f"  FE {yf:+.2f} [{yf - 1.96 / np.sqrt(w.sum()):+.2f}, {yf + 1.96 / np.sqrt(w.sum()):+.2f}]")
