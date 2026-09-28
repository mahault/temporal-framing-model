"""Reported temporal orientation and momentary valence (Mulholland et al. 2023)
with the inference stated (round 2, 2026-09-27). Within-person correlations,
and the cross-level interaction of past-orientation with trait positivity
(person-mean valence) from a pooled within-person regression with
participant-clustered sandwich SEs. Writes reviews/mulholland_stats.md.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from diathesis_stats_v2 import cluster_ols, boot_r

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data_raw" / "mulholland2023" / "conscious_cogn_data_mendeley.csv"


def _f(x):
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def main():
    rows = list(csv.DictReader(open(DATA, encoding="latin-1")))
    parts = defaultdict(list)
    for r in rows:
        p, fu, e = _f(r["dimension_past"]), _f(r["dimension_future"]), _f(r["dimension_emotion"])
        if None in (p, fu, e):
            continue
        parts[r["secret_user_id"]].append((p, fu, e))
    X, Y, G = [], [], []
    per = []
    n_probe = 0
    for pid, seq in parts.items():
        if len(seq) < 3:
            continue
        A = np.array(seq, float)
        pm = A[:, 2].mean()
        pc = A[:, 0] - A[:, 0].mean()
        fc = A[:, 1] - A[:, 1].mean()
        vc = A[:, 2] - A[:, 2].mean()
        per.append((pm, np.polyfit(pc, vc, 1)[0] if pc.std() > 1e-9 else np.nan))
        for k in range(len(A)):
            X.append([pc[k], fc[k], pm])
            Y.append(vc[k])
            G.append(pid)
        n_probe += len(A)
    X = np.array(X)
    Y = np.array(Y)
    G = np.array(G)
    gm = np.mean([p for p, _ in per])
    r_past = np.corrcoef(X[:, 0], Y)[0, 1]
    r_fut = np.corrcoef(X[:, 1], Y)[0, 1]
    Xa = np.column_stack([X[:, 0], X[:, 0] * (X[:, 2] - gm), X[:, 1], np.ones(len(Y))])
    b, se, se_naive, ncl = cluster_ols(Xa, Y, G)
    per = np.array([p for p in per if not np.isnan(p[1])])
    r_sl = boot_r(per[:, 0], per[:, 1])
    L = ["# Reported orientation and valence, Mulholland et al. 2023 (2026-09-27)", "",
         "Script: `mulholland_stats.py`. Variables person-centred; trait positivity = person-mean valence.", "",
         f"- participants {ncl}, probes {n_probe}",
         f"- within-person r(past-orientation, valence) = {r_past:+.3f}; r(future-orientation, valence) = {r_fut:+.3f}",
         f"- pooled within-person regression valence_c ~ past_c + past_c x trait + future_c: past slope b = {b[0]:+.4f} "
         f"(cluster-robust SE {se[0]:.4f}, z = {b[0]/se[0]:+.2f}); past x trait interaction b = {b[1]:+.4f} "
         f"(SE {se[1]:.4f}, z = {b[1]/se[1]:+.2f}); future slope b = {b[2]:+.4f} (SE {se[2]:.4f}, z = {b[2]/se[2]:+.2f})",
         f"- between-person r(trait positivity, per-person past slope) = {r_sl[0]:+.3f} [{r_sl[1]:+.3f}, {r_sl[2]:+.3f}]", ""]
    (ROOT / "reviews" / "mulholland_stats.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
