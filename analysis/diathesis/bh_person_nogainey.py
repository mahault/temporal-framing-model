"""Person-level (two-stage) vulnerability associations without the Gainey sample.

The Gainey person-level trait scores came from openESM release 1.0.0, withdrawn in release 2.0.0
(17 September 2026) pending privacy review, and the authors dropped every Gainey trait result.

From the saved per-sample relations in analysis/diathesis/out/summary.json this script
  * pools the Geschwind and Kane neuroticism associations by DerSimonian-Laird, with
    Hartung-Knapp-Sidik-Jonkman intervals,
  * runs Benjamini-Hochberg at q = 0.05 over the family of every per-sample association and every
    pooled association (Geschwind and Kane only).

Run from the repo root:  python analysis/diathesis/bh_person_nogainey.py
Writes analysis/diathesis/out/person_nogainey.json.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

R = Path(__file__).resolve().parents[2]
S = json.load(open(R / "analysis" / "diathesis" / "out" / "summary.json"))
SAMPLES = ("Geschwind", "Kane")


def pool(est):
    b = np.array([e[0] for e in est]); se = np.array([e[1] for e in est]); k = len(b)
    w = 1 / se ** 2; bf = (w * b).sum() / w.sum(); q = (w * (b - bf) ** 2).sum()
    c = w.sum() - (w ** 2).sum() / w.sum(); tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 else 0.0
    ws = 1 / (se ** 2 + tau2); bd = (ws * b).sum() / ws.sum(); sd = math.sqrt(1 / ws.sum())
    out = dict(b=float(bd), lo=float(bd - 1.96 * sd), hi=float(bd + 1.96 * sd),
               p=float(2 * stats.norm.sf(abs(bd / sd))), tau2=float(tau2), k=k)
    hv = float((ws * (b - bd) ** 2).sum() / ((k - 1) * ws.sum()))
    sh = math.sqrt(max(hv, 1e-300)); tq = stats.t.ppf(0.975, k - 1)
    out["hk_lo"], out["hk_hi"] = float(bd - tq * sh), float(bd + tq * sh)
    return out


tests, pooled = [], {}
keys = sorted(set().union(*[S[s]["relations"]["neuroticism"].keys() for s in SAMPLES]))
for key in keys:
    est = []
    for s in SAMPLES:
        r = S[s]["relations"]["neuroticism"].get(key)
        if not r:
            continue
        z = r["beta"] / r["se"]
        tests.append(dict(kind="sample", sample=s, key=key, b=r["beta"], se=r["se"],
                          p=float(2 * stats.norm.sf(abs(z)))))
        est.append((r["beta"], r["se"]))
    if len(est) == 2:
        pooled[key] = pool(est)
        tests.append(dict(kind="pooled", sample="pooled", key=key, b=pooled[key]["b"], p=pooled[key]["p"]))

p = np.array([t["p"] for t in tests]); n = len(p)
order = np.argsort(p); adj = np.empty(n); prev = 1.0
for rank in range(n, 0, -1):
    i = order[rank - 1]; prev = min(prev, p[i] * n / rank); adj[i] = prev
for t, a in zip(tests, adj):
    t["p_adj"] = float(a); t["survive"] = bool(a < 0.05)
out = dict(n_tests=n, survive=int((adj < 0.05).sum()), pooled=pooled, tests=tests)
json.dump(out, open(R / "analysis" / "diathesis" / "out" / "person_nogainey.json", "w"), indent=1)
print("family size", n, "survive", out["survive"])
for k, m in pooled.items():
    print(f"POOLED {k:20s} {m['b']:+.3f} [{m['lo']:+.3f}, {m['hi']:+.3f}] p={m['p']:.4f} HK [{m['hk_lo']:+.3f}, {m['hk_hi']:+.3f}]")
for t in sorted(tests, key=lambda t: t["p"]):
    print(f"{t['kind']:6s} {t['sample']:9s} {t['key']:20s} b={t['b']:+.3f} p={t['p']:.4f} adj={t['p_adj']:.4f} {'*' if t['survive'] else ''}")
