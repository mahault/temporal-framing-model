"""Benjamini-Hochberg over every trait-interaction test in the round-6 vulnerability analysis.

Family: every per-sample trait x slope interaction (reactivity, lagged reactivity, inertia with and
without variability controls, persistence), every DerSimonian-Laird pooled interaction, and every
mega-analysis interaction. Hartung-Knapp intervals are a second interval for the same pooled
estimate and are not counted as separate tests. Writes out/bh_r6.json.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "out", "results.json")))
TRAITS = ("neuroticism", "dysphoria", "brooding")
tests = []
for block in ("reactivity", "lagged", "inertia", "persistence"):
    for key, m in d.get(block, {}).items():
        for name, fe in m.get("fixed", {}).items():
            if ":" in name and any(t in name for t in TRAITS) and "pm_" not in name and "psd_" not in name:
                parts = name.split(":")
                # keep only the focal slope x trait term (exclude person mean or SD x trait, if any)
                tests.append({"family_part": block, "model": key, "term": name,
                              "b": fe["b"], "lo": fe["lo"], "hi": fe["hi"], "p": fe["p"]})
for key, m in d.get("pooled", {}).items():
    if "DL" in m:
        tests.append({"family_part": "pooled_DL", "model": key, "term": "pooled", "b": m["DL"]["b"],
                      "lo": m["DL"]["lo"], "hi": m["DL"]["hi"], "p": m["DL"]["p"]})
for key, m in d.get("mega", {}).items():
    fe = m.get("fixed", {}) if isinstance(m, dict) else {}
    for name, v in fe.items():
        if ":" in name and any(t in name for t in TRAITS):
            tests.append({"family_part": "mega", "model": key, "term": name, "b": v["b"],
                          "lo": v["lo"], "hi": v["hi"], "p": v["p"]})
    if isinstance(m, dict) and "b" in m and "p" in m:
        tests.append({"family_part": "mega", "model": key, "term": "mega", "b": m["b"],
                      "lo": m.get("lo"), "hi": m.get("hi"), "p": m["p"]})
p = np.array([t["p"] for t in tests])
n = len(p)
order = np.argsort(p)
adj = np.empty(n)
prev = 1.0
for rank in range(n, 0, -1):
    i = order[rank - 1]
    prev = min(prev, p[i] * n / rank)
    adj[i] = prev
for t, a in zip(tests, adj):
    t["p_adj"] = float(a)
    t["survive"] = bool(a < 0.05)
out = {"q": 0.05, "n_tests": n, "nominal_below_05": int((p < 0.05).sum()),
       "survive": int((adj < 0.05).sum()), "tests": tests}
json.dump(out, open(os.path.join(HERE, "out", "bh_r6.json"), "w"), indent=1)
print("n_tests", n, "nominal", out["nominal_below_05"], "survive", out["survive"])
for t in sorted(tests, key=lambda t: t["p"]):
    print(f"{t['family_part']:12s} {t['model']:35s} {t['term']:35s} b={t['b']:+.3f} p={t['p']:.4f} adj={t['p_adj']:.4f} {'*' if t['survive'] else ''}")
