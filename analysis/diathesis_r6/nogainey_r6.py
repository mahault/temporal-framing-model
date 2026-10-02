"""Vulnerability round 6 without the Gainey sample.

The Gainey person-level trait scores came from openESM release 1.0.0, which release 2.0.0
(17 September 2026) withdrew pending privacy review. The authors decided to drop every Gainey
trait result. This script recomputes, without any Gainey data:

  * the DerSimonian-Laird and Hartung-Knapp pooled interactions from the saved per-sample
    terms of the Geschwind and Kane models (out/results.json, no refit needed),
  * the inertia mega-analysis on Geschwind and Kane only (refit; the reactivity mega-analysis
    never included Gainey and is copied),
  * Benjamini-Hochberg over every remaining test in the family.

Run from the repo root:  python analysis/diathesis_r6/nogainey_r6.py
Writes analysis/diathesis_r6/out/results_nogainey.json and out/bh_r6_nogainey.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diathesis_r6 as D  # noqa: E402

SAMPLES = ("Geschwind", "Kane")
TRAITS = ("neuroticism", "dysphoria", "brooding")


def term(fit, name):
    if not fit or name not in fit.get("fixed", {}):
        return None
    x = fit["fixed"][name]
    return (x["b"], x["se"])


def main():
    res = json.load(open(HERE / "out" / "results.json"))
    out = {"note": "Gainey removed; per-sample Geschwind and Kane terms as in results.json"}

    sets = {
        "R1_NA": ("reactivity", "{s}_NA", "STR_s_pc:neuroticism_z"),
        "R1_PA": ("reactivity", "{s}_PA", "STR_s_pc:neuroticism_z"),
        "R2_NA": ("lagged", "{s}_NA", "STR_s_lagpc:neuroticism_z"),
        "I_NA": ("inertia", "{s}_NA_neuroticism", "NA_s_lagpc:neuroticism_z"),
        "I_PA": ("inertia", "{s}_PA_neuroticism", "PA_s_lagpc:neuroticism_z"),
        "I_NA_nocontrols": ("inertia", "{s}_NA_neuroticism_nocontrols", "NA_s_lagpc:neuroticism_z"),
        "P_thought": ("persistence", "{s}_neuroticism", "TH_s_lagpc:neuroticism_z"),
    }
    pooled = {}
    for key, (block, pat, name) in sets.items():
        est = [term(res[block].get(pat.format(s=s)), name) for s in SAMPLES]
        est = [e for e in est if e is not None]
        if len(est) >= 2:
            pooled[key] = D.pool(est)
    out["pooled"] = pooled

    # mega-analysis of NA inertia on the two remaining samples
    data = {}
    for s in SAMPLES:
        raw, traits = D.LOADERS[s]()
        data[s] = D.prepare(raw, traits)
    cols_i = ["NA_s", "NA_s_lagpc", "neuroticism_z", "NA_s_pmz", "NA_s_psdz"]
    di = pd.concat([data[s].assign(sample=s)[cols_i + ["pid", "sample"]] for s in SAMPLES])
    di["pid"] = di["sample"] + "_" + di["pid"]
    f = D.fit_mixed(di, "NA_s ~ C(sample) + C(sample):NA_s_lagpc + NA_s_lagpc:(neuroticism_z + NA_s_pmz + NA_s_psdz)"
                        " + neuroticism_z + NA_s_pmz + NA_s_psdz",
                    "NA_s_lagpc", cols_i, "MEGA I NA (no Gainey)")
    out["mega"] = {"R1_NA": res["mega"]["R1_NA"], "I_NA": D.strip(f)}
    json.dump(out, open(HERE / "out" / "results_nogainey.json", "w"), indent=1)

    # Benjamini-Hochberg over every remaining test
    tests = []
    for block in ("reactivity", "lagged", "inertia", "persistence"):
        for key, m in res.get(block, {}).items():
            if key.startswith("Gainey"):
                continue
            for name, fe in (m or {}).get("fixed", {}).items():
                if ":" in name and any(t in name for t in TRAITS) and "pm_" not in name and "psd_" not in name:
                    tests.append(dict(family_part=block, model=key, term=name, b=fe["b"],
                                      lo=fe["lo"], hi=fe["hi"], p=fe["p"]))
    for key, m in pooled.items():
        tests.append(dict(family_part="pooled_DL", model=key, term="pooled", **m["DL"]))
    for key, m in out["mega"].items():
        for name, v in (m or {}).get("fixed", {}).items():
            if ":" in name and any(t in name for t in TRAITS):
                tests.append(dict(family_part="mega", model=key, term=name, b=v["b"], lo=v["lo"],
                                  hi=v["hi"], p=v["p"]))
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
    bh = {"q": 0.05, "n_tests": n, "nominal_below_05": int((p < 0.05).sum()),
          "survive": int((adj < 0.05).sum()), "tests": tests}
    json.dump(bh, open(HERE / "out" / "bh_r6_nogainey.json", "w"), indent=1)
    print("n_tests", n, "nominal", bh["nominal_below_05"], "survive", bh["survive"])
    for t in sorted(tests, key=lambda t: t["p"]):
        print(f"{t['family_part']:11s} {t['model']:38s} {t['term'][:28]:28s} b={t['b']:+.4f} "
              f"[{t['lo']:+.4f}, {t['hi']:+.4f}] p={t['p']:.4f} adj={t['p_adj']:.4f} {'*' if t['survive'] else ''}")
    for k, m in pooled.items():
        hk = m.get("HK", {})
        print(f"POOLED {k}: DL {m['DL']['b']:+.4f} [{m['DL']['lo']:+.4f}, {m['DL']['hi']:+.4f}] p={m['DL']['p']:.4f}"
              f"  HK [{hk.get('lo', float('nan')):+.4f}, {hk.get('hi', float('nan')):+.4f}]  tau2={m['tau2']:.5f}")


if __name__ == "__main__":
    main()
