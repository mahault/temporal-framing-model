"""Forest plot of round-6 vulnerability interactions. Run from the repo root:
python analysis/diathesis_r6/figures_r6.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
r = json.load(open(ROOT / "analysis" / "diathesis_r6" / "out" / "results.json"))
# Gainey dropped (openESM 2.0.0 withdrew its person-level trait file); pooled values without Gainey
r["pooled"] = json.load(open(ROOT / "analysis" / "diathesis_r6" / "out" / "results_nogainey.json"))["pooled"]

rows = []  # (label, b, lo, hi, kind)


def add(title, sec, keys, term, pool_key):
    rows.append((title, None, None, None, "head"))
    for s in ("Geschwind", "Kane"):
        f = r[sec].get(keys.format(s=s))
        if f and term in f["fixed"]:
            x = f["fixed"][term]
            rows.append((f"  {s} (N={f['n_people']})", x["b"], x["lo"], x["hi"], "sample"))
    p = r["pooled"].get(pool_key)
    if p:
        rows.append(("  Pooled, random effects", p["DL"]["b"], p["DL"]["lo"], p["DL"]["hi"], "pool"))


add("Stress reactivity, negative affect (concurrent)", "reactivity", "{s}_NA", "STR_s_pc:neuroticism_z", "R1_NA")
add("Stress reactivity, positive affect (concurrent)", "reactivity", "{s}_PA", "STR_s_pc:neuroticism_z", "R1_PA")
add("Stress reactivity, negative affect (next signal)", "lagged", "{s}_NA", "STR_s_lagpc:neuroticism_z", "R2_NA")
add("Inertia of negative affect, variability controlled", "inertia", "{s}_NA_neuroticism",
    "NA_s_lagpc:neuroticism_z", "I_NA")
add("Inertia of negative affect, no controls", "inertia", "{s}_NA_neuroticism_nocontrols",
    "NA_s_lagpc:neuroticism_z", "I_NA_nocontrols")
add("Persistence of worry", "persistence", "{s}_neuroticism", "TH_s_lagpc:neuroticism_z", "P_thought")

fig, ax = plt.subplots(figsize=(7.2, 0.27 * len(rows) + 0.8))
y = len(rows)
ticks, labels = [], []
for lab, b, lo, hi, kind in rows:
    y -= 1
    ticks.append(y)
    labels.append(lab)
    if kind == "head":
        continue
    col = "black" if kind == "pool" else "0.35"
    ax.plot([lo, hi], [y, y], color=col, lw=1.4 if kind == "pool" else 1.0)
    ax.plot(b, y, "D" if kind == "pool" else "o", color=col, ms=5 if kind == "pool" else 4)
ax.axvline(0, color="0.6", lw=0.8, ls="--")
ax.set_yticks(ticks)
ax.set_yticklabels(labels, fontsize=7.5)
for t, (lab, *_rest) in zip(ax.get_yticklabels(), rows):
    if not lab.startswith("  "):
        t.set_fontweight("bold")
ax.set_xlabel("Change in within-person slope per SD of neuroticism\n(affect in SD units, 95% CI)", fontsize=8)
ax.tick_params(axis="x", labelsize=8)
fig.tight_layout()
out = ROOT / "figures" / "diathesis_r6_forest.png"
fig.savefig(out, dpi=200)
print(out)
