"""Figure for the fairer test of active selection (shared valence mean), 2026-09-30."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "orientation" / "out_frame"
ev = {ds: json.load(open(OUT / f"eval_{ds}.json")) for ds in ("baumeister1", "bayer")}
pool = json.load(open(OUT / "pooled.json"))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
ax = axes[0]
rows = [("Baumeister S1", ev["baumeister1"]["contrasts"]["M2_shared-M1b_shared"]["ll"]),
        ("Bayer", ev["bayer"]["contrasts"]["M2_shared-M1b_shared"]["ll"]),
        ("Bayer, dense labels", ev["bayer"]["contrasts_dense"]["M2_shared-M1b_shared"]["ll"]),
        ("pooled (participants)", pool["M2_shared-M1b_shared"]["pooled_participant"]),
        ("pooled (inverse variance)", pool["M2_shared-M1b_shared"]["inverse_variance"])]
y = np.arange(len(rows))[::-1]
for yi, (lab, (m, lo, hi)) in zip(y, rows):
    ax.errorbar(m, yi, xerr=[[m - lo], [hi - m]], fmt="o", color="k" if "pooled" in lab else "C0", capsize=3)
ax.axvline(0, color="grey", lw=0.8); ax.locator_params(axis="x", nbins=5)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel("agent minus chain with valence (nats per signal)")
ax.set_title("Next reported orientation", fontsize=10)

ax = axes[1]
names = [("flat\npreferences", "M2_shared_flatpref-M2_shared"),
         ("no epistemic\nterm", "M2_shared_noepist-M2_shared"),
         ("zero\nprecision", "M2_shared_noprecision-M2_shared")]
for j, ds in enumerate(("baumeister1", "bayer")):
    vals = [ev[ds]["contrasts"][k]["ll"] for _, k in names]
    x = np.arange(3) + j * 0.35
    ax.bar(x, [v[0] for v in vals], 0.3, yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
           capsize=3, label="Baumeister S1" if ds == "baumeister1" else "Bayer")
ax.axhline(0, color="grey", lw=0.8)
ax.set_xticks(np.arange(3) + 0.17); ax.set_xticklabels([n for n, _ in names])
ax.set_ylabel("variant minus full agent (nats per signal)")
ax.set_title("Removing a component of the agent", fontsize=10); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(ROOT / "figures" / "frame_orientation_fairtest.png", dpi=150)
print("saved")
