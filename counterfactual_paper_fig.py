"""Publication figure for counterfactual switching (both complete-feedback datasets).
Reads reviews/regret_participant_holdout.json (written by regret_participant_holdout.py).
Run: python counterfactual_paper_fig.py  ->  figures/fig_counterfactual_signature.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
res = json.load(open(ROOT / "reviews" / "regret_participant_holdout.json"))
MODELS = [("factual", "factual\nlearner"), ("cfvalue", "counterfactual\nvalue learner"),
          ("regret", "regret\nbias"), ("regret_frame", "regret bias,\nframe-gated")]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
for ax, o in zip(axes, res):
    vals = [o["human"]] + [o["models"][m]["generated"] for m, _ in MODELS]
    x = np.arange(len(vals))
    m = [v[0] for v in vals]
    ax.bar(x, m, yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
           color=["#333333"] + ["#0072B2"] * len(MODELS), capsize=3, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(["people"] + [n for _, n in MODELS], fontsize=8)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title(f"{o['name']} (n = {o['n']})", fontsize=9.5, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("switching after a loss,\nforegone better minus foregone same")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig_counterfactual_signature.png", dpi=200, bbox_inches="tight")
print("written")
