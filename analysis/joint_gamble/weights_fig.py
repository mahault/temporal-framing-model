"""Channel-weight figure for the restructured paper (gamble task).

Left: group weights of the backward, present and forward channels (mean over the five training folds,
error bars = SD across folds), with the equal-weight fit dashed. Right: distribution of the shrunk
per-person weights (all held-out participants). Units: happiness on its 0 to 100 scale per task point.
Run from the repo root: python analysis/joint_gamble/weights_fig.py -> figures/joint_gamble_weights.png
Inputs: out/summary.json, out/person_weights_fold*.npy (person weights ordered forward, present, backward).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
S = json.loads((HERE / "out/summary.json").read_text())
W = np.concatenate([np.load(HERE / f"out/person_weights_fold{f}.npy") for f in range(5)])
order = [2, 1, 0]                      # backward, present, forward
names = ["backward", "present", "forward"]
cols = ["#0072B2", "#E69F00", "#D55E00"]
g = S["params"]["H_RM"]["w"]
gm = [g["mean"][i] for i in order]
gs = [g["sd"][i] for i in order]
weq = S["params"]["H_RM_eq"]["w_eq"]["mean"][0]

fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.4))
axs[0].bar(range(3), gm, yerr=gs, color=cols, capsize=4)
axs[0].axhline(weq, ls="--", color="k", lw=0.8, label="all three weights forced equal")
for i, v in enumerate(gm):
    axs[0].text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
axs[0].set_xticks(range(3)); axs[0].set_xticklabels(names)
axs[0].set_ylabel("happiness (0 to 100) per task point")
axs[0].legend(fontsize=7, frameon=False, loc="upper left"); axs[0].set_ylim(0, 0.115)
axs[0].set_title("Group weights", fontsize=9.5)
parts = axs[1].violinplot([W[:, i] for i in order], showmedians=True, showextrema=False)
for b, c in zip(parts["bodies"], cols):
    b.set_facecolor(c); b.set_alpha(0.55)
axs[1].axhline(0, color="grey", lw=0.8)
axs[1].set_xticks([1, 2, 3]); axs[1].set_xticklabels(names)
axs[1].set_ylabel("happiness (0 to 100) per task point")
axs[1].set_title(f"Shrunk per-person weights (N = {len(W):,})", fontsize=9.5)
fig.tight_layout()
out = ROOT / "figures" / "joint_gamble_weights.png"
fig.savefig(out, dpi=180)
print("written", out)
