"""Thought-content channel figure for the restructured paper (Figure 6).

Left: weight of thought pleasantness on current valence for past and future thought relative to present
thought, from the within-person estimator with participant-bootstrap 95% intervals (1,000 resamples), next
to the gamble-task ratios. Present is the reference and has no interval. Right: held-out log-likelihood of
three separate weights minus one pleasantness term, in the regression and in the state-space model.
Run from the repo root: python analysis/channels_content/content_fig.py -> figures/channels_content_weights.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
S = json.loads((HERE / "out/static.json").read_text())["split"]
H = json.loads((HERE / "out/heldout_static.json").read_text())["three_minus_one"]
D = json.loads((HERE / "out/dynamic.json").read_text())["three_minus_one"]
GAMBLE = dict(B=0.017 / 0.097, F=0.049 / 0.097)

fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.7))
names = ["backward\n(past thought)", "present\n(reference)", "forward\n(future thought)"]
rB, rF = S["ratio_B_P"], S["ratio_F_P"]
vals = [rB[0], 1.0, rF[0]]
cols = ["#0072B2", "#E69F00", "#D55E00"]
ax[0].bar(range(3), vals, color=cols, width=0.55, label="daily life")
ax[0].errorbar([0, 2], [rB[0], rF[0]], yerr=[[rB[0] - rB[1], rF[0] - rF[1]], [rB[2] - rB[0], rF[2] - rF[0]]],
               fmt="none", ecolor="k", capsize=3)
ax[0].scatter([0, 1, 2], [GAMBLE["B"], 1.0, GAMBLE["F"]], marker="D", color="k", zorder=3, label="gamble task")
ax[0].axhline(1.0, color="grey", lw=0.6, ls=":")
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(names, fontsize=8)
ax[0].set_ylabel("weight relative to present")
ax[0].legend(fontsize=8, frameon=False, loc="upper right")
ax[0].set_title("Channel weights", fontsize=10)
labs = ["regression", "state-space model"]
v = [H, D]
ax[1].errorbar(range(2), [x[0] for x in v], yerr=[[x[0] - x[1] for x in v], [x[2] - x[0] for x in v]],
               fmt="o", color="k", capsize=4)
ax[1].axhline(0, color="grey", lw=0.8)
ax[1].set_xticks(range(2)); ax[1].set_xticklabels(labs, fontsize=8); ax[1].set_xlim(-0.5, 1.5)
ax[1].set_ylabel("held-out log-likelihood gain\n(nats per beep)")
ax[1].set_title("Three weights minus one, held-out participants", fontsize=10)
plt.tight_layout()
out = ROOT / "figures" / "channels_content_weights.png"
plt.savefig(out, dpi=170)
print("written", out, "ratios", rB, rF)
