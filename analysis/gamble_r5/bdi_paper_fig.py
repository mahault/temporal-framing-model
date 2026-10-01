"""Publication figure for the depression analysis: BDI effects on the gamble-model parameters, first play
and second play, with 200-replicate participant bootstrap intervals. Effects are shown divided by their
bootstrap standard error, so that parameters on different scales share one axis.
Run: python bdi_paper_fig.py  ->  figures/gamble_bdi_effects.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
s = json.load(open(Path(__file__).parent / "out" / "gamble_r5_summary.json"))
plays = [("first play, N = 1,838", s["bdi_bootstrap"]["effects"], "C0", 0.12),
         ("second play, N = 929", s["bdi_play2_bootstrap"]["effects"], "C1", -0.12)]
params = [("a", "baseline mood"), ("wP", "present weight"), ("wF", "forward weight"),
          ("wB", "backward weight"), ("eta", "optimism coupling")]

fig, ax = plt.subplots(figsize=(6.4, 3.6))
y = np.arange(len(params))[::-1]
for lab, eff, col, off in plays:
    for yi, (k, _) in zip(y, params):
        e = eff[k]
        se = e["boot_se"]
        m, lo, hi = e["estimate"] / se, e["boot_lo"] / se, e["boot_hi"] / se
        ax.errorbar(m, yi + off, xerr=[[m - lo], [hi - m]], fmt="o", color=col, capsize=3,
                    label=lab if k == "a" else None)
ax.axvline(0, color="grey", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([n for _, n in params])
ax.set_xlabel("effect of one SD of BDI, in bootstrap SE units")
ax.legend(fontsize=8, loc="lower left")
plt.tight_layout()
out = ROOT / "figures" / "gamble_bdi_effects.png"
plt.savefig(out, dpi=200)
print("written", out)
