"""Publication figure for the orientation results (main text Figure 3).

Panel a: within-person contrasts of momentary valence with present-oriented thought, past and future,
in the three orientation samples, as a fraction of each sample's response range.
Panel b: Baumeister Study 1, valence by content of thought, with the present-oriented mean.
Panel c: Baumeister Study 1 and mCog, valence at the signal before, at and after each orientation.
Run: python orientation_paper_fig.py  (reads out/descriptives.json)  ->  figures/orientation_summary_paper.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
d = json.load(open(Path(__file__).parent / "out" / "descriptives.json"))
b1, mc, s2 = d["baumeister1"], d["bayer"], d["baumeister1"]["study2"]

fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.6))

# a. contrasts as fraction of response range
samples = [("Baumeister\nStudy 1", b1["within_person_contrast_vs_present"], 6.0),
           ("Baumeister\nStudy 2", s2["within_contrast_vs_present"], 100.0),
           ("mCog", mc["within_person_contrast_vs_present"], 4.0)]
x = np.arange(len(samples))
for k, (lab, col, off) in enumerate((("past", "C3", -0.12), ("future", "C0", 0.12))):
    m = [c[lab]["beta"] / r for _, c, r in samples]
    e = [1.96 * c[lab]["se"] / r for _, c, r in samples]
    ax[0].errorbar(x + off, m, yerr=e, fmt="o", color=col, capsize=3, label=f"{lab} minus present")
ax[0].axhline(0, color="grey", lw=0.8)
ax[0].set_xticks(x); ax[0].set_xticklabels([s[0] for s in samples])
ax[0].set_ylabel("valence contrast (fraction of scale)")
ax[0].set_title("a  Orientation and valence", loc="left", fontsize=10)
ax[0].legend(fontsize=8, loc="lower right")

# b. content of thought, Study 1
sub = b1["subitem_valence"]
order = [("regret", "regret"), ("replaying", "replaying"), ("worries", "worry"), ("fear", "fear"), ("planning", "planning")]
vals = [sub[k]["mean_valence"] for k, _ in order]
cols = ["C3", "C3", "C0", "C0", "C2"]
ax[1].bar(range(len(order)), vals, color=cols)
for i, (k, _) in enumerate(order):
    ax[1].text(i, vals[i] + 0.03, f"n={sub[k]['n']}", ha="center", fontsize=7)
pres = b1["valence_mean_by_orientation"]["present"]["mean"]
ax[1].axhline(pres, color="k", ls="--", lw=0.8)
ax[1].text(len(order) - 0.5, pres + 0.03, "present-oriented mean", ha="right", fontsize=7)
ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels([n for _, n in order])
ax[1].set_ylabel("mean valence (-3 to 3)")
ax[1].set_title("b  Content of thought, Study 1", loc="left", fontsize=10)

# c. before, at, after, Study 1 and mCog, centred on each sample's present-oriented mean
for ds, data, rng, mk in (("Study 1", b1, 6.0, "o"), ("mCog", mc, 4.0, "s")):
    pm = data["valence_mean_by_orientation"]["present"]["mean"]
    for lab, col in (("past", "C3"), ("present", "grey"), ("future", "C0")):
        bat = data["valence_before_at_after"][lab]
        y = [(bat[k] - pm) / rng for k in ("before", "at", "after")]
        ax[2].plot([0, 1, 2], y, marker=mk, color=col, ls="-" if ds == "Study 1" else ":",
                   label=f"{lab}, {ds}")
ax[2].axhline(0, color="grey", lw=0.6)
ax[2].set_xticks([0, 1, 2]); ax[2].set_xticklabels(["signal before", "at signal", "signal after"])
ax[2].set_ylabel("valence minus present mean (fraction of scale)")
ax[2].set_title("c  Valence around each orientation", loc="left", fontsize=10)
ax[2].legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))

plt.tight_layout()
out = ROOT / "figures" / "orientation_summary_paper.png"
plt.savefig(out, dpi=200)
print("written", out)
