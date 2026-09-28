"""Figure for round 3: held-out R2 by horizon, strongest baseline vs model-based predictors,
both samples, both protocols. Reads reviews/esm_eval_v3.json, writes figures/fig_esm_v3.png."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
d = json.load(open(ROOT / "reviews" / "esm_eval_v3.json", encoding="utf-8"))
H = d["horizons"]
panels = [("Geschwind", "P/level/level", ["kitchen", "ar2", "msar2", "model_full", "model_inert", "aug"]),
          ("osf_83cfk", "P/level/level", ["kitchen", "ar2", "msar2", "model_full", "model_inert", "aug"]),
          ("Geschwind", "A/level/level", ["A_kitchen", "A_kitchen_pooled", "A_model", "A_model_inert", "A_aug"]),
          ("osf_83cfk", "A/level/level", ["A_kitchen", "A_kitchen_pooled", "A_model", "A_model_inert", "A_aug"])]
labels = {"kitchen": "ridge: 6 lags + events + time of day", "ar2": "two-lag regression", "msar2": "two-regime switching AR",
          "model_full": "frame model (gated)", "model_inert": "frame model (inert)", "aug": "ridge + model state features",
          "A_kitchen": "adapted ridge", "A_kitchen_pooled": "pooled ridge", "A_model": "adapted frame model",
          "A_model_inert": "adapted frame model (inert)", "A_aug": "adapted ridge + model features"}
styles = {"kitchen": "k-", "A_kitchen": "k-", "A_kitchen_pooled": "k--", "ar2": "k:", "msar2": "k-.",
          "model_full": "C3-", "A_model": "C3-", "model_inert": "C1-", "A_model_inert": "C1-", "aug": "C0-", "A_aug": "C0-"}
fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
for ax, (sample, tag, names) in zip(axes.ravel(), panels):
    tab = d[sample]["tables"][tag]
    for nm in names:
        ys = [tab[nm][str(h)][0] for h in H]
        ax.plot(H, ys, styles[nm], label=labels[nm], lw=1.6)
    ax.set_title(f"{'Remitted-depression ESM' if sample == 'Geschwind' else 'Reliability ESM'}, "
                 f"{'pooled' if tag.startswith('P') else 'adapted'} protocol", fontsize=10)
    ax.set_ylabel("held-out R$^2$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
for ax in axes[1]:
    ax.set_xlabel("horizon (beeps ahead)")
fig.tight_layout()
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures" / "fig_esm_v3.png", dpi=180)
print("written figures/fig_esm_v3.png")
