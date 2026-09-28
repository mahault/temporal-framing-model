"""Figure for model v2.1: held-out R2 by horizon, v2 (gated, inert), key ablations, the six-lag ridge
and the round-3 discrete model, both samples, pooled protocol. Reads reviews/model_v2_forecast.json,
writes figures/fig_forecast_v2.png."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
d = json.load(open(ROOT / "reviews" / "model_v2_forecast.json", encoding="utf-8"))
H = [1, 2, 3, 4, 5, 6]
series = [("kitchen_tt", "ridge: 6 lags + events + time of day", "k-"),
          ("ar2", "two-lag regression", "k:"),
          ("v1 model_full", "discrete model (round 3)", "C7-"),
          ("P:v2_gated", "v2 gated", "C3-"),
          ("P:v2_inert", "v2 inert (g = 0)", "C1--"),
          ("P:v2_no_level2", "v2 without slow mood", "C2-"),
          ("P:v2_clampPRESENT", "v2 frame clamped PRESENT", "C4-.")]


def val(x):
    if isinstance(x, (list, tuple)):
        return x[0]
    if isinstance(x, dict):
        return x.get("r2", next(iter(x.values())))
    return x


fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
for ax, sample, title in zip(axes, ["Geschwind", "osf_83cfk"], ["Remitted-depression ESM", "Reliability ESM"]):
    tab = d[sample]["tables"]
    for nm, lab, st in series:
        ax.plot(H, [val(tab[nm][str(h)] if str(h) in tab[nm] else tab[nm][h]) for h in H], st, label=lab, lw=1.6)
    ax.set_title(f"{title}, pooled protocol", fontsize=10)
    ax.set_xlabel("horizon (beeps ahead)")
    ax.set_ylabel("held-out R$^2$")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig_forecast_v2.png", dpi=180)
print("written figures/fig_forecast_v2.png")
