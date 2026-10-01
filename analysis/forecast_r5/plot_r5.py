"""Figure for round-5 forecasting: held-out R2 by horizon (top) and difference from the best baseline
with participant-bootstrap 95% intervals (bottom), both ESM samples.
Run:  python plot_r5.py  (after score.py)  ->  figures/fig_forecast_r5.png
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import HORIZONS, OUT, ROOT, SAMPLES  # noqa: E402

LABEL = {"ridge_tt": "Ridge (six lags, events, time of day)", "ridge_pm_all": "Ridge plus running person mean",
         "kf_all_h1": "Local level plus AR(1) filter", "pmean_all": "Running person mean", "ebar1": "Per-person AR(1), shrunk",
         "v21_gated": "Model as first specified", "r5_sp_h6_vol": "Model with set-point and precision state",
         "kfx_all_h6": "Two-timescale filter, precision state, channels"}
SAMPLE_LABEL = {"Geschwind": "Geschwind et al.", "osf_83cfk": "Reliability sample"}


def main():
    rep = json.loads((OUT / "score.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for j, s in enumerate(SAMPLES):
        r = rep[s]["r2"]
        for n in LABEL:
            if n in r:
                ls = "-" if n.startswith(("r5", "v21", "kfx")) else "--"
                ax[0, j].plot(HORIZONS, [r[n][str(h)] if str(h) in r[n] else r[n][h] for h in HORIZONS], ls, marker="o", ms=3, label=LABEL[n])
        ax[0, j].set_title(SAMPLE_LABEL[s])
        ax[0, j].set_ylabel("held-out $R^2$")
        best = rep[s]["best_baseline"]
        for j2, n in enumerate(("v21_gated", "r5_sp_h6_vol", "kfx_all_h6")):
            pts = [c for c in rep[s]["cis"] if c["a"] == n and c["b"] == best[str(c["h"])]]
            pts.sort(key=lambda c: c["h"])
            if pts:
                hs = [c["h"] + 0.1 * (j2 - 1) for c in pts]
                ax[1, j].errorbar(hs, [c["point"] for c in pts], yerr=[[c["point"] - c["lo"] for c in pts], [c["hi"] - c["point"] for c in pts]],
                                  fmt="o", capsize=2, label=LABEL[n])
        ax[1, j].axhline(0, color="grey", lw=0.8)
        ax[1, j].set_xlabel("horizon (beeps ahead)")
        ax[1, j].set_ylabel("$R^2$ minus strongest competitor")
    ax[0, 0].legend(fontsize=7, loc="upper right")
    ax[1, 0].legend(fontsize=7)
    fig.tight_layout()
    f = ROOT / "figures" / "fig_forecast_r5.png"
    fig.savefig(f, dpi=200)
    print("written", f)


if __name__ == "__main__":
    main()
