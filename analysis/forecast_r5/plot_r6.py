"""Forecasting figure for the paper (round 6): competitors and the reported forecaster only.

Top: held-out R2 by horizon. Middle: the reported forecaster minus three competitors in R2.
Bottom: held-out log-likelihood of the reported forecaster minus the local-level filter with constant
noise and minus the ridge with person statistics. Participant-bootstrap 95% intervals.
Run after score.py and contrasts_r6.py:  python plot_r6.py  ->  figures/fig_forecast_r6.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import HORIZONS, OUT, ROOT, SAMPLES  # noqa: E402

MODEL = "kfx_all_h6"
LABEL = {MODEL: "Local-level filter with precision state and channels (reported)",
         "ridge_tt": "Ridge, six lags, events, time of day", "pmean_all": "Running person mean",
         "ridge_pm_all": "Ridge with person mean and SD", "ebar1": "Per-person AR(1), shrunk",
         "kf_all_h1": "Local-level filter, constant noise"}
SAMPLE_LABEL = {"Geschwind": "Geschwind sample", "osf_83cfk": "Reliability sample"}
COL = {"ridge_tt": "#0072B2", "pmean_all": "#D55E00", "ridge_pm_all": "#E69F00", "ebar1": "#CC79A7",
       "kf_all_h1": "#009E73", MODEL: "#000000"}


def main():
    rep = json.loads((OUT / "contrasts_r6.json").read_text())
    fig, ax = plt.subplots(3, 2, figsize=(10, 9), sharex=True)
    for j, s in enumerate(SAMPLES):
        r2 = rep[s]["r2"]
        for n in LABEL:
            ys = [r2[n][str(h)] if str(h) in r2[n] else r2[n][h] for h in HORIZONS]
            ax[0, j].plot(HORIZONS, ys, "-" if n == MODEL else "--", marker="o", ms=3, color=COL[n],
                          lw=3.2 if n == MODEL else 1.2, alpha=0.45 if n == MODEL else 1.0, label=LABEL[n])
        ax[0, j].set_title(SAMPLE_LABEL[s])
        ax[0, j].set_ylabel("held-out $R^2$")
        for k, b in enumerate(("ridge_pm_all", "ebar1", "kf_all_h1")):
            pts = sorted([c for c in rep[s]["r2_diff"] if c["b"] == b], key=lambda c: c["h"])
            hs = [c["h"] + 0.12 * (k - 1) for c in pts]
            ax[1, j].errorbar(hs, [c["point"] for c in pts], yerr=[[c["point"] - c["lo"] for c in pts],
                              [c["hi"] - c["point"] for c in pts]], fmt="o", ms=4, capsize=2, color=COL[b],
                              label="minus " + LABEL[b][0].lower() + LABEL[b][1:])
        ax[1, j].axhline(0, color="grey", lw=0.8)
        ax[1, j].set_ylabel("difference in $R^2$")
        for k, b in enumerate(("kf_all_h1/own", "ridge_pm_all/lfo")):
            pts = sorted([c for c in rep[s]["ll_diff"] if c["b"] == b], key=lambda c: c["h"])
            base = b.split("/")[0]
            hs = [c["h"] + 0.12 * (k - 0.5) for c in pts]
            ax[2, j].errorbar(hs, [c["point"] for c in pts], yerr=[[c["point"] - c["lo"] for c in pts],
                              [c["hi"] - c["point"] for c in pts]], fmt="o", ms=4, capsize=2, color=COL[base],
                              label="minus " + LABEL[base][0].lower() + LABEL[base][1:])
        ax[2, j].axhline(0, color="grey", lw=0.8)
        ax[2, j].set_ylabel("log-likelihood gain\n(nats per beep)")
        ax[2, j].set_xlabel("horizon (beeps ahead)")
    ax[0, 1].legend(fontsize=7, loc="upper right")
    ax[1, 0].legend(fontsize=7, loc="upper left")
    ax[2, 0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    f = ROOT / "figures" / "fig_forecast_r6.png"
    fig.savefig(f, dpi=200)
    print("written", f)


if __name__ == "__main__":
    main()
