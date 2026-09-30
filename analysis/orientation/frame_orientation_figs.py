"""Figures for the frame-orientation model (reads out_frame/eval_*.json).

    python analysis/orientation/frame_orientation_figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "orientation" / "out_frame"
FIG = ROOT / "figures"
DS = {"baumeister1": "Baumeister S1", "bayer": "Bayer (openESM 0076)"}
EV = {ds: json.load(open(OUT / f"eval_{ds}.json")) for ds in DS}
# empirical within-person contrasts vs present, rescaled to the unit valence scale
EMP = {"baumeister1": (-0.384 / 6, -0.149 / 6), "bayer": (-0.556 / 4, -0.145 / 4)}
COL = {"baumeister1": "#3b6ea8", "bayer": "#c8702a"}


def ci(c):
    m, lo, hi = c
    return m, [[m - lo], [hi - m]]


def summary():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    # (a) next-signal orientation LL gain over base rates
    ax = axes[0]
    names = ["M1", "M1b", "M2", "M1_collapsed"]
    labels = ["Markov", "Markov\n+valence", "active\ninference", "Markov,\nshared valence"]
    for j, ds in enumerate(DS):
        for i, n in enumerate(names):
            m, err = ci(EV[ds]["contrasts"][f"{n}-M0"]["ll"])
            ax.bar(i + j * 0.38, m, 0.34, yerr=err, capsize=3, color=COL[ds], label=DS[ds] if i == 0 else None)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(np.arange(len(names)) + 0.19); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("next-signal orientation log-lik. gain\nover base rates (nats/signal)")
    ax.set_title("(a) predicting the next reported orientation", fontsize=9); ax.legend(fontsize=8)
    # (b) ablations of the active-inference model relative to full M2
    ax = axes[1]
    abl = ["M2_flatpref", "M2_noepist", "M2_noprecision", "M1"]
    alab = ["flat\npreferences", "no epistemic\nterm", "zero policy\nprecision", "passive\nMarkov (M1)"]
    for j, ds in enumerate(DS):
        for i, n in enumerate(abl):
            key = f"{n}-M2" if n != "M1" else "M2-M1"
            m, err = ci(EV[ds]["contrasts"][key]["ll"])
            if n == "M1":
                m, err = -m, [err[1], err[0]]
            ax.bar(i + j * 0.38, m, 0.34, yerr=err, capsize=3, color=COL[ds])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(np.arange(len(abl)) + 0.19); ax.set_xticklabels(alab, fontsize=8)
    ax.set_ylabel("log-lik. change vs full active-inference model")
    ax.set_title("(b) variant minus full M2 (positive = variant better)", fontsize=9)
    # (c) fitted transition tendencies vs asserted
    ax = axes[2]
    for j, ds in enumerate(DS):
        m = EV[ds]["models"]["M2"]["params_mean"]["tendency"]
        s = EV[ds]["models"]["M2"]["params_sd"]["tendency"]
        ax.bar(np.arange(3) + j * 0.38, m, 0.34, yerr=s, capsize=3, color=COL[ds], label=DS[ds])
    ax.plot(np.arange(3) + 0.19, [0.70, 0.75, 0.90], "k_", ms=22, mew=2, label="asserted 0.70 / 0.75 / 0.90")
    ax.set_xticks(np.arange(3) + 0.19); ax.set_xticklabels(["RECALL", "ENGAGE", "FUTURATE"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("P(target frame | action), mean over folds, bar = SD")
    ax.set_title("(c) fitted transition tendencies", fontsize=9); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "frame_orientation_summary.png", dpi=150); plt.close()


def drift():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    names = ["M1_free", "M1", "M1_anchor5x", "M1_collapsed"]
    labs = ["no anchor", "anchor 100", "anchor 500", "anchor 100,\nshared valence"]
    for j, ds in enumerate(DS):
        vals = [EV[ds]["models"][n]["params_mean"]["Ao"][0][0] for n in names]
        ax.bar(np.arange(4) + j * 0.38, vals, 0.34, color=COL[ds], label=DS[ds])
    ax.set_xticks(np.arange(4) + 0.19); ax.set_xticklabels(labs, fontsize=8); ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(report past | past frame)"); ax.set_title("(a) what the past frame emits", fontsize=9)
    ax.legend(fontsize=8)
    ax = axes[1]
    for j, ds in enumerate(DS):
        for k, n in enumerate(["M1", "M1_anchor5x"]):
            mu = np.array(EV[ds]["models"][n]["params_mean"]["mu"])
            x = np.array([0, 1]) + j * 0.38 + k * 0.17 - 0.08
            ax.bar(x, [mu[0] - mu[1], mu[2] - mu[1]], 0.15, color=COL[ds], alpha=1.0 if k == 0 else 0.5,
                   label=f"{DS[ds]}, {'anchor 100' if k == 0 else 'anchor 500'}")
        ax.plot(np.array([0, 1]) + j * 0.38, EMP[ds], "k_", ms=18, mew=2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks([0.19, 1.19]); ax.set_xticklabels(["past minus present", "future minus present"])
    ax.set_ylabel("fitted frame valence difference (unit scale)")
    ax.set_title("(b) fitted frame valence; black = empirical within-person", fontsize=9)
    ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "frame_orientation_drift.png", dpi=150); plt.close()


if __name__ == "__main__":
    summary(); drift(); print("figures written")
