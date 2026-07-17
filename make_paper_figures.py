"""Publication figures for the unified temporal-framing paper.

Numbers are the held-out results computed this session (see EMPIRICAL_RECORD.md):
  - Rutledge GBE affect prediction (rutledge_affect_fit.py)
  - ESM affect-dynamics prediction (eval_fitted_cv.py / empirical_rebuild.py)
  - Counterfactual/regret signature (regret diagnostic on S2021c)
Design follows the dataviz skill: Okabe-Ito colorblind-safe palette, thin marks,
direct value labels, single axis, hero series (our model) highlighted, muted
comparisons. Saved to figures/.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIG = Path("figures"); FIG.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.dpi": 150, "savefig.dpi": 200, "font.family": "DejaVu Sans",
})
# Okabe-Ito (colorblind-safe)
BLUE, ORANGE, GREEN, GRAY, VERM = "#0072B2", "#E69F00", "#009E73", "#9AA0A6", "#D55E00"


def _label_bars(ax, bars, fmt="{:.3f}", dy=0.004):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=9.5)


def _esm_panel(ax, letter, title, baseline, model):
    horizons = ["1 step", "2 steps", "3 steps"]
    x = np.arange(len(horizons)); w = 0.38
    b1 = ax.bar(x - w/2, baseline, w, color=GRAY, edgecolor="white", linewidth=1.5,
                label="Best linear/AR baseline")
    b2 = ax.bar(x + w/2, model, w, color=BLUE, edgecolor="white", linewidth=1.5,
                label="Full model")
    _label_bars(ax, b1); _label_bars(ax, b2)
    ax.set_xticks(x); ax.set_xticklabels(horizons)
    ax.set_ylabel("Held-out $R^2$ (next-moment valence)")
    ax.set_xlabel("Prediction horizon")
    ax.set_ylim(0, 0.56)
    ax.set_title(f"{letter}  {title}", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")


def fig_model_advantage():
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 4.5))

    # --- Panel A: head-to-head, each predecessor = one channel (Rutledge) ---
    names = ["Joffily\n(backward)", "Pattisapu\n(present)", "Hesp\n(forward)", "OURS\n(all 3)"]
    vals = [0.0001, 0.111, 0.034, 0.144]
    bars = axA.bar(names, vals, color=[GRAY, GRAY, GRAY, BLUE], width=0.66,
                   edgecolor="white", linewidth=1.5)
    _label_bars(axA, bars)
    axA.axhline(0.146, ls="--", lw=1.2, color="#555")
    axA.text(-0.35, 0.150, "happiness eq. (0.146)", ha="left", va="bottom", fontsize=8, color="#555")
    axA.set_ylabel("Held-out $R^2$ (momentary happiness)")
    axA.set_ylim(0, 0.18)
    axA.set_title("A  Subsumption: Rutledge GBE\n(each predecessor = one channel; 14,803 subj)",
                  fontsize=10.5, loc="left")

    # --- Panels B, C: ESM affect-dynamics on two independent samples ---
    _esm_panel(axB, "B", "Extension: remitted-depression ESM\n(Geschwind–Bringmann, n=129)",
               [0.089, 0.274, 0.097], [0.193, 0.302, 0.170])
    axB.text(-0.44, 0.235, "2.2x", ha="left", color=VERM, fontsize=11, fontweight="bold")
    _esm_panel(axC, "C", "Replication: reliability ESM\n(osf.io/83cfk, n=91)",
               [0.475, 0.338, 0.276], [0.486, 0.377, 0.322])

    fig.suptitle("Subsumes standard reward-affect models (A); the affect-dynamics advantage is real "
                 "but sample-dependent: large where affect carries dynamics (B), near parity where "
                 "affect is already highly persistent (C)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig_model_advantage.png", bbox_inches="tight")
    plt.close(fig)
    print("saved figures/fig_model_advantage.png")


def fig_counterfactual_signature():
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    cats = ["After RELIEF\n(foregone worse)", "After REGRET\n(foregone better)"]
    vals = [0.256, 0.453]
    bars = ax.bar(cats, vals, color=[GREEN, VERM], width=0.6, edgecolor="white", linewidth=1.5)
    _label_bars(ax, bars, fmt="{:.2f}", dy=0.008)
    ax.set_ylabel("P(switch on next choice)")
    ax.set_ylim(0, 0.55)
    ax.set_title("Counterfactual emotion drives behaviour\n"
                 "Sugawara & Katahira (n=143): regret predicts switching (t=10.4)", fontsize=10.5, loc="left")
    ax.annotate("", xy=(1, 0.47), xytext=(0, 0.47),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.4))
    ax.text(0.5, 0.485, "1.8×", ha="center", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig_counterfactual_signature.png", bbox_inches="tight")
    plt.close(fig)
    print("saved figures/fig_counterfactual_signature.png")


if __name__ == "__main__":
    fig_model_advantage()
    fig_counterfactual_signature()
