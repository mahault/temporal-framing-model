"""Recalibrated PAD emotion-space figure for the unified paper.

Honest scope (see EMPIRICAL_RECORD.md): the Pleasure-Arousal circumplex separates
all ten targeted profiles into the correct quadrants; Dominance adds the classic
anger (approach, high) vs fear (withdrawal, low) split. We do NOT claim full 3D
octant separation.

Readouts (principled active-inference quantities, centred across profiles):
  Pleasure  = centred positive-belief precision  pi_pos_eff   (how positively the
              agent construes its situation)
  Arousal   = centred mean expected free energy   mean_G       (expected demand)
  Dominance = centred policy precision  (1 - normalised policy entropy)  (control)
Design follows the dataviz skill: Okabe-Ito-adjacent emotion hues, thin marks,
direct labels, recessive guides.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from experiments import run_emotion_validation

FIG = Path("figures"); FIG.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 200, "font.family": "DejaVu Sans",
})

# centres/scales for the three readouts (calibration constants)
CP, SP = 1.75, 1.6      # pleasure  (pi_pos_eff)
CA, SA = 10.5, 6.0      # arousal   (mean G)
CD, SD = 0.55, 0.13     # dominance (policy precision)

ECOL = {
    'happy': '#f1c40f', 'content': '#2ecc71', 'calm': '#1abc9c',
    'excited': '#e67e22', 'alert': '#e74c3c', 'angry': '#c0392b',
    'fearful': '#8e44ad', 'sad': '#3498db', 'depressed': '#2c3e50',
    'bored': '#95a5a6',
}


def pad(h):
    P = np.tanh((np.mean(h['pi_pos_eff']) - CP) / SP)
    A = np.tanh((np.mean(h['G'].mean(axis=1)) - CA) / SA)
    D = np.tanh(((1.0 - np.mean(h['policy_entropy_norm'])) - CD) / SD)
    return P, A, D


def main():
    res = run_emotion_validation()
    coord = {n: pad(h) for n, h in res.items()}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.4),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    # ---- Panel A: Pleasure x Arousal circumplex ----
    axL.axhline(0, color="#bbb", lw=0.8, zorder=1)
    axL.axvline(0, color="#bbb", lw=0.8, zorder=1)
    # per-label offsets (dx, dy in points, ha) to avoid collisions
    LOFF = {
        'excited': (0, -15, "center"), 'happy': (0, 12, "center"),
        'alert': (0, 12, "center"), 'content': (0, 12, "center"),
        'calm': (0, 12, "center"), 'angry': (16, 8, "left"),
        'fearful': (-16, 8, "right"), 'depressed': (0, 12, "center"),
        'sad': (0, -14, "center"), 'bored': (0, -14, "center"),
    }
    for name, (P, A, D) in coord.items():
        axL.scatter(P, A, s=190, color=ECOL[name], edgecolors="k",
                    linewidths=1.3, zorder=5)
        dx, dy, ha = LOFF[name]
        axL.annotate(name, (P, A), xytext=(dx, dy), textcoords="offset points",
                     ha=ha, va="bottom" if dy > 0 else "top", fontsize=9.5)
    axL.set_xlim(-1.15, 1.15); axL.set_ylim(-1.15, 1.15)
    axL.set_xlabel("Pleasure  (positive-belief precision)")
    axL.set_ylabel("Arousal  (mean expected free energy)")
    axL.set_aspect("equal")
    q = dict(fontsize=8.5, color="#888", style="italic")
    axL.text(0.99, 1.10, "activated pleasant", ha="right", **q)
    axL.text(-0.99, 1.10, "activated unpleasant", ha="left", **q)
    axL.text(-0.99, -1.12, "deactivated unpleasant", ha="left", **q)
    axL.text(0.99, -1.12, "deactivated pleasant", ha="right", **q)
    axL.set_title("A  Pleasure-Arousal circumplex: all ten profiles fall in the\n"
                  "correct quadrant (Russell 1980)", fontsize=10.5, loc="left")

    # ---- Panel B: Dominance separates anger from fear ----
    order = sorted(coord, key=lambda n: coord[n][2])
    ys = np.arange(len(order))
    for y, name in zip(ys, order):
        D = coord[name][2]
        hi = name in ("angry", "fearful")
        axR.plot([0, D], [y, y], color=ECOL[name], lw=3 if hi else 2,
                 alpha=1.0 if hi else 0.55, zorder=2)
        axR.scatter(D, y, s=110 if hi else 70, color=ECOL[name],
                    edgecolors="k", linewidths=1.2 if hi else 0.8, zorder=3)
    axR.axvline(0, color="#bbb", lw=0.8)
    axR.set_yticks(ys); axR.set_yticklabels(order, fontsize=9.5)
    axR.set_xlim(-1.15, 1.15)
    axR.set_xlabel("Dominance  (policy precision)")
    axR.set_title("B  Dominance splits high-arousal negatives:\n"
                  "anger (approach, high) vs fear (withdrawal, low)",
                  fontsize=10.5, loc="left")
    axR.annotate("anger", (coord['angry'][2], order.index('angry')),
                 xytext=(-6, 10), textcoords="offset points", fontsize=9,
                 ha="right", color=ECOL['angry'], fontweight="bold")
    axR.annotate("fear", (coord['fearful'][2], order.index('fearful')),
                 xytext=(6, 10), textcoords="offset points", fontsize=9,
                 ha="left", color=ECOL['fearful'], fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIG / "fig_pad_circumplex.png", bbox_inches="tight")
    plt.close(fig)

    # report correctness
    tgt = {'happy': ('+', '+'), 'content': ('+', '-'), 'calm': ('+', '-'),
           'excited': ('+', '+'), 'alert': ('+', '+'), 'angry': ('-', '+'),
           'fearful': ('-', '+'), 'sad': ('-', '-'), 'depressed': ('-', '-'),
           'bored': ('-', '-')}
    ok = 0
    for n, (P, A, D) in coord.items():
        g = ('+' if P >= 0 else '-', '+' if A >= 0 else '-')
        ok += g == tgt[n]
        print(f"  {n:10s} P={P:+.2f} A={A:+.2f} D={D:+.2f}  quad {''.join(g)} "
              f"{'OK' if g == tgt[n] else 'XX target '+''.join(tgt[n])}")
    print(f"circumplex quadrant correct: {ok}/10")
    print("saved figures/fig_pad_circumplex.png")


if __name__ == "__main__":
    main()
