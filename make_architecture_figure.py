"""Architecture schematic for the restructured paper (Figure 1).

Shows the model as layers: a task model supplies inputs, three channels read affect off it, the
temporal frame weights all three channels, their weighted sum drives a fast valence state that relaxes
toward a slow mood state, and a precision state widens or narrows the forecast after recent surprise.
The right column marks which channel each earlier account covers.
Run: python make_architecture_figure.py  ->  figures/fig_architecture.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

FIG = Path("figures"); FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 200})

BLUE, ORANGE, GREEN, VERM, PURP, GOLD = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#8e44ad", "#b8860b"

fig, ax = plt.subplots(figsize=(12.5, 7.8))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, text, fc, ec="k", fs=10, bold=False, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2.2",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=4)


def arrow(x1, y1, x2, y2, ls="-", color="#333", lw=1.8, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, lw=lw,
                                 color=color, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=2))


def note(x, y, text, color, ha="left"):
    ax.text(x, y, text, fontsize=7.6, color=color, ha=ha, va="center", style="italic")


# task model
box(4, 5, 60, 10, "TASK MODEL\nthe gamble model, or the experience-sampling filter", "#EFEFEF", fs=10)
note(34, 1.6, "supplies outcomes, expectations and forecast errors to the channels", "#666", ha="center")

# three channels
cy, ch = 27, 12
box(24, cy, 12, ch, "BACKWARD $v_B$\nchange in\nmodel evidence", "#DCEBF5", ec=BLUE, fs=9)
box(38, cy, 12, ch, "PRESENT $v_P$\nreward\nprediction error", "#FBE6CC", ec=ORANGE, fs=9)
box(52, cy, 12, ch, "FORWARD $v_F$\nrevision\nof plans", "#F6D9CC", ec=VERM, fs=9)
ax.text(24, 43, "THREE CHANNELS", ha="left", va="center", fontsize=9, fontweight="bold", color="#444")
for xc in (30, 44, 58):
    arrow(xc, 15, xc, cy)

# temporal frame
box(4, cy, 15, ch, "TEMPORAL\nFRAME $f$\npast, present\nor future", "#D9F0EA", ec=GREEN, fs=8.8)

# valence drive
box(24, 49, 40, 7, "VALENCE DRIVE  $\\sum_c w_c\\,\\beta_c\\,v_c$", "#E7E7E7", fs=10)
for xc in (30, 44, 58):
    arrow(xc, cy + ch, xc, 49)
arrow(12, cy + ch, 24, 52.5, color=GREEN, ls=(0, (4, 2)), rad=-0.2)
note(4.5, 47.5, "frame weights $w_c$\nfor all three channels", GREEN)

# fast and slow states
box(6, 66, 26, 9, "FAST VALENCE STATE $x_t$\nread out as reported valence $y_t$", "#EFEFEF", fs=9)
box(40, 66, 24, 9, "SLOW MOOD STATE $m_t$\nthe level valence relaxes to", "#EDE3F3", ec=PURP, fs=9)
arrow(40, 56, 22, 66, rad=-0.05)
arrow(40, 70.5, 32.6, 70.5, color=PURP, lw=1.4)
note(36.3, 73.2, "pull", PURP, ha="center")
arrow(32.6, 68, 40, 68, color="#555", lw=1.0)
note(36.3, 64.6, "update", "#555", ha="center")
arrow(64.5, 66, 62, 39.5, ls=(0, (5, 3)), color=PURP, lw=1.3, rad=-0.35)
note(67.5, 53, "optimism term\nof the forward\nchannel", PURP)

# precision state
box(6, 84, 26, 8, "PRECISION STATE $z_t$\nrecent surprise widens or\nnarrows the forecast", "#FFF4C2", ec=GOLD, fs=8.6)
arrow(19, 75.5, 19, 84, color=GOLD, lw=1.4)
note(20.3, 79.7, "squared forecast error", GOLD)

# right column: earlier accounts
RX = 76
ax.text(RX + 11, 93, "Earlier accounts", ha="center", fontsize=11, fontweight="bold")
ax.text(RX + 11, 89.4, "(each covers one channel)", ha="center", fontsize=8.2, color="#666", style="italic")
rows = [
    ("Joffily & Coricelli 2013", "backward channel\n(rate of change of free energy)", BLUE),
    ("Pattisapu et al. 2024", "present channel\n(reward prediction error)", ORANGE),
    ("Hesp et al. 2021", "forward channel\n(affective charge of plans)", VERM),
    ("This model", "three channels, temporal frame,\ntwo timescales, precision state", "#111"),
]
for i, (name, desc, col) in enumerate(rows):
    y = 81 - i * 15
    hero = name == "This model"
    box(RX, y - 9.5, 23, 10.5, f"{name}\n" + desc, "#FFF7E6" if hero else "white", ec=col,
        lw=2.6 if hero else 1.6, fs=8.4, bold=hero)
    ax.add_patch(plt.Rectangle((RX - 2.4, y - 9.5), 1.6, 10.5, color=col, zorder=4))

fig.tight_layout()
fig.savefig(FIG / "fig_architecture.png", bbox_inches="tight")
plt.close(fig)
print("saved figures/fig_architecture.png")
