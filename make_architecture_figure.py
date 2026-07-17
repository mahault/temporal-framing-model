"""Layered-architecture schematic for the unified paper.

Shows the whole model as a stack of layers and marks which single layer each prior
model corresponds to (Alejandro's request: 'see the whole architecture with layers,
where the prior models plug in'). This is the readable companion to the dense SPM
factor graph (Figure 1).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

FIG = Path("figures"); FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 200})

# Okabe-Ito
BLUE, ORANGE, GREEN, GRAY, VERM, SKY, YEL = (
    "#0072B2", "#E69F00", "#009E73", "#9AA0A6", "#D55E00", "#56B4E9", "#F0E442")

fig, ax = plt.subplots(figsize=(12.5, 7.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, text, fc, ec="k", fs=10.5, bold=False, tc="k", lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.6,rounding_size=2.2",
                 fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal",
            color=tc, zorder=4, wrap=True)


def arrow(x1, y1, x2, y2, style="-|>", ls="-", color="#333", lw=1.8, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=16, lw=lw, color=color, ls=ls,
                 connectionstyle=f"arc3,rad={rad}", zorder=2))


LX, LW = 4, 60           # left column for the stack
# ---- Layer 0: task-specific generative model ----
box(LX, 6, LW, 11,
    "TASK-SPECIFIC GENERATIVE MODEL (POMDP)\n"
    "states $s$, observations $o$, policies $\\pi$, expected free energy $G$",
    "#EFEFEF", fs=10.5, bold=False)
ax.text(LX + LW / 2, 3.0, "e.g. the gamble task-model (§9), an ESM affect model, "
        "any decision task", ha="center", va="center", fontsize=8.2, color="#666",
        style="italic")

# ---- temporal-frame modulator (left) ----
box(LX, 30, 17, 12,
    "TEMPORAL\nFRAME $f$\npast/present/\nfuture\n(precision over\nhorizons)",
    "#D9F0EA", ec=GREEN, fs=8.6)

# ---- Layer 1: three channels ----
cy, ch = 30, 12
box(24, cy, 12, ch, "BACKWARD\n$-dF/dt$\n(VFE)", "#DCEBF5", ec=BLUE, fs=9)
box(37.5, cy, 12, ch, "PRESENT\nRPE", "#FBE6CC", ec=ORANGE, fs=9)
box(51, cy, 13, ch, "FORWARD\nEFE affective\ncharge", "#F6D9CC", ec=VERM, fs=9)
ax.text(4, 46.5, "AFFECTIVE READOUT LAYER\n(three channels)",
        ha="left", va="center", fontsize=9, fontweight="bold", color="#444")

# ---- Layer 2: composite ----
box(20, 52, 44, 8,
    "COMPOSITE VALENCE  $V=\\tanh(v_{model}+v_{reward}+v_{action})$",
    "#E7E7E7", fs=10)

# ---- Layer 3: readout state + mood ----
box(6, 68, 26, 9, "REPRESENTED VALENCE $v$\nfelt valence $o_{val}$ (report, gating)",
    "#EFEFEF", fs=9)
box(40, 68, 24, 9, "MOOD LAYER (M5)\nslow $\\pi_{pos}$ over trials", "#EDE3F3",
    ec="#8e44ad", fs=9)

# ---- arrows (upward flow) ----
for xc in (30, 43.5, 57.5):
    arrow(xc, 17, xc, 30)                       # task model -> channels
    arrow(xc, 42, 42 if xc != 43.5 else 43.5, 52, rad=0.0)  # channels -> composite
arrow(42, 60, 19, 68, rad=-0.05)                # composite -> represented valence
arrow(44, 60, 52, 68, rad=0.05)                 # composite -> mood
arrow(21, 36, 24, 36, color=GREEN, ls=(0, (4, 2)))   # frame -> channels (modulate)
# mood feedback down to task model (slow)
arrow(52, 68, 66, 40, style="-|>", ls=(0, (5, 3)), color="#8e44ad", lw=1.3, rad=0.35)
ax.text(70.5, 54, "slow mood\nfeedback", fontsize=7.6, color="#8e44ad",
        ha="left", va="center", style="italic")

# ================= right column: prior models as slices =================
RX = 76
ax.text(RX + 11, 92, "Prior models = single layers", ha="center",
        fontsize=11, fontweight="bold")
ax.text(RX + 11, 88.4, "(our model integrates all of them)", ha="center",
        fontsize=8.2, color="#666", style="italic")

rows = [
    ("Joffily & Coricelli 2013", "backward channel only\n(perception-only VFE)", BLUE),
    ("Pattisapu et al. 2024", "present channel only\n(RPE, POMDP)", ORANGE),
    ("Hesp et al. 2021", "forward channel + mood layer\n(no reward / no frame)", VERM),
    ("Ours", "all three channels + temporal\nframe + readout state", "#111"),
]
y0 = 80
for i, (name, desc, col) in enumerate(rows):
    y = y0 - i * 15
    hero = name == "Ours"
    box(RX, y - 9.5, 23, 10.5, f"{name}\n" + desc,
        "#FFF7E6" if hero else "white", ec=col,
        lw=2.6 if hero else 1.6, fs=8.4, bold=hero)
    ax.add_patch(plt.Rectangle((RX - 2.4, y - 9.5), 1.6, 10.5, color=col, zorder=4))

fig.suptitle("Affect as a readout layer over a task-specific model: the whole "
             "architecture, and where prior models sit",
             fontsize=12.5, fontweight="bold", y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(FIG / "fig_architecture.png", bbox_inches="tight")
plt.close(fig)
print("saved figures/fig_architecture.png")
