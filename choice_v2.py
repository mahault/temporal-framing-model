"""Frame-sensitive test (c): the delayed-choice task with frame posteriors inferred
by model v2 from affect histories (2026-09-28).

Round 2 (frame_choice_task.py) showed that the gated EFE discounts the delayed
option differently under clamped frames and under frames set by one framing
action. Here the frame posterior comes from v2's recogniser (gated, Geschwind
group parameters from fit_v2.py) after twelve beeps of a synthetic affect
history, so the link runs from affect data to choice: history -> q(f) -> gated
EFE -> P(DELAYED) and the implied discount (delayed:immediate reward ratio at
indifference).

Histories (valence in [0, 1], event pleasantness in [-3, 3]):
  neutral      flat 0.6, no events
  loss streak  0.6 falling to 0.3 with negative events
  gain streak  0.4 rising to 0.7 with positive events
  volatile     alternating 0.3 / 0.7 with alternating events
  recovery     0.3 flat after a drop (mood above the fast state)

Writes reviews/choice_v2.md, .json and figures/fig_choice_v2.png.
Run:  python choice_v2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from frame_choice_task import indifference_ratio, p_delayed
from model_v2 import ModelV2, drive_sequence, load_group

ROOT = Path(__file__).resolve().parent


def histories():
    T = 12
    h = {}
    h["neutral"] = (np.full(T, 0.6), np.zeros(T))
    h["loss streak"] = (np.linspace(0.6, 0.3, T), np.where(np.arange(T) % 2 == 0, -2.0, -1.0))
    h["gain streak"] = (np.linspace(0.4, 0.7, T), np.where(np.arange(T) % 2 == 0, 2.0, 1.0))
    h["volatile"] = (np.where(np.arange(T) % 2 == 0, 0.3, 0.7), np.where(np.arange(T) % 2 == 0, -2.0, 2.0))
    rec = np.full(T, 0.3)
    rec[:4] = 0.6
    h["recovery"] = (rec, np.zeros(T))
    return h


def main():
    gp = json.loads((ROOT / "reviews" / "model_v2_forecast.json").read_text(encoding="utf-8"))["Geschwind"]["group"]["gated"]
    model = load_group(ModelV2(1, True, g=1.0), gp)
    gains = np.linspace(0, 1, 11)
    res = {"gains": gains.tolist(), "histories": {}}
    lines = ["# Frame posteriors from affect histories and intertemporal choice, model v2 (2026-09-28)", "",
             "Script: `choice_v2.py`. q(f) after twelve beeps of each history under v2 (gated, Geschwind group "
             "parameters); choice under the gated EFE of frame_choice_task.py (reward ratio 2:1 for the P(DELAYED) "
             "columns; the implied discount is the ratio at indifference).", "",
             "| history | q(PAST) | q(PRESENT) | q(FUTURE) | P(DELAYED) g=0 | P(DELAYED) g=1 | implied discount g=0 | g=1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for nm, (y, e) in histories().items():
        r = drive_sequence(model, y.astype(np.float32), e=e.astype(np.float32))
        qf = np.asarray(r["q"][-1], float)
        curve = [p_delayed(qf, g) for g in gains]
        ir0, ir1 = indifference_ratio(qf, 0.0), indifference_ratio(qf, 1.0)
        res["histories"][nm] = dict(q=qf.tolist(), curve=curve, discount_g0=ir0, discount_g1=ir1,
                                    channels=dict(vB=float(r["vB"][-1]), vP=float(r["vP"][-1]), vF=float(r["vF"][-1])))
        lines.append(f"| {nm} | {qf[0]:.2f} | {qf[1]:.2f} | {qf[2]:.2f} | {curve[0]:.3f} | {curve[-1]:.3f} | {ir0:.2f} | {ir1:.2f} |")
    lines.append("")
    (ROOT / "reviews" / "choice_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "reviews" / "choice_v2.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    for nm, d in res["histories"].items():
        ax.plot(gains, d["curve"], marker="o", ms=3, label=nm)
    ax.set_xlabel("frame gain g")
    ax.set_ylabel("P(choose DELAYED), ratio 2:1")
    ax.legend(fontsize=7.5, frameon=False)
    ax = axes[1]
    names = list(res["histories"])
    x = np.arange(len(names))
    vals = [min(res["histories"][n]["discount_g1"], 20.0) for n in names]
    ax.bar(x, vals, color="#0072B2")
    ax.axhline(res["histories"]["neutral"]["discount_g0"], color="k", ls="--", lw=0.8, label="g = 0 (any history)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=15)
    ax.set_ylabel("implied discount at g = 1 (capped at 20)")
    ax.legend(fontsize=7.5, frameon=False)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig_choice_v2.png", dpi=200, bbox_inches="tight")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
