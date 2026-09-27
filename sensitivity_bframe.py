"""Sensitivity of the qualitative simulation results to the hand-set constants
(revision 2026-09-27, AAAI-27 reviewer 6A6h weakness 2).

Sweeps
  (A) frame stickiness s (the RECALL->PAST 0.70, ENGAGE->PRESENT 0.75,
      FUTURATE->FUTURE 0.90 and ABSTRACT->FUTURE 0.80 self-transition entries,
      all set to s) x valence pull-weight scale k (FUTURATE 0.5*k, ABSTRACT 0.65*k)
  (B) frame gain g (the new gating parameter) at default s and k

and records, per grid point and seed, three qualitative results of the paper:
  recall_collapse : RECALL%(healthy) - RECALL%(recall-impaired)     [Fig. feedback]
  future_fixation : mean q(FUTURE)(stressed) - q(FUTURE)(healthy)    [Fig. stress]
  diathesis       : final E[pi_pos] of vulnerable+stress < theta=2 AND
                    healthy+stress > theta                            [Fig. mood]

Outputs figures/fig_sensitivity.png and reviews/sensitivity_results.md.
Run:  python sensitivity_bframe.py [--workers 8] [--seeds 4] [--quick]
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
STICKY = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
PULL = [0.6, 0.8, 1.0, 1.2, 1.4]
GAINS = [0.0, 0.25, 0.5, 0.75, 1.0]
THETA = 2.0


def _job(args):
    """Runs in a worker process: set the module overrides, then simulate."""
    sticky, pull, gain, seed, T, T_sd = args
    import generative_model as gm
    gm.FRAME_STICKINESS = sticky
    gm.PULL_WEIGHT_SCALE = pull
    from experiments import (run_trial, FEEDBACK_PROFILES, STRESS_PROFILES,
                             STRESS_DECAY_PROFILES)
    from generative_model import RECALL
    out = dict(sticky=sticky, pull=pull, gain=gain, seed=seed)
    fb = {n: run_trial(**p, T=T, seed=seed, frame_gain=gain)
          for n, p in FEEDBACK_PROFILES.items()}
    out["recall_collapse"] = (np.mean(fb["healthy"]["action"] == RECALL)
                              - np.mean(fb["recall_impaired"]["action"] == RECALL))
    out["entropy_diff"] = (np.mean(fb["recall_impaired"]["policy_entropy_norm"])
                           - np.mean(fb["healthy"]["policy_entropy_norm"]))
    st = {n: run_trial(**p, T=T, seed=seed, frame_gain=gain)
          for n, p in STRESS_PROFILES.items()}
    out["future_fixation"] = (np.mean(st["stressed"]["frame_belief"][:, 2])
                              - np.mean(st["healthy"]["frame_belief"][:, 2]))
    out["past_stressed"] = float(np.mean(st["stressed"]["frame_belief"][:, 0]))
    sd = {n: run_trial(**STRESS_DECAY_PROFILES[n], T=T_sd, seed=seed, frame_gain=gain)
          for n in ("healthy_stress", "vulnerable_stress")}
    out["pi_healthy_stress"] = float(sd["healthy_stress"]["pi_pos"][-200:].mean())
    out["pi_vulnerable_stress"] = float(sd["vulnerable_stress"]["pi_pos"][-200:].mean())
    out["diathesis"] = float(out["pi_vulnerable_stress"] < THETA < out["pi_healthy_stress"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    T, T_sd = (150, 1000) if args.quick else (300, 3000)
    seeds = list(range(42, 42 + args.seeds))
    jobs = [(s, k, 1.0, sd, T, T_sd) for s, k, sd in itertools.product(STICKY, PULL, seeds)]
    jobs += [(None, None, g, sd, T, T_sd) for g, sd in itertools.product(GAINS, seeds)]
    print(f"{len(jobs)} simulations")
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(_job, jobs, chunksize=2))
    else:
        res = [_job(j) for j in jobs]

    # ── aggregate ─────────────────────────────────────────
    def grid(metric):
        M = np.full((len(STICKY), len(PULL)), np.nan)
        for i, s in enumerate(STICKY):
            for j, k in enumerate(PULL):
                vals = [r[metric] for r in res if r["sticky"] == s and r["pull"] == k]
                M[i, j] = np.mean(vals)
        return M

    A = dict(recall_collapse=grid("recall_collapse"),
             future_fixation=grid("future_fixation"),
             diathesis=grid("diathesis"),
             entropy_diff=grid("entropy_diff"))
    B = {}
    for g in GAINS:
        vals = [r for r in res if r["sticky"] is None and r["gain"] == g]
        B[g] = {m: (np.mean([v[m] for v in vals]), np.std([v[m] for v in vals]))
                for m in ("recall_collapse", "future_fixation", "diathesis",
                          "entropy_diff", "past_stressed")}

    # ── figure ────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    titles = [("recall_collapse", "RECALL collapse\n(healthy - impaired)", "Blues"),
              ("future_fixation", "Future fixation\n(stressed - healthy q(FUTURE))", "Oranges"),
              ("diathesis", "Diathesis-stress holds\n(fraction of seeds)", "Greens")]
    for ax, (m, title, cmap) in zip(axes[:3], titles):
        im = ax.imshow(A[m], origin="lower", cmap=cmap, aspect="auto",
                       vmin=0 if m == "diathesis" else None, vmax=1 if m == "diathesis" else None)
        ax.set_xticks(range(len(PULL))); ax.set_xticklabels(PULL)
        ax.set_yticks(range(len(STICKY))); ax.set_yticklabels(STICKY)
        ax.set_xlabel("pull-weight scale k"); ax.set_ylabel("frame stickiness s")
        ax.set_title(title, fontsize=10)
        for i in range(len(STICKY)):
            for j in range(len(PULL)):
                ax.text(j, i, f"{A[m][i, j]:.2f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046)
        # mark the paper's defaults (s = 0.70..0.90 -> nearest 0.8, k = 1.0)
        ax.scatter([PULL.index(1.0)], [STICKY.index(0.8)], marker="s", s=120,
                   facecolors="none", edgecolors="k", linewidths=1.5)
    ax = axes[3]
    for m, c in (("recall_collapse", "#0072B2"), ("future_fixation", "#E69F00"),
                 ("diathesis", "#009E73")):
        mu = [B[g][m][0] for g in GAINS]; sd = [B[g][m][1] for g in GAINS]
        ax.errorbar(GAINS, mu, yerr=sd, marker="o", color=c, label=m.replace("_", " "))
    ax.set_xlabel("frame gain g"); ax.set_ylabel("effect size")
    ax.set_title("Sensitivity to the gating gain", fontsize=10)
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "fig_sensitivity.png", dpi=200, bbox_inches="tight")

    # ── markdown ──────────────────────────────────────────
    lines = ["# Sensitivity of the qualitative results (2026-09-27)", "",
             f"Script: `sensitivity_bframe.py`, {args.seeds} seeds per cell, T={T}, "
             f"T_diathesis={T_sd}. Paper defaults: stickiness 0.70/0.75/0.90/0.80, "
             "pull scale 1.0, gain 1.0.", ""]
    for m in ("recall_collapse", "future_fixation", "diathesis", "entropy_diff"):
        lines.append(f"## {m} (rows: stickiness s; cols: pull scale k)")
        lines.append("")
        lines.append("| s \\ k | " + " | ".join(str(k) for k in PULL) + " |")
        lines.append("|---|" + "---:|" * len(PULL))
        for i, s in enumerate(STICKY):
            lines.append(f"| {s} | " + " | ".join(f"{A[m][i, j]:.3f}" for j in range(len(PULL))) + " |")
        lines.append("")
    lines.append("## Frame gain g (default s, k)")
    lines.append("")
    lines.append("| g | recall_collapse | future_fixation | diathesis | entropy_diff (impaired - healthy) | q(PAST) stressed |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for g in GAINS:
        b = B[g]
        lines.append(f"| {g} | {b['recall_collapse'][0]:.3f} ({b['recall_collapse'][1]:.3f}) | "
                     f"{b['future_fixation'][0]:.3f} ({b['future_fixation'][1]:.3f}) | "
                     f"{b['diathesis'][0]:.2f} | {b['entropy_diff'][0]:+.3f} ({b['entropy_diff'][1]:.3f}) | "
                     f"{b['past_stressed'][0]:.3f} |")
    lines.append("")
    frac = {m: float(np.mean(A[m] > 0)) for m in ("recall_collapse", "future_fixation")}
    lines.append(f"Fraction of the s x k grid with recall_collapse > 0: {frac['recall_collapse']:.2f}; "
                 f"future_fixation > 0: {frac['future_fixation']:.2f}; "
                 f"diathesis holding in all seeds: {float(np.mean(A['diathesis'] == 1.0)):.2f}; "
                 f"in a majority of seeds: {float(np.mean(A['diathesis'] > 0.5)):.2f}.")
    out = ROOT / "reviews" / "sensitivity_results.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
