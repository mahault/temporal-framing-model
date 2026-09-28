"""Frame-sensitive test (a): predicted vs observed variance of affect after events (Geschwind).

Uses the held-out per-row arrays written by fit_v2.py (protocol P) for the gated
and inert v2 variants. For each held-out beep t, S1[t] is the model's one-step
predictive variance of y_t given y_{1:t-1} and eps[t] the realised innovation.
Rows are grouped by whether the previous beep (same segment) carried a reported
event, and by whether that event was in the top quartile of |e|. Reports mean
predicted variance, mean squared innovation, their ratio, participant-bootstrap
CIs on the after-event minus no-event differences, and the one-step Gaussian
NLL with the model's own variance on after-event rows, gated vs inert.

Writes reviews/variance_v2.md, .json and figures/fig_variance_v2.png.
Run:  python variance_v2.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import empirical_rebuild as er
from esm_eval_v3 import add_time_features

ROOT = Path(__file__).resolve().parent
N_BOOT = 2000
VARIANTS = ("gated", "inert", "clampPAST", "clampPRESENT", "clampFUTURE", "no_channels")


def load_rows(sample, variant):
    rows = {}
    for f in sorted((ROOT / "reviews").glob(f"v2_rows_{sample}_{variant}_fold*.npz")):
        z = np.load(f, allow_pickle=False)
        by = defaultdict(dict)
        for k in z.files:
            pid, key = k.split("__", 1)
            by[pid][key] = z[k]
        rows.update(by)
    return rows


def boot(per_part, fn, seed=0):
    pids = sorted(per_part)
    rng = np.random.RandomState(seed)
    point = fn([per_part[p] for p in pids])
    b = np.array([fn([per_part[p] for p in rng.choice(pids, len(pids), replace=True)]) for _ in range(N_BOOT)])
    return float(point), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    parts = er.load_participants()
    add_time_features(parts, True)
    e_abs = np.array([abs(b["e"]) for s in parts.values() for b in s if b["e"] is not None and b["e"] != 0])
    q75 = float(np.percentile(e_abs, 75))
    out = {}
    lines = ["# Variance after events, model v2 (2026-09-28)", "",
             "Script: `variance_v2.py`. Held-out one-step predictive variance S1 (model's own Kalman variance) and "
             "realised squared innovation, Geschwind, protocol P rows from fit_v2.py. 'after event' = the previous "
             f"beep in the same segment carried a reported event; 'after big event' = |e| at or above the top quartile ({q75:.2f}). "
             "CIs: participant bootstrap (2000 resamples).", "",
             "| variant | subset | rows | mean predicted S1 | mean eps^2 | ratio eps^2/S1 | NLL own S |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for v in VARIANTS:
        rows = load_rows("Geschwind", v)
        if not rows:
            continue
        per = {"no_event": {}, "after_event": {}, "after_big": {}}
        for pid, rw in rows.items():
            seq = parts[pid]
            S1, eps = rw["S1"], rw["eps"]
            for t in range(1, len(S1)):
                if seq[t].get("p") != seq[t - 1].get("p"):
                    continue
                e = seq[t - 1]["e"]
                had = e is not None and e != 0.0
                keys = ["after_event" if had else "no_event"]
                if had and abs(e) >= q75:
                    keys.append("after_big")
                for k in keys:
                    per[k].setdefault(pid, []).append((float(S1[t]), float(eps[t] ** 2)))
        res = {}
        for k, d in per.items():
            arr = np.array([x for vals in d.values() for x in vals])
            nll = np.mean(0.5 * np.log(2 * np.pi * arr[:, 0]) + arr[:, 1] / (2 * arr[:, 0]))
            res[k] = dict(n=len(arr), S1=float(arr[:, 0].mean()), eps2=float(arr[:, 1].mean()),
                          ratio=float(arr[:, 1].mean() / arr[:, 0].mean()), nll=float(nll))
            lines.append(f"| {v} | {k} | {len(arr)} | {res[k]['S1']:.5f} | {res[k]['eps2']:.5f} | {res[k]['ratio']:.3f} | {res[k]['nll']:.4f} |")
        # CIs: after-event minus no-event, predicted and observed
        both = {p: (np.array(per["after_event"][p]), np.array(per["no_event"][p]))
                for p in per["after_event"] if p in per["no_event"]}

        def dpred(lst):
            a = np.concatenate([x[0] for x in lst]); b = np.concatenate([x[1] for x in lst])
            return a[:, 0].mean() - b[:, 0].mean()

        def dobs(lst):
            a = np.concatenate([x[0] for x in lst]); b = np.concatenate([x[1] for x in lst])
            return a[:, 1].mean() - b[:, 1].mean()
        res["ci_pred_after_minus_none"] = boot(both, dpred)
        res["ci_obs_after_minus_none"] = boot(both, dobs)
        out[v] = res
    lines.append("")
    for v, res in out.items():
        cp, co = res["ci_pred_after_minus_none"], res["ci_obs_after_minus_none"]
        lines.append(f"- {v}: after event minus no event, predicted S1 {cp[0]:+.5f} [{cp[1]:+.5f}, {cp[2]:+.5f}]; "
                     f"observed eps^2 {co[0]:+.5f} [{co[1]:+.5f}, {co[2]:+.5f}]")
    # gated vs inert NLL on after-event rows, participant bootstrap
    if "gated" in out and "inert" in out:
        rg, ri = load_rows("Geschwind", "gated"), load_rows("Geschwind", "inert")
        per = {}
        for pid in rg:
            if pid not in ri:
                continue
            seq = parts[pid]
            vals = []
            for t in range(1, len(rg[pid]["S1"])):
                if seq[t].get("p") != seq[t - 1].get("p"):
                    continue
                e = seq[t - 1]["e"]
                if not (e is not None and e != 0.0):
                    continue
                ng = 0.5 * np.log(2 * np.pi * rg[pid]["S1"][t]) + rg[pid]["eps"][t] ** 2 / (2 * rg[pid]["S1"][t])
                ni = 0.5 * np.log(2 * np.pi * ri[pid]["S1"][t]) + ri[pid]["eps"][t] ** 2 / (2 * ri[pid]["S1"][t])
                vals.append((ng, ni))
            if vals:
                per[pid] = np.array(vals)
        d = boot(per, lambda lst: float(np.concatenate(lst)[:, 0].mean() - np.concatenate(lst)[:, 1].mean()))
        out["nll_after_event_gated_minus_inert"] = d
        lines.append(f"- one-step NLL after events, gated minus inert: {d[0]:+.5f} [{d[1]:+.5f}, {d[2]:+.5f}] (negative favours gated)")
    lines.append("")
    (ROOT / "reviews" / "variance_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "reviews" / "variance_v2.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vs = [v for v in ("gated", "inert") if v in out]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    subsets = ("no_event", "after_event", "after_big")
    x = np.arange(len(subsets))
    wdt = 0.2
    for j, v in enumerate(vs):
        ax.bar(x + (2 * j - 1.5) * wdt, [out[v][s]["S1"] for s in subsets], wdt, label=f"{v}: predicted S1",
               color=["#0072B2", "#D55E00"][j], alpha=0.85)
        ax.bar(x + (2 * j - 0.5) * wdt, [out[v][s]["eps2"] for s in subsets], wdt, label=f"{v}: observed eps^2",
               color=["#0072B2", "#D55E00"][j], alpha=0.35, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(["no event before", "after event", "after top-quartile event"])
    ax.set_ylabel("one-step variance of valence")
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig_variance_v2.png", dpi=200, bbox_inches="tight")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
