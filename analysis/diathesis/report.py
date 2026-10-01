"""Tables and figures for the diathesis data test (reads out/summary.json).

Run from the repo root:  python analysis/diathesis/report.py
Writes figures/diathesis_forest.png, figures/diathesis_reliability.png and
prints markdown tables used in reviews/DIATHESIS_DATA_TEST.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
S = json.loads((Path(__file__).resolve().parent / "out" / "summary.json").read_text())
DS = [k for k in S if not k.startswith("_")]
LABEL = {
    "a_level_model": "(a) level, model baseline", "a_mood_mean_model": "(a) level, model mean mood",
    "a_level_free": "(a) level, person mean",
    "b_inertia_model": "(b) inertia, model log timescale", "b_inertia_free": "(b) inertia, AR(1)",
    "c_react_model": "(c) reactivity, model event weight", "c_react_free": "(c) reactivity, event slope",
    "d_persist_free": "(d) thought persistence, lag-1 r", "d_level_free": "(d) thought level, person mean",
}
ORDER = list(LABEL)


def f(x, d=2):
    return "NA" if x is None else f"{x:+.{d}f}"


def ci(r):
    return f"{f(r['beta'])} [{f(r['lo'])}, {f(r['hi'])}]" if r and r.get("beta") is not None else "NA"


def tables():
    out = []
    out.append("| Outcome | " + " | ".join(f"{d} (N={S[d]['n']})" for d in DS) + " | Pooled |")
    out.append("|---|" + "---|" * (len(DS) + 1))
    P = S.get("_pooled_neuroticism", {})
    for k in ORDER:
        cells = [ci(S[d]["relations"].get("neuroticism", {}).get(k)) for d in DS]
        if all(c == "NA" for c in cells):
            continue
        p = P.get(k)
        pc = ci(p) if p else "NA"
        out.append(f"| {LABEL[k]} | " + " | ".join(cells) + f" | {pc} |")
    out.append("")
    if "Gainey" in S:
        out.append("| Gainey outcome | neuroticism | brooding (RRS) | dysphoria (IDAS) |")
        out.append("|---|---|---|---|")
        R = S["Gainey"]["relations"]
        for k in ORDER:
            if k in R.get("neuroticism", {}):
                out.append(f"| {LABEL[k]} | {ci(R['neuroticism'][k])} | {ci(R.get('brooding_trait', {}).get(k))} | {ci(R.get('dysphoria', {}).get(k))} |")
        out.append("")
    out.append("| Split-half reliability (Spearman-Brown) | " + " | ".join(DS) + " |")
    out.append("|---|" + "---|" * len(DS))
    for k in ("a_model", "a_free", "b_model", "b_free", "c_model", "c_free", "d_free"):
        cells = [S[d]["reliability"].get(k) for d in DS]
        out.append(f"| {k} | " + " | ".join("NA" if c is None or c.get('spearman_brown') is None else f"{c['spearman_brown']:.2f} (n={c['n']})" for c in cells) + " |")
    out.append("")
    out.append("| Incremental (trait ~ model + model-free, neuroticism) | " + " | ".join(DS) + " |")
    out.append("|---|" + "---|" * len(DS))
    for lab in ("a", "b", "c"):
        cells = []
        for d in DS:
            r = S[d]["incremental"].get("neuroticism", {}).get(lab)
            cells.append("NA" if not r else f"model {f(r['b_model'])} (SE {r['se_model']:.2f}); free {f(r['b_free'])} (SE {r['se_free']:.2f}); r {r['r_model_free']:.2f}")
        out.append(f"| ({lab}) | " + " | ".join(cells) + " |")
    out.append("")
    out.append("| Diathesis-stress interaction (neg. event x neuroticism) | kind | b_int [95% CI] | z | obs | people |")
    out.append("|---|---|---|---|---|---|")
    for d in DS:
        for kind, r in S[d].get("interaction", {}).get("neuroticism", {}).items():
            out.append(f"| {d} | {kind} | {r['b_int']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['z']:+.2f} | {r['n_obs']} | {r['n_part']} |")
    pe = P.get("e_interaction_lagged_std")
    if pe:
        out.append(f"| Pooled (lagged, in SD of valence) | lagged | {pe['beta']:+.4f} [{pe['lo']:+.4f}, {pe['hi']:+.4f}] | | | |")
    return "\n".join(out)


def forest():
    P = S.get("_pooled_neuroticism", {})
    ks = [k for k in ORDER if any(k in S[d]["relations"].get("neuroticism", {}) for d in DS)]
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(ks) * (len(DS) + 1) / 2 + 1.5))
    y = 0
    ticks, labs = [], []
    cols = {"Geschwind": "#1f77b4", "Kane": "#ff7f0e", "Gainey": "#2ca02c", "Pooled": "black"}
    for k in reversed(ks):
        for d in list(DS) + ["Pooled"]:
            r = P.get(k) if d == "Pooled" else S[d]["relations"].get("neuroticism", {}).get(k)
            if r and r.get("beta") is not None:
                ax.errorbar(r["beta"], y, xerr=[[r["beta"] - r["lo"]], [r["hi"] - r["beta"]]], fmt="D" if d == "Pooled" else "o",
                            color=cols[d], ms=5, capsize=2, label=d)
            y += 1
        ticks.append(y - (len(DS) + 1) / 2 - 0.5)
        labs.append(LABEL[k])
        y += 1
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labs, fontsize=8)
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="lower right")
    ax.set_xlabel("standardized association with neuroticism (95% CI)")
    ax.set_title("Vulnerability and person-level mood dynamics", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "diathesis_forest.png", dpi=160)


def reliability():
    ks = ("a_model", "a_free", "b_model", "b_free", "c_model", "c_free", "d_free")
    fig, ax = plt.subplots(figsize=(7, 3))
    w = 0.8 / len(DS)
    for j, d in enumerate(DS):
        vals = [S[d]["reliability"].get(k, {}).get("spearman_brown") for k in ks]
        vals = [np.nan if v is None else v for v in vals]
        ax.bar(np.arange(len(ks)) + j * w, vals, w, label=d)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(np.arange(len(ks)) + w * (len(DS) - 1) / 2)
    ax.set_xticklabels(ks, fontsize=8)
    ax.set_ylabel("split-half reliability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "diathesis_reliability.png", dpi=160)


if __name__ == "__main__":
    print(tables())
    forest()
    reliability()
    for d in DS:
        r = S[d]
        print(d, "n", r["n"], "signals", r["signals"], "theta 5/50/95", [round(x, 1) for x in r["theta_range"]],
              "r(level model, free)", round(r["corr_level_model_free"], 2), "lam", round(r["lam"], 1))
