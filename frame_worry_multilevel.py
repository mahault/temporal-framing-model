"""Frame-to-worry association, done honestly (revision 2026-09-27).

The AAAI-27 reviewers pointed out that the valence composite fed to the model
(cheerful + relaxed - worried - fearful - sad) contains the worry item, so the
paper's "non-circular" r = 0.166 between the future-frame posterior and worry
was not non-circular. This script:

  1. drives the model with worry REMOVED from the composite
     (cheerful + relaxed - fearful - sad), for the gated (g = 1) and the
     pre-revision ungated (g = 0) model;
  2. reports the raw Pearson r, the partial r after regressing both variables
     on concurrent valence v_t and event pleasantness e_t, the within-person
     (participant-demeaned) r, and a cluster-robust (participant-clustered
     sandwich) z for the partial slope;
  3. writes the numbers to reviews/frame_worry_multilevel_results.md.

Only the Geschwind sample has a worry item; the osf_83cfk replication sample
has 12 emotion sliders and no worry item, so no second-sample test exists.

Run:  python frame_worry_multilevel.py [--workers 12] [--quick]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import mean

import numpy as np

import empirical_rebuild as er

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reviews" / "frame_worry_multilevel_results.md"


def _valence_no_worry(row):
    pos = [row[k] for k in ("cheerful", "relaxed") if row[k] is not None]
    neg = [row[k] for k in ("fearful", "sad") if row[k] is not None]
    if not pos or not neg:
        return None
    return mean(pos) - mean(neg)


_VALENCE_ORIG = er._valence


def load(exclude_worry):
    er._valence = _valence_no_worry if exclude_worry else _VALENCE_ORIG
    parts = er.load_participants()
    er._valence = _VALENCE_ORIG
    return parts


def _drive_one(args):
    pid, seq, variant, seed = args
    return pid, er.drive(seq, variant, seed=seed)["frame_future"]


def collect(parts, pids, variant, workers):
    jobs = [(pid, parts[pid], variant, 100 + i) for i, pid in enumerate(pids)]
    out = {}
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for pid, ff in ex.map(_drive_one, jobs, chunksize=4):
                out[pid] = ff
    else:
        for j in jobs:
            pid, ff = _drive_one(j)
            out[pid] = ff
    return out


def _cluster_robust_slope(y, x, covs, groups):
    """OLS of y on [1, x, covs]; returns slope on x with participant-clustered
    sandwich SE (Liang-Zeger, small-sample factor G/(G-1))."""
    X = np.column_stack([np.ones(len(y)), x] + covs)
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    gids = np.unique(groups)
    for g in gids:
        m = groups == g
        u = X[m].T @ resid[m]
        meat += np.outer(u, u)
    G = len(gids)
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1))
    se = math.sqrt(V[1, 1])
    return float(beta[1]), float(se), float(beta[1] / se)


def stats(parts, ff_by_pid, pids):
    ff, ww, vv, ee, pp = [], [], [], [], []
    for pid in pids:
        for i, b in enumerate(parts[pid]):
            if b["w"] is None:
                continue
            ff.append(ff_by_pid[pid][i]); ww.append(b["w"])
            vv.append(b["v"]); ee.append(b["e"] if b["e"] is not None else 0.0)
            pp.append(pid)
    ff, ww, vv, ee = map(lambda a: np.array(a, float), (ff, ww, vv, ee))
    pp = np.array(pp)
    r = float(np.corrcoef(ff, ww)[0, 1])
    X = np.column_stack([np.ones(len(ff)), vv, ee])

    def resid(y):
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ c
    partial_r = float(np.corrcoef(resid(ff), resid(ww))[0, 1])
    ffd, wwd = ff.copy(), ww.copy()
    for pid in np.unique(pp):
        m = pp == pid
        ffd[m] -= ffd[m].mean(); wwd[m] -= wwd[m].mean()
    within_r = float(np.corrcoef(ffd, wwd)[0, 1])
    # standardised slopes with cluster-robust SEs
    zf = (ff - ff.mean()) / ff.std(); zw = (ww - ww.mean()) / ww.std()
    b_raw, se_raw, z_raw = _cluster_robust_slope(zw, zf, [], pp)
    b_adj, se_adj, z_adj = _cluster_robust_slope(
        zw, zf, [(vv - vv.mean()) / vv.std(), (ee - ee.mean()) / (ee.std() + 1e-12)], pp)
    # within-person slope with cluster-robust SE (participant fixed effects via demeaning)
    zfd = ffd / (ffd.std() + 1e-12); zwd = wwd / (wwd.std() + 1e-12)
    b_w, se_w, z_w = _cluster_robust_slope(zwd, zfd, [], pp)
    per = []
    for pid in np.unique(pp):
        m = pp == pid
        if m.sum() > 5 and ff[m].std() > 1e-9 and ww[m].std() > 1e-9:
            per.append(np.corrcoef(ff[m], ww[m])[0, 1])
    per = np.array(per)
    return dict(n=len(ff), n_participants=len(np.unique(pp)), r=r,
                partial_r=partial_r, within_r=within_r,
                slope_raw=b_raw, se_raw=se_raw, z_raw=z_raw,
                slope_adj=b_adj, se_adj=se_adj, z_adj=z_adj,
                slope_within=b_w, se_within=se_w, z_within=z_w,
                per_participant_mean=float(per.mean()),
                per_participant_se=float(per.std(ddof=1) / math.sqrt(len(per))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    parts_with = load(exclude_worry=False)
    parts_without = load(exclude_worry=True)
    pids = sorted(set(parts_with) & set(parts_without))
    if args.quick:
        pids = pids[:30]
    rows = []
    for variant in ("full", "ungated"):
        for label, parts in (("with worry in input", parts_with),
                             ("worry removed from input", parts_without)):
            ff = collect(parts, pids, variant, args.workers)
            s = stats(parts, ff, pids)
            s.update(variant=variant, input=label)
            rows.append(s)
            print(variant, label, {k: (round(v, 3) if isinstance(v, float) else v)
                                   for k, v in s.items()})

    lines = ["# Future-frame posterior vs worry: honest re-analysis (2026-09-27)", "",
             "Script: `frame_worry_multilevel.py`. Geschwind ESM only (the replication "
             "sample has no worry item). Cluster-robust z = standardised slope / "
             "participant-clustered sandwich SE.", "",
             "| model | input | n beeps | n pp | raw r | partial r (v_t, e_t) | within-person r | "
             "z raw | z adjusted | z within | per-pp mean r (SE) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in rows:
        lines.append(
            f"| {s['variant']} | {s['input']} | {s['n']} | {s['n_participants']} | "
            f"{s['r']:.3f} | {s['partial_r']:.3f} | {s['within_r']:.3f} | "
            f"{s['z_raw']:.2f} | {s['z_adj']:.2f} | {s['z_within']:.2f} | "
            f"{s['per_participant_mean']:.3f} ({s['per_participant_se']:.3f}) |")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
