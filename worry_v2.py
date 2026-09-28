"""Frame-sensitive test (d): worry as a secondary target with v2 state features
(Geschwind, worry never fed to the model), 2026-09-28.

Mirror of esm_worry_pred_v3.py with model v2 in place of the discrete model.
v2 (gated) is fitted per fold on the training participants using the
worry-free valence composite, and its held-out state at each beep (fast state,
mood, frame posterior, three channels, one-step predictive variance, h-step
expectation) is appended to the ridge baseline (six worry lags, six valence
lags, events with two lags, time of day).

Writes reviews/worry_v2.md.  Run: python worry_v2.py --workers 5
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import empirical_rebuild as er
from empirical_rebuild import same_segment
from frame_worry_multilevel import _valence_no_worry
from esm_eval_v3 import HORIZONS, NLAG, Ridge, add_time_features, bootstrap_diff, pick_lambda, r2
from fit_v2 import folds_of
from model_v2 import ModelV2, empirical_bayes_lambda, fit, make_tensors, subset

ROOT = Path(__file__).resolve().parent
FE = ["pred", "x", "m", "q_past", "q_pres", "q_fut", "vB", "vP", "vF", "S1", "rho"]


def _fit_fold(a):
    k, parts, pids, fold, iters = a
    torch.set_num_threads(2)
    dt = make_tensors(parts, pids, True)
    tr = np.array([i for i, p in enumerate(pids) if fold[p] != k])
    te = np.array([i for i, p in enumerate(pids) if fold[p] == k])
    model = ModelV2(len(pids), True, g=1.0)
    fit(model, subset(dt, tr), tr, iters=iters, lam=10.0, seed=k)
    lam = empirical_bayes_lambda(model, tr)
    fit(model, subset(dt, tr), tr, iters=iters // 2, lam=lam, seed=k + 100)
    model.eval()
    with torch.no_grad():
        res = model(subset(dt, te), torch.as_tensor(te))
    feats = {}
    for n, i in enumerate(te):
        p = pids[i]
        T = len(parts[p])
        feats[p] = [dict(pred=res["yhat"][n, t].numpy(), x=float(res["x"][n, t]), m=float(res["m"][n, t]),
                         q_past=float(res["q"][n, t, 0]), q_pres=float(res["q"][n, t, 1]), q_fut=float(res["q"][n, t, 2]),
                         vB=float(res["vB"][n, t]), vP=float(res["vP"][n, t]), vF=float(res["vF"][n, t]),
                         S1=float(res["S1"][n, t]), rho=float(res["rho"][n, t])) for t in range(T)]
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()
    orig = er._valence
    er._valence = _valence_no_worry
    parts = er.load_participants()
    er._valence = orig
    add_time_features(parts, True)
    pids = sorted(parts)
    fold = folds_of(pids)
    jobs = [(k, parts, pids, fold, args.iters) for k in range(5)]
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            outs = list(ex.map(_fit_fold, jobs))
    else:
        outs = [_fit_fold(j) for j in jobs]
    driven = {}
    for o in outs:
        driven.update(o)

    recs = []
    for pid in pids:
        seq = parts[pid]
        for i, b in enumerate(seq):
            if b["w"] is None:
                continue
            wl, vl, miss = [], [], []
            for L in range(1, NLAG + 1):
                j = i - L
                ok = j >= 0 and same_segment(seq, j, i) and seq[j]["w"] is not None
                wl.append(seq[j]["w"] if ok else (wl[-1] if wl else b["w"]))
                vl.append(seq[j]["v"] if ok else (vl[-1] if vl else b["v"]))
                miss.append(0.0 if ok else 1.0)
            e = b["e"] if b["e"] is not None else 0.0
            el = []
            for L in (1, 2):
                j = i - L
                ok = j >= 0 and same_segment(seq, j, i) and seq[j]["e"] is not None
                el.append(seq[j]["e"] if ok else 0.0)
            r = dict(pid=pid, i=i, w=b["w"], v=b["v"], e=e, wl=wl, vl=vl, miss=miss, el=el,
                     tod=b.get("tod", 0.5), first=b.get("first", 0.0))
            for h in HORIZONS:
                j = i + h
                r[f"y{h}"] = seq[j]["w"] if (j < len(seq) and same_segment(seq, i, j) and seq[j]["w"] is not None) else None
            recs.append(r)

    def base(r):
        return [r["w"], r["v"]] + r["wl"] + r["vl"] + r["miss"] + [r["e"], max(r["e"], 0.0)] + r["el"] + [r["tod"], r["tod"] ** 2, r["first"]]

    def mf(r, h):
        d = driven[r["pid"]][r["i"]]
        o = dict(d)
        o["pred"] = float(d["pred"][h - 1])
        return o

    def aug(r, h, keys):
        m = mf(r, h)
        return base(r) + [m[k] for k in keys]

    kinds = {"base": lambda r, h: base(r), "aug": lambda r, h: aug(r, h, FE),
             "aug_frame_only": lambda r, h: aug(r, h, ["q_past", "q_pres", "q_fut"]),
             "aug_noframe": lambda r, h: aug(r, h, [k for k in FE if not k.startswith("q_")])}
    by = {nm: {h: defaultdict(list) for h in HORIZONS} for nm in kinds}
    for k in range(5):
        tr = [r for r in recs if fold[r["pid"]] != k]
        te = [r for r in recs if fold[r["pid"]] == k]
        for h in HORIZONS:
            trh = [r for r in tr if r[f"y{h}"] is not None]
            teh = [r for r in te if r[f"y{h}"] is not None]
            ytr = np.array([r[f"y{h}"] for r in trh])
            for nm, f in kinds.items():
                Xtr = np.array([f(r, h) for r in trh])
                Xte = np.array([f(r, h) for r in teh])
                p = Ridge(pick_lambda(Xtr, ytr, [r["pid"] for r in trh])).fit(Xtr, ytr).predict(Xte)
                for r, pv in zip(teh, p):
                    by[nm][h][r["pid"]].append((r[f"y{h}"], float(pv)))
    lines = ["# Worry prediction with v2 state features (2026-09-28)", "",
             "Target: worried item h beeps ahead (Geschwind, worry never fed to the model; v2 driven by the worry-free "
             "composite). Baseline: ridge on six worry lags, six valence lags, events, time of day. Augmented: plus v2's "
             "held-out state (fast state, mood, frame posterior, three channels, one-step variance, expectation). "
             "5 participant folds, participant bootstrap CIs.", "",
             "| predictor | " + " | ".join(f"h={h} R2" for h in HORIZONS) + " |", "|---|" + "---:|" * len(HORIZONS)]
    for nm in kinds:
        row = []
        for h in HORIZONS:
            arr = np.concatenate([np.array(v) for v in by[nm][h].values()])
            row.append(r2(arr[:, 1], arr[:, 0]))
        lines.append(f"| {nm} | " + " | ".join(f"{x:.3f}" for x in row) + " |")
    lines.append("")
    for h in HORIZONS:
        for a, b in (("aug", "base"), ("aug_frame_only", "base"), ("aug_noframe", "base")):
            both = {p: np.column_stack([np.array(by[a][h][p])[:, 0], np.array(by[a][h][p])[:, 1], np.array(by[b][h][p])[:, 1]])
                    for p in by[a][h]}
            d = bootstrap_diff(both, a, b)
            lines.append(f"- h={h}: {a} - {b}: {d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}], P(<=0)={d['p_le_0']:.3f}")
    lines.append(f"\nrows at h=1: {sum(len(v) for v in by['base'][1].values())}")
    (ROOT / "reviews" / "worry_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
