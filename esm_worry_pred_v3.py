"""Round 3, joint target: does the model help predict the next-beep WORRY item (Geschwind)?

Worry is never fed to the model here (valence composite without the worried
item, as in the round-2 worry test). Baseline: ridge on six lags of worry, six
lags of the valence composite, event pleasantness with two lags, time of day.
Augmented: the same plus the model's state features at the current beep
(h-step valence expectation, three channels, q(f), state entropy, effective
rho_pos), driven with the pooled parameters selected in esm_eval_v3. Same
participant folds as esm_eval_v3 (seed-0 shuffle, 5 folds), ridge penalty by
inner participant-grouped CV, participant bootstrap CIs.

Writes reviews/esm_worry_pred_v3.md.  Run: python esm_worry_pred_v3.py --workers 20
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import empirical_rebuild as er
from empirical_rebuild import same_segment
from frame_worry_multilevel import _valence_no_worry
from esm_eval_v3 import (COMBOS, HORIZONS, NLAG, Ridge, add_time_features, bootstrap_diff, drive,
                         model_feats, pick_lambda, r2)

ROOT = Path(__file__).resolve().parent
PARAMS = dict(pi_pos=3.0, valence_inertia=0.65, omega_e=3.0, asym=(0.6, 1.6))


def _job(a):
    pid, seq, seed = a
    return pid, drive(seq, PARAMS, "full", seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    orig = er._valence
    er._valence = _valence_no_worry
    parts = er.load_participants()
    er._valence = orig
    add_time_features(parts, True)
    pids = sorted(parts)
    jobs = [(pid, parts[pid], 500 + i) for i, pid in enumerate(pids)]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        driven = dict(ex.map(_job, jobs, chunksize=4))

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
    rng = np.random.RandomState(0)
    order = list(pids)
    rng.shuffle(order)
    fold = {p: k % 5 for k, p in enumerate(order)}

    def base(r):
        return [r["w"], r["v"]] + r["wl"] + r["vl"] + r["miss"] + [r["e"], max(r["e"], 0.0)] + r["el"] + [r["tod"], r["tod"] ** 2, r["first"]]

    def aug(r, h, keys):
        mf = model_feats(driven[r["pid"]][r["i"]], h)
        return base(r) + [mf[k] for k in keys]

    FE = ["pred", "v_model", "v_reward", "v_action", "qf_past", "qf_pres", "qf_fut", "arousal", "rho_eff"]
    kinds = {"persistence": None, "base": lambda r, h: base(r), "aug": lambda r, h: aug(r, h, FE),
             "aug_frame_only": lambda r, h: aug(r, h, ["qf_past", "qf_pres", "qf_fut"]),
             "aug_noframe": lambda r, h: aug(r, h, [k for k in FE if not k.startswith("qf")])}
    by = {nm: {h: defaultdict(list) for h in HORIZONS} for nm in kinds}
    for k in range(5):
        tr = [r for r in recs if fold[r["pid"]] != k]
        te = [r for r in recs if fold[r["pid"]] == k]
        for h in HORIZONS:
            trh = [r for r in tr if r[f"y{h}"] is not None]
            teh = [r for r in te if r[f"y{h}"] is not None]
            ytr = np.array([r[f"y{h}"] for r in trh])
            for nm, f in kinds.items():
                if f is None:
                    p = np.array([r["w"] for r in teh])
                else:
                    Xtr = np.array([f(r, h) for r in trh])
                    Xte = np.array([f(r, h) for r in teh])
                    p = Ridge(pick_lambda(Xtr, ytr, [r["pid"] for r in trh])).fit(Xtr, ytr).predict(Xte)
                for r, pv in zip(teh, p):
                    by[nm][h][r["pid"]].append((r[f"y{h}"], float(pv)))
    lines = ["# Worry prediction, round 3 (2026-09-28)", "",
             "Target: worried item h beeps ahead (Geschwind, worry never fed to the model). Baseline: ridge on six "
             "worry lags, six valence lags, events, time of day. Augmented: plus the model's state features. "
             f"Pooled parameters {PARAMS}. 5 participant folds, participant bootstrap CIs.", "",
             "| predictor | " + " | ".join(f"h={h} R2" for h in HORIZONS) + " |", "|---|" + "---:|" * len(HORIZONS)]
    out = {}
    for nm in kinds:
        row = []
        for h in HORIZONS:
            arr = np.concatenate([np.array(v) for v in by[nm][h].values()])
            row.append(r2(arr[:, 1], arr[:, 0]))
        out[nm] = row
        lines.append(f"| {nm} | " + " | ".join(f"{x:.3f}" for x in row) + " |")
    lines.append("")
    for h in HORIZONS:
        for a, b in (("aug", "base"), ("aug_frame_only", "base"), ("aug_noframe", "base")):
            both = {p: np.column_stack([np.array(by[a][h][p])[:, 0], np.array(by[a][h][p])[:, 1], np.array(by[b][h][p])[:, 1]])
                    for p in by[a][h]}
            d = bootstrap_diff(both, a, b)
            lines.append(f"- h={h}: {a} - {b}: {d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}], P(<=0)={d['p_le_0']:.3f}")
    lines.append(f"\nrows at h=1: {sum(len(v) for v in by['base'][1].values())}")
    (ROOT / "reviews" / "esm_worry_pred_v3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
