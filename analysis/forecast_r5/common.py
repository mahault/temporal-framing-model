"""Shared data, folds and scoring for the round-5 forecasting comparison.

Same participant folds (seed-0 shuffle, five folds), preprocessing, records and
horizons as esm_eval_v3.py and fit_v2.py, so every predictor here is scored on
exactly the rows that v2.1 was scored on.
"""
from __future__ import annotations

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import empirical_rebuild as er  # noqa: E402
from esm_eval_v3 import HORIZONS, add_time_features, build_records, load_osf, r2  # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
SAMPLES = ("Geschwind", "osf_83cfk")
N_BOOT = 1000


def folds_of(pids):
    rng = np.random.RandomState(0)
    order = list(pids)
    rng.shuffle(order)
    return {p: k % 5 for k, p in enumerate(order)}


def load_sample(name):
    if name == "Geschwind":
        parts = er.load_participants()
        add_time_features(parts, True)
        return parts, True
    parts = load_osf()
    add_time_features(parts, False)
    return parts, False


def records(parts, has_event):
    recs = build_records(parts, has_event)
    for r in recs:
        seq = parts[r["pid"]]
        for h in HORIZONS:
            j = r["i"] + h
            ok = j < len(seq) and seq[j].get("p") == seq[r["i"]].get("p")
            r[f"tt{h}"] = seq[j].get("tod", 0.5) if ok else 0.5
            r[f"ft{h}"] = seq[j].get("first", 0.0) if ok else 0.0
        # running person statistics from every beep up to and including i
        past = np.array([b["v"] for b in seq[: r["i"] + 1]], float)
        seg = np.array([b["v"] for b in seq[: r["i"] + 1] if b.get("p") == seq[r["i"]].get("p")], float)
        r["pm_all"], r["ps_all"], r["pn_all"] = float(past.mean()), float(past.std()), len(past)
        r["pm_seg"], r["ps_seg"], r["pn_seg"] = float(seg.mean()), float(seg.std()), len(seg)
    return recs


def load_v21_rows(name, variant, pids):
    """v2.1 held-out predictions saved by fit_v2.py (protocol P): {(pid, i): (yhat[6], S[6])}."""
    out = {}
    for k in range(5):
        f = ROOT / "reviews" / f"v2_rows_{name}_{variant}_fold{k}.npz"
        if not f.exists():
            return None
        z = np.load(f)
        keys = {kk.split("__")[0] for kk in z.files}
        for p in keys:
            ii, yh, S = z[f"{p}__i"], z[f"{p}__yhat"], z[f"{p}__S"]
            for j, i in enumerate(ii):
                out[(p, int(i))] = (yh[j], S[j])
    return out


def save(obj, name):
    with open(OUT / name, "wb") as fh:
        pickle.dump(obj, fh)


def load(name):
    with open(OUT / name, "rb") as fh:
        return pickle.load(fh)


# ── scoring ─────────────────────────────────────────────────────────────
def rows_by_pid(recs, oof_h, h):
    """{pid: array (n, 2) of y, yhat} over records that have a target at h and a prediction."""
    by = defaultdict(list)
    for r in recs:
        y = r[f"y{h}"]
        key = (r["pid"], r["i"])
        if y is None or key not in oof_h:
            continue
        by[r["pid"]].append((y, oof_h[key]))
    return {p: np.array(v, float) for p, v in by.items()}


def common_keys(recs, oofs, h):
    keys = None
    for o in oofs:
        ks = set(o[h].keys())
        keys = ks if keys is None else keys & ks
    valid = {(r["pid"], r["i"]) for r in recs if r[f"y{h}"] is not None}
    return keys & valid


def r2_of(recs, oof, h, keys=None):
    ys, ps = [], []
    for r in recs:
        key = (r["pid"], r["i"])
        if r[f"y{h}"] is None or key not in oof[h] or (keys is not None and key not in keys):
            continue
        ys.append(r[f"y{h}"])
        ps.append(oof[h][key])
    return r2(np.array(ps), np.array(ys)), len(ys)


def boot_r2_diff(recs, oa, ob, h, keys, seed=0):
    """Participant bootstrap of pooled R2(a) - R2(b) on the shared rows."""
    by = defaultdict(list)
    for r in recs:
        key = (r["pid"], r["i"])
        if key in keys:
            by[r["pid"]].append((r[f"y{h}"], oa[h][key], ob[h][key]))
    by = {p: np.array(v, float) for p, v in by.items()}
    pids = sorted(by)

    def diff(sel):
        arr = np.concatenate([by[p] for p in sel])
        return r2(arr[:, 1], arr[:, 0]) - r2(arr[:, 2], arr[:, 0])

    point = diff(pids)
    rng = np.random.RandomState(seed)
    bs = np.array([diff(rng.choice(pids, len(pids), replace=True)) for _ in range(N_BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def nll_rows(recs, oof, var, h, keys):
    """Per-row Gaussian NLL with variance var[h][key] (a predictor's own or leave-fold-out residual)."""
    by = defaultdict(list)
    for r in recs:
        key = (r["pid"], r["i"])
        if key in keys:
            s = max(var[h][key], 1e-6)
            by[r["pid"]].append(0.5 * np.log(2 * np.pi * s) + (r[f"y{h}"] - oof[h][key]) ** 2 / (2 * s))
    return {p: np.array(v) for p, v in by.items()}


def lfo_var(recs, oof, h, fold):
    """Leave-fold-out residual variance: rows of fold k get the residual variance of the same
    predictor on the other four folds' held-out rows (uniform across predictors; no fold-k data)."""
    res = defaultdict(list)
    for r in recs:
        key = (r["pid"], r["i"])
        if r[f"y{h}"] is not None and key in oof[h]:
            res[fold[r["pid"]]].append(r[f"y{h}"] - oof[h][key])
    allk = {k: np.array(v) for k, v in res.items()}
    vk = {k: float(np.var(np.concatenate([allk[j] for j in allk if j != k]))) for k in allk}
    return {key: vk[fold[key[0]]] for key in oof[h]}


def boot_mean_diff(a, b, seed=0):
    """Participant bootstrap of mean(a) - mean(b), rows pooled; a, b: {pid: array} on the same rows."""
    pids = sorted(set(a) & set(b))

    def diff(sel):
        return float(np.concatenate([a[p] for p in sel]).mean() - np.concatenate([b[p] for p in sel]).mean())

    point = diff(pids)
    rng = np.random.RandomState(seed)
    bs = np.array([diff(rng.choice(pids, len(pids), replace=True)) for _ in range(N_BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return point, float(lo), float(hi)
