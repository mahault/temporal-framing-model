"""Unified ESM evaluation, round 2 (2026-09-27).

Why a new script. The Geschwind loader used to interleave the two six-day
sampling periods (one row per (participant, day, beep) for each period), so
the "previous beep" of the pipeline was the same slot two months earlier and
the true previous beep sat at lag 2. That is what produced a two-step baseline
R2 three times the one-step value. ``empirical_rebuild.load_participants`` now
orders by (period, day, beep), the agent is reset at the period boundary, and
no prediction target crosses it. Every ESM number in the paper is recomputed
here, on both samples, with one protocol:

  * nested parameter selection: the model's three global parameters
    (rho_pos, valence inertia, omega_e) are chosen on the training participants
    of each fold from an 18-point grid (highest mean held-in R2 over horizons
    of the frame-gated model), then every variant is evaluated with those
    parameters on the held-out participants;
  * 5-fold participant CV, train-fit affine calibration for every predictor;
  * baselines: persistence, iterated AR(1), direct h-step regression on v_t,
    direct h-step regression on (v_t, v_{t-1}), and on Geschwind the linear
    event and asymmetric event models;
  * model variants: frame-gated readout (g=1), frame-inert (g=0), gating
    posterior clamped to PAST / PRESENT / FUTURE, transition-gated
    (g=1 plus g_B in {1, 4}: the frame also sets valence persistence),
    no-inertia (g=1, inertia 0), and a channels-only regression;
  * participant-bootstrap 95 percent CIs on R2 differences;
  * the frame-worry test with worry removed from the model input, for the
    gated, inert and transition-gated models.

Writes reviews/esm_eval_v2.md and reviews/esm_eval_v2.json.
Run:  python esm_eval_v2.py --workers 22
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from agent import Agent
from generative_model import EPS, N_FRAMES, build_model
import empirical_rebuild as er
from empirical_rebuild import _bin_e, _bin_v, HORIZONS, target, same_segment
from esm_replication import load_participants as load_osf
from frame_worry_multilevel import stats as worry_stats, _valence_no_worry

ROOT = Path(__file__).resolve().parent
GRID = dict(pi_pos=[2.0, 3.0, 4.0], valence_inertia=[0.2, 0.35, 0.5], omega_e=[3.0, 5.0])
COMBOS = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
VARIANTS = {
    "full": dict(g=1.0, gt=0.0, clamp=None),
    "ungated": dict(g=0.0, gt=0.0, clamp=None),
    "clamp_present": dict(g=1.0, gt=0.0, clamp=1),
    "clamp_past": dict(g=1.0, gt=0.0, clamp=0),
    "clamp_future": dict(g=1.0, gt=0.0, clamp=2),
    "trans_g1": dict(g=1.0, gt=1.0, clamp=None),
    "trans_g4": dict(g=1.0, gt=4.0, clamp=None),
    "noinertia": dict(g=1.0, gt=0.0, clamp=None, inertia=0.0),
}
N_BOOT = 2000
K = M = 8


def _lin(X, y):
    coef, *_ = np.linalg.lstsq(np.asarray(X, float), np.asarray(y, float), rcond=None)
    return coef


def _r2(p, y):
    y = np.asarray(y, float)
    sst = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if sst < EPS else 1 - float(np.sum((y - np.asarray(p, float)) ** 2)) / sst


def drive(seq, params, variant, seed):
    spec = VARIANTS[variant]
    inertia = spec.get("inertia", params["valence_inertia"])
    model = build_model(K=K, M=M, pi_pos=params["pi_pos"], omega_e=params["omega_e"],
                        gamma=16.0, c_pos=1.0, c_neg=1.0, neg_val_precision=1.0,
                        valence_inertia=inertia)
    agent = Agent(model, gamma=16.0, pi_pos=params["pi_pos"], omega_e=params["omega_e"],
                  c_pos=1.0, c_neg=1.0, neg_val_precision=1.0, valence_inertia=inertia,
                  counterfactual_horizon=1, adaptive_counterfactual_horizon=False,
                  frame_gain=spec["g"], frame_clamp=spec["clamp"],
                  frame_transition_gain=spec["gt"], seed=seed)
    v_axis = np.arange(K)
    preds = {h: [] for h in HORIZONS}
    ch, ff = [], []
    prev = None
    for beep in seq:
        if prev is not None and beep.get("p") != prev:
            agent.reset()
        prev = beep.get("p")
        _, info = agent.step([_bin_e(beep["e"]), 1, _bin_v(beep["v"], K)])
        ch.append((info["v_model"], info["v_reward"], info["v_action"]))
        ff.append(float(info["beliefs"].reshape(K, M, N_FRAMES).sum(axis=(0, 1))[2]))
        pi = info["pi"]
        B = sum(pi[a] * model.B[a] for a in range(len(pi)))
        q = info["beliefs"].copy()
        for h in range(1, max(HORIZONS) + 1):
            q = B @ q
            q = np.maximum(q, EPS)
            q /= q.sum()
            if h in preds:
                vm = q.reshape(K, M, N_FRAMES).sum(axis=(1, 2))
                preds[h].append(float(vm @ v_axis / (K - 1)))
    return preds, ch, ff


def _job(a):
    pid, seq, gi, variant, seed = a
    p, ch, ff = drive(seq, COMBOS[gi], variant, seed)
    return pid, gi, variant, p, ch, ff


def run_jobs(jobs, workers):
    out = {}
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for pid, gi, v, p, ch, ff in ex.map(_job, jobs, chunksize=4):
                out[(pid, gi, v)] = (p, ch, ff)
    else:
        for j in jobs:
            pid, gi, v, p, ch, ff = _job(j)
            out[(pid, gi, v)] = (p, ch, ff)
    return out


def descriptives(parts, has_event):
    pids = sorted(parts)
    recs = sum(len(parts[p]) for p in pids)
    segs = sum(len({b.get("p") for b in parts[p]}) for p in pids)
    days = [len({(b.get("p"), b.get("d")) for b in parts[p]}) for p in pids]
    n_targets = {h: sum(1 for p in pids for i in range(len(parts[p])) if target(parts[p], i, h) is not None)
                 for h in HORIZONS}
    d = dict(participants=len(pids), records=recs, segments=segs,
             days_median=float(np.median(days)), days_min=int(min(days)), days_max=int(max(days)),
             beeps_per_participant_median=float(np.median([len(parts[p]) for p in pids])),
             targets={int(h): int(n) for h, n in n_targets.items()})
    if has_event:
        d["with_worry"] = sum(1 for p in pids for b in parts[p] if b["w"] is not None)
        d["with_event"] = sum(1 for p in pids for b in parts[p] if b["e"] is not None)
        d["with_neuroticism"] = len([p for p in pids if parts[p][0].get("n") is not None])
    return d


def evaluate(name, parts, workers, has_event):
    pids = sorted(parts)
    seeds = {pid: 500 + i for i, pid in enumerate(pids)}
    # stage 1: the gated model on every grid point
    jobs = [(pid, parts[pid], gi, "full", seeds[pid]) for pid in pids for gi in range(len(COMBOS))]
    print(f"[{name}] stage 1: {len(jobs)} drives")
    driven = run_jobs(jobs, workers)

    # records
    recs = []
    for pid in pids:
        seq = parts[pid]
        for i in range(len(seq)):
            r = dict(pid=pid, v_t=seq[i]["v"], e_t=(seq[i]["e"] if seq[i]["e"] is not None else 0.0),
                     w_t=seq[i].get("w"), _i=len(recs),
                     v_prev=(seq[i - 1]["v"] if i > 0 and same_segment(seq, i - 1, i) else None))
            for h in HORIZONS:
                r[f"y{h}"] = target(seq, i, h)
            recs.append(r)
    rng = np.random.RandomState(0)
    order = list(pids)
    rng.shuffle(order)
    fold = {p: k % 5 for k, p in enumerate(order)}

    # per-fold selection of the grid point on training participants
    first = {}
    for r in recs:
        first.setdefault(r["pid"], r["_i"])
    selected = []
    for k in range(5):
        tr = [r for r in recs if fold[r["pid"]] != k]
        best, best_gi = -np.inf, None
        for gi in range(len(COMBOS)):
            score = []
            for h in HORIZONS:
                rows = [r for r in tr if r[f"y{h}"] is not None]
                x = [driven[(r["pid"], gi, "full")][0][h][r["_i"] - first[r["pid"]]] for r in rows]
                y = [r[f"y{h}"] for r in rows]
                c = _lin([[1, v] for v in x], y)
                score.append(_r2(c[0] + c[1] * np.array(x), y))
            m = float(np.mean(score))
            if m > best:
                best, best_gi = m, gi
        selected.append(best_gi)
    print(f"[{name}] selected grid points per fold: {[COMBOS[g] for g in selected]}")

    # stage 2: every variant at the selected grid points
    need = sorted(set(selected))
    jobs = [(pid, parts[pid], gi, v, seeds[pid]) for pid in pids for gi in need
            for v in VARIANTS if v != "full"]
    print(f"[{name}] stage 2: {len(jobs)} drives")
    driven.update(run_jobs(jobs, workers))

    names = ["persistence", "ar1_iter", "direct_v", "direct_v2"] \
        + (["linear_event", "linear_event_asym"] if has_event else []) \
        + [f"model_{v}" for v in VARIANTS] + ["channels_only"]
    r2 = {nm: {h: [] for h in HORIZONS} for nm in names}
    oof = {nm: {h: {} for h in HORIZONS} for nm in names}
    e_mean = float(np.mean([r["e_t"] for r in recs]))
    for k in range(5):
        gi = selected[k]
        tr = [r for r in recs if fold[r["pid"]] != k]
        te = [r for r in recs if fold[r["pid"]] == k]
        tr1 = [r for r in tr if r["y1"] is not None]
        ca = _lin([[1, r["v_t"]] for r in tr1], [r["y1"] for r in tr1])
        if has_event:
            ce = _lin([[1, r["v_t"], r["e_t"]] for r in tr1], [r["y1"] for r in tr1])
            cas = _lin([[1, r["v_t"], max(r["e_t"], 0), min(r["e_t"], 0)] for r in tr1],
                       [r["y1"] for r in tr1])
        for h in HORIZONS:
            trh = [r for r in tr if r[f"y{h}"] is not None]
            teh = [r for r in te if r[f"y{h}"] is not None]
            yte = np.array([r[f"y{h}"] for r in teh])
            v_te = np.array([r["v_t"] for r in teh], float)
            e_te = np.array([r["e_t"] for r in teh], float)
            preds = {"persistence": v_te.copy()}
            pa = v_te.copy()
            for _ in range(h):
                pa = ca[0] + ca[1] * pa
            preds["ar1_iter"] = pa
            cdv = _lin([[1, r["v_t"]] for r in trh], [r[f"y{h}"] for r in trh])
            preds["direct_v"] = cdv[0] + cdv[1] * v_te
            tr2 = [r for r in trh if r["v_prev"] is not None]
            c2 = _lin([[1, r["v_t"], r["v_prev"]] for r in tr2], [r[f"y{h}"] for r in tr2])
            preds["direct_v2"] = np.array([
                c2[0] + c2[1] * r["v_t"] + c2[2] * r["v_prev"] if r["v_prev"] is not None
                else cdv[0] + cdv[1] * r["v_t"] for r in teh])
            if has_event:
                p = v_te.copy()
                for step in range(h):
                    p = ce[0] + ce[1] * p + ce[2] * (e_te if step == 0 else e_mean)
                preds["linear_event"] = p
                p = v_te.copy()
                for step in range(h):
                    ee = e_te if step == 0 else np.full(len(teh), e_mean)
                    p = cas[0] + cas[1] * p + cas[2] * np.maximum(ee, 0) + cas[3] * np.minimum(ee, 0)
                preds["linear_event_asym"] = p
            for v in VARIANTS:
                mx_tr = [driven[(r["pid"], gi, v)][0][h][r["_i"] - first[r["pid"]]] for r in trh]
                mx_te = [driven[(r["pid"], gi, v)][0][h][r["_i"] - first[r["pid"]]] for r in teh]
                cm = _lin([[1, x] for x in mx_tr], [r[f"y{h}"] for r in trh])
                preds[f"model_{v}"] = cm[0] + cm[1] * np.array(mx_te)

            def cfeat(r):
                c1, c2_, c3 = driven[(r["pid"], gi, "full")][1][r["_i"] - first[r["pid"]]]
                return [1, r["v_t"], c1, c2_, c3]
            cc = _lin([cfeat(r) for r in trh], [r[f"y{h}"] for r in trh])
            preds["channels_only"] = np.asarray([cfeat(r) for r in teh], float) @ cc
            for nm, p in preds.items():
                r2[nm][h].append(_r2(p, yte))
                for r, pv in zip(teh, p):
                    oof[nm][h][r["_i"]] = float(pv)
    return recs, r2, oof, names, selected, driven, first


def _first_index(recs, pid):
    for r in recs:
        if r["pid"] == pid:
            return r["_i"]
    raise KeyError(pid)


def bootstrap_diff(recs, oof, a, b, h, seed=0):
    rows = [(r["pid"], r[f"y{h}"], oof[a][h][r["_i"]], oof[b][h][r["_i"]])
            for r in recs if r[f"y{h}"] is not None]
    pids = sorted({p for p, *_ in rows})
    by = {p: [] for p in pids}
    for p, y, x, z in rows:
        by[p].append((y, x, z))
    by = {p: np.array(v, float) for p, v in by.items()}

    def diff(sel):
        arr = np.concatenate([by[p] for p in sel])
        return _r2(arr[:, 1], arr[:, 0]) - _r2(arr[:, 2], arr[:, 0])
    point = diff(pids)
    rng = np.random.RandomState(seed)
    boots = np.array([diff(rng.choice(pids, len(pids), replace=True)) for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(point=float(point), lo=float(lo), hi=float(hi), p_le_0=float(np.mean(boots <= 0)))


def report(name, recs, r2, oof, names, selected, lines, out):
    base = [n for n in names if not n.startswith("model_") and n != "channels_only"]
    lines.append(f"## {name} (n={len({r['pid'] for r in recs})} participants, {len(recs)} records)")
    lines.append("")
    lines.append("Selected parameters per fold: " + "; ".join(
        f"fold {k}: rho_pos {COMBOS[g]['pi_pos']}, inertia {COMBOS[g]['valence_inertia']}, "
        f"omega_e {COMBOS[g]['omega_e']}" for k, g in enumerate(selected)))
    lines.append("")
    lines.append("| predictor | " + " | ".join(f"h={h} R2 (fold SD)" for h in HORIZONS) + " |")
    lines.append("|---|" + "---:|" * len(HORIZONS))
    table = {}
    for nm in names:
        row = {}
        for h in HORIZONS:
            row[int(h)] = (float(np.nanmean(r2[nm][h])), float(np.nanstd(r2[nm][h])))
        table[nm] = row
        lines.append(f"| {nm} | " + " | ".join(f"{row[h][0]:.3f} ({row[h][1]:.3f})" for h in HORIZONS) + " |")
    lines.append("")
    lines.append(f"Participant-bootstrap 95% CIs on R2 differences ({N_BOOT} resamples):")
    lines.append("")
    cis = {}
    for h in HORIZONS:
        best = max(base, key=lambda nm: np.nanmean(r2[nm][h]))
        pairs = [(f"full - best baseline ({best})", "model_full", best),
                 ("full - direct (v_t, v_t-1)", "model_full", "direct_v2"),
                 ("full - frame-inert", "model_full", "model_ungated"),
                 ("full - clamped FUTURE", "model_full", "model_clamp_future"),
                 ("transition-gated g_B=4 - frame-inert", "model_trans_g4", "model_ungated"),
                 ("transition-gated g_B=1 - frame-inert", "model_trans_g1", "model_ungated"),
                 ("full - channels only", "model_full", "channels_only"),
                 ("full - no inertia", "model_full", "model_noinertia"),
                 ("no inertia - best baseline", "model_noinertia", best)]
        cis[int(h)] = {}
        for label, a, b in pairs:
            d = bootstrap_diff(recs, oof, a, b, h)
            cis[int(h)][label] = d
            lines.append(f"- h={h}: {label}: {d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}], "
                         f"P(diff<=0)={d['p_le_0']:.3f}")
    lines.append("")
    out[name] = dict(table=table, cis=cis, selected=[COMBOS[g] for g in selected])


def worry_analysis(workers, params, lines, out):
    """Frame-worry association, worry removed from the model input."""
    orig = er._valence
    er._valence = _valence_no_worry
    parts = er.load_participants()
    er._valence = orig
    pids = sorted(parts)
    gi = COMBOS.index(params)
    rows = {}
    for v in ("full", "ungated", "trans_g4", "clamp_future"):
        jobs = [(pid, parts[pid], gi, v, 500 + i) for i, pid in enumerate(pids)]
        d = run_jobs(jobs, workers)
        ff = {pid: d[(pid, gi, v)][2] for pid in pids}
        s = worry_stats(parts, ff, pids)
        rows[v] = s
    lines.append("## Frame-worry test, worry removed from the input (Geschwind)")
    lines.append("")
    lines.append(f"Parameters: {params}. Partial r controls for concurrent valence and event; "
                 "z = standardised slope / participant-clustered sandwich SE.")
    lines.append("")
    lines.append("| model | n beeps | raw r | partial r | within-person r | z raw | z adjusted | z within |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for v, s in rows.items():
        lines.append(f"| {v} | {s['n']} | {s['r']:.3f} | {s['partial_r']:.3f} | {s['within_r']:.3f} | "
                     f"{s['z_raw']:.2f} | {s['z_adj']:.2f} | {s['z_within']:.2f} |")
    lines.append("")
    out["worry"] = {v: {k: (float(x) if isinstance(x, (float, np.floating)) else x) for k, x in s.items()}
                    for v, s in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    out = {"grid": COMBOS, "variants": VARIANTS}
    lines = ["# Unified ESM evaluation, round 2 (2026-09-27)", "",
             "Script: `esm_eval_v2.py`. Geschwind sequences ordered by (period, day, beep), "
             "agent reset at the period boundary, no target crosses it. Nested selection of "
             "(rho_pos, inertia, omega_e) on training participants per fold; 5-fold participant "
             "CV; train-fit affine calibration for every predictor; rollout off (horizon 1), "
             "policy-averaged predictive transition.", ""]
    g = er.load_participants()
    out["geschwind_descriptives"] = descriptives(g, True)
    lines.append("Geschwind descriptives: " + json.dumps(out["geschwind_descriptives"]))
    recs, r2, oof, names, selected, driven, first = evaluate("Geschwind", g, args.workers, True)
    report("Geschwind remitted-depression ESM", recs, r2, oof, names, selected, lines, out)
    modal = Counter(selected).most_common(1)[0][0]
    o = load_osf()
    out["osf_descriptives"] = descriptives(o, False)
    lines.append("osf_83cfk descriptives: " + json.dumps(out["osf_descriptives"]))
    recs, r2, oof, names, selected, driven, first = evaluate("osf_83cfk", o, args.workers, False)
    report("Reliability ESM (osf.io/83cfk)", recs, r2, oof, names, selected, lines, out)
    worry_analysis(args.workers, COMBOS[modal], lines, out)
    (ROOT / "reviews").mkdir(exist_ok=True)
    (ROOT / "reviews" / "esm_eval_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "reviews" / "esm_eval_v2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
