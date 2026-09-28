"""Frame ablation for the ESM affect-dynamics prediction (revision 2026-09-27).

Answers AAAI-27 reviewers DoMi 6, 6A6h 3 and the AI reviewer's item 2/3: does the
inferred temporal frame contribute to the out-of-sample prediction, and is the
second-sample "replication" distinguishable from sampling variability?

Uses the paper's fitted global parameters and pipeline (eval_fitted_cv.py /
esm_replication.py: policy-averaged one-step predictive transition, rollout off,
5-fold participant CV, train-fit affine calibration), on both ESM samples, for:

  full            frame-gated model (g = 1, the revised paper model)
  ungated         g = 0, the pre-revision frame-inert model
  frame_present   gated, gating posterior clamped to PRESENT
  frame_past      gated, clamped to PAST
  frame_future    gated, clamped to FUTURE
  channels_only   affine regression on v_t and the three readout channels
                  (no state rollout)
baselines: iterated AR(1), direct h-step regression on v_t, and on Geschwind
also linear+event and linear+asymmetric-event.

Reports held-out R2 per horizon with fold SD, and participant-level bootstrap
95 percent CIs (2000 resamples) for full - best baseline, full - ungated,
full - frame_present, full - channels_only. Writes reviews/esm_frame_ablation.md.

Run:  python esm_frame_ablation.py [--workers 12]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from agent import Agent
from generative_model import EPS, N_FRAMES, build_model
from empirical_rebuild import load_participants as load_geschwind, _bin_e, _bin_v, HORIZONS, target
from esm_replication import load_participants as load_osf

ROOT = Path(__file__).resolve().parent
FITTED = dict(pi_pos=2.0, valence_inertia=0.5, omega_e=5.0, c_pos=1.0, c_neg=1.0)
VARIANTS = ("full", "ungated", "frame_present", "frame_past", "frame_future")
N_BOOT = 2000


def _lin(X, y):
    coef, *_ = np.linalg.lstsq(np.asarray(X, float), np.asarray(y, float), rcond=None)
    return coef


def _r2(p, y):
    y = np.asarray(y, float); sst = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if sst < EPS else 1 - float(np.sum((y - np.asarray(p, float)) ** 2)) / sst


def drive(seq, seed, variant):
    gain, clamp = 1.0, None
    if variant == "ungated":
        gain = 0.0
    elif variant.startswith("frame_"):
        clamp = {"frame_past": 0, "frame_present": 1, "frame_future": 2}[variant]
    K = M = 8
    model = build_model(K=K, M=M, pi_pos=FITTED["pi_pos"], omega_e=FITTED["omega_e"],
                        gamma=16.0, c_pos=FITTED["c_pos"], c_neg=FITTED["c_neg"],
                        neg_val_precision=1.0, valence_inertia=FITTED["valence_inertia"])
    agent = Agent(model, gamma=16.0, pi_pos=FITTED["pi_pos"], omega_e=FITTED["omega_e"],
                  c_pos=FITTED["c_pos"], c_neg=FITTED["c_neg"], neg_val_precision=1.0,
                  valence_inertia=FITTED["valence_inertia"],
                  counterfactual_horizon=1, adaptive_counterfactual_horizon=False,
                  frame_gain=gain, frame_clamp=clamp, seed=seed)
    v_axis = np.arange(K); preds = {h: [] for h in HORIZONS}
    ch = []
    prev_p = None
    for beep in seq:
        if prev_p is not None and beep.get("p") != prev_p:
            agent.reset()
        prev_p = beep.get("p")
        _, info = agent.step([_bin_e(beep["e"]), 1, _bin_v(beep["v"], K)])
        ch.append((info["v_model"], info["v_reward"], info["v_action"]))
        pi = info["pi"]; B = sum(pi[a] * model.B[a] for a in range(len(pi)))
        q = info["beliefs"].copy()
        for h in range(1, max(HORIZONS) + 1):
            q = B @ q; q = np.maximum(q, EPS); q /= q.sum()
            if h in preds:
                vm = q.reshape(K, M, N_FRAMES).sum(axis=(1, 2))
                preds[h].append(float(vm @ v_axis / max(K - 1, 1)))
    return preds, ch


def _job(args):
    pid, seq, seed, variant = args
    p, ch = drive(seq, seed, variant)
    return pid, variant, p, ch


def evaluate(sample, parts, workers, has_event):
    pids = sorted(parts)
    jobs = [(pid, parts[pid], 500 + i, v) for i, pid in enumerate(pids) for v in VARIANTS]
    driven = {}
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for pid, v, p, ch in ex.map(_job, jobs, chunksize=4):
                driven[(pid, v)] = (p, ch)
    else:
        for j in jobs:
            pid, v, p, ch = _job(j)
            driven[(pid, v)] = (p, ch)
    recs = []
    for pid in pids:
        seq = parts[pid]; n = len(seq)
        for idx in range(n):
            e = seq[idx]["e"] if seq[idx]["e"] is not None else 0.0
            r = {"pid": pid, "v_t": seq[idx]["v"], "e_t": e, "_i": len(recs)}
            ch = driven[(pid, "full")][1][idx]
            r["c1"], r["c2"], r["c3"] = ch
            for h in HORIZONS:
                r[f"y{h}"] = target(seq, idx, h)
                for v in VARIANTS:
                    r[f"m_{v}_{h}"] = driven[(pid, v)][0][h][idx]
            recs.append(r)
    rng = np.random.RandomState(0); order = list(pids); rng.shuffle(order)
    fold = {p: k % 5 for k, p in enumerate(order)}
    names = ["ar1_iter", "direct_v"] + (["linear_event", "linear_event_asym"] if has_event else []) \
        + [f"model_{v}" for v in VARIANTS] + ["channels_only"]
    r2 = {nm: {h: [] for h in HORIZONS} for nm in names}
    oof = {nm: {h: {} for h in HORIZONS} for nm in names}
    e_mean = float(np.mean([r["e_t"] for r in recs]))
    for k in range(5):
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
            yte = np.array([r[f"y{h}"] for r in teh]); v_te = np.array([r["v_t"] for r in teh], float)
            e_te = np.array([r["e_t"] for r in teh], float)
            preds = {}
            pa = v_te.copy()
            for _ in range(h):
                pa = ca[0] + ca[1] * pa
            preds["ar1_iter"] = pa
            cdv = _lin([[1, r["v_t"]] for r in trh], [r[f"y{h}"] for r in trh])
            preds["direct_v"] = cdv[0] + cdv[1] * v_te
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
                cm = _lin([[1, r[f"m_{v}_{h}"]] for r in trh], [r[f"y{h}"] for r in trh])
                preds[f"model_{v}"] = cm[0] + cm[1] * np.array([r[f"m_{v}_{h}"] for r in teh])
            cc = _lin([[1, r["v_t"], r["c1"], r["c2"], r["c3"]] for r in trh],
                      [r[f"y{h}"] for r in trh])
            preds["channels_only"] = np.asarray([[1, r["v_t"], r["c1"], r["c2"], r["c3"]] for r in teh], float) @ cc
            for nm, p in preds.items():
                r2[nm][h].append(_r2(p, yte))
                for r, pv in zip(teh, p):
                    oof[nm][h][r["_i"]] = float(pv)
    return recs, r2, oof, names


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
    return point, lo, hi, float(np.mean(boots <= 0))


def report(sample, recs, r2, oof, names, lines):
    lines.append(f"## {sample} (n={len({r['pid'] for r in recs})} participants, {len(recs)} records)")
    lines.append("")
    lines.append("| predictor | " + " | ".join(f"h={h} R2 (fold SD)" for h in HORIZONS) + " |")
    lines.append("|---|" + "---:|" * len(HORIZONS))
    for nm in names:
        lines.append(f"| {nm} | " + " | ".join(
            f"{np.nanmean(r2[nm][h]):.3f} ({np.nanstd(r2[nm][h]):.3f})" for h in HORIZONS) + " |")
    lines.append("")
    lines.append(f"Participant-bootstrap 95% CIs on R2 differences ({N_BOOT} resamples):")
    lines.append("")
    base_names = [n for n in names if not n.startswith("model_") and n != "channels_only"]
    for h in HORIZONS:
        best = max(base_names, key=lambda nm: np.nanmean(r2[nm][h]))
        for label, b in ((f"best baseline ({best})", best), ("ungated", "model_ungated"),
                         ("frame clamped PRESENT", "model_frame_present"),
                         ("channels only", "channels_only")):
            pt, lo, hi, p = bootstrap_diff(recs, oof, "model_full", b, h)
            lines.append(f"- h={h}: full - {label}: {pt:+.3f} [{lo:+.3f}, {hi:+.3f}], P(diff<=0)={p:.3f}")
    lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    lines = ["# ESM frame ablation with the paper's fitted parameters (2026-09-27)", "",
             f"Script: `esm_frame_ablation.py`. FITTED = {FITTED}. Rollout off (horizon 1), "
             "policy-averaged predictive transition, 5-fold participant CV, train-fit affine "
             "calibration. `full` = frame-gated (g=1); `ungated` = pre-revision model; "
             "`frame_*` clamp the gating posterior; `channels_only` = affine regression on v_t "
             "and the three readout channels of the full model.", ""]
    g = load_geschwind()
    recs, r2, oof, names = evaluate("Geschwind", g, args.workers, has_event=True)
    report("Geschwind-Bringmann remitted-depression ESM", recs, r2, oof, names, lines)
    o = load_osf()
    recs, r2, oof, names = evaluate("osf_83cfk", o, args.workers, has_event=False)
    report("Reliability ESM (osf.io/83cfk), valence only", recs, r2, oof, names, lines)
    out = ROOT / "reviews" / "esm_frame_ablation.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
