"""Rebuilt empirical validation for the temporal-framing model.

Motivation
----------
The previous harness (``empirical_validation.py``) benchmarked one-step
next-valence *level* prediction, a target that a trivial persistence baseline
already dominates. Its "win" came from a persistence-like ``valence_inertia``
prior, and it scored the Joffily/Hesp *derivative* readouts on a level task
they were never designed for. This rebuild fixes that:

1. Fair baselines, each given an optimal train-fit linear calibration:
     - naive persistence (v_{t+1} = v_t, no fit)
     - AR(1)                (a + b v_t)
     - linear event model   (a + b v_t + c event_t)
     - linear ASYMMETRIC    (a + b v_t + p*relu(e) + n*relu(-e))
2. Targets where the model's structure can actually matter:
     - multi-step-ahead prediction (h = 1, 2, 3): persistence/AR(1) decay,
       a correct dynamical model should degrade more gracefully;
     - transition ASYMMETRY: do negative events move valence more than
       positive events? A symmetric baseline structurally predicts zero
       asymmetry; the c_pos != c_neg mechanism can capture it.
     - a NON-CIRCULAR test: does the model's future-frame belief (driven only
       by valence+event observations) track the independently measured worry
       item, which is never fed to the model?
3. Proper generalisation: k-fold cross-validation with whole PARTICIPANTS held
   out (calibration fit on train participants, scored on unseen participants).

The script reports the truth. If the model does not beat a baseline on a
target, it says so.

Run:  python -B empirical_rebuild.py [--quick] [--folds 5]
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean

import numpy as np

from agent import Agent
from generative_model import EPS, N_FRAMES, build_model

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data_raw"
GESCHWIND_PATH = DATA_RAW / "geschwind_2013_s004.csv"
REPORT = ROOT / "empirical_rebuild_report.md"

# Column indices in the PLOS S4 rows (rows carry one extra leading field).
COL = dict(participant=1, day=2, beep=3, cheerful=6, pleasantness=7,
           worried=8, fearful=9, sad=10, relaxed=11)

HORIZONS = (1, 2, 3)

# Model parameterisation. One global setting for the residual-depression
# sample (no per-participant fitting yet). The FULL model uses asymmetric
# hedonic sensitivity; the SYMMETRIC ablation forces c_pos = c_neg.
FULL = dict(K=8, M=8, pi_pos=3.0, omega_e=3.0, gamma=16.0,
            c_pos=0.6, c_neg=1.6, neg_val_precision=1.3, valence_inertia=0.2)
# Frame-gated precision (revision 2026-09-27): the FULL model is gated with
# g = 1; the "ungated" variant (g = 0) is the pre-revision model.
FRAME_GAIN = 1.0
N_BOOT = 2000


# ── small numeric helpers ──────────────────────────────────
def _f(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _pearson(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pts) < 3:
        return float("nan")
    x = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    if x.std() < EPS or y.std() < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _fit_linear(X, y):
    """Ordinary least squares. X: (n, k) with intercept column included."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _rmse(pred, y):
    pred = np.asarray(pred, float)
    y = np.asarray(y, float)
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def _r2(pred, y):
    pred = np.asarray(pred, float)
    y = np.asarray(y, float)
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst < EPS:
        return float("nan")
    sse = float(np.sum((y - pred) ** 2))
    return 1.0 - sse / sst


# ── data ───────────────────────────────────────────────────
def _valence(row):
    pos = [row[k] for k in ("cheerful", "relaxed") if row[k] is not None]
    neg = [row[k] for k in ("worried", "fearful", "sad") if row[k] is not None]
    if not pos or not neg:
        return None
    return mean(pos) - mean(neg)


def _norm_v(v):
    """Composite valence (~[-6, 6]) -> [0, 1]."""
    return None if v is None else float(np.clip((v + 6.0) / 12.0, 0.0, 1.0))


def _bin_e(e):
    if e is None:
        return 1
    return 0 if e < 0 else (2 if e > 0 else 1)


def _bin_v(v_norm, K):
    return int(np.clip(round(v_norm * (K - 1)), 0, K - 1))


def load_participants():
    """Return {pid: [ordered beep dicts with v_norm, event, worry]}."""
    if not GESCHWIND_PATH.exists():
        raise FileNotFoundError(f"missing {GESCHWIND_PATH}")
    raw = {}
    with GESCHWIND_PATH.open(encoding="utf-8-sig", newline="") as h:
        r = csv.reader(h)
        next(r)
        for vals in r:
            if len(vals) < 13:
                continue
            row = dict(
                participant=vals[COL["participant"]],
                day=_f(vals[COL["day"]]), beep=_f(vals[COL["beep"]]),
                cheerful=_f(vals[COL["cheerful"]]),
                pleasantness=_f(vals[COL["pleasantness"]]),
                worried=_f(vals[COL["worried"]]),
                fearful=_f(vals[COL["fearful"]]),
                sad=_f(vals[COL["sad"]]),
                relaxed=_f(vals[COL["relaxed"]]),
            )
            raw.setdefault(row["participant"], []).append(row)

    out = {}
    for pid, rows in raw.items():
        rows.sort(key=lambda r: (r["day"] or -1, r["beep"] or -1))
        seq = []
        for row in rows:
            v = _norm_v(_valence(row))
            if v is None:            # can't observe valence -> drop beep
                continue
            seq.append(dict(v=v, e=row["pleasantness"], w=row["worried"]))
        if len(seq) >= 12:
            out[pid] = seq
    return out


# ── drive the generative model over one participant's sequence ──
def _future_frame(beliefs, K, M):
    f = beliefs.reshape(K, M, N_FRAMES).sum(axis=(0, 1))
    return float(f[2])          # PAST, PRESENT, FUTURE -> index 2


def drive(seq, variant, seed):
    """Return per-beep model predictions for a participant.

    Produces, aligned to each beep position i:
      m[h] : h-step-ahead predicted normalised valence (continue-policy rollout)
      v_model / v_reward / v_action : the three readout channels at i
      frame_future : posterior future-frame belief at i
    """
    p = dict(FULL)
    horizon, adaptive = 2, True
    frame_gain, frame_clamp = FRAME_GAIN, None
    if variant == "symmetric":
        p.update(c_pos=1.0, c_neg=1.0, neg_val_precision=1.0)
    elif variant == "one_step":
        horizon, adaptive = 1, False
    elif variant == "no_inertia":
        p.update(valence_inertia=0.0)
    elif variant == "ungated":
        # pre-revision model: the frame is inferred but gates nothing
        frame_gain = 0.0
    elif variant == "frame_present":
        # gated model with the gating posterior clamped to PRESENT
        frame_clamp = 1
    elif variant == "frame_past":
        frame_clamp = 0
    elif variant == "frame_future":
        frame_clamp = 2
    # "full" uses defaults (frame-gated, g = FRAME_GAIN)

    K, M = p["K"], p["M"]
    model = build_model(K=K, M=M, pi_pos=p["pi_pos"], omega_e=p["omega_e"],
                        gamma=p["gamma"], c_pos=p["c_pos"], c_neg=p["c_neg"],
                        neg_val_precision=p["neg_val_precision"],
                        valence_inertia=p["valence_inertia"])
    agent = Agent(model, gamma=p["gamma"], pi_pos=p["pi_pos"],
                  omega_e=p["omega_e"], c_pos=p["c_pos"], c_neg=p["c_neg"],
                  neg_val_precision=p["neg_val_precision"],
                  valence_inertia=p["valence_inertia"],
                  counterfactual_horizon=horizon,
                  adaptive_counterfactual_horizon=adaptive,
                  max_counterfactual_horizon=3,
                  frame_gain=frame_gain, frame_clamp=frame_clamp, seed=seed)

    v_axis = np.arange(K)
    preds = {h: [] for h in HORIZONS}
    v_model, v_reward, v_action, frame_future = [], [], [], []
    for beep in seq:
        obs = [_bin_e(beep["e"]), 1, _bin_v(beep["v"], K)]
        action, info = agent.step(obs)
        beliefs = info["beliefs"]
        v_model.append(info["v_model"])
        v_reward.append(info["v_reward"])
        v_action.append(info["v_action"])
        frame_future.append(_future_frame(beliefs, K, M))
        # Policy-averaged predictive transition B_bar = sum_a pi(a) B[a].
        # This is the model's actual one-step predictive under its policy
        # posterior (deterministic; no action-sampling noise), and a
        # Bayesian-average rollout for multi-step.
        pi = info["pi"]
        B = sum(pi[a] * model.B[a] for a in range(len(pi)))
        q = beliefs.copy()
        for h in range(1, max(HORIZONS) + 1):
            q = B @ q
            q = np.maximum(q, EPS)
            q /= q.sum()
            if h in preds:
                v_marg = q.reshape(K, M, N_FRAMES).sum(axis=(1, 2))
                preds[h].append(float(v_marg @ v_axis / max(K - 1, 1)))
    return dict(m=preds, v_model=v_model, v_reward=v_reward,
                v_action=v_action, frame_future=frame_future)


# ── assemble a flat table of prediction records ────────────
def _drive_job(args):
    pid, seq, variant, seed = args
    return pid, variant, drive(seq, variant, seed)


def build_records(participants, variants, quick=False, workers=1):
    pids = sorted(participants)
    if quick:
        pids = pids[:40]
    driven = {v: {} for v in variants}
    jobs = [(pid, participants[pid], v, 100 + i)
            for i, pid in enumerate(pids) for v in variants]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for pid, v, d in ex.map(_drive_job, jobs, chunksize=4):
                driven[v][pid] = d
    else:
        for job in jobs:
            pid, v, d = _drive_job(job)
            driven[v][pid] = d

    records = []
    for pid in pids:
        seq = participants[pid]
        n = len(seq)
        for i in range(n):
            base = dict(pid=pid, v_t=seq[i]["v"],
                        e_t=(seq[i]["e"] if seq[i]["e"] is not None else 0.0),
                        w_t=seq[i]["w"],
                        vfut=driven["full"][pid]["frame_future"][i],
                        r_joffily=driven["full"][pid]["v_model"][i],
                        r_pattisapu=driven["full"][pid]["v_reward"][i],
                        r_hesp=driven["full"][pid]["v_action"][i])
            for h in HORIZONS:
                base[f"y{h}"] = seq[i + h]["v"] if i + h < n else None
            for v in variants:
                for h in HORIZONS:
                    base[f"m_{v}_{h}"] = driven[v][pid]["m"][h][i]
            records.append(base)
    return records, pids


# ── predictors (train -> calibrated prediction on test) ────
def _apply_affine(coef, x):
    return coef[0] + coef[1] * np.asarray(x, float)


def evaluate(records, pids, folds, variants):
    rng = np.random.RandomState(0)
    order = list(pids)
    rng.shuffle(order)
    fold_of = {pid: k % folds for k, pid in enumerate(order)}

    # predictor -> horizon -> list of (rmse, r2, r) per fold
    metrics = {}
    # out-of-fold predictions, predictor -> horizon -> {record index: pred}
    oof = {}

    def add(name, h, rmse, r2, r):
        metrics.setdefault(name, {}).setdefault(h, []).append((rmse, r2, r))

    def keep(name, h, te_idx, pred):
        d = oof.setdefault(name, {}).setdefault(h, {})
        for i, p in zip(te_idx, np.asarray(pred, float)):
            d[i] = float(p)

    e_mean_all = mean(r["e_t"] for r in records)
    for i, r in enumerate(records):
        r["_i"] = i

    for k in range(folds):
        train = [r for r in records if fold_of[r["pid"]] != k]
        test = [r for r in records if fold_of[r["pid"]] == k]

        for h in HORIZONS:
            tr = [r for r in train if r[f"y{h}"] is not None]
            te = [r for r in test if r[f"y{h}"] is not None]
            if len(tr) < 10 or len(te) < 10:
                continue
            ytr = np.array([r[f"y{h}"] for r in tr])
            yte = np.array([r[f"y{h}"] for r in te])
            te_idx = [r["_i"] for r in te]

            # naive persistence (no fit)
            add("persistence", h, _rmse([r["v_t"] for r in te], yte),
                _r2([r["v_t"] for r in te], yte),
                _pearson([r["v_t"] for r in te], yte))
            keep("persistence", h, te_idx, [r["v_t"] for r in te])

            # mean
            add("mean", h, _rmse(np.full(len(te), ytr.mean()), yte),
                _r2(np.full(len(te), ytr.mean()), yte), float("nan"))

            # AR(1): iterate a + b v
            c = _fit_linear([[1, r["v_t"]] for r in tr], ytr)
            p = np.array([r["v_t"] for r in te], float)
            for _ in range(h):
                p = c[0] + c[1] * p
            add("ar1", h, _rmse(p, yte), _r2(p, yte), _pearson(p, yte))
            keep("ar1", h, te_idx, p)

            # direct h-step regression on v_t (strongest simple predictor)
            c = _fit_linear([[1, r["v_t"]] for r in tr], ytr)
            p = c[0] + c[1] * np.array([r["v_t"] for r in te], float)
            add("direct_v", h, _rmse(p, yte), _r2(p, yte), _pearson(p, yte))
            keep("direct_v", h, te_idx, p)

            # linear event: a + b v + c e ; iterate with mean event for h>1
            c = _fit_linear([[1, r["v_t"], r["e_t"]] for r in tr], ytr)
            p = np.array([r["v_t"] for r in te], float)
            e_now = np.array([r["e_t"] for r in te], float)
            for step in range(h):
                p = c[0] + c[1] * p + c[2] * (e_now if step == 0 else e_mean_all)
            add("linear_event", h, _rmse(p, yte), _r2(p, yte), _pearson(p, yte))
            keep("linear_event", h, te_idx, p)

            # linear asymmetric
            def feat(r):
                e = r["e_t"]
                return [1, r["v_t"], max(e, 0.0), min(e, 0.0)]
            c = _fit_linear([feat(r) for r in tr], ytr)
            p = np.array([r["v_t"] for r in te], float)
            for step in range(h):
                ee = (np.array([r["e_t"] for r in te], float) if step == 0
                      else np.full(len(te), e_mean_all))
                p = c[0] + c[1] * p + c[2] * np.maximum(ee, 0) + c[3] * np.minimum(ee, 0)
            add("linear_event_asym", h, _rmse(p, yte), _r2(p, yte), _pearson(p, yte))
            keep("linear_event_asym", h, te_idx, p)

            # model variants (train-fit affine on the model's own output)
            for v in variants:
                key = f"m_{v}_{h}"
                c = _fit_linear([[1, r[key]] for r in tr], ytr)
                p = _apply_affine(c, [r[key] for r in te])
                add(f"model_{v}", h, _rmse(p, yte), _r2(p, yte), _pearson(p, yte))
                keep(f"model_{v}", h, te_idx, p)

            # channels-only: the three readout channels of the full model as
            # regressors (plus v_t), no state rollout -> isolates channel
            # integration from the frame-gated dynamics
            def cfeat(r):
                return [1, r["v_t"], r["r_joffily"], r["r_pattisapu"], r["r_hesp"]]
            c = _fit_linear([cfeat(r) for r in tr], ytr)
            p = np.asarray([cfeat(r) for r in te], float) @ c
            add("channels_only", h, _rmse(p, yte), _r2(p, yte), _pearson(p, yte))
            keep("channels_only", h, te_idx, p)

            # readout baselines: 1-step only (they are instantaneous channels)
            if h == 1:
                for nm, col in (("readout_joffily", "r_joffily"),
                                ("readout_pattisapu", "r_pattisapu"),
                                ("readout_hesp", "r_hesp")):
                    c = _fit_linear([[1, r[col]] for r in tr], ytr)
                    p = _apply_affine(c, [r[col] for r in te])
                    add(nm, h, _rmse(p, yte), _r2(p, yte), _pearson(p, yte))

    # aggregate
    agg = {}
    for name, hs in metrics.items():
        agg[name] = {}
        for h, vals in hs.items():
            arr = np.array(vals, float)
            agg[name][h] = dict(rmse=float(np.nanmean(arr[:, 0])),
                                 rmse_sd=float(np.nanstd(arr[:, 0])),
                                 r2=float(np.nanmean(arr[:, 1])),
                                 r2_sd=float(np.nanstd(arr[:, 1])),
                                 r2_folds=[float(x) for x in arr[:, 1]],
                                 r=float(np.nanmean(arr[:, 2])))
    return agg, oof


def bootstrap_r2_diff(records, oof, name_a, name_b, h, n_boot=N_BOOT, seed=0):
    """Participant-level bootstrap CI for R2(name_a) - R2(name_b) at horizon h,
    using pooled out-of-fold predictions. Returns (point, lo, hi, p_le_0)."""
    rows = [(r["pid"], r[f"y{h}"], oof[name_a][h][r["_i"]], oof[name_b][h][r["_i"]])
            for r in records
            if r[f"y{h}"] is not None and r["_i"] in oof[name_a][h]
            and r["_i"] in oof[name_b][h]]
    pids = sorted({p for p, *_ in rows})
    by = {p: [] for p in pids}
    for p, y, a, b in rows:
        by[p].append((y, a, b))
    by = {p: np.array(v, float) for p, v in by.items()}

    def diff(sel):
        arr = np.concatenate([by[p] for p in sel])
        return _r2(arr[:, 1], arr[:, 0]) - _r2(arr[:, 2], arr[:, 0])

    point = diff(pids)
    rng = np.random.RandomState(seed)
    boots = np.array([diff(rng.choice(pids, size=len(pids), replace=True))
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(point=float(point), lo=float(lo), hi=float(hi),
                p_le_0=float(np.mean(boots <= 0)), n_participants=len(pids))


# ── asymmetry + non-circular frame->worry analyses ─────────
def asymmetry(records):
    """Regress delta-valence on positive/negative event parts.

    Reports empirical vs model-implied (full vs symmetric) event sensitivity,
    separately for positive and negative events. A symmetric mechanism gives
    |beta_pos| ~ |beta_neg|; the data / asymmetric model can differ.
    """
    def fit(dep_key, is_model):
        X, y = [], []
        for r in records:
            if r["y1"] is None:
                continue
            e = r["e_t"]
            if is_model:
                dv = r[dep_key] - r["v_t"]          # model implied change
            else:
                dv = r["y1"] - r["v_t"]             # empirical change
            X.append([1, max(e, 0.0), min(e, 0.0)])
            y.append(dv)
        c = _fit_linear(X, y)
        return dict(beta_pos=float(c[1]), beta_neg=float(c[2]))

    emp = fit("y1", is_model=False)
    full = fit("m_full_1", is_model=True)
    sym = fit("m_symmetric_1", is_model=True)
    return dict(empirical=emp, model_full=full, model_symmetric=sym)


def frame_worry(records):
    """Raw frame-worry association. NOTE (revision 2026-09-27): the valence
    composite fed to the model INCLUDES the worried item, so this raw r is not
    a non-circular test. The honest analysis (worry removed from the input,
    covariate-adjusted, cluster-robust) is in frame_worry_multilevel.py."""
    ff = [r["vfut"] for r in records if r["w_t"] is not None]
    ww = [r["w_t"] for r in records if r["w_t"] is not None]
    # compare with the reward readout as a control
    rew = [r["r_pattisapu"] for r in records if r["w_t"] is not None]
    return dict(future_frame_vs_worry=_pearson(ff, ww),
                reward_readout_vs_worry=_pearson(rew, ww),
                n=len(ww))


# ── report ─────────────────────────────────────────────────
def _fmt(x, d=4):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def write_report(agg, asym, fw, n_participants, n_records, quick, cis=None):
    order = ["persistence", "mean", "ar1", "direct_v", "linear_event", "linear_event_asym",
             "model_full", "model_ungated", "model_frame_present", "model_frame_past",
             "model_frame_future", "channels_only",
             "model_symmetric", "model_one_step", "model_no_inertia",
             "readout_joffily", "readout_pattisapu", "readout_hesp"]
    lines = ["# Rebuilt Empirical Validation Report", ""]
    lines.append(f"- Model: frame-gated (g={FRAME_GAIN}); `model_ungated` is the "
                 f"pre-revision frame-inert model; `model_frame_*` clamp the gating "
                 f"posterior; `channels_only` regresses on the three readout channels "
                 f"with no state rollout.")
    lines.append(f"- Dataset: Geschwind/Bringmann residual-depression ESM "
                 f"(`data_raw/geschwind_2013_s004.csv`).")
    lines.append(f"- Participants used: {n_participants}; prediction records: {n_records}"
                 f"{'  (QUICK subset)' if quick else ''}.")
    lines.append(f"- Cross-validation: whole participants held out per fold.")
    lines.append("- Valence normalised to [0,1]; RMSE and R2 in that scale; "
                 "each predictor gets an optimal train-fit linear calibration.")
    lines.append("")

    ar1 = agg.get("ar1", {})
    for h in HORIZONS:
        lines.append(f"## Horizon h = {h} step(s) ahead")
        lines.append("")
        lines.append("| Predictor | RMSE | R2 (fold SD) | r | skill vs AR(1) |")
        lines.append("|---|---:|---:|---:|---:|")
        base = ar1.get(h, {}).get("rmse")
        for name in order:
            m = agg.get(name, {}).get(h)
            if not m:
                continue
            skill = ("n/a" if not base or base == 0
                     else f"{100.0 * (1 - m['rmse'] / base):+.1f}%")
            lines.append(f"| {name} | {_fmt(m['rmse'])} | {_fmt(m['r2'],3)} "
                         f"({_fmt(m.get('r2_sd'),3)}) | "
                         f"{_fmt(m['r'],3)} | {skill} |")
        lines.append("")
        if cis and h in cis:
            lines.append("Participant-bootstrap 95% CI on R2 differences "
                         f"({N_BOOT} resamples of participants):")
            lines.append("")
            for label, d in cis[h].items():
                lines.append(f"- {label}: {d['point']:+.3f} [{d['lo']:+.3f}, "
                             f"{d['hi']:+.3f}], P(diff<=0)={d['p_le_0']:.3f}")
            lines.append("")

    lines.append("## Transition asymmetry (effect of event on 1-step valence change)")
    lines.append("")
    lines.append("Positive vs negative event sensitivity. A symmetric mechanism "
                 "predicts |beta_pos| ~ |beta_neg|.")
    lines.append("")
    lines.append("| Source | beta_pos | beta_neg | |neg|/|pos| |")
    lines.append("|---|---:|---:|---:|")
    for label, key in (("empirical data", "empirical"),
                       ("model (full, asymmetric)", "model_full"),
                       ("model (symmetric ablation)", "model_symmetric")):
        d = asym[key]
        bp, bn = d["beta_pos"], d["beta_neg"]
        ratio = abs(bn) / abs(bp) if abs(bp) > EPS else float("nan")
        lines.append(f"| {label} | {_fmt(bp)} | {_fmt(bn)} | {_fmt(ratio,2)} |")
    lines.append("")

    lines.append("## Raw future-frame belief vs measured worry (NOT non-circular)")
    lines.append("")
    lines.append("The valence composite fed to the model includes the worried item. "
                 "See frame_worry_multilevel.py for the analysis with worry removed "
                 "from the input and covariates controlled.")
    lines.append("")
    lines.append(f"- corr(future-frame belief, worry item) = "
                 f"{_fmt(fw['future_frame_vs_worry'],3)}  (n={fw['n']})")
    lines.append(f"- control corr(reward readout, worry item) = "
                 f"{_fmt(fw['reward_readout_vs_worry'],3)}")
    lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    variants = ["full", "ungated", "frame_present", "frame_past", "frame_future",
                "symmetric", "one_step", "no_inertia"]
    print("Loading participants ...")
    participants = load_participants()
    print(f"  {len(participants)} participants with >=12 valence beeps")
    print("Driving model variants (this is the slow part) ...")
    records, pids = build_records(participants, variants, quick=args.quick,
                                  workers=args.workers)
    print(f"  {len(records)} beep records across {len(pids)} participants")
    print("Cross-validated evaluation ...")
    agg, oof = evaluate(records, pids, args.folds, variants)
    cis = {}
    for h in HORIZONS:
        best_base = max(["ar1", "direct_v", "linear_event", "linear_event_asym"],
                        key=lambda nm: agg[nm][h]["r2"])
        cis[h] = {
            f"full - best baseline ({best_base})":
                bootstrap_r2_diff(records, oof, "model_full", best_base, h),
            "full - ungated (frame-inert)":
                bootstrap_r2_diff(records, oof, "model_full", "model_ungated", h),
            "full - frame clamped to PRESENT":
                bootstrap_r2_diff(records, oof, "model_full", "model_frame_present", h),
            "full - channels only":
                bootstrap_r2_diff(records, oof, "model_full", "channels_only", h),
        }
    asym = asymmetry(records)
    fw = frame_worry(records)
    text = write_report(agg, asym, fw, len(pids), len(records), args.quick, cis=cis)
    print("\n" + text)
    print(f"\nReport written to {REPORT}")


if __name__ == "__main__":
    main()
