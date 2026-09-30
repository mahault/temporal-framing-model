"""
Direct test of the latent temporal frame against reported per-beep temporal
orientation (Baumeister et al. 2020 Study 1 and 2; openESM 0076_bayer), plus a
frame-local affective precision prototype (Shah & Pashea 2026 style).

Usage (from repo root):
    python analysis/orientation/orientation_test.py load       # loaders + descriptives
    python analysis/orientation/orientation_test.py drive      # v1 model drives (cached)
    python analysis/orientation/orientation_test.py eval       # held-out orientation tests
    python analysis/orientation/orientation_test.py precision  # affective precision prototype
    python analysis/orientation/orientation_test.py all

Nothing in the repo's model code is modified; the v1 agent is imported as is.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "analysis" / "orientation" / "out"
OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures"

from agent import Agent  # noqa: E402
from generative_model import EPS, N_FRAMES, build_model  # noqa: E402
from empirical_rebuild import _bin_e, _bin_v  # noqa: E402

K = M = 8
# pooled parameters selected in round 3 (reviews/esm_eval_v3.md, Geschwind folds)
PARAMS = dict(pi_pos=3.0, valence_inertia=0.65, omega_e=3.0, asym=(0.6, 1.6))
LABELS = {1: "past", 2: "present", 3: "future"}
N_BOOT = 1000
RNG = np.random.default_rng(0)


# ─────────────────────────────── loaders ───────────────────────────────
def load_baumeister1():
    """Study 1: 492 adults, random signals over 3 days. One row per signal.

    valence  : 'How happy/sad do you feel right now?' -3..+3 -> v in [0,1]
    contentpn: 'to what extent were your thoughts about something pleasant/
               unpleasant?' -3..+3 (used as the event/pleasantness channel)
    onefactor: 1 past, 2 present, 3 future, 4 past+present, 5 past+future,
               6 present+future, 7 all three, 8 no time aspect
    timefactor: single-focus label (1/2/3) else NaN
    sub-items: regret, replaying, worries, fear, planning (branch checklists),
               checklist worry / repeating (60% random checklist), timespan.
    """
    import pyreadstat
    df, _ = pyreadstat.read_sav(str(ROOT / "data_raw/candidates/baumeister2020_ett/Study1/ETT_ESM_Study1.sav"))
    parts = {}
    for pid, g in df.groupby("PID"):
        g = g.sort_values(["DAY", "SIG"])
        seq = []
        for _, r in g.iterrows():
            if np.isnan(r["valence"]):
                continue
            ori = None if np.isnan(r["onefactor"]) else int(r["onefactor"])
            single = None if np.isnan(r["timefactor"]) else int(r["timefactor"])
            e = None if np.isnan(r["contentpn"]) else float(r["contentpn"])
            hour = None if np.isnan(r["hour"]) else float(r["hour"])
            seq.append(dict(
                v=float((r["valence"] + 3.0) / 6.0), v_raw=float(r["valence"]), e=e,
                d=int(r["DAY"]), sig=int(r["SIG"]), hour=hour,
                ori=ori, single=single,
                any_past=int(ori in (1, 4, 5, 7)) if ori else None,
                any_future=int(ori in (3, 5, 6, 7)) if ori else None,
                any_present=int(ori in (2, 4, 6, 7)) if ori else None,
                notime=int(ori == 8) if ori else None,
                regret=float(r["typepast_1"] == 1.0), replaying=float(r["typepast_2"] == 1.0),
                worries=float(r["typefuture_10"] == 1.0), fear=float(r["typefuture_9"] == 1.0),
                planning=float(r["typefuture_1"] == 1.0),
                cl_worry=None if np.isnan(r["checklist1_31"]) and np.isnan(r["checklist1_1"]) else float(r["checklist1_31"] == 1.0),
                timespan=None if np.isnan(r["timespan"]) else float(r["timespan"]),
                anxious=None if np.isnan(r["anxious"]) else float(r["anxious"]),
                stress=None if np.isnan(r["stress"]) else float(r["stress"]),
            ))
        if len(seq) >= 4:
            parts[str(int(pid))] = seq
    return parts


def load_baumeister2():
    """Study 2: 35 adults, 14 days, forced-choice orientation, bad-good 0-100.
    Date only (no clock time), so used descriptively and not as sequences."""
    import pyreadstat
    df, _ = pyreadstat.read_sav(str(ROOT / "data_raw/candidates/baumeister2020_ett/Study2/Study 2 ESM all trials FOR OSF.sav"))
    rows = []
    for _, r in df.iterrows():
        if np.isnan(r["Affect"]) or np.isnan(r["time"]):
            continue
        rows.append(dict(pid=str(int(r["ID"])), v=float(r["Affect"] / 100.0), single=int(r["time"]),
                         date=str(r["Date_Time"])))
    return rows


def load_bayer():
    """openESM 0076_bayer: 519 adults, 6/day x 14 days.
    pa: 'Right now, how positive or negative do you feel?' 1..5 -> v in [0,1]
    time_orientation: 1 past / 2 present / 3 future (single choice), asked on a
    subset of beeps and participants. survey_error rows dropped."""
    import pandas as pd
    d = pd.read_csv(ROOT / "data_raw/candidates/openesm_0076_bayer/0076_bayer_ts.tsv", sep="\t", low_memory=False)
    d = d[(d.survey_error != 1)]
    d["ts"] = pd.to_datetime(d["start_time"], utc=True, errors="coerce")
    parts = {}
    for pid, g in d.groupby("id"):
        g = g.sort_values(["day", "beep"])
        seq = []
        for _, r in g.iterrows():
            if np.isnan(r["pa"]):
                continue
            ori = None if np.isnan(r["time_orientation"]) else int(r["time_orientation"])
            hour = None if pd.isna(r["ts"]) else float(r["ts"].hour + r["ts"].minute / 60)
            seq.append(dict(v=float((r["pa"] - 1.0) / 4.0), v_raw=float(r["pa"]), e=None,
                            d=int(r["day"]), sig=int(r["beep"]), hour=hour, ori=ori, single=ori,
                            any_past=None if ori is None else int(ori == 1),
                            any_future=None if ori is None else int(ori == 3),
                            any_present=None if ori is None else int(ori == 2), notime=None,
                            problem=None if np.isnan(r["problem_thoughts"]) else float(r["problem_thoughts"]),
                            offtask=None if np.isnan(r["focus_activity"]) else float(r["focus_activity"])))
        if len(seq) >= 4:
            parts[str(int(pid))] = seq
    return parts


# ─────────────────────────── statistics helpers ───────────────────────────
def ols_cluster(X, y, groups):
    """OLS with cluster-robust (CR1) standard errors. Returns beta, se."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    meat = np.zeros_like(XtX_inv)
    gs = np.unique(groups)
    for g in gs:
        m = groups == g
        s = X[m].T @ u[m]
        meat += np.outer(s, s)
    G = len(gs); n, k = X.shape
    adj = G / (G - 1) * (n - 1) / (n - k)
    V = adj * XtX_inv @ meat @ XtX_inv
    return beta, np.sqrt(np.diag(V))


def logit_cluster(X, y, groups, l2=1e-4, iters=50):
    """Logistic regression (IRLS) with cluster-robust SEs."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, k = X.shape
    b = np.zeros(k)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + l2 * np.eye(k)
        g = X.T @ (y - p) - l2 * b
        step = np.linalg.solve(H, g)
        b = b + step
        if np.abs(step).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-X @ b))
    W = p * (1 - p)
    Hinv = np.linalg.pinv(X.T @ (X * W[:, None]))
    meat = np.zeros((k, k))
    gs = np.unique(groups)
    for gid in gs:
        m = groups == gid
        s = X[m].T @ (y[m] - p[m])
        meat += np.outer(s, s)
    G = len(gs)
    V = G / (G - 1) * Hinv @ meat @ Hinv
    return b, np.sqrt(np.diag(V))


def demean_by(X, groups):
    X = np.asarray(X, float).copy()
    for g in np.unique(groups):
        m = groups == g
        X[m] -= X[m].mean(axis=0)
    return X


def boot_ci(vals_by_pid, fn, n_boot=N_BOOT, seed=0):
    """Participant bootstrap of a statistic fn(list_of_participant_entries)."""
    rng = np.random.default_rng(seed)
    pids = list(vals_by_pid)
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(len(pids), len(pids), replace=True)
        stats.append(fn([vals_by_pid[pids[i]] for i in pick]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# ─────────────────────────── descriptives ───────────────────────────
def descriptives(name, parts, rows2=None):
    rep = {}
    n_sig = sum(len(s) for s in parts.values())
    labelled = [(pid, b) for pid, s in parts.items() for b in s if b["single"] is not None]
    rep["n_participants"] = len(parts)
    rep["n_signals_with_valence"] = n_sig
    rep["n_single_focus"] = len(labelled)
    counts = defaultdict(int)
    for _, b in labelled:
        counts[LABELS[b["single"]]] += 1
    rep["single_focus_counts"] = dict(counts)
    if name == "baumeister1":
        oc = defaultdict(int)
        for s in parts.values():
            for b in s:
                if b["ori"] is not None:
                    oc[b["ori"]] += 1
        rep["onefactor_counts"] = {int(k): int(v) for k, v in sorted(oc.items())}

    # valence by orientation: pooled means, within-person (participant FE) contrasts vs present
    pid_arr = np.array([p for p, _ in labelled])
    v = np.array([b["v_raw"] for _, b in labelled])
    lab = np.array([b["single"] for _, b in labelled])
    rep["valence_mean_by_orientation"] = {LABELS[k]: dict(mean=float(v[lab == k].mean()), n=int((lab == k).sum())) for k in (1, 2, 3)}
    X = np.column_stack([(lab == 1).astype(float), (lab == 3).astype(float)])
    Xd = demean_by(X, pid_arr); yd = demean_by(v[:, None], pid_arr)[:, 0]
    b, se = ols_cluster(Xd, yd, pid_arr)
    rep["within_person_contrast_vs_present"] = {"past": dict(beta=float(b[0]), se=float(se[0])),
                                                "future": dict(beta=float(b[1]), se=float(se[1]))}
    Xp = np.column_stack([np.ones(len(v)), X])
    b2, se2 = ols_cluster(Xp, v, pid_arr)
    rep["pooled_contrast_vs_present"] = {"past": dict(beta=float(b2[1]), se=float(se2[1])),
                                        "future": dict(beta=float(b2[2]), se=float(se2[2]))}

    # valence before and after each orientation (adjacent signals, same day)
    ba = {LABELS[k]: dict(before=[], at=[], after=[]) for k in (1, 2, 3)}
    trans = np.zeros((3, 3)); trans_any = defaultdict(lambda: defaultdict(int))
    pers_rows = []
    for pid, s in parts.items():
        for i, b in enumerate(s):
            if b["single"] is None:
                continue
            k = LABELS[b["single"]]
            ba[k]["at"].append(b["v_raw"])
            if i > 0 and s[i - 1]["d"] == b["d"]:
                ba[k]["before"].append(s[i - 1]["v_raw"])
            if i + 1 < len(s) and s[i + 1]["d"] == b["d"]:
                ba[k]["after"].append(s[i + 1]["v_raw"])
                nb = s[i + 1]
                if nb["single"] is not None:
                    trans[b["single"] - 1, nb["single"] - 1] += 1
                if nb["any_future"] is not None and b["any_future"] is not None:
                    pers_rows.append((pid, b["any_future"], nb["any_future"], b["v"], nb["v"], b["any_past"], nb["any_past"]))
    rep["valence_before_at_after"] = {k: {kk: (float(np.mean(vv)) if vv else None) for kk, vv in d.items()} for k, d in ba.items()}
    rep["transition_counts"] = trans.astype(int).tolist()
    rowsum = trans.sum(axis=1, keepdims=True); rowsum[rowsum == 0] = 1
    rep["transition_rownorm"] = (trans / rowsum).round(3).tolist()
    # persistence: future_{t+1} ~ future_t + v_t + v_{t+1}, participant FE (demeaned LPM) and logit with Mundlak mean
    if pers_rows:
        pr = np.array([(r[1], r[2], r[3], r[4], r[5], r[6]) for r in pers_rows], float)
        gp = np.array([r[0] for r in pers_rows])
        X = np.column_stack([pr[:, 0], pr[:, 2], pr[:, 3]])
        b, se = ols_cluster(demean_by(X, gp), demean_by(pr[:, 1:2], gp)[:, 0], gp)
        rep["persistence_future_LPM_within"] = dict(beta_future_t=float(b[0]), se=float(se[0]), beta_v_t=float(b[1]), beta_v_t1=float(b[2]), n=int(len(gp)))
        mean_f = np.array([pr[gp == g, 0].mean() for g in gp])
        Xl = np.column_stack([np.ones(len(gp)), pr[:, 0], pr[:, 2], pr[:, 3], mean_f])
        bl, sel = logit_cluster(Xl, pr[:, 1], gp)
        rep["persistence_future_logit_mundlak"] = dict(beta_future_t=float(bl[1]), se=float(sel[1]), or_=float(np.exp(bl[1])))
        X = np.column_stack([pr[:, 4], pr[:, 2], pr[:, 3]])
        b, se = ols_cluster(demean_by(X, gp), demean_by(pr[:, 5:6], gp)[:, 0], gp)
        rep["persistence_past_LPM_within"] = dict(beta_past_t=float(b[0]), se=float(se[0]), n=int(len(gp)))
        # raw conditional rates
        rep["P_future_t1_given_future_t"] = float(pr[pr[:, 0] == 1, 1].mean())
        rep["P_future_t1_given_not_future_t"] = float(pr[pr[:, 0] == 0, 1].mean())
        rep["P_past_t1_given_past_t"] = float(pr[pr[:, 4] == 1, 5].mean())
        rep["P_past_t1_given_not_past_t"] = float(pr[pr[:, 4] == 0, 5].mean())
    if name == "baumeister1":
        # sub-item valence: regret / replaying / worries / fear / planning among single-focus
        sub = {}
        for key in ("regret", "replaying", "worries", "fear", "planning"):
            vv = np.array([b["v_raw"] for _, b in labelled if b[key] == 1.0])
            sub[key] = dict(n=int(len(vv)), mean_valence=(float(vv.mean()) if len(vv) else None))
        rep["subitem_valence"] = sub
        # temporal distance vs valence within past and within future
        for k, nm in ((1, "past"), (3, "future")):
            xs = [(b["timespan"], b["v_raw"], p) for p, b in labelled if b["single"] == k and b["timespan"] is not None]
            if len(xs) > 30:
                a = np.array([(x[0], x[1]) for x in xs], float); g = np.array([x[2] for x in xs])
                bb, ss = ols_cluster(demean_by(a[:, :1], g), demean_by(a[:, 1:2], g)[:, 0], g)
                rep[f"valence_vs_timespan_within_{nm}"] = dict(beta=float(bb[0]), se=float(ss[0]), n=int(len(g)))
    if rows2 is not None:
        v2 = np.array([r["v"] for r in rows2]); l2 = np.array([r["single"] for r in rows2]); p2 = np.array([r["pid"] for r in rows2])
        X = np.column_stack([(l2 == 1).astype(float), (l2 == 3).astype(float)])
        b, se = ols_cluster(demean_by(X, p2), demean_by(v2[:, None] * 100, p2)[:, 0], p2)
        rep["study2"] = dict(n_participants=int(len(np.unique(p2))), n_reports=int(len(v2)),
                             counts={LABELS[k]: int((l2 == k).sum()) for k in (1, 2, 3)},
                             mean_by_orientation={LABELS[k]: float(v2[l2 == k].mean() * 100) for k in (1, 2, 3)},
                             within_contrast_vs_present={"past": dict(beta=float(b[0]), se=float(se[0])), "future": dict(beta=float(b[1]), se=float(se[1]))})
    return rep


# ─────────────────────────── v1 model drive ───────────────────────────
def make_agent(g, seed=0, gamma=16.0):
    c_pos, c_neg = PARAMS["asym"]
    model = build_model(K=K, M=M, pi_pos=PARAMS["pi_pos"], omega_e=PARAMS["omega_e"], gamma=gamma,
                        c_pos=c_pos, c_neg=c_neg, neg_val_precision=1.0, valence_inertia=PARAMS["valence_inertia"])
    agent = Agent(model, gamma=gamma, pi_pos=PARAMS["pi_pos"], omega_e=PARAMS["omega_e"], c_pos=c_pos, c_neg=c_neg,
                  neg_val_precision=1.0, valence_inertia=PARAMS["valence_inertia"], counterfactual_horizon=1,
                  adaptive_counterfactual_horizon=False, frame_gain=g, frame_clamp=None, frame_transition_gain=0.0, seed=seed)
    return model, agent


def drive_seq(seq, g, precision=None, seed=0):
    """Drive the v1 agent over one participant's valence sequence.
    precision: None | 'tracked' | 'full' (frame-local affective precision prototype).

    Frame-local trackers: one log-precision multiplier L_k per frame k in
    (past, present, future). Charge phi_k = alpha * channel_k, where the three
    v1 channels are the backward (v_model), present (v_reward) and forward
    (v_action) valence signals, each in [-1, 1] with 0 as the neutral baseline.
    Persistence prior rho keeps L_k smooth. In the 'full' variant the dominant
    frame's tracker scales policy precision gamma before the next step; in
    'tracked' the trackers run but gamma stays at its base value.

    Note: in the v1 model the valence likelihood is identical across frames,
    so a per-frame predictive surprisal of the observed valence (the Shah and
    Pashea construction) is identical for the three frames and carries no
    frame-specific evidence; the channel-based charge is used instead."""
    model, agent = make_agent(g, seed)
    gamma_base = agent.gamma
    alpha, rho = 1.0, 0.8            # charge gain, persistence prior
    L = np.zeros(3)                  # log precision multipliers per frame
    out = []
    for beep in seq:
        obs = [_bin_e(beep["e"]), 1, _bin_v(beep["v"], K)]
        qf_prev = agent.beliefs.reshape(K, M, N_FRAMES).sum(axis=(0, 1))
        dom = int(np.argmax(qf_prev))
        if precision == "full":
            agent.gamma = gamma_base * float(np.exp(L[dom]))
        else:
            agent.gamma = gamma_base
        act, info = agent.step(obs)
        phi = alpha * np.array([info["v_model"], info["v_reward"], info["v_action"]], float)
        if precision is not None:
            L = np.clip(rho * L + phi, -2.0, 2.0)
        qf = info["beliefs"].reshape(K, M, N_FRAMES).sum(axis=(0, 1))
        out.append(dict(qf=[float(x) for x in qf], dom=int(np.argmax(qf)), act=int(act),
                        v_model=float(info["v_model"]), v_reward=float(info["v_reward"]), v_action=float(info["v_action"]),
                        L=[float(x) for x in L], phi=[float(x) for x in phi], gamma=float(agent.gamma)))
    return out


def _job(a):
    name, pid, seq, g, precision = a
    return name, pid, g, precision, drive_seq(seq, g, precision)


def drive_all(datasets, variants, workers=12):
    jobs = [(name, pid, seq, g, prec) for name, parts in datasets.items() for pid, seq in parts.items() for g, prec in variants]
    out = {}
    t0 = time.time()
    for j in jobs:
        name, pid, g, prec, d = _job(j)
        out[(name, pid, g, prec)] = d
    print(f"drove {len(jobs)} sequences in {time.time() - t0:.0f}s")
    return out


# ─────────────────────────── held-out orientation test ───────────────────────────
def softmax_rows(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z); return E / E.sum(axis=1, keepdims=True)


def fit_multinomial(X, y, C=3, lam=1.0):
    from scipy.optimize import minimize
    n, d = X.shape
    Y = np.zeros((n, C)); Y[np.arange(n), y] = 1

    def f(w):
        W = w.reshape(d, C)
        P = softmax_rows(X @ W)
        ll = -(Y * np.log(P + 1e-12)).sum() + 0.5 * lam * (W[1:] ** 2).sum()
        G = X.T @ (P - Y); G[1:] += lam * W[1:]
        return ll, G.ravel()
    r = minimize(f, np.zeros(d * C), jac=True, method="L-BFGS-B")
    return r.x.reshape(d, C)


def auc(scores, labels):
    scores = np.asarray(scores); labels = np.asarray(labels)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def build_rows(parts, drives, g, precision=None, name=""):
    rows = []
    for pid, seq in parts.items():
        d = drives[(name, pid, g, precision)]
        for i, b in enumerate(seq):
            if b["single"] is None:
                continue
            v1 = seq[i - 1]["v"] if i > 0 and seq[i - 1]["d"] == b["d"] else b["v"]
            v2 = seq[i - 2]["v"] if i > 1 and seq[i - 2]["d"] == b["d"] else v1
            tod = 0.5 if b["hour"] is None else (b["hour"] - 8.0) / 12.0
            e = 0.0 if b.get("e") is None else b["e"] / 3.0
            rows.append(dict(pid=pid, y=b["single"] - 1, base=[b["v"], v1, v2, b["v"] - v1, tod, e],
                             qf=d[i]["qf"][:2], ch=[d[i]["v_model"], d[i]["v_reward"], d[i]["v_action"]],
                             L=d[i]["L"], dom=d[i]["dom"]))
    return rows


def heldout_eval(rows, featsets, n_folds=5, lam=1.0, seed=0):
    """Participant-held-out multinomial log-likelihood (nats/signal, relative to
    training base rates) and one-vs-rest AUCs, per feature set. Returns per-pid
    summaries for bootstrap."""
    pids = sorted({r["pid"] for r in rows})
    rng = np.random.default_rng(seed); rng.shuffle(pids)
    folds = [set(pids[i::n_folds]) for i in range(n_folds)]
    res = {fs: defaultdict(lambda: dict(ll=[], p=[], y=[])) for fs in featsets}
    base_pp = defaultdict(lambda: dict(ll=[], y=[]))
    for k in range(n_folds):
        te = folds[k]
        tr_rows = [r for r in rows if r["pid"] not in te]; te_rows = [r for r in rows if r["pid"] in te]
        ytr = np.array([r["y"] for r in tr_rows]); rates = np.bincount(ytr, minlength=3) / len(ytr)
        for fs in featsets:
            def mk(rs):
                X = []
                for r in rs:
                    f = [1.0]
                    if "base" in fs: f += r["base"]
                    if "qf" in fs: f += r["qf"]
                    if "ch" in fs: f += r["ch"]
                    if "L" in fs: f += r["L"]
                    X.append(f)
                return np.array(X)
            Xtr = mk(tr_rows); Xte = mk(te_rows)
            mu = Xtr[:, 1:].mean(axis=0); sd = Xtr[:, 1:].std(axis=0) + 1e-9
            Xtr[:, 1:] = (Xtr[:, 1:] - mu) / sd; Xte[:, 1:] = (Xte[:, 1:] - mu) / sd
            W = fit_multinomial(Xtr, ytr, lam=lam)
            P = softmax_rows(Xte @ W)
            for r, p in zip(te_rows, P):
                res[fs][r["pid"]]["ll"].append(np.log(p[r["y"]] + 1e-12) - np.log(rates[r["y"]] + 1e-12))
                res[fs][r["pid"]]["p"].append(p.tolist()); res[fs][r["pid"]]["y"].append(r["y"])
        for r in te_rows:
            base_pp[r["pid"]]["ll"].append(0.0)
    summary = {}
    for fs in featsets:
        d = res[fs]
        allp = np.array([p for pid in d for p in d[pid]["p"]]); ally = np.array([y for pid in d for y in d[pid]["y"]])
        ll_by_pid = {pid: float(np.mean(d[pid]["ll"])) for pid in d}
        summary[fs] = dict(ll_gain_mean=float(np.mean([np.mean(d[pid]["ll"]) for pid in d])),
                           ll_gain_pooled=float(np.mean([x for pid in d for x in d[pid]["ll"]])),
                           auc_past=auc(allp[:, 0], ally == 0), auc_present=auc(allp[:, 1], ally == 1), auc_future=auc(allp[:, 2], ally == 2),
                           _ll_by_pid=ll_by_pid, _pooled=(allp, ally))
    return summary


def diff_ci(sumA, sumB):
    pids = list(sumA["_ll_by_pid"])
    a = sumA["_ll_by_pid"]; b = sumB["_ll_by_pid"]
    return boot_ci({p: (a[p], b[p]) for p in pids}, lambda xs: float(np.mean([x[0] - x[1] for x in xs])))


def clean(s):
    return {k: v for k, v in s.items() if not k.startswith("_")}


# ─────────────────────────── main stages ───────────────────────────
def stage_load():
    b1 = load_baumeister1(); b2 = load_baumeister2(); by = load_bayer()
    json.dump(dict(baumeister1=b1, bayer=by), open(OUT / "parts.json", "w"))
    json.dump(b2, open(OUT / "study2.json", "w"))
    desc = dict(baumeister1=descriptives("baumeister1", b1, rows2=b2), bayer=descriptives("bayer", by))
    json.dump(desc, open(OUT / "descriptives.json", "w"), indent=1)
    print(json.dumps(desc, indent=1))


def stage_drive():
    parts = json.load(open(OUT / "parts.json"))
    variants = [(1.0, None), (0.0, None), (1.0, "tracked"), (1.0, "full")]
    drives = drive_all(parts, variants)
    ser = {f"{n}|{p}|{g}|{prec}": d for (n, p, g, prec), d in drives.items()}
    json.dump(ser, open(OUT / "drives.json", "w"))


def load_drives():
    raw = json.load(open(OUT / "drives.json"))
    out = {}
    for key, d in raw.items():
        n, p, g, prec = key.split("|")
        out[(n, p, float(g), None if prec == "None" else prec)] = d
    return out


def stage_eval():
    parts = json.load(open(OUT / "parts.json")); drives = load_drives()
    featsets = ["base", "qf", "base+qf", "base+ch", "base+qf+ch"]
    report = {}
    for name in ("baumeister1", "bayer"):
        report[name] = {}
        for g, tag in ((1.0, "gated"), (0.0, "inert")):
            rows = build_rows(parts[name], drives, g, None, name)
            s = heldout_eval(rows, featsets)
            r = {fs: clean(s[fs]) for fs in featsets}
            r["diff_base+qf_minus_base"] = dict(mean=s["base+qf"]["ll_gain_mean"] - s["base"]["ll_gain_mean"], ci=diff_ci(s["base+qf"], s["base"]))
            r["diff_base+qf+ch_minus_base"] = dict(mean=s["base+qf+ch"]["ll_gain_mean"] - s["base"]["ll_gain_mean"], ci=diff_ci(s["base+qf+ch"], s["base"]))
            r["diff_qf_minus_rates"] = dict(mean=s["qf"]["ll_gain_mean"], ci=boot_ci(s["qf"]["_ll_by_pid"], lambda xs: float(np.mean(xs))))
            r["n_rows"] = len(rows); r["n_pids"] = len({x["pid"] for x in rows})
            # raw association: mean q(f) by reported orientation, and dominant-frame confusion
            qf = np.array([x["qf"] for x in rows]); y = np.array([x["y"] for x in rows]); dom = np.array([x["dom"] for x in rows])
            r["mean_qf_by_orientation"] = {LABELS[k + 1]: dict(q_past=float(qf[y == k, 0].mean()), q_present=float(qf[y == k, 1].mean()), q_future=float(1 - qf[y == k].sum(axis=1).mean())) for k in range(3)}
            conf = np.zeros((3, 3), int)
            for yy, dd in zip(y, dom): conf[yy, dd] += 1
            r["dominant_frame_confusion_rows_reported_cols_model"] = conf.tolist()
            r["dominant_frame_accuracy"] = float((y == dom).mean()); r["majority_accuracy"] = float(np.bincount(y).max() / len(y))
            report[name][tag] = r
            print(name, tag, json.dumps({k: v for k, v in r.items() if k.startswith("diff") or k in ("dominant_frame_accuracy", "majority_accuracy")}))
    json.dump(report, open(OUT / "eval.json", "w"), indent=1)


def stage_precision():
    parts = json.load(open(OUT / "parts.json")); drives = load_drives()
    report = {}
    for name in ("baumeister1", "bayer"):
        report[name] = {}
        # (a) orientation prediction with tracker states
        rows_full = build_rows(parts[name], drives, 1.0, "full", name)
        rows_tr = build_rows(parts[name], drives, 1.0, "tracked", name)
        rows_no = build_rows(parts[name], drives, 1.0, None, name)
        for tag, rows in (("full", rows_full), ("tracked", rows_tr), ("no_affect", rows_no)):
            fs = ["base", "base+qf", "base+qf+L"] if tag != "no_affect" else ["base", "base+qf"]
            s = heldout_eval(rows, fs)
            r = {f: clean(s[f]) for f in fs}
            if "base+qf+L" in fs:
                r["diff_L_minus_base+qf"] = dict(mean=s["base+qf+L"]["ll_gain_mean"] - s["base+qf"]["ll_gain_mean"], ci=diff_ci(s["base+qf+L"], s["base+qf"]))
            r["diff_qf_minus_base"] = dict(mean=s["base+qf"]["ll_gain_mean"] - s["base"]["ll_gain_mean"], ci=diff_ci(s["base+qf"], s["base"]))
            report[name][tag] = r
        # (b) persistence of the model's dominant frame vs empirical orientation persistence
        pers = {}
        for tag, prec in (("full", "full"), ("tracked", "tracked"), ("no_affect", None)):
            same = []; diff = []; lag1 = []
            for pid, seq in parts[name].items():
                d = drives[(name, pid, 1.0, prec)]
                doms = [x["dom"] for x in d]
                for i in range(len(seq) - 1):
                    if seq[i + 1]["d"] != seq[i]["d"]:
                        continue
                    (same if doms[i] == 2 else diff).append(int(doms[i + 1] == 2))
                    lag1.append((doms[i] == doms[i + 1]))
            pers[tag] = dict(P_future_next_given_future=float(np.mean(same)) if same else None,
                             P_future_next_given_not=float(np.mean(diff)) if diff else None,
                             lag1_same_frame=float(np.mean(lag1)), n=len(lag1),
                             frame_rates=[float(np.mean([x["dom"] == f for pid in parts[name] for x in drives[(name, pid, 1.0, prec)]])) for f in range(3)])
        desc = json.load(open(OUT / "descriptives.json"))[name]
        pers["empirical"] = dict(P_future_next_given_future=desc.get("P_future_t1_given_future_t"), P_future_next_given_not=desc.get("P_future_t1_given_not_future_t"),
                                 P_past_next_given_past=desc.get("P_past_t1_given_past_t"), P_past_next_given_not=desc.get("P_past_t1_given_not_past_t"))
        report[name]["persistence"] = pers
        # tracker dynamics: mean L by reported orientation
        Ls = defaultdict(list)
        for x in rows_full: Ls[LABELS[x["y"] + 1]].append(x["L"])
        report[name]["mean_L_by_orientation_full"] = {k: [float(v) for v in np.mean(np.array(vs), axis=0)] for k, vs in Ls.items()}
        gam = defaultdict(list)
        for pid in parts[name]:
            for x in drives[(name, pid, 1.0, "full")]: gam[x["dom"]].append(x["gamma"])
        report[name]["gamma_by_dominant_frame_full"] = {LABELS[f + 1]: float(np.mean(v)) for f, v in gam.items()}
        print(name, json.dumps(pers))
    json.dump(report, open(OUT / "precision.json", "w"), indent=1)


def stage_figures():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    desc = json.load(open(OUT / "descriptives.json")); ev = json.load(open(OUT / "eval.json")); pr = json.load(open(OUT / "precision.json"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    ax = axes[0]
    for i, (name, lab, scale) in enumerate((("baumeister1", "Baumeister S1 (happy/sad, -3..3)", 1.0), ("bayer", "Bayer (1..5)", 1.0))):
        m = desc[name]["valence_mean_by_orientation"]
        ax.plot(["past", "present", "future"], [m[k]["mean"] for k in ("past", "present", "future")], marker="o", label=lab)
    ax.set_title("Mean valence by reported orientation"); ax.legend(fontsize=7)
    ax = axes[1]
    for name, off in (("baumeister1", -0.15), ("bayer", 0.15)):
        for tag, c in (("gated", "C0"), ("inert", "C1")):
            r = ev[name][tag]
            xs = ["base", "base+qf", "base+qf+ch", "qf"]
            ys = [r[x]["ll_gain_mean"] for x in xs]
            ax.bar(np.arange(len(xs)) + off + (0.0 if tag == "gated" else 0.07), ys, width=0.07, color=c, alpha=0.9 if name == "baumeister1" else 0.5,
                   label=f"{name} {tag}")
    ax.set_xticks(range(4)); ax.set_xticklabels(["valence\nlags", "+ q(f)", "+ q(f)+ch", "q(f) only"], fontsize=8)
    ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("held-out LL gain over base rates (nats/signal)"); ax.legend(fontsize=6); ax.set_title("Held-out prediction of orientation")
    ax = axes[2]
    for name, mk in (("baumeister1", "o"), ("bayer", "s")):
        p = pr[name]["persistence"]
        emp = p["empirical"]
        ax.scatter(["empirical"], [emp["P_future_next_given_future"] - emp["P_future_next_given_not"]], marker=mk, color="k", label=f"{name} data")
        for tag, c in (("no_affect", "C2"), ("tracked", "C1"), ("full", "C0")):
            q = p[tag]
            if q["P_future_next_given_future"] is not None and q["P_future_next_given_not"] is not None:
                ax.scatter([tag], [q["P_future_next_given_future"] - q["P_future_next_given_not"]], marker=mk, color=c)
    ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("P(future at t+1 | future at t) - P(future | not)"); ax.set_title("Orientation persistence"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "orientation_summary.png", dpi=160)
    # transition heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for ax, name in zip(axes, ("baumeister1", "bayer")):
        T = np.array(desc[name]["transition_rownorm"])
        im = ax.imshow(T, vmin=0, vmax=1, cmap="Blues")
        for i in range(3):
            for j in range(3): ax.text(j, i, f"{T[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(["past", "present", "future"]); ax.set_yticklabels(["past", "present", "future"])
        ax.set_xlabel("orientation at t+1"); ax.set_ylabel("orientation at t"); ax.set_title(name)
    plt.tight_layout(); plt.savefig(FIG / "orientation_transitions.png", dpi=160)
    print("figures written")


if __name__ == "__main__":
    stages = sys.argv[1:] or ["all"]
    if "all" in stages: stages = ["load", "drive", "eval", "precision", "figures"]
    for s in stages:
        {"load": stage_load, "drive": stage_drive, "eval": stage_eval, "precision": stage_precision, "figures": stage_figures}[s]()
