"""ESM prediction, round 3 (2026-09-28): can the model predict better than strong baselines?

Round 2 (esm_eval_v2.py) established parity between the frame model and a
two-lag regression on next-beep valence. This script asks whether any
principled extension of the model gives a REAL participant-held-out gain over
STRONG baselines, and reports the answer either way. Everything is evaluated
with the same participant folds, the same preprocessing, and participant
bootstrap CIs on R2 differences.

Baselines (all train-fit on the training participants of each fold):
  persistence; AR2 (direct h-step on v_t, v_{t-1}); AR6 (lags 1..6, missing
  lags filled with the last available lag plus an indicator); AR6+event (event
  pleasantness and its two lags, Geschwind only); "kitchen" = ridge on lags,
  events, time-of-day, first-beep-of-day; MS-AR(2), a two-regime switching
  regression fitted by EM with a causal (filtered) regime posterior at test.

Model-based predictors:
  full (frame-gated, affine-calibrated expected valence); inert (g=0);
  transition-gated (g_B=4); channels-only; "aug" = kitchen ridge plus the
  model's state features (h-step expectation, three channels, q(f), state
  entropy, effective rho_pos); aug ablations dropping the frame features, the
  channels, or everything but the h-step expectation.

Protocols:
  P  pooled: parameters and coefficients from other participants only.
  A  adaptation: the first segment of a held-out participant (period 0 on
     Geschwind, the first half of beeps on osf) is used to adapt per-participant
     coefficients (ridge shrinkage toward the pooled fit) and, for the model, to
     select per-participant generative parameters from a 36-point grid; scoring
     on the rest. Baselines get exactly the same adaptation.

Targets: valence level at h = 1..6; valence change after an event beep
(Geschwind), and after the largest-magnitude events (top quartile of |e|).

Writes reviews/esm_eval_v3.md, reviews/esm_eval_v3.json, figures/fig_esm_v3.png.
Run:  python esm_eval_v3.py --workers 20
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from agent import Agent
from generative_model import EPS, N_FRAMES, build_model
import empirical_rebuild as er
from empirical_rebuild import _bin_e, _bin_v, same_segment, target

ROOT = Path(__file__).resolve().parent
HORIZONS = (1, 2, 3, 4, 5, 6)
NLAG = 6
K = M = 8
N_BOOT = 1000
GRID = dict(pi_pos=[2.0, 3.0, 4.0], valence_inertia=[0.35, 0.5, 0.65], omega_e=[3.0, 5.0],
            asym=[(1.0, 1.0), (0.6, 1.6)])
COMBOS = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
VARIANTS = {"full": dict(g=1.0, gt=0.0), "ungated": dict(g=0.0, gt=0.0), "trans_g4": dict(g=1.0, gt=4.0)}
LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0]
FEAT_MODEL = ["pred", "v_model", "v_reward", "v_action", "qf_past", "qf_pres", "qf_fut", "arousal", "rho_eff"]


# ── data ───────────────────────────────────────────────────────────────
def load_osf():
    """osf.io/83cfk with day and hour recovered from the timestamp."""
    from esm_replication import DATA, POS, NEG, _f
    rows = defaultdict(list)
    with open(ROOT / DATA, encoding="utf-8") as fh:
        r = csv.reader(fh)
        next(r)
        for row in r:
            pid, ts = row[0], _f(row[1])
            pos = [_f(row[i]) for i in POS if _f(row[i]) is not None]
            neg = [_f(row[i]) for i in NEG if _f(row[i]) is not None]
            if not pos or not neg or ts is None:
                continue
            v = np.mean(pos) - np.mean(neg)
            t = dt.datetime.fromtimestamp(ts, dt.timezone.utc)
            rows[pid].append((ts, float(np.clip((v + 100) / 200, 0.0, 1.0)), t.toordinal(), t.hour + t.minute / 60))
    parts = {}
    for pid, seq in rows.items():
        seq.sort(key=lambda t: t[0])
        parts[pid] = [dict(v=vn, e=None, w=None, p=0, d=day, hour=hr) for _, vn, day, hr in seq]
    return parts


def add_time_features(parts, geschwind):
    for pid, seq in parts.items():
        groups = defaultdict(list)
        for i, b in enumerate(seq):
            groups[(b.get("p"), b.get("d"))].append(i)
        for key, idx in groups.items():
            for rank, i in enumerate(idx):
                seq[i]["tod"] = rank / max(len(idx) - 1, 1)
                seq[i]["first"] = 1.0 if rank == 0 else 0.0
        if not geschwind:
            for b in seq:
                b["tod"] = (b["hour"] - 9.0) / 9.0


# ── generative model drive ────────────────────────────────────────────
def drive(seq, params, variant, seed):
    spec = VARIANTS[variant]
    c_pos, c_neg = params["asym"]
    model = build_model(K=K, M=M, pi_pos=params["pi_pos"], omega_e=params["omega_e"], gamma=16.0,
                        c_pos=c_pos, c_neg=c_neg, neg_val_precision=1.0,
                        valence_inertia=params["valence_inertia"])
    agent = Agent(model, gamma=16.0, pi_pos=params["pi_pos"], omega_e=params["omega_e"],
                  c_pos=c_pos, c_neg=c_neg, neg_val_precision=1.0,
                  valence_inertia=params["valence_inertia"], counterfactual_horizon=1,
                  adaptive_counterfactual_horizon=False, frame_gain=spec["g"], frame_clamp=None,
                  frame_transition_gain=spec["gt"], seed=seed)
    v_axis = np.arange(K)
    out = []
    prev = None
    for beep in seq:
        if prev is not None and beep.get("p") != prev:
            agent.reset()
        prev = beep.get("p")
        _, info = agent.step([_bin_e(beep["e"]), 1, _bin_v(beep["v"], K)])
        pi = info["pi"]
        B = sum(pi[a] * model.B[a] for a in range(len(pi)))
        q = info["beliefs"].copy()
        preds = []
        for h in range(1, max(HORIZONS) + 1):
            q = B @ q
            q = np.maximum(q, EPS)
            q /= q.sum()
            vm = q.reshape(K, M, N_FRAMES).sum(axis=(1, 2))
            preds.append(float(vm @ v_axis / (K - 1)))
        qf = info["beliefs"].reshape(K, M, N_FRAMES).sum(axis=(0, 1))
        out.append(dict(preds=preds, v_model=info["v_model"], v_reward=info["v_reward"],
                        v_action=info["v_action"], qf=[float(x) for x in qf],
                        arousal=float(info["arousal_norm"]), rho_eff=float(info["pi_pos_eff"])))
    return out


def _job(a):
    pid, seq, gi, variant, seed = a
    return pid, gi, variant, drive(seq, COMBOS[gi], variant, seed)


def run_jobs(jobs, workers):
    out = {}
    if workers > 1 and len(jobs) > 8:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for pid, gi, v, d in ex.map(_job, jobs, chunksize=8):
                out[(pid, gi, v)] = d
    else:
        for j in jobs:
            pid, gi, v, d = _job(j)
            out[(pid, gi, v)] = d
    return out


# ── records and features ──────────────────────────────────────────────
def build_records(parts, has_event):
    recs = []
    for pid in sorted(parts):
        seq = parts[pid]
        for i, b in enumerate(seq):
            lags, miss = [], []
            for L in range(1, NLAG + 1):
                j = i - L
                ok = j >= 0 and same_segment(seq, j, i)
                lags.append(seq[j]["v"] if ok else (lags[-1] if lags else b["v"]))
                miss.append(0.0 if ok else 1.0)
            e = b["e"] if (has_event and b["e"] is not None) else 0.0
            elags = []
            for L in (1, 2):
                j = i - L
                ok = j >= 0 and same_segment(seq, j, i) and has_event and seq[j]["e"] is not None
                elags.append(seq[j]["e"] if ok else 0.0)
            r = dict(pid=pid, i=i, v=b["v"], e=e, elags=elags, lags=lags, miss=miss,
                     tod=b.get("tod", 0.5), first=b.get("first", 0.0), seg=b.get("p"))
            for h in HORIZONS:
                r[f"y{h}"] = target(seq, i, h)
            recs.append(r)
    return recs


def feats(r, kind, mf=None):
    """Feature vector without intercept."""
    if kind == "ar2":
        return [r["v"], r["lags"][0]]
    if kind == "ar6":
        return [r["v"]] + r["lags"][:NLAG - 1] + r["miss"][:NLAG - 1]
    if kind == "ar6e":
        return feats(r, "ar6") + [r["e"], max(r["e"], 0.0)] + r["elags"]
    if kind == "kitchen":
        return feats(r, "ar6e") + [r["tod"], r["tod"] ** 2, r["first"]]
    if kind.startswith("aug"):
        base = feats(r, "kitchen")
        sel = {"aug": FEAT_MODEL, "aug_noframe": [f for f in FEAT_MODEL if not f.startswith("qf")],
               "aug_nochan": [f for f in FEAT_MODEL if not f.startswith("v_")],
               "aug_predonly": ["pred"]}[kind]
        return base + [mf[f] for f in sel]
    raise KeyError(kind)


def model_feats(d, h):
    return dict(pred=d["preds"][h - 1], v_model=d["v_model"], v_reward=d["v_reward"], v_action=d["v_action"],
                qf_past=d["qf"][0], qf_pres=d["qf"][1], qf_fut=d["qf"][2], arousal=d["arousal"], rho_eff=d["rho_eff"])


# ── linear tools ───────────────────────────────────────────────────────
class Ridge:
    def __init__(self, lam):
        self.lam = lam

    def fit(self, X, y, w=None, prior=None):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        w = np.ones(len(y)) if w is None else np.asarray(w, float)
        Zw = Z * w[:, None]
        self.b0 = float(np.sum(w * y) / np.sum(w))
        yc = y - self.b0
        A = Zw.T @ Z + self.lam * np.eye(Z.shape[1])
        rhs = Zw.T @ yc
        if prior is not None:           # shrink toward prior standardised coefficients
            rhs = rhs + self.lam * prior
        self.beta = np.linalg.solve(A, rhs)
        return self

    def predict(self, X):
        Z = (np.asarray(X, float) - self.mu) / self.sd
        return self.b0 + Z @ self.beta


def r2(p, y):
    y = np.asarray(y, float)
    sst = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if sst < EPS else 1 - float(np.sum((y - np.asarray(p, float)) ** 2)) / sst


def pick_lambda(X, y, groups, lams=LAMBDAS, inner=4):
    """Inner participant-grouped CV over the ridge penalty."""
    g = np.asarray(groups)
    ug = np.unique(g)
    rng = np.random.RandomState(1)
    rng.shuffle(ug)
    fold_of = {p: k % inner for k, p in enumerate(ug)}
    f = np.array([fold_of[p] for p in g])
    best, best_l = -np.inf, lams[0]
    for lam in lams:
        sc = []
        for k in range(inner):
            tr, te = f != k, f == k
            if te.sum() < 5:
                continue
            m = Ridge(lam).fit(X[tr], y[tr])
            sc.append(r2(m.predict(X[te]), y[te]))
        s = float(np.nanmean(sc))
        if s > best:
            best, best_l = s, lam
    return best_l


# ── two-regime switching regression ───────────────────────────────────
class MSAR:
    """Two-regime Markov-switching direct-h regression on (1, v_t, v_{t-1}).
    Regimes learned by EM on the one-step residual structure of the training
    participants; test-time regime posterior is causal (filtered)."""

    def __init__(self, n_iter=40):
        self.n_iter = n_iter

    @staticmethod
    def _chains(rows):
        chains, cur, last = [], [], None
        for r in rows:
            key = (r["pid"], r["seg"])
            if last is not None and (key != last or r["i"] != prev_i + 1):
                chains.append(cur)
                cur = []
            cur.append(r)
            last, prev_i = key, r["i"]
        if cur:
            chains.append(cur)
        return chains

    def _emis(self, X, y):
        ll = np.zeros((len(y), 2))
        for s in range(2):
            mu = X @ self.beta[s]
            ll[:, s] = -0.5 * np.log(2 * np.pi * self.var[s]) - 0.5 * (y - mu) ** 2 / self.var[s]
        return ll

    def fit(self, rows):
        rows = [r for r in rows if r["y1"] is not None]
        X = np.array([[1, r["v"], r["lags"][0]] for r in rows])
        y = np.array([r["y1"] for r in rows])
        base, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ base
        hi = np.abs(res) > np.median(np.abs(res))
        self.beta = [base.copy(), base.copy()]
        self.var = [float(np.var(res[~hi])) + 1e-6, float(np.var(res[hi])) + 1e-6]
        self.P = np.array([[0.9, 0.1], [0.1, 0.9]])
        self.p0 = np.array([0.5, 0.5])
        chains = self._chains(rows)
        idx_of = {id(r): k for k, r in enumerate(rows)}
        for _ in range(self.n_iter):
            ll = self._emis(X, y)
            gam = np.zeros((len(rows), 2))
            xi = np.zeros((2, 2))
            p0 = np.zeros(2)
            for ch in chains:
                ids = [idx_of[id(r)] for r in ch]
                L = np.exp(ll[ids] - ll[ids].max(1, keepdims=True))
                T = len(ids)
                a = np.zeros((T, 2))
                c = np.zeros(T)
                a[0] = self.p0 * L[0]
                c[0] = a[0].sum()
                a[0] /= c[0]
                for t in range(1, T):
                    a[t] = (a[t - 1] @ self.P) * L[t]
                    c[t] = a[t].sum()
                    a[t] /= c[t]
                b = np.ones((T, 2))
                for t in range(T - 2, -1, -1):
                    b[t] = (self.P @ (L[t + 1] * b[t + 1])) / c[t + 1]
                g = a * b
                g /= g.sum(1, keepdims=True)
                gam[ids] = g
                p0 += g[0]
                for t in range(T - 1):
                    xi += (a[t][:, None] * self.P * (L[t + 1] * b[t + 1])[None, :]) / c[t + 1]
            self.p0 = p0 / p0.sum()
            self.P = xi / xi.sum(1, keepdims=True)
            for s in range(2):
                w = gam[:, s] + 1e-9
                Xw = X * w[:, None]
                self.beta[s] = np.linalg.solve(X.T @ Xw + 1e-6 * np.eye(3), Xw.T @ y)
                self.var[s] = float(np.sum(w * (y - X @ self.beta[s]) ** 2) / w.sum()) + 1e-6
        self.gam = {id(r): gam[k] for k, r in enumerate(rows)}
        self._rows = rows
        self.betah = {}
        for h in HORIZONS:
            rh = [r for r in rows if r[f"y{h}"] is not None]
            Xh = np.array([[1, r["v"], r["lags"][0]] for r in rh])
            yh = np.array([r[f"y{h}"] for r in rh])
            self.betah[h] = []
            for s in range(2):
                w = np.array([self.gam[id(r)][s] for r in rh]) + 1e-9
                Xw = Xh * w[:, None]
                self.betah[h].append(np.linalg.solve(Xh.T @ Xw + 1e-6 * np.eye(3), Xw.T @ yh))
        return self

    def filtered(self, rows):
        """Causal regime posterior at each row, using targets of earlier rows only."""
        out = {}
        for ch in self._chains(rows):
            a = self.p0.copy()
            for k, r in enumerate(ch):
                pred = a @ self.P if k > 0 else a
                out[id(r)] = pred.copy()
                if r["y1"] is None:
                    a = pred
                    continue
                x = np.array([1, r["v"], r["lags"][0]])
                ll = np.array([-0.5 * np.log(2 * np.pi * self.var[s]) - 0.5 * (r["y1"] - x @ self.beta[s]) ** 2 / self.var[s]
                               for s in range(2)])
                L = np.exp(ll - ll.max())
                a = pred * L
                a /= a.sum()
        return out

    def predict(self, rows, h, post):
        X = np.array([[1, r["v"], r["lags"][0]] for r in rows])
        w = np.array([post[id(r)] for r in rows])
        return sum(w[:, s] * (X @ self.betah[h][s]) for s in range(2))


# ── evaluation ─────────────────────────────────────────────────────────
def bootstrap_diff(rows_by_pid, a, b, seed=0):
    """rows_by_pid: {pid: array (n, 3) of y, pred_a, pred_b}."""
    pids = sorted(rows_by_pid)
    if not pids:
        return dict(point=float("nan"), lo=float("nan"), hi=float("nan"), p_le_0=float("nan"))

    def diff(sel):
        arr = np.concatenate([rows_by_pid[p] for p in sel])
        return r2(arr[:, 1], arr[:, 0]) - r2(arr[:, 2], arr[:, 0])
    point = diff(pids)
    rng = np.random.RandomState(seed)
    boots = np.array([diff(rng.choice(pids, len(pids), replace=True)) for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(point=float(point), lo=float(lo), hi=float(hi), p_le_0=float(np.mean(boots <= 0)))


def evaluate(name, parts, has_event, workers, log):
    t0 = time.time()
    pids = sorted(parts)
    seeds = {pid: 500 + i for i, pid in enumerate(pids)}
    recs = build_records(parts, has_event)
    first = {}
    for k, r in enumerate(recs):
        first.setdefault(r["pid"], k)
    rng = np.random.RandomState(0)
    order = list(pids)
    rng.shuffle(order)
    fold = {p: k % 5 for k, p in enumerate(order)}

    # stage 1: the gated model on the whole grid (needed for protocol A anyway)
    jobs = [(pid, parts[pid], gi, "full", seeds[pid]) for pid in pids for gi in range(len(COMBOS))]
    log(f"[{name}] stage 1: {len(jobs)} drives")
    driven = run_jobs(jobs, workers)

    def mrow(r, gi, v="full"):
        return driven[(r["pid"], gi, v)][r["i"]]

    # nested pooled selection per fold: mean R2 over horizons of affine-calibrated full model
    selected = []
    for k in range(5):
        tr = [r for r in recs if fold[r["pid"]] != k]
        best, bgi = -np.inf, 0
        for gi in range(len(COMBOS)):
            sc = []
            for h in HORIZONS:
                rows = [r for r in tr if r[f"y{h}"] is not None]
                x = np.array([mrow(r, gi)["preds"][h - 1] for r in rows])
                y = np.array([r[f"y{h}"] for r in rows])
                c = np.polyfit(x, y, 1)
                sc.append(r2(np.polyval(c, x), y))
            if np.mean(sc) > best:
                best, bgi = float(np.mean(sc)), gi
        selected.append(bgi)
    log(f"[{name}] selected: {[COMBOS[g] for g in selected]}")
    need = sorted(set(selected))
    jobs = [(pid, parts[pid], gi, v, seeds[pid]) for pid in pids for gi in need for v in ("ungated", "trans_g4")]
    log(f"[{name}] stage 2: {len(jobs)} drives")
    driven.update(run_jobs(jobs, workers))

    names_P = ["persistence", "ar2", "ar6", "ar6e", "kitchen", "msar2", "model_full", "model_inert", "model_trans_g4",
               "channels_only", "aug", "aug_noframe", "aug_nochan", "aug_predonly"]
    if not has_event:
        names_P = [n for n in names_P if n != "ar6e"]
    oof = {"P": defaultdict(dict), "A": defaultdict(dict)}   # oof[proto][(name,h)][row_index] = pred
    nll_rows = defaultdict(list)   # Gaussian NLL per held-out row, sigma^2 from training residuals
    lam_used = defaultdict(list)

    # ── protocol P ──
    for k in range(5):
        gi = selected[k]
        tr = [r for r in recs if fold[r["pid"]] != k]
        te = [r for r in recs if fold[r["pid"]] == k]
        ms = MSAR().fit(tr)
        post = ms.filtered(te)
        for h in HORIZONS:
            trh = [r for r in tr if r[f"y{h}"] is not None]
            teh = [r for r in te if r[f"y{h}"] is not None]
            ytr = np.array([r[f"y{h}"] for r in trh])
            gtr = [r["pid"] for r in trh]
            P, PT = {}, {}
            P["persistence"] = np.array([r["v"] for r in teh])
            PT["persistence"] = np.array([r["v"] for r in trh])
            P["msar2"] = ms.predict(teh, h, post)
            PT["msar2"] = ms.predict(trh, h, ms.filtered(trh))
            for kind in ("ar2", "ar6", "ar6e", "kitchen"):
                if kind == "ar6e" and not has_event:
                    continue
                Xtr = np.array([feats(r, kind) for r in trh])
                Xte = np.array([feats(r, kind) for r in teh])
                lam = 0.0 if kind == "ar2" else pick_lambda(Xtr, ytr, gtr)
                lam_used[kind].append(lam)
                mdl = Ridge(max(lam, 1e-6)).fit(Xtr, ytr)
                P[kind], PT[kind] = mdl.predict(Xte), mdl.predict(Xtr)
            for nm, v in (("model_full", "full"), ("model_inert", "ungated"), ("model_trans_g4", "trans_g4")):
                x = np.array([mrow(r, gi, v)["preds"][h - 1] for r in trh])
                c = np.polyfit(x, ytr, 1)
                P[nm] = np.polyval(c, np.array([mrow(r, gi, v)["preds"][h - 1] for r in teh]))
                PT[nm] = np.polyval(c, x)
            Xtr = np.array([[r["v"], mrow(r, gi)["v_model"], mrow(r, gi)["v_reward"], mrow(r, gi)["v_action"]] for r in trh])
            Xte = np.array([[r["v"], mrow(r, gi)["v_model"], mrow(r, gi)["v_reward"], mrow(r, gi)["v_action"]] for r in teh])
            mdl = Ridge(1e-6).fit(Xtr, ytr)
            P["channels_only"], PT["channels_only"] = mdl.predict(Xte), mdl.predict(Xtr)
            for kind in ("aug", "aug_noframe", "aug_nochan", "aug_predonly"):
                Xtr = np.array([feats(r, kind, model_feats(mrow(r, gi), h)) for r in trh])
                Xte = np.array([feats(r, kind, model_feats(mrow(r, gi), h)) for r in teh])
                lam = pick_lambda(Xtr, ytr, gtr)
                lam_used[kind].append(lam)
                mdl = Ridge(lam).fit(Xtr, ytr)
                P[kind], PT[kind] = mdl.predict(Xte), mdl.predict(Xtr)
            for nm, p in P.items():
                s2 = float(np.mean((ytr - PT[nm]) ** 2)) + 1e-9   # training residual variance
                for r, pv in zip(teh, p):
                    oof["P"][(nm, h)][first[r["pid"]] + r["i"]] = float(pv)
                    nll_rows[(nm, h)].append(0.5 * np.log(2 * np.pi * s2) + (r[f"y{h}"] - float(pv)) ** 2 / (2 * s2))
    log(f"[{name}] protocol P done ({time.time() - t0:.0f}s)")

    # ── protocol A: adaptation on the first segment of each held-out participant ──
    def split_adapt(pid):
        seq = parts[pid]
        rows = [r for r in recs if r["pid"] == pid]
        if has_event:
            segs = sorted({b.get("p") for b in seq})
            if len(segs) == 2:
                return [r for r in rows if r["seg"] == segs[0]], [r for r in rows if r["seg"] == segs[1]]
        n = len(rows) // 2
        return rows[:n], rows[n:]

    names_A = ["A_kitchen_pooled", "A_kitchen", "A_ar2", "A_msar2_pooled", "A_model_pooled", "A_model", "A_model_inert",
               "A_aug_pooled", "A_aug", "A_aug_noframe"]
    lam_adapt = {}
    for k in range(5):
        gi = selected[k]
        tr = [r for r in recs if fold[r["pid"]] != k]
        te_pids = [p for p in pids if fold[p] == k]
        # choose the adaptation shrinkage on training participants by simulating the protocol
        ms = MSAR().fit(tr)
        for h in HORIZONS:
            trh = [r for r in tr if r[f"y{h}"] is not None]
            ytr = np.array([r[f"y{h}"] for r in trh])
            gtr = [r["pid"] for r in trh]
            pooled = {}
            for kind in ("kitchen", "aug", "aug_noframe"):
                mf = (lambda r: model_feats(mrow(r, gi), h)) if kind != "kitchen" else (lambda r: None)
                Xtr = np.array([feats(r, kind, mf(r)) for r in trh])
                pooled[kind] = Ridge(pick_lambda(Xtr, ytr, gtr)).fit(Xtr, ytr)
            x = np.array([mrow(r, gi)["preds"][h - 1] for r in trh])
            pooled_aff = np.polyfit(x, ytr, 1)
            x = np.array([mrow(r, gi, "ungated")["preds"][h - 1] for r in trh])
            pooled_aff_in = np.polyfit(x, ytr, 1)
            # adaptation penalty: chosen once per (fold, h) on training participants
            if (k, h) not in lam_adapt:
                tr_pids = sorted({r["pid"] for r in tr})
                rng2 = np.random.RandomState(7)
                sub = list(rng2.choice(tr_pids, min(30, len(tr_pids)), replace=False))
                best, bl = -np.inf, 1.0
                for lam in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
                    ys, ps = [], []
                    for p in sub:
                        ad, ev = split_adapt(p)
                        ad = [r for r in ad if r[f"y{h}"] is not None]
                        ev = [r for r in ev if r[f"y{h}"] is not None]
                        if len(ad) < 8 or len(ev) < 4:
                            continue
                        m = pooled["kitchen"]
                        Xa = np.array([feats(r, "kitchen") for r in ad])
                        ya = np.array([r[f"y{h}"] for r in ad])
                        loc = Ridge(lam)
                        loc.mu, loc.sd = m.mu, m.sd
                        Za = (Xa - m.mu) / m.sd
                        ya_c = ya - m.b0
                        A_ = Za.T @ Za + lam * np.eye(Za.shape[1])
                        loc.beta = np.linalg.solve(A_, Za.T @ ya_c + lam * m.beta)
                        loc.b0 = m.b0 + float(np.mean(ya - (m.b0 + Za @ loc.beta))) * (len(ad) / (len(ad) + lam))
                        Xe = np.array([feats(r, "kitchen") for r in ev])
                        ps.extend(loc.predict(Xe))
                        ys.extend(r[f"y{h}"] for r in ev)
                    s = r2(ps, ys)
                    if s > best:
                        best, bl = s, lam
                lam_adapt[(k, h)] = bl
            lam = lam_adapt[(k, h)]

            def adapt(m, kind, ad, mf=None):
                Xa = np.array([feats(r, kind, mf(r) if mf else None) for r in ad])
                ya = np.array([r[f"y{h}"] for r in ad])
                loc = Ridge(lam)
                loc.mu, loc.sd = m.mu, m.sd
                Za = (Xa - m.mu) / m.sd
                A_ = Za.T @ Za + lam * np.eye(Za.shape[1])
                loc.beta = np.linalg.solve(A_, Za.T @ (ya - m.b0) + lam * m.beta)
                loc.b0 = m.b0 + float(np.mean(ya - (m.b0 + Za @ loc.beta))) * (len(ad) / (len(ad) + lam))
                return loc

            def adapt_affine(c, x, y):
                # shrunk per-participant affine calibration of a model expectation
                n = len(x)
                w = n / (n + lam)
                cl = np.polyfit(x, y, 1) if n >= 4 else c
                return (1 - w) * np.asarray(c) + w * np.asarray(cl)

            for pid in te_pids:
                ad, ev = split_adapt(pid)
                ad = [r for r in ad if r[f"y{h}"] is not None]
                ev = [r for r in ev if r[f"y{h}"] is not None]
                if len(ad) < 8 or len(ev) < 4:
                    continue
                P = {}
                Xe = np.array([feats(r, "kitchen") for r in ev])
                P["A_kitchen_pooled"] = pooled["kitchen"].predict(Xe)
                P["A_kitchen"] = adapt(pooled["kitchen"], "kitchen", ad).predict(Xe)
                Xa2 = np.array([feats(r, "ar2") for r in ad])
                ya = np.array([r[f"y{h}"] for r in ad])
                P["A_ar2"] = Ridge(1e-3).fit(Xa2, ya).predict(np.array([feats(r, "ar2") for r in ev]))
                P["A_msar2_pooled"] = ms.predict(ev, h, ms.filtered(ad + ev))
                # pooled model with pooled calibration
                xe = np.array([mrow(r, gi)["preds"][h - 1] for r in ev])
                P["A_model_pooled"] = np.polyval(pooled_aff, xe)
                # per-participant grid selection on the adaptation segment (shrunk affine)
                best, bgi, bc = -np.inf, gi, pooled_aff
                for g2 in range(len(COMBOS)):
                    xa = np.array([mrow(r, g2)["preds"][h - 1] for r in ad])
                    c = adapt_affine(pooled_aff, xa, ya)
                    s = r2(np.polyval(c, xa), ya)
                    if s > best:
                        best, bgi, bc = s, g2, c
                P["A_model"] = np.polyval(bc, np.array([mrow(r, bgi)["preds"][h - 1] for r in ev]))
                xa = np.array([mrow(r, gi, "ungated")["preds"][h - 1] for r in ad])
                c = adapt_affine(pooled_aff_in, xa, ya)
                P["A_model_inert"] = np.polyval(c, np.array([mrow(r, gi, "ungated")["preds"][h - 1] for r in ev]))
                mf = lambda r: model_feats(mrow(r, gi), h)
                Xe = np.array([feats(r, "aug", mf(r)) for r in ev])
                P["A_aug_pooled"] = pooled["aug"].predict(Xe)
                P["A_aug"] = adapt(pooled["aug"], "aug", ad, mf).predict(Xe)
                Xe = np.array([feats(r, "aug_noframe", mf(r)) for r in ev])
                P["A_aug_noframe"] = adapt(pooled["aug_noframe"], "aug_noframe", ad, mf).predict(Xe)
                for nm, p in P.items():
                    for r, pv in zip(ev, p):
                        oof["A"][(nm, h)][first[r["pid"]] + r["i"]] = float(pv)
    log(f"[{name}] protocol A done ({time.time() - t0:.0f}s)")

    # ── scoring ──
    def rows_for(proto, nm, h, subset=None, change=False):
        by = defaultdict(list)
        d = oof[proto].get((nm, h), {})
        for r in recs:
            key = first[r["pid"]] + r["i"]
            if r[f"y{h}"] is None or key not in d:
                continue
            if subset is not None and not subset(r):
                continue
            y, p = r[f"y{h}"], d[key]
            if change:
                y, p = y - r["v"], p - r["v"]
            by[r["pid"]].append((y, p))
        return {p: np.array(v, float) for p, v in by.items()}

    def score(proto, nm, h, subset=None, change=False):
        by = rows_for(proto, nm, h, subset, change)
        if not by:
            return float("nan"), float("nan"), 0
        arr = np.concatenate(list(by.values()))
        per = [r2(v[:, 1], v[:, 0]) for v in by.values() if len(v) >= 5]
        return r2(arr[:, 1], arr[:, 0]), float(np.nanmedian(per)), len(arr)

    def ci(proto, a, b, h, subset=None, change=False):
        A_ = rows_for(proto, a, h, subset, change)
        B_ = rows_for(proto, b, h, subset, change)
        both = {p: np.column_stack([A_[p][:, 0], A_[p][:, 1], B_[p][:, 1]]) for p in A_ if p in B_ and len(A_[p]) == len(B_[p])}
        return bootstrap_diff(both, a, b)

    e_abs = np.array([abs(r["e"]) for r in recs if r["e"] != 0.0])
    q75 = float(np.percentile(e_abs, 75)) if len(e_abs) else 0.0
    subsets = {"level": None}
    if has_event:
        subsets["after_event"] = lambda r: r["e"] != 0.0
        subsets["after_big_event"] = lambda r: abs(r["e"]) >= q75
    out = dict(selected=[COMBOS[g] for g in selected], lam_used={k: sorted(set(v)) for k, v in lam_used.items()},
               lam_adapt={f"{k}": v for k, v in lam_adapt.items()}, q75_abs_e=q75, tables={}, cis={})
    lines = [f"## {name} (n={len(pids)} participants, {len(recs)} records)", "",
             "Selected pooled parameters per fold: " + "; ".join(str(COMBOS[g]) for g in selected), ""]
    for proto, names in (("P", names_P), ("A", names_A)):
        for sname, sub in subsets.items():
            for change in ((False, True) if sname != "level" else (False,)):
                tag = f"{proto}/{sname}/{'change' if change else 'level'}"
                out["tables"][tag] = {}
                lines.append(f"### {tag}")
                lines.append("")
                lines.append("| predictor | " + " | ".join(f"h={h} R2 (median per-person)" for h in HORIZONS) + " |")
                lines.append("|---|" + "---:|" * len(HORIZONS))
                for nm in names:
                    row = {}
                    for h in HORIZONS:
                        row[h] = score(proto, nm, h, sub, change)
                    out["tables"][tag][nm] = {int(h): row[h][:2] for h in HORIZONS}
                    lines.append(f"| {nm} | " + " | ".join(f"{row[h][0]:.3f} ({row[h][1]:.3f})" for h in HORIZONS) + " |")
                n1 = score(proto, names[0], 1, sub, change)[2]
                lines.append(f"\nrows at h=1: {n1}\n")
                # CIs on the pairs that matter
                base_names = [n for n in names if not any(t in n for t in ("model", "aug", "channels"))]
                pairs = []
                for h in HORIZONS:
                    best_base = max(base_names, key=lambda n: score(proto, n, h, sub, change)[0])
                    if proto == "P":
                        pairs += [(h, "model_full", best_base), (h, "aug", best_base), (h, "aug", "kitchen"),
                                  (h, "aug", "aug_noframe"), (h, "aug", "aug_nochan"), (h, "aug", "aug_predonly"),
                                  (h, "model_full", "model_inert"), (h, "model_trans_g4", "model_inert"),
                                  (h, "kitchen", "ar2"), (h, "msar2", "ar2")]
                    else:
                        pairs += [(h, "A_model", best_base), (h, "A_aug", best_base), (h, "A_aug", "A_kitchen"),
                                  (h, "A_model", "A_kitchen"), (h, "A_model", "A_model_pooled"),
                                  (h, "A_model", "A_model_inert"), (h, "A_aug", "A_aug_noframe"),
                                  (h, "A_kitchen", "A_kitchen_pooled"), (h, "A_aug", "A_aug_pooled")]
                out["cis"][tag] = []
                for h, a, b in pairs:
                    d = ci(proto, a, b, h, sub, change)
                    d.update(h=h, a=a, b=b)
                    out["cis"][tag].append(d)
                    lines.append(f"- h={h}: {a} - {b}: {d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}], P(<=0)={d['p_le_0']:.3f}")
                lines.append("")
    out["nll"] = {nm: {int(h): float(np.mean(nll_rows[(nm, h)])) for h in HORIZONS} for nm in names_P}
    lines.append("### P/level: held-out Gaussian NLL per row (sigma^2 from training residuals; lower is better)")
    lines.append("")
    lines.append("| predictor | " + " | ".join(f"h={h}" for h in HORIZONS) + " |")
    lines.append("|---|" + "---:|" * len(HORIZONS))
    for nm in names_P:
        lines.append(f"| {nm} | " + " | ".join(f"{out['nll'][nm][h]:.4f}" for h in HORIZONS) + " |")
    lines.append("")
    log(f"[{name}] scoring done ({time.time() - t0:.0f}s)")
    return out, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    logf = open(ROOT / "reviews" / "esm_eval_v3.log", "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    lines = ["# ESM prediction, round 3 (2026-09-28)", "",
             "Script: `esm_eval_v3.py`. Same participant folds as round 2 (seed 0 shuffle, 5 folds). "
             "Horizons 1 to 6. Ridge penalties chosen by inner participant-grouped CV on the training "
             "participants; adaptation penalty chosen on 30 training participants by simulating the "
             "adaptation protocol. R2 is pooled over held-out rows; the value in parentheses is the median "
             "per-participant R2. CIs are participant bootstraps (1000 resamples) on the pooled R2 difference.", ""]
    out = {"grid": COMBOS, "horizons": HORIZONS}
    g = er.load_participants()
    add_time_features(g, True)
    if args.quick:
        g = {p: g[p] for p in sorted(g)[:30]}
    o, l = evaluate("Geschwind", g, True, args.workers, log)
    out["Geschwind"] = o
    lines += l
    osf = load_osf()
    add_time_features(osf, False)
    if args.quick:
        osf = {p: osf[p] for p in sorted(osf)[:30]}
    o, l = evaluate("osf_83cfk", osf, False, args.workers, log)
    out["osf_83cfk"] = o
    lines += l
    (ROOT / "reviews" / "esm_eval_v3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "reviews" / "esm_eval_v3.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    log("written reviews/esm_eval_v3.md")


if __name__ == "__main__":
    main()
