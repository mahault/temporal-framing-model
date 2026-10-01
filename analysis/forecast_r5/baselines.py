"""Round-5 forecasting baselines (2026-10-01).

Competitors the cold review asked for, on the same records, folds and horizons as v2.1:

  ridge_tt      round-3 kitchen ridge plus the target beep's time of day (the baseline v2.1 was compared with)
  pmean_all     running person mean of every earlier beep (both Geschwind periods), shrunk toward the
                training grand mean with a pseudo-count chosen on the training participants
  pmean_seg     the same restricted to the current sampling segment
  ridge_pm_all  ridge_tt plus the shrunk running person mean, running person SD and log beep count
  ridge_pm_seg  the same with segment-restricted statistics
  kf_all_h1     local level plus AR(1) Kalman filter (level = latent person set-point with a slow random
                walk, carried across Geschwind periods), group parameters fitted by maximum likelihood
                (one-step NLL) on the training participants
  kf_all_h6     the same fitted on the h = 1..6 forecast NLL
  kf_seg_h1/h6  level reset at each segment start
  ebar1         per-person AR(1) with empirical-Bayes shrinkage toward the pooled AR(1), updated online
                from all earlier beeps; prior strength chosen on the training participants

Writes out/baselines_<sample>.pkl = {"oof": {name: {h: {(pid, i): yhat}}}, "S": {name: {h: {...}}}}.
Run:  python baselines.py --workers 8
"""
from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from common import HORIZONS, SAMPLES, folds_of, load_sample, records, save, ROOT
from esm_eval_v3 import Ridge, feats, pick_lambda, r2

import sys
sys.path.insert(0, str(ROOT))
from model_v2 import make_tensors, subset  # noqa: E402

H = len(HORIZONS)


# ── ridge baselines ─────────────────────────────────────────────────────
def shrunk_mean(r, which, k, gm):
    n = r[f"pn_{which}"]
    return (n * r[f"pm_{which}"] + k * gm) / (n + k)


def ridge_family(recs, fold):
    oof = {nm: {h: {} for h in HORIZONS} for nm in ("ridge_tt", "pmean_all", "pmean_seg", "ridge_pm_all", "ridge_pm_seg")}
    info = {}
    for k in range(5):
        tr = [r for r in recs if fold[r["pid"]] != k]
        te = [r for r in recs if fold[r["pid"]] == k]
        gm = float(np.mean([r["v"] for r in tr]))
        # pseudo-count for the shrunk person mean, chosen on training rows (pooled over horizons)
        kk = {}
        for which in ("all", "seg"):
            best, bk = -np.inf, 1.0
            for cand in (0.0, 1, 2, 4, 8, 16, 32):
                sc = []
                for h in (1, 3, 6):
                    rows = [r for r in tr if r[f"y{h}"] is not None]
                    sc.append(r2(np.array([shrunk_mean(r, which, cand, gm) for r in rows]), np.array([r[f"y{h}"] for r in rows])))
                if np.mean(sc) > best:
                    best, bk = float(np.mean(sc)), cand
            kk[which] = bk
        info[k] = dict(gm=gm, k_all=kk["all"], k_seg=kk["seg"])
        for h in HORIZONS:
            trh = [r for r in tr if r[f"y{h}"] is not None]
            teh = [r for r in te if r[f"y{h}"] is not None]
            ytr = np.array([r[f"y{h}"] for r in trh])
            gtr = [r["pid"] for r in trh]
            base = lambda r: feats(r, "kitchen") + [r[f"tt{h}"], r[f"tt{h}"] ** 2, r[f"ft{h}"]]
            fams = {
                "ridge_tt": base,
                "ridge_pm_all": lambda r: base(r) + [shrunk_mean(r, "all", kk["all"], gm), r["ps_all"], math.log(r["pn_all"])],
                "ridge_pm_seg": lambda r: base(r) + [shrunk_mean(r, "seg", kk["seg"], gm), r["ps_seg"], math.log(r["pn_seg"])],
            }
            for nm, ff in fams.items():
                Xtr = np.array([ff(r) for r in trh])
                Xte = np.array([ff(r) for r in teh])
                lam = pick_lambda(Xtr, ytr, gtr)
                p = Ridge(max(lam, 1e-6)).fit(Xtr, ytr).predict(Xte)
                for r, pv in zip(teh, p):
                    oof[nm][h][(r["pid"], r["i"])] = float(pv)
            for r in teh:
                oof["pmean_all"][h][(r["pid"], r["i"])] = shrunk_mean(r, "all", kk["all"], gm)
                oof["pmean_seg"][h][(r["pid"], r["i"])] = shrunk_mean(r, "seg", kk["seg"], gm)
    return oof, info


# ── empirical-Bayes online AR(1) ────────────────────────────────────────
def pairs_of(seq):
    """(x_prev, y) within-segment consecutive pairs, indexed by the later beep."""
    out = []
    for j in range(1, len(seq)):
        if seq[j].get("p") == seq[j - 1].get("p"):
            out.append((j, seq[j - 1]["v"], seq[j]["v"]))
    return out


def ebar1_predict(parts, pids, beta0, M0, kappa):
    """Online posterior-mean AR(1) per participant; h-step iterated forecasts from beep i."""
    pred = {}
    for p in pids:
        seq = parts[p]
        prs = pairs_of(seq)
        XtX = np.zeros((2, 2))
        Xty = np.zeros(2)
        q = 0
        for i in range(len(seq)):
            while q < len(prs) and prs[q][0] <= i:
                x = np.array([1.0, prs[q][1]])
                XtX += np.outer(x, x)
                Xty += x * prs[q][2]
                q += 1
            A = XtX + kappa * M0
            c, phi = np.linalg.solve(A, Xty + kappa * M0 @ beta0)
            y = seq[i]["v"]
            fc = []
            for _ in HORIZONS:
                y = c + phi * y
                fc.append(y)
            pred[(p, i)] = fc
    return pred


def ebar1(parts, recs, fold):
    oof = {h: {} for h in HORIZONS}
    info = {}
    for k in range(5):
        trp = sorted({r["pid"] for r in recs if fold[r["pid"]] != k})
        tep = sorted({r["pid"] for r in recs if fold[r["pid"]] == k})

        def pooled(pp):
            X, Y = [], []
            for p in pp:
                for _, x, y in pairs_of(parts[p]):
                    X.append([1.0, x])
                    Y.append(y)
            X, Y = np.array(X), np.array(Y)
            return np.linalg.solve(X.T @ X, X.T @ Y), X.T @ X / len(Y)

        # prior strength chosen by inner split of the training participants
        rng = np.random.RandomState(1)
        sh = list(trp)
        rng.shuffle(sh)
        inner = {p: j % 4 for j, p in enumerate(sh)}
        best, bk = -np.inf, 10.0
        for kap in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
            sc = []
            for j in range(4):
                b0, M = pooled([p for p in trp if inner[p] != j])
                vp = [p for p in trp if inner[p] == j]
                pr = ebar1_predict(parts, vp, b0, M, kap)
                for h in (1, 3, 6):
                    rows = [r for r in recs if r["pid"] in set(vp) and r[f"y{h}"] is not None]
                    sc.append(r2(np.array([pr[(r["pid"], r["i"])][h - 1] for r in rows]), np.array([r[f"y{h}"] for r in rows])))
            if np.mean(sc) > best:
                best, bk = float(np.mean(sc)), kap
        b0, M = pooled(trp)
        pr = ebar1_predict(parts, tep, b0, M, bk)
        for r in recs:
            if r["pid"] in set(tep):
                for h in HORIZONS:
                    if r[f"y{h}"] is not None:
                        oof[h][(r["pid"], r["i"])] = float(pr[(r["pid"], r["i"])][h - 1])
        info[k] = dict(kappa=bk, c=float(b0[0]), phi=float(b0[1]))
    return oof, info


# ── local level + AR(1) Kalman filter ───────────────────────────────────
class KF(torch.nn.Module):
    """y_t = l_t + a_t + c1 tod + c2 tod^2 + c3 first + nu;  a_{t+1} = phi a_t + eta;  l_{t+1} = l_t + xi.
    l_1 ~ N(mu, pl). carry=True keeps l across segment starts (adds qseg variance); False resets it."""

    def __init__(self, carry):
        super().__init__()
        self.carry = carry
        self.mu = torch.nn.Parameter(torch.tensor(0.6))
        self.a_pl = torch.nn.Parameter(torch.tensor(math.log(0.01)))
        self.a_ql = torch.nn.Parameter(torch.tensor(math.log(1e-4)))
        self.a_phi = torch.nn.Parameter(torch.tensor(0.5))
        self.a_qa = torch.nn.Parameter(torch.tensor(math.log(0.01)))
        self.a_r = torch.nn.Parameter(torch.tensor(math.log(0.005)))
        self.a_qseg = torch.nn.Parameter(torch.tensor(math.log(1e-3)))
        self.c = torch.nn.Parameter(torch.zeros(3))

    def forward(self, d):
        y, tod, first, mask, reset = d["y"], d["tod"], d["first"], d["mask"], d["reset"]
        N, T = y.shape
        pl, ql, qa, r, qseg = (torch.exp(v) for v in (self.a_pl, self.a_ql, self.a_qa, self.a_r, self.a_qseg))
        phi = 0.99 * torch.tanh(self.a_phi)
        va = qa / (1 - phi ** 2)
        c = self.c
        ro = lambda td, fr: c[0] * td + c[1] * td * td + c[2] * fr
        l = torch.full((N,), float("nan"))
        a = torch.zeros(N)
        Pll, Paa, Pla = torch.zeros(N), torch.zeros(N), torch.zeros(N)
        yh_all, S_all = [], []
        started = torch.zeros(N, dtype=torch.bool)
        for t in range(T):
            rs = reset[:, t] > 0.5
            fresh = rs & ~started
            seg = rs & started
            # predict one step (not at a reset)
            a_p, Paa_p, Pla_p, Pll_p = phi * a, phi ** 2 * Paa + qa, phi * Pla, Pll + ql
            l_p = l
            # first beep of a participant
            l_p = torch.where(fresh, self.mu.expand(N), l_p)
            Pll_p = torch.where(fresh, pl.expand(N), Pll_p)
            # later segment start: AR part reset, level carried (or reset)
            if self.carry:
                Pll_p = torch.where(seg, Pll + qseg, Pll_p)
                l_p = torch.where(seg, l, l_p)
            else:
                l_p = torch.where(seg, self.mu.expand(N), l_p)
                Pll_p = torch.where(seg, pl.expand(N), Pll_p)
            a_p = torch.where(rs, torch.zeros(N), a_p)
            Paa_p = torch.where(rs, va.expand(N), Paa_p)
            Pla_p = torch.where(rs, torch.zeros(N), Pla_p)
            started = started | rs
            # observe
            yhat = l_p + a_p + ro(tod[:, t], first[:, t])
            S = Pll_p + Paa_p + 2 * Pla_p + r
            e = y[:, t] - yhat
            kl = (Pll_p + Pla_p) / S
            ka = (Paa_p + Pla_p) / S
            ok = mask[:, t] > 0.5
            l_u, a_u = l_p + kl * e, a_p + ka * e
            Pll_u = Pll_p - kl * (Pll_p + Pla_p)
            Paa_u = Paa_p - ka * (Paa_p + Pla_p)
            Pla_u = Pla_p - kl * (Pla_p + Paa_p)
            l = torch.where(ok, l_u, l_p)
            a = torch.where(ok, a_u, a_p)
            Pll = torch.where(ok, Pll_u, Pll_p)
            Paa = torch.where(ok, Paa_u, Paa_p)
            Pla = torch.where(ok, Pla_u, Pla_p)
            yh, Sh = [], []
            for h in HORIZONS:
                ph = phi ** h
                yh.append(l + ph * a + ro(d["todt"][h][:, t], d["firstt"][h][:, t]))
                Sh.append(Pll + h * ql + ph ** 2 * Paa + 2 * ph * Pla + va * (1 - ph ** 2) + r)
            yh_all.append(torch.stack(yh, 1))
            S_all.append(torch.stack(Sh, 1))
        return torch.stack(yh_all, 1), torch.stack(S_all, 1)   # N, T, H

    def nll(self, d, yh, S, hs):
        tot, n = 0.0, 0.0
        for h in hs:
            m = d["mt"][h]
            l = 0.5 * torch.log(2 * math.pi * S[:, :, h - 1]) + (d["yt"][h] - yh[:, :, h - 1]) ** 2 / (2 * S[:, :, h - 1])
            tot = tot + (l * m).sum()
            n += float(m.sum())
        return tot / n


def _kf_job(a):
    name, k, carry, hs, parts, pids, fold = a
    torch.set_num_threads(2)
    torch.manual_seed(k)
    t0 = time.time()
    dt = make_tensors(parts, pids, False)
    tr = [i for i, p in enumerate(pids) if fold[p] != k]
    te = [i for i, p in enumerate(pids) if fold[p] == k]
    dtr, dte = subset(dt, tr), subset(dt, te)
    m = KF(carry)
    opt = torch.optim.Adam(m.parameters(), lr=0.03)
    best, state = float("inf"), None
    for it in range(250):
        opt.zero_grad()
        yh, S = m(dtr)
        loss = m.nll(dtr, yh, S, hs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
        if float(loss) < best:
            best, state = float(loss), {kk: v.detach().clone() for kk, v in m.state_dict().items()}
    m.load_state_dict(state)
    with torch.no_grad():
        yh, S = m(dte)
    pred, var = {h: {} for h in HORIZONS}, {h: {} for h in HORIZONS}
    for n, i in enumerate(te):
        p = pids[i]
        T = len(parts[p])
        for t in range(T):
            for h in HORIZONS:
                if dte["mt"][h][n, t] > 0.5:
                    pred[h][(p, t)] = float(yh[n, t, h - 1])
                    var[h][(p, t)] = float(S[n, t, h - 1])
    tag = f"kf_{'all' if carry else 'seg'}_h{max(hs)}"
    params = {kk: (float(v) if v.dim() == 0 else [float(x) for x in v]) for kk, v in m.state_dict().items()}
    return dict(name=name, k=k, tag=tag, pred=pred, var=var, loss=best, secs=time.time() - t0, params=params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--samples", default=",".join(SAMPLES))
    args = ap.parse_args()
    for name in args.samples.split(","):
        t0 = time.time()
        parts, has_event = load_sample(name)
        pids = sorted(parts)
        fold = folds_of(pids)
        recs = records(parts, has_event)
        print(f"[{name}] {len(pids)} participants, {len(recs)} records", flush=True)
        oof, info = ridge_family(recs, fold)
        print(f"[{name}] ridge family done ({time.time() - t0:.0f}s) {info}", flush=True)
        oof["ebar1"], info_eb = ebar1(parts, recs, fold)
        print(f"[{name}] ebar1 done ({time.time() - t0:.0f}s) {info_eb}", flush=True)
        jobs = [(name, k, carry, hs, parts, pids, fold) for carry in (True, False) for hs in ((1,), tuple(HORIZONS)) for k in range(5)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(_kf_job, jobs))
        S = {}
        kf_params = {}
        for rr in res:
            oof.setdefault(rr["tag"], {h: {} for h in HORIZONS})
            S.setdefault(rr["tag"], {h: {} for h in HORIZONS})
            for h in HORIZONS:
                oof[rr["tag"]][h].update(rr["pred"][h])
                S[rr["tag"]][h].update(rr["var"][h])
            kf_params.setdefault(rr["tag"], {})[rr["k"]] = rr["params"]
        print(f"[{name}] Kalman fits done ({time.time() - t0:.0f}s)", flush=True)
        save(dict(oof=oof, S=S, info=dict(ridge=info, ebar1=info_eb, kf=kf_params)), f"baselines_{name}.pkl")
        print(f"[{name}] saved", flush=True)


if __name__ == "__main__":
    main()
