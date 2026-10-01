"""The paper's three channels added to the strongest competitor (2026-10-01).

Core: local level plus AR(1) filter with the volatility state (kfvol). Added: the three valence channels
of the paper drive the fast deviation, as in the model,

    a_{t+1} = phi a_t + beta_B vB_t + beta_P vP_t + beta_F vF_t + eta_t
    vB_t = tanh(-(eps_t^2 - eps_{t-1}^2) / tau_B)       backward: change in surprise
    vP_t = tanh((e_t - ebar_t) / tau_P)                 present: event prediction error (Geschwind);
           tanh(eps_t / tau_P) where no events are recorded (reliability sample)
    vF_t = tanh((-(1 - phi) a_t + b_opt) / tau_F)       forward: expected next change plus optimism

with the drive decaying by a fitted factor per forecast step. kfx_* is this model; kfx0_* is the same
code with the channel weights fixed at zero (it reproduces kfvol and checks the implementation).
Adds kfx_all_h1, kfx_all_h6, kfx0_all_h1 to out/baselines_<sample>.pkl.
Run:  python kfx.py --workers 10
"""
from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor

import torch

from common import HORIZONS, SAMPLES, folds_of, load, load_sample, save, ROOT
from kfvol import KFVol

import sys
sys.path.insert(0, str(ROOT))
from model_v2 import EBAR_RATE, make_tensors, subset  # noqa: E402


class KFX(KFVol):
    def __init__(self, has_event, channels=True):
        super().__init__(True)
        self.has_event = has_event
        self.channels = channels
        self.beta = torch.nn.Parameter(torch.zeros(3))
        self.a_tau = torch.nn.Parameter(torch.tensor([math.log(math.e - 1)] * 3))
        self.b_opt = torch.nn.Parameter(torch.tensor(0.0))
        self.du_logit = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, d):
        y, e, tod, first, mask, reset = d["y"], d["e"], d["tod"], d["first"], d["mask"], d["reset"]
        N, T = y.shape
        pl, ql, qa0, r0, qseg = (torch.exp(v) for v in (self.a_pl, self.a_ql, self.a_qa, self.a_r, self.a_qseg))
        phi = 0.99 * torch.tanh(self.a_phi)
        lam_v = 0.5 * torch.sigmoid(self.vol_logit)
        gv = torch.nn.functional.softplus(self.a_gv)
        tau = torch.nn.functional.softplus(self.a_tau) + 1e-3
        beta = self.beta if self.channels else self.beta * 0.0
        du = torch.sigmoid(self.du_logit)
        c = self.c
        ro = lambda td, fr: c[0] * td + c[1] * td * td + c[2] * fr
        l, a = torch.zeros(N), torch.zeros(N)
        Pll, Paa, Pla = torch.zeros(N), torch.zeros(N), torch.zeros(N)
        z = torch.ones(N)
        eps_prev, ebar, u = torch.zeros(N), torch.zeros(N), torch.zeros(N)
        yh_all, S_all = [], []
        started = torch.zeros(N, dtype=torch.bool)
        for t in range(T):
            rs = reset[:, t] > 0.5
            fresh = rs & ~started
            seg = rs & started
            mult = z ** gv
            qa, r = qa0 * mult, r0 * mult
            va = qa / (1 - phi ** 2)
            a_p, Paa_p, Pla_p, Pll_p, l_p = phi * a + u, phi ** 2 * Paa + qa, phi * Pla, Pll + ql, l
            l_p = torch.where(fresh, self.mu.expand(N), l_p)
            Pll_p = torch.where(fresh, pl.expand(N), Pll_p)
            Pll_p = torch.where(seg, Pll + qseg, Pll_p)
            l_p = torch.where(seg, l, l_p)
            a_p = torch.where(rs, torch.zeros(N), a_p)
            Paa_p = torch.where(rs, va, Paa_p)
            Pla_p = torch.where(rs, torch.zeros(N), Pla_p)
            z = torch.where(fresh, torch.ones_like(z), z)
            eps_prev = torch.where(rs, torch.zeros(N), eps_prev)
            ebar = torch.where(fresh, torch.zeros(N), ebar)
            started = started | rs
            yhat = l_p + a_p + ro(tod[:, t], first[:, t])
            S = Pll_p + Paa_p + 2 * Pla_p + r
            eps = y[:, t] - yhat
            kl = (Pll_p + Pla_p) / S
            ka = (Paa_p + Pla_p) / S
            ok = mask[:, t] > 0.5
            l = torch.where(ok, l_p + kl * eps, l_p)
            a = torch.where(ok, a_p + ka * eps, a_p)
            Pll = torch.where(ok, Pll_p - kl * (Pll_p + Pla_p), Pll_p)
            Paa = torch.where(ok, Paa_p - ka * (Paa_p + Pla_p), Paa_p)
            Pla = torch.where(ok, Pla_p - kl * (Pla_p + Paa_p), Pla_p)
            eps = torch.where(ok, eps, torch.zeros(N))
            z = torch.where(ok, (1 - lam_v) * z + lam_v * torch.clamp(eps ** 2 / S, max=20.0), z)
            vB = torch.tanh(-(eps ** 2 - eps_prev ** 2) / tau[0])
            if self.has_event:
                vP = torch.tanh((e[:, t] - ebar) / tau[1])
                ebar = torch.where(ok, (1 - EBAR_RATE) * ebar + EBAR_RATE * e[:, t], ebar)
            else:
                vP = torch.tanh(eps / tau[1])
            vF = torch.tanh((-(1 - phi) * a + self.b_opt) / tau[2])
            u = torch.where(ok, beta[0] * vB + beta[1] * vP + beta[2] * vF, torch.zeros(N))
            eps_prev = torch.where(ok, eps, eps_prev)
            mult = z ** gv
            qa, r = qa0 * mult, r0 * mult
            va = qa / (1 - phi ** 2)
            yh, Sh = [], []
            fa, fu = a, u
            for h in HORIZONS:
                fa = phi * fa + fu
                fu = fu * du
                ph = phi ** h
                yh.append(l + fa + ro(d["todt"][h][:, t], d["firstt"][h][:, t]))
                Sh.append(Pll + h * ql + ph ** 2 * Paa + 2 * ph * Pla + va * (1 - ph ** 2) + r)
            yh_all.append(torch.stack(yh, 1))
            S_all.append(torch.stack(Sh, 1))
        return torch.stack(yh_all, 1), torch.stack(S_all, 1)


def _job(a):
    name, k, hs, channels, parts, pids, fold, has_event = a
    torch.set_num_threads(2)
    torch.manual_seed(k)
    t0 = time.time()
    dt = make_tensors(parts, pids, has_event)
    tr = [i for i, p in enumerate(pids) if fold[p] != k]
    te = [i for i, p in enumerate(pids) if fold[p] == k]
    dtr, dte = subset(dt, tr), subset(dt, te)
    m = KFX(has_event, channels)
    opt = torch.optim.Adam(m.parameters(), lr=0.03)
    best, state = float("inf"), None
    for _ in range(250):
        opt.zero_grad()
        yh, S = m(dtr)
        loss = m.nll(dtr, yh, S, hs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
        if float(loss.detach()) < best:
            best, state = float(loss.detach()), {kk: v.detach().clone() for kk, v in m.state_dict().items()}
    m.load_state_dict(state)
    with torch.no_grad():
        yh, S = m(dte)
    pred, var = {h: {} for h in HORIZONS}, {h: {} for h in HORIZONS}
    for n, i in enumerate(te):
        p = pids[i]
        for t in range(len(parts[p])):
            for h in HORIZONS:
                if dte["mt"][h][n, t] > 0.5:
                    pred[h][(p, t)] = float(yh[n, t, h - 1])
                    var[h][(p, t)] = float(S[n, t, h - 1])
    params = {kk: (float(v) if v.dim() == 0 else [float(x) for x in v]) for kk, v in m.state_dict().items()}
    tag = f"{'kfx' if channels else 'kfx0'}_all_h{max(hs)}"
    return dict(k=k, tag=tag, pred=pred, var=var, params=params, secs=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    for name in SAMPLES:
        parts, has_event = load_sample(name)
        pids = sorted(parts)
        fold = folds_of(pids)
        jobs = [(name, k, hs, ch, parts, pids, fold, has_event)
                for hs, ch in (((1,), True), (tuple(HORIZONS), True), ((1,), False)) for k in range(5)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(_job, jobs))
        b = load(f"baselines_{name}.pkl")
        for rr in res:
            b["oof"].setdefault(rr["tag"], {h: {} for h in HORIZONS})
            b["S"].setdefault(rr["tag"], {h: {} for h in HORIZONS})
            for h in HORIZONS:
                b["oof"][rr["tag"]][h].update(rr["pred"][h])
                b["S"][rr["tag"]][h].update(rr["var"][h])
            b["info"]["kf"].setdefault(rr["tag"], {})[rr["k"]] = rr["params"]
        save(b, f"baselines_{name}.pkl")
        print(f"[{name}] kfx saved", flush=True)


if __name__ == "__main__":
    main()
