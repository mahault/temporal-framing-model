"""Local level plus AR(1) filter with the same volatility state as the round-5 model (2026-10-01).

The fair competitor for the round-5 density forecasts: an exponentially weighted ratio z_t of squared
innovations to their predicted variance scales the AR and observation noise by z_t ** gamma_v. Level
carried across segments, maximum likelihood on one-step innovations and, separately, on h = 1..6.
Adds "kfvol_all_h1" and "kfvol_all_h6" to out/baselines_<sample>.pkl.
Run:  python kfvol.py --workers 10
"""
from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor

import torch

from baselines import KF
from common import HORIZONS, SAMPLES, folds_of, load, load_sample, save, ROOT

import sys
sys.path.insert(0, str(ROOT))
from model_v2 import make_tensors, subset  # noqa: E402


class KFVol(KF):
    def __init__(self, carry):
        super().__init__(carry)
        self.vol_logit = torch.nn.Parameter(torch.tensor(-2.0))
        self.a_gv = torch.nn.Parameter(torch.tensor(-1.0))

    def forward(self, d):
        y, tod, first, mask, reset = d["y"], d["tod"], d["first"], d["mask"], d["reset"]
        N, T = y.shape
        pl, ql, qa0, r0, qseg = (torch.exp(v) for v in (self.a_pl, self.a_ql, self.a_qa, self.a_r, self.a_qseg))
        phi = 0.99 * torch.tanh(self.a_phi)
        lam_v = 0.5 * torch.sigmoid(self.vol_logit)
        gv = torch.nn.functional.softplus(self.a_gv)
        c = self.c
        ro = lambda td, fr: c[0] * td + c[1] * td * td + c[2] * fr
        l = torch.zeros(N)
        a = torch.zeros(N)
        Pll, Paa, Pla = torch.zeros(N), torch.zeros(N), torch.zeros(N)
        z = torch.ones(N)
        yh_all, S_all = [], []
        started = torch.zeros(N, dtype=torch.bool)
        for t in range(T):
            rs = reset[:, t] > 0.5
            fresh = rs & ~started
            seg = rs & started
            mult = z ** gv
            qa, r = qa0 * mult, r0 * mult
            va = qa / (1 - phi ** 2)
            a_p, Paa_p, Pla_p, Pll_p, l_p = phi * a, phi ** 2 * Paa + qa, phi * Pla, Pll + ql, l
            l_p = torch.where(fresh, self.mu.expand(N), l_p)
            Pll_p = torch.where(fresh, pl.expand(N), Pll_p)
            if self.carry:
                Pll_p = torch.where(seg, Pll + qseg, Pll_p)
                l_p = torch.where(seg, l, l_p)
            else:
                l_p = torch.where(seg, self.mu.expand(N), l_p)
                Pll_p = torch.where(seg, pl.expand(N), Pll_p)
            a_p = torch.where(rs, torch.zeros(N), a_p)
            Paa_p = torch.where(rs, va, Paa_p)
            Pla_p = torch.where(rs, torch.zeros(N), Pla_p)
            z = torch.where(fresh, torch.ones_like(z), z)
            started = started | rs
            yhat = l_p + a_p + ro(tod[:, t], first[:, t])
            S = Pll_p + Paa_p + 2 * Pla_p + r
            e = y[:, t] - yhat
            kl = (Pll_p + Pla_p) / S
            ka = (Paa_p + Pla_p) / S
            ok = mask[:, t] > 0.5
            l = torch.where(ok, l_p + kl * e, l_p)
            a = torch.where(ok, a_p + ka * e, a_p)
            Pll = torch.where(ok, Pll_p - kl * (Pll_p + Pla_p), Pll_p)
            Paa = torch.where(ok, Paa_p - ka * (Paa_p + Pla_p), Paa_p)
            Pla = torch.where(ok, Pla_p - kl * (Pla_p + Paa_p), Pla_p)
            z = torch.where(ok, (1 - lam_v) * z + lam_v * torch.clamp(e ** 2 / S, max=20.0), z)
            mult = z ** gv
            qa, r = qa0 * mult, r0 * mult
            va = qa / (1 - phi ** 2)
            yh, Sh = [], []
            for h in HORIZONS:
                ph = phi ** h
                yh.append(l + ph * a + ro(d["todt"][h][:, t], d["firstt"][h][:, t]))
                Sh.append(Pll + h * ql + ph ** 2 * Paa + 2 * ph * Pla + va * (1 - ph ** 2) + r)
            yh_all.append(torch.stack(yh, 1))
            S_all.append(torch.stack(Sh, 1))
        return torch.stack(yh_all, 1), torch.stack(S_all, 1)


def _job(a):
    name, k, hs, parts, pids, fold = a
    torch.set_num_threads(2)
    torch.manual_seed(k)
    t0 = time.time()
    dt = make_tensors(parts, pids, False)
    tr = [i for i, p in enumerate(pids) if fold[p] != k]
    te = [i for i, p in enumerate(pids) if fold[p] == k]
    dtr, dte = subset(dt, tr), subset(dt, te)
    m = KFVol(True)
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
    return dict(k=k, tag=f"kfvol_all_h{max(hs)}", pred=pred, var=var, params=params, secs=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    for name in SAMPLES:
        parts, _ = load_sample(name)
        pids = sorted(parts)
        fold = folds_of(pids)
        jobs = [(name, k, hs, parts, pids, fold) for hs in ((1,), tuple(HORIZONS)) for k in range(5)]
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
        print(f"[{name}] kfvol saved", flush=True)


if __name__ == "__main__":
    main()
