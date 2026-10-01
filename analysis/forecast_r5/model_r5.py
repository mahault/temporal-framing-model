"""Round-5 extensions of model v2.1 (model_v2.py is imported, not modified).

Options added to ModelV2:
  carry  the slow mood m and the participant set-point b persist across sampling segments
         (the second Geschwind period starts from the person's state at the end of the first, with
         added variance qseg); v2.1 reset both to the group mean, discarding what the filter had learned
         about the person
  vol    a volatility (precision) state: an exponentially weighted ratio z_t of squared innovations to
         their predicted variance scales the process and observation noise by z_t ** gamma_v, so the
         predictive variance adapts to each person
  setpoint  the participant set-point b becomes a proper level-3 state: its prior variance (between-person
         spread) starts at a realistic value and is fitted, it drifts slowly with fitted variance qb, and the
         slow mood reverts to it at a fitted rate; in v2.1 the fitted prior variance of b collapsed to 0.001,
         so the model kept no long-run person level
  latent_mood  slow mood m is an inferred latent state (fitted prior variance, fitted drift, reversion to
         the set-point) updated through the filter, instead of a running average of fast valence
         (1/theta = 0); this nests the local-level plus AR(1) filter as a special case
  chan_shrink  an L2 penalty on the three channel weights (prior centred on zero), added in fit()
The forecasting objective horizons are set in the fit call (v2.1 used h = 1..3).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model_v2 import P0, PAST, EBAR_RATE, ModelV2, _sp  # noqa: E402


class ModelR5(ModelV2):
    def __init__(self, n_part, has_event, carry=False, vol=False, setpoint=False, latent_mood=False,
                 null_init=False, **kw):
        super().__init__(n_part, has_event, **kw)
        self.latent_mood = latent_mood
        self.a_pm = torch.nn.Parameter(torch.tensor(math.log(0.01)))
        if latent_mood:
            with torch.no_grad():
                self.km_logit.fill_(1.15)   # fast valence reverts to mood at about 0.38 per beep
        self.carry = carry
        self.vol = vol
        self.setpoint = setpoint
        self.a_qb = torch.nn.Parameter(torch.tensor(math.log(1e-4)))
        if setpoint:
            with torch.no_grad():
                self.a_pb.fill_(-2.48)      # softplus(-2.48) * 0.1 = 0.008, the between-person variance scale
                self.ab_logit.fill_(0.0)    # reversion of slow mood to the set-point, 0.1 per beep
        if null_init:
            # start at generic values of the local-level plus AR(1) filter (fast reversion 0.4 per beep,
            # level variance 0.01, slow drift 1e-4, observation and process noise 0.01), channels at zero,
            # so that fitting starts from the null competitor and can only add to it
            with torch.no_grad():
                self.km_logit.fill_(1.386)
                self.a_qx.fill_(0.31)
                self.log_r.fill_(math.log(0.01))
                self.a_qm.fill_(-2.06)
                self.a_pm.fill_(math.log(0.01))
                self.ab_logit.fill_(-3.0)
                for b_ in (self.beta_B, self.beta_P, self.beta_F):
                    b_.fill_(0.0)
        self.a_qseg = torch.nn.Parameter(torch.tensor(math.log(1e-3)))
        self.vol_logit = torch.nn.Parameter(torch.tensor(-2.0))     # EWMA rate of the volatility state
        self.a_gv = torch.nn.Parameter(torch.tensor(-1.0))          # exponent gamma_v (softplus)

    def forward(self, d, idx, H=None):
        y, e, tod, first, mask, reset = d["y"], d["e"], d["tod"], d["first"], d["mask"], d["reset"]
        N, T = y.shape
        H = H or self.H
        mu = self._pp("mu", idx)
        theta = torch.exp(self._pp("log_theta", idx)) + 2.0
        km = 0.5 * torch.sigmoid(self._pp("km_logit", idx))
        r0 = torch.exp(self._pp("log_r", idx)) + 1e-5
        beta = torch.stack([self._pp("beta_B", idx), self._pp("beta_P", idx), self._pp("beta_F", idx)], 1)
        if self.no_channels:
            beta = beta * 0.0
        if self.no_level2:
            km = km * 0.0
            inv_theta = torch.zeros_like(theta)
        else:
            inv_theta = 1.0 / theta
        if self.latent_mood:
            inv_theta = torch.zeros_like(theta)
        qx0 = _sp(self.a_qx) * 1e-2 + 1e-6
        qu = _sp(self.a_qu)
        qm = _sp(self.a_qm) * 1e-3 + 1e-7
        tau = _sp(self.a_tau) + 1e-3
        kappa = _sp(self.a_kappa)
        s = torch.sigmoid(self.s_logit)
        du = torch.sigmoid(self.du_logit)
        c = self.c * 0.0 if self.no_tod else self.c
        qseg = torch.exp(self.a_qseg)
        lam_v = 0.5 * torch.sigmoid(self.vol_logit)
        gv = _sp(self.a_gv) if self.vol else torch.zeros(())

        def readout(x, td, fr):
            return x + c[0] * td + c[1] * td * td + c[2] * fr

        pb = torch.zeros(()) if self.no_baseline else _sp(self.a_pb) * 0.1
        ab = torch.zeros(()) if self.no_baseline else 0.2 * torch.sigmoid(self.ab_logit)
        if self.no_level2:
            ab = ab * 0.0
        pm0 = torch.exp(self.a_pm) if self.latent_mood else torch.tensor(P0[1])
        P0_ = torch.diag(torch.stack([torch.tensor(P0[0]), pm0, pb]))

        def Amat(kmg):
            A = torch.zeros(N, 3, 3)
            A[:, 0, 0] = 1 - kmg
            A[:, 0, 1] = kmg
            A[:, 1, 0] = inv_theta
            A[:, 1, 1] = 1 - inv_theta - ab
            A[:, 1, 2] = ab
            A[:, 2, 2] = 1.0
            return A

        qb = torch.exp(self.a_qb) if self.setpoint else torch.tensor(1e-7)

        def Qmat(uu, mult):
            Q = torch.zeros(N, 3, 3)
            Q[:, 0, 0] = (qx0 + qu * uu ** 2) * mult
            Q[:, 1, 1] = qm
            Q[:, 2, 2] = qb
            return Q

        x, m, b = mu.clone(), mu.clone(), mu.clone()
        P = P0_.expand(N, 3, 3).clone()
        eps_prev = torch.zeros(N)
        ebar = torch.zeros(N)
        q = torch.full((N, 3), 1.0 / 3.0)
        u_prev = torch.zeros(N)
        w_prev = torch.ones(N, 3)
        z = torch.ones(N)
        started = torch.zeros(N, dtype=torch.bool)
        out = {k: [] for k in ("yhat", "S", "eps", "S1", "vB", "vP", "vF", "q", "x", "m", "b", "u", "w", "rho")}
        for t in range(T):
            rs = reset[:, t] > 0.5
            fresh = rs & ~started if self.carry else rs
            seg = rs & started if self.carry else torch.zeros_like(rs)
            x = torch.where(fresh, mu, torch.where(seg, m, x))
            m = torch.where(fresh, mu, m)
            b = torch.where(fresh, mu, b)
            Pseg = P.clone()
            Pseg[:, 0, :] = 0.0
            Pseg[:, :, 0] = 0.0
            Pseg[:, 0, 0] = P0[0]
            Pseg[:, 1, 1] = P[:, 1, 1] + qseg
            Pseg[:, 2, 2] = P[:, 2, 2] + qseg
            P = torch.where(fresh[:, None, None], P0_.expand(N, 3, 3), torch.where(seg[:, None, None], Pseg, P))
            z = torch.where(fresh, torch.ones_like(z), z)
            eps_prev = torch.where(rs, torch.zeros_like(eps_prev), eps_prev)
            ebar = torch.where(fresh, torch.zeros_like(ebar), ebar)
            q = torch.where(rs[:, None], torch.full_like(q, 1.0 / 3.0), q)
            u_prev = torch.where(rs, torch.zeros_like(u_prev), u_prev)
            w_prev = torch.where(rs[:, None], torch.ones_like(w_prev), w_prev)
            started = started | rs
            mult = z ** gv
            r = r0 * mult
            kmg = w_prev[:, PAST] * km
            A = Amat(kmg)
            mean = torch.stack([x, m, b], 1)
            z0 = torch.zeros_like(u_prev)
            mean_p = torch.einsum("nij,nj->ni", A, mean) + torch.stack([u_prev, z0, z0], 1)
            P_p = torch.einsum("nij,njk,nlk->nil", A, P, A) + Qmat(u_prev, mult)
            mean_p = torch.where(rs[:, None], mean, mean_p)
            P_p = torch.where(rs[:, None, None], P, P_p)
            yhat1 = readout(mean_p[:, 0], tod[:, t], first[:, t])
            S1 = P_p[:, 0, 0] + r
            eps = y[:, t] - yhat1
            K = P_p[:, :, 0] / S1[:, None]
            mean_u = mean_p + K * eps[:, None]
            P_u = P_p - K[:, :, None] * P_p[:, 0, :][:, None, :]
            ok = mask[:, t] > 0.5
            mean_new = torch.where(ok[:, None], mean_u, mean_p)
            P = torch.where(ok[:, None, None], P_u, P_p)
            x, m, b = mean_new[:, 0], mean_new[:, 1], mean_new[:, 2]
            eps = torch.where(ok, eps, torch.zeros_like(eps))
            if self.vol:
                z = torch.where(ok, (1 - lam_v) * z + lam_v * torch.clamp(eps ** 2 / S1, max=20.0), z)
            vB = torch.tanh(-(eps ** 2 - eps_prev ** 2) / tau[0])
            if self.has_event:
                vP = torch.tanh((e[:, t] - ebar) / tau[1])
                ebar = torch.where(ok, (1 - EBAR_RATE) * ebar + EBAR_RATE * e[:, t], ebar)
            else:
                vP = torch.tanh(eps / tau[1])
            rho = torch.sigmoid(self.rho_pos + self.gamma_m * (m - b))
            vF = torch.tanh((km * (m - x) + rho * self.b_opt) / tau[2])
            sal = torch.stack([vB.abs(), vP.abs(), vF.abs()], 1)
            qn = (s * q + (1 - s) / 3.0) * torch.exp(kappa * sal)
            qn = qn / qn.sum(1, keepdim=True)
            q = torch.where(ok[:, None], qn, q)
            if self.clamp is not None:
                qe = torch.zeros_like(q)
                qe[:, self.clamp] = 1.0
            else:
                qe = q
            w = torch.clamp(1.0 + self.g * (3.0 * qe - 1.0), min=0.0)
            u = (w * beta * torch.stack([vB, vP, vF], 1)).sum(1)
            u = torch.where(ok, u, torch.zeros_like(u))
            eps_prev = torch.where(ok, eps, eps_prev)
            u_prev, w_prev = u, w
            mult_f = z ** gv
            fm, fP, fu = torch.stack([x, m, b], 1), P, u
            yh, Sh = [], []
            Ah = Amat(w[:, PAST] * km)
            for h in range(1, H + 1):
                zz = torch.zeros_like(fu)
                fm = torch.einsum("nij,nj->ni", Ah, fm) + torch.stack([fu, zz, zz], 1)
                fP = torch.einsum("nij,njk,nlk->nil", Ah, fP, Ah) + Qmat(fu, mult_f)
                yh.append(readout(fm[:, 0], d["todt"][h][:, t], d["firstt"][h][:, t]))
                Sh.append(fP[:, 0, 0] + r0 * mult_f)
                fu = fu * du
            for k_, v_ in (("yhat", torch.stack(yh, 1)), ("S", torch.stack(Sh, 1)), ("eps", eps), ("S1", S1),
                           ("vB", vB), ("vP", vP), ("vF", vF), ("q", q), ("x", x), ("m", m), ("b", b),
                           ("u", u), ("w", w), ("rho", rho)):
                out[k_].append(v_)
        return {k: torch.stack(v, 1) for k, v in out.items()}


def fit_r5(model, d, idx, iters=300, lr=0.03, horizons=(1, 2, 3), chan_l2=0.0, seed=0):
    """Group fit (participant deviations frozen at zero), full-batch Adam, best-iterate kept.
    chan_l2 > 0 adds a Gaussian prior centred on zero to the three channel weights."""
    torch.manual_seed(seed)
    for n_, p_ in model.named_parameters():
        p_.requires_grad_(n_ != "delta")
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    idx_t = torch.as_tensor(idx)
    best, best_state = float("inf"), None
    for _ in range(iters):
        opt.zero_grad()
        res = model(d, idx_t, H=max(horizons))
        loss = model.nll(d, idx_t, res, horizons)
        if chan_l2 > 0:
            loss = loss + chan_l2 * (model.beta_B ** 2 + model.beta_P ** 2 + model.beta_F ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if float(loss.detach()) < best:
            best = float(loss.detach())
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best
