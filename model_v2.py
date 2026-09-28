"""Model v2: hierarchical continuous state-space version of the temporal-framing model.

See reviews/model_v2_spec.md for the factorisation. Two-state Kalman filter on
(fast valence x, slow mood m) whose level-1 dynamics are driven by the three
valence channels of the paper (backward, present, forward), each with a
frame-gated precision. The frame is a sticky three-state recogniser fed by
channel salience. Frame gain g = 0 reproduces the inert model; clamp fixes the
frame one-hot. Participant-level deviations delta_i on a subset of parameters
give the hierarchical fit.

The discrete model (generative_model.py, agent.py) is untouched.
"""
from __future__ import annotations

import math

import numpy as np
import torch

PAST, PRESENT, FUTURE = 0, 1, 2
HIER = ("mu", "log_theta", "km_logit", "beta_B", "beta_P", "beta_F", "log_r")   # per-participant deviations
EBAR_RATE = 0.1
P0 = (0.05, 0.02)


def _sp(x):
    return torch.nn.functional.softplus(x)


class ModelV2(torch.nn.Module):
    def __init__(self, n_part, has_event, g=1.0, clamp=None, no_level2=False, no_tod=False,
                 no_channels=False, no_hier=False, horizons=6):
        super().__init__()
        self.has_event = bool(has_event)
        self.g = float(g)
        self.clamp = clamp
        self.no_level2 = no_level2
        self.no_tod = no_tod
        self.no_channels = no_channels
        self.no_hier = no_hier
        self.H = horizons
        # group parameters (unconstrained)
        self.mu = torch.nn.Parameter(torch.tensor(0.6))
        self.log_theta = torch.nn.Parameter(torch.tensor(math.log(20.0)))
        self.km_logit = torch.nn.Parameter(torch.tensor(-1.5))
        self.a_qx = torch.nn.Parameter(torch.tensor(0.0))
        self.a_qu = torch.nn.Parameter(torch.tensor(-2.0))
        self.a_qm = torch.nn.Parameter(torch.tensor(0.0))
        self.log_r = torch.nn.Parameter(torch.tensor(math.log(0.01)))
        self.beta_B = torch.nn.Parameter(torch.tensor(0.0))
        self.beta_P = torch.nn.Parameter(torch.tensor(0.02))
        self.beta_F = torch.nn.Parameter(torch.tensor(0.0))
        self.a_tau = torch.nn.Parameter(torch.tensor([math.log(math.e - 1)] * 3))   # softplus -> 1
        self.b_opt = torch.nn.Parameter(torch.tensor(0.0))
        self.rho_pos = torch.nn.Parameter(torch.tensor(0.0))
        self.gamma_m = torch.nn.Parameter(torch.tensor(0.0))
        self.a_kappa = torch.nn.Parameter(torch.tensor(0.5))
        self.s_logit = torch.nn.Parameter(torch.tensor(1.5))
        self.du_logit = torch.nn.Parameter(torch.tensor(0.0))
        self.c = torch.nn.Parameter(torch.zeros(3))
        # participant deviations
        self.delta = torch.nn.Parameter(torch.zeros(n_part, len(HIER)))

    # ── parameter access with per-participant deviations ──
    def _pp(self, name, idx):
        base = getattr(self, name)
        if self.no_hier or name not in HIER:
            return base.expand(len(idx)) if base.dim() == 0 else base
        return base + self.delta[idx, HIER.index(name)]

    def forward(self, d, idx, H=None):
        """d: dict of tensors (N, T) [y, e, tod, first, mask, reset] and targets
        yt[h], mt[h], todt[h], firstt[h] for h = 1..H. idx: participant indices (N,).
        Returns dict of per-step tensors."""
        y, e, tod, first, mask, reset = d["y"], d["e"], d["tod"], d["first"], d["mask"], d["reset"]
        N, T = y.shape
        mu = self._pp("mu", idx)
        theta = torch.exp(self._pp("log_theta", idx)) + 2.0
        km = 0.5 * torch.sigmoid(self._pp("km_logit", idx))
        r = torch.exp(self._pp("log_r", idx)) + 1e-5
        beta = torch.stack([self._pp("beta_B", idx), self._pp("beta_P", idx), self._pp("beta_F", idx)], 1)  # N,3
        if self.no_channels:
            beta = beta * 0.0
        if self.no_level2:
            km = km * 0.0
            inv_theta = torch.zeros_like(theta)
        else:
            inv_theta = 1.0 / theta
        qx = _sp(self.a_qx) * 1e-2 + 1e-6
        qu = _sp(self.a_qu)
        qm = _sp(self.a_qm) * 1e-3 + 1e-7
        tau = _sp(self.a_tau) + 1e-3
        kappa = _sp(self.a_kappa)
        s = torch.sigmoid(self.s_logit)
        du = torch.sigmoid(self.du_logit)
        c = self.c * 0.0 if self.no_tod else self.c

        def readout(x, td, fr):
            return x + c[0] * td + c[1] * td * td + c[2] * fr

        # state
        x = mu.clone()
        m = mu.clone()
        P = torch.zeros(N, 2, 2)
        P[:, 0, 0] = P0[0]
        P[:, 1, 1] = P0[1]
        eps_prev = torch.zeros(N)
        ebar = torch.zeros(N)
        q = torch.full((N, 3), 1.0 / 3.0)
        u_prev = torch.zeros(N)
        w_prev = torch.ones(N, 3)
        out = {k: [] for k in ("yhat", "S", "eps", "S1", "vB", "vP", "vF", "q", "x", "m", "u", "w", "rho")}
        for t in range(T):
            rs = reset[:, t] > 0.5
            # reset at segment starts
            x = torch.where(rs, mu, x)
            m = torch.where(rs, mu, m)
            P = torch.where(rs[:, None, None], torch.diag(torch.tensor(P0)).expand(N, 2, 2), P)
            eps_prev = torch.where(rs, torch.zeros_like(eps_prev), eps_prev)
            ebar = torch.where(rs, torch.zeros_like(ebar), ebar)
            q = torch.where(rs[:, None], torch.full_like(q, 1.0 / 3.0), q)
            u_prev = torch.where(rs, torch.zeros_like(u_prev), u_prev)
            w_prev = torch.where(rs[:, None], torch.ones_like(w_prev), w_prev)
            # predict (skipped at reset)
            kmg = w_prev[:, PAST] * km
            A = torch.zeros(N, 2, 2)
            A[:, 0, 0] = 1 - kmg
            A[:, 0, 1] = kmg
            A[:, 1, 0] = inv_theta
            A[:, 1, 1] = 1 - inv_theta
            mean = torch.stack([x, m], 1)
            mean_p = torch.einsum("nij,nj->ni", A, mean) + torch.stack([u_prev, torch.zeros_like(u_prev)], 1)
            Qt = torch.zeros(N, 2, 2)
            Qt[:, 0, 0] = qx + qu * u_prev ** 2
            Qt[:, 1, 1] = qm
            P_p = torch.einsum("nij,njk,nlk->nil", A, P, A) + Qt
            mean_p = torch.where(rs[:, None], mean, mean_p)
            P_p = torch.where(rs[:, None, None], P, P_p)
            # observe
            yhat1 = readout(mean_p[:, 0], tod[:, t], first[:, t])
            S1 = P_p[:, 0, 0] + r
            eps = y[:, t] - yhat1
            K = P_p[:, :, 0] / S1[:, None]
            mean_u = mean_p + K * eps[:, None]
            P_u = P_p - K[:, :, None] * P_p[:, 0, :][:, None, :]
            ok = mask[:, t] > 0.5
            mean_new = torch.where(ok[:, None], mean_u, mean_p)
            P = torch.where(ok[:, None, None], P_u, P_p)
            x, m = mean_new[:, 0], mean_new[:, 1]
            eps = torch.where(ok, eps, torch.zeros_like(eps))
            # channels
            vB = torch.tanh(-(eps ** 2 - eps_prev ** 2) / tau[0])
            if self.has_event:
                vP = torch.tanh((e[:, t] - ebar) / tau[1])
                ebar = torch.where(ok, (1 - EBAR_RATE) * ebar + EBAR_RATE * e[:, t], ebar)
            else:
                vP = torch.tanh(eps / tau[1])
            rho = torch.sigmoid(self.rho_pos + self.gamma_m * (m - mu))
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
            # h-step forecasts from the filtered state
            fm, fP, fu = torch.stack([x, m], 1), P, u
            yh, Sh = [], []
            for h in range(1, (H or self.H) + 1):
                kmg2 = w[:, PAST] * km
                Ah = torch.zeros(N, 2, 2)
                Ah[:, 0, 0] = 1 - kmg2
                Ah[:, 0, 1] = kmg2
                Ah[:, 1, 0] = inv_theta
                Ah[:, 1, 1] = 1 - inv_theta
                fm = torch.einsum("nij,nj->ni", Ah, fm) + torch.stack([fu, torch.zeros_like(fu)], 1)
                Qh = torch.zeros(N, 2, 2)
                Qh[:, 0, 0] = qx + qu * fu ** 2
                Qh[:, 1, 1] = qm
                fP = torch.einsum("nij,njk,nlk->nil", Ah, fP, Ah) + Qh
                yh.append(readout(fm[:, 0], d["todt"][h][:, t], d["firstt"][h][:, t]))
                Sh.append(fP[:, 0, 0] + r)
                fu = fu * du
            out["yhat"].append(torch.stack(yh, 1))
            out["S"].append(torch.stack(Sh, 1))
            out["eps"].append(eps)
            out["S1"].append(S1)
            out["vB"].append(vB)
            out["vP"].append(vP)
            out["vF"].append(vF)
            out["q"].append(q)
            out["x"].append(x)
            out["m"].append(m)
            out["u"].append(u)
            out["w"].append(w)
            out["rho"].append(rho)
        res = {k: torch.stack(v, 1) for k, v in out.items()}   # (N, T, ...)
        return res

    def nll(self, d, idx, res, horizons=(1, 2, 3)):
        tot, n = 0.0, 0
        for h in horizons:
            yt, mt = d["yt"][h], d["mt"][h]
            yh = res["yhat"][:, :, h - 1]
            S = res["S"][:, :, h - 1]
            l = 0.5 * torch.log(2 * math.pi * S) + (yt - yh) ** 2 / (2 * S)
            tot = tot + (l * mt).sum()
            n += int(mt.sum())
        return tot / max(n, 1)


def make_tensors(parts, pids, has_event, H=6):
    """Pad participant sequences into (N, T) tensors with targets at h = 1..H."""
    T = max(len(parts[p]) for p in pids)
    N = len(pids)
    z = lambda: np.zeros((N, T), np.float32)
    y, e, tod, first, mask, reset = z(), z(), z(), z(), z(), z()
    yt = {h: z() for h in range(1, H + 1)}
    mt = {h: z() for h in range(1, H + 1)}
    todt = {h: z() for h in range(1, H + 1)}
    firstt = {h: z() for h in range(1, H + 1)}
    for n, p in enumerate(pids):
        seq = parts[p]
        for i, b in enumerate(seq):
            y[n, i] = b["v"]
            e[n, i] = b["e"] if (has_event and b.get("e") is not None) else 0.0
            tod[n, i] = b.get("tod", 0.5)
            first[n, i] = b.get("first", 0.0)
            mask[n, i] = 1.0
            reset[n, i] = 1.0 if (i == 0 or b.get("p") != seq[i - 1].get("p")) else 0.0
            for h in range(1, H + 1):
                j = i + h
                if j < len(seq) and seq[j].get("p") == b.get("p"):
                    yt[h][n, i] = seq[j]["v"]
                    mt[h][n, i] = 1.0
                    todt[h][n, i] = seq[j].get("tod", 0.5)
                    firstt[h][n, i] = seq[j].get("first", 0.0)
    tt = lambda a: torch.tensor(a)
    return dict(y=tt(y), e=tt(e), tod=tt(tod), first=tt(first), mask=tt(mask), reset=tt(reset),
                yt={h: tt(v) for h, v in yt.items()}, mt={h: tt(v) for h, v in mt.items()},
                todt={h: tt(v) for h, v in todt.items()}, firstt={h: tt(v) for h, v in firstt.items()})


def subset(d, rows):
    """Select participant rows of a tensor dict."""
    rows = torch.as_tensor(rows)
    o = {}
    for k, v in d.items():
        o[k] = {h: t[rows] for h, t in v.items()} if isinstance(v, dict) else v[rows]
    return o


def fit(model, d, idx, iters=300, lr=0.03, lam=10.0, horizons=(1, 2, 3), log=None, seed=0):
    """Full-batch Adam with the hierarchical L2 penalty on delta[idx]."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    idx_t = torch.as_tensor(idx)
    best, best_state = float("inf"), None
    for it in range(iters):
        opt.zero_grad()
        res = model(d, idx_t, H=max(horizons))
        loss = model.nll(d, idx_t, res, horizons)
        # penalty per target row, so lam is on the same scale as the mean NLL
        n_rows = float(sum(float(d["mt"][h].sum()) for h in horizons))
        pen = 0.0 if model.no_hier else lam * (model.delta[idx_t] ** 2).sum() / max(n_rows, 1.0)
        tot = loss + pen
        tot.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if float(tot.detach()) < best:
            best = float(tot.detach())
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if log and (it % 50 == 0 or it == iters - 1):
            log(f"  it {it} nll {float(loss):.4f} pen {float(pen):.4f}")
    model.load_state_dict(best_state)
    return best


def load_group(model, gp):
    """Set the group parameters of a model from a dict of unconstrained values (fit_v2 output)."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n == "delta" or n not in gp:
                continue
            v = gp[n]
            p.copy_(torch.tensor(v, dtype=p.dtype) if isinstance(v, list) else torch.tensor(float(v)))
    return model


def drive_sequence(model, y, e=None, tod=None, first=None, reset=None, H=1):
    """Run the filter on one sequence (numpy arrays) with the model's group
    parameters and no participant deviation. Returns per-step arrays."""
    T = len(y)
    z = lambda a, fill=0.0: torch.tensor((np.full(T, fill, np.float32) if a is None else np.asarray(a, np.float32)))[None, :]
    d = dict(y=z(y), e=z(e), tod=z(tod, 0.5), first=z(first), mask=torch.ones(1, T), reset=z(reset))
    if reset is None:
        d["reset"][0, 0] = 1.0
    d["yt"] = {h: torch.zeros(1, T) for h in range(1, H + 1)}
    d["mt"] = {h: torch.zeros(1, T) for h in range(1, H + 1)}
    d["todt"] = {h: torch.full((1, T), 0.5) for h in range(1, H + 1)}
    d["firstt"] = {h: torch.zeros(1, T) for h in range(1, H + 1)}
    model.eval()
    with torch.no_grad():
        res = model(d, torch.tensor([0]), H=H)
    return {k: v[0].numpy() for k, v in res.items()}


def empirical_bayes_lambda(model, idx, floor=1.0, ceil=1e4):
    """lambda = 1 / var(delta) across the training participants (per-dimension mean)."""
    with torch.no_grad():
        dv = model.delta[torch.as_tensor(idx)]
        var = float((dv ** 2).mean())
    return float(np.clip(1.0 / max(var, 1e-8), floor, ceil))
