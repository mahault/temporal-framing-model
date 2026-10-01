"""Joint model of momentary happiness and risky choice on the Rutledge GBE task.

One generative model per participant with group-level parameters and marginalised
person-level random effects.

Affect (three channels, the paper's readout on the gamble task-model, utility linear in points):
  forward  F_t = E[U | chosen option]      (EV of the gamble, or the certain amount)
  present  P_t = U(outcome) - E[U|chosen]   (reward prediction error, 0 after a safe choice)
  backward B_t = -|P_t|                     (poor model fit, change in model evidence)
  instantaneous affect  v_t = w_F F_t + w_P P_t + w_B B_t
  fast readout          r_t = sum_s gamma^(t-s) v_s
Slow mood (two timescales, as in model v2.1):
  M_t = rho M_{t-1} + k v_t + noise(q)
Happiness rating after trial t:
  h_t = a + M_t + r_t + noise(sigma^2),   a ~ N(a0, tau^2)  (person baseline, marginalised)
The latent state (a, M) is tracked with a Kalman filter, so every rating is scored one step
ahead from the task history and the participant's own earlier ratings.

Choice on trial t (prospect theory, 50/50 gamble vs certain amount):
  U(x) = x^alpha for x >= 0, -lambda (-x)^alpha for x < 0
  current mood  m_t = E[a - a0 + M_{t-1} | data before trial t]
  optimism      p_t = sigmoid(eta m_t)            subjective probability of the better outcome
  precision     mu_t = mu exp(zeta m_t)
  P(gamble) = sigmoid(mu_t [p_t U(win) + (1-p_t) U(lose) - U(certain)] + b)
Mood enters choice through the positive-belief (optimism) channel eta and the policy-precision
channel zeta. With eta = zeta = 0 the choice model is ordinary prospect theory.
Optionally each choice is also an observation of mood (Laplace / extended-Kalman update).
"""
from __future__ import annotations
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
T = 30


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self, npz, idx):
        g = lambda k: torch.tensor(npz[k][idx])
        self.cert, self.win, self.lose = g("cert"), g("win"), g("lose")
        self.chose, self.out = g("chose"), g("out")
        hap = npz["hap"][idx]
        self.hmask = torch.tensor(~np.isnan(hap))
        self.hap = torch.tensor(np.nan_to_num(hap))
        self.hpre = g("hap_pre")
        self.N = len(idx)
        evg = 0.5 * (self.win + self.lose)
        g1 = self.chose == 1
        F = torch.where(g1, evg, self.cert)
        P = torch.where(g1, self.out - evg, torch.zeros_like(evg))
        B = -P.abs()
        self.chan = torch.stack([F, P, B], -1)
        CR = torch.where(g1, torch.zeros_like(evg), self.cert)
        EV = torch.where(g1, evg, torch.zeros_like(evg))
        self.he = torch.stack([CR, EV, P], -1)

    def zhap(self):
        """Per-participant z-scored post-trial ratings (the paper's target)."""
        h = self.hap.clone(); m = self.hmask
        n = m.sum(1, keepdim=True)
        mu = (h * m).sum(1, keepdim=True) / n
        sd = (((h - mu) ** 2) * m).sum(1, keepdim=True).div(n).sqrt().clamp_min(1e-6)
        return torch.where(m, (h - mu) / sd, torch.zeros_like(h))


# ----------------------------------------------------------------------------- params
def init_params():
    p = {
        "w": [0.05, 0.10, 0.02], "w_eq": [0.05], "gam": [0.0],
        "a0": [0.6], "logtau": [np.log(0.15)], "logsig": [np.log(0.08)],
        "rho": [2.0], "k": [0.0], "logq": [np.log(1e-4)],
        "logmu": [np.log(5.0)], "loglam": [np.log(1.5)], "alpha": [0.0], "b": [0.3],
        "eta": [0.0], "zeta": [0.0],
    }
    return {k: torch.tensor(v, requires_grad=True) for k, v in p.items()}


def softplus_alpha(x):
    return 0.2 + 1.3 * torch.sigmoid(x)       # alpha in (0.2, 1.5), 0 -> 0.85


# ----------------------------------------------------------------------------- forward
def forward(D, cfg, P, dper=None):
    """Return per-trial log-likelihood arrays and open-loop predictions.

    cfg keys: readout 'chan'|'he'; chmask (3 floats); equal_w; mood; use_hap;
              choice None|'ev'|'pt'; couple None|'filt'|'task'; ekf
    dper: optional per-participant offsets {'w':(N,3), 'b':(N,), 'logmu':(N,)}
    """
    N = D.N
    X = D.chan if cfg.get("readout", "chan") == "chan" else D.he
    X = X * torch.tensor(cfg.get("chmask", [1.0, 1.0, 1.0]))
    w = P["w_eq"].expand(3) if cfg.get("equal_w") else P["w"]
    w = w.expand(N, 3)
    if dper is not None and "w" in dper:
        w = w + dper["w"]
    gam = torch.sigmoid(P["gam"])
    rho = torch.sigmoid(P["rho"]); k = P["k"]; q = P["logq"].exp()
    if not cfg.get("mood", True):
        k = torch.zeros_like(k); q = torch.zeros_like(q)
    a0 = P["a0"]; tau2 = (2 * P["logtau"]).exp(); s2 = (2 * P["logsig"]).exp()

    choice = cfg.get("choice")
    couple = cfg.get("couple")
    ekf = cfg.get("ekf", False)
    if choice is not None:
        mu = P["logmu"].exp(); b = P["b"].expand(N)
        logmu_i = P["logmu"].expand(N)
        if dper is not None and "b" in dper:
            b = b + dper["b"]
        if dper is not None and "logmu" in dper:
            logmu_i = logmu_i + dper["logmu"]
        if choice == "pt":
            lam = P["loglam"].exp(); alpha = softplus_alpha(P["alpha"])
            U = lambda x: torch.where(x >= 0, (x.abs() + 1e-9) ** alpha, -lam * (x.abs() + 1e-9) ** alpha)
        else:
            U = lambda x: x
        eta = P["eta"] if couple else torch.zeros(1)
        zeta = P["zeta"] if couple else torch.zeros(1)

    # Kalman state for (a, M)
    ma = a0.expand(N).clone(); mM = torch.zeros(N)
    P11 = tau2.expand(N).clone(); P12 = torch.zeros(N); P22 = torch.zeros(N)
    fast = torch.zeros(N); Mopen = torch.zeros(N)

    def rating_update(y, pred_extra, mask):
        nonlocal ma, mM, P11, P12, P22
        pred = ma + mM + pred_extra
        S = P11 + 2 * P12 + P22 + s2
        e = y - pred
        ll = -0.5 * (torch.log(2 * np.pi * S) + e ** 2 / S)
        K1 = (P11 + P12) / S; K2 = (P12 + P22) / S
        ma_n = ma + K1 * e; mM_n = mM + K2 * e
        P11_n = P11 - K1 * K1 * S; P12_n = P12 - K1 * K2 * S; P22_n = P22 - K2 * K2 * S
        ma = torch.where(mask, ma_n, ma); mM = torch.where(mask, mM_n, mM)
        P11 = torch.where(mask, P11_n, P11); P12 = torch.where(mask, P12_n, P12)
        P22 = torch.where(mask, P22_n, P22)
        return torch.where(mask, ll, torch.zeros_like(ll)), pred

    use_hap = cfg.get("use_hap", True)
    ones = torch.ones(N, dtype=torch.bool)
    hll_pre, _ = rating_update(D.hpre, torch.zeros(N), ones) if use_hap else (torch.zeros(N), None)

    hll = []; cll = []; popen = []; pfilt = []
    for t in range(T):
        # ---- choice on trial t
        if choice is not None:
            if couple == "filt":
                m = (ma - a0) + mM
            elif couple == "task":
                m = Mopen
            else:
                m = torch.zeros(N)
            p = torch.sigmoid(eta * m)
            uw, ul, uc = U(D.win[:, t]), U(D.lose[:, t]), U(D.cert[:, t])
            du = p * uw + (1 - p) * ul - uc
            mut = logmu_i.exp() * torch.exp(zeta * m)
            z = mut * du + b
            y = D.chose[:, t]
            cll.append(y * torch.nn.functional.logsigmoid(z) + (1 - y) * torch.nn.functional.logsigmoid(-z))
            if ekf and couple == "filt":
                pi = torch.sigmoid(z)
                g = mut * ((uw - ul) * p * (1 - p) * eta) + zeta * mut * du
                W = (pi * (1 - pi)).clamp_min(1e-6)
                sP = P11 + 2 * P12 + P22
                s = g * g * sP + 1.0 / W
                h1 = (P11 + P12) * g; h2 = (P12 + P22) * g
                P11 = P11 - h1 * h1 / s; P12 = P12 - h1 * h2 / s; P22 = P22 - h2 * h2 / s
                r = y - pi
                ma = ma + (P11 + P12) * g * r; mM = mM + (P12 + P22) * g * r
        # ---- outcome, affect, mood transition
        v = (X[:, t, :] * w).sum(-1)
        fast = gam * fast + v
        mM = rho * mM + k * v
        P22 = rho * rho * P22 + q; P12 = rho * P12
        Mopen = rho * Mopen + k * v
        popen.append(a0 + Mopen + fast)
        # ---- rating after trial t
        if use_hap:
            l, pr = rating_update(D.hap[:, t], fast, D.hmask[:, t])
            hll.append(l); pfilt.append(pr)
    out = {
        "hll": torch.stack(hll, 1) if hll else torch.zeros(N, T),
        "hll_pre": hll_pre,
        "cll": torch.stack(cll, 1) if cll else torch.zeros(N, T),
        "popen": torch.stack(popen, 1),
        "pfilt": torch.stack(pfilt, 1) if pfilt else torch.zeros(N, T),
    }
    return out


def objective(out, cfg):
    tot = torch.zeros(())
    if cfg.get("use_hap", True) and cfg.get("fit_hap", True):
        tot = tot + out["hll"].sum() + out["hll_pre"].sum()
    if cfg.get("choice") is not None and cfg.get("fit_choice", True):
        tot = tot + out["cll"].sum()
    return tot


ACTIVE = {  # parameters each part of the model uses
    "hap": ["gam", "a0", "logtau", "logsig"],
    "w": ["w"], "w_eq": ["w_eq"],
    "mood": ["rho", "k", "logq"],
    "ev": ["logmu", "b"], "pt": ["logmu", "loglam", "alpha", "b"],
    "couple": ["eta", "zeta"],
}


def active_names(cfg):
    names = []
    if cfg.get("use_hap", True) or cfg.get("couple") == "task":
        names += ["gam"] + (["w_eq"] if cfg.get("equal_w") else ["w"])
        if cfg.get("mood", True):
            names += ACTIVE["mood"]
    if cfg.get("use_hap", True):
        names += ["a0", "logtau", "logsig"]
    if cfg.get("choice") is not None:
        names += ACTIVE[cfg["choice"]]
        if cfg.get("couple"):
            names += ACTIVE["couple"]
    for f in cfg.get("frozen", []):
        if f in names:
            names.remove(f)
    return sorted(set(names))


def fit(D, cfg, P=None, iters=500, lr=0.03, verbose=False):
    P = P if P is not None else init_params()
    names = active_names(cfg)
    opt = torch.optim.Adam([P[n] for n in names], lr=lr)
    nobs = D.N
    last = None
    for it in range(iters):
        opt.zero_grad()
        out = forward(D, cfg, P)
        loss = -objective(out, cfg) / nobs
        loss.backward()
        opt.step()
        if verbose and it % 100 == 0:
            print(f"    it {it} loss {loss.item():.5f}", flush=True)
        last = loss.item()
    return P, last


def fit_person_offsets(D, cfg, P, which, sd, tmask, iters=250, lr=0.05):
    """MAP per-participant offsets (group parameters fixed) using only trials in tmask.

    which: subset of {'w','b','logmu'}; sd: dict of prior sds; tmask: (T,) bool of trials
    whose choices and ratings enter the fit.
    """
    N = D.N
    dper = {}
    if "w" in which:
        dper["w"] = torch.zeros(N, 3, requires_grad=True)
    if "b" in which:
        dper["b"] = torch.zeros(N, requires_grad=True)
    if "logmu" in which:
        dper["logmu"] = torch.zeros(N, requires_grad=True)
    opt = torch.optim.Adam(list(dper.values()), lr=lr)
    Pd = {k: v.detach() for k, v in P.items()}
    tm = torch.tensor(tmask)
    for it in range(iters):
        opt.zero_grad()
        out = forward(D, cfg, Pd, dper)
        ll = torch.zeros(())
        if cfg.get("use_hap", True) and "w" in which:
            ll = ll + (out["hll"] * tm).sum() + out["hll_pre"].sum()
        if cfg.get("choice") is not None and ("b" in which or "logmu" in which):
            ll = ll + (out["cll"] * tm).sum()
        prior = sum(-0.5 * (v ** 2).sum() / sd[k] ** 2 for k, v in dper.items())
        loss = -(ll + prior) / N
        loss.backward()
        opt.step()
    return {k: v.detach() for k, v in dper.items()}


def pack(P):
    out = {}
    for k, v in P.items():
        x = v.detach().numpy().tolist()
        out[k] = x
    out["_derived"] = {
        "gamma": float(torch.sigmoid(P["gam"])),
        "rho": float(torch.sigmoid(P["rho"])),
        "mood_timescale_trials": float(1.0 / (1.0 - torch.sigmoid(P["rho"]) + 1e-9)),
        "fast_timescale_trials": float(1.0 / (1.0 - torch.sigmoid(P["gam"]) + 1e-9)),
        "tau": float(P["logtau"].exp()), "sigma": float(P["logsig"].exp()), "q": float(P["logq"].exp()),
        "mu": float(P["logmu"].exp()), "lambda": float(P["loglam"].exp()),
        "alpha": float(softplus_alpha(P["alpha"])),
    }
    return out
