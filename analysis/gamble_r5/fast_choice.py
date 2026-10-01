"""Fast-affect coupling to choice on the Rutledge gamble task (one fold).

The joint model J (analysis/joint_gamble/joint_model.py) lets only the slow mood estimate
m_t = E[a - a0 + M_{t-1}] touch choice, through optimism (eta) and precision (zeta). The joint-fit
report flagged that recent outcomes predict choice beyond prospect theory. Here the fast affect
readout r_{t-1} = sum_s gamma^(t-1-s) v_s (the same quantity that enters the happiness rating)
also enters choice:

  optimism   p_t  = sigmoid(eta m_t + eta_f r_{t-1})
  precision  mu_t = mu exp(zeta m_t + zeta_f r_{t-1})
  bias       z_t  = mu_t [p_t U(win) + (1-p_t) U(lose) - U(certain)] + b + b_f r_{t-1}

Nested test: every parameter of the fold's fitted J or C_PT (analysis/joint_gamble/out/fold{F}.json)
is held fixed and only the new coupling terms are fitted, by maximum likelihood on a random subsample
of the fold's training participants (--sub, default 12,000). The added model therefore contains the
reference model exactly (new terms at zero). Models:
  J_fast      J plus eta_f, zeta_f, b_f (happiness and choice fitted jointly)
  C_PT_fast   prospect-theory choice only, plus eta_f, zeta_f, b_f with the fast readout computed
              from the fold's H_RM happiness weights (held fixed)
  C_PT_lag    prospect-theory choice plus model-free lags: last outcome and last unsigned
              prediction error
Scored on the held-out participants: choice log-likelihood per choice (all trials), and on trials
16-30 after per-person choice offsets (b, logmu) are fitted on trials 1-15, as in run_fold.py.

Usage: python analysis/gamble_r5/fast_choice.py FOLD [--iters 150] [--sub 12000] [--threads 4] [--set 1|2]
Set 2 adds choice perseveration (the previous choice) as a control: C_PT_pers, C_PT_pers_lag,
C_PT_pers_fast, J_pers, J_pers_fast.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis/joint_gamble"))
import joint_model as jm  # noqa: E402

torch.set_default_dtype(torch.float64)
T = jm.T
HALF = np.arange(T) < 15


def forward_fast(D, cfg, P, dper=None):
    """jm.forward with fast-affect terms in the choice rule. cfg: choice_only (bool), lag (bool)."""
    N = D.N
    X = D.chan
    w = P["w"].expand(N, 3)
    gam = torch.sigmoid(P["gam"])
    rho = torch.sigmoid(P["rho"]); k = P["k"]; q = P["logq"].exp()
    a0 = P["a0"]; tau2 = (2 * P["logtau"]).exp(); s2 = (2 * P["logsig"]).exp()
    mu_i = P["logmu"].expand(N); b = P["b"].expand(N)
    if dper is not None:
        if "b" in dper:
            b = b + dper["b"]
        if "logmu" in dper:
            mu_i = mu_i + dper["logmu"]
    lam = P["loglam"].exp(); alpha = jm.softplus_alpha(P["alpha"])
    U = lambda x: torch.where(x >= 0, (x.abs() + 1e-9) ** alpha, -lam * (x.abs() + 1e-9) ** alpha)
    choice_only = cfg.get("choice_only", False)
    eta = P["eta"] if not choice_only else torch.zeros(1)
    zeta = P["zeta"] if not choice_only else torch.zeros(1)
    eta_f = P.get("eta_f", torch.zeros(1)); zeta_f = P.get("zeta_f", torch.zeros(1)); bf = P.get("bf", torch.zeros(1))
    lag = cfg.get("lag", False)
    c_out = P.get("c_out", torch.zeros(1)); c_abs = P.get("c_abs", torch.zeros(1))
    c_ch = P.get("c_ch", torch.zeros(1))          # choice perseveration (previous choice coded +1/-1)

    ma = a0.expand(N).clone(); mM = torch.zeros(N)
    P11 = tau2.expand(N).clone(); P12 = torch.zeros(N); P22 = torch.zeros(N)
    fast = torch.zeros(N)

    def rating_update(y, extra, mask):
        nonlocal ma, mM, P11, P12, P22
        pred = ma + mM + extra
        S = P11 + 2 * P12 + P22 + s2
        e = y - pred
        ll = -0.5 * (torch.log(2 * np.pi * S) + e ** 2 / S)
        K1 = (P11 + P12) / S; K2 = (P12 + P22) / S
        ma = torch.where(mask, ma + K1 * e, ma); mM = torch.where(mask, mM + K2 * e, mM)
        P11n = P11 - K1 * K1 * S; P12n = P12 - K1 * K2 * S; P22n = P22 - K2 * K2 * S
        P11 = torch.where(mask, P11n, P11); P12 = torch.where(mask, P12n, P12); P22 = torch.where(mask, P22n, P22)
        return torch.where(mask, ll, torch.zeros_like(ll))

    use_hap = not choice_only
    ones = torch.ones(N, dtype=torch.bool)
    hll_pre = rating_update(D.hpre, torch.zeros(N), ones) if use_hap else torch.zeros(N)
    hll, cll = [], []
    prev_out = torch.zeros(N); prev_abs = torch.zeros(N); prev_ch = torch.zeros(N)
    for t in range(T):
        m = (ma - a0) + mM if not choice_only else torch.zeros(N)
        p = torch.sigmoid(eta * m + eta_f * fast)
        uw, ul, uc = U(D.win[:, t]), U(D.lose[:, t]), U(D.cert[:, t])
        du = p * uw + (1 - p) * ul - uc
        mut = mu_i.exp() * torch.exp(zeta * m + zeta_f * fast)
        z = mut * du + b + bf * fast
        if lag:
            z = z + c_out * prev_out + c_abs * prev_abs
        z = z + c_ch * prev_ch
        y = D.chose[:, t]
        cll.append(y * torch.nn.functional.logsigmoid(z) + (1 - y) * torch.nn.functional.logsigmoid(-z))
        if use_hap:   # EKF update of the slow state from the choice, as in model J
            pi = torch.sigmoid(z)
            g = mut * ((uw - ul) * p * (1 - p) * eta) + zeta * mut * du
            W = (pi * (1 - pi)).clamp_min(1e-6)
            sP = P11 + 2 * P12 + P22
            s = g * g * sP + 1.0 / W
            h1 = (P11 + P12) * g; h2 = (P12 + P22) * g
            P11 = P11 - h1 * h1 / s; P12 = P12 - h1 * h2 / s; P22 = P22 - h2 * h2 / s
            r = y - pi
            ma = ma + (P11 + P12) * g * r; mM = mM + (P12 + P22) * g * r
        v = (X[:, t, :] * w).sum(-1)
        fast = gam * fast + v
        mM = rho * mM + k * v
        P22 = rho * rho * P22 + q; P12 = rho * P12
        prev_out = D.out[:, t]; prev_abs = -X[:, t, 2]; prev_ch = 2 * D.chose[:, t] - 1
        if use_hap:
            hll.append(rating_update(D.hap[:, t], fast, D.hmask[:, t]))
    return {"cll": torch.stack(cll, 1),
            "hll": torch.stack(hll, 1) if hll else torch.zeros(N, T), "hll_pre": hll_pre}


def to_params(d, extra):
    P = {k: torch.tensor(v, requires_grad=True) for k, v in d.items() if not k.startswith("_")}
    for k in extra:
        P[k] = torch.zeros(1, requires_grad=True)
    return P


def fit(D, cfg, P, names, iters, lr):
    opt = torch.optim.Adam([P[n] for n in names], lr=lr)
    for it in range(iters):
        opt.zero_grad()
        o = forward_fast(D, cfg, P)
        ll = o["cll"].sum()
        if not cfg.get("choice_only", False):
            ll = ll + o["hll"].sum() + o["hll_pre"].sum()
        loss = -ll / D.N
        loss.backward(); opt.step()
        if it % 50 == 0:
            print(f"    it {it} loss {loss.item():.6f}", flush=True)
    return P, loss.item()


def ho_choice(D, cfg, P, sb, sm, iters=250, lr=0.05):
    dper = {"b": torch.zeros(D.N, requires_grad=True), "logmu": torch.zeros(D.N, requires_grad=True)}
    Pd = {k: v.detach() for k, v in P.items()}
    tm = torch.tensor(HALF)
    opt = torch.optim.Adam(list(dper.values()), lr=lr)
    for it in range(iters):
        opt.zero_grad()
        o = forward_fast(D, cfg, Pd, dper)
        ll = (o["cll"] * tm).sum()
        if not cfg.get("choice_only", False):
            ll = ll + (o["hll"] * tm).sum()
        prior = -0.5 * ((dper["b"] ** 2).sum() / sb ** 2 + (dper["logmu"] ** 2).sum() / sm ** 2)
        loss = -(ll + prior) / D.N
        loss.backward(); opt.step()
    o = forward_fast(D, cfg, Pd, {k: v.detach() for k, v in dper.items()})
    return (o["cll"].detach() * torch.tensor(~HALF)).sum(1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fold", type=int); ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sub", type=int, default=12000)
    ap.add_argument("--set", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    npz = np.load(ROOT / "analysis/joint_gamble/out/gbe_first_play.npz")
    N = len(npz["cert"])
    folds = np.array_split(np.random.RandomState(0).permutation(N), 5)
    te = np.sort(folds[a.fold]); tr = np.sort(np.concatenate([f for i, f in enumerate(folds) if i != a.fold]))
    tr = np.sort(np.random.RandomState(100 + a.fold).choice(tr, min(a.sub, len(tr)), replace=False))
    Dtr, Dte = jm.Data(npz, tr), jm.Data(npz, te)
    fj = json.loads((ROOT / f"analysis/joint_gamble/out/fold{a.fold}.json").read_text())
    pri = fj["choice_offset_prior"]; sb, sm = pri["sd_b"], pri["sd_logmu"]
    res = {"fold": a.fold, "ntest": Dte.N, "models": {}}
    scores = {}

    # reference: J and C_PT as fitted in the joint run, rescored with this code path
    refs = [("J_ref", "J", {}), ("C_PT_ref", "C_PT", {"choice_only": True})] if a.set == 1 else []
    for name, src, cfg in refs:
        P = to_params(fj["models"][src]["params"], [])
        if src == "C_PT":   # happiness parameters irrelevant; fast readout off
            pass
        o = forward_fast(Dte, cfg, {k: v.detach() for k, v in P.items()})
        scores[name] = {"cll": o["cll"].sum(1).detach().numpy(),
                        "ho": ho_choice(Dte, cfg, P, sb, sm)}
        res["models"][name] = {"test_cll_per_choice": float(o["cll"].mean())}
        print(name, res["models"][name], flush=True)

    specs = [
        ("J_fast", "J", {}, ["eta_f", "zeta_f", "bf"], ["eta_f", "zeta_f", "bf"]),
        ("C_PT_fast", "C_PT", {"choice_only": True}, ["eta_f", "zeta_f", "bf"], ["eta_f", "zeta_f", "bf"]),
        ("C_PT_lag", "C_PT", {"choice_only": True, "lag": True}, ["c_out", "c_abs"], ["c_out", "c_abs"]),
    ]
    if a.set == 2:   # perseveration controls: does fast affect add beyond repeating the last choice?
        specs = [
            ("C_PT_pers", "C_PT", {"choice_only": True}, ["c_ch"], ["c_ch"]),
            ("C_PT_pers_lag", "C_PT", {"choice_only": True, "lag": True}, ["c_ch", "c_out", "c_abs"], ["c_ch", "c_out", "c_abs"]),
            ("C_PT_pers_fast", "C_PT", {"choice_only": True}, ["c_ch", "eta_f", "zeta_f", "bf"], ["c_ch", "eta_f", "zeta_f", "bf"]),
            ("J_pers", "J", {}, ["c_ch"], ["c_ch"]),
            ("J_pers_fast", "J", {}, ["c_ch", "eta_f", "zeta_f", "bf"], ["c_ch", "eta_f", "zeta_f", "bf"]),
        ]
    for name, src, cfg, extra, names in specs:
        t0 = time.time()
        P = to_params(fj["models"][src]["params"], extra)
        if src == "C_PT":   # the fast readout uses the fold's H_RM happiness weights, held fixed
            for kk in ["w", "gam"]:
                P[kk] = torch.tensor(fj["models"]["H_RM"]["params"][kk])
        P, loss = fit(Dtr, cfg, P, names, a.iters, 0.02)
        o = forward_fast(Dte, cfg, {k: v.detach() for k, v in P.items()})
        hll = float((o["hll"].sum() + o["hll_pre"].sum()) / (Dte.hmask.sum() + Dte.N)) if not cfg.get("choice_only") else None
        scores[name] = {"cll": o["cll"].sum(1).detach().numpy(), "ho": ho_choice(Dte, cfg, P, sb, sm)}
        res["models"][name] = {"test_cll_per_choice": float(o["cll"].mean()), "test_hll_per_rating": hll,
                               "train_loss": loss, "seconds": time.time() - t0,
                               "params": {kk: P[kk].detach().numpy().tolist() for kk in names}}
        print(name, {kk: vv for kk, vv in res["models"][name].items() if kk != "params"},
              {kk: res["models"][name]["params"][kk] for kk in extra}, flush=True)
    tag = "fast" if a.set == 1 else "fast2"
    np.savez_compressed(HERE / f"out/{tag}_scores_fold{a.fold}.npz",
                        **{f"{m}__{k}": v for m, s in scores.items() for k, v in s.items()})
    (HERE / f"out/{tag}_fold{a.fold}.json").write_text(json.dumps(res, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
