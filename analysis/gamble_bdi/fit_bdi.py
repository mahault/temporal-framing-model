"""Per-person parameters of the joint happiness and choice model for the BDI participants.

Each BDI participant was a held-out participant in exactly one fold of the joint gamble fit
(analysis/joint_gamble/run_fold.py, permutation seed 0, 5 folds). Their per-person parameters are
estimated with the group parameters of model J from that fold, so the group fit never saw them.

Per-person MAP offsets around the group values, Gaussian shrinkage priors:
  a      person baseline                         prior sd = tau (the fitted between-person sd)
  rho    mood persistence, logit scale            prior sd 1.0
  wF wP wB channel weights                         prior sd 0.03 (chosen by split-half CV in the joint fit)
  eta    mood -> optimism coupling                prior sd 1.0
  b      choice bias                              prior sd 1.0
  logmu  log choice precision                     prior sd 0.5
Posterior sds from the diagonal of the Hessian of the negative log posterior (exact per person,
because people are independent given the group parameters).

Usage: python analysis/gamble_bdi/fit_bdi.py
Writes analysis/gamble_bdi/out/person_params.npz (git-ignored).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis/joint_gamble"))
import joint_model as jm  # noqa: E402  (read-only import)

torch.set_default_dtype(torch.float64)
T = jm.T
SD = {"a": None, "rho": 1.0, "w": 0.03, "eta": 1.0, "b": 1.0, "logmu": 0.5}


def load_bdi():
    d = sio.loadmat(ROOT / "data_raw/rutledge_gbe/Rutledge_GBE_risk_data_TOD.mat",
                    squeeze_me=True, struct_as_record=False)
    rows = []
    for s in d["depData"]:
        rows.append({
            "id": float(np.atleast_1d(s.id)[0]),
            "bdi": float(s.bdiTotal),
            "age": float(np.atleast_1d(s.age)[0]),
            "female": float(np.atleast_1d(s.isFemale)[0]),
            "depStatus": float(s.depStatus),
            "lifeSat": float(np.atleast_1d(s.lifeSatisfaction)[0]),
        })
    return rows


def forward_person(D, P, off):
    """Joint model J forward pass with per-person a, rho, w, eta, b, logmu.

    Mirrors jm.forward for cfg readout 'chan', mood on, choice 'pt', couple 'filt', ekf True.
    """
    N = D.N
    X = D.chan
    w = P["w"].expand(N, 3) + off["w"]
    gam = torch.sigmoid(P["gam"])
    rho = torch.sigmoid(P["rho"] + off["rho"])
    k = P["k"]; q = P["logq"].exp()
    a0_grp = P["a0"]
    a0 = a0_grp + off["a"]
    tau2 = (2 * P["logtau"]).exp(); s2 = (2 * P["logsig"]).exp()
    lam = P["loglam"].exp(); alpha = jm.softplus_alpha(P["alpha"])
    U = lambda x: torch.where(x >= 0, (x.abs() + 1e-9) ** alpha, -lam * (x.abs() + 1e-9) ** alpha)
    eta = P["eta"] + off["eta"]; zeta = P["zeta"]
    b = P["b"] + off["b"]; logmu = P["logmu"] + off["logmu"]

    # the person baseline is now a point estimate: state (a, M) with a known, so P11 = 0
    ma = a0.clone(); mM = torch.zeros(N)
    P11 = torch.zeros(N); P12 = torch.zeros(N); P22 = torch.zeros(N)
    fast = torch.zeros(N)
    hll = []; cll = []

    def rating_update(y, extra, mask):
        nonlocal ma, mM, P11, P12, P22
        pred = ma + mM + extra
        S = P11 + 2 * P12 + P22 + s2
        e = y - pred
        ll = -0.5 * (torch.log(2 * np.pi * S) + e ** 2 / S)
        K1 = (P11 + P12) / S; K2 = (P12 + P22) / S
        ma_n = ma + K1 * e; mM_n = mM + K2 * e
        P11_n = P11 - K1 * K1 * S; P12_n = P12 - K1 * K2 * S; P22_n = P22 - K2 * K2 * S
        ma = torch.where(mask, ma_n, ma); mM = torch.where(mask, mM_n, mM)
        P11 = torch.where(mask, P11_n, P11); P12 = torch.where(mask, P12_n, P12)
        P22 = torch.where(mask, P22_n, P22)
        return torch.where(mask, ll, torch.zeros_like(ll))

    hpre = rating_update(D.hpre, torch.zeros(N), torch.ones(N, dtype=torch.bool))
    for t in range(T):
        m = (ma - a0_grp) + mM
        p = torch.sigmoid(eta * m)
        uw, ul, uc = U(D.win[:, t]), U(D.lose[:, t]), U(D.cert[:, t])
        du = p * uw + (1 - p) * ul - uc
        mut = logmu.exp() * torch.exp(zeta * m)
        z = mut * du + b
        y = D.chose[:, t]
        cll.append(y * torch.nn.functional.logsigmoid(z) + (1 - y) * torch.nn.functional.logsigmoid(-z))
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
        hll.append(rating_update(D.hap[:, t], fast, D.hmask[:, t]))
    return torch.stack(hll, 1).sum(1) + hpre + torch.stack(cll, 1).sum(1)


def neg_log_post(D, P, off, sd):
    ll = forward_person(D, P, off)
    pr = 0.0
    for k, v in off.items():
        s = sd[k]
        pr = pr - 0.5 * (v ** 2 / s ** 2).reshape(D.N, -1).sum(1)
    return -(ll + pr)          # per person


def fit_fold(D, P, iters=600, lr=0.03):
    N = D.N
    tau = float(P["logtau"].exp())
    sd = dict(SD); sd["a"] = tau
    off = {"a": torch.zeros(N), "rho": torch.zeros(N), "w": torch.zeros(N, 3),
           "eta": torch.zeros(N), "b": torch.zeros(N), "logmu": torch.zeros(N)}
    for v in off.values():
        v.requires_grad_(True)
    opt = torch.optim.Adam(list(off.values()), lr=lr)
    for it in range(iters):
        opt.zero_grad()
        loss = neg_log_post(D, P, off, sd).sum() / N
        loss.backward()
        opt.step()
        if it % 150 == 0:
            print(f"    it {it} loss/person {loss.item():.4f}", flush=True)
    # diagonal Hessian per person via Hessian-vector products (people are independent)
    total = lambda: neg_log_post(D, P, off, sd).sum()
    params = list(off.values())
    grads = torch.autograd.grad(total(), params, create_graph=True)
    hdiag = {}
    names = list(off.keys())
    for name, gi, pi in zip(names, grads, params):
        if pi.dim() == 1:
            hv = torch.autograd.grad(gi, pi, grad_outputs=torch.ones_like(gi), retain_graph=True)[0]
            hdiag[name] = hv.detach()
        else:
            cols = []
            for c in range(pi.shape[1]):
                vec = torch.zeros_like(gi); vec[:, c] = 1.0
                hv = torch.autograd.grad(gi, pi, grad_outputs=vec, retain_graph=True)[0]
                cols.append(hv[:, c].detach())
            hdiag[name] = torch.stack(cols, 1)
    psd = {k: (1.0 / v.clamp_min(1e-9)).sqrt() for k, v in hdiag.items()}
    return {k: v.detach() for k, v in off.items()}, psd


def main():
    npz = np.load(ROOT / "analysis/joint_gamble/out/gbe_first_play.npz")
    ids = npz["id"]
    Nall = len(ids)
    perm = np.random.RandomState(0).permutation(Nall)
    folds = np.array_split(perm, 5)
    fold_of = np.empty(Nall, int)
    for f, ix in enumerate(folds):
        fold_of[ix] = f
    pos = {v: i for i, v in enumerate(ids)}
    bdi_rows = load_bdi()
    kept = [r for r in bdi_rows if r["id"] in pos]
    print(f"BDI participants {len(bdi_rows)}, in the first-play set {len(kept)}")
    rowidx = np.array([pos[r["id"]] for r in kept])

    res = {k: [] for k in ["row", "fold", "a", "rho", "wF", "wP", "wB", "eta", "b", "logmu",
                           "sd_a", "sd_rho", "sd_wF", "sd_wP", "sd_wB", "sd_eta", "sd_b", "sd_logmu",
                           "a_grp", "rho_grp", "eta_grp", "w_grp_F", "w_grp_P", "w_grp_B"]}
    group = {}
    for f in range(5):
        sel = rowidx[fold_of[rowidx] == f]
        if len(sel) == 0:
            continue
        J = json.loads((ROOT / f"analysis/joint_gamble/out/fold{f}.json").read_text())["models"]["J"]["params"]
        P = {k: torch.tensor(v) for k, v in J.items() if not k.startswith("_")}
        group[f] = {k: v for k, v in J.items() if not k.startswith("_")}
        D = jm.Data(npz, np.sort(sel))
        order = np.sort(sel)
        print(f"fold {f}: {D.N} BDI participants", flush=True)
        off, psd = fit_fold(D, P)
        res["row"] += order.tolist(); res["fold"] += [f] * D.N
        res["a"] += (P["a0"] + off["a"]).tolist()
        res["rho"] += (P["rho"] + off["rho"]).tolist()           # logit scale
        for c, nm in enumerate("FPB"):
            res["w" + nm] += (P["w"][c] + off["w"][:, c]).tolist()
            res["sd_w" + nm] += psd["w"][:, c].tolist()
            res["w_grp_" + nm] += [float(P["w"][c])] * D.N
        res["eta"] += (P["eta"] + off["eta"]).tolist()
        res["b"] += (P["b"] + off["b"]).tolist()
        res["logmu"] += (P["logmu"] + off["logmu"]).tolist()
        for k in ["a", "rho", "eta", "b", "logmu"]:
            res["sd_" + k] += psd[k].tolist()
        res["a_grp"] += [float(P["a0"])] * D.N
        res["rho_grp"] += [float(P["rho"])] * D.N
        res["eta_grp"] += [float(P["eta"])] * D.N
    out = {k: np.asarray(v, float) for k, v in res.items()}
    byrow = {r["id"]: r for r in kept}
    for k in ["bdi", "age", "female", "depStatus", "lifeSat"]:
        out[k] = np.array([byrow[ids[int(i)]][k] for i in out["row"]], float)
    out["id"] = np.array([ids[int(i)] for i in out["row"]], float)
    (HERE / "out").mkdir(exist_ok=True)
    np.savez_compressed(HERE / "out/person_params.npz", **out)
    (HERE / "out/group_params.json").write_text(json.dumps(group, indent=1))
    print("saved", len(out["row"]))


if __name__ == "__main__":
    main()
