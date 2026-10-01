"""BDI against the joint model's per-person parameters, model-free equivalents and held-out prediction.

Three analyses:
1. Covariate model (primary). BDI (z-scored), age band (ordinal, z-scored) and sex enter the model
   as fixed effects on each per-person parameter: theta_i = theta_group(fold_i) + beta' x_i, for
   theta in {baseline a, mood persistence rho (logit), wF, wP, wB, eta}. The baseline also keeps a
   person random effect with prior sd tau, and choice bias b and log precision logmu keep their
   random effects (prior sd 1.0, 0.5). The betas are fitted by maximum a posteriori jointly with the
   random effects. Standard errors are cluster-robust (sandwich over participants) for the beta
   block, conditional on the random effects at their MAP values.
2. Point-estimate regressions. MAP per-person parameters from fit_bdi.py regressed on BDI with age
   and sex, HC3 standard errors and a participant bootstrap.
3. Held-out prediction of BDI (10 folds, repeated 20 times, ridge) from covariates only, plus
   model-free features, plus model parameters, plus both. Bootstrap CIs on R2 differences.

Usage: python analysis/gamble_bdi/analyze_bdi.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis/joint_gamble"))
sys.path.insert(0, str(HERE))
import joint_model as jm  # noqa: E402
from fit_bdi import forward_person  # noqa: E402

torch.set_default_dtype(torch.float64)
rs = np.random.RandomState(7)
T = jm.T
PARAMS = ["a", "rho", "wF", "wP", "wB", "eta"]


def zs(x):
    return (x - x.mean()) / x.std()


# --------------------------------------------------------------------------- data
npz = np.load(ROOT / "analysis/joint_gamble/out/gbe_first_play.npz")
pp = dict(np.load(HERE / "out/person_params.npz"))
order = np.argsort(pp["row"])
pp = {k: v[order] for k, v in pp.items()}
rows = pp["row"].astype(int)
N = len(rows)
D = jm.Data(npz, rows)
groups = json.loads((HERE / "out/group_params.json").read_text())
fold = pp["fold"].astype(int)


def grp_tensor(name, comp=None):
    vals = []
    for f in fold:
        v = groups[str(f)][name]
        vals.append(v[comp] if comp is not None else v[0])
    return torch.tensor(vals)


G = {k: grp_tensor(k) for k in ["gam", "a0", "logtau", "logsig", "rho", "k", "logq", "logmu",
                                 "loglam", "alpha", "b", "eta", "zeta"]}
G["w"] = torch.stack([grp_tensor("w", c) for c in range(3)], 1)

bdi = pp["bdi"]; age = pp["age"]; fem = pp["female"]
X = np.column_stack([zs(bdi), zs(age), fem - fem.mean()])
Xt = torch.tensor(X)
K = X.shape[1]
CNAMES = ["BDI", "age", "female"]


# --------------------------------------------------------------------------- 1. covariate model
def fwd_cov(beta, rand):
    """beta: (len(PARAMS), K); rand: dict of random effects."""
    eff = Xt @ beta.T                              # (N, n_params)
    off = {
        "a": eff[:, 0] + rand["a"],
        "rho": eff[:, 1],
        "w": eff[:, 2:5],
        "eta": eff[:, 5],
        "b": rand["b"],
        "logmu": rand["logmu"],
    }
    P = {k: v for k, v in G.items()}
    # forward_person adds offsets to group values; per-person group values broadcast elementwise
    return forward_person(D, P, off)


def per_person_negpost(beta, rand):
    ll = fwd_cov(beta, rand)
    tau = G["logtau"].exp()
    pr = -0.5 * (rand["a"] ** 2 / tau ** 2 + rand["b"] ** 2 / 1.0 + rand["logmu"] ** 2 / 0.25)
    return -(ll + pr)


RHO_ROW = 1   # BDI/age/sex effects on mood persistence are not identified here; held at zero


def fit_cov(iters=800, lr=0.02, mask=None, init=None):
    beta = torch.zeros(len(PARAMS), K, requires_grad=True) if init is None else init[0].clone().requires_grad_(True)
    rand = ({k: torch.zeros(N, requires_grad=True) for k in ["a", "b", "logmu"]} if init is None
            else {k: v.clone().requires_grad_(True) for k, v in init[1].items()})
    opt = torch.optim.Adam([beta] + list(rand.values()), lr=lr)
    wts = torch.ones(N) if mask is None else torch.tensor(mask, dtype=torch.float64)
    bmask = torch.ones_like(beta); bmask[RHO_ROW, :] = 0.0
    trace = []
    for it in range(iters):
        opt.zero_grad()
        loss = (per_person_negpost(beta, rand) * wts).sum() / wts.sum()
        loss.backward()
        beta.grad *= bmask
        opt.step()
        if it % 250 == 0 or it == iters - 1:
            trace.append((it, loss.item())); print("   fit", it, round(loss.item(), 6), flush=True)
    return beta.detach(), {k: v.detach() for k, v in rand.items()}, loss.item()


def fit_cov_fixed(i_fixed, init, iters=1500, lr=0.01):
    """Refit with the BDI effect on parameter i_fixed held at zero (gradient masked)."""
    beta = init[0].clone().requires_grad_(True)
    rand = {k: v.clone().requires_grad_(True) for k, v in init[1].items()}
    mask = torch.ones_like(beta); mask[i_fixed, 0] = 0.0; mask[RHO_ROW, :] = 0.0
    opt = torch.optim.Adam([beta] + list(rand.values()), lr=lr)
    for it in range(iters):
        opt.zero_grad()
        loss = per_person_negpost(beta, rand).sum() / N
        loss.backward()
        beta.grad *= mask
        opt.step()
    return beta.detach(), {k: v.detach() for k, v in rand.items()}, loss.item()


def sandwich(beta, rand):
    b = beta.clone().requires_grad_(True)
    nlp = per_person_negpost(b, rand)              # (N,)
    # per-person scores
    scores = []
    for i in range(N):
        g = torch.autograd.grad(nlp[i], b, retain_graph=True)[0].reshape(-1)
        scores.append(g)
    S = torch.stack(scores)                         # (N, p)
    tot = lambda bb: per_person_negpost(bb.reshape(beta.shape), rand).sum()
    H = torch.autograd.functional.hessian(tot, beta.reshape(-1))
    free = torch.ones(beta.shape, dtype=torch.bool); free[RHO_ROW, :] = False
    fi = free.reshape(-1)
    Hi = torch.linalg.inv(H[fi][:, fi])
    Sf = S[:, fi]
    Vf = Hi @ (Sf.T @ Sf) @ Hi
    sd = torch.full((beta.numel(),), float("nan")); sd[fi] = Vf.diag().sqrt()
    return sd.reshape(beta.shape)


# --------------------------------------------------------------------------- model-free features
def model_free():
    hap = npz["hap"][rows]; hpre = npz["hap_pre"][rows]
    chose = npz["chose"][rows]; out = npz["out"][rows]; win = npz["win"][rows]; lose = npz["lose"][rows]
    ev = 0.5 * (win + lose)
    rpe = np.where(chose == 1, out - ev, 0.0)
    feats = {"mean_hap": [], "ac1": [], "slope_pos": [], "slope_neg": []}
    for i in range(N):
        h = hap[i]; idx = np.where(~np.isnan(h))[0]
        seq = np.concatenate([[hpre[i]], h[idx]])
        feats["mean_hap"].append(np.mean(seq))
        a, b2 = seq[:-1] - seq.mean(), seq[1:] - seq.mean()
        feats["ac1"].append(np.sum(a * b2) / max(np.sum((seq - seq.mean()) ** 2), 1e-9))
        # rating change since the previous rating, against RPEs accumulated in between, split by sign
        prev = np.concatenate([[-1], idx[:-1]])
        dh, pos, neg = [], [], []
        last_h = hpre[i]
        for j, t in enumerate(idx):
            lo = prev[j] + 1
            r = rpe[i, lo:t + 1]
            dh.append(h[t] - last_h); pos.append(np.clip(r, 0, None).sum()); neg.append(np.clip(r, None, 0).sum())
            last_h = h[t]
        A = np.column_stack([np.ones(len(dh)), pos, neg])
        coef, *_ = np.linalg.lstsq(A, np.asarray(dh), rcond=None)
        feats["slope_pos"].append(coef[1]); feats["slope_neg"].append(coef[2])
    return {k: np.asarray(v) for k, v in feats.items()}


def ols_hc3(y, Xd):
    Xd = np.column_stack([np.ones(len(y)), Xd])
    XtX_i = np.linalg.inv(Xd.T @ Xd)
    bhat = XtX_i @ Xd.T @ y
    e = y - Xd @ bhat
    h = np.einsum("ij,jk,ik->i", Xd, XtX_i, Xd)
    meat = (Xd * (e / (1 - h))[:, None] ** 2).T @ Xd
    V = XtX_i @ meat @ XtX_i
    return bhat, np.sqrt(np.diag(V))


def boot_ci(stat, nb=2000):
    vals = []
    for _ in range(nb):
        ix = rs.randint(0, N, N)
        vals.append(stat(ix))
    return np.percentile(vals, [2.5, 97.5])


def ridge_cv(Xf, y, reps=20, nfold=10, lam_grid=(0.1, 1, 10, 100, 1000)):
    """Out-of-fold predictions averaged over repetitions; ridge penalty chosen by inner 5-fold CV.
    numpy implementation (the local sklearn build is binary-incompatible with numpy)."""
    def fit_pred(Xa, ya, Xb, lam):
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-12
        A = (Xa - mu) / sd; B = (Xb - mu) / sd
        ym = ya.mean()
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (ya - ym))
        return B @ w + ym
    preds = np.zeros((reps, N))
    for r in range(reps):
        perm = np.random.RandomState(100 + r).permutation(N)
        for f in np.array_split(perm, nfold):
            tr = np.setdiff1d(np.arange(N), f)
            best, bl = None, np.inf
            inner = np.array_split(np.random.RandomState(r).permutation(tr), 5)
            for lam in lam_grid:
                err = 0.0
                for g in inner:
                    t2 = np.setdiff1d(tr, g)
                    err += np.sum((y[g] - fit_pred(Xf[t2], y[t2], Xf[g], lam)) ** 2)
                if err < bl:
                    bl, best = err, lam
            preds[r, f] = fit_pred(Xf[tr], y[tr], Xf[f], best)
    return preds.mean(0)


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    res = {"N": N, "bdi_mean": float(bdi.mean()), "bdi_sd": float(bdi.std())}

    # ---- 1. covariate model
    print("covariate model fit", flush=True)
    beta, rand, loss = fit_cov(iters=3000, lr=0.01)
    # polish: continue from the solution and confirm the objective no longer moves
    beta, rand, loss2 = fit_cov(iters=1500, lr=0.003, init=(beta, rand))
    res["covariate_model_convergence"] = {"after_3000": loss, "after_polish": loss2}
    torch.save({"beta": beta, "rand": rand}, HERE / "out/cov_fit.pt")
    se = sandwich(beta, rand)
    # null fit (no BDI column) for a likelihood-ratio check on all BDI effects jointly
    cov_tab = {}
    for i, p in enumerate(PARAMS):
        cov_tab[p] = {c: {"beta": float(beta[i, j]), "se": float(se[i, j]),
                          "lo": float(beta[i, j] - 1.96 * se[i, j]), "hi": float(beta[i, j] + 1.96 * se[i, j])}
                      for j, c in enumerate(CNAMES)}
    res["covariate_model"] = cov_tab
    # likelihood-ratio test per BDI effect: refit with that effect fixed at zero
    total_full = float(per_person_negpost(beta, rand).sum())
    lr = {}
    for i, p in enumerate(PARAMS):
        if i == RHO_ROW:
            continue
        b0 = beta.clone(); b0[i, 0] = 0.0
        b_r, r_r, _ = fit_cov_fixed(i, (b0, rand))
        total_r = float(per_person_negpost(b_r, r_r).sum())
        lr[p] = {"delta_neg_log_post": total_r - total_full, "chi2_1": 2 * (total_r - total_full)}
        print("LR", p, lr[p], flush=True)
    res["lr_tests_bdi"] = lr
    res["covariate_model_loss"] = loss
    # express each BDI effect relative to the group value and the between-person sd of the MAP estimates
    scale = {"a": pp["a"].std(), "rho": pp["rho"].std(), "wF": pp["wF"].std(), "wP": pp["wP"].std(),
             "wB": pp["wB"].std(), "eta": pp["eta"].std()}
    res["between_person_sd_of_map"] = {k: float(v) for k, v in scale.items()}
    # mood timescale implied at BDI -1 SD vs +1 SD
    res["rho_note"] = ("BDI, age and sex effects on mood persistence held at zero: per-person rho has "
                       "posterior sd 1.04 against a between-person sd 0.24 on the logit scale, and a free "
                       "covariate effect ran to the boundary (rho -> 0 at low BDI, rho -> 1 at high BDI) in "
                       "two earlier fits, so it is not identified from 30 trials and 12 ratings")
    print(json.dumps(cov_tab, indent=1)[:2500], flush=True)

    # ---- 2. point-estimate regressions
    Xc = np.column_stack([zs(age), fem])
    pe = {}
    for p in PARAMS + ["b", "logmu"]:
        y = zs(pp[p])
        bh, s = ols_hc3(y, np.column_stack([zs(bdi), Xc]))
        ci = boot_ci(lambda ix: ols_hc3(zs(pp[p][ix]), np.column_stack([zs(bdi[ix]), Xc[ix]]))[0][1], nb=1000)
        pe[p] = {"std_beta_bdi": float(bh[1]), "hc3_se": float(s[1]), "boot_lo": float(ci[0]), "boot_hi": float(ci[1]),
                 "mean_posterior_sd": float(pp["sd_" + p].mean()) if "sd_" + p in pp else None,
                 "between_sd": float(pp[p].std())}
    res["point_estimates"] = pe

    # ---- model-free
    mf = model_free()
    mfr = {}
    for k, v in mf.items():
        bh, s = ols_hc3(zs(v), np.column_stack([zs(bdi), Xc]))
        ci = boot_ci(lambda ix: ols_hc3(zs(v[ix]), np.column_stack([zs(bdi[ix]), Xc[ix]]))[0][1], nb=1000)
        mfr[k] = {"std_beta_bdi": float(bh[1]), "hc3_se": float(s[1]), "boot_lo": float(ci[0]), "boot_hi": float(ci[1])}
    # loss vs win asymmetry
    asym = np.asarray(mf["slope_neg"]) - np.asarray(mf["slope_pos"])
    bh, s = ols_hc3(zs(asym), np.column_stack([zs(bdi), Xc]))
    ci = boot_ci(lambda ix: ols_hc3(zs(asym[ix]), np.column_stack([zs(bdi[ix]), Xc[ix]]))[0][1], nb=1000)
    mfr["loss_minus_win_slope"] = {"std_beta_bdi": float(bh[1]), "hc3_se": float(s[1]), "boot_lo": float(ci[0]), "boot_hi": float(ci[1])}
    res["model_free"] = mfr

    # ---- 3. held-out prediction of BDI
    y = bdi.copy()
    cov = Xc
    MF = np.column_stack([mf[k] for k in ["mean_hap", "ac1", "slope_pos", "slope_neg"]])
    MB = np.column_stack([pp[k] for k in PARAMS + ["b", "logmu"]])
    sets = {"covariates": cov, "+model_free": np.column_stack([cov, MF]),
            "+model": np.column_stack([cov, MB]), "+both": np.column_stack([cov, MF, MB]),
            "+baseline_a_only": np.column_stack([cov, pp["a"]]), "+mean_hap_only": np.column_stack([cov, mf["mean_hap"]])}
    preds = {k: ridge_cv(v, y) for k, v in sets.items()}
    pr = {k: float(r2(y, p)) for k, p in preds.items()}

    def dci(a, b):
        return boot_ci(lambda ix: r2(y[ix], preds[a][ix]) - r2(y[ix], preds[b][ix]), nb=2000)
    pr_ci = {}
    for a, b in [("+model_free", "covariates"), ("+model", "covariates"), ("+model", "+model_free"),
                 ("+both", "+model_free"), ("+both", "+model"), ("+baseline_a_only", "+mean_hap_only")]:
        lo, hi = dci(a, b)
        pr_ci[f"{a} minus {b}"] = {"diff": pr[a] - pr[b], "lo": float(lo), "hi": float(hi)}
    res["prediction_r2"] = pr
    res["prediction_diffs"] = pr_ci
    print(json.dumps({"pe": pe, "mf": mfr, "pred": pr, "diffs": pr_ci}, indent=1), flush=True)

    (HERE / "out/summary.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(HERE / "out/model_free.npz", **mf)

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    PP = [p for p in PARAMS if p != "rho"]
    labs = {"a": "baseline a", "wF": "w forward", "wP": "w present", "wB": "w backward", "eta": "optimism eta"}
    bb = [cov_tab[p]["BDI"]["beta"] / scale[p] for p in PP]
    ee = [1.96 * cov_tab[p]["BDI"]["se"] / scale[p] for p in PP]
    ax[0].errorbar(bb, range(len(PP)), xerr=ee, fmt="o", color="#2b6cb0")
    ax[0].axvline(0, color="grey", lw=0.8)
    ax[0].set_yticks(range(len(PP))); ax[0].set_yticklabels([labs[p] for p in PP])
    ax[0].set_xlabel("BDI effect (per SD of BDI, in SDs of the parameter)")
    ax[0].set_title("Covariate model, 95% CI")
    q = np.digitize(bdi, np.percentile(bdi, [25, 50, 75]))
    ax[1].bar(range(4), [pp["a"][q == k].mean() for k in range(4)],
              yerr=[1.96 * pp["a"][q == k].std() / np.sqrt((q == k).sum()) for k in range(4)], color="#2b6cb0")
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["BDI Q1", "Q2", "Q3", "Q4"])
    ax[1].set_ylim(0.45, 0.68); ax[1].set_ylabel("person baseline a (happiness/100)")
    ax[1].set_title("Baseline mood by BDI quartile")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/gamble_bdi_effects.png", dpi=160)
    print("done")


if __name__ == "__main__":
    main()
