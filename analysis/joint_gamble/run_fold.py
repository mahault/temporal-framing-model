"""Fit every model on the training participants of one fold and score the held-out participants.

Usage: python run_fold.py FOLD [--nfold 5] [--iters 500] [--threads 4]
Writes out/fold{F}.json (parameters, fit losses, pooled metrics) and out/scores_fold{F}.npz
(per-participant log-likelihood sums used for the bootstrap; not committed).
"""
from __future__ import annotations
import argparse, json, time, copy
import numpy as np
import torch
from joint_model import Data, init_params, forward, fit, fit_person_offsets, pack, T

HERE = __import__("pathlib").Path(__file__).resolve().parent
NPZ = HERE / "out/gbe_first_play.npz"

HAP = dict(use_hap=True)
MODELS = [
    # name, cfg, init_from, frozen-group
    ("H_HE", dict(readout="he", mood=False), None, None),
    ("H_HE_mood", dict(readout="he", mood=True), "H_HE", None),
    ("H_R", dict(readout="chan", mood=False), None, None),
    ("H_RM", dict(readout="chan", mood=True), "H_R", None),
    ("H_RM_eq", dict(readout="chan", mood=True, equal_w=True), "H_RM", None),
    ("H_RM_noF", dict(readout="chan", mood=True, chmask=[0.0, 1.0, 1.0]), "H_RM", None),
    ("H_RM_noP", dict(readout="chan", mood=True, chmask=[1.0, 0.0, 1.0]), "H_RM", None),
    ("H_RM_noB", dict(readout="chan", mood=True, chmask=[1.0, 1.0, 0.0]), "H_RM", None),
    ("C_EV", dict(use_hap=False, choice="ev"), None, None),
    ("C_PT", dict(use_hap=False, choice="pt"), None, None),
    ("C_PT_task", dict(use_hap=False, choice="pt", couple="task", readout="chan", mood=True), "C_PT", None),
    ("Plug", dict(readout="chan", mood=True, choice="pt", couple="filt", ekf=False, fit_hap=False), "H_RM+C_PT", "hap"),
    ("J_noEKF", dict(readout="chan", mood=True, choice="pt", couple="filt", ekf=False), "Plug", None),
    ("J", dict(readout="chan", mood=True, choice="pt", couple="filt", ekf=True), "J_noEKF", None),
    ("J_nomood", dict(readout="chan", mood=False, choice="pt", couple="filt", ekf=True), "J", None),
    ("J_noF", dict(readout="chan", mood=True, choice="pt", couple="filt", ekf=True, chmask=[0.0, 1.0, 1.0]), "J", None),
    ("J_noP", dict(readout="chan", mood=True, choice="pt", couple="filt", ekf=True, chmask=[1.0, 0.0, 1.0]), "J", None),
    ("J_noB", dict(readout="chan", mood=True, choice="pt", couple="filt", ekf=True, chmask=[1.0, 1.0, 0.0]), "J", None),
]
HAP_PARAMS = ["w", "w_eq", "gam", "a0", "logtau", "logsig", "rho", "k", "logq"]
HALF = np.arange(T) < 15          # trials 1-15 fit per-person offsets, 16-30 are held out


def clone(P):
    return {k: v.detach().clone().requires_grad_(True) for k, v in P.items()}


def openloop_r2(Dtr, otr, Dte, ote):
    """Paper-comparable task-only R2: z-scored ratings on the model's centred open-loop prediction."""
    def rows(D, out):
        m = D.hmask; y = D.zhap(); x = out["popen"].detach()
        n = m.sum(1, keepdim=True)
        xc = x - (x * m).sum(1, keepdim=True) / n
        return xc[m].numpy(), y[m].numpy()
    xtr, ytr = rows(Dtr, otr); xte, yte = rows(Dte, ote)
    A = np.c_[np.ones_like(xtr), xtr]
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    pred = coef[0] + coef[1] * xte
    return float(1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2))


def paper_r2(Dtr, Dte, which):
    """The paper's own pipeline: pooled OLS of z-scored ratings on gamma=0.6 discounted regressors."""
    def rows(D):
        X = D.chan if which == "chan" else D.he
        acc = torch.zeros(D.N, 3); feats = []
        for t in range(T):
            acc = 0.6 * acc + X[:, t, :]; feats.append(acc.clone())
        F = torch.stack(feats, 1)
        m = D.hmask
        return F[m].numpy(), D.zhap()[m].numpy()
    xtr, ytr = rows(Dtr); xte, yte = rows(Dte)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-9
    A = np.c_[np.ones(len(xtr)), (xtr - mu) / sd]
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    pred = np.c_[np.ones(len(xte)), (xte - mu) / sd] @ coef
    return float(1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2))


def person_scores(D, out):
    m = D.hmask.float()
    hll = out["hll"].detach(); cll = out["cll"].detach()
    h2 = torch.tensor(~HALF, dtype=torch.float64)
    pf = out["pfilt"].detach()
    return {
        "hll": (hll.sum(1) + out["hll_pre"].detach()).numpy(), "nh": (m.sum(1) + 1).numpy(),
        "hll2": (hll * h2).sum(1).numpy(), "nh2": (m * h2).sum(1).numpy(),
        "cll": cll.sum(1).numpy(), "nc": np.full(D.N, T, float),
        "cll2": (cll * h2).sum(1).numpy(), "nc2": np.full(D.N, float((~HALF).sum())),
        "sse_f": (((D.hap - pf) ** 2) * m).sum(1).numpy(),
        "hsum": (D.hap * m).sum(1).numpy(), "hsq": ((D.hap ** 2) * m).sum(1).numpy(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fold", type=int); ap.add_argument("--nfold", type=int, default=5)
    ap.add_argument("--iters", type=int, default=500); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--nmax", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    npz = np.load(NPZ)
    N = len(npz["cert"])
    perm = np.random.RandomState(0).permutation(N)
    if a.nmax:
        perm = perm[: a.nmax]
    folds = np.array_split(perm, a.nfold)
    te = np.sort(folds[a.fold]); tr = np.sort(np.concatenate([f for i, f in enumerate(folds) if i != a.fold]))
    Dtr, Dte = Data(npz, tr), Data(npz, te)
    print(f"fold {a.fold}: train {Dtr.N}, test {Dte.N}", flush=True)
    res = {"fold": a.fold, "ntrain": Dtr.N, "ntest": Dte.N, "models": {}}
    res["paper_r2"] = {"he": paper_r2(Dtr, Dte, "he"), "chan": paper_r2(Dtr, Dte, "chan")}
    print("paper pipeline R2", res["paper_r2"], flush=True)
    fitted = {}; scores = {}
    for name, cfg, init_from, frozen in MODELS:
        t0 = time.time()
        if init_from is None:
            P0 = init_params()
        elif init_from == "H_RM+C_PT":
            P0 = clone(fitted["H_RM"])
            for k in ["logmu", "loglam", "alpha", "b"]:
                P0[k] = fitted["C_PT"][k].detach().clone().requires_grad_(True)
        else:
            P0 = clone(fitted[init_from])
        cfg = dict(cfg)
        if frozen == "hap":
            cfg["frozen"] = HAP_PARAMS
        P, loss = fit(Dtr, cfg, P0, iters=a.iters, lr=0.03)
        # convergence check: 50 more iterations
        P, loss2 = fit(Dtr, cfg, P, iters=50, lr=0.01)
        fitted[name] = P
        otr = forward(Dtr, cfg, {k: v.detach() for k, v in P.items()})
        ote = forward(Dte, cfg, {k: v.detach() for k, v in P.items()})
        r = {"cfg": {k: v for k, v in cfg.items() if k != "frozen"}, "train_loss": loss, "train_loss_extra50": loss2,
             "params": pack(P), "seconds": time.time() - t0}
        if cfg.get("use_hap", True):
            r["openloop_r2"] = openloop_r2(Dtr, otr, Dte, ote)
        scores[name] = person_scores(Dte, ote)
        s = scores[name]
        r["test_hll_per_rating"] = float(s["hll"].sum() / s["nh"].sum()) if cfg.get("use_hap", True) else None
        r["test_cll_per_choice"] = float(s["cll"].sum() / s["nc"].sum()) if cfg.get("choice") else None
        if cfg.get("use_hap", True):
            sst = s["hsq"].sum() - s["hsum"].sum() ** 2 / (s["nh"].sum() - Dte.N)
            r["test_filtered_r2_raw"] = float(1 - s["sse_f"].sum() / sst)
        res["models"][name] = r
        print(f"  {name:10s} loss {loss:.5f}->{loss2:.5f} hll {r['test_hll_per_rating']} "
              f"cll {r['test_cll_per_choice']} r2open {r.get('openloop_r2')} ({r['seconds']:.0f}s)", flush=True)

    # ---- held-out trials: per-person offsets fitted on trials 1-15 of the test participants
    # prior sds chosen on a train subsample by split-half (empirical Bayes by cross-validation)
    rs = np.random.RandomState(a.fold)
    sub = np.sort(rs.choice(tr, size=min(4000, len(tr)), replace=False))
    Dsub = Data(npz, sub)
    grid_b = [(0.5, 0.25), (1.0, 0.5), (2.0, 1.0)]
    best = None
    cfgC = dict(MODELS[9][1])
    for sb, sm in grid_b:
        dper = fit_person_offsets(Dsub, cfgC, fitted["C_PT"], {"b", "logmu"}, {"b": sb, "logmu": sm}, HALF)
        o = forward(Dsub, cfgC, {k: v.detach() for k, v in fitted["C_PT"].items()}, dper)
        ll2 = float((o["cll"] * torch.tensor(~HALF)).sum() / (Dsub.N * (~HALF).sum()))
        if best is None or ll2 > best[0]:
            best = (ll2, sb, sm)
    res["choice_offset_prior"] = {"sd_b": best[1], "sd_logmu": best[2], "subsample_ll2": best[0]}
    grid_w = [0.01, 0.03, 0.1]
    bestw = None
    cfgH = dict(MODELS[3][1])
    for sw in grid_w:
        dper = fit_person_offsets(Dsub, cfgH, fitted["H_RM"], {"w"}, {"w": sw}, HALF)
        o = forward(Dsub, cfgH, {k: v.detach() for k, v in fitted["H_RM"].items()}, dper)
        ll2 = float((o["hll"] * torch.tensor(~HALF)).sum() / (Dsub.hmask.float() * torch.tensor(~HALF)).sum())
        if bestw is None or ll2 > bestw[0]:
            bestw = (ll2, sw)
    o0 = forward(Dsub, cfgH, {k: v.detach() for k, v in fitted["H_RM"].items()})
    res["weight_offset_prior"] = {"sd_w": bestw[1], "subsample_ll2": bestw[0],
                                  "subsample_ll2_group_weights": float((o0["hll"] * torch.tensor(~HALF)).sum() / (Dsub.hmask.float() * torch.tensor(~HALF)).sum())}
    print("offset priors", res["choice_offset_prior"], res["weight_offset_prior"], flush=True)

    ho = {}
    for name in ["C_EV", "C_PT", "C_PT_task", "Plug", "J_noEKF", "J", "J_nomood"]:
        cfg = dict([m for m in MODELS if m[0] == name][0][1])
        cfg.pop("frozen", None)
        dper = fit_person_offsets(Dte, cfg, fitted[name], {"b", "logmu"},
                                  {"b": best[1], "logmu": best[2]}, HALF)
        o = forward(Dte, cfg, {k: v.detach() for k, v in fitted[name].items()}, dper)
        ho[name] = (o["cll"].detach() * torch.tensor(~HALF)).sum(1).numpy()
    for name in ["H_RM", "H_RM_eq"]:
        cfg = dict([m for m in MODELS if m[0] == name][0][1])
        dper = fit_person_offsets(Dte, cfg, fitted[name], {"w"}, {"w": bestw[1]}, HALF)
        o = forward(Dte, cfg, {k: v.detach() for k, v in fitted[name].items()}, dper)
        ho[name + "_pw"] = (o["hll"].detach() * torch.tensor(~HALF)).sum(1).numpy()
    # between-person weight distribution: MAP on all of each test participant's ratings
    cfg = dict(MODELS[3][1])
    dall = fit_person_offsets(Dte, cfg, fitted["H_RM"], {"w"}, {"w": bestw[1]}, np.ones(T, bool))
    wi = (fitted["H_RM"]["w"].detach() + dall["w"]).numpy()
    res["person_weights"] = {
        "quantiles_5_25_50_75_95": np.percentile(wi, [5, 25, 50, 75, 95], axis=0).T.tolist(),
        "mean": wi.mean(0).tolist(), "sd": wi.std(0).tolist(),
        "corr": np.corrcoef(wi.T).tolist(),
        "frac_positive": (wi > 0).mean(0).tolist(),
    }
    np.save(HERE / f"out/person_weights_fold{a.fold}.npy", wi)
    flat = {f"{m}__{k}": v for m, s in scores.items() for k, v in s.items()}
    flat.update({f"HO__{k}": v for k, v in ho.items()})
    np.savez_compressed(HERE / f"out/scores_fold{a.fold}.npz", **flat)
    (HERE / f"out/fold{a.fold}.json").write_text(json.dumps(res, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
