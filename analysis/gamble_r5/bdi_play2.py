"""Replication of the BDI covariate model on each participant's SECOND play.

Same covariate model as analysis/gamble_bdi/analyze_bdi.py (BDI, age band and sex as fixed effects
on baseline a, the three channel weights and optimism eta; person random effects on a, choice bias
and choice precision; group parameters from the fold in which the participant was held out), fitted
from scratch to the second play of every BDI participant who has one. The BDI questionnaire was
completed once, so play 2 is an independent behavioural sample against the same score.
Inference: likelihood-ratio tests (refit with the BDI effect at zero) and a participant bootstrap.

Usage: python analysis/gamble_r5/bdi_play2.py                         # full fit and LR tests
       python analysis/gamble_r5/bdi_play2.py --worker W --nworkers K --nboot 200   # bootstrap workers
The bootstrap replicates are combined by aggregate_r5.py.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis/gamble_bdi"))
sys.path.insert(0, str(ROOT / "analysis/joint_gamble"))
import analyze_bdi as ab  # noqa: E402
import joint_model as jm  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--worker", type=int, default=-1); ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    f2 = np.load(HERE / "out/gbe_second_play.npz")
    pp = dict(np.load(ROOT / "analysis/gamble_bdi/out/person_params.npz"))
    row2idx = {int(r): i for i, r in enumerate(f2["row"])}
    keep = [i for i, r in enumerate(pp["row"].astype(int)) if int(r) in row2idx]
    idx2 = np.array([row2idx[int(pp["row"][i])] for i in keep])
    keep = np.array(keep)
    # override the analysis module's globals with the play-2 sample
    ab.D = jm.Data(f2, idx2)
    ab.N = len(keep)
    ab.fold = pp["fold"][keep].astype(int)
    G = {}
    for k in ["gam", "a0", "logtau", "logsig", "rho", "k", "logq", "logmu", "loglam", "alpha", "b", "eta", "zeta"]:
        G[k] = torch.tensor([ab.groups[str(f)][k][0] for f in ab.fold])
    G["w"] = torch.stack([torch.tensor([ab.groups[str(f)]["w"][c] for f in ab.fold]) for c in range(3)], 1)
    ab.G = G
    bdi = pp["bdi"][keep]; age = pp["age"][keep]; fem = pp["female"][keep]
    X = np.column_stack([ab.zs(bdi), ab.zs(age), fem - fem.mean()])
    ab.Xt = torch.tensor(X)
    print("play-2 BDI participants:", ab.N, flush=True)
    if a.worker >= 0:   # bootstrap worker: start from the saved full fit
        fitd = torch.load(HERE / "out/bdi_play2_fit.pt"); beta, rand = fitd["beta"], fitd["rand"]
        path = HERE / f"out/boot_play2_w{a.worker}.json"
        out = json.loads(path.read_text()) if path.exists() else []
        for r in range(a.worker, a.nboot, a.nworkers):
            if any(o["rep"] == r for o in out):
                continue
            w = np.bincount(np.random.RandomState(5000 + r).randint(0, ab.N, ab.N), minlength=ab.N).astype(float)
            bb, _, _ = ab.fit_cov(iters=600, lr=0.005, mask=w, init=(beta, rand))
            out.append({"rep": r, "beta": bb.numpy().tolist()}); path.write_text(json.dumps(out))
            print("rep", r, "done", flush=True)
        print("worker done", flush=True)
        return
    beta, rand, loss = ab.fit_cov(iters=3000, lr=0.01)
    beta, rand, loss2 = ab.fit_cov(iters=1500, lr=0.003, init=(beta, rand))
    torch.save({"beta": beta, "rand": rand}, HERE / "out/bdi_play2_fit.pt")
    full = float(ab.per_person_negpost(beta, rand).sum())
    res = {"N": int(ab.N), "loss": loss2, "beta_bdi": {}, "lr": {}}
    for i, p in enumerate(ab.PARAMS):
        if i == ab.RHO_ROW:
            continue
        res["beta_bdi"][p] = float(beta[i, 0])
        b0 = beta.clone(); b0[i, 0] = 0.0
        b1, r1, _ = ab.fit_cov_fixed(i, (b0, rand))
        tot_r = float(ab.per_person_negpost(b1, r1).sum())
        res["lr"][p] = {"chi2_1": 2 * (tot_r - full)}
        print("LR", p, res["lr"][p], flush=True)
    res["beta_all"] = beta.numpy().tolist()
    (HERE / "out/bdi_play2.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
