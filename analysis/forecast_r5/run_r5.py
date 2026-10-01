"""Round-5 model fits: candidate selection on inner validation, then held-out test (2026-10-01).

Stage "cand": for each sample and outer fold, every candidate configuration is fitted twice:
  inner  on 80% of the training participants, scored on the remaining 20% (selection only)
  outer  on all training participants, scored on the held-out fold (reported once)
Stage "abl": ablations of the selected configuration (outer fits only).

Each fit writes out/r5_<sample>_<config>_<stage>_k<k>.pkl and is skipped if that file exists, so the run
can be restarted after an interruption without losing finished fits.
Run:  python run_r5.py --stage cand --workers 10
      python run_r5.py --stage abl --base <selected config> --workers 10
"""
from __future__ import annotations

import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from common import HORIZONS, OUT, SAMPLES, folds_of, load_sample
from model_r5 import ModelR5, fit_r5

import sys
from common import ROOT
sys.path.insert(0, str(ROOT))
from model_v2 import make_tensors, subset  # noqa: E402

H6 = tuple(HORIZONS)
CANDIDATES = {
    # cumulative additions to v2.1 (v2.1 itself is scored from its saved held-out rows)
    "sp":          dict(kw=dict(g=1.0, carry=True, setpoint=True), hs=(1, 2, 3)),
    "sp_h6":       dict(kw=dict(g=1.0, carry=True, setpoint=True), hs=H6),
    "sp_h6_vol":   dict(kw=dict(g=1.0, carry=True, setpoint=True, vol=True), hs=H6),
    "lm_h6":       dict(kw=dict(g=1.0, carry=True, setpoint=True, latent_mood=True), hs=H6),
    "lm_h6_vol":   dict(kw=dict(g=1.0, carry=True, setpoint=True, latent_mood=True, vol=True), hs=H6),
    # latent mood, inert gate, channel weights shrunk toward zero, started at generic null-filter values
    "lmn_h1":      dict(kw=dict(g=0.0, carry=True, setpoint=True, latent_mood=True, null_init=True), hs=(1,), chan_l2=0.05),
    "lmn_h6":      dict(kw=dict(g=0.0, carry=True, setpoint=True, latent_mood=True, null_init=True), hs=H6, chan_l2=0.05),
}
ABLATIONS = {"inert": dict(g=0.0), "no_channels": dict(no_channels=True), "no_level2": dict(no_level2=True),
             "no_setpoint": dict(setpoint=False), "no_carry": dict(carry=False)}


def config_of(name, base=None):
    if name in CANDIDATES:
        c = CANDIDATES[name]
        return dict(c["kw"]), c["hs"], c.get("chan_l2", 0.0)
    b = CANDIDATES[base]
    kw = dict(b["kw"])
    kw.update(ABLATIONS[name.split("+", 1)[1]])
    return kw, b["hs"], b.get("chan_l2", 0.0)


def _job(a):
    sample, k, cfg, stage, base, iters = a
    f = OUT / f"r5_{sample}_{cfg}_{stage}_k{k}.pkl"
    if f.exists():
        return str(f) + " (exists)"
    torch.set_num_threads(2)
    t0 = time.time()
    parts, has_event = load_sample(sample)
    pids = sorted(parts)
    fold = folds_of(pids)
    tr = [p for p in pids if fold[p] != k]
    te = [p for p in pids if fold[p] == k]
    if stage == "inner":
        rng = np.random.RandomState(100 + k)
        sh = list(tr)
        rng.shuffle(sh)
        nv = max(len(sh) // 5, 5)
        te, tr = sorted(sh[:nv]), sorted(sh[nv:])
    kw, hs, cl2 = config_of(cfg, base)
    dt = make_tensors(parts, pids, has_event)
    itr = [pids.index(p) for p in tr]
    ite = [pids.index(p) for p in te]
    model = ModelR5(len(pids), has_event, **kw)
    loss = fit_r5(model, subset(dt, itr), itr, iters=iters, horizons=hs, chan_l2=cl2, seed=k)
    model.eval()
    dte = subset(dt, ite)
    with torch.no_grad():
        res = model(dte, torch.as_tensor(ite), H=6)
    pred, var, state = {h: {} for h in HORIZONS}, {h: {} for h in HORIZONS}, {}
    for n, p in enumerate(te):
        T = len(parts[p])
        for t in range(T):
            for h in HORIZONS:
                if dte["mt"][h][n, t] > 0.5:
                    pred[h][(p, t)] = float(res["yhat"][n, t, h - 1])
                    var[h][(p, t)] = float(res["S"][n, t, h - 1])
        if stage != "inner":
            state[p] = {kk: res[kk][n, :T].numpy() for kk in ("u", "vB", "vP", "vF", "m", "x", "b", "q")}
    params = {n_: (float(v) if v.dim() == 0 else [float(x) for x in v.flatten()][:6])
              for n_, v in model.named_parameters() if n_ != "delta"}
    with open(f, "wb") as fh:
        pickle.dump(dict(sample=sample, k=k, cfg=cfg, stage=stage, kw=kw, hs=hs, chan_l2=cl2, loss=loss,
                         pred=pred, var=var, state=state, params=params, secs=time.time() - t0), fh)
    return f"{f.name} {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("cand", "abl"), required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--only", default=None, help="comma-separated candidate names")
    args = ap.parse_args()
    jobs = []
    for sample in SAMPLES:
        for k in range(5):
            if args.stage == "cand":
                for cfg in (args.only.split(",") if args.only else CANDIDATES):
                    jobs.append((sample, k, cfg, "outer", None, args.iters))
                    jobs.append((sample, k, cfg, "inner", None, args.iters))
            else:
                for ab in ABLATIONS:
                    jobs.append((sample, k, f"{args.base}+{ab}", "outer", args.base, args.iters))
    # longest first: Geschwind and six-horizon objectives
    jobs.sort(key=lambda j: (j[0] != "Geschwind", "h6" not in j[2]))
    print(f"{len(jobs)} jobs", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for msg in ex.map(_job, jobs):
            print(msg, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
