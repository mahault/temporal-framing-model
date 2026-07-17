"""Counterfactual vs factual learning: fair nested model comparison.

Dataset: Sugawara & Katahira (2021) complete-feedback bandit (S2021c.mat,
143 subjects) and Palminteri (2017) complete feedback (P2017b.mat, 20 subjects),
from hrl-team/decay_1. Per trial: state (pair), choice (1/2; other codes =
missed, excluded per the lab's own script), obtained outcome r, foregone
outcome c.

Models (Q per state x 2 options; softmax on Q difference; matches the lab script
Models_Complete_Final_Decay_Anneal.m):
  FACTUAL        params (alpha, beta): update only chosen option from r.
  COUNTERFACTUAL params (alpha, alpha_c, beta): also update the UNCHOSEN option
                 from the foregone outcome c, with its own rate alpha_c.
Factual is nested (alpha_c = 0), so the honest test is whether the extra
foregone-learning generalises: held-out log-likelihood and BIC. If counterfactual
wins, humans demonstrably learn from foregone outcomes -- the behavioural
signature our counterfactual-depth machinery implements.
"""
from __future__ import annotations
import numpy as np
import scipy.io as sio
from pathlib import Path
from scipy.optimize import minimize

D = Path("data_raw/hrl_decay1")
EPS = 1e-12


def load(matfile, keys):
    d = sio.loadmat(str(D / matfile))
    cho, out, cou, sta = (d[k][0] for k in keys)
    subs = []
    for i in range(len(cho)):
        c = np.asarray(cho[i], float).ravel()
        o = np.asarray(out[i], float).ravel()
        f = np.asarray(cou[i], float).ravel()
        s = np.asarray(sta[i], float).ravel()
        n = min(len(c), len(o), len(f), len(s))
        subs.append((c[:n], o[:n], f[:n], s[:n]))
    return subs


def nll(params, trials, counterfactual, eval_from):
    """Warm-start Q through the whole sequence; sum -log p(choice) for valid
    trials with index >= eval_from. Outcomes -1/+1 -> reward scale 0/1."""
    if counterfactual:
        alpha, alpha_c, beta = params
    else:
        alpha, beta = params; alpha_c = 0.0
    if not (0 <= alpha <= 1) or not (0 <= alpha_c <= 1) or beta <= 0:
        return 1e7
    cho, out, cou, sta = trials
    nstate = int(max(sta)) if len(sta) else 1
    Q = np.zeros((nstate + 1, 2))
    total = 0.0; n = 0
    for t in range(len(cho)):
        c = cho[t]
        if c not in (1.0, 2.0):
            continue
        s = int(sta[t]); ci = int(c) - 1; ui = 1 - ci
        p = 1.0 / (1.0 + np.exp(-beta * (Q[s, ci] - Q[s, ui])))
        if t >= eval_from:
            total -= np.log(p + EPS); n += 1
        r = (out[t] + 1) / 2.0; rf = (cou[t] + 1) / 2.0
        Q[s, ci] += alpha * (r - Q[s, ci])
        if counterfactual:
            Q[s, ui] += alpha_c * (rf - Q[s, ui])
    return total if n else 1e7


def fit(trials, counterfactual, eval_from):
    x0s = ([[0.2, 1.0], [0.5, 3.0]] if not counterfactual
           else [[0.2, 0.1, 1.0], [0.4, 0.3, 3.0], [0.3, 0.0, 2.0]])
    best = None
    for x0 in x0s:
        r = minimize(nll, x0, args=(trials, counterfactual, eval_from),
                     method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-3, maxiter=600))
        if best is None or r.fun < best.fun:
            best = r
    return best


def run(name, matfile, keys):
    subs = load(matfile, keys)
    ho_fac, ho_cf, bic_fac, bic_cf, cf_better, ac_list = [], [], [], [], 0, []
    for trials in subs:
        n = len(trials[0]); split = int(n * 0.6)
        fit_tr = tuple(a[:split] for a in trials)
        rf = fit(fit_tr, False, 0)
        rc = fit(fit_tr, True, 0)
        # held-out negative LL (trials >= split), warm-started through full seq
        hf = nll(rf.x, trials, False, split)
        hc = nll(rc.x, trials, True, split)
        ho_fac.append(hf); ho_cf.append(hc)
        # count valid held-out trials for per-trial normalisation
        cho, _, _, sta = trials
        nval = sum(1 for t in range(split, len(cho)) if cho[t] in (1.0, 2.0))
        if nval:
            ac_list.append((hf - hc) / nval)   # LL gain per trial (cf better if >0)
        # BIC on full data (k params: factual 2, cf 3)
        ff = fit(trials, False, 0); fc = fit(trials, True, 0)
        nfull = sum(1 for c in cho if c in (1.0, 2.0))
        bic_fac.append(2 * ff.fun + 2 * np.log(nfull))
        bic_cf.append(2 * fc.fun + 3 * np.log(nfull))
        if hc < hf:
            cf_better += 1
    ho_fac = np.array(ho_fac); ho_cf = np.array(ho_cf)
    ac = np.array(ac_list); bdiff = np.array(bic_fac) - np.array(bic_cf)
    t = ac.mean() / (ac.std(ddof=1) / np.sqrt(len(ac)))
    print(f"\n=== {name} (n={len(subs)}) ===")
    print(f"  held-out NLL: factual={ho_fac.mean():.2f}  counterfactual={ho_cf.mean():.2f} "
          f"(lower=better)")
    print(f"  held-out LL gain per trial (cf-factual): {ac.mean():+.4f}  (paired t={t:.2f})")
    print(f"  subjects better (held-out) by counterfactual: {cf_better}/{len(subs)} "
          f"= {100*cf_better/len(subs):.0f}%")
    print(f"  BIC (full data): factual={np.mean(bic_fac):.1f}  counterfactual={np.mean(bic_cf):.1f} "
          f"(lower=better; cf-favored subjects: {int((bdiff>0).sum())}/{len(subs)})")


if __name__ == "__main__":
    run("Sugawara & Katahira 2021 (complete feedback)", "S2021c.mat",
        ("choc", "outc", "couc", "stac"))
    run("Palminteri 2017 (complete feedback)", "P2017b.mat",
        ("cho", "out", "cou", "sta"))
