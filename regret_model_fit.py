"""Reworked counterfactual architecture: regret/relief as a choice signal.

Diagnosis (diagnostic in the analysis log): foregone outcomes drive the NEXT
choice (switch ~ foregone: t=+10.4 over 143 subjects; P(switch|regret)=0.45 vs
P(switch|relief)=0.26), but NOT via value learning (that washed out). So the
counterfactual mechanism belongs as a regret/relief signal that biases policy,
not as counterfactual value-rollout depth.

Architecture (per state s, options 1/2):
  Value:   factual Q-learning, update chosen option only (alpha).
  Regret:  after a visit to s with obtained o and foregone f, store a regret
           signal rho = (f - o) attached to the option that was chosen. On the
           NEXT visit to s, the choice logit is biased AWAY from the previously
           chosen option in proportion to kappa * rho (regret -> switch;
           relief -> stay). This is the counterfactual affective charge feeding
           policy selection -- the paper's regret = F_actual - F_counterfactual
           comparison, wired into choice.

  logit(opt1>opt2) = beta*(Q[s,0]-Q[s,1]) + kappa * regret_bias[s]

Models (nested): FACTUAL (alpha, beta) == REGRET with kappa=0.
Comparison: held-out log-likelihood (fit first 60%, predict last 40%) + BIC.
If the regret model wins, the counterfactual mechanism does real, generalising
behavioural work -- reworked so that it finally does.
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
        arrs = [np.asarray(x[i], float).ravel() for x in (cho, out, cou, sta)]
        n = min(len(a) for a in arrs)
        subs.append(tuple(a[:n] for a in arrs))
    return subs


def nll(params, trials, regret, eval_from):
    if regret:
        alpha, beta, kappa = params
    else:
        alpha, beta = params; kappa = 0.0
    if not (0 <= alpha <= 1) or beta <= 0 or abs(kappa) > 10:
        return 1e7
    cho, out, cou, sta = trials
    nstate = int(max(sta)) if len(sta) else 1
    Q = np.zeros((nstate + 1, 2))
    reg_bias = np.zeros(nstate + 1)   # signed toward option-1 preference
    total = 0.0; n = 0
    for t in range(len(cho)):
        c = cho[t]
        if c not in (1.0, 2.0):
            continue
        s = int(sta[t]); ci = int(c) - 1; ui = 1 - ci
        logit = beta * (Q[s, 0] - Q[s, 1]) + kappa * reg_bias[s]
        p1 = 1.0 / (1.0 + np.exp(-logit))           # P(choose option 1)
        p_choice = p1 if ci == 0 else (1 - p1)
        if t >= eval_from:
            total -= np.log(p_choice + EPS); n += 1
        # value update (factual)
        r = (out[t] + 1) / 2.0
        Q[s, ci] += alpha * (r - Q[s, ci])
        # regret signal: rho = foregone - obtained (on raw -1/+1 scale)
        rho = cou[t] - out[t]                        # >0 = regret, <0 = relief
        # bias AWAY from the chosen option next time: if chose opt1 (ci=0),
        # regret should reduce preference for opt1 -> negative contribution.
        reg_bias[s] = -rho if ci == 0 else rho
    return total if n else 1e7


def fit(trials, regret, eval_from):
    x0s = ([[0.3, 2.0]] if not regret else [[0.3, 2.0, 0.3], [0.2, 1.0, 0.6], [0.5, 3.0, 0.1]])
    best = None
    for x0 in x0s:
        r = minimize(nll, x0, args=(trials, regret, eval_from), method="Nelder-Mead",
                     options=dict(xatol=1e-3, fatol=1e-3, maxiter=800))
        if best is None or r.fun < best.fun:
            best = r
    return best


def run(name, matfile, keys):
    subs = load(matfile, keys)
    ho_f, ho_r, bic_f, bic_r, better, kappas, gains = [], [], [], [], 0, [], []
    for trials in subs:
        n = len(trials[0]); split = int(n * 0.6)
        ft = tuple(a[:split] for a in trials)
        rf = fit(ft, False, 0); rr = fit(ft, True, 0)
        hf = nll(rf.x, trials, False, split); hr = nll(rr.x, trials, True, split)
        ho_f.append(hf); ho_r.append(hr)
        cho = trials[0]
        nval = sum(1 for t in range(split, len(cho)) if cho[t] in (1.0, 2.0))
        if nval:
            gains.append((hf - hr) / nval)
        ff = fit(trials, False, 0); fr = fit(trials, True, 0)
        nfull = sum(1 for c in cho if c in (1.0, 2.0))
        bic_f.append(2 * ff.fun + 2 * np.log(nfull)); bic_r.append(2 * fr.fun + 3 * np.log(nfull))
        kappas.append(fr.x[2])
        if hr < hf:
            better += 1
    ho_f = np.array(ho_f); ho_r = np.array(ho_r); g = np.array(gains); k = np.array(kappas)
    bdiff = np.array(bic_f) - np.array(bic_r)
    t = g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))
    tk = k.mean() / (k.std(ddof=1) / np.sqrt(len(k)))
    print(f"\n=== {name} (n={len(subs)}) ===")
    print(f"  held-out NLL: factual={ho_f.mean():.2f}  REGRET={ho_r.mean():.2f}  (lower=better)")
    print(f"  held-out LL gain/trial (regret-factual): {g.mean():+.4f}  (paired t={t:.2f})")
    print(f"  subjects better (held-out) by regret model: {better}/{len(subs)} = {100*better/len(subs):.0f}%")
    print(f"  fitted kappa (regret weight): mean={k.mean():+.3f}  (t={tk:.2f}; >0 = regret->switch)")
    print(f"  BIC favors regret in {int((bdiff>0).sum())}/{len(subs)} subjects "
          f"(mean BIC factual={np.mean(bic_f):.1f} regret={np.mean(bic_r):.1f})")


if __name__ == "__main__":
    run("Sugawara & Katahira 2021 (complete feedback)", "S2021c.mat",
        ("choc", "outc", "couc", "stac"))
    run("Palminteri 2017 (complete feedback)", "P2017b.mat",
        ("cho", "out", "cou", "sta"))
