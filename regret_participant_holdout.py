"""Counterfactual regret as a choice signal, participant-level holdout (round 2,
2026-09-27).

The earlier comparison (regret_model_fit.py) fitted each participant on the
first 60 percent of trials and scored the last 40 percent, which is a temporal
split, not participant holdout. Here the models are fitted at the group level
on training participants and scored on held-out participants (5-fold CV), so
the comparison matches the "participants held out throughout" protocol of the
ESM analyses.

Models (per state s, options 1 and 2, binary outcomes -1 / +1):
  factual       Q-learning on the chosen option (alpha, beta)
  cfvalue       Q-learning on chosen and foregone options (alpha, alpha_cf, beta)
  regret        factual plus a regret bias: after a visit with obtained o and
                foregone f, rho = f - o is attached to the chosen option and the
                next choice in s is biased away from it by kappa * rho
                (alpha, beta, kappa). Under binary outcomes and a fixed outcome
                preference, Eq. 1 of the paper, F_actual - min F_cf, reduces to
                a constant times (f - o), so this is Eq. 1 wired into policy
                selection.
  regret_frame  the regret bias scaled by the horizon precision of the PAST
                frame, w_past = 3 q(PAST), where q(PAST) is the posterior of
                the paper's affect model driven by the trial outcomes
                (alpha, beta, kappa). This is the frame-gated retrospective
                precision variant.
  regret_sal    the regret bias scaled by a decaying counterfactual-salience
                trace, q <- s q + (1 - s) 1[f != o], multiplier 3 q
                (alpha, beta, kappa, s).

Outputs held-out NLL per trial with participant-bootstrap CIs on the difference
from the factual learner, group-level parameters per fold, and the model-
generated after-loss switching contrast (foregone better minus foregone same)
simulated on the held-out participants' trial sequences with the fold's fitted
parameters, mean and 95 percent CI over participants at the human trial count.
Writes reviews/regret_participant_holdout.md and .json, and
figures/fig_counterfactual_signature.png.
Run:  python regret_participant_holdout.py --workers 22
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from regret_model_fit import load
from agent import Agent
from generative_model import N_FRAMES, build_model
from empirical_rebuild import _bin_v

ROOT = Path(__file__).resolve().parent
EPS = 1e-12
MODELS = ("factual", "cfvalue", "regret", "regret_frame", "regret_sal")
NPAR = dict(factual=2, cfvalue=3, regret=3, regret_frame=3, regret_sal=4)
N_SIM = 20
N_BOOT = 2000
DATASETS = (("Sugawara & Katahira 2021", "S2021c.mat", ("choc", "outc", "couc", "stac")),
            ("Palminteri et al. 2017", "P2017b.mat", ("cho", "out", "cou", "sta")))


def stack(subs):
    """Subjects -> arrays (n, T); invalid choices masked."""
    T = min(len(s[0]) for s in subs)
    cho = np.array([s[0][:T] for s in subs])
    out = np.array([s[1][:T] for s in subs])
    cou = np.array([s[2][:T] for s in subs])
    sta = np.array([s[3][:T] for s in subs]).astype(int) - 1
    valid = np.isin(cho, (1.0, 2.0))
    ci = np.where(valid, cho - 1, 0).astype(int)
    return dict(ci=ci, out=out, cou=cou, sta=sta, valid=valid, n=len(subs), T=T)


def frame_weights(d, workers=1):
    """w_past[t] = 3 q(PAST) of the affect model after the outcomes of trials
    < t (1 before the first outcome)."""
    n, T = d["n"], d["T"]
    W = np.ones((n, T))
    K = M_ = 8
    for i in range(n):
        model = build_model(K=K, M=M_, pi_pos=2.0, omega_e=5.0, gamma=16.0,
                            valence_inertia=0.5)
        ag = Agent(model, gamma=16.0, pi_pos=2.0, omega_e=5.0, valence_inertia=0.5,
                   counterfactual_horizon=1, frame_gain=1.0, seed=i)
        for t in range(T):
            if not d["valid"][i, t]:
                W[i, t] = W[i, t - 1] if t > 0 else 1.0
                continue
            o = d["out"][i, t]
            _, info = ag.step([0 if o < 0 else 2, 1, _bin_v(0.25 if o < 0 else 0.75, K)])
            qf = info["beliefs"].reshape(K, M_, N_FRAMES).sum(axis=(0, 1))
            if t + 1 < T:
                W[i, t + 1] = 3.0 * qf[0]
    return W


def unpack(model, params):
    alpha_cf, kappa, s = 0.0, 0.0, 0.0
    if model == "factual":
        alpha, beta = params
    elif model == "cfvalue":
        alpha, alpha_cf, beta = params
    elif model in ("regret", "regret_frame"):
        alpha, beta, kappa = params
    else:
        alpha, beta, kappa, s = params
    ok = (0 <= alpha <= 1) and (0 <= alpha_cf <= 1) and (0 < beta <= 50) \
        and (abs(kappa) <= 10) and (0 <= s <= 1)
    return alpha, alpha_cf, beta, kappa, s, ok


def run(model, params, d, W=None, simulate=None):
    """Vectorised over subjects. Returns P(chosen) per trial (n, T) for the human
    choices, or, with simulate=rng, simulated choices and their obtained /
    foregone outcomes."""
    alpha, alpha_cf, beta, kappa, s, ok = unpack(model, params)
    n, T = d["n"], d["T"]
    idx = np.arange(n)
    Q = np.zeros((n, 8, 2))
    reg = np.zeros((n, 8))
    sal = np.full(n, 1.0 / 3.0)
    p_choice = np.full((n, T), np.nan)
    sim_c = np.zeros((n, T), int)
    sim_o = np.zeros((n, T))
    sim_f = np.zeros((n, T))
    for t in range(T):
        st = d["sta"][:, t]
        v = d["valid"][:, t]
        q0, q1 = Q[idx, st, 0], Q[idx, st, 1]
        if model == "regret_frame":
            mult = W[:, t]
        elif model == "regret_sal":
            mult = 3.0 * sal
        else:
            mult = 1.0
        logit = beta * (q0 - q1) + kappa * mult * reg[idx, st]
        p1 = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
        if simulate is None:
            ci = d["ci"][:, t]
            o = d["out"][:, t]
            f = d["cou"][:, t]
        else:
            ci = (simulate.random_sample(n) >= p1).astype(int)   # 0 -> option 1
            same = ci == d["ci"][:, t]
            o = np.where(same, d["out"][:, t], d["cou"][:, t])
            f = np.where(same, d["cou"][:, t], d["out"][:, t])
            sim_c[:, t] = ci
            sim_o[:, t] = o
            sim_f[:, t] = f
        p_choice[:, t] = np.where(v, np.where(ci == 0, p1, 1 - p1), np.nan)
        r = (o + 1) / 2.0
        dq = alpha * (r - Q[idx, st, ci])
        Q[idx, st, ci] = np.where(v, Q[idx, st, ci] + dq, Q[idx, st, ci])
        if alpha_cf > 0:
            rf = (f + 1) / 2.0
            ui = 1 - ci
            dq = alpha_cf * (rf - Q[idx, st, ui])
            Q[idx, st, ui] = np.where(v, Q[idx, st, ui] + dq, Q[idx, st, ui])
        rho = f - o
        reg[idx, st] = np.where(v, np.where(ci == 0, -rho, rho), reg[idx, st])
        if model == "regret_sal":
            sal = np.where(v, s * sal + (1 - s) * (rho != 0), sal)
    if simulate is None:
        return p_choice
    return sim_c, sim_o, sim_f


def nll_group(params, model, d, W, subset):
    ok = unpack(model, params)[5]
    if not ok:
        return 1e9
    p = run(model, params, d, W)[subset]
    return float(-np.nansum(np.log(p + EPS)))


def fit_group(model, d, W, subset):
    x0s = dict(factual=[[0.3, 2.0], [0.6, 1.0], [0.15, 4.0]],
               cfvalue=[[0.3, 0.3, 2.0], [0.5, 0.1, 1.0], [0.2, 0.5, 3.0]],
               regret=[[0.3, 2.0, 0.3], [0.2, 1.0, 0.6], [0.5, 3.0, 0.1]],
               regret_frame=[[0.3, 2.0, 0.3], [0.2, 1.0, 0.6], [0.5, 3.0, 0.1]],
               regret_sal=[[0.3, 2.0, 0.3, 0.5], [0.2, 1.0, 0.6, 0.8], [0.5, 3.0, 0.3, 0.2]])[model]
    best = None
    for x0 in x0s:
        r = minimize(nll_group, x0, args=(model, d, W, subset), method="Nelder-Mead",
                     options=dict(xatol=1e-4, fatol=1e-4, maxiter=2000))
        if best is None or r.fun < best.fun:
            best = r
    return best


def contrast(ci, o, f, sta, valid):
    """After-loss switching contrast per subject: P(switch | foregone better) -
    P(switch | foregone same), next visit to the same state."""
    n, T = ci.shape
    out = np.full(n, np.nan)
    for i in range(n):
        last = {}
        fb, fs = [], []
        for t in range(T):
            if not valid[i, t]:
                continue
            s = sta[i, t]
            if s in last:
                lc, lo, lf = last[s]
                if lo < 0:
                    (fb if lf > 0 else fs).append(float(ci[i, t] != lc))
            last[s] = (ci[i, t], o[i, t], f[i, t])
        if fb and fs:
            out[i] = np.mean(fb) - np.mean(fs)
    return out


def boot_ci(x, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    rng = np.random.RandomState(seed)
    b = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(N_BOOT)])
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _fold_job(a):
    model, d, W, tr, te = a
    r = fit_group(model, d, W, tr)
    p = run(model, r.x, d, W)
    nll_te = -np.nanmean(np.log(p[te] + EPS), axis=1)
    rng = np.random.RandomState(7)
    sims = []
    for _ in range(N_SIM):
        c, o, f = run(model, r.x, d, W, simulate=rng)
        sims.append(contrast(c[te], o[te], f[te], d["sta"][te], d["valid"][te]))
    return model, te, list(r.x), nll_te, np.nanmean(np.array(sims), axis=0)


def analyse(name, matfile, keys, workers):
    subs = load(matfile, keys)
    d = stack(subs)
    W = frame_weights(d)
    n = d["n"]
    rng = np.random.RandomState(0)
    order = np.arange(n)
    rng.shuffle(order)
    folds = [order[k::5] for k in range(5)]
    jobs = []
    for k in range(5):
        te = np.sort(folds[k])
        tr = np.sort(np.concatenate([folds[j] for j in range(5) if j != k]))
        for m in MODELS:
            jobs.append((m, d, W, tr, te))
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_fold_job, jobs))
    else:
        res = [_fold_job(j) for j in jobs]
    nll = {m: np.full(n, np.nan) for m in MODELS}
    gen = {m: np.full(n, np.nan) for m in MODELS}
    params = {m: [] for m in MODELS}
    for m, te, x, nll_te, g in res:
        nll[m][te] = nll_te
        gen[m][te] = g
        params[m].append(x)
    human = contrast(d["ci"], d["out"], d["cou"], d["sta"], d["valid"])
    out = dict(name=name, n=n, T=d["T"], human=boot_ci(human), models={})
    for m in MODELS:
        diff = nll[m] - nll["factual"]
        out["models"][m] = dict(
            nll=boot_ci(nll[m]), diff_vs_factual=boot_ci(diff),
            better_frac=float(np.mean(diff < 0)),
            generated=boot_ci(gen[m]),
            params_per_fold=params[m],
            params_mean=[float(v) for v in np.mean(params[m], axis=0)])
    out["frame_weight_stats"] = dict(mean=float(W.mean()), sd=float(W.std()),
                                     min=float(W.min()), max=float(W.max()))
    return out


def fmt(o):
    L = [f"### {o['name']} (n={o['n']}, {o['T']} trials each)", "",
         f"Human after-loss contrast: {o['human'][0]:+.3f} [{o['human'][1]:+.3f}, {o['human'][2]:+.3f}]",
         f"Affect-model w_past over trials: mean {o['frame_weight_stats']['mean']:.2f}, "
         f"SD {o['frame_weight_stats']['sd']:.2f}, range {o['frame_weight_stats']['min']:.2f} to "
         f"{o['frame_weight_stats']['max']:.2f}", "",
         "| model | held-out NLL/trial [CI] | vs factual [CI] | subjects better | generated contrast [CI] | group params (mean over folds) |",
         "|---|---:|---:|---:|---:|---|"]
    for m, r in o["models"].items():
        L.append(f"| {m} | {r['nll'][0]:.4f} [{r['nll'][1]:.4f}, {r['nll'][2]:.4f}] | "
                 f"{r['diff_vs_factual'][0]:+.4f} [{r['diff_vs_factual'][1]:+.4f}, {r['diff_vs_factual'][2]:+.4f}] | "
                 f"{r['better_frac']:.2f} | {r['generated'][0]:+.3f} [{r['generated'][1]:+.3f}, {r['generated'][2]:+.3f}] | "
                 f"{', '.join(f'{v:.3f}' for v in r['params_mean'])} |")
    L.append("")
    return L


def figure(o):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Human"] + [dict(factual="Factual", cfvalue="Counterfactual\nvalue", regret="Regret bias",
                               regret_frame="Regret bias,\nframe-gated", regret_sal="Regret bias,\nsalience-gated")[m]
                          for m in MODELS]
    vals = [o["human"]] + [o["models"][m]["generated"] for m in MODELS]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    x = np.arange(len(vals))
    m = [v[0] for v in vals]
    lo = [v[0] - v[1] for v in vals]
    hi = [v[2] - v[0] for v in vals]
    colors = ["#333333"] + ["#0072B2"] * len(MODELS)
    ax.bar(x, m, yerr=[lo, hi], color=colors, capsize=3, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("after-loss switching contrast\n(foregone better minus foregone same)")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title(f"{o['name']}: data and model-generated contrasts on held-out participants "
                 f"(95% CIs over participants)", fontsize=9.5, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "fig_counterfactual_signature.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    lines = ["# Regret mechanism, participant-level holdout (2026-09-27)", "",
             "Script: `regret_participant_holdout.py`. Group-level parameters fitted on training "
             "participants, held-out NLL per trial on test participants, 5 folds. Generated contrasts: "
             f"{N_SIM} simulated runs per held-out participant on their own trial sequence with the "
             "fold's parameters; CIs are participant bootstraps (2000 resamples).", ""]
    results = []
    for name, mf, keys in DATASETS:
        o = analyse(name, mf, keys, args.workers)
        results.append(o)
        lines += fmt(o)
        print("\n".join(fmt(o)))
    (ROOT / "reviews").mkdir(exist_ok=True)
    (ROOT / "reviews" / "regret_participant_holdout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "reviews" / "regret_participant_holdout.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    figure(results[0])


if __name__ == "__main__":
    main()
