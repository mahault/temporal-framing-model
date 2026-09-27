"""Counterfactual regret as a choice signal: nested model comparison AND
model-generated switching signature (revision 2026-09-27).

Diagnosis (analysis log): foregone outcomes drive the NEXT choice
(switch ~ foregone: t=+10.4 over 143 subjects; P(switch|regret)=0.45 vs
P(switch|relief)=0.26), but NOT via value learning. So the counterfactual
mechanism belongs as a regret/relief signal that biases policy.

Models (per subject, options 1/2, state s):
  FACTUAL   (alpha, beta):        Q-learning on the chosen option only.
  CF-VALUE  (alpha, alpha_cf, beta): Q-learning on chosen AND foregone option
             (the reviewer's "model that receives obtained and foregone
             outcomes" without the free-energy regret construction).
  REGRET    (alpha, beta, kappa): FACTUAL plus a regret bias. After a visit to
             s with obtained o and foregone f, rho = f - o is attached to the
             chosen option; on the NEXT visit to s the choice logit is biased
             away from that option by kappa * rho (regret -> switch, relief ->
             stay). This is the paper's Regret = F_actual - F_counterfactual
             wired into policy selection.
      logit(opt1 > opt2) = beta (Q[s,0] - Q[s,1]) + kappa regret_bias[s]

Comparison: held-out log-likelihood (fit first 60 percent, predict last 40),
BIC on the full sequence, and the model-generated switching signature
P(switch | regret) vs P(switch | relief), computed from each fitted model's
choice probabilities on the same trials as the human rates. Writes
reviews/regret_model_results.md and figures/fig_counterfactual_signature.png.

Run:  python regret_model_fit.py [--workers 12]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
D = ROOT / "data_raw" / "hrl_decay1"
EPS = 1e-12
MODELS = ("factual", "cfvalue", "regret")
NPAR = dict(factual=2, cfvalue=3, regret=3)


def load(matfile, keys):
    d = sio.loadmat(str(D / matfile))
    cho, out, cou, sta = (d[k][0] for k in keys)
    subs = []
    for i in range(len(cho)):
        arrs = [np.asarray(x[i], float).ravel() for x in (cho, out, cou, sta)]
        n = min(len(a) for a in arrs)
        subs.append(tuple(a[:n] for a in arrs))
    return subs


def _unpack(model, params):
    if model == "factual":
        alpha, beta = params; alpha_cf, kappa = 0.0, 0.0
    elif model == "cfvalue":
        alpha, alpha_cf, beta = params; kappa = 0.0
    else:
        alpha, beta, kappa = params; alpha_cf = 0.0
    ok = (0 <= alpha <= 1) and (0 <= alpha_cf <= 1) and beta > 0 and abs(kappa) <= 10
    return alpha, alpha_cf, beta, kappa, ok


def run_model(model, params, trials):
    """Returns per-trial P(chosen option) and the model's P(choose option 1)."""
    alpha, alpha_cf, beta, kappa, ok = _unpack(model, params)
    cho, out, cou, sta = trials
    nstate = int(max(sta)) if len(sta) else 1
    Q = np.zeros((nstate + 1, 2))
    reg_bias = np.zeros(nstate + 1)
    p_choice = np.full(len(cho), np.nan)
    p1_all = np.full(len(cho), np.nan)
    for t in range(len(cho)):
        c = cho[t]
        if c not in (1.0, 2.0):
            continue
        s = int(sta[t]); ci = int(c) - 1; ui = 1 - ci
        logit = beta * (Q[s, 0] - Q[s, 1]) + kappa * reg_bias[s]
        p1 = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
        p1_all[t] = p1
        p_choice[t] = p1 if ci == 0 else (1 - p1)
        r = (out[t] + 1) / 2.0
        Q[s, ci] += alpha * (r - Q[s, ci])
        if alpha_cf > 0:
            r_cf = (cou[t] + 1) / 2.0
            Q[s, ui] += alpha_cf * (r_cf - Q[s, ui])
        rho = cou[t] - out[t]
        reg_bias[s] = -rho if ci == 0 else rho
    return p_choice, p1_all


def nll(params, model, trials, eval_from):
    ok = _unpack(model, params)[4]
    if not ok:
        return 1e7
    p_choice, _ = run_model(model, params, trials)
    valid = ~np.isnan(p_choice)
    valid[:eval_from] = False
    if valid.sum() == 0:
        return 1e7
    return float(-np.sum(np.log(p_choice[valid] + EPS)))


def fit(model, trials, eval_from=0):
    x0s = dict(factual=[[0.3, 2.0], [0.6, 1.0]],
               cfvalue=[[0.3, 0.3, 2.0], [0.5, 0.1, 1.0], [0.2, 0.5, 3.0]],
               regret=[[0.3, 2.0, 0.3], [0.2, 1.0, 0.6], [0.5, 3.0, 0.1]])[model]
    best = None
    for x0 in x0s:
        r = minimize(nll, x0, args=(model, trials, eval_from), method="Nelder-Mead",
                     options=dict(xatol=1e-3, fatol=1e-3, maxiter=800))
        if best is None or r.fun < best.fun:
            best = r
    return best


def switching_signature(trials, p1_all=None):
    """Human (p1_all None) or model-generated switching rates on the next visit
    to the same state, stratified by the PREVIOUS obtained outcome.

    Outcomes are binary (-1/+1), so "regret" (foregone > obtained) is
    confounded with "obtained = -1". The counterfactual signature proper is
    therefore the contrast AT FIXED OBTAINED OUTCOME:
        after a loss:  P(switch | foregone = +1) vs P(switch | foregone = -1)
        after a win:   P(switch | foregone = +1) vs P(switch | foregone = -1)
    A factual learner cannot produce either difference by construction.
    Returns dict with keys 'loss_fb' (loss, foregone better), 'loss_fs'
    (loss, foregone same), 'win_fb', 'win_fs', plus the unstratified
    'regret' (foregone > obtained) and 'relief' (foregone < obtained).
    For a model the switch probability on a trial is 1 - P(previous option)."""
    cho, out, cou, sta = trials
    last = {}
    cells = dict(loss_fb=[], loss_fs=[], win_fb=[], win_fs=[], regret=[], relief=[])
    for t in range(len(cho)):
        c = cho[t]
        if c not in (1.0, 2.0):
            continue
        s = int(sta[t])
        if s in last:
            lc, lo, lf = last[s]
            if p1_all is None:
                switch = float(c != lc)
            else:
                p_prev = p1_all[t] if lc == 1.0 else 1 - p1_all[t]
                switch = 1.0 - p_prev
            if lo < 0:
                cells["loss_fb" if lf > 0 else "loss_fs"].append(switch)
            else:
                cells["win_fb" if lf > 0 else "win_fs"].append(switch)
            if lf > lo:
                cells["regret"].append(switch)
            elif lf < lo:
                cells["relief"].append(switch)
        last[s] = (c, out[t], cou[t])
    return {k: (np.mean(v) if v else np.nan) for k, v in cells.items()}


def _subject_job(trials):
    n = len(trials[0]); split = int(n * 0.6)
    ft = tuple(a[:split] for a in trials)
    cho = trials[0]
    nval = sum(1 for t in range(split, n) if cho[t] in (1.0, 2.0))
    nfull = sum(1 for c in cho if c in (1.0, 2.0))
    res = {}
    for m in MODELS:
        r_train = fit(m, ft)
        ho = nll(r_train.x, m, trials, split)
        r_full = fit(m, trials)
        bic = 2 * r_full.fun + NPAR[m] * np.log(nfull)
        _, p1 = run_model(m, r_full.x, trials)
        res[m] = dict(ho=ho, ho_per_trial=ho / max(nval, 1), bic=bic,
                      params=list(r_full.x), sw=switching_signature(trials, p1))
    res["human"] = dict(sw=switching_signature(trials))
    return res


def run(name, matfile, keys, workers):
    subs = load(matfile, keys)
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_subject_job, subs))
    else:
        results = [_subject_job(s) for s in subs]
    n = len(results)
    out = dict(name=name, n=n)
    KEYS = ("loss_fb", "loss_fs", "win_fb", "win_fs", "regret", "relief")

    def agg_sw(key_model):
        return {k: float(np.nanmean([r[key_model]["sw"][k] for r in results])) for k in KEYS}

    def paired_t(key_model, a, b):
        d = np.array([r[key_model]["sw"][a] - r[key_model]["sw"][b] for r in results])
        d = d[~np.isnan(d)]
        return float(d.mean()), float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
    out["human"] = dict(sw=agg_sw("human"),
                        loss_contrast=paired_t("human", "loss_fb", "loss_fs"),
                        win_contrast=paired_t("human", "win_fb", "win_fs"))
    for m in MODELS:
        ho = np.array([r[m]["ho_per_trial"] for r in results])
        bic = np.array([r[m]["bic"] for r in results])
        out[m] = dict(ho_per_trial=ho.mean(), bic=bic.mean(), sw=agg_sw(m),
                      loss_contrast=paired_t(m, "loss_fb", "loss_fs"),
                      win_contrast=paired_t(m, "win_fb", "win_fs"),
                      params=np.mean([r[m]["params"] for r in results], axis=0))
    # paired comparisons vs factual
    for m in ("cfvalue", "regret"):
        g = np.array([r["factual"]["ho_per_trial"] - r[m]["ho_per_trial"] for r in results])
        out[m]["ll_gain"] = g.mean()
        out[m]["ll_t"] = g.mean() / (g.std(ddof=1) / np.sqrt(n))
        out[m]["better_frac"] = float(np.mean(g > 0))
        bd = np.array([r["factual"]["bic"] - r[m]["bic"] for r in results])
        out[m]["bic_frac"] = float(np.mean(bd > 0))
    k = np.array([r["regret"]["params"][2] for r in results])
    out["regret"]["kappa_t"] = k.mean() / (k.std(ddof=1) / np.sqrt(n))
    # per-subject correlation of model-generated and human loss-stratified contrast
    d_model = np.array([r["regret"]["sw"]["loss_fb"] - r["regret"]["sw"]["loss_fs"] for r in results])
    d_human = np.array([r["human"]["sw"]["loss_fb"] - r["human"]["sw"]["loss_fs"] for r in results])
    ok = ~np.isnan(d_model) & ~np.isnan(d_human)
    out["signature_corr"] = float(np.corrcoef(d_model[ok], d_human[ok])[0, 1])
    return out


def _fmt(o):
    lines = [f"### {o['name']} (n={o['n']})", "",
             "Switching on the next visit to the same state. Unstratified regret/relief is "
             "confounded with the obtained outcome (binary outcomes), so the counterfactual "
             "signature is the contrast at fixed obtained outcome (paired t over subjects).", "",
             "| model | held-out NLL/trial | mean BIC | LL gain vs factual (t) | subjects better | BIC favours | "
             "after LOSS: foregone better / same (diff, t) | after WIN: foregone better / same (diff, t) | "
             "unstratified regret / relief |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def row(label, d, extra=("", "", "")):
        s = d["sw"]; lc = d["loss_contrast"]; wc = d["win_contrast"]
        return (f"| {label} | {extra[0]} | {extra[1]} | {extra[2]} | "
                f"{s['loss_fb']:.3f} / {s['loss_fs']:.3f} ({lc[0]:+.3f}, {lc[1]:.1f}) | "
                f"{s['win_fb']:.3f} / {s['win_fs']:.3f} ({wc[0]:+.3f}, {wc[1]:.1f}) | "
                f"{s['regret']:.3f} / {s['relief']:.3f} |")
    lines.append(row("human", o["human"]).replace("|  |  |  |", "| | | | | |", 1))
    for m in MODELS:
        d = o[m]
        gain = f"{d['ll_gain']:+.4f} ({d['ll_t']:.2f})" if "ll_gain" in d else ""
        bf = f"{d['better_frac']:.2f}" if "better_frac" in d else ""
        bic = f"{d['bic_frac']:.2f}" if "bic_frac" in d else ""
        s = d["sw"]; lc = d["loss_contrast"]; wc = d["win_contrast"]
        lines.append(f"| {m} | {d['ho_per_trial']:.4f} | {d['bic']:.1f} | {gain} | {bf} | {bic} | "
                     f"{s['loss_fb']:.3f} / {s['loss_fs']:.3f} ({lc[0]:+.3f}, {lc[1]:.1f}) | "
                     f"{s['win_fb']:.3f} / {s['win_fs']:.3f} ({wc[0]:+.3f}, {wc[1]:.1f}) | "
                     f"{s['regret']:.3f} / {s['relief']:.3f} |")
    lines.append("")
    lines.append(f"Fitted kappa t = {o['regret']['kappa_t']:.2f}; per-subject correlation of the "
                 f"model-generated and human after-loss (foregone better - same) switching "
                 f"difference r = {o['signature_corr']:.2f}.")
    lines.append("")
    return lines


def figure(o):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    GREEN, VERM = "#009E73", "#D55E00"
    groups = [("Human\n(n=%d)" % o["n"], o["human"]), ("Factual\nmodel", o["factual"]),
              ("Counterfactual-\nvalue model", o["cfvalue"]), ("Regret-bias\nmodel (ours)", o["regret"])]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    x = np.arange(len(groups)); w = 0.36
    same = [g[1]["sw"]["loss_fs"] for g in groups]; better = [g[1]["sw"]["loss_fb"] for g in groups]
    b1 = ax.bar(x - w / 2, same, w, color=GREEN, edgecolor="white",
                label="after a loss, foregone equally bad (no regret)")
    b2 = ax.bar(x + w / 2, better, w, color=VERM, edgecolor="white",
                label="after a loss, foregone better (regret)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006, f"{b.get_height():.2f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylabel("P(switch on next visit)")
    ax.set_ylim(0, max(better + same) * 1.3)
    ax.set_title("Counterfactual switching at fixed obtained outcome: data and model-generated\n"
                 "(Sugawara & Katahira 2021, complete feedback)", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
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
    o1 = run("Sugawara & Katahira 2021 (complete feedback)", "S2021c.mat",
             ("choc", "outc", "couc", "stac"), args.workers)
    o2 = run("Palminteri 2017 (complete feedback)", "P2017b.mat",
             ("cho", "out", "cou", "sta"), args.workers)
    lines = ["# Regret mechanism: nested comparison and model-generated switching (2026-09-27)",
             "", "Script: `regret_model_fit.py`. Held-out = fit on first 60 percent of trials, "
             "scored on the last 40. Switching rates: human from choices, model from fitted "
             "choice probabilities on the same trials.", ""]
    lines += _fmt(o1) + _fmt(o2)
    out = ROOT / "reviews" / "regret_model_results.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    figure(o1)
    print("saved figures/fig_counterfactual_signature.png")


if __name__ == "__main__":
    main()
