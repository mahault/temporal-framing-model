"""Two figures for the restructured paper (round 8, 2026-10-01).

(a) A day in the model. One held-out Geschwind participant's day replayed with the continuous form of the
    model (configuration sp_h6_vol, the configuration ablated in the paper) using the group parameters
    fitted on the four folds that did not contain this participant. Panels: reported valence with the
    model's one-step forecast and +/- 1 predictive SD band and the reported filter's one-step forecast
    (kfx_all_h6, saved held-out predictions); the fast state x, slow mood m and set point b; the precision
    state z with the predictive SD; the frame posterior q. Events are marked on the top panel.
    The replay is checked against the saved held-out predictions of the same run (max abs difference
    printed and written to out/day_example.json).
    Day choice (deterministic): among held-out participants whose valence SD lies within the
    interquartile range of participants' SDs, the day (one sampling period and day number) with at least
    eight prompts that contains the most unpleasant reported event with the largest negative one-step
    forecast error at that prompt.

(b) Precision calibration. Held-out one-step forecasts of the reported filter (kfx_all_h6, with the
    precision state) and of the local-level filter with constant noise (kf_all_h1), both samples. Signals
    binned by the model's own predictive SD (deciles); empirical root mean squared error per bin with
    participant-bootstrap 95% bands. Coverage of central 50% and 90% Gaussian intervals at h = 1 and h = 6
    with participant-bootstrap intervals.

Run from analysis/forecast_r5:  python day_and_calibration.py
Writes out/day_example.json, out/calibration.json, ../../figures/fig_day.png, ../../figures/fig_calibration.png
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from common import HORIZONS, OUT, ROOT, SAMPLES, folds_of, load, load_sample, records
from model_r5 import ModelR5
import sys
sys.path.insert(0, str(ROOT))
from model_v2 import make_tensors, subset, _sp  # noqa: E402

FIG = ROOT / "figures"
CFG = "sp_h6_vol"
N_BOOT = 1000


# ---------------------------------------------------------------- (a) a day in the model
def replay(sample="Geschwind"):
    parts, has_event = load_sample(sample)
    pids = sorted(parts)
    fold = folds_of(pids)
    dt = make_tensors(parts, pids, has_event)
    sds = {p: float(np.std([b["v"] for b in parts[p]])) for p in pids}
    q1, q3 = np.percentile(list(sds.values()), [25, 75])
    best = None
    runs = {}
    for k in range(5):
        z = pickle.load(open(OUT / f"r5_{sample}_{CFG}_outer_k{k}.pkl", "rb"))
        model = ModelR5(len(pids), has_event, **z["kw"])
        with torch.no_grad():
            for n_, p_ in model.named_parameters():
                if n_ == "delta" or n_ not in z["params"]:
                    continue  # a_pm is unused without latent mood and was not saved
                v = z["params"][n_]
                p_.copy_(torch.tensor(v, dtype=p_.dtype).reshape(p_.shape))
        model.eval()
        te = [p for p in pids if fold[p] == k]
        ite = [pids.index(p) for p in te]
        dte = subset(dt, ite)
        with torch.no_grad():
            res = model(dte, torch.as_tensor(ite), H=6)
        lam_v = float(0.5 * torch.sigmoid(model.vol_logit))
        gv = float(_sp(model.a_gv))
        # check against saved held-out predictions
        diffs = []
        for n, p in enumerate(te):
            for t in range(len(parts[p])):
                key = (p, t)
                if key in z["pred"][1]:
                    diffs.append(abs(float(res["yhat"][n, t, 0]) - z["pred"][1][key]))
        runs[k] = dict(maxdiff=float(max(diffs)), lam_v=lam_v, gv=gv,
                       theta=float(torch.exp(model.log_theta) + 2.0))
        for n, p in enumerate(te):
            if not (q1 <= sds[p] <= q3):
                continue
            seq = parts[p]
            T = len(seq)
            # precision state z, recomputed from the run's own innovations (same update as the model)
            eps = res["eps"][n, :T].numpy()
            S1 = res["S1"][n, :T].numpy()
            zz = np.ones(T)
            zcur = 1.0
            for t in range(T):
                if t == 0 or seq[t].get("p") != seq[t - 1].get("p"):
                    if t == 0:
                        zcur = 1.0
                zcur = (1 - lam_v) * zcur + lam_v * min(eps[t] ** 2 / S1[t], 20.0)
                zz[t] = zcur
            days = defaultdict(list)
            for t, b in enumerate(seq):
                days[(b["p"], b["d"])].append(t)
            for key, ts in days.items():
                if len(ts) < 8:
                    continue
                for t in ts[1:]:
                    e = seq[t]["e"]
                    if e is None:
                        continue
                    score = (e, eps[t])  # most unpleasant event, then largest negative error
                    if best is None or score < best[0]:
                        best = (score, dict(pid=p, k=k, n=n, ts=ts, t_event=t, res=res, zz=zz, seq=seq))
    _, b = best
    p, n, ts, res, zz, seq = b["pid"], b["n"], b["ts"], b["res"], b["zz"], b["seq"]
    base = load(f"baselines_{sample}.pkl")
    kfx = base["oof"]["kfx_all_h6"][1]
    S1all = res["S1"][n].numpy()
    # one-step forecast for prompt t is made at t-1: yhat[t-1, h=1], S[t-1, h=1]
    yhat = res["yhat"][n, :, 0].numpy()
    Sh = res["S"][n, :, 0].numpy()
    rows = []
    for j, t in enumerate(ts):
        r = dict(prompt=j + 1, t=t, y=seq[t]["v"], event=seq[t]["e"],
                 x=float(res["x"][n, t]), m=float(res["m"][n, t]), b=float(res["b"][n, t]), z=float(zz[t]),
                 q=[float(v) for v in res["q"][n, t]])
        if j > 0:
            r["fc"] = float(yhat[ts[j - 1]])
            r["fc_sd"] = float(np.sqrt(Sh[ts[j - 1]]))
            r["kfx_fc"] = kfx.get((p, ts[j - 1]))
        rows.append(r)
    info = dict(sample=sample, pid_hash=abs(hash(p)) % 10000, fold=b["k"], cfg=CFG, runs=runs,
                selection="IQR of valence SD; day with >=8 prompts; most unpleasant event, then largest negative "
                          "one-step error", rows=rows)
    json.dump(info, open(OUT / "day_example.json", "w"), indent=1)

    x_ = np.arange(1, len(ts) + 1)
    fig, ax = plt.subplots(4, 1, figsize=(7.0, 8.2), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.5, 1.1, 0.9, 0.9]))
    fc = np.array([r.get("fc", np.nan) for r in rows])
    sd = np.array([r.get("fc_sd", np.nan) for r in rows])
    kf = np.array([np.nan if r.get("kfx_fc") is None else r["kfx_fc"] for r in rows])
    ax[0].fill_between(x_, fc - sd, fc + sd, color="0.85", label="model forecast $\\pm$1 SD")
    ax[0].plot(x_, fc, color="0.4", lw=1.2, label="model one-step forecast")
    ax[0].plot(x_, kf, color="tab:blue", lw=1.0, ls="--", label="reported filter forecast")
    ax[0].plot(x_, [r["y"] for r in rows], "ko-", ms=4, lw=1.0, label="reported valence")
    for r in rows:
        if r["event"] is not None and r["event"] <= -1:
            ax[0].annotate(f"event {int(r['event']):+d}", (r["prompt"], r["y"]), textcoords="offset points",
                           xytext=(0, -14), ha="center", fontsize=7, color="tab:red")
    ax[0].set_ylabel("valence (0 to 1)", fontsize=8)
    ax[0].legend(fontsize=6.5, loc="upper right", ncol=2, frameon=False)
    ax[1].plot(x_, [r["x"] for r in rows], color="tab:orange", lw=1.4, label="fast state $x_t$")
    ax[1].plot(x_, [r["m"] for r in rows], color="tab:green", lw=1.4, label="slow mood $m_t$")
    ax[1].plot(x_, [r["b"] for r in rows], color="0.5", lw=1.0, ls=":", label="set point")
    ax[1].set_ylabel("state", fontsize=8)
    ax[1].legend(fontsize=6.5, loc="lower left", ncol=3, frameon=False)
    ax[2].plot(x_, [r["z"] for r in rows], color="tab:purple", lw=1.4, label="precision state $z_t$")
    ax[2].axhline(1.0, color="0.7", lw=0.8, ls="--")
    ax[2].set_ylabel("$z_t$", fontsize=8)
    ax[2].legend(fontsize=6.5, loc="upper left", frameon=False)
    qq = np.array([r["q"] for r in rows])
    for i, (lab, col) in enumerate((("past", "tab:red"), ("present", "0.4"), ("future", "tab:blue"))):
        ax[3].plot(x_, qq[:, i], color=col, lw=1.3, label=lab)
    ax[3].set_ylim(0, 1)
    ax[3].set_ylabel("frame posterior", fontsize=8)
    ax[3].legend(fontsize=6.5, loc="upper right", ncol=3, frameon=False)
    ax[3].set_xlabel("prompt of the day", fontsize=8)
    for a in ax:
        a.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_day.png", dpi=200)
    plt.close(fig)
    print("day", info["runs"], "event prompt", [r["prompt"] for r in rows if r["t"] == b["t_event"]])
    return info


# ---------------------------------------------------------------- (b) precision calibration
def calibration():
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for ax, sample in zip(axes, SAMPLES):
        parts, has_event = load_sample(sample)
        recs = records(parts, has_event)
        base = load(f"baselines_{sample}.pkl")
        res = {}
        for tag, lab, col in (("kfx_all_h6", "with precision state", "tab:purple"),
                              ("kf_all_h1", "constant noise", "0.5")):
            res[tag] = {}
            for h in (1, 6):
                by = defaultdict(list)
                for r in recs:
                    key = (r["pid"], r["i"])
                    y = r[f"y{h}"]
                    if y is None or key not in base["oof"][tag][h] or key not in base["oof"]["kf_all_h1"][h] \
                            or key not in base["oof"]["kfx_all_h6"][h]:
                        continue
                    by[r["pid"]].append((y, base["oof"][tag][h][key], base["S"][tag][h][key]))
                by = {p: np.array(v, float) for p, v in by.items()}
                pids = sorted(by)

                def cover(sel, level):
                    a = np.concatenate([by[p] for p in sel])
                    zc = stats.norm.ppf(0.5 + level / 2)
                    return float(np.mean(np.abs(a[:, 0] - a[:, 1]) <= zc * np.sqrt(a[:, 2])))

                rng = np.random.RandomState(0)
                boots = [rng.choice(pids, len(pids), replace=True) for _ in range(N_BOOT)]
                cv = {}
                for lev in (0.5, 0.9):
                    pt = cover(pids, lev)
                    bs = [cover(s, lev) for s in boots]
                    cv[str(lev)] = (pt, *[float(v) for v in np.percentile(bs, [2.5, 97.5])])
                entry = dict(coverage=cv, n_rows=int(sum(len(v) for v in by.values())), n_people=len(pids))
                if h == 1:
                    allr = np.concatenate([by[p] for p in pids])
                    edges = np.unique(np.percentile(np.sqrt(allr[:, 2]), np.linspace(0, 100, 11)))
                    if len(edges) < 3:
                        edges = np.array([allr[:, 2].min() ** 0.5 - 1e-9, allr[:, 2].max() ** 0.5 + 1e-9])

                    def binned(sel):
                        a = np.concatenate([by[p] for p in sel])
                        s = np.sqrt(a[:, 2])
                        idx = np.clip(np.digitize(s, edges[1:-1]), 0, len(edges) - 2)
                        pred, emp = [], []
                        for j in range(len(edges) - 1):
                            m = idx == j
                            if m.sum() < 10:
                                pred.append(np.nan); emp.append(np.nan); continue
                            pred.append(float(np.sqrt(np.mean(a[m, 2]))))
                            emp.append(float(np.sqrt(np.mean((a[m, 0] - a[m, 1]) ** 2))))
                        return np.array(pred), np.array(emp)

                    pr, em = binned(pids)
                    bse = np.array([binned(s)[1] for s in boots[:300]])
                    lo, hi = np.nanpercentile(bse, [2.5, 97.5], axis=0)
                    entry.update(pred_sd=pr.tolist(), emp_rmse=em.tolist(), lo=lo.tolist(), hi=hi.tolist(),
                                 sd_range=[float(np.sqrt(allr[:, 2]).min()), float(np.sqrt(allr[:, 2]).max())])
                    ax.fill_between(pr, lo, hi, color=col, alpha=0.25, lw=0)
                    ax.plot(pr, em, "o-", color=col, ms=3.5, lw=1.2, label=lab)
                res[tag][h] = entry
        lim = [0, max(np.nanmax(res[t][1]["emp_rmse"]) for t in res) * 1.1]
        ax.plot(lim, lim, color="0.75", lw=0.8, ls="--")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title("Geschwind" if sample == "Geschwind" else "Reliability sample", fontsize=9)
        ax.set_xlabel("predicted SD of the one-step forecast", fontsize=8)
        ax.set_ylabel("observed RMS error", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        out[sample] = res
    fig.tight_layout()
    fig.savefig(FIG / "fig_calibration.png", dpi=200)
    plt.close(fig)
    json.dump(out, open(OUT / "calibration.json", "w"), indent=1)
    for s, r in out.items():
        for tag, hh in r.items():
            for h, e in hh.items():
                c5, c9 = e["coverage"]["0.5"], e["coverage"]["0.9"]
                print(f"{s:10s} {tag:12s} h={h} cov50 {c5[0]:.3f} [{c5[1]:.3f},{c5[2]:.3f}] "
                      f"cov90 {c9[0]:.3f} [{c9[1]:.3f},{c9[2]:.3f}] rows {e['n_rows']}"
                      + (f" sd range {e['sd_range'][0]:.3f}-{e['sd_range'][1]:.3f}" if h == 1 else ""))
    return out


if __name__ == "__main__":
    replay()
    calibration()
