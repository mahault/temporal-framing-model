"""Does the precision state beat a person's own variability? (written 2026-11-04, before any run)

The paper's one clear gain for the model is a better forecast variance than a filter whose noise is the same for everyone. But z_t is an
exponentially weighted average of squared errors, a GARCH-type state, and the comparison gives every person one noise level. Its gain could
therefore be just that some people's ratings vary more than others', a well known quantity, and rating-scale bounds also tie variance to the
mean (Mestdagh et al. 2018; Dejonckheere et al. 2019 found complex dynamics add little beyond mean and variance).

All point forecasts are kept fixed at the constant-noise filter's, so only the variance differs, except in z_model, the paper's model as run.
Variance rules, all using only information available at forecast time, with parameters fitted on training folds only:
  constant   the constant-noise filter's variance, rescaled
  z_model    the volatility filter with its own point forecasts, as stored
  person     running mean of the person's own squared errors so far, shrunk toward the group by a fitted n0
  bound      constant, times a fitted function of how near the forecast is to the ends of the scale
  person_bound
  person_z   person, times a recent-volatility state z^gamma, with z the same running ratio
  person_bound_z
Scored by held-out Gaussian log-likelihood at 1 and 6 beeps, with 95% participant-bootstrap intervals on paired differences.

V1 person alone recovers at least 60% of z_model's gain over constant at 1 beep in both samples.
V2 person_z beats person, interval above zero, in both samples at 1 beep, by less than .03. So recent volatility is real but smaller. Low confidence.
V3 person_bound beats person. Low confidence on size.
V4 at least half the variance of z_model's log forecast variance is between people.
V5 in Geschwind, a person's mean log volatility adds nothing to cross-validated prediction of neuroticism beyond their mean and SD of valence.

Run: python variance_competitors.py
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize

from common import OUT, folds_of, load, load_sample

SAMPLES = ("Geschwind", "osf_83cfk")
N0 = (1, 2, 5, 10, 20, 50, 100)
LAM = (.02, .05, .1, .2, .35, .5)
GAM = (.25, .5, .75, 1.0, 1.5)
N_BOOT = 2000


def rows(sample, h):
    parts, _ = load_sample(sample)
    b = load(f"baselines_{sample}.pkl")
    fold = folds_of(sorted(parts))
    tag = "kf_all_h6" if h == 6 else "kf_all_h1"
    vtag = "kfvol_all_h6" if h == 6 else "kfvol_all_h1"
    out = {}
    for (p, t), yh in b["oof"][tag][h].items():
        if (p, t) not in b["oof"][vtag][h]:
            continue
        y = parts[p][t + h]["v"]
        out.setdefault(p, []).append((t, y, yh, b["S"][tag][h][(p, t)], b["oof"][vtag][h][(p, t)], b["S"][vtag][h][(p, t)]))
    res = {}
    for p, r in out.items():
        a = np.array(sorted(r))
        res[p] = dict(t=a[:, 0], y=a[:, 1], yh=a[:, 2], S=a[:, 3], yv=a[:, 4], Sv=a[:, 5], e2=(a[:, 1] - a[:, 2]) ** 2, k=fold[p])
    return res, parts


def person_var(d, n0, h):
    """Group variance S shrunk with the person's squared errors that are already known at forecast time."""
    t, e2, S = d["t"], d["e2"], d["S"]
    v = np.zeros(len(t))
    known = t + h  # time at which each row's error is known
    for i in range(len(t)):
        m = known <= t[i]
        n = m.sum()
        v[i] = (n0 * S[i] + e2[m].sum()) / (n0 + n)
    return v


def z_state(d, base, lam, h):
    t, e2 = d["t"], d["e2"]
    known = t + h
    z = np.ones(len(t))
    cur, j = 1.0, 0
    for i in range(len(t)):
        while j < len(t) and known[j] <= t[i]:
            cur = (1 - lam) * cur + lam * min(e2[j] / base[j], 20.0)
            j += 1
        z[i] = cur
    return z


def ll(e2, v):
    return -0.5 * (np.log(2 * np.pi * v) + e2 / v)


def bound_g(yh, b):
    x = np.clip(yh, 0, 1) - .5
    return np.exp(b[0] * x + b[1] * x * x)


def fit_fold(data, h, k):
    tr = [d for d in data.values() if d["k"] != k]
    e2 = np.concatenate([d["e2"] for d in tr])
    yh = np.concatenate([d["yh"] for d in tr])
    S = np.concatenate([d["S"] for d in tr])
    P = {}
    P["a_const"] = float(np.mean(e2 / S))
    best = (-1e18, None)
    for n0 in N0:
        v = np.concatenate([person_var(d, n0, h) for d in tr])
        a = float(np.mean(e2 / v))
        s = ll(e2, a * v).sum()
        if s > best[0]:
            best = (s, n0, a)
    P["n0"], P["a_person"] = best[1], best[2]
    pv = {id(d): person_var(d, P["n0"], h) for d in tr}
    pv_all = np.concatenate([pv[id(d)] for d in tr])

    def fit_b(base):
        f = lambda b: -ll(e2, base * bound_g(yh, b) * np.mean(e2 / (base * bound_g(yh, b)))).sum()  # noqa: E731
        r = minimize(f, [0.0, 0.0], method="Nelder-Mead", options=dict(maxiter=300))
        return r.x.tolist()

    P["b_const"] = fit_b(S)
    P["a_bound"] = float(np.mean(e2 / (S * bound_g(yh, P["b_const"]))))
    P["b_person"] = fit_b(pv_all)
    P["a_pb"] = float(np.mean(e2 / (pv_all * bound_g(yh, P["b_person"]))))

    def fit_z(baser, label):
        best = (-1e18,)
        for lam in LAM:
            zz = np.concatenate([z_state(d, baser(d), lam, h) for d in tr])
            base = np.concatenate([baser(d) for d in tr])
            for g in GAM:
                v = base * zz ** g
                a = float(np.mean(e2 / v))
                s = ll(e2, a * v).sum()
                if s > best[0]:
                    best = (s, lam, g, a)
        P[f"lam_{label}"], P[f"gam_{label}"], P[f"a_{label}"] = best[1], best[2], best[3]

    fit_z(lambda d: pv[id(d)], "pz")
    fit_z(lambda d: pv[id(d)] * bound_g(d["yh"], P["b_person"]), "pbz")
    return P


def score_person(d, P, h):
    pv = person_var(d, P["n0"], h)
    gb = bound_g(d["yh"], P["b_person"])
    e2 = d["e2"]
    out = {"constant": ll(e2, P["a_const"] * d["S"]), "z_model": ll((d["y"] - d["yv"]) ** 2, d["Sv"]), "person": ll(e2, P["a_person"] * pv),
           "bound": ll(e2, P["a_bound"] * d["S"] * bound_g(d["yh"], P["b_const"])), "person_bound": ll(e2, P["a_pb"] * pv * gb)}
    z = z_state(d, pv, P["lam_pz"], h)
    out["person_z"] = ll(e2, P["a_pz"] * pv * z ** P["gam_pz"])
    b2 = pv * gb
    z2 = z_state(d, b2, P["lam_pbz"], h)
    out["person_bound_z"] = ll(e2, P["a_pbz"] * b2 * z2 ** P["gam_pbz"])
    return out


def diff(per, a, b, rng):
    pids = sorted(per)
    sa = np.array([per[p][a].sum() for p in pids])
    sb = np.array([per[p][b].sum() for p in pids])
    n = np.array([len(per[p][a]) for p in pids])
    pt = (sa - sb).sum() / n.sum()
    bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(pids), len(pids))
        bs.append((sa[i] - sb[i]).sum() / n[i].sum())
    return [round(float(pt), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def main():
    res = {}
    rng = np.random.default_rng(0)
    for s in SAMPLES:
        res[s] = {}
        for h in (1, 6):
            data, parts = rows(s, h)
            Ps = {k: fit_fold(data, h, k) for k in range(5)}
            per = {p: score_person(d, Ps[d["k"]], h) for p, d in data.items()}
            names = list(next(iter(per.values())))
            contrasts = [("z_model", "constant"), ("person", "constant"), ("bound", "constant"), ("person_bound", "person"), ("person_z", "person"), ("person_bound_z", "person_bound"), ("z_model", "person"), ("z_model", "person_bound"), ("z_model", "person_bound_z")]
            lv = np.log(np.concatenate([d["Sv"] for d in data.values()]))
            means = {p: np.log(d["Sv"]).mean() for p, d in data.items()}
            within = np.concatenate([np.log(d["Sv"]) - means[p] for p, d in data.items()])
            res[s][str(h)] = {"n_people": len(data), "n_rows": int(sum(len(d["t"]) for d in data.values())), "mean_ll": {n: round(float(np.concatenate([per[p][n] for p in per]).mean()), 4) for n in names},
                              "contrasts": {f"{a} - {b}": diff(per, a, b, rng) for a, b in contrasts}, "between_share_logvar": round(float(1 - within.var() / lv.var()), 3),
                              "params_k0": Ps[0]}
            print(s, h, json.dumps(res[s][str(h)]["contrasts"]), res[s][str(h)]["between_share_logvar"], flush=True)
    # V5: neuroticism in Geschwind
    data, parts = rows("Geschwind", 1)
    X, y = [], []
    for p, d in data.items():
        nv = parts[p][0].get("n")
        if nv is None or not np.isfinite(nv):
            continue
        v = np.array([b["v"] for b in parts[p]], float)
        X.append([v.mean(), v.std(), np.log(d["Sv"]).mean(), np.mean(np.diff(v) ** 2) ** .5])
        y.append(nv)
    X, y = np.array(X), np.array(y)

    def cvr2(cols, seed):
        r = np.random.default_rng(seed)
        idx = r.permutation(len(y))
        pred = np.zeros(len(y))
        for k in range(5):
            te = idx[k::5]
            tr = np.setdiff1d(idx, te)
            A = np.c_[np.ones(len(tr)), X[tr][:, cols]]
            w = np.linalg.lstsq(A, y[tr], rcond=None)[0]
            pred[te] = np.c_[np.ones(len(te)), X[te][:, cols]] @ w
        return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)

    sets = {"mean": [0], "mean_sd": [0, 1], "mean_sd_rmssd": [0, 1, 3], "mean_sd_vol": [0, 1, 2]}
    res["neuroticism_cv_r2"] = {k: [round(float(np.mean([cvr2(c, s) for s in range(50)])), 3)] for k, c in sets.items()}
    res["neuroticism_n"] = len(y)
    print(res["neuroticism_cv_r2"], len(y), flush=True)
    json.dump(res, open(OUT / "variance_competitors.json", "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
