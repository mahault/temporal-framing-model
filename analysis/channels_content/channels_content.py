"""
Channels from thought content in daily life (Baumeister et al. 2020 Study 1; Bayer partial replication).
Predictions committed before analysis in reviews/CHANNELS_CONTENT.md (commit fba8116).

Usage (repo root):
    python analysis/channels_content/channels_content.py static    # mixed models, equality, ratios, carry-over
    python analysis/channels_content/channels_content.py heldout   # static held-out comparison
    python analysis/channels_content/channels_content.py dynamic   # state-space held-out comparison
    python analysis/channels_content/channels_content.py bayer     # partial replication
    python analysis/channels_content/channels_content.py report    # BH, tables, figures
Outputs in analysis/channels_content/out/ (aggregate json only; participant rows are git-ignored).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "channels_content" / "out"
OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures"
RNG = np.random.default_rng(20261001)
N_BOOT = 1000
GAMBLE = dict(P=0.097, F=0.049, B=0.017)

POS_PAST = ["typepast_3", "typepast_7", "typepast_8", "typepast_6"]           # happy proud relieved nostalgia
NEG_PAST = ["typepast_1", "typepast_4", "typepast_5", "typepast_9", "typepast_2", "typepast_11"]
POS_FUT = ["typefuture_1", "typefuture_7", "typefuture_8"]                    # planning, hope to do, hope will happen
NEG_FUT = ["typefuture_10", "typefuture_9"]                                   # worries, fear


# ───────────────────────────── data ─────────────────────────────
def load_s1() -> pd.DataFrame:
    import pyreadstat
    df, _ = pyreadstat.read_sav(str(ROOT / "data_raw/candidates/baumeister2020_ett/Study1/ETT_ESM_Study1.sav"))
    df = df[df["valence"].notna()].copy()
    df = df.sort_values(["PID", "DAY", "SIG"])
    n = df.groupby("PID")["valence"].transform("size")
    df = df[n >= 4].copy()
    df["pid"] = df["PID"].astype(int).astype(str)
    df["y"] = df["valence"].astype(float)
    of = df["onefactor"]
    df["past"] = of.isin([1, 4, 5, 7]).astype(float)
    df["pres"] = of.isin([2, 4, 6, 7]).astype(float)
    df["fut"] = of.isin([3, 5, 6, 7]).astype(float)
    df["ori_known"] = of.notna()
    df["k"] = df[["past", "pres", "fut"]].sum(axis=1)
    c = df["contentpn"].astype(float)
    df["c"] = c
    for rule in ("split", "copy"):
        div = df["k"].where(df["k"] > 0, 1.0) if rule == "split" else 1.0
        for ch, col in (("B", "past"), ("P", "pres"), ("F", "fut")):
            df[f"u{ch}_{rule}"] = (c * df[col] / div).fillna(0.0)
    df["single_or_none"] = df["k"] <= 1
    # signed content inputs (branch checklists)
    t = lambda cols: df[cols].fillna(0).eq(1).sum(axis=1).astype(float)
    df["uB_signed"] = (t(POS_PAST) - t(NEG_PAST)) * df["past"]
    df["uF_signed"] = (t(POS_FUT) - t(NEG_FUT)) * df["fut"]
    df["uP_signed"] = df["uP_split"]
    df["tod"] = (df["hour"] - 9.0) / 9.0
    df["tod"] = df["tod"].fillna(df["tod"].mean())
    # previous mood within day
    df["prev"] = df.groupby(["pid", "DAY"])["y"].shift(1)
    df["has_prev"] = df["prev"].notna().astype(float)
    pm = df.groupby("pid")["y"].transform("mean")
    df["prev_c"] = (df["prev"] - pm).fillna(0.0)
    # next mood within day (carry-over)
    df["next"] = df.groupby(["pid", "DAY"])["y"].shift(-1)
    df["y_c"] = df["y"] - pm
    # causal running person level for held-out scoring (shrunk expanding mean of earlier signals)
    g = df.groupby("pid")["y"]
    csum = g.cumsum() - df["y"]
    cnt = g.cumcount()
    df["_csum"], df["_cnt"] = csum, cnt
    return df.reset_index(drop=True)


def model_rows(df, rule="split"):
    d = df[df["c"].notna() & df["ori_known"]].copy()
    if rule == "single":
        d = d[d["single_or_none"]].copy()
        r = "split"
    else:
        r = rule
    if rule == "signed":
        r = "signed"
    for ch in "BPF":
        d[f"u{ch}"] = d[f"u{ch}_{r}"]
    return d.reset_index(drop=True)


# ───────────────────────────── static models ─────────────────────────────
FIX = ["past", "pres", "fut", "prev_c", "has_prev", "tod"]


def mixed(d, inputs, slopes=False):
    import statsmodels.formula.api as smf
    form = "y ~ " + " + ".join(inputs + FIX)
    re = ("~" + " + ".join(inputs)) if slopes else "~1"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = smf.mixedlm(form, d, groups=d["pid"], re_formula=re)
        for meth in ("bfgs", "powell", "nm"):
            try:
                m = mod.fit(reml=True, method=meth)
                if m.converged:
                    break
            except Exception:  # noqa: BLE001
                continue
    return m


def within_ols(d, inputs, yname="y"):
    """Within-person centred OLS with participant-clustered SEs (check and bootstrap engine)."""
    cols = inputs + FIX
    X = d[cols].to_numpy(float)
    y = d[yname].to_numpy(float)
    g = d["pid"].to_numpy()
    Xc = X - pd.DataFrame(X).groupby(g).transform("mean").to_numpy()
    yc = y - pd.Series(y).groupby(g).transform("mean").to_numpy()
    XtX = Xc.T @ Xc
    b = np.linalg.solve(XtX, Xc.T @ yc)
    e = yc - Xc @ b
    inv = np.linalg.inv(XtX)
    meat = np.zeros_like(XtX)
    for pid in np.unique(g):
        s = g == pid
        sc = Xc[s].T @ e[s]
        meat += np.outer(sc, sc)
    G = len(np.unique(g))
    V = inv @ meat @ inv * G / (G - 1)
    return dict(zip(cols, b)), pd.DataFrame(V, index=cols, columns=cols)


def wald_equal(b, V, names):
    """H0: all named coefficients equal."""
    from scipy import stats
    k = len(names)
    R = np.zeros((k - 1, k))
    for i in range(k - 1):
        R[i, i], R[i, i + 1] = 1, -1
    bb = np.array([b[n] for n in names])
    VV = V.loc[names, names].to_numpy()
    r = R @ bb
    W = float(r @ np.linalg.solve(R @ VV @ R.T, r))
    return W, float(stats.chi2.sf(W, k - 1))


def boot_within(d, cols, yname, n=N_BOOT):
    """Participant bootstrap of the within-person OLS: per-participant cross-products, resampled."""
    pids = d["pid"].unique()
    XtX, Xty = [], []
    for p in pids:
        g = d[d["pid"] == p]
        X = g[cols].to_numpy(float); y = g[yname].to_numpy(float)
        Xc = X - X.mean(0); yc = y - y.mean()
        XtX.append(Xc.T @ Xc); Xty.append(Xc.T @ yc)
    XtX = np.array(XtX); Xty = np.array(Xty)
    out = []
    for _ in range(n):
        i = RNG.integers(0, len(pids), len(pids))
        try:
            out.append(np.linalg.solve(XtX[i].sum(0), Xty[i].sum(0)))
        except np.linalg.LinAlgError:
            pass
    return np.array(out)


def boot_weights(d, inputs, n=N_BOOT, yname="y"):
    return boot_within(d, inputs + FIX, yname, n)[:, :len(inputs)]


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def stage_static():
    df = load_s1()
    res = {"n_participants": int(df["pid"].nunique()), "n_signals": int(len(df))}
    inputs = ["uB", "uP", "uF"]
    for rule in ("split", "copy", "single", "signed"):
        d = model_rows(df, rule)
        if rule == "signed":
            d = d.copy()
        r = {"n_rows": int(len(d)), "n_participants": int(d["pid"].nunique())}
        m = mixed(d, inputs)
        r["mixed"] = {k: [float(m.params[k]), float(m.bse[k])] for k in inputs}
        W, p = wald_equal(m.params, m.cov_params(), inputs)
        r["mixed_equal"] = [W, p]
        b, V = within_ols(d, inputs)
        r["within"] = {k: [float(b[k]), float(np.sqrt(V.loc[k, k]))] for k in inputs}
        W2, p2 = wald_equal(b, V, inputs)
        r["within_equal_cluster"] = [W2, p2]
        if rule == "split":
            # one undifferentiated pleasantness term
            d1 = d.copy().reset_index(drop=True)
            d1["uC"] = d1["uB"] + d1["uP"] + d1["uF"]
            m1 = mixed(d1, ["uC"])
            r["mixed_undiff"] = [float(m1.params["uC"]), float(m1.bse["uC"])]
            r["llf_mixed_ML"] = {}
            import statsmodels.formula.api as smf
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f3 = smf.mixedlm("y ~ " + " + ".join(inputs + FIX), d, groups=d["pid"]).fit(reml=False)
                f1 = smf.mixedlm("y ~ " + " + ".join(["uC"] + FIX), d1, groups=d1["pid"]).fit(reml=False)
            from scipy import stats
            LR = 2 * (f3.llf - f1.llf)
            r["LR_3_vs_1"] = [float(LR), float(stats.chi2.sf(LR, 2))]
            # random slopes
            try:
                ms = mixed(d, inputs, slopes=True)
                r["mixed_slopes"] = {k: [float(ms.params[k]), float(ms.bse[k])] for k in inputs}
                Ws, ps = wald_equal(ms.params, ms.cov_params(), inputs)
                r["mixed_slopes_equal"] = [Ws, ps]
                r["mixed_slopes_converged"] = bool(ms.converged)
            except Exception as ex:  # noqa: BLE001
                r["mixed_slopes"] = f"failed: {ex}"
            # bootstrap ratios and ordered differences (within estimator)
            B = boot_weights(d, inputs)
            wB, wP, wF = B[:, 0], B[:, 1], B[:, 2]
            r["boot_n"] = int(len(B))
            r["boot_weights_ci"] = {"uB": ci(wB), "uP": ci(wP), "uF": ci(wF)}
            r["ratio_F_P"] = [float(np.median(wF / wP)), *ci(wF / wP)]
            r["ratio_B_P"] = [float(np.median(wB / wP)), *ci(wB / wP)]
            r["diff_P_minus_F"] = [float(np.mean(wP - wF)), *ci(wP - wF)]
            r["diff_F_minus_B"] = [float(np.mean(wF - wB)), *ci(wF - wB)]
            r["diff_P_minus_B"] = [float(np.mean(wP - wB)), *ci(wP - wB)]
            # carry-over: next mood within day
            dc = d[d["next"].notna()].copy().reset_index(drop=True)
            dc["ycur_c"] = dc["y_c"]
            import statsmodels.formula.api as smf
            form = "next ~ uB + uP + uF + ycur_c + past + pres + fut + tod"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mc = smf.mixedlm(form, dc, groups=dc["pid"]).fit(reml=True)
            r["carry"] = {k: [float(mc.params[k]), float(mc.bse[k]), float(mc.pvalues[k])] for k in inputs + ["ycur_c"]}
            r["carry_n"] = int(len(dc))
            # F minus B carry difference, bootstrap with within estimator on next
            CB = boot_within(dc, ["uB", "uP", "uF", "ycur_c", "past", "pres", "fut", "tod"], "next")[:, :3]
            r["carry_boot_ci"] = {"uB": ci(CB[:, 0]), "uP": ci(CB[:, 1]), "uF": ci(CB[:, 2])}
            r["carry_F_minus_B"] = [float(np.mean(CB[:, 2] - CB[:, 0])), *ci(CB[:, 2] - CB[:, 0])]
            # descriptive: correlation of inputs with mood
            r["corr_inputs_mood"] = {k: float(np.corrcoef(d[k], d["y"])[0, 1]) for k in inputs}
            r["n_by_orientation"] = {k: int((d[k] != 0).sum()) for k in inputs}
        res[rule] = r
        print(rule, json.dumps({k: r[k] for k in ("mixed", "mixed_equal")}, indent=None))
    json.dump(res, open(OUT / "static.json", "w"), indent=1)


# ───────────────────────────── held-out static ─────────────────────────────
def causal_features(d, prior_k=3.0, grand=None):
    d = d.copy()
    grand = d["y"].mean() if grand is None else grand
    d["lvl"] = (d["_csum"] + prior_k * grand) / (d["_cnt"] + prior_k)
    d["prev_dev"] = (d["prev"] - d["lvl"]).fillna(0.0)
    return d


def stage_heldout():
    df = load_s1()
    d = model_rows(df, "split")
    d["uC"] = d["uB"] + d["uP"] + d["uF"]
    pids = np.array(sorted(d["pid"].unique()))
    perm = RNG.permutation(len(pids))
    folds = np.array_split(pids[perm], 5)
    base = ["lvl", "prev_dev", "has_prev", "tod", "past", "pres", "fut"]
    specs = {"three": ["uB", "uP", "uF"], "one": ["uC"], "none": []}
    ll = {k: {} for k in specs}
    for f in folds:
        test = d[d["pid"].isin(f)]
        train = d[~d["pid"].isin(f)]
        grand = train["y"].mean()
        tr = causal_features(train, grand=grand)
        te = causal_features(test, grand=grand)
        for name, inp in specs.items():
            cols = base + inp
            X = np.c_[np.ones(len(tr)), tr[cols].to_numpy(float)]
            b = np.linalg.lstsq(X, tr["y"].to_numpy(float), rcond=None)[0]
            s2 = float(np.mean((tr["y"].to_numpy() - X @ b) ** 2))
            Xt = np.c_[np.ones(len(te)), te[cols].to_numpy(float)]
            e = te["y"].to_numpy() - Xt @ b
            lls = -0.5 * (np.log(2 * np.pi * s2) + e ** 2 / s2)
            for p, v in zip(te["pid"].to_numpy(), lls):
                ll[name].setdefault(p, []).append(float(v))
    res = {}
    P = sorted(ll["three"])
    for a, bname in (("three", "one"), ("three", "none"), ("one", "none")):
        sa = np.array([np.sum(ll[a][p]) for p in P]); sb = np.array([np.sum(ll[bname][p]) for p in P])
        nn = np.array([len(ll[a][p]) for p in P])
        diff = (sa.sum() - sb.sum()) / nn.sum()
        boots = []
        for _ in range(N_BOOT):
            i = RNG.integers(0, len(P), len(P))
            boots.append((sa[i].sum() - sb[i].sum()) / nn[i].sum())
        res[f"{a}_minus_{bname}"] = [float(diff), *ci(boots)]
    res["n_participants"] = len(P); res["n_signals"] = int(sum(len(v) for v in ll["three"].values()))
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / "heldout_static.json", "w"), indent=1)


# ───────────────────────────── dynamic state space ─────────────────────────────
def pad(d, inputs):
    """Arrays (P, T) for vectorised Kalman: y, inputs, indicators, tod, day-start flag, mask."""
    groups = [g for _, g in d.groupby("pid", sort=True)]
    P = len(groups); T = max(len(g) for g in groups)
    Y = np.zeros((P, T)); M = np.zeros((P, T), bool); DS = np.zeros((P, T), bool)
    U = np.zeros((len(inputs), P, T)); IND = np.zeros((3, P, T)); TOD = np.zeros((P, T))
    for i, g in enumerate(groups):
        n = len(g)
        Y[i, :n] = g["y"].to_numpy(); M[i, :n] = True
        day = g["DAY"].to_numpy()
        DS[i, :n] = np.r_[True, day[1:] != day[:-1]]
        for j, c in enumerate(inputs):
            U[j, i, :n] = g[c].to_numpy()
        for j, c in enumerate(("past", "pres", "fut")):
            IND[j, i, :n] = g[c].to_numpy()
        TOD[i, :n] = g["tod"].to_numpy()
    return dict(Y=Y, M=M, DS=DS, U=U, IND=IND, TOD=TOD, pids=[g["pid"].iloc[0] for g in groups])


def kalman_ll(theta, A, n_in, per_signal=False):
    """Two-state model: y = L + f + b.ind + c*tod + e; L random walk (q_L), f = phi f + g.u + eta,
    f reset to its stationary distribution at day start. Returns total log-lik (or per-signal array)."""
    i = 0
    phi = np.tanh(theta[i]); i += 1
    g = theta[i:i + n_in]; i += n_in
    bind = theta[i:i + 3]; i += 3
    ctod = theta[i]; i += 1
    mu0 = theta[i]; i += 1
    sd0 = np.exp(theta[i]); i += 1
    qL = np.exp(theta[i]); i += 1
    sf = np.exp(theta[i]); i += 1
    se = np.exp(theta[i]); i += 1
    Y, M, DS, U, IND, TOD = A["Y"], A["M"], A["DS"], A["U"], A["IND"], A["TOD"]
    P, T = Y.shape
    mL = np.full(P, mu0); vL = np.full(P, sd0 ** 2)
    mf = np.zeros(P); vf = np.full(P, sf ** 2 / max(1e-6, 1 - phi ** 2)); cLf = np.zeros(P)
    vstat = sf ** 2 / max(1e-6, 1 - phi ** 2)
    out = np.zeros((P, T))
    first = np.ones(P, bool)
    for t in range(T):
        m = M[:, t]
        drive = np.tensordot(g, U[:, :, t], axes=1)
        # predict
        reset = DS[:, t] & ~first
        mL_p = mL; vL_p = vL + np.where(first, 0.0, qL ** 2)
        mf_p = np.where(reset | first, drive, phi * mf + drive)
        vf_p = np.where(reset | first, vstat, phi ** 2 * vf + sf ** 2)
        c_p = np.where(reset | first, 0.0, phi * cLf)
        mean = mL_p + mf_p + np.tensordot(bind, IND[:, :, t], axes=1) + ctod * TOD[:, t]
        S = vL_p + vf_p + 2 * c_p + se ** 2
        e = Y[:, t] - mean
        ll = -0.5 * (np.log(2 * np.pi * S) + e ** 2 / S)
        out[:, t] = np.where(m, ll, 0.0)
        KL = (vL_p + c_p) / S; Kf = (vf_p + c_p) / S
        mL_n = mL_p + KL * e; mf_n = mf_p + Kf * e
        vL_n = vL_p - KL * (vL_p + c_p); vf_n = vf_p - Kf * (vf_p + c_p); c_n = c_p - KL * (vf_p + c_p)
        mL = np.where(m, mL_n, mL); mf = np.where(m, mf_n, mf)
        vL = np.where(m, vL_n, vL); vf = np.where(m, vf_n, vf); cLf = np.where(m, c_n, cLf)
        first = first & ~m
    return out if per_signal else float(out.sum())


def fit_dyn(A, n_in, x0=None):
    from scipy.optimize import minimize
    k = 1 + n_in + 3 + 1 + 1 + 1 + 1 + 1 + 1
    if x0 is None:
        x0 = np.zeros(k)
        x0[0] = 0.3; x0[1:1 + n_in] = 0.3
        x0[1 + n_in + 4] = float(A["Y"][A["M"]].mean())
        x0[-5] = np.log(1.0); x0[-4] = np.log(0.1); x0[-3] = np.log(0.8); x0[-1] = np.log(0.8)
    f = lambda th: -kalman_ll(th, A, n_in)
    r = minimize(f, x0, method="L-BFGS-B", options=dict(maxiter=3000))
    return r.x, float(r.fun), bool(r.success)


def stage_dynamic():
    df = load_s1()
    # all valence signals kept as observations; inputs 0 where pleasantness or orientation missing
    d = df.copy()
    for ch in "BPF":
        d[f"u{ch}"] = d[f"u{ch}_split"].fillna(0.0)
    d.loc[~d["ori_known"] | d["c"].isna(), ["uB", "uP", "uF"]] = 0.0
    for col in ("past", "pres", "fut"):
        d.loc[~d["ori_known"], col] = 0.0
    d["uC"] = d["uB"] + d["uP"] + d["uF"]
    pids = np.array(sorted(d["pid"].unique()))
    perm = RNG.permutation(len(pids))
    folds = np.array_split(pids[perm], 5)
    specs = {"three": ["uB", "uP", "uF"], "one": ["uC"]}
    per = {k: {} for k in specs}; params = {k: [] for k in specs}
    for fi, f in enumerate(folds):
        for name, inp in specs.items():
            Atr = pad(d[~d["pid"].isin(f)], inp); Ate = pad(d[d["pid"].isin(f)], inp)
            th, nll, ok = fit_dyn(Atr, len(inp))
            params[name].append(dict(theta=th.tolist(), train_nll=nll, success=ok))
            L = kalman_ll(th, Ate, len(inp), per_signal=True)
            for i, p in enumerate(Ate["pids"]):
                per[name][p] = [float(L[i].sum()), int(Ate["M"][i].sum())]
            print(f"fold {fi} {name} ok={ok} nll/sig={nll / Atr['M'].sum():.4f} gains={th[1:1 + len(inp)]}", flush=True)
    P = sorted(per["three"])
    sa = np.array([per["three"][p][0] for p in P]); sb = np.array([per["one"][p][0] for p in P])
    nn = np.array([per["three"][p][1] for p in P])
    diff = (sa.sum() - sb.sum()) / nn.sum()
    boots = []
    for _ in range(N_BOOT):
        i = RNG.integers(0, len(P), len(P))
        boots.append((sa[i].sum() - sb[i].sum()) / nn[i].sum())
    # full-data fit for reported gains
    Afull = pad(d, specs["three"])
    thf, nllf, okf = fit_dyn(Afull, 3)
    res = dict(three_minus_one=[float(diff), *ci(boots)], n_participants=len(P), n_signals=int(nn.sum()),
               fold_params=params, full_fit=dict(phi=float(np.tanh(thf[0])), gains=thf[1:4].tolist(),
                                                   success=okf, nll_per_signal=nllf / Afull["M"].sum()))
    print(json.dumps({k: res[k] for k in ("three_minus_one", "full_fit")}, indent=1))
    json.dump(res, open(OUT / "dynamic.json", "w"), indent=1)


# ───────────────────────────── Bayer partial replication ─────────────────────────────
def stage_bayer():
    import statsmodels.formula.api as smf
    d = pd.read_csv(ROOT / "data_raw/candidates/openesm_0076_bayer/0076_bayer_ts.tsv", sep="\t", low_memory=False)
    d = d[(d.survey_error != 1) & d["pa"].notna()].copy()
    d = d.sort_values(["id", "day", "beep"])
    n = d.groupby("id")["pa"].transform("size")
    d = d[n >= 4].copy()
    d["pid"] = d["id"].astype(int).astype(str)
    d["y"] = d["pa"].astype(float)
    ts = pd.to_datetime(d["start_time"], utc=True, errors="coerce")
    d["tod"] = ((ts.dt.hour + ts.dt.minute / 60) - 9.0) / 9.0
    d["tod"] = d["tod"].fillna(d["tod"].mean())
    d["prev"] = d.groupby(["pid", "day"])["y"].shift(1)
    d["has_prev"] = d["prev"].notna().astype(float)
    pm = d.groupby("pid")["y"].transform("mean")
    d["prev_c"] = (d["prev"] - pm).fillna(0.0)
    m = d[d["time_orientation"].notna() & d["problem_thoughts"].notna()].copy().reset_index(drop=True)
    o = m["time_orientation"].astype(int)
    m["past"] = (o == 1).astype(float); m["pres"] = (o == 2).astype(float); m["fut"] = (o == 3).astype(float)
    prob = -(m["problem_thoughts"] > 0).astype(float)
    m["uB"] = prob * m["past"]; m["uP"] = prob * m["pres"]; m["uF"] = prob * m["fut"]
    res = dict(n_rows=int(len(m)), n_participants=int(m["pid"].nunique()),
               n_problem={k: int((m[k] != 0).sum()) for k in ("uB", "uP", "uF")})
    # orientation is single choice here, so present is the reference category
    form = "y ~ uB + uP + uF + past + fut + prev_c + has_prev + tod"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mm = smf.mixedlm(form, m, groups=m["pid"]).fit(reml=True)
    res["mixed"] = {k: [float(mm.params[k]), float(mm.bse[k])] for k in ("uB", "uP", "uF")}
    W, p = wald_equal(mm.params, mm.cov_params(), ["uB", "uP", "uF"])
    res["mixed_equal"] = [W, p]
    global FIX
    keep = FIX
    FIX = ["past", "fut", "prev_c", "has_prev", "tod"]
    b, V = within_ols(m, ["uB", "uP", "uF"])
    FIX = keep
    res["within"] = {k: [float(b[k]), float(np.sqrt(V.loc[k, k]))] for k in ("uB", "uP", "uF")}
    res["within_equal_cluster"] = list(wald_equal(b, V, ["uB", "uP", "uF"]))
    # present-only check: interaction pleasantness
    q = d[d["interaction_pleasant"].notna()].copy().reset_index(drop=True)
    q["ip"] = q["interaction_pleasant"] - 3.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mq = smf.mixedlm("y ~ ip + prev_c + has_prev + tod", q, groups=q["pid"]).fit(reml=True)
    res["interaction_pleasant"] = [float(mq.params["ip"]), float(mq.bse["ip"]), int(len(q))]
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / "bayer.json", "w"), indent=1)


# ───────────────────────────── report ─────────────────────────────
def bh(ps):
    ps = np.asarray(ps, float); n = len(ps); o = np.argsort(ps)
    adj = np.empty(n); prev = 1.0
    for rank in range(n, 0, -1):
        i = o[rank - 1]
        prev = min(prev, ps[i] * n / rank)
        adj[i] = prev
    return adj


def stage_report():
    from scipy import stats
    S = json.load(open(OUT / "static.json")); H = json.load(open(OUT / "heldout_static.json"))
    D = json.load(open(OUT / "dynamic.json")); B = json.load(open(OUT / "bayer.json"))
    tests = []
    for rule in ("split", "copy", "single", "signed"):
        tests.append((f"equal weights, {rule}", S[rule]["mixed_equal"][1]))
    tests.append(("3 vs 1 term, LR, split", S["split"]["LR_3_vs_1"][1]))
    for k in ("uB", "uP", "uF"):
        est, se, p = S["split"]["carry"][k]
        tests.append((f"carry-over {k}", p))
    tests.append(("equal weights, Bayer problem input", B["mixed_equal"][1]))
    # bootstrap-CI based tests converted to two-sided p via normal approx of the bootstrap
    def p_from_ci(est, lo, hi):
        se = (hi - lo) / (2 * 1.96)
        return float(2 * stats.norm.sf(abs(est) / se)) if se > 0 else 1.0
    tests.append(("held-out static 3 vs 1", p_from_ci(*H["three_minus_one"])))
    tests.append(("held-out dynamic 3 vs 1", p_from_ci(*D["three_minus_one"])))
    tests.append(("P minus F", p_from_ci(*S["split"]["diff_P_minus_F"])))
    tests.append(("F minus B", p_from_ci(*S["split"]["diff_F_minus_B"])))
    tests.append(("carry F minus B", p_from_ci(*S["split"]["carry_F_minus_B"])))
    adj = bh([t[1] for t in tests])
    rows = [dict(test=t[0], p=t[1], p_bh=float(a)) for t, a in zip(tests, adj)]
    json.dump(rows, open(OUT / "bh.json", "w"), indent=1)
    for r in rows:
        print(f"{r['test']:40s} p={r['p']:.3g}  BH={r['p_bh']:.3g}")
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    names = ["backward\n(past thoughts)", "present", "forward\n(future thoughts)"]
    w = [S["split"]["mixed"][k][0] for k in ("uB", "uP", "uF")]
    cis = [S["split"]["boot_weights_ci"][k] for k in ("uB", "uP", "uF")]
    wP = w[1]
    ax[0].bar(range(3), np.array(w) / wP, color=["#6a8caf", "#c9a227", "#7aa874"], width=0.55, label="daily life")
    ax[0].errorbar(range(3), np.array(w) / wP, yerr=[[(wi - c[0]) / wP for wi, c in zip(w, cis)],
                                                  [(c[1] - wi) / wP for wi, c in zip(w, cis)]],
                   fmt="none", ecolor="k", capsize=3)
    g = [GAMBLE["B"] / GAMBLE["P"], 1.0, GAMBLE["F"] / GAMBLE["P"]]
    ax[0].scatter(range(3), g, marker="D", color="k", zorder=3, label="gamble task")
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(names, fontsize=8)
    ax[0].set_ylabel("weight relative to present"); ax[0].legend(fontsize=8, frameon=False)
    ax[0].set_title("Channel weights", fontsize=10)
    labs = ["static model", "state-space model"]
    vals = [H["three_minus_one"], D["three_minus_one"]]
    ax[1].errorbar(range(2), [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                   fmt="o", color="k", capsize=4)
    ax[1].axhline(0, color="grey", lw=0.8)
    ax[1].set_xticks(range(2)); ax[1].set_xticklabels(labs, fontsize=8); ax[1].set_xlim(-0.5, 1.5)
    ax[1].set_ylabel("held-out log-lik gain per signal\n(three channels minus one term)")
    ax[1].set_title("Held-out participants", fontsize=10)
    plt.tight_layout(); plt.savefig(FIG / "channels_content_weights.png", dpi=170)
    print("figure written")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "static"
    {"static": stage_static, "heldout": stage_heldout, "dynamic": stage_dynamic,
     "bayer": stage_bayer, "report": stage_report}[stage]()
