"""Vulnerability round 6: standard multilevel tests of stress reactivity and emotional inertia.

Predictions and plan: reviews/DIATHESIS_R6.md, committed in 138df02 before this script was written.

Run from the repo root:  python analysis/diathesis_r6/diathesis_r6.py
Outputs: analysis/diathesis_r6/out/results.json (aggregate), out/run.log,
         out/blups_*.csv (participant level, git-ignored).
"""
from __future__ import annotations

import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
CAND = ROOT / "data_raw" / "candidates"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
LOG = open(OUT / "run.log", "a", encoding="utf-8")
MIN_SIG = 12
RNG = np.random.default_rng(20261001)


def log(s):
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def mean_items(df, cols):
    return df[cols].mean(axis=1, skipna=True).where(df[cols].notna().any(axis=1))


# ── loaders: one signal-level frame per sample ──────────────────────────────
# Columns: pid, seg (lag allowed only within seg), pos (position within seg),
# NA, PA, V (valence composite as used by v2.1), STR (stressor), TH (thought item),
# and person-level traits.

def load_geschwind():
    df = pd.read_csv(ROOT / "data_raw" / "geschwind_2013_s004.csv", na_values=["NA"])
    df = df.rename(columns=dict(subjno="pid", dayno="day", beepno="beep", st_period="period",
                                opgewkt_="cheerful", onplplez="pleas", pieker="worried",
                                angstig_="fearful", somber__="sad", ontspann="relaxed", neur="neuroticism"))
    df["pid"] = df["pid"].astype(int).astype(str)
    df = df.sort_values(["pid", "period", "day", "beep"])
    df["NA"] = mean_items(df, ["worried", "fearful", "sad"])
    df["PA"] = mean_items(df, ["cheerful", "relaxed"])
    df["V"] = df["PA"] - df["NA"]
    df["STR"] = (-df["pleas"]).clip(lower=0)
    df.loc[df["pleas"].isna(), "STR"] = np.nan
    df["TH"] = df["worried"]
    df["seg"] = df["pid"] + "_" + df["period"].astype(str) + "_" + df["day"].astype(str)
    df["pos"] = df["beep"]
    tr = df.groupby("pid")["neuroticism"].first().dropna()
    traits = pd.DataFrame(dict(neuroticism=tr))
    return df, traits


def load_kane():
    k = pd.read_csv(CAND / "kane2017_esm" / "Kane_ESM_L1.csv", na_values=[" ", ""], skipinitialspace=True)
    k["pid"] = k["subjnumb"].astype(int).astype(str)
    k["pos"] = k.groupby("pid").cumcount()
    k["NA"] = mean_items(k, ["esm18", "esm20", "esm22"])
    k["PA"] = k["esm16"]
    k["V"] = k["PA"] - k["NA"]
    k["STR"] = k["esm34"]
    mw = k["esm01"]
    k["TH"] = np.where(mw == 1, k["esm04"], np.where(mw == 2, 1.0, np.nan))
    k["seg"] = k["pid"]   # no day index: consecutive rows, some cross a night
    L2 = pd.read_csv(CAND / "kane2017_esm" / "Kane_ESM_and_NEO_L2.csv", na_values=[" ", ""], skipinitialspace=True)
    L2["pid"] = L2["subjnumb"].astype(int).astype(str)
    traits = L2.set_index("pid")[["N"]].rename(columns=dict(N="neuroticism")).dropna()
    return k, traits


def load_gainey():
    g = pd.read_csv(CAND / "openesm_0058_gainey" / "0058_gainey_ts.tsv", sep="\t")
    g["pid"] = g["id"].astype(int).astype(str)
    g = g.sort_values(["pid", "day", "beep"])
    g["NA"] = mean_items(g, ["irritable", "upset", "afraid_anxious", "sad"])
    g["PA"] = mean_items(g, ["active", "interested", "excited", "strong"])
    g["V"] = g["PA"] - g["NA"]
    g["STR"] = np.nan
    g["TH"] = g["brooding"]
    g["seg"] = g["pid"] + "_" + g["day"].astype(str)
    g["pos"] = g["beep"]
    st = pd.read_csv(CAND / "openesm_0058_gainey" / "0058_gainey_static.tsv", sep="\t", low_memory=False)
    st = st.groupby("ID").first()
    st.index = st.index.astype(int).astype(str)
    traits = pd.DataFrame(dict(neuroticism=pd.to_numeric(st["BFI_N"], errors="coerce"),
                               dysphoria=pd.to_numeric(st["IDASDys10"], errors="coerce"),
                               brooding_trait=pd.to_numeric(st["RRS"], errors="coerce")))
    traits = traits[traits["neuroticism"].notna()]
    return g, traits


LOADERS = dict(Geschwind=load_geschwind, Kane=load_kane, Gainey=load_gainey)


def prepare(df, traits):
    df = df[df["pid"].isin(traits.index)].copy()
    keep = df.groupby("pid")["NA"].apply(lambda s: s.notna().sum() >= MIN_SIG)
    df = df[df["pid"].isin(keep[keep].index)].copy()
    # scale affect by within-sample total SD; thought item too
    for c in ("NA", "PA", "V", "TH", "STR"):
        sd = df[c].std()
        df[c + "_s"] = df[c] / sd if sd and not math.isnan(sd) else np.nan
    # lag-1 within segment, adjacent positions only
    df = df.sort_values(["pid", "seg", "pos"])
    grp = df.groupby("seg")
    adj = grp["pos"].diff() == 1
    for c in ("NA_s", "PA_s", "V_s", "TH_s", "STR_s"):
        df[c + "_lag"] = grp[c].shift(1).where(adj)
    # person means and SDs, person-mean centring
    for c in ("NA_s", "PA_s", "V_s", "TH_s", "STR_s"):
        pm = df.groupby("pid")[c].transform("mean")
        df[c + "_pm"] = pm
        df[c + "_pc"] = df[c] - pm
        df[c + "_lagpc"] = df[c + "_lag"] - pm
        df[c + "_psd"] = df.groupby("pid")[c].transform("std")
    t = traits.loc[df["pid"].unique()].copy()
    for c in t.columns:
        t[c + "_z"] = (t[c] - t[c].mean()) / t[c].std()
    df = df.join(t[[c for c in t.columns if c.endswith("_z")]], on="pid")
    # person-level z of person mean and person SD (for Koval controls)
    for c in ("NA_s", "PA_s", "V_s", "TH_s"):
        pp = df.groupby("pid")[[c + "_pm", c + "_psd"]].first()
        pz = (pp - pp.mean()) / pp.std()
        pz.columns = [c + "_pmz", c + "_psdz"]
        df = df.join(pz, on="pid")
    return df


# ── mixed models ────────────────────────────────────────────────────────────
def fit_mixed(df, formula, slope, needed, label):
    d = df.dropna(subset=needed).copy()
    d = d[d.groupby("pid")["pid"].transform("size") >= 5]
    t0 = time.time()
    # Several optimizers; keep the converged fit with the highest REML log-likelihood and finite
    # fixed-effect SEs. A random-slope variance at the zero boundary is a valid REML solution
    # (its own SE is then undefined); it is flagged, not rejected.
    res, struct, best = None, "correlated", -np.inf
    md = smf.mixedlm(formula, d, groups=d["pid"], re_formula="~" + slope)
    for meth in ("lbfgs", "bfgs", "nm", "powell"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = md.fit(reml=True, method=meth, maxiter=5000)
            fe_se = r.bse[r.fe_params.index].values
            if r.converged and np.all(np.isfinite(fe_se)) and r.llf > best + 1e-6:
                res, best = r, r.llf
        except Exception as e:  # noqa: BLE001
            log(f"    {label}: {meth} failed: {e}")
    if res is None:
        return None
    try:
        sl_var = float(res.cov_re.loc[slope, slope])
        if sl_var < 1e-6 * max(float(res.cov_re.iloc[0, 0]), 1e-12):
            struct = "correlated (slope variance at the zero boundary)"
    except Exception:  # noqa: BLE001
        pass
    out = dict(label=label, n_obs=int(d.shape[0]), n_people=int(d["pid"].nunique()), structure=struct,
               seconds=round(time.time() - t0, 1), fixed={})
    ci = res.conf_int()
    for name in res.fe_params.index:
        out["fixed"][name] = dict(b=float(res.fe_params[name]), se=float(res.bse[name]),
                                  lo=float(ci.loc[name, 0]), hi=float(ci.loc[name, 1]),
                                  p=float(res.pvalues[name]))
    # random-slope SD
    try:
        cov = res.cov_re
        out["slope_sd"] = float(math.sqrt(max(cov.loc[slope, slope], 0.0)))
        out["intercept_sd"] = float(math.sqrt(max(cov.iloc[0, 0], 0.0)))
    except Exception:  # noqa: BLE001
        out["slope_sd"] = None
    out["llf"] = float(res.llf)
    out["_res"] = res
    out["_data"] = d
    log(f"    {label}: n={out['n_obs']} people={out['n_people']} {struct} {out['seconds']}s")
    return out


def person_slopes(fit, slope):
    """Per-person slope (fixed + BLUP) for the correlated structure."""
    res = fit["_res"]
    re = res.random_effects
    rows = {}
    base = float(res.fe_params[slope])
    for pid, v in re.items():
        if slope in v.index:
            rows[pid] = base + float(v[slope])
        elif len(v) > 1:
            rows[pid] = base + float(v.iloc[-1])
    return pd.Series(rows)


def pool(estimates):
    """DerSimonian-Laird and Hartung-Knapp random-effects pooling of (b, se) pairs."""
    b = np.array([e[0] for e in estimates], float)
    se = np.array([e[1] for e in estimates], float)
    k = len(b)
    w = 1 / se ** 2
    bf = float((w * b).sum() / w.sum())
    q = float((w * (b - bf) ** 2).sum())
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 else 0.0
    ws = 1 / (se ** 2 + tau2)
    bdl = float((ws * b).sum() / ws.sum())
    sedl = float(math.sqrt(1 / ws.sum()))
    out = dict(k=k, tau2=tau2, Q=q, DL=dict(b=bdl, lo=bdl - 1.96 * sedl, hi=bdl + 1.96 * sedl,
                                             p=float(2 * stats.norm.sf(abs(bdl / sedl)))))
    if k > 1:
        hk_var = float((ws * (b - bdl) ** 2).sum() / ((k - 1) * ws.sum()))
        sehk = math.sqrt(max(hk_var, 1e-300))
        tq = stats.t.ppf(0.975, k - 1)
        out["HK"] = dict(b=bdl, lo=bdl - tq * sehk, hi=bdl + tq * sehk,
                         p=float(2 * stats.t.sf(abs(bdl / sehk), k - 1)))
    return out


def strip(f):
    return {k: v for k, v in f.items() if not k.startswith("_")} if f else None


# ── main ────────────────────────────────────────────────────────────────────
def main():
    log(f"=== run {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    data = {}
    desc = {}
    for name, loader in LOADERS.items():
        raw, traits = loader()
        df = prepare(raw, traits)
        data[name] = df
        lagn = int(df["NA_s_lag"].notna().sum())
        desc[name] = dict(people=int(df["pid"].nunique()), signals=int(df["NA"].notna().sum()),
                          lag_pairs=lagn, stressor_signals=int(df["STR"].notna().sum()),
                          thought_signals=int(df["TH"].notna().sum()),
                          sd=dict(NA=float(df["NA"].std()), PA=float(df["PA"].std())))
        log(f"[{name}] {desc[name]}")

    results = dict(desc=desc, reactivity={}, lagged={}, inertia={}, persistence={}, link={})

    # R1 concurrent stress reactivity
    for name in ("Geschwind", "Kane"):
        df = data[name]
        for aff in ("NA", "PA"):
            f = fit_mixed(df, f"{aff}_s ~ STR_s_pc * neuroticism_z + STR_s_pm", "STR_s_pc",
                          [f"{aff}_s", "STR_s_pc", "neuroticism_z"], f"{name} R1 {aff}")
            results["reactivity"][f"{name}_{aff}"] = strip(f)
        # R2 lagged
        f = fit_mixed(df, "NA_s ~ NA_s_lagpc + STR_s_lagpc * neuroticism_z + STR_s_pc + STR_s_pm",
                      "STR_s_lagpc", ["NA_s", "NA_s_lagpc", "STR_s_lagpc", "STR_s_pc", "neuroticism_z"],
                      f"{name} R2 NA lagged")
        results["lagged"][f"{name}_NA"] = strip(f)

    # I1/I2/I3 inertia with Koval controls
    inertia_fits = {}
    for name, df in data.items():
        traits = ["neuroticism"] + (["dysphoria", "brooding_trait"] if name == "Gainey" else [])
        for aff in ("NA", "PA"):
            for tr in traits:
                if aff == "PA" and tr != "neuroticism":
                    continue
                form = (f"{aff}_s ~ {aff}_s_lagpc * ({tr}_z + {aff}_s_pmz + {aff}_s_psdz)")
                f = fit_mixed(df, form, f"{aff}_s_lagpc",
                              [f"{aff}_s", f"{aff}_s_lagpc", f"{tr}_z", f"{aff}_s_pmz", f"{aff}_s_psdz"],
                              f"{name} I {aff} x {tr}")
                results["inertia"][f"{name}_{aff}_{tr}"] = strip(f)
                inertia_fits[(name, aff, tr)] = f
                # without Koval controls, for comparison with the literature's simpler form
                f2 = fit_mixed(df, f"{aff}_s ~ {aff}_s_lagpc * {tr}_z", f"{aff}_s_lagpc",
                               [f"{aff}_s", f"{aff}_s_lagpc", f"{tr}_z"], f"{name} I {aff} x {tr} (no controls)")
                results["inertia"][f"{name}_{aff}_{tr}_nocontrols"] = strip(f2)

    # P1 persistence of negative thought
    for name, df in data.items():
        traits = ["neuroticism"] + (["dysphoria", "brooding_trait"] if name == "Gainey" else [])
        for tr in traits:
            form = f"TH_s ~ TH_s_lagpc * ({tr}_z + TH_s_pmz + TH_s_psdz)"
            f = fit_mixed(df, form, "TH_s_lagpc", ["TH_s", "TH_s_lagpc", f"{tr}_z", "TH_s_pmz", "TH_s_psdz"],
                          f"{name} P thought x {tr}")
            results["persistence"][f"{name}_{tr}"] = strip(f)

    # pooling of neuroticism interactions
    def term(fit, name):
        if not fit or name not in fit["fixed"]:
            return None
        x = fit["fixed"][name]
        return (x["b"], x["se"])

    pooled = {}
    sets = {
        "R1_NA": [(results["reactivity"].get(f"{s}_NA"), "STR_s_pc:neuroticism_z") for s in ("Geschwind", "Kane")],
        "R1_PA": [(results["reactivity"].get(f"{s}_PA"), "STR_s_pc:neuroticism_z") for s in ("Geschwind", "Kane")],
        "R2_NA": [(results["lagged"].get(f"{s}_NA"), "STR_s_lagpc:neuroticism_z") for s in ("Geschwind", "Kane")],
        "I_NA": [(results["inertia"].get(f"{s}_NA_neuroticism"), "NA_s_lagpc:neuroticism_z") for s in data],
        "I_PA": [(results["inertia"].get(f"{s}_PA_neuroticism"), "PA_s_lagpc:neuroticism_z") for s in data],
        "I_NA_nocontrols": [(results["inertia"].get(f"{s}_NA_neuroticism_nocontrols"), "NA_s_lagpc:neuroticism_z") for s in data],
        "P_thought": [(results["persistence"].get(f"{s}_neuroticism"), "TH_s_lagpc:neuroticism_z") for s in data],
    }
    for key, items in sets.items():
        est = [term(f, t) for f, t in items]
        est = [e for e in est if e is not None]
        if len(est) >= 2:
            pooled[key] = pool(est)
    results["pooled"] = pooled

    # mega-analysis: NA reactivity (two samples) and NA inertia (three samples)
    mega = {}
    cols_r = ["NA_s", "STR_s_pc", "STR_s_pm", "neuroticism_z"]
    dr = pd.concat([data[s].assign(sample=s)[cols_r + ["pid", "sample"]] for s in ("Geschwind", "Kane")])
    dr["pid"] = dr["sample"] + "_" + dr["pid"]
    f = fit_mixed(dr, "NA_s ~ C(sample) + C(sample):STR_s_pc + STR_s_pc:neuroticism_z + neuroticism_z + STR_s_pm",
                  "STR_s_pc", cols_r, "MEGA R1 NA")
    mega["R1_NA"] = strip(f)
    cols_i = ["NA_s", "NA_s_lagpc", "neuroticism_z", "NA_s_pmz", "NA_s_psdz"]
    di = pd.concat([data[s].assign(sample=s)[cols_i + ["pid", "sample"]] for s in data])
    di["pid"] = di["sample"] + "_" + di["pid"]
    f = fit_mixed(di, "NA_s ~ C(sample) + C(sample):NA_s_lagpc + NA_s_lagpc:(neuroticism_z + NA_s_pmz + NA_s_psdz)"
                      " + neuroticism_z + NA_s_pmz + NA_s_psdz",
                  "NA_s_lagpc", cols_i, "MEGA I NA")
    mega["I_NA"] = strip(f)
    results["mega"] = mega

    # link to v2.1 per-person parameters: valence slopes from mixed models
    for name, df in data.items():
        pfile = ROOT / "analysis" / "diathesis" / "out" / f"person_{name}.json"
        if not pfile.exists():
            continue
        persons = json.load(open(pfile))["persons"]
        pp = pd.DataFrame({pid: dict(log_theta=math.log(max(v["theta"] - 2.0, 1e-9)), beta_P=v["beta_P"])
                           for pid, v in persons.items()}).T
        link = {}
        fv = fit_mixed(df, "V_s ~ V_s_lagpc", "V_s_lagpc", ["V_s", "V_s_lagpc"], f"{name} link V inertia")
        if fv and fv["structure"].startswith("correlated"):
            sl = person_slopes(fv, "V_s_lagpc")
            j = pp.join(sl.rename("v_inertia"), how="inner").dropna()
            link["log_timescale_vs_inertia_slope"] = spearman_boot(j["log_theta"], j["v_inertia"])
            pd.DataFrame(dict(v_inertia=sl)).to_csv(OUT / f"blups_{name}_inertia.csv")
        if name in ("Geschwind", "Kane"):
            fr = fit_mixed(df, "V_s ~ STR_s_pc + STR_s_pm", "STR_s_pc", ["V_s", "STR_s_pc"], f"{name} link V reactivity")
            if fr and fr["structure"].startswith("correlated"):
                sl = person_slopes(fr, "STR_s_pc")
                j = pp.join(sl.rename("v_react"), how="inner").dropna()
                link["event_weight_vs_reactivity_slope"] = spearman_boot(j["beta_P"], j["v_react"])
                pd.DataFrame(dict(v_react=sl)).to_csv(OUT / f"blups_{name}_react.csv")
        results["link"][name] = link

    json.dump(results, open(OUT / "results.json", "w"), indent=1)
    log("results written")


def spearman_boot(x, y, B=2000):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    r = float(stats.spearmanr(x, y).statistic)
    n = len(x)
    bs = []
    for _ in range(B):
        i = RNG.integers(0, n, n)
        bs.append(stats.spearmanr(x[i], y[i]).statistic)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(rho=r, lo=float(lo), hi=float(hi), n=n)


if __name__ == "__main__":
    main()
