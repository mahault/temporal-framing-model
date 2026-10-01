"""Aggregate the round-5 gamble analyses into out/gamble_r5_summary.json.

Inputs: out/fast_fold{0..4}.json and fast_scores_fold{0..4}.npz (fast_choice.py), the joint run's
analysis/joint_gamble/out/scores_fold{F}.npz (for J_noP and J_nomood choice scores on the same
participants), out/play2.json, out/bdi_sign.json, out/boot_bdi_w*.json (bdi_boot.py),
out/bdi_play2.json (bdi_play2.py), analysis/gamble_bdi/out/summary.json.

Usage: python analysis/gamble_r5/aggregate_r5.py
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
rs = np.random.RandomState(3)
T = 30


def boot_mean(x, nb=2000):
    x = np.asarray(x)
    m = np.array([x[rs.randint(0, len(x), len(x))].mean() for _ in range(nb)])
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    out = {}
    # ---------------------------------------------------------------- fast-affect coupling to choice
    fj = [json.loads((HERE / f"out/fast_fold{f}.json").read_text()) for f in range(5)]
    sc = [dict(np.load(HERE / f"out/fast_scores_fold{f}.npz")) for f in range(5)]
    js = [dict(np.load(ROOT / f"analysis/joint_gamble/out/scores_fold{f}.npz")) for f in range(5)]
    cat = lambda key: np.concatenate([s[key] for s in sc])
    jcat = lambda key: np.concatenate([s[key] for s in js])
    # per-choice differences, all trials (30 choices per person) and held-out trials 16-30 (15)
    pairs = {"J_fast_minus_J": ("J_fast", "J_ref"), "C_PT_fast_minus_C_PT": ("C_PT_fast", "C_PT_ref"),
             "C_PT_lag_minus_C_PT": ("C_PT_lag", "C_PT_ref"), "J_fast_minus_C_PT_fast": ("J_fast", "C_PT_fast")}
    fc = {}
    for lab, (a, b) in pairs.items():
        fc[lab] = {"all_trials_per_choice": boot_mean((cat(f"{a}__cll") - cat(f"{b}__cll")) / T),
                   "heldout_trials_per_choice": boot_mean((cat(f"{a}__ho") - cat(f"{b}__ho")) / 15)}
    # the flagged effect: J_noP and J_nomood beat J on choice. Does J_fast close that gap?
    fc["J_noP_minus_J"] = {"all_trials_per_choice": boot_mean((jcat("J_noP__cll") - jcat("J__cll")) / T)}
    fc["J_nomood_minus_J"] = {"all_trials_per_choice": boot_mean((jcat("J_nomood__cll") - jcat("J__cll")) / T)}
    fc["J_fast_minus_J_noP"] = {"all_trials_per_choice": boot_mean((cat("J_fast__cll") - jcat("J_noP__cll")) / T)}
    fc["params_by_fold"] = {m: [f["models"][m]["params"] for f in fj] for m in ["J_fast", "C_PT_fast", "C_PT_lag"]}
    # set 2: perseveration controls
    if all((HERE / f"out/fast2_fold{f}.json").exists() for f in range(5)):
        fj2 = [json.loads((HERE / f"out/fast2_fold{f}.json").read_text()) for f in range(5)]
        sc2 = [dict(np.load(HERE / f"out/fast2_scores_fold{f}.npz")) for f in range(5)]
        both = [{**a_, **b_} for a_, b_ in zip(sc, sc2)]
        cat2 = lambda key: np.concatenate([s_[key] for s_ in both])
        pairs2 = {"C_PT_pers_minus_C_PT": ("C_PT_pers", "C_PT_ref"),
                  "C_PT_pers_lag_minus_C_PT_pers": ("C_PT_pers_lag", "C_PT_pers"),
                  "C_PT_pers_fast_minus_C_PT_pers": ("C_PT_pers_fast", "C_PT_pers"),
                  "J_pers_minus_J": ("J_pers", "J_ref"),
                  "J_pers_fast_minus_J_pers": ("J_pers_fast", "J_pers"),
                  "J_pers_fast_minus_C_PT_pers_fast": ("J_pers_fast", "C_PT_pers_fast")}
        for lab, (a_, b_) in pairs2.items():
            fc[lab] = {"all_trials_per_choice": boot_mean((cat2(f"{a_}__cll") - cat2(f"{b_}__cll")) / T),
                       "heldout_trials_per_choice": boot_mean((cat2(f"{a_}__ho") - cat2(f"{b_}__ho")) / 15)}
        fc["params_by_fold_set2"] = {m: [f["models"][m]["params"] for f in fj2]
                                     for m in ["C_PT_pers", "C_PT_pers_lag", "C_PT_pers_fast", "J_pers", "J_pers_fast"]}
        fc["happiness_ll_per_rating_J_pers_fast"] = [f["models"]["J_pers_fast"]["test_hll_per_rating"] for f in fj2]
    fc["happiness_ll_per_rating_J_fast"] = [f["models"]["J_fast"]["test_hll_per_rating"] for f in fj]
    out["fast_choice"] = fc
    # ---------------------------------------------------------------- BDI bootstrap
    reps = []
    for p in sorted(glob.glob(str(HERE / "out/boot_bdi_w*.json"))):
        reps += json.loads(Path(p).read_text())
    B = np.array([r["beta"] for r in reps])            # (nrep, 6 params, 3 covariates)
    s = json.loads((ROOT / "analysis/gamble_bdi/out/summary.json").read_text())
    params = ["a", "rho", "wF", "wP", "wB", "eta"]
    bb = {}
    for i, p in enumerate(params):
        if p == "rho":
            continue
        x = B[:, i, 0]
        est = s["covariate_model"][p]["BDI"]["beta"]
        se = float(x.std())
        z = est / se
        from math import erf, sqrt
        pz = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
        bb[p] = {"estimate": est, "boot_se": se, "boot_lo": float(np.percentile(x, 2.5)),
                 "boot_hi": float(np.percentile(x, 97.5)), "z": float(z), "p_boot_normal": float(pz),
                 "sandwich_se_previous": s["covariate_model"][p]["BDI"]["se"],
                 "lr_chi2_previous": s["lr_tests_bdi"][p]["chi2_1"]}
    out["bdi_bootstrap"] = {"n_replicates": int(len(reps)), "effects": bb}
    # play-2 BDI replication bootstrap
    reps2 = []
    for p_ in sorted(glob.glob(str(HERE / "out/boot_play2_w*.json"))):
        reps2 += json.loads(Path(p_).read_text())
    if reps2 and (HERE / "out/bdi_play2.json").exists():
        d2 = json.loads((HERE / "out/bdi_play2.json").read_text())
        B2 = np.array([r["beta"] for r in reps2])
        from math import erf, sqrt
        eff2 = {}
        for i, p in enumerate(params):
            if p == "rho":
                continue
            x = B2[:, i, 0]; est = d2["beta_bdi"][p]; se = float(x.std())
            eff2[p] = {"estimate": est, "boot_se": se, "boot_lo": float(np.percentile(x, 2.5)),
                       "boot_hi": float(np.percentile(x, 97.5)), "z": est / se,
                       "p_boot_normal": float(2 * (1 - 0.5 * (1 + erf(abs(est / se) / sqrt(2))))),
                       "lr_chi2": d2["lr"][p]["chi2_1"]}
        out["bdi_play2_bootstrap"] = {"n_replicates": len(reps2), "N": d2["N"], "effects": eff2}
    for k, f in [("bdi_sign", "out/bdi_sign.json"), ("play2", "out/play2.json"), ("bdi_play2", "out/bdi_play2.json")]:
        if (HERE / f).exists():
            out[k] = json.loads((HERE / f).read_text())
    (HERE / "out/gamble_r5_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k in ("fast_choice", "bdi_bootstrap")}, indent=1)[:6000])


if __name__ == "__main__":
    main()
