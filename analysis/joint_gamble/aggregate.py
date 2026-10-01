"""Pool the five folds, bootstrap participant-level CIs, write tables and figures.

Usage: python aggregate.py  ->  out/summary.json, ../../figures/joint_gamble_*.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE.parents[1] / "figures"
B = 2000
rng = np.random.RandomState(42)


def load():
    folds = [json.loads((HERE / f"out/fold{f}.json").read_text()) for f in range(5)]
    sc = [dict(np.load(HERE / f"out/scores_fold{f}.npz")) for f in range(5)]
    keys = sc[0].keys()
    S = {k: np.concatenate([s[k] for s in sc]) for k in keys}
    W = np.concatenate([np.load(HERE / f"out/person_weights_fold{f}.npy") for f in range(5)])
    return folds, S, W


def boot_diff(num_a, num_b, den):
    """Per-observation difference sum(a-b)/sum(den) with participant bootstrap CI."""
    d = num_a - num_b
    n = len(d)
    est = d.sum() / den.sum()
    bs = np.empty(B)
    for i in range(B):
        j = rng.randint(0, n, n)
        bs[i] = d[j].sum() / den[j].sum()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"est": float(est), "lo": float(lo), "hi": float(hi)}


def fmt(r, nd=4):
    return f"{r['est']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}]"


def main():
    folds, S, W = load()
    N = len(S["H_RM__hll"])
    out = {"n_participants": int(N)}
    models = list(folds[0]["models"].keys())

    # pooled per-observation metrics
    pooled = {}
    for m in models:
        r = {}
        if f"{m}__hll" in S and folds[0]["models"][m]["test_hll_per_rating"] is not None:
            r["hll_per_rating"] = float(S[f"{m}__hll"].sum() / S[f"{m}__nh"].sum())
            r["openloop_r2_mean"] = float(np.mean([f["models"][m]["openloop_r2"] for f in folds]))
            r["openloop_r2_sd"] = float(np.std([f["models"][m]["openloop_r2"] for f in folds]))
            r["filtered_r2_raw_mean"] = float(np.mean([f["models"][m]["test_filtered_r2_raw"] for f in folds]))
        if folds[0]["models"][m]["test_cll_per_choice"] is not None:
            r["cll_per_choice"] = float(S[f"{m}__cll"].sum() / S[f"{m}__nc"].sum())
        pooled[m] = r
    out["pooled"] = pooled
    out["paper_pipeline_r2"] = {k: float(np.mean([f["paper_r2"][k] for f in folds])) for k in ["he", "chan"]}
    out["paper_pipeline_r2_sd"] = {k: float(np.std([f["paper_r2"][k] for f in folds])) for k in ["he", "chan"]}

    H = lambda m: (S[f"{m}__hll"], S[f"{m}__nh"])
    C = lambda m: (S[f"{m}__cll"], S[f"{m}__nc"])
    def dh(a, b): return boot_diff(S[f"{a}__hll"], S[f"{b}__hll"], S[f"{b}__nh"])
    def dc(a, b): return boot_diff(S[f"{a}__cll"], S[f"{b}__cll"], S[f"{b}__nc"])
    tests = {
        "happiness": {
            "readout_vs_equation (H_R - H_HE)": dh("H_R", "H_HE"),
            "slow_mood_added_to_readout (H_RM - H_R)": dh("H_RM", "H_R"),
            "slow_mood_added_to_equation (H_HE_mood - H_HE)": dh("H_HE_mood", "H_HE"),
            "two_timescale_readout_vs_equation_with_mood (H_RM - H_HE_mood)": dh("H_RM", "H_HE_mood"),
            "DIR2 shared parameters (J_noEKF - H_RM)": dh("J_noEKF", "H_RM"),
            "DIR2 shared parameters + choices update mood (J - H_RM)": dh("J", "H_RM"),
            "DIR2 choices as mood evidence (J - J_noEKF)": dh("J", "J_noEKF"),
            "ablation equal weights (H_RM_eq - H_RM)": dh("H_RM_eq", "H_RM"),
            "ablation no forward (H_RM_noF - H_RM)": dh("H_RM_noF", "H_RM"),
            "ablation no present (H_RM_noP - H_RM)": dh("H_RM_noP", "H_RM"),
            "ablation no backward (H_RM_noB - H_RM)": dh("H_RM_noB", "H_RM"),
            "joint ablation no mood (J_nomood - J)": dh("J_nomood", "J"),
            "joint ablation no forward (J_noF - J)": dh("J_noF", "J"),
            "joint ablation no present (J_noP - J)": dh("J_noP", "J"),
            "joint ablation no backward (J_noB - J)": dh("J_noB", "J"),
        },
        "choice": {
            "prospect_theory_vs_EV (C_PT - C_EV)": dc("C_PT", "C_EV"),
            "mood_from_outcomes_only (C_PT_task - C_PT)": dc("C_PT_task", "C_PT"),
            "DIR1 mood filtered from ratings, happiness fitted separately (Plug - C_PT)": dc("Plug", "C_PT"),
            "DIR1 joint fit (J_noEKF - C_PT)": dc("J_noEKF", "C_PT"),
            "DIR1 joint fit + choices update mood (J - C_PT)": dc("J", "C_PT"),
            "joint fitting over plug-in (J_noEKF - Plug)": dc("J_noEKF", "Plug"),
            "joint ablation no mood level (J_nomood - J)": dc("J_nomood", "J"),
            "joint ablation no forward (J_noF - J)": dc("J_noF", "J"),
            "joint ablation no present (J_noP - J)": dc("J_noP", "J"),
            "joint ablation no backward (J_noB - J)": dc("J_noB", "J"),
        },
    }
    # held-out trials (trials 16-30), per-person offsets fitted on trials 1-15
    nc2 = S["C_PT__nc2"]; nh2 = S["H_RM__nh2"]
    hot = lambda a, b, den: boot_diff(S[f"HO__{a}"], S[f"HO__{b}"], den)
    tests["heldout_trials_choice"] = {
        "PT vs EV": hot("C_PT", "C_EV", nc2),
        "DIR1 plug-in (Plug - C_PT)": hot("Plug", "C_PT", nc2),
        "DIR1 joint (J_noEKF - C_PT)": hot("J_noEKF", "C_PT", nc2),
        "DIR1 joint + choice updates (J - C_PT)": hot("J", "C_PT", nc2),
        "mood from outcomes only (C_PT_task - C_PT)": hot("C_PT_task", "C_PT", nc2),
    }
    tests["heldout_trials_happiness"] = {
        "DIR2 (J_noEKF - H_RM), no per-person offsets": boot_diff(S["J_noEKF__hll2"], S["H_RM__hll2"], nh2),
        "DIR2 (J - H_RM), no per-person offsets": boot_diff(S["J__hll2"], S["H_RM__hll2"], nh2),
        "per-person weights vs group weights (H_RM_pw - H_RM)": boot_diff(S["HO__H_RM_pw"], S["H_RM__hll2"], nh2),
        "equal weights with per-person weights vs free (H_RM_eq_pw - H_RM_pw)": boot_diff(S["HO__H_RM_eq_pw"], S["HO__H_RM_pw"], nh2),
    }
    out["tests"] = tests
    out["heldout_trials_pooled"] = {
        "choice_ll_per_choice": {m: float(S[f"HO__{m}"].sum() / nc2.sum()) for m in ["C_EV", "C_PT", "C_PT_task", "Plug", "J_noEKF", "J", "J_nomood"]},
    }

    # parameters across folds
    def par(m, key, i=None):
        vals = []
        for f in folds:
            p = f["models"][m]["params"]
            v = p["_derived"][key] if key in p["_derived"] else p[key]
            vals.append(v if i is None else v[i])
        vals = np.array(vals, float)
        return {"mean": vals.mean(0).tolist(), "sd": vals.std(0).tolist()}
    params = {}
    for m in ["H_HE", "H_R", "H_RM", "H_RM_eq", "C_EV", "C_PT", "Plug", "J_noEKF", "J"]:
        pm = {}
        for key in ["w", "w_eq", "gamma", "rho", "mood_timescale_trials", "k", "q", "tau", "sigma", "mu", "lambda", "alpha", "b", "eta", "zeta"]:
            try:
                pm[key] = par(m, key)
            except KeyError:
                pass
        params[m] = pm
    out["params"] = params
    out["offset_priors"] = [{"choice": f["choice_offset_prior"], "weights": f["weight_offset_prior"]} for f in folds]

    # person-level channel weights (shrinkage MAP, all ratings)
    out["person_weights"] = {
        "n": int(len(W)),
        "median": np.median(W, 0).tolist(),
        "q05_q25_q75_q95": np.percentile(W, [5, 25, 75, 95], 0).T.tolist(),
        "frac_positive": (W > 0).mean(0).tolist(),
        "sd": W.std(0).tolist(),
        "corr": np.corrcoef(W.T).tolist(),
    }
    (HERE / "out/summary.json").write_text(json.dumps(out, indent=1))

    # console tables
    print(f"participants {N}")
    print("paper pipeline R2 (he, chan):", out["paper_pipeline_r2"], out["paper_pipeline_r2_sd"])
    for m, r in pooled.items():
        print(f"  {m:10s} " + "  ".join(f"{k}={v:.4f}" for k, v in r.items()))
    for grp, d in tests.items():
        print(grp)
        for k, v in d.items():
            print(f"   {k:75s} {fmt(v)}")
    print("person weights median", out["person_weights"]["median"], "frac>0", out["person_weights"]["frac_positive"])
    figures(out, W)


def figures(out, W):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    t = out["tests"]
    # 1. unification tests, both directions
    items = [
        ("Choice | ratings-filtered mood,\nhappiness fitted alone", t["choice"]["DIR1 mood filtered from ratings, happiness fitted separately (Plug - C_PT)"]),
        ("Choice | joint fit", t["choice"]["DIR1 joint fit (J_noEKF - C_PT)"]),
        ("Choice | joint fit,\nchoices update mood", t["choice"]["DIR1 joint fit + choice updates mood (J - C_PT)"] if "DIR1 joint fit + choice updates mood (J - C_PT)" in t["choice"] else t["choice"]["DIR1 joint fit + choices update mood (J - C_PT)"]),
        ("Choice | mood from\noutcomes only", t["choice"]["mood_from_outcomes_only (C_PT_task - C_PT)"]),
        ("Happiness | shared\nparameters", t["happiness"]["DIR2 shared parameters (J_noEKF - H_RM)"]),
        ("Happiness | choices\nupdate mood", t["happiness"]["DIR2 shared parameters + choices update mood (J - H_RM)"]),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, (lab, r) in enumerate(items):
        col = "#3b6ea5" if lab.startswith("Choice") else "#b5651d"
        ax.errorbar(i, r["est"], yerr=[[r["est"] - r["lo"]], [r["hi"] - r["est"]]], fmt="o", color=col, capsize=4)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(len(items))); ax.set_xticklabels([l for l, _ in items], fontsize=7)
    ax.set_ylabel("held-out log-lik gain\n(nats per choice or rating)")
    ax.set_title("Joint fitting of happiness and choice, held-out participants", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "joint_gamble_unification.png", dpi=180); plt.close(fig)
    # 2. channel weights
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    g = out["params"]["H_RM"]["w"]
    axs[0].bar(range(3), g["mean"], yerr=g["sd"], color=["#6a8caf", "#c9a227", "#8c5a8c"], capsize=4)
    axs[0].axhline(out["params"]["H_RM_eq"]["w_eq"]["mean"][0], ls="--", color="k", lw=0.8, label="equal-weight fit")
    axs[0].set_xticks(range(3)); axs[0].set_xticklabels(["forward", "present", "backward"])
    axs[0].set_ylabel("weight (happiness per point, both /100)"); axs[0].legend(fontsize=7)
    axs[0].set_title("Group weights, common utility scale", fontsize=9)
    axs[1].violinplot([W[:, 0], W[:, 1], W[:, 2]], showmedians=True)
    axs[1].axhline(0, color="grey", lw=0.8)
    axs[1].set_xticks([1, 2, 3]); axs[1].set_xticklabels(["forward", "present", "backward"])
    axs[1].set_title("Per-participant weights (shrinkage estimates)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "joint_gamble_weights.png", dpi=180); plt.close(fig)
    # 3. ablations
    ab = [("equal weights", t["happiness"]["ablation equal weights (H_RM_eq - H_RM)"]),
          ("no forward", t["happiness"]["ablation no forward (H_RM_noF - H_RM)"]),
          ("no present", t["happiness"]["ablation no present (H_RM_noP - H_RM)"]),
          ("no backward", t["happiness"]["ablation no backward (H_RM_noB - H_RM)"]),
          ("no slow mood", {k: -v for k, v in t["happiness"]["slow_mood_added_to_readout (H_RM - H_R)"].items()} if False else None)]
    r = t["happiness"]["slow_mood_added_to_readout (H_RM - H_R)"]
    ab[-1] = ("no slow mood", {"est": -r["est"], "lo": -r["hi"], "hi": -r["lo"]})
    fig, ax = plt.subplots(figsize=(6, 3))
    for i, (lab, r) in enumerate(ab):
        ax.errorbar(i, r["est"], yerr=[[r["est"] - r["lo"]], [r["hi"] - r["est"]]], fmt="o", color="#444", capsize=4)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(len(ab))); ax.set_xticklabels([l for l, _ in ab], fontsize=8)
    ax.set_ylabel("change in held-out happiness\nlog-lik per rating")
    ax.set_title("Ablations of the two-timescale readout", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "joint_gamble_ablations.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
