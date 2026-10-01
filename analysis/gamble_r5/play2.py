"""What the three-channel readout does that the happiness equation does not, on existing data.

Models (group parameters from the fold of analysis/joint_gamble in which each participant was
held out, so no group parameter ever saw the participant):
  H_RM       three-channel readout with slow mood (forward EV, present RPE, backward -|RPE|)
  H_HE_mood  Rutledge happiness equation readout (certain reward, EV, RPE) with the same slow mood

1. Transfer to a second play. Per-person readout weights are fitted (MAP offsets, prior sd chosen
   by split-half cross-validation on training participants' first plays) on each person's first
   play and used to predict happiness on that person's second play, recorded on a later occasion.
   Score: log-likelihood per rating on play 2. Compared: group weights vs play-1 person weights,
   within each model, and between models.
2. Test-retest reliability of per-person weights: weights fitted separately on play 1 and play 2,
   Spearman correlation per weight, participant bootstrap.
3. Where the two readouts differ: per-rating log-likelihood difference (H_RM minus H_HE_mood) on
   held-out first plays, by the size of the unsigned prediction error since the previous rating
   and after outcome streaks.

Usage: python analysis/gamble_r5/play2.py [--threads 8]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis/joint_gamble"))
import joint_model as jm  # noqa: E402

torch.set_default_dtype(torch.float64)
T = jm.T
HALF = np.arange(T) < 15
CFG = {"H_RM": dict(readout="chan", mood=True), "H_HE_mood": dict(readout="he", mood=True)}
rs = np.random.RandomState(5)


def params(d):
    return {k: torch.tensor(v) for k, v in d.items() if not k.startswith("_")}


def hll_rows(D, cfg, P, dper=None):
    o = jm.forward(D, cfg, P, dper)
    return o["hll"].detach().numpy(), o["hll_pre"].detach().numpy()


def boot(x, nb=2000):
    x = np.asarray(x)
    m = [x[rs.randint(0, len(x), len(x))].mean() for _ in range(nb)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_ratio(num, den, nb=2000):
    num = np.asarray(num); den = np.asarray(den)
    vals = []
    for _ in range(nb):
        ix = rs.randint(0, len(num), len(num))
        vals.append(num[ix].sum() / den[ix].sum())
    return float(num.sum() / den.sum()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    f1 = np.load(ROOT / "analysis/joint_gamble/out/gbe_first_play.npz")
    f2 = np.load(HERE / "out/gbe_second_play.npz")
    N1 = len(f1["cert"])
    perm = np.random.RandomState(0).permutation(N1)
    folds = np.array_split(perm, 5)
    fold_of = np.empty(N1, int)
    for f, ix in enumerate(folds):
        fold_of[ix] = f
    rows2 = f2["row"].astype(int)
    res = {"n_second_play": int(len(rows2)), "sd_grid": {}, "transfer": {}, "reliability": {}, "where": {}}
    acc = {k: [] for k in ["p2_grp_RM", "p2_pw_RM", "p2_grp_HE", "p2_pw_HE", "nh2"]}
    W1 = {"H_RM": [], "H_HE_mood": []}; W2 = {"H_RM": [], "H_HE_mood": []}
    where = {"absP": [], "d": [], "streak": [], "pid": []}
    pid_off = 0
    t0 = time.time()
    for f in range(5):
        fj = json.loads((ROOT / f"analysis/joint_gamble/out/fold{f}.json").read_text())
        Pm = {m: params(fj["models"][m]["params"]) for m in CFG}
        # prior sd per model by split-half CV on a subsample of this fold's training participants
        tr = np.sort(np.concatenate([folds[i] for i in range(5) if i != f]))
        sub = np.sort(rs.choice(tr, 3000, replace=False))
        Dsub = jm.Data(f1, sub)
        sd = {}
        for m in CFG:
            best = None
            for s in [0.003, 0.01, 0.03, 0.1, 0.3]:
                dp = jm.fit_person_offsets(Dsub, CFG[m], Pm[m], {"w"}, {"w": s}, HALF)
                h, _ = hll_rows(Dsub, CFG[m], Pm[m], dp)
                ll2 = float((h * (~HALF)).sum() / (Dsub.hmask.numpy() * (~HALF)).sum())
                if best is None or ll2 > best[0]:
                    best = (ll2, s)
            sd[m] = best[1]
        res["sd_grid"][f] = sd
        # second-play participants held out in this fold
        sel = np.where(fold_of[rows2] == f)[0]
        D1 = jm.Data(f1, rows2[sel]); D2 = jm.Data(f2, sel)
        nh2 = D2.hmask.numpy().sum(1) + 1
        acc["nh2"].append(nh2)
        for m, tag in [("H_RM", "RM"), ("H_HE_mood", "HE")]:
            h, hp = hll_rows(D2, CFG[m], Pm[m])
            acc[f"p2_grp_{tag}"].append(h.sum(1) + hp)
            dp1 = jm.fit_person_offsets(D1, CFG[m], Pm[m], {"w"}, {"w": sd[m]}, np.ones(T, bool))
            h, hp = hll_rows(D2, CFG[m], Pm[m], dp1)
            acc[f"p2_pw_{tag}"].append(h.sum(1) + hp)
            dp2 = jm.fit_person_offsets(D2, CFG[m], Pm[m], {"w"}, {"w": sd[m]}, np.ones(T, bool))
            W1[m].append((Pm[m]["w"] + dp1["w"]).numpy()); W2[m].append((Pm[m]["w"] + dp2["w"]).numpy())
        # where the readouts differ, on the fold's held-out first plays
        te = np.sort(folds[f]); te = np.sort(rs.choice(te, 4000, replace=False))
        Dt = jm.Data(f1, te)
        hR, _ = hll_rows(Dt, CFG["H_RM"], Pm["H_RM"]); hH, _ = hll_rows(Dt, CFG["H_HE_mood"], Pm["H_HE_mood"])
        P = Dt.chan[:, :, 1].numpy(); out = Dt.out.numpy(); chose = Dt.chose.numpy(); mask = Dt.hmask.numpy()
        for i in range(Dt.N):
            sA = 0.0; signs = []
            for t in range(T):
                sA += abs(P[i, t])
                if chose[i, t] == 1 and P[i, t] != 0:
                    signs.append(np.sign(P[i, t]))
                if mask[i, t]:
                    st = len(signs) >= 3 and abs(sum(signs[-3:])) == 3
                    where["absP"].append(sA); where["d"].append(hR[i, t] - hH[i, t]); where["streak"].append(st); where["pid"].append(pid_off + i)
                    sA = 0.0
        pid_off += Dt.N
        print(f"fold {f} done ({time.time() - t0:.0f}s) sd={sd}", flush=True)
    for k in acc:
        acc[k] = np.concatenate(acc[k])
    nh = acc["nh2"]
    tr_ = res["transfer"]
    for tag in ["RM", "HE"]:
        tr_[f"play2_ll_per_rating_group_{tag}"] = float(acc[f"p2_grp_{tag}"].sum() / nh.sum())
        tr_[f"play2_ll_per_rating_person_{tag}"] = float(acc[f"p2_pw_{tag}"].sum() / nh.sum())
        tr_[f"person_minus_group_{tag}"] = boot_ratio(acc[f"p2_pw_{tag}"] - acc[f"p2_grp_{tag}"], nh)
    tr_["RM_minus_HE_group_weights"] = boot_ratio(acc["p2_grp_RM"] - acc["p2_grp_HE"], nh)
    tr_["RM_minus_HE_person_weights"] = boot_ratio(acc["p2_pw_RM"] - acc["p2_pw_HE"], nh)
    for m in CFG:
        a1 = np.concatenate(W1[m]); a2 = np.concatenate(W2[m])
        rel = {}
        names = ["forward", "present", "backward"] if m == "H_RM" else ["certain", "EV", "RPE"]
        for j, nm in enumerate(names):
            r0 = spearman(a1[:, j], a2[:, j])
            bs = []
            for _ in range(500):
                ix = rs.randint(0, len(a1), len(a1)); bs.append(spearman(a1[ix, j], a2[ix, j]))
            rel[nm] = {"spearman": r0, "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
        res["reliability"][m] = rel
    d = np.array(where["d"]); aP = np.array(where["absP"]); st = np.array(where["streak"]); pid = np.array(where["pid"])
    npid = pid.max() + 1

    def cboot(m, nb=1000):
        num = np.bincount(pid[m], weights=d[m], minlength=npid); den = np.bincount(pid[m], minlength=npid).astype(float)
        return boot_ratio(num, den, nb)
    qs = np.percentile(aP[aP > 0], [50, 75, 90])
    bins = {"no_gamble_since_last_rating": aP == 0, "below_median": (aP > 0) & (aP <= qs[0]),
            "median_to_75": (aP > qs[0]) & (aP <= qs[1]), "75_to_90": (aP > qs[1]) & (aP <= qs[2]), "top_10pct": aP > qs[2]}
    for k, m in bins.items():
        res["where"][k] = {"n": int(m.sum()), "RM_minus_HE_per_rating": cboot(m)}
    res["where"]["after_streak_of_3"] = {"n": int(st.sum()), "RM_minus_HE_per_rating": cboot(st)}
    res["where"]["not_after_streak"] = {"n": int((~st).sum()), "RM_minus_HE_per_rating": cboot(~st)}
    res["where"]["quantiles_unsigned_PE_50_75_90"] = qs.tolist()
    (HERE / "out/play2.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
