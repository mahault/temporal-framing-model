"""Direction of the BDI effect on the backward channel, analytically and model-free.

Analytic: in analysis/joint_gamble/joint_model.py the backward channel is B_t = -|P_t| with P_t the
reward prediction error, and it enters instantaneous affect as w_B B_t = -w_B |P_t|. The group
w_B is positive (about +0.017 to +0.019 in every fold), so a surprise of either sign lowers
happiness by w_B |P_t|. In analysis/gamble_bdi/analyze_bdi.py BDI enters additively,
w_B,i = w_B,group + beta_BDI z_BDI,i. A negative beta_BDI therefore makes w_B smaller in more
depressed people: they lose LESS happiness after a surprise.

Model-free check: for every happiness rating, the change since the previous rating is regressed on
the summed signed prediction error, the summed unsigned prediction error and the summed expected
value of the chosen options over the trials since that previous rating, with interactions of each
with z-scored BDI. Participant cluster bootstrap. A positive unsigned x BDI interaction means more
depressed participants lose less happiness after surprise.

Usage: python analysis/gamble_r5/bdi_sign.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
rs = np.random.RandomState(11)


def design(npz, rows):
    cert, win, lose = npz["cert"][rows], npz["win"][rows], npz["lose"][rows]
    chose, out, hap, hpre = npz["chose"][rows], npz["out"][rows], npz["hap"][rows], npz["hap_pre"][rows]
    evg = 0.5 * (win + lose)
    P = np.where(chose == 1, out - evg, 0.0)
    F = np.where(chose == 1, evg, cert)
    recs = []
    for i in range(len(rows)):
        prev = hpre[i]; sP = sA = sF = 0.0; n = 0
        for t in range(hap.shape[1]):
            sP += P[i, t]; sA += abs(P[i, t]); sF += F[i, t]; n += 1
            if not np.isnan(hap[i, t]):
                recs.append((i, hap[i, t] - prev, sP, sA, sF, n))
                prev = hap[i, t]; sP = sA = sF = 0.0; n = 0
    return np.array(recs)


def ols(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def main():
    npz = np.load(ROOT / "analysis/joint_gamble/out/gbe_first_play.npz")
    pp = np.load(ROOT / "analysis/gamble_bdi/out/person_params.npz")
    rows = pp["row"].astype(int); bdi = pp["bdi"]
    zb = (bdi - bdi.mean()) / bdi.std()
    R = design(npz, rows)
    pid = R[:, 0].astype(int)
    y = R[:, 1]; sP, sA, sF = R[:, 2], R[:, 3], R[:, 4]
    z = zb[pid]
    names = ["const", "signed_PE", "unsigned_PE", "chosen_EV", "BDI", "signed_PE_x_BDI", "unsigned_PE_x_BDI", "chosen_EV_x_BDI"]
    X = np.column_stack([np.ones_like(y), sP, sA, sF, z, sP * z, sA * z, sF * z])
    b = ols(y, X)
    # participant cluster bootstrap
    groups = [np.where(pid == i)[0] for i in range(len(rows))]
    B = []
    for r in range(2000):
        ix = rs.randint(0, len(rows), len(rows))
        idx = np.concatenate([groups[i] for i in ix])
        B.append(ols(y[idx], X[idx]))
    B = np.array(B)
    lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
    se = B.std(0)
    res = {"n_participants": int(len(rows)), "n_ratings": int(len(y)),
           "coef": {n: {"b": float(b[j]), "lo": float(lo[j]), "hi": float(hi[j]),
                        "z": float(b[j] / se[j])} for j, n in enumerate(names)}}
    # implied happiness change after one unit (100 points) of unsigned surprise at BDI z = -1, 0, +1
    res["unsigned_effect_at_bdi_z"] = {str(zz): float(b[2] + b[6] * zz) for zz in (-1, 0, 1)}
    # tertile split, model-free and assumption-light: mean rating change per unit unsigned PE
    t1, t2 = np.percentile(bdi, [33.3, 66.7])
    tert = {}
    for lab, m in [("low", bdi <= t1), ("high", bdi > t2)]:
        sel = m[pid]
        bb = ols(y[sel], X[sel][:, :4])
        tert[lab] = {"n": int(m.sum()), "unsigned_PE_slope": float(bb[2]), "signed_PE_slope": float(bb[1])}
    res["tertiles"] = tert
    # whole 46k first-play sample: does unsigned surprise lower happiness at all (model-free)?
    allrows = np.arange(len(npz["cert"]))
    sub = np.sort(rs.choice(allrows, 12000, replace=False))
    R2 = design(npz, sub)
    X2 = np.column_stack([np.ones(len(R2)), R2[:, 2], R2[:, 3], R2[:, 4]])
    b2 = ols(R2[:, 1], X2)
    res["all_participants_subsample_12000"] = {"signed_PE": float(b2[1]), "unsigned_PE": float(b2[2]), "chosen_EV": float(b2[3])}
    # same model-free check on the second play of BDI participants (independent behavioural sample)
    f2 = np.load(HERE / "out/gbe_second_play.npz")
    has = ~np.isnan(f2["bdi"])
    idx2 = np.where(has)[0]; b2 = f2["bdi"][idx2]; zb2 = (b2 - bdi.mean()) / bdi.std()
    R3 = design(f2, idx2)
    pid3 = R3[:, 0].astype(int); y3 = R3[:, 1]; z3 = zb2[pid3]
    X3 = np.column_stack([np.ones_like(y3), R3[:, 2], R3[:, 3], R3[:, 4], z3, R3[:, 2] * z3, R3[:, 3] * z3, R3[:, 4] * z3])
    b3 = ols(y3, X3)
    g3 = [np.where(pid3 == i)[0] for i in range(len(idx2))]
    B3 = []
    for r in range(2000):
        ix = rs.randint(0, len(idx2), len(idx2)); ii = np.concatenate([g3[i] for i in ix]); B3.append(ols(y3[ii], X3[ii]))
    B3 = np.array(B3); lo3, hi3 = np.percentile(B3, [2.5, 97.5], axis=0)
    res["play2"] = {"n_participants": int(len(idx2)), "n_ratings": int(len(y3)),
                    "coef": {n: {"b": float(b3[j]), "lo": float(lo3[j]), "hi": float(hi3[j])} for j, n in enumerate(names)}}
    # analytic: group w_B per fold
    wb = [json.loads((ROOT / f"analysis/joint_gamble/out/fold{f}.json").read_text())["models"]["J"]["params"]["w"][2]
          for f in range(5)]
    res["group_wB_by_fold_model_J"] = wb
    s = json.loads((ROOT / "analysis/gamble_bdi/out/summary.json").read_text())
    res["covariate_model_bdi_on_wB"] = s["covariate_model"]["wB"]["BDI"]
    (HERE / "out/bdi_sign.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
