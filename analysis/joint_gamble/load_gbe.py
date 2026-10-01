"""Load the first completed play of every Rutledge GBE participant into padded arrays.

Output: out/gbe_first_play.npz with, per participant i and trial t (T=30):
  cert, win, lose  (points / 100)
  chose            1 = gamble, 0 = safe
  out              outcome (points / 100)
  hap_pre          the rating taken before trial 1 (0-1)
  hap              rating after trial t (0-1), NaN where no rating
  ttype            0 gain, 1 mixed, 2 loss
  age, design      participant covariates
Column conventions follow Rutledge_GBE_risk_data_code.m: the rating stored on row 1 is the
pre-task rating; ratings on later rows follow that row's outcome.
"""
from __future__ import annotations
import numpy as np
import scipy.io as sio
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAT = ROOT / "data_raw/rutledge_gbe/Rutledge_GBE_risk_data_TOD.mat"
OUT = Path(__file__).resolve().parent / "out/gbe_first_play.npz"
C_CERTAIN, C_WIN, C_LOSE, C_CHOSE, C_OUT, C_HAP = 2, 3, 4, 6, 7, 9
T = 30


def main():
    d = sio.loadmat(MAT, squeeze_me=True, struct_as_record=False)
    sd = d["subjData"]
    rows = {k: [] for k in ["cert", "win", "lose", "chose", "out", "hap", "hap_pre", "ttype", "age", "design", "id"]}
    skipped = 0
    for s in sd:
        dd = s.data
        p = dd[0] if (isinstance(dd, np.ndarray) and dd.dtype == object) else dd
        m = np.asarray(p, float)
        if m.ndim != 2 or m.shape[0] != T or m.shape[1] < 10:
            skipped += 1
            continue
        hap = m[:, C_HAP] / 100.0
        rated = np.where(~np.isnan(hap))[0]
        chose = m[:, C_CHOSE]
        if len(rated) < 6 or rated[0] != 0 or np.any(np.isnan(chose)) or np.nanstd(hap) < 1e-6:
            skipped += 1
            continue
        hap_pre = hap[0]
        h = hap.copy(); h[0] = np.nan          # row-1 rating is the pre-task rating
        cert = m[:, C_CERTAIN]
        tt = np.where(cert > 0, 0, np.where(cert == 0, 1, 2))
        rows["cert"].append(cert / 100); rows["win"].append(m[:, C_WIN] / 100)
        rows["lose"].append(m[:, C_LOSE] / 100); rows["chose"].append(chose)
        rows["out"].append(m[:, C_OUT] / 100); rows["hap"].append(h); rows["hap_pre"].append(hap_pre)
        rows["ttype"].append(tt); rows["age"].append(float(np.atleast_1d(s.age)[0])); rows["design"].append(float(np.atleast_1d(s.designVersion)[0]))
        rows["id"].append(float(np.atleast_1d(s.id)[0]))
    arr = {k: np.asarray(v, float) for k, v in rows.items()}
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, **arr)
    n = len(arr["cert"])
    print(f"participants kept {n}, skipped {skipped}; ratings after trials {np.sum(~np.isnan(arr['hap']))}")
    print(f"gamble rate {arr['chose'].mean():.3f}; by type gain/mixed/loss",
          [round(arr['chose'][arr['ttype'] == k].mean(), 3) for k in range(3)])


if __name__ == "__main__":
    main()
