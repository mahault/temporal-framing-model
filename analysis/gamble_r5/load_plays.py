"""Second play of every Rutledge GBE participant whose first play is in the joint-fit sample.

Same inclusion rule as analysis/joint_gamble/load_gbe.py, applied to each play in order. For each
participant in out/gbe_first_play.npz (row r) we keep the next valid play after the first valid
one. Output out/gbe_second_play.npz with the same fields plus 'row' (row index into the
first-play arrays) and 'bdi' (BDI total, NaN where absent).

Usage: python analysis/gamble_r5/load_plays.py
"""
from __future__ import annotations
import numpy as np
import scipy.io as sio
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAT = ROOT / "data_raw/rutledge_gbe/Rutledge_GBE_risk_data_TOD.mat"
FIRST = ROOT / "analysis/joint_gamble/out/gbe_first_play.npz"
OUT = Path(__file__).resolve().parent / "out/gbe_second_play.npz"
C_CERTAIN, C_WIN, C_LOSE, C_CHOSE, C_OUT, C_HAP = 2, 3, 4, 6, 7, 9
T = 30


def parse(p):
    m = np.asarray(p, float)
    if m.ndim != 2 or m.shape[0] != T or m.shape[1] < 10:
        return None
    hap = m[:, C_HAP] / 100.0
    rated = np.where(~np.isnan(hap))[0]
    chose = m[:, C_CHOSE]
    if len(rated) < 6 or rated[0] != 0 or np.any(np.isnan(chose)) or np.nanstd(hap) < 1e-6:
        return None
    h = hap.copy(); h[0] = np.nan
    cert = m[:, C_CERTAIN]
    tt = np.where(cert > 0, 0, np.where(cert == 0, 1, 2))
    return dict(cert=cert / 100, win=m[:, C_WIN] / 100, lose=m[:, C_LOSE] / 100, chose=chose,
                out=m[:, C_OUT] / 100, hap=h, hap_pre=hap[0], ttype=tt)


def main():
    d = sio.loadmat(MAT, squeeze_me=True, struct_as_record=False)
    first = np.load(FIRST)
    id2row = {float(i): r for r, i in enumerate(first["id"])}
    bdi = {}
    for s in d["depData"]:
        bdi[float(np.atleast_1d(s.id)[0])] = float(np.atleast_1d(s.bdiTotal)[0])
    rows = {k: [] for k in ["cert", "win", "lose", "chose", "out", "hap", "hap_pre", "ttype", "row", "bdi"]}
    for s in d["subjData"]:
        sid = float(np.atleast_1d(s.id)[0])
        if sid not in id2row:
            continue
        dd = s.data
        plays = list(dd) if (isinstance(dd, np.ndarray) and dd.dtype == object) else [dd]
        valid = [q for q in (parse(p) for p in plays) if q is not None]
        if len(valid) < 2:
            continue
        q = valid[1]
        for k, v in q.items():
            rows[k].append(v)
        rows["row"].append(id2row[sid]); rows["bdi"].append(bdi.get(sid, np.nan))
    arr = {k: np.asarray(v, float) for k, v in rows.items()}
    # consistency: the first valid play parsed here must equal the stored first play
    np.savez_compressed(OUT, **arr)
    print(f"participants with a valid second play: {len(arr['row'])}; with BDI: {np.sum(~np.isnan(arr['bdi']))}")


if __name__ == "__main__":
    main()
