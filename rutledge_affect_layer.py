"""Affect as a readout layer over a task-specific generative model (Rutledge GBE).

The three affect channels are general operators on ANY active-inference agent's
inference dynamics. Here we put them on top of the *gamble* task-model (the agent
whose policies are {safe, gamble} and whose preferences are increasing in reward):

  forward (EFE-based)  = anticipated value of the chosen option = E[U|chosen]
                         (= EV for a gamble, = certain value for safe).
                         This is -risk(G_chosen): expected free energy natively
                         computes the chosen option's expected value.
  present  (reward PE)  = U(outcome) - E[U|chosen]  (= RPE for gambles, 0 for safe)
  backward (VFE change) = surprise at the outcome, -|RPE| (worse fit -> neg valence)

No EV is injected as a raw regressor; each channel is DERIVED from the task-model's
VFE/EFE by its standard definition. If the three channels recover / beat the
Rutledge happiness equation (CR+EV+RPE), the affect-readout layer is general: it
subsumes the happiness equation as a special case.

Utility is linear in reward (raw points), matching the happiness equation's units.
Channels integrated with forgetting gamma=0.6; happiness z-scored per subject;
held out across subjects.
"""
from __future__ import annotations
import numpy as np
import scipy.io as sio

MAT = "data_raw/rutledge_gbe/Rutledge_GBE_risk_data_TOD.mat"
MAXSUBJ = 15000
GAMMA = 0.6
C_CERTAIN, C_WIN, C_LOSE, C_CHOSE, C_OUT, C_HAP = 2, 3, 4, 6, 7, 9


def r2(pred, y):
    y = np.asarray(y); sst = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - pred) ** 2) / sst if sst > 0 else np.nan


def fit_eval(train, test, cols):
    Xtr = np.array([[1.0] + [r[0][c] for c in cols] for r in train]); ytr = np.array([r[1] for r in train])
    Xte = np.array([[1.0] + [r[0][c] for c in cols] for r in test]); yte = np.array([r[1] for r in test])
    mu = Xtr[:, 1:].mean(0); sd = Xtr[:, 1:].std(0) + 1e-9
    Xtr[:, 1:] = (Xtr[:, 1:] - mu) / sd; Xte[:, 1:] = (Xte[:, 1:] - mu) / sd
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return r2(Xte @ w, yte)


def build_rows(m):
    n = m.shape[0]
    cert, win, lose = m[:, C_CERTAIN], m[:, C_WIN], m[:, C_LOSE]
    chose, out, hap = m[:, C_CHOSE], m[:, C_OUT], m[:, C_HAP]
    ev = 0.5 * (win + lose)
    # task-model affect channels (linear utility U(x)=x)
    E_U = np.where(chose == 1, ev, cert)          # forward: anticipated value of chosen option
    RPE = np.where(chose == 1, out - ev, 0.0)     # present: reward prediction error
    surprise = -np.abs(RPE)                        # backward: VFE / poor-fit signal
    # happiness-equation regressors (for comparison)
    CR = np.where(chose == 0, cert, 0.0)
    EVc = np.where(chose == 1, ev, 0.0)
    def disc(x):
        acc = np.zeros(n); run = 0.0
        for t in range(n):
            run = GAMMA * run + x[t]; acc[t] = run
        return acc
    dfwd, dpres, dback = disc(E_U), disc(RPE), disc(surprise)
    dCR, dEV, dRPE = disc(CR), disc(EVc), disc(RPE)
    rated = ~np.isnan(hap)
    if rated.sum() < 3 or np.nanstd(hap[rated]) < 1e-6:
        return []
    mu, sd = np.nanmean(hap[rated]), np.nanstd(hap[rated])
    rows = []
    for t in np.where(rated)[0]:
        # feature vector: [0]fwd [1]present [2]backward | [3]CR [4]EV [5]RPE
        feats = [dfwd[t], dpres[t], dback[t], dCR[t], dEV[t], dRPE[t]]
        rows.append((feats, (hap[t] - mu) / sd))
    return rows


def main():
    print("loading Rutledge (~30s)...")
    d = sio.loadmat(MAT, squeeze_me=True, struct_as_record=False)
    sd = d["subjData"]
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(sd))[:MAXSUBJ]
    train, test, nsub = [], [], 0
    for k, i in enumerate(idx):
        s = sd[i]; dd = s.data
        plays = dd if (isinstance(dd, np.ndarray) and dd.dtype == object) else [dd]
        rows = []
        for p in (plays if np.ndim(plays) > 0 else [plays]):
            m = np.asarray(p, float)
            if m.ndim == 2 and m.shape[1] >= 10:
                rows += build_rows(m)
        if not rows:
            continue
        nsub += 1
        (train if (k % 10) < 7 else test).extend(rows)
    print(f"subjects used: {nsub}; train={len(train)}, test={len(test)}\n")
    print("=== FAIR head-to-head: each predecessor = one channel, over the same task-model ===")
    print("  (held out across subjects; feature cols: 0=forward/Hesp 1=present/Pattisapu 2=backward/Joffily)")
    print(f"  Joffily  (backward channel only, -dF/dt):         R2 = {fit_eval(train,test,[2]):.4f}")
    print(f"  Pattisapu(present channel only, RPE):             R2 = {fit_eval(train,test,[1]):.4f}")
    print(f"  Hesp     (forward channel only, EFE anticipation):R2 = {fit_eval(train,test,[0]):.4f}")
    print(f"  Pattisapu+Hesp (present+forward):                 R2 = {fit_eval(train,test,[0,1]):.4f}")
    print(f"  OURS     (all three channels):                    R2 = {fit_eval(train,test,[0,1,2]):.4f}")
    print(f"  [ref] happiness equation (CR + EV + RPE):         R2 = {fit_eval(train,test,[3,4,5]):.4f}")


if __name__ == "__main__":
    main()
