"""Fair test of the model's generality: OUR POMDP's three channels vs the
happiness equation, on Rutledge GBE.

Fix vs rutledge_pomdp_validate.py: (1) feed graded reward (magnitude preserved via
the K-level valence observation + signed external feedback), not a coarse bin;
(2) read the model's THREE channels separately (v_model backward, v_reward present,
v_action forward) -- the model's own theoretical commitments -- rather than the
tanh-compressed composite; (3) integrate each channel with the same forgetting
(gamma) the happiness equation uses; (4) regress happiness on the discounted
channels, held out across subjects.

If the model is a general affect model, its channels (which include a reward-
prediction-error channel) should predict happiness at least as well as the
happiness equation (R2 ~ 0.146), not worse.
"""
from __future__ import annotations
import numpy as np
import scipy.io as sio
from generative_model import build_model
from agent import Agent

MAT = "data_raw/rutledge_gbe/Rutledge_GBE_risk_data_TOD.mat"
MAXSUBJ = 3000
GAMMA = 0.6
C_CERTAIN, C_WIN, C_LOSE, C_CHOSE, C_OUT, C_HAP = 2, 3, 4, 6, 7, 9
K, M = 8, 8


def r2(pred, y):
    y = np.asarray(y); sst = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - pred) ** 2) / sst if sst > 0 else np.nan


def fit_eval(train, test, ncol):
    Xtr = np.array([[1.0] + list(r[0]) for r in train]); ytr = np.array([r[1] for r in train])
    Xte = np.array([[1.0] + list(r[0]) for r in test]); yte = np.array([r[1] for r in test])
    mu = Xtr[:, 1:].mean(0); sd = Xtr[:, 1:].std(0) + 1e-9
    Xtr[:, 1:] = (Xtr[:, 1:] - mu) / sd; Xte[:, 1:] = (Xte[:, 1:] - mu) / sd
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return r2(Xte @ w, yte)


def main():
    print("loading Rutledge (~30s)...")
    d = sio.loadmat(MAT, squeeze_me=True, struct_as_record=False)
    sd = d["subjData"]
    rng = np.random.RandomState(1)
    idx = rng.permutation(len(sd))[:MAXSUBJ]
    model = build_model(K=K, M=M, pi_pos=3.0, omega_e=3.0, gamma=16.0,
                        c_pos=1.0, c_neg=1.0, neg_val_precision=1.0, valence_inertia=0.3)
    # channel rows (model) and reward-equation rows (baselines), same rated trials
    tr_ch, te_ch, tr_eq, te_eq, tr_rpe, te_rpe = [], [], [], [], [], []
    nsub = 0
    for k, i in enumerate(idx):
        s = sd[i]
        dd = s.data
        plays = dd if (isinstance(dd, np.ndarray) and dd.dtype == object) else [dd]
        rows_ch, rows_eq, rows_rpe = [], [], []
        for p in (plays if np.ndim(plays) > 0 else [plays]):
            m = np.asarray(p, float)
            if m.ndim != 2 or m.shape[1] < 10:
                continue
            n = m.shape[0]
            scale = max(1.0, np.nanmax(np.abs(m[:, [C_CERTAIN, C_WIN, C_LOSE, C_OUT]])))
            cert, win, lose = m[:, C_CERTAIN], m[:, C_WIN], m[:, C_LOSE]
            chose, out, hap = m[:, C_CHOSE], m[:, C_OUT], m[:, C_HAP]
            ev = 0.5 * (win + lose)
            CR = np.where(chose == 0, cert, 0.0)
            EV = np.where(chose == 1, ev, 0.0)
            RPE = np.where(chose == 1, out - ev, 0.0)
            agent = Agent(model, gamma=16.0, pi_pos=3.0, omega_e=3.0, valence_inertia=0.3,
                          counterfactual_horizon=1, adaptive_counterfactual_horizon=False, seed=k)
            vm = vr = va = 0.0  # discounted channel accumulators
            dcr = dev = drpe = 0.0
            rated = ~np.isnan(hap)
            if rated.sum() < 3 or np.nanstd(hap[rated]) < 1e-6:
                continue
            mu, sdv = np.nanmean(hap[rated]), np.nanstd(hap[rated])
            for t in range(n):
                o = out[t]
                oe = 0 if o < -0.05 * scale else (2 if o > 0.05 * scale else 1)
                ov = int(np.clip(round((o / scale + 1) / 2 * (K - 1)), 0, K - 1))
                _, info = agent.step([oe, 1, ov])
                vm = GAMMA * vm + info["v_model"]
                vr = GAMMA * vr + info["v_reward"]
                va = GAMMA * va + info["v_action"]
                dcr = GAMMA * dcr + CR[t]; dev = GAMMA * dev + EV[t]; drpe = GAMMA * drpe + RPE[t]
                if rated[t]:
                    hz = (hap[t] - mu) / sdv
                    rows_ch.append(([vm, vr, va], hz))
                    rows_eq.append(([dcr, dev, drpe], hz))
                    rows_rpe.append(([drpe], hz))
        if not rows_ch:
            continue
        nsub += 1
        bucket = 0 if (k % 10) < 7 else 1
        (tr_ch if bucket == 0 else te_ch).extend(rows_ch)
        (tr_eq if bucket == 0 else te_eq).extend(rows_eq)
        (tr_rpe if bucket == 0 else te_rpe).extend(rows_rpe)
    print(f"subjects used: {nsub}; train={len(tr_ch)}, test={len(te_ch)}")
    print("\n=== Rutledge happiness prediction, held out ===")
    print(f"  RPE only (single channel):                    R2 = {fit_eval(tr_rpe, te_rpe, 1):.4f}")
    print(f"  Happiness equation (CR+EV+RPE):               R2 = {fit_eval(tr_eq, te_eq, 3):.4f}")
    print(f"  OUR model 3 channels (v_model,v_reward,v_action): R2 = {fit_eval(tr_ch, te_ch, 3):.4f}")


if __name__ == "__main__":
    main()
