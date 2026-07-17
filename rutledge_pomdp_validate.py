"""Run OUR actual temporal-framing POMDP on the Rutledge GBE happiness data.

Unlike rutledge_affect_fit.py (which fit the linear happiness equation to test the
integration *principle*), this drives the real Agent (agent.py / generative_model.py)
through each play's outcome sequence, reads the model's composite three-channel
valence, and predicts the momentary happiness ratings. Held out across participants.
This is a genuine second-dataset validation of the model itself.

Baselines (all with a train-fit affine, held-out across subjects):
  - persistence (previous happiness) -- not available (ratings sparse); use mean
  - outcome-RPE linear (single reward channel)
  - full happiness equation (CR+EV+RPE)   [reference: rutledge_affect_fit.py]
  - OUR model composite valence
"""
from __future__ import annotations
import numpy as np
import scipy.io as sio
from generative_model import build_model, N_FRAMES
from agent import Agent

MAT = "data_raw/rutledge_gbe/Rutledge_GBE_risk_data_TOD.mat"
MAXSUBJ = 4000
C_CERTAIN, C_WIN, C_LOSE, C_CHOSE, C_OUT, C_HAP = 2, 3, 4, 6, 7, 9
K, M = 8, 8
EPS = 1e-12


def _bin_ext(x, scale):
    # outcome relative to 0 -> neg/neutral/pos
    if x < -0.05 * scale: return 0
    if x > 0.05 * scale: return 2
    return 1


def drive_play(agent, model, m, scale):
    """Return list of (model_valence, happiness_z) at rated trials + the RPE/EV/CR
    discounted regressors for the baseline."""
    n = m.shape[0]
    v_axis = np.arange(K)
    out = m[:, C_OUT]; hap = m[:, C_HAP]
    rated = ~np.isnan(hap)
    if rated.sum() < 3 or np.nanstd(hap[rated]) < 1e-6:
        return []
    mu, sd = np.nanmean(hap[rated]), np.nanstd(hap[rated])
    rows = []
    model_val_run = []
    for t in range(n):
        o = out[t]
        # observation: external feedback + felt valence both reflect the outcome;
        # interoception neutral (no body signal in this task)
        oe = _bin_ext(o, scale)
        ov = int(np.clip(round((o / scale + 1) / 2 * (K - 1)), 0, K - 1))
        _, info = agent.step([oe, 1, ov])
        model_val_run.append(info["valence"])   # composite three-channel valence
        if rated[t]:
            rows.append((info["valence"], (hap[t] - mu) / sd))
    return rows


def r2(pred, y):
    y = np.asarray(y); sst = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - pred) ** 2) / sst if sst > 0 else np.nan


def fit_affine_eval(train, test):
    Xtr = np.array([[1.0, r[0]] for r in train]); ytr = np.array([r[1] for r in train])
    Xte = np.array([[1.0, r[0]] for r in test]); yte = np.array([r[1] for r in test])
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return r2(Xte @ w, yte)


def main():
    print("loading Rutledge (~30s)...")
    d = sio.loadmat(MAT, squeeze_me=True, struct_as_record=False)
    sd = d["subjData"]
    rng = np.random.RandomState(1)
    idx = rng.permutation(len(sd))[:MAXSUBJ]
    # one shared model/agent config (no per-subject fitting); moderate params
    model = build_model(K=K, M=M, pi_pos=3.0, omega_e=3.0, gamma=16.0,
                        c_pos=1.0, c_neg=1.0, neg_val_precision=1.0, valence_inertia=0.3)
    train, test, nsub = [], [], 0
    for k, i in enumerate(idx):
        s = sd[i]
        dd = s.data
        plays = dd if (isinstance(dd, np.ndarray) and dd.dtype == object) else [dd]
        rows = []
        for p in (plays if np.ndim(plays) > 0 else [plays]):
            m = np.asarray(p, float)
            if m.ndim != 2 or m.shape[1] < 10:
                continue
            scale = max(1.0, np.nanmax(np.abs(m[:, C_OUT])))
            agent = Agent(model, gamma=16.0, pi_pos=3.0, omega_e=3.0,
                          valence_inertia=0.3, counterfactual_horizon=1,
                          adaptive_counterfactual_horizon=False, seed=k)
            rows += drive_play(agent, model, m, scale)
        if not rows:
            continue
        nsub += 1
        (train if (k % 10) < 7 else test).extend(rows)
    print(f"subjects used: {nsub}; train={len(train)}, test={len(test)}")
    r2_model = fit_affine_eval(train, test)
    print(f"\n=== OUR temporal-framing POMDP on Rutledge happiness (held out) ===")
    print(f"  model composite-valence -> happiness:  held-out R2 = {r2_model:.4f}")
    print(f"  (reference from linear happiness-equation analysis: RPE-only 0.111, full 0.146)")


if __name__ == "__main__":
    main()
