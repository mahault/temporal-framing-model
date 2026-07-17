"""Does counterfactual/regret improve AFFECT prediction? (Rutledge GBE happiness data)

Task: safe vs risky gamble, momentary happiness rated every 2-3 trials.
47,067 participants, ~1.1M happiness ratings. Data: Rutledge GBE (Dryad, CC0).

The Rutledge (2014) happiness equation models momentary happiness as a weighted
sum of forgetting-discounted reward terms:
  CR  = certain reward received (safe chosen)
  EV  = expected value of chosen gamble
  RPE = reward prediction error = outcome - EV (gamble chosen)

We add a COUNTERFACTUAL / regret term:
  CF  = obtained - foregone = outcome - certainValue   (when the gamble was chosen;
        >0 relief, <0 regret) -- the comparison to the safe option that was given up.
CF is distinct from RPE (RPE compares to the gamble's own expectation; CF compares
to the foregone alternative), so it is the counterfactual-emotion signal.

Nested models (each adds a term; all share forgetting gamma):
  M1 RPE-only            (single prediction-error channel; 'insufficient')
  M2 CR+EV+RPE           (full Rutledge reward model)
  M3 CR+EV+RPE+CF        (+ counterfactual/regret)
Held-out across SUBJECTS (fit on train subjects, predict test subjects' ratings).
If M3 > M2 out-of-sample, counterfactual improves AFFECT prediction (Q1);
if M2 > M1, the reduced sub-model is insufficient (Q2).
"""
from __future__ import annotations
import numpy as np
import scipy.io as sio
from pathlib import Path

MAT = "data_raw/rutledge_gbe/Rutledge_GBE_risk_data_TOD.mat"
GAMMA = 0.6           # Rutledge forgetting factor
MAXSUBJ = 15000       # subset for speed (still ~350k ratings)
# column indices (0-based) from dataHdr
C_CERTAIN, C_WIN, C_LOSE, C_CHOSE, C_OUT, C_HAP = 2, 3, 4, 6, 7, 9


def build_rows(play):
    """Return list of (features, happiness_z) for each rated trial in a play."""
    m = np.asarray(play, float)
    if m.ndim != 2 or m.shape[1] < 10:
        return []
    n = m.shape[0]
    certain, win, lose = m[:, C_CERTAIN], m[:, C_WIN], m[:, C_LOSE]
    chose, out, hap = m[:, C_CHOSE], m[:, C_OUT], m[:, C_HAP]
    ev_full = 0.5 * (win + lose)
    CR = np.where(chose == 0, certain, 0.0)
    EV = np.where(chose == 1, ev_full, 0.0)
    RPE = np.where(chose == 1, out - ev_full, 0.0)
    CF = np.where(chose == 1, out - certain, 0.0)     # counterfactual/regret
    # forgetting-discounted cumulative sums up to each trial
    def discounted(x):
        acc = np.zeros(n); run = 0.0
        for t in range(n):
            run = GAMMA * run + x[t]
            acc[t] = run
        return acc
    dCR, dEV, dRPE, dCF = discounted(CR), discounted(EV), discounted(RPE), discounted(CF)
    hz = hap.copy()
    rated = ~np.isnan(hz)
    if rated.sum() < 3 or np.nanstd(hz) < 1e-6:
        return []
    mu, sd = np.nanmean(hz), np.nanstd(hz)
    rows = []
    for t in np.where(rated)[0]:
        rows.append(([dCR[t], dEV[t], dRPE[t], dCF[t]], (hz[t] - mu) / sd))
    return rows


def r2(pred, y):
    y = np.asarray(y); sst = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - pred) ** 2) / sst if sst > 0 else np.nan


def fit_eval(train, test, cols):
    Xtr = np.array([[1.0] + [r[0][c] for c in cols] for r in train])
    ytr = np.array([r[1] for r in train])
    Xte = np.array([[1.0] + [r[0][c] for c in cols] for r in test])
    yte = np.array([r[1] for r in test])
    # standardize regressors (not intercept) using train stats
    mu = Xtr[:, 1:].mean(0); sd = Xtr[:, 1:].std(0) + 1e-9
    Xtr[:, 1:] = (Xtr[:, 1:] - mu) / sd; Xte[:, 1:] = (Xte[:, 1:] - mu) / sd
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return r2(Xte @ w, yte), r2(Xtr @ w, ytr)


def main():
    print("loading Rutledge GBE (~30s)...")
    d = sio.loadmat(MAT, squeeze_me=True, struct_as_record=False)
    sd = d["subjData"]
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(sd))[:MAXSUBJ]
    train_rows, test_rows, nsub = [], [], 0
    for k, i in enumerate(idx):
        s = sd[i]
        dd = s.data
        plays = dd if (isinstance(dd, np.ndarray) and dd.dtype == object) else [dd]
        subj_rows = []
        for p in (plays if np.ndim(plays) > 0 else [plays]):
            subj_rows += build_rows(p)
        if not subj_rows:
            continue
        nsub += 1
        (train_rows if (k % 10) < 7 else test_rows).extend(subj_rows)
    print(f"subjects used: {nsub}; train ratings={len(train_rows)}, test ratings={len(test_rows)}")

    models = {
        "M1  RPE only (single channel)":            [2],
        "M2  CR+EV+RPE (full Rutledge reward)":     [0, 1, 2],
        "M3  CR+EV+RPE+CF (+counterfactual)":       [0, 1, 2, 3],
        "  CF only (counterfactual alone)":          [3],
    }
    print(f"\n{'model':<40}{'held-out R2':>14}{'train R2':>12}")
    res = {}
    for name, cols in models.items():
        ho, tr = fit_eval(train_rows, test_rows, cols)
        res[name] = ho
        print(f"{name:<40}{ho:>14.4f}{tr:>12.4f}")
    m1 = res["M1  RPE only (single channel)"]
    m2 = res["M2  CR+EV+RPE (full Rutledge reward)"]
    m3 = res["M3  CR+EV+RPE+CF (+counterfactual)"]
    print(f"\n  Q2 full reward vs single channel:  +{100*(m2-m1)/abs(m1):.1f}% R2 (M2>M1 => single insufficient)")
    print(f"  Q1 counterfactual gain (M3 vs M2): {m3-m2:+.4f} R2 ({100*(m3-m2)/abs(m2):+.1f}%)  <-- does CF help AFFECT?")


if __name__ == "__main__":
    main()
