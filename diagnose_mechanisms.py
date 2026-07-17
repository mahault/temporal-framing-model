"""Diagnose WHY asymmetry (c_pos!=c_neg) and counterfactual depth show no
effect on the ESM valence-prediction test.

Hypothesis: both mechanisms act on POLICY SELECTION (the EFE / action
distribution), while the ESM test reads out policy-AVERAGED valence-level
dynamics. If pi barely moves between variants, or if the counterfactual
horizon rarely deepens under passive observation-driven filtering, the
mechanisms have no leverage on this particular target.

We measure, over real Geschwind sequences:
  1. mean selected-action distribution per variant
  2. L1 distance between the full and symmetric policy posteriors pi
  3. distribution of the adaptive counterfactual horizon actually used
  4. correlation of the predicted next-valence sequences across variants
"""

from __future__ import annotations

import numpy as np

from agent import Agent
from generative_model import (EPS, N_ACTIONS, N_FRAMES, ACTION_NAMES,
                              build_model)
from empirical_rebuild import load_participants, _bin_e, _bin_v, FULL


def _params(variant):
    p = dict(FULL)
    horizon, adaptive = 2, True
    if variant == "symmetric":
        p.update(c_pos=1.0, c_neg=1.0, neg_val_precision=1.0)
    elif variant == "one_step":
        horizon, adaptive = 1, False
    return p, horizon, adaptive


def drive_diag(seq, variant, seed):
    p, horizon, adaptive = _params(variant)
    K, M = p["K"], p["M"]
    model = build_model(K=K, M=M, pi_pos=p["pi_pos"], omega_e=p["omega_e"],
                        gamma=p["gamma"], c_pos=p["c_pos"], c_neg=p["c_neg"],
                        neg_val_precision=p["neg_val_precision"],
                        valence_inertia=p["valence_inertia"])
    agent = Agent(model, gamma=p["gamma"], pi_pos=p["pi_pos"],
                  omega_e=p["omega_e"], c_pos=p["c_pos"], c_neg=p["c_neg"],
                  neg_val_precision=p["neg_val_precision"],
                  valence_inertia=p["valence_inertia"],
                  counterfactual_horizon=horizon,
                  adaptive_counterfactual_horizon=adaptive,
                  max_counterfactual_horizon=3, seed=seed)
    v_axis = np.arange(K)
    pis, horizons, preds = [], [], []
    for beep in seq:
        obs = [_bin_e(beep["e"]), 1, _bin_v(beep["v"], K)]
        action, info = agent.step(obs)
        pis.append(info["pi"].copy())
        horizons.append(info.get("counterfactual_horizon", horizon))
        pi = info["pi"]
        B = sum(pi[a] * model.B[a] for a in range(len(pi)))
        q = B @ info["beliefs"]
        q = np.maximum(q, EPS); q /= q.sum()
        vm = q.reshape(K, M, N_FRAMES).sum(axis=(1, 2))
        preds.append(float(vm @ v_axis / max(K - 1, 1)))
    return np.array(pis), np.array(horizons), np.array(preds)


def main():
    parts = load_participants()
    pids = sorted(parts)[:30]
    variants = ["full", "symmetric", "one_step"]
    acc = {v: {"pi": [], "hz": [], "pred": []} for v in variants}
    # align predictions across variants per participant
    pred_by_variant = {v: [] for v in variants}
    for i, pid in enumerate(pids):
        seq = parts[pid]
        per = {}
        for v in variants:
            pis, hz, pred = drive_diag(seq, v, seed=100 + i)
            acc[v]["pi"].append(pis)
            acc[v]["hz"].append(hz)
            per[v] = pred
        n = min(len(per[v]) for v in variants)
        for v in variants:
            pred_by_variant[v].append(per[v][:n])

    print("=== Mean selected-policy distribution (pi) per variant ===")
    hdr = "  ".join(f"{a:>10}" for a in ACTION_NAMES)
    print(f"{'variant':>12}  {hdr}")
    mean_pi = {}
    for v in variants:
        allpi = np.vstack(acc[v]["pi"])
        mp = allpi.mean(axis=0)
        mean_pi[v] = mp
        print(f"{v:>12}  " + "  ".join(f"{x:10.3f}" for x in mp))

    print("\n=== Policy divergence full vs symmetric ===")
    # per-timestep L1 between full and symmetric pi (participant-matched)
    l1s = []
    for pf, ps in zip(acc["full"]["pi"], acc["symmetric"]["pi"]):
        n = min(len(pf), len(ps))
        l1s.append(np.abs(pf[:n] - ps[:n]).sum(axis=1))
    l1 = np.concatenate(l1s)
    print(f"mean per-step L1(pi_full, pi_symmetric) = {l1.mean():.4f} "
          f"(0 = identical, 2 = disjoint); max = {l1.max():.4f}")
    print(f"mean L1(mean_pi_full, mean_pi_symmetric) = "
          f"{np.abs(mean_pi['full'] - mean_pi['symmetric']).sum():.4f}")

    print("\n=== Adaptive counterfactual horizon actually used (full) ===")
    hz = np.concatenate(acc["full"]["hz"])
    vals, counts = np.unique(hz, return_counts=True)
    for val, c in zip(vals, counts):
        print(f"  horizon {int(val)}: {100*c/len(hz):5.1f}%")

    print("\n=== Predicted-valence sequence agreement across variants ===")
    def corr(a, b):
        a = np.concatenate(a); b = np.concatenate(b)
        n = min(len(a), len(b)); a, b = a[:n], b[:n]
        return float(np.corrcoef(a, b)[0, 1]), float(np.sqrt(np.mean((a-b)**2)))
    r_fs, rmse_fs = corr(pred_by_variant["full"], pred_by_variant["symmetric"])
    r_fo, rmse_fo = corr(pred_by_variant["full"], pred_by_variant["one_step"])
    print(f"full vs symmetric: r = {r_fs:.4f}, RMSE = {rmse_fs:.5f}")
    print(f"full vs one_step : r = {r_fo:.4f}, RMSE = {rmse_fo:.5f}")


if __name__ == "__main__":
    main()
