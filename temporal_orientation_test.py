"""Direct temporal-orientation test (the paper's named 'missing keystone').

Dataset: Mulholland et al. 2023, Consciousness & Cognition (Mendeley
10.17632/zpmm72bg6s.1, CC BY-NC). N=101, 1,458 per-probe mDES rows. Each probe
records momentary valence (dimension_emotion, 1-5) AND the temporal orientation of
thought (dimension_past, dimension_future, 1-5) -- an independent, per-beep report
of the very frame variable the model infers.

Two non-circular tests:

  T1 (keystone) -- does the model's latent temporal-frame belief track the
     INDEPENDENTLY REPORTED temporal orientation? Drive the model on valence only;
     the reported past/future orientation is never given to it. Correlate the
     model's posterior past-frame belief with reported past-orientation, and its
     future-frame belief with reported future-orientation. Reported both pooled and
     within-person (participant-demeaned), since pooled mixes between/within variance.

  T2 (theory sign check) -- rumination/worry theory predicts past-orientation
     associates with negative valence. Does the model reproduce the sign of the
     reported past~valence and future~valence relationships?

Also a per-channel readout comparison: which of the three valence channels
(v_model backward, v_reward present, v_action forward), read concurrently while the
model is driven, best matches reported momentary valence -- the head-to-head done on
a temporal-structure dataset rather than the static Rutledge gamble.
"""
from __future__ import annotations
import csv
import numpy as np
from collections import defaultdict
from agent import Agent
from generative_model import EPS, N_FRAMES, build_model

DATA = "data_raw/mulholland2023/conscious_cogn_data_mendeley.csv"
FITTED = dict(pi_pos=2.0, valence_inertia=0.5, omega_e=5.0, c_pos=1.0, c_neg=1.0)


def _f(x):
    try:
        v = float(x); return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def load():
    rows = list(csv.DictReader(open(DATA, encoding="latin-1")))
    parts = defaultdict(list)
    for r in rows:
        p, fu, e = _f(r["dimension_past"]), _f(r["dimension_future"]), _f(r["dimension_emotion"])
        t = _f(r["activity_start_time"])
        if None in (p, fu, e):
            continue
        parts[r["secret_user_id"]].append(dict(t=t if t is not None else 0.0,
                                               past=p, future=fu, v=e))
    for pid in parts:
        parts[pid].sort(key=lambda d: d["t"])
    return parts


def _bin_v(v_norm, K):
    return int(np.clip(round(v_norm * (K - 1)), 0, K - 1))


def drive(seq, seed):
    """Drive model on valence (event neutral). Return per-beep frame beliefs + channels."""
    K = M = 8
    model = build_model(K=K, M=M, pi_pos=FITTED["pi_pos"], omega_e=FITTED["omega_e"],
                        gamma=16.0, c_pos=FITTED["c_pos"], c_neg=FITTED["c_neg"],
                        neg_val_precision=1.0, valence_inertia=FITTED["valence_inertia"])
    agent = Agent(model, gamma=16.0, pi_pos=FITTED["pi_pos"], omega_e=FITTED["omega_e"],
                  c_pos=FITTED["c_pos"], c_neg=FITTED["c_neg"], neg_val_precision=1.0,
                  valence_inertia=FITTED["valence_inertia"], counterfactual_horizon=1,
                  adaptive_counterfactual_horizon=False, seed=seed)
    out = []
    for beep in seq:
        vn = (beep["v"] - 1.0) / 4.0                      # 1..5 -> 0..1
        _, info = agent.step([1, 1, _bin_v(vn, K)])       # event neutral
        fr = info["beliefs"].reshape(K, M, N_FRAMES).sum(axis=(0, 1))
        out.append(dict(f_past=fr[0], f_pres=fr[1], f_fut=fr[2],
                        v_model=info["v_model"], v_reward=info["v_reward"],
                        v_action=info["v_action"]))
    return out


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < EPS or b.std() < EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _within(pairs_by_pid):
    """Participant-demeaned pooled correlation."""
    xs, ys = [], []
    for pid, (x, y) in pairs_by_pid.items():
        x, y = np.asarray(x, float), np.asarray(y, float)
        if len(x) < 3:
            continue
        xs.append(x - x.mean()); ys.append(y - y.mean())
    return _pearson(np.concatenate(xs), np.concatenate(ys))


def main():
    parts = load()
    pids = sorted(parts)
    print(f"Mulholland 2023: {len(pids)} participants, "
          f"{sum(len(parts[p]) for p in pids)} beeps\n")

    # collect model outputs aligned with reports
    rep_past, rep_fut, rep_v = [], [], []
    m_fpast, m_ffut, m_vm, m_vr, m_va = [], [], [], [], []
    bypid = defaultdict(lambda: defaultdict(list))
    for i, pid in enumerate(pids):
        seq = parts[pid]; o = drive(seq, 500 + i)
        for beep, oo in zip(seq, o):
            rep_past.append(beep["past"]); rep_fut.append(beep["future"]); rep_v.append(beep["v"])
            m_fpast.append(oo["f_past"]); m_ffut.append(oo["f_fut"])
            m_vm.append(oo["v_model"]); m_vr.append(oo["v_reward"]); m_va.append(oo["v_action"])
            d = bypid[pid]
            d["rep_past"].append(beep["past"]); d["rep_fut"].append(beep["future"]); d["rep_v"].append(beep["v"])
            d["f_past"].append(oo["f_past"]); d["f_fut"].append(oo["f_fut"])

    # ---- T1: latent frame belief vs reported orientation ----
    print("=== T1  latent frame belief vs INDEPENDENTLY REPORTED orientation ===")
    print(f"  past-frame belief ~ reported past    pooled r={_pearson(m_fpast, rep_past):+.3f}   "
          f"within-person r={_within({p:(bypid[p]['f_past'],bypid[p]['rep_past']) for p in bypid}):+.3f}")
    print(f"  future-frame belief ~ reported future pooled r={_pearson(m_ffut, rep_fut):+.3f}   "
          f"within-person r={_within({p:(bypid[p]['f_fut'],bypid[p]['rep_fut']) for p in bypid}):+.3f}")
    # cross (discriminant): past-frame should NOT track future report as well
    print(f"  [discriminant] past-frame ~ reported FUTURE pooled r={_pearson(m_fpast, rep_fut):+.3f}; "
          f"future-frame ~ reported PAST pooled r={_pearson(m_ffut, rep_past):+.3f}")

    # ---- T2: theory sign check (raw reported relationships + model channels) ----
    print("\n=== T2  rumination/worry sign check (reported) ===")
    print(f"  reported past ~ valence   r={_pearson(rep_past, rep_v):+.3f}  "
          f"(within {_within({p:(bypid[p]['rep_past'],bypid[p]['rep_v']) for p in bypid}):+.3f})")
    print(f"  reported future ~ valence r={_pearson(rep_fut, rep_v):+.3f}  "
          f"(within {_within({p:(bypid[p]['rep_fut'],bypid[p]['rep_v']) for p in bypid}):+.3f})")

    # ---- per-channel readout vs reported momentary valence ----
    print("\n=== per-channel valence readout ~ reported momentary valence (pooled) ===")
    for nm, ch in (("v_model (backward)", m_vm), ("v_reward (present)", m_vr), ("v_action (forward)", m_va)):
        print(f"  {nm:20s} r={_pearson(ch, rep_v):+.3f}")


if __name__ == "__main__":
    main()
