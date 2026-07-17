"""
Targeted sweep for three remaining issues:
1. Depressed: tip PAST over FUTURE without saturating RECALL or losing neg valence
2. Sad: push valence negative
3. Excited: reduce RECALL share, boost FUTURATE
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from experiments import run_trial
from generative_model import (RECALL, ENGAGE, FUTURATE, FEEL, DISSOCIATE,
                              ABSTRACT, N_ACTIONS, ACTION_NAMES, FRAME_NAMES)


def run_avg(params, n_seeds=20, base_seed=42, T=500):
    agg = {k: [] for k in ['valence', 'v_reward']}
    act_all, frame_all = [], []
    p = {k: v for k, v in params.items() if k != 'desc'}
    for s in range(n_seeds):
        h = run_trial(**p, T=T, seed=base_seed + s)
        agg['valence'].append(np.mean(h['valence']))
        agg['v_reward'].append(np.mean(h['v_reward']))
        act_all.append([np.mean(h['action'] == a) for a in range(N_ACTIONS)])
        frame_all.append(np.mean(h['frame_belief'], axis=0))
    af = np.mean(act_all, axis=0)
    fb = np.mean(frame_all, axis=0)
    return {
        'valence': np.mean(agg['valence']),
        'val_std': np.std(agg['valence']),
        'v_reward': np.mean(agg['v_reward']),
        'actions': af, 'frame': fb,
        'neg_seeds': sum(1 for v in agg['valence'] if v < 0),
    }


def pr(name, r):
    af, fb = r['actions'], r['frame']
    print(f"  {name:<35} val={r['valence']:>+.4f}+-{r['val_std']:.3f} neg={r['neg_seeds']:>2}/20 "
          f"vr={r['v_reward']:>+.4f} "
          f"REC={af[0]:.2f} ENG={af[1]:.2f} FUT={af[2]:.2f} FEL={af[3]:.2f} DIS={af[4]:.2f} ABS={af[5]:.2f} "
          f"| P={fb[0]:.2f} PR={fb[1]:.2f} F={fb[2]:.2f} "
          f"| {ACTION_NAMES[np.argmax(af)]}/{FRAME_NAMES[np.argmax(fb)]}")


# ── 1. Depressed: add small FUTURATE/ABSTRACT penalty to tip PAST ──
print("="*130)
print("DEPRESSED: E vector variants (base: E_rec=10)")
print("  Goal: PAST dominant, RECALL dominant, val < 0")
print("="*130)

base_dep = dict(K=4, M=8, pi_pos=0.2, omega_e=5.0, gamma=16.0, c_scale=0.1,
                c_pos=0.1, c_neg=2.0, neg_val_precision=1.0, volatility=0.45)

for tag, E in [
    ("E=[10,0,0,0,0,0] (current)",     np.array([10,0,0,0,0,0], dtype=float)),
    ("E=[10,0,-1,0,0,-1]",             np.array([10,0,-1,0,0,-1], dtype=float)),
    ("E=[10,0,-2,0,0,-2]",             np.array([10,0,-2,0,0,-2], dtype=float)),
    ("E=[10,0,-3,0,0,-3]",             np.array([10,0,-3,0,0,-3], dtype=float)),
    ("E=[12,0,-1,0,0,-1]",             np.array([12,0,-1,0,0,-1], dtype=float)),
    ("E=[12,0,-2,0,0,-2]",             np.array([12,0,-2,0,0,-2], dtype=float)),
    ("E=[10,0,-1,2,0,-1]",             np.array([10,0,-1,2,0,-1], dtype=float)),
    ("E=[10,0,-2,2,0,-2]",             np.array([10,0,-2,2,0,-2], dtype=float)),
    ("E=[8,0,-2,2,0,-2]",              np.array([8,0,-2,2,0,-2], dtype=float)),
    ("E=[8,0,-3,2,0,-3]",              np.array([8,0,-3,2,0,-3], dtype=float)),
]:
    r = run_avg({**base_dep, 'habit_E': E})
    pr(tag, r)


# ── 2. Sad: push valence negative ──
print("\n" + "="*130)
print("SAD: c_neg and E vector variants")
print("  Goal: val < 0, FEEL or RECALL dominant, PRESENT or PAST frame")
print("="*130)

base_sad = dict(K=4, M=8, pi_pos=0.1, omega_e=3.0, gamma=16.0, c_scale=0.25,
                neg_val_precision=1.0, volatility=0.6)

for tag, extra in [
    ("current (cp=0.25,cn=1.0,E=[5,0,0,0,0,0])",
     dict(c_pos=0.25, c_neg=1.0, habit_E=np.array([5,0,0,0,0,0], dtype=float))),
    ("cn=1.5, E=[5,0,0,0,0,0]",
     dict(c_pos=0.25, c_neg=1.5, habit_E=np.array([5,0,0,0,0,0], dtype=float))),
    ("cn=2.0, E=[5,0,0,0,0,0]",
     dict(c_pos=0.25, c_neg=2.0, habit_E=np.array([5,0,0,0,0,0], dtype=float))),
    ("cn=1.5, E=[5,0,-2,0,0,-2]",
     dict(c_pos=0.25, c_neg=1.5, habit_E=np.array([5,0,-2,0,0,-2], dtype=float))),
    ("cn=1.0, E=[8,0,0,0,0,0]",
     dict(c_pos=0.25, c_neg=1.0, habit_E=np.array([8,0,0,0,0,0], dtype=float))),
    ("cn=1.0, E=[6,0,-2,0,0,-2]",
     dict(c_pos=0.25, c_neg=1.0, habit_E=np.array([6,0,-2,0,0,-2], dtype=float))),
    ("cn=1.5, E=[3,0,0,0,0,0]",
     dict(c_pos=0.25, c_neg=1.5, habit_E=np.array([3,0,0,0,0,0], dtype=float))),
    ("cn=1.5, E=[3,0,-2,0,0,-2]",
     dict(c_pos=0.25, c_neg=1.5, habit_E=np.array([3,0,-2,0,0,-2], dtype=float))),
    ("cn=2.0, E=[3,0,0,0,0,0]",
     dict(c_pos=0.25, c_neg=2.0, habit_E=np.array([3,0,0,0,0,0], dtype=float))),
    ("cn=2.0, E=[3,0,-2,0,0,-2]",
     dict(c_pos=0.25, c_neg=2.0, habit_E=np.array([3,0,-2,0,0,-2], dtype=float))),
]:
    r = run_avg({**base_sad, **extra})
    pr(tag, r)


# ── 3. Excited: reduce RECALL, boost FUTURATE ──
print("\n" + "="*130)
print("EXCITED: reduce RECALL dominance")
print("  Goal: FUTURATE clearly dominant, positive val, low dominance")
print("="*130)

base_exc = dict(K=4, M=8, omega_e=0.5, gamma=4.0, volatility=0.45)

for tag, extra in [
    ("current (pi=5, cs=4)",
     dict(pi_pos=5.0, c_scale=4.0)),
    ("pi=4, cs=4",
     dict(pi_pos=4.0, c_scale=4.0)),
    ("pi=3.5, cs=4",
     dict(pi_pos=3.5, c_scale=4.0)),
    ("pi=5, cs=4, E=[0,0,3,0,0,0]",
     dict(pi_pos=5.0, c_scale=4.0, habit_E=np.array([0,0,3,0,0,0], dtype=float))),
    ("pi=5, cs=4, E=[-2,0,2,0,0,0]",
     dict(pi_pos=5.0, c_scale=4.0, habit_E=np.array([-2,0,2,0,0,0], dtype=float))),
    ("pi=5, cs=4, E=[-3,0,3,0,0,0]",
     dict(pi_pos=5.0, c_scale=4.0, habit_E=np.array([-3,0,3,0,0,0], dtype=float))),
    ("pi=5, cs=3, E=[-2,0,2,0,0,0]",
     dict(pi_pos=5.0, c_scale=3.0, habit_E=np.array([-2,0,2,0,0,0], dtype=float))),
    ("pi=4, cs=3, E=[-2,0,2,0,0,0]",
     dict(pi_pos=4.0, c_scale=3.0, habit_E=np.array([-2,0,2,0,0,0], dtype=float))),
]:
    r = run_avg({**base_exc, **extra})
    pr(tag, r)


if __name__ == '__main__':
    pass
