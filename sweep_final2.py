"""
Refined sweep round 2: narrow in on best configs from round 1.

1. Depressed: E=[10,0,-1,0,0,-1] tips PAST but val goes +0.012.
   Try combining FUT/ABS penalty with slightly higher c_neg to keep val negative.
2. Sad: Need val<0 with FEEL or RECALL dominant, PRESENT frame.
   Try intermediate E_rec (6-7) and c_neg (1.0-1.3).
3. Excited: pi=3.5-4.0 with mild E vector to cleanly separate FUTURATE.
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
    print(f"  {name:<45} val={r['valence']:>+.4f}+-{r['val_std']:.3f} neg={r['neg_seeds']:>2}/20 "
          f"vr={r['v_reward']:>+.4f} "
          f"REC={af[0]:.2f} ENG={af[1]:.2f} FUT={af[2]:.2f} FEL={af[3]:.2f} DIS={af[4]:.2f} ABS={af[5]:.2f} "
          f"| P={fb[0]:.2f} PR={fb[1]:.2f} F={fb[2]:.2f} "
          f"| {ACTION_NAMES[np.argmax(af)]}/{FRAME_NAMES[np.argmax(fb)]}")


# ── 1. Depressed: E with FUT/ABS penalty + higher c_neg ──
print("="*140)
print("DEPRESSED: E+FUT/ABS penalty combined with c_neg increase")
print("  Goal: PAST > FUTURE, RECALL dominant, val < 0")
print("="*140)

base_dep = dict(K=4, M=8, pi_pos=0.2, omega_e=5.0, gamma=16.0, c_scale=0.1,
                c_pos=0.1, neg_val_precision=1.0, volatility=0.45)

for tag, extra in [
    # Baseline from round 1
    ("c_neg=2.0, E=[10,0,0,0,0,0] (current)",
     dict(c_neg=2.0, habit_E=np.array([10,0,0,0,0,0], dtype=float))),
    ("c_neg=2.0, E=[10,0,-1,0,0,-1] (R1 best frame)",
     dict(c_neg=2.0, habit_E=np.array([10,0,-1,0,0,-1], dtype=float))),
    # Bump c_neg to compensate for val going positive
    ("c_neg=2.5, E=[10,0,-1,0,0,-1]",
     dict(c_neg=2.5, habit_E=np.array([10,0,-1,0,0,-1], dtype=float))),
    ("c_neg=3.0, E=[10,0,-1,0,0,-1]",
     dict(c_neg=3.0, habit_E=np.array([10,0,-1,0,0,-1], dtype=float))),
    ("c_neg=2.5, E=[10,0,-2,0,0,-2]",
     dict(c_neg=2.5, habit_E=np.array([10,0,-2,0,0,-2], dtype=float))),
    ("c_neg=3.0, E=[10,0,-2,0,0,-2]",
     dict(c_neg=3.0, habit_E=np.array([10,0,-2,0,0,-2], dtype=float))),
    # Also try neg_val_precision for rumination asymmetry
    ("c_neg=2.5, nvp=1.5, E=[10,0,-1,0,0,-1]",
     dict(c_neg=2.5, neg_val_precision=1.5, habit_E=np.array([10,0,-1,0,0,-1], dtype=float))),
    ("c_neg=3.0, nvp=1.5, E=[10,0,-1,0,0,-1]",
     dict(c_neg=3.0, neg_val_precision=1.5, habit_E=np.array([10,0,-1,0,0,-1], dtype=float))),
]:
    r = run_avg({**base_dep, **extra})
    pr(tag, r)


# ── 2. Sad: intermediate E_rec with mild c_neg ──
print("\n" + "="*140)
print("SAD: intermediate E_rec + c_neg tuning")
print("  Goal: val < 0, FEEL or RECALL dominant, PRESENT frame")
print("="*140)

base_sad = dict(K=4, M=8, pi_pos=0.1, omega_e=3.0, gamma=16.0, c_scale=0.25,
                neg_val_precision=1.0, volatility=0.6)

for tag, extra in [
    # Current baseline
    ("current (cp=0.25, cn=1.0, E=[5,0,0,0,0,0])",
     dict(c_pos=0.25, c_neg=1.0, habit_E=np.array([5,0,0,0,0,0], dtype=float))),
    # Increase E_rec slightly to push val negative
    ("cn=1.0, E=[6,0,0,0,0,0]",
     dict(c_pos=0.25, c_neg=1.0, habit_E=np.array([6,0,0,0,0,0], dtype=float))),
    ("cn=1.0, E=[7,0,0,0,0,0]",
     dict(c_pos=0.25, c_neg=1.0, habit_E=np.array([7,0,0,0,0,0], dtype=float))),
    # Small c_neg increase
    ("cn=1.2, E=[5,0,0,0,0,0]",
     dict(c_pos=0.25, c_neg=1.2, habit_E=np.array([5,0,0,0,0,0], dtype=float))),
    ("cn=1.2, E=[6,0,0,0,0,0]",
     dict(c_pos=0.25, c_neg=1.2, habit_E=np.array([6,0,0,0,0,0], dtype=float))),
    ("cn=1.3, E=[5,0,0,0,0,0]",
     dict(c_pos=0.25, c_neg=1.3, habit_E=np.array([5,0,0,0,0,0], dtype=float))),
    ("cn=1.3, E=[6,0,0,0,0,0]",
     dict(c_pos=0.25, c_neg=1.3, habit_E=np.array([6,0,0,0,0,0], dtype=float))),
    # Try with FUT/ABS penalty too
    ("cn=1.2, E=[6,0,-1,0,0,-1]",
     dict(c_pos=0.25, c_neg=1.2, habit_E=np.array([6,0,-1,0,0,-1], dtype=float))),
    ("cn=1.3, E=[5,0,-1,0,0,-1]",
     dict(c_pos=0.25, c_neg=1.3, habit_E=np.array([5,0,-1,0,0,-1], dtype=float))),
    # Slightly higher c_neg for stronger negative valence
    ("cn=1.5, E=[6,0,-1,0,0,-1]",
     dict(c_pos=0.25, c_neg=1.5, habit_E=np.array([6,0,-1,0,0,-1], dtype=float))),
]:
    r = run_avg({**base_sad, **extra})
    pr(tag, r)


# ── 3. Excited: mild E vector to tip FUTURATE ──
print("\n" + "="*140)
print("EXCITED: mild E vector + pi_pos tuning")
print("  Goal: FUTURATE clearly dominant (>40%), val>0, FUTURE frame")
print("="*140)

base_exc = dict(K=4, M=8, omega_e=0.5, gamma=4.0, volatility=0.45)

for tag, extra in [
    # Current baseline
    ("current (pi=5, cs=4)",
     dict(pi_pos=5.0, c_scale=4.0)),
    # Mild E vectors
    ("pi=5, cs=4, E=[0,0,1,0,0,0]",
     dict(pi_pos=5.0, c_scale=4.0, habit_E=np.array([0,0,1,0,0,0], dtype=float))),
    ("pi=5, cs=4, E=[-1,0,1,0,0,0]",
     dict(pi_pos=5.0, c_scale=4.0, habit_E=np.array([-1,0,1,0,0,0], dtype=float))),
    ("pi=5, cs=4, E=[-1,0,2,0,0,0]",
     dict(pi_pos=5.0, c_scale=4.0, habit_E=np.array([-1,0,2,0,0,0], dtype=float))),
    # Reduced pi_pos
    ("pi=4, cs=4, E=[0,0,1,0,0,0]",
     dict(pi_pos=4.0, c_scale=4.0, habit_E=np.array([0,0,1,0,0,0], dtype=float))),
    ("pi=4, cs=4, E=[-1,0,1,0,0,0]",
     dict(pi_pos=4.0, c_scale=4.0, habit_E=np.array([-1,0,1,0,0,0], dtype=float))),
    ("pi=4.5, cs=4, E=[-1,0,1,0,0,0]",
     dict(pi_pos=4.5, c_scale=4.0, habit_E=np.array([-1,0,1,0,0,0], dtype=float))),
    ("pi=3.5, cs=4 (no E)",
     dict(pi_pos=3.5, c_scale=4.0)),
]:
    r = run_avg({**base_exc, **extra})
    pr(tag, r)


if __name__ == '__main__':
    pass
