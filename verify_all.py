"""Verify all emotion + clinical profiles after final parameter updates."""

import numpy as np
import sys
sys.path.insert(0, '.')

from experiments import run_trial, PROFILES, EMOTION_PROFILES
from generative_model import N_ACTIONS, ACTION_NAMES, FRAME_NAMES


def run_avg(params, n_seeds=20, base_seed=42, T=500):
    agg = {k: [] for k in ['valence', 'v_reward', 'v_model', 'v_action']}
    act_all, frame_all = [], []
    p = {k: v for k, v in params.items() if k != 'desc'}
    for s in range(n_seeds):
        h = run_trial(**p, T=T, seed=base_seed + s)
        for k in agg:
            agg[k].append(np.mean(h[k]))
        act_all.append([np.mean(h['action'] == a) for a in range(N_ACTIONS)])
        frame_all.append(np.mean(h['frame_belief'], axis=0))
    af = np.mean(act_all, axis=0)
    fb = np.mean(frame_all, axis=0)
    return {
        'valence': np.mean(agg['valence']),
        'val_std': np.std(agg['valence']),
        'v_reward': np.mean(agg['v_reward']),
        'v_model': np.mean(agg['v_model']),
        'v_action': np.mean(agg['v_action']),
        'actions': af, 'frame': fb,
        'neg_seeds': sum(1 for v in agg['valence'] if v < 0),
    }


def pr(name, r):
    af, fb = r['actions'], r['frame']
    print(f"  {name:<14} val={r['valence']:>+.4f}+-{r['val_std']:.3f} "
          f"neg={r['neg_seeds']:>2}/20 "
          f"vm={r['v_model']:>+.3f} vr={r['v_reward']:>+.3f} va={r['v_action']:>+.3f} "
          f"REC={af[0]:.2f} ENG={af[1]:.2f} FUT={af[2]:.2f} "
          f"FEL={af[3]:.2f} DIS={af[4]:.2f} ABS={af[5]:.2f} "
          f"| P={fb[0]:.2f} PR={fb[1]:.2f} F={fb[2]:.2f} "
          f"| {ACTION_NAMES[np.argmax(af)]}/{FRAME_NAMES[np.argmax(fb)]}")


# ── Clinical profiles ──
print("=" * 150)
print("CLINICAL PROFILES (experiments.py)")
print("=" * 150)
for name, params in PROFILES.items():
    r = run_avg(params)
    pr(name, r)

# ── Emotion profiles ──
print("\n" + "=" * 150)
print("EMOTION PROFILES (experiments.py / emotion_diagnostic.py)")
print("=" * 150)
for name, params in EMOTION_PROFILES.items():
    r = run_avg(params)
    pr(name, r)

# ── Summary table ──
print("\n" + "=" * 150)
print("QUICK SUMMARY: key qualitative checks")
print("=" * 150)
print("  Profile      | Expected                                    | Check")
print("  " + "-" * 100)
checks = [
    ("happy",      "val>0, ENGAGE dominant, PRESENT frame"),
    ("content",    "val>0, balanced actions, PRESENT frame"),
    ("calm",       "val~0, balanced, PRESENT frame"),
    ("excited",    "val>0, FUTURATE dominant, FUTURE frame"),
    ("alert",      "val~0, high D, diverse actions"),
    ("angry",      "val<0, high D, ENGAGE dominant"),
    ("fearful",    "val<0, low D, diverse/DISSOCIATE"),
    ("sad",        "val<0, RECALL dominant, PRESENT/PAST frame"),
    ("depressed",  "val~0 (anhedonia), RECALL dominant, PAST frame"),
    ("bored",      "val~0, low arousal, DISSOCIATE/FEEL"),
]
for name, expected in checks:
    print(f"  {name:<14}| {expected}")
