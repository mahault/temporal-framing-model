"""Verify current state: all profiles with new mechanisms."""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from experiments import run_trial, PROFILES, STRESS_PROFILES
from generative_model import N_ACTIONS, ACTION_NAMES

T = 300
N_RUNS = 5

# Combine all profiles we need to verify
ALL = {}
for k, v in PROFILES.items():
    ALL[k] = {kk: vv for kk, vv in v.items() if kk != 'desc'}
# Add stressed from STRESS_PROFILES
ALL['chronic_stress'] = {kk: vv for kk, vv in STRESS_PROFILES['stressed'].items() if kk != 'desc'}
# Add recall_impaired
ALL['recall_impaired'] = dict(K=8, M=8, pi_pos=0.2, omega_e=5.0, gamma=16.0, c_scale=1.0, volatility=0.45)

print("=" * 110)
print("  FULL STATE VERIFICATION (N=%d runs, T=%d)" % (N_RUNS, T))
print("=" * 110)

for pname, prof in ALL.items():
    vals, vms, vrs, vas = [], [], [], []
    energies, sws, doms, arousals = [], [], [], []
    all_props = {a: [] for a in ACTION_NAMES}
    all_fb = []
    all_v = []

    for r in range(N_RUNS):
        h = run_trial(**prof, T=T, seed=42 + r)
        vals.append(np.mean(h['valence']))
        vms.append(np.mean(h['v_model']))
        vrs.append(np.mean(h['v_reward']))
        vas.append(np.mean(h['v_action']))
        energies.append(np.mean(h['energy_true']))
        sws.append(np.sum(h['action'][1:] != h['action'][:-1]) / (T-1))
        all_fb.append(np.mean(h['frame_belief'], axis=0))
        arousals.append(np.mean(h['arousal_norm']))
        pi_arr = np.maximum(h['pi'], 1e-16)
        H_pi = -np.sum(pi_arr * np.log(pi_arr), axis=1)
        doms.append(np.mean(1.0 - H_pi / np.log(N_ACTIONS)))
        for i, a_name in enumerate(ACTION_NAMES):
            all_props[a_name].append(np.mean(h['action'] == i))
        all_v.extend(h['valence'].tolist())

    p = {k: np.mean(v) for k, v in all_props.items()}
    fb = np.mean(all_fb, axis=0)
    v_arr = np.array(all_v)

    print(f"\n  {'=' * 50}")
    print(f"  {pname.upper()}  params: {prof}")
    print(f"  {'=' * 50}")
    print(f"    Valence:     mean={np.mean(vals):+.3f}  std={np.std(vals):.3f}  "
          f"median={np.median(v_arr):+.3f}  frac<0={np.mean(v_arr<0):.0%}")
    print(f"    Channels:    vm={np.mean(vms):+.4f}  vr={np.mean(vrs):+.4f}  va={np.mean(vas):+.4f}")
    print(f"    Energy:      {np.mean(energies):.3f}")
    print(f"    Arousal:     {np.mean(arousals):.3f}")
    print(f"    Dominance:   {np.mean(doms):.3f}")
    print(f"    Switching:   {np.mean(sws):.0%}")
    print(f"    Frame:       PAST={fb[0]:.2f}  PRESENT={fb[1]:.2f}  FUTURE={fb[2]:.2f}")
    print(f"    Actions:     REC={p['RECALL']:.0%}  ENG={p['ENGAGE']:.0%}  "
          f"FUT={p['FUTURATE']:.0%}  FEL={p['FEEL']:.0%}  "
          f"DIS={p['DISSOCIATE']:.0%}  ABS={p['ABSTRACT']:.0%}")
