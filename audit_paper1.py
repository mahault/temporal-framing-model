"""
Audit: check every numerical claim in affective_valence_temporal_framing.tex
against current simulation output.
"""

import numpy as np
from generative_model import (N_ACTIONS, ACTION_NAMES, RECALL, ENGAGE, FUTURATE,
                               FEEL, DISSOCIATE, ABSTRACT)
from experiments import (run_trial, run_phenotype_experiment,
                          run_feedback_reliance_experiment,
                          run_chronic_stress_experiment,
                          run_stress_decay_experiment,
                          PROFILES, FEEDBACK_PROFILES, STRESS_PROFILES,
                          STRESS_DECAY_PROFILES)

T = 300
T_SD = 3000
SEED = 42


def action_stats(h, label=""):
    """Print action frequencies, frame beliefs, run lengths."""
    actions = h['action']
    freqs = {ACTION_NAMES[a]: np.mean(actions == a) for a in range(N_ACTIONS)}
    fb = np.mean(h['frame_belief'], axis=0)
    mv = np.mean(h['valence_belief'])
    me = np.mean(h['energy_true'])

    # Run lengths
    from collections import defaultdict
    runs = defaultdict(list)
    current = actions[0]
    length = 1
    for t in range(1, len(actions)):
        if actions[t] == current:
            length += 1
        else:
            runs[current].append(length)
            current = actions[t]
            length = 1
    runs[current].append(length)

    print(f"\n  {label}")
    print(f"    Action freqs: ", end="")
    for a in range(N_ACTIONS):
        if freqs[ACTION_NAMES[a]] > 0.005:
            print(f"{ACTION_NAMES[a]}={freqs[ACTION_NAMES[a]]:.1%}  ", end="")
    print()
    print(f"    Frame beliefs: PAST={fb[0]:.2f}  PRESENT={fb[1]:.2f}  FUTURE={fb[2]:.2f}")
    print(f"    mean_v={mv:.3f}  mean_e={me:.3f}")
    for a in range(N_ACTIONS):
        if len(runs[a]) > 0:
            print(f"    Run length {ACTION_NAMES[a]:12s}: mean={np.mean(runs[a]):.1f}  max={max(runs[a])}")

    # Mean VFE conditioned on action
    print(f"    VFE|action: ", end="")
    for a in range(N_ACTIONS):
        mask = actions == a
        if mask.sum() > 0:
            print(f"{ACTION_NAMES[a]}={np.mean(h['vfe'][mask]):.3f}  ", end="")
    print()

    # Policy entropy
    pi_h = []
    for t in range(len(actions)):
        p = np.maximum(h['pi'][t], 1e-10)
        pi_h.append(float(-np.dot(p, np.log(p))))
    print(f"    Policy entropy: {np.mean(pi_h):.3f} / {np.log(N_ACTIONS):.3f}")

    # Transition matrix (key transitions)
    trans = np.zeros((N_ACTIONS, N_ACTIONS))
    for t in range(len(actions) - 1):
        trans[actions[t+1], actions[t]] += 1
    col_sums = trans.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    trans_norm = trans / col_sums
    # Print RECALL->ENGAGE and other key transitions
    key_pairs = [(RECALL, ENGAGE), (RECALL, FEEL), (FEEL, RECALL),
                 (FUTURATE, FUTURATE), (ABSTRACT, FUTURATE)]
    print(f"    Key transitions: ", end="")
    for (a_from, a_to) in key_pairs:
        if col_sums[0, a_from] > 1:
            print(f"{ACTION_NAMES[a_from]}->{ACTION_NAMES[a_to]}={trans_norm[a_to,a_from]:.2f}  ", end="")
    print()

    return freqs, fb


# ══════════════════════════════════════════════════════════
print("=" * 70)
print("PAPER 1 AUDIT: affective_valence_temporal_framing.tex")
print("=" * 70)


# ── 1. Phenotype comparison (used in fig11 framing dynamics) ──
print("\n" + "=" * 70)
print("EXPERIMENT 1: Phenotype comparison (fig11 source)")
print("=" * 70)
results = {}
for name, prof in PROFILES.items():
    results[name] = run_trial(**prof, T=T, seed=SEED)
    action_stats(results[name], label=f"{name.upper()} ({prof['desc'][:50]}...)")

print("\n  PAPER CLAIMS TO CHECK:")
print("  - Line 232: recall-impaired RECALL drops from 43% to 20%")
print("  - Line 262: healthy never selects FUTURATE at all")
print("  - Line 262: FUTURATE run lengths ~2.2 in manic, RECALL ~1.7 in manic")
print("  - Line 266: healthy frame = ~0.47 PRESENT, ~0.34 PAST, ~0.20 FUTURE")
print("  - Line 266: depressive elevated future ~0.33")
print("  - Line 266: manic dominant future ~0.53")
print("  - Line 372: manic FUTURATE 51% selection rate")


# ── 2. Feedback reliance (fig10) ──────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT 6: Feedback reliance (fig10)")
print("=" * 70)
for name, prof in FEEDBACK_PROFILES.items():
    h = run_trial(**prof, T=T, seed=SEED)
    action_stats(h, label=f"{name.upper()}")

print("\n  PAPER CLAIMS (Line 232, 239):")
print("  - Healthy: RECALL ~43%, FEEL ~57%")
print("  - Recall-impaired: RECALL drops to ~20%, FEEL ~76%, FUTURATE ~4%")
print("  - mean_v: 0.63 vs 0.55")


# ── 3. Chronic stress (fig12) ────────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT 8: Chronic stress (fig12)")
print("=" * 70)
for name, prof in STRESS_PROFILES.items():
    h = run_trial(**prof, T=T, seed=SEED)
    action_stats(h, label=f"{name.upper()}")

print("\n  PAPER CLAIMS (Line 294, 313):")
print("  - Stressed: elevated PAST (0.42 vs 0.34), depleted PRESENT (0.38 vs 0.46)")
print("  - v_model persistently negative in stressed")


# ── 4. Stress decay (fig14) ─────────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT 10: Stress decay (fig14, T=3000)")
print("=" * 70)
for name, prof in STRESS_DECAY_PROFILES.items():
    print(f"  Running {name} (T={T_SD})...", flush=True)
    h = run_trial(**prof, T=T_SD, seed=SEED)
    mv = np.mean(h['valence_belief'])
    mp = np.mean(h['pi_pos'])
    final_pp = h['pi_pos'][-100:].mean()
    print(f"    {name}: mean_v={mv:.3f}  mean_pi_pos={mp:.3f}  final_pi_pos={final_pp:.3f}")

print("\n  PAPER CLAIMS (Line 361):")
print("  - Healthy: pi_pos drifts upward to ~5.5 (final ~6.0)")
print("  - Stressed: pi_pos shifts to ~4.3 (final ~4.4)")
