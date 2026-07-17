"""
Diagnostic: why doesn't the healthy agent cycle between temporal modes?

Ben's metastability insight: health = ability to switch between attractors,
not any fixed parameter regime. This script analyses action switching rates,
run lengths, policy entropy, and per-action EFE to understand what's locking
the healthy agent into dominant modes.
"""

import numpy as np
from generative_model import (build_model, N_ACTIONS, ACTION_NAMES,
                               RECALL, ENGAGE, FUTURATE, FEEL,
                               DISSOCIATE, ABSTRACT)
from experiments import run_trial, PROFILES

T = 300
SEED = 42


def action_transitions(actions):
    """Compute action-to-action transition matrix."""
    n = N_ACTIONS
    trans = np.zeros((n, n))
    for t in range(len(actions) - 1):
        trans[actions[t + 1], actions[t]] += 1
    # Normalise columns (from → to)
    col_sums = trans.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    return trans / col_sums, trans


def run_lengths(actions):
    """Compute run lengths (consecutive same-action sequences)."""
    runs = []
    current = actions[0]
    length = 1
    for t in range(1, len(actions)):
        if actions[t] == current:
            length += 1
        else:
            runs.append((current, length))
            current = actions[t]
            length = 1
    runs.append((current, length))
    return runs


def switching_rate(actions):
    """Fraction of timesteps where action changes."""
    switches = sum(1 for t in range(1, len(actions)) if actions[t] != actions[t-1])
    return switches / (len(actions) - 1)


def action_entropy(actions):
    """Entropy of action frequency distribution."""
    counts = np.array([np.mean(actions == a) for a in range(N_ACTIONS)])
    counts = np.maximum(counts, 1e-10)
    return float(-np.dot(counts, np.log(counts)))


def mean_policy_entropy(pi_history):
    """Mean per-step policy entropy (decision uncertainty)."""
    entropies = []
    for t in range(len(pi_history)):
        p = np.maximum(pi_history[t], 1e-10)
        entropies.append(float(-np.dot(p, np.log(p))))
    return np.mean(entropies), np.array(entropies)


# ── Run all profiles ────────────────────────────────────────
print("=" * 70)
print("METASTABILITY DIAGNOSTIC")
print("=" * 70)

all_results = {}
for name, prof in PROFILES.items():
    print(f"\n{'─' * 60}")
    print(f"  Profile: {name.upper()}")
    print(f"  pi_pos={prof['pi_pos']}, omega_e={prof['omega_e']}, "
          f"gamma={prof['gamma']}, c_scale={prof['c_scale']}")
    print(f"{'─' * 60}")

    h = run_trial(**prof, T=T, seed=SEED)
    all_results[name] = h
    actions = h['action']

    # 1. Action frequencies
    print("\n  Action frequencies:")
    for a in range(N_ACTIONS):
        frac = np.mean(actions == a)
        bar = '█' * int(frac * 50)
        print(f"    {ACTION_NAMES[a]:12s} {frac:5.1%}  {bar}")

    # 2. Switching rate
    sr = switching_rate(actions)
    print(f"\n  Switching rate: {sr:.1%} of timesteps")

    # 3. Mean policy entropy (per-step decision uncertainty)
    mean_pe, pe_series = mean_policy_entropy(h['pi'])
    max_pe = np.log(N_ACTIONS)
    print(f"  Mean policy entropy: {mean_pe:.3f} / {max_pe:.3f} "
          f"({mean_pe/max_pe:.1%} of max)")

    # 4. Action entropy (empirical distribution)
    ae = action_entropy(actions)
    print(f"  Action entropy: {ae:.3f} / {max_pe:.3f} "
          f"({ae/max_pe:.1%} of max)")

    # 5. Mean run lengths per action
    runs = run_lengths(actions)
    print("\n  Mean run lengths:")
    for a in range(N_ACTIONS):
        a_runs = [r[1] for r in runs if r[0] == a]
        if a_runs:
            print(f"    {ACTION_NAMES[a]:12s}  mean={np.mean(a_runs):.1f}  "
                  f"max={max(a_runs):3d}  count={len(a_runs)}")

    # 6. Mean G (EFE) per action
    mean_G = np.mean(h['G'], axis=0)
    print("\n  Mean EFE per action (lower = preferred):")
    rank = np.argsort(mean_G)
    for idx in rank:
        delta = mean_G[idx] - mean_G[rank[0]]
        bar = '▓' * int(min(delta * 10, 40))
        print(f"    {ACTION_NAMES[idx]:12s}  G={mean_G[idx]:.3f}  "
              f"(+{delta:.3f})  {bar}")

    # 7. Action transition matrix
    trans_norm, trans_raw = action_transitions(actions)
    print("\n  Action transition matrix (FROM → TO, normalised):")
    header = "    FROM\\TO   " + "".join(f"{ACTION_NAMES[a][:5]:>7s}" for a in range(N_ACTIONS))
    print(header)
    for a_from in range(N_ACTIONS):
        row = f"    {ACTION_NAMES[a_from]:10s}"
        for a_to in range(N_ACTIONS):
            val = trans_norm[a_to, a_from]
            row += f" {val:6.2f}"
        print(row)

    # 8. Frame belief dynamics — how much does the frame distribution change?
    fb = h['frame_belief']
    frame_change = np.array([np.linalg.norm(fb[t] - fb[t-1])
                             for t in range(1, T)])
    print(f"\n  Frame belief change per step: mean={np.mean(frame_change):.4f}  "
          f"max={np.max(frame_change):.4f}")
    print(f"  Mean frame beliefs: PAST={np.mean(fb[:,0]):.3f}  "
          f"PRESENT={np.mean(fb[:,1]):.3f}  FUTURE={np.mean(fb[:,2]):.3f}")


# ── Cross-profile comparison ────────────────────────────────
print("\n" + "=" * 70)
print("CROSS-PROFILE COMPARISON")
print("=" * 70)
print(f"\n  {'Profile':12s} {'Switch%':>8s} {'PolicyH':>8s} {'ActionH':>8s} "
      f"{'DomAct':>10s} {'DomFrac':>8s}")
for name in PROFILES:
    h = all_results[name]
    actions = h['action']
    sr = switching_rate(actions)
    mean_pe, _ = mean_policy_entropy(h['pi'])
    ae = action_entropy(actions)
    dom_act = np.argmax([np.mean(actions == a) for a in range(N_ACTIONS)])
    dom_frac = np.mean(actions == dom_act)
    print(f"  {name:12s} {sr:7.1%} {mean_pe:8.3f} {ae:8.3f} "
          f"{ACTION_NAMES[dom_act]:>10s} {dom_frac:7.1%}")


# ── THE KEY QUESTION: why is healthy locked? ────────────────
print("\n" + "=" * 70)
print("DIAGNOSIS: Why doesn't healthy cycle?")
print("=" * 70)

h = all_results['healthy']
G = h['G']  # (T, N_ACTIONS)

# G gap: how much better is the best action vs second-best?
G_sorted = np.sort(G, axis=1)
g_gap = G_sorted[:, 1] - G_sorted[:, 0]  # second-best minus best
print(f"\n  G gap (2nd best - best): mean={np.mean(g_gap):.3f}  "
      f"std={np.std(g_gap):.3f}")
print(f"  When gap > 0.5, softmax with gamma=16 gives >99.97% to winner")
print(f"  Fraction of steps with gap > 0.5: {np.mean(g_gap > 0.5):.1%}")
print(f"  Fraction of steps with gap > 0.2: {np.mean(g_gap > 0.2):.1%}")
print(f"  Fraction of steps with gap > 0.1: {np.mean(g_gap > 0.1):.1%}")
print(f"  Fraction of steps with gap < 0.05: {np.mean(g_gap < 0.05):.1%}")

# Per-action G spread over time
print("\n  G statistics per action across time:")
for a in range(N_ACTIONS):
    g_a = G[:, a]
    print(f"    {ACTION_NAMES[a]:12s}  mean={np.mean(g_a):.3f}  "
          f"std={np.std(g_a):.3f}  min={np.min(g_a):.3f}  max={np.max(g_a):.3f}")

# Check if gamma is the culprit: what would switching rate be with different gamma?
print("\n  Hypothetical switching rates with different gamma:")
for gamma_test in [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
    # Recompute policy from stored G values
    switches = 0
    prev_a = None
    for t in range(T):
        log_pi = -gamma_test * G[t]
        log_pi -= log_pi.max()
        pi = np.exp(log_pi)
        pi /= pi.sum()
        a = np.argmax(pi)  # deterministic for comparison
        if prev_a is not None and a != prev_a:
            switches += 1
        prev_a = a
    sr = switches / (T - 1)
    print(f"    gamma={gamma_test:5.1f}  switching={sr:.1%}")
