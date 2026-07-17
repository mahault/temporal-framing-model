"""
Diagnostic part 2: What mechanisms could create metastability?

Tests several candidate interventions to see which produces healthy cycling:
1. Frame-dependent C vectors (each frame has intrinsic value)
2. Novelty/anti-perseveration (habit penalty in EFE)
3. State-dependent frame preferences (wanting different frames at different valences)
4. Rebalanced B matrices (weaken RECALL, strengthen ENGAGE/FUTURATE intrinsic value)

For now, we focus on understanding the EFE landscape more deeply.
"""

import numpy as np
from generative_model import (build_model, N_ACTIONS, ACTION_NAMES, EPS,
                               RECALL, ENGAGE, FUTURATE, FEEL,
                               DISSOCIATE, ABSTRACT, N_FRAMES,
                               B_valence, B_energy, B_frame,
                               _gaussian_col, _softmax, flat_idx)
from experiments import run_trial

T = 300
SEED = 42

# ── Decompose EFE per modality for healthy agent ────────────
print("=" * 70)
print("PER-MODALITY EFE DECOMPOSITION (Healthy agent)")
print("=" * 70)

model = build_model(K=8, M=8, pi_pos=5.0, omega_e=5.0, gamma=16.0, c_scale=1.0)
K, M = model.K, model.M

# Initialise beliefs to D (starting state)
beliefs = model.D.copy()

# Compute EFE per action per modality at step 0
print("\n  Initial state (prior beliefs):")
for a in range(N_ACTIONS):
    q_pred = model.B[a] @ beliefs
    q_pred = np.maximum(q_pred, EPS)
    q_pred /= q_pred.sum()

    risks = []
    ambs = []
    for m, label in enumerate(['o_ext', 'o_int', 'o_val']):
        Am = model.A[m]
        Cm = model.C[m]
        q_o = Am @ q_pred
        q_o = np.maximum(q_o, EPS)
        q_o /= q_o.sum()
        p_pref = np.exp(Cm - Cm.max())
        p_pref /= (p_pref.sum() + EPS)
        risk = float(np.dot(q_o, np.log(q_o + EPS) - np.log(p_pref + EPS)))
        H_cols = -np.sum(Am * np.log(Am + EPS), axis=0)
        amb = float(np.dot(q_pred, H_cols))
        risks.append(risk)
        ambs.append(amb)

    total = sum(risks) + sum(ambs)
    print(f"\n  {ACTION_NAMES[a]:12s}  G={total:.4f}")
    for m, label in enumerate(['o_ext', 'o_int', 'o_val']):
        print(f"    {label:6s}  risk={risks[m]:.4f}  amb={ambs[m]:.4f}  "
              f"sum={risks[m]+ambs[m]:.4f}")


# ── Key insight: what does each action PREDICT? ─────────────
print("\n" + "=" * 70)
print("PREDICTED OBSERVATIONS (Healthy agent, from prior)")
print("=" * 70)

for a in range(N_ACTIONS):
    q_pred = model.B[a] @ beliefs
    q_pred = np.maximum(q_pred, EPS)
    q_pred /= q_pred.sum()

    # Marginals
    joint = q_pred.reshape(K, M, 3)
    v_marg = joint.sum(axis=(1, 2))
    e_marg = joint.sum(axis=(0, 2))
    f_marg = joint.sum(axis=(0, 1))

    E_v = np.dot(np.arange(K), v_marg) / max(K-1, 1)
    E_e = np.dot(np.arange(M), e_marg) / max(M-1, 1)

    q_o_ext = model.A[0] @ q_pred
    q_o_int = model.A[1] @ q_pred
    q_o_val = model.A[2] @ q_pred

    print(f"\n  {ACTION_NAMES[a]:12s}:")
    print(f"    E[valence]={E_v:.3f}  E[energy]={E_e:.3f}  "
          f"frame=[{f_marg[0]:.2f}, {f_marg[1]:.2f}, {f_marg[2]:.2f}]")
    print(f"    P(o_ext)=[neg:{q_o_ext[0]:.3f}, neu:{q_o_ext[1]:.3f}, pos:{q_o_ext[2]:.3f}]")
    print(f"    P(o_int)=[dep:{q_o_int[0]:.3f}, neu:{q_o_int[1]:.3f}, ene:{q_o_int[2]:.3f}]")


# ── What would frame-sensitive preferences look like? ────────
print("\n" + "=" * 70)
print("THOUGHT EXPERIMENT: Frame-sensitive preferences")
print("=" * 70)
print("""
Currently: C vectors are frame-blind. The agent doesn't PREFER any frame.
Proposal: Add a 4th observation modality o_frame (3 outcomes: PAST, PRESENT, FUTURE)
with a C_frame preference that prefers DIVERSITY, not any specific frame.

But this is tricky — how do you express "I want to switch frames" in C vectors?
C vectors express preferences over single observations, not sequences.

Alternative principled mechanisms:
1. FEEL should have diminishing returns (if you FEEL too much, load is already low,
   nothing to process -> ambiguity rises -> other actions become competitive)
2. RECALL should have diminishing returns (if you've recalled recently, the
   positive pull weakens or accuracy drops -> Bayesian habituation)
3. The generative model itself should encode that BEING IN ONE FRAME TOO LONG
   degrades observations (attention fatigue -> A matrix blurs for current frame)

Option 3 is the most principled: it says "precision drops on the channel you're
overusing, making other channels relatively more informative."
""")


# ── Test: what if FEEL had diminishing returns? ──────────────
print("=" * 70)
print("MECHANISM ANALYSIS: Why FEEL dominates")
print("=" * 70)
print("""
FEEL: B_valence = 0.3*toward_neutral + 0.7*stay
      B_energy = delta=+1.2 (big load reduction)
      B_frame  = 50-60% PRESENT

The healthy agent's prior is peaked at high valence, high energy.
FEEL predicts: "energy goes UP (already high -> stays high = good),
valence stays near current (high = good), frame goes PRESENT."

At high energy (low load), FEEL's +1.2 delta means predicted energy
overshoots the top -> clipped -> very peaked prediction at max energy.
This gives LOW risk on o_int (predicted obs matches C_int preference
for "energised") and LOW ambiguity (peaked prediction).

FEEL "wins" not because the agent needs to FEEL, but because FEEL's
predictions are the most precise and preference-aligned when load is
already low. It's the safest bet.
""")

# Verify: at different energy levels, how does FEEL's G compare?
print("\n  FEEL vs RECALL G at different energy levels:")
for e_level in [0, 2, 4, 6, 7]:  # M=8, so 0=depleted, 7=energised
    # Construct beliefs peaked at (v=6, e=e_level, f=PRESENT)
    test_beliefs = np.zeros(K * M * 3)
    for v in range(K):
        for f in range(3):
            idx = flat_idx(v, e_level, f, M)
            test_beliefs[idx] = _gaussian_col(K, 6.0, 5.0)[v] * [0.2, 0.6, 0.2][f]
    test_beliefs = np.maximum(test_beliefs, EPS)
    test_beliefs /= test_beliefs.sum()

    G_feel = 0.0
    G_recall = 0.0
    for action, G_target in [(FEEL, 'G_feel'), (RECALL, 'G_recall')]:
        q_pred = model.B[action] @ test_beliefs
        q_pred = np.maximum(q_pred, EPS)
        q_pred /= q_pred.sum()
        g = 0.0
        for m_idx in range(len(model.A)):
            Am = model.A[m_idx]
            Cm = model.C[m_idx]
            q_o = Am @ q_pred
            q_o = np.maximum(q_o, EPS)
            q_o /= q_o.sum()
            p_pref = np.exp(Cm - Cm.max())
            p_pref /= (p_pref.sum() + EPS)
            risk = float(np.dot(q_o, np.log(q_o + EPS) - np.log(p_pref + EPS)))
            H_cols = -np.sum(Am * np.log(Am + EPS), axis=0)
            amb = float(np.dot(q_pred, H_cols))
            g += risk + amb
        if action == FEEL:
            G_feel = g
        else:
            G_recall = g

    e_label = f"e={e_level}/{M-1}"
    winner = "FEEL" if G_feel < G_recall else "RECALL"
    print(f"    {e_label:8s}  FEEL={G_feel:.4f}  RECALL={G_recall:.4f}  "
          f"gap={abs(G_feel-G_recall):.4f}  winner={winner}")


# ── What would happen with frame observation modality? ───────
print("\n" + "=" * 70)
print("PROPOSED MECHANISM: Add o_frame observation + C_frame preference")
print("=" * 70)
print("""
A 4th observation modality o_frame in {PAST, PRESENT, FUTURE} would let
the agent's EFE explicitly account for frame transitions.

If C_frame is UNIFORM (no frame preference), it adds only ambiguity terms.
But if C_frame prefers the CURRENT FRAME'S COMPLEMENT (i.e., "I want to
be somewhere other than where I am"), it creates a principled switching drive.

More principled: C_frame prefers a balanced distribution. Since KL(q_o || p_pref)
is minimised when q_o = p_pref, setting C_frame = [1/3, 1/3, 1/3] (uniform)
means any action that KEEPS you in one frame has high risk (concentrated
prediction vs uniform preference). Actions that SPREAD predictions across
frames have lower risk.

This is "frame diversity as preference" — formally identical to epistemic
drive: the agent prefers to be surprised about which frame it's in.

Let's test this numerically.
""")

# Simulate frame observation modality
# A_frame: P(o_frame | v, e, f) = delta(f) (deterministic read of current frame)
A_frame = np.zeros((3, K * M * 3))
for v in range(K):
    for e in range(M):
        for f in range(3):
            A_frame[f, flat_idx(v, e, f, M)] = 0.95
            for f2 in range(3):
                if f2 != f:
                    A_frame[f2, flat_idx(v, e, f, M)] = 0.025
A_frame /= (A_frame.sum(axis=0, keepdims=True) + EPS)

# Test uniform preference C_frame = [0, 0, 0] (= uniform after softmax)
C_frame_uniform = np.array([0.0, 0.0, 0.0])
p_pref_frame = np.array([1/3, 1/3, 1/3])

print("\n  Frame-modality risk per action (C_frame = uniform):")
for a in range(N_ACTIONS):
    q_pred = model.B[a] @ beliefs
    q_pred = np.maximum(q_pred, EPS)
    q_pred /= q_pred.sum()

    q_o_frame = A_frame @ q_pred
    q_o_frame = np.maximum(q_o_frame, EPS)
    q_o_frame /= q_o_frame.sum()

    risk_frame = float(np.dot(q_o_frame,
                               np.log(q_o_frame + EPS) - np.log(p_pref_frame + EPS)))

    f_marg = q_pred.reshape(K, M, 3).sum(axis=(0, 1))
    print(f"    {ACTION_NAMES[a]:12s}  risk_frame={risk_frame:.4f}  "
          f"pred_frame=[{f_marg[0]:.2f}, {f_marg[1]:.2f}, {f_marg[2]:.2f}]  "
          f"P(o_frame)=[{q_o_frame[0]:.2f}, {q_o_frame[1]:.2f}, {q_o_frame[2]:.2f}]")

print("""
  -> If risk_frame differs significantly across actions, adding this modality
     would break the RECALL/FEEL monopoly by penalising actions that
     concentrate the agent in one frame.
""")


# ── Quantify: how much would frame risk rebalance the competition? ──
print("  Total G with frame modality (healthy, from prior):")
for a in range(N_ACTIONS):
    q_pred = model.B[a] @ beliefs
    q_pred = np.maximum(q_pred, EPS)
    q_pred /= q_pred.sum()

    G_base = 0.0
    for m_idx in range(len(model.A)):
        Am = model.A[m_idx]
        Cm = model.C[m_idx]
        q_o = Am @ q_pred
        q_o = np.maximum(q_o, EPS)
        q_o /= q_o.sum()
        p_pref = np.exp(Cm - Cm.max())
        p_pref /= (p_pref.sum() + EPS)
        risk = float(np.dot(q_o, np.log(q_o + EPS) - np.log(p_pref + EPS)))
        H_cols = -np.sum(Am * np.log(Am + EPS), axis=0)
        amb = float(np.dot(q_pred, H_cols))
        G_base += risk + amb

    # Add frame modality
    q_o_frame = A_frame @ q_pred
    q_o_frame = np.maximum(q_o_frame, EPS)
    q_o_frame /= q_o_frame.sum()
    risk_frame = float(np.dot(q_o_frame,
                               np.log(q_o_frame + EPS) - np.log(p_pref_frame + EPS)))
    H_cols_frame = -np.sum(A_frame * np.log(A_frame + EPS), axis=0)
    amb_frame = float(np.dot(q_pred, H_cols_frame))
    G_new = G_base + risk_frame + amb_frame

    print(f"    {ACTION_NAMES[a]:12s}  G_old={G_base:.4f}  "
          f"+frame_risk={risk_frame:.4f}  +frame_amb={amb_frame:.4f}  "
          f"G_new={G_new:.4f}")
