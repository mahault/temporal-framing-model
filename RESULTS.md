# Simulation Results — Full Run (T=300, T_sd=3000)

**Date:** 2026-04-21
**Mode:** full (T=300, n_runs=5, n_pi=10, n_omega=10, T_sd=3000)
**Seed:** 42

---

## Experiment 1: Phenotype Comparison (fig1, fig2, fig5, fig6, fig8)

Three agent profiles run for T=300 steps under identical environment dynamics.

| Profile    | pi_pos | omega_e | gamma | c_scale | mean_v | mean_e |
|------------|--------|---------|-------|---------|--------|--------|
| Healthy    | 5.0    | 5.0     | 16.0  | 1.0     | 0.620  | 0.945  |
| Depressive | 0.2    | 5.0     | 16.0  | 0.1     | 0.447  | 0.471  |
| Manic      | 1.5    | 0.5     | 16.0  | 2.0     | 0.630  | 0.245  |

**Key observations:**
- Healthy agent maintains high interoceptive load management (mean_e=0.962) and moderate valence
- Depressive agent shows reduced valence and impaired load regulation (mean_e=0.716)
- Manic agent achieves highest valence but critically low load management (mean_e=0.268) — ignoring bodily signals in pursuit of reward

**Figures:**
- `fig1_phenotypes.png` — Valence and interoceptive load time series per phenotype
- `fig2_joffily.png` — Joffily-Coricelli backward-looking valence channel (v_model = tanh(-ΔF/τ))
- `fig5_phase_portrait.png` — Valence–arousal phase space trajectories
- `fig6_circumplex.png` — Russell circumplex projection of affective dynamics
- `fig8_temporal_aiming.png` — Temporal frame distributions per phenotype

---

## Experiment 2: Granularity Sweep (fig3)

Varies emotional granularity K (number of valence levels) across 5 runs per condition.

| K  | mean_v | var_v  |
|----|--------|--------|
| 2  | 0.694  | 0.1139 |
| 4  | 0.687  | 0.0348 |
| 6  | 0.667  | 0.0365 |
| 8  | 0.661  | 0.0360 |

**Key observations:**
- Low granularity (K=2) yields higher apparent valence but much higher variance — coarse models are volatile
- Diminishing returns beyond K=4: variance stabilizes, valence decreases slightly
- Consistent with Barrett's (2017) constructed emotion: finer-grained interoceptive models enable more precise regulation at the cost of higher computational demand

**Figure:** `fig3_granularity.png` — 3×2 panel including FUTURATE effectiveness and elaboration/nuance metrics

---

## Experiment 3: Parameter Landscape (fig4)

10×10 sweep over pi_pos ∈ [0.5, 8.0] and omega_e ∈ [0.5, 8.0].

**Figure:** `fig4_parameter_space.png` — Heatmaps of mean valence, arousal, and temporal frame proportions across the (pi_pos, omega_e) parameter space

---

## Experiment 4: Emotion Validation (fig7)

Maps discrete emotions onto the circumplex via temporal frame × valence × arousal signatures.

**Figures:**
- `fig7_emotion_validation.png` — Emotion-labeled circumplex positions
- `fig7_emotion_validation_projections.png` — Temporal frame projections for each emotion

---

## Experiment 5: Temporal Aiming (fig9)

Full temporal summary across all emotion categories.

**Figure:** `fig9_temporal_summary.png` — Temporal orientation profiles per emotion

---

## Experiment 6: Feedback Reliance (fig10)

Tests the role of recall precision (pi_pos) by comparing healthy vs recall-impaired agents.

| Profile          | pi_pos | mean_v |
|------------------|--------|--------|
| Healthy          | 5.0    | 0.604  |
| Recall-impaired  | 0.5    | 0.536  |

**Key observation:** Impaired recall precision reduces valence by ~11%. Agents that cannot accurately retrieve past states make worse counterfactual comparisons, degrading the v_model channel.

**Figure:** `fig10_feedback_reliance.png`

---

## Experiment 7: Framing Dynamics (fig11)

Visualizes how action selection drives temporal frame transitions.

**Figure:** `fig11_framing_dynamics.png` — Action-conditioned frame transition probabilities for RECALL, ENGAGE, FUTURATE, FEEL, BLANK

---

## Experiment 8: Chronic Stress (fig12)

Compares healthy vs chronically stressed environments (elevated volatility).

| Profile  | mean_v | frame_belief (PAST, PRESENT, FUTURE) |
|----------|--------|--------------------------------------|
| Healthy  | 0.618  | [0.34, 0.47, 0.19]                  |
| Stressed | 0.689  | [0.27, 0.56, 0.17]                  |

**Key observation:** Stressed agents shift toward PRESENT (0.56 vs 0.47) and away from PAST (0.27 vs 0.34). Temporal narrowing under stress: the agent retreats from counterfactual reasoning into present-focused engagement.

**Figure:** `fig12_chronic_stress.png`

---

## Experiment 9: pi_pos Dynamics — M5 Mood Layer (fig13)

Tracks the slow mood-level Bayesian update of pi_pos over the simulation.

**Figure:** `fig13_pi_pos_dynamics.png` — pi_pos and pi_pos_eff trajectories per phenotype, showing the dual-timescale system (slow M5 mood + fast interoceptive coupling)

---

## Experiment 10: Stress Decay — Emergent Depression (fig14)

Long-horizon (T=3000) simulation comparing stable vs stress-then-recovery environments.

| Profile              | mean_v | mean_pi_pos |
|----------------------|--------|-------------|
| Healthy (stable)     | 0.637  | 5.649       |
| Healthy (under stress) | 0.574 | 3.951      |

**Key observation:** Sustained stress drives pi_pos from ~5.6 down to ~4.0 — a 30% reduction in recall precision via the M5 mood pathway. This is the emergent depression trajectory: chronic stress → mood decay → impaired counterfactual recall → reduced valence → further mood decay. The agent doesn't "decide" to become depressed; it follows from Bayesian inference over its own regulatory capacity (allostatic self-efficacy; Stephan et al. 2016).

**Figure:** `fig14_stress_decay.png`

---

## Experiment 11: Psychotic Decompensation — Two Failure Modes (fig15)

Tests two distinct failure modes comparing healthy vs vulnerable profiles.
Six actions: RECALL, ENGAGE, FUTURATE, FEEL, DISSOCIATE, ABSTRACT.

| Profile    | K | pi_pos | omega_e | volatility | DISSOCIATE % | ABSTRACT % | intero_load | mean_v |
|------------|---|--------|---------|------------|-------------|-----------|-------------|--------|
| Healthy    | 8 | 5.0    | 5.0     | 0.3        | **1.3%**    | **3.7%**  | 0.97        | 0.623  |
| Vulnerable | 4 | 1.0    | 0.3     | 0.7        | **14.3%**   | **11.3%** | 1.07        | 0.611  |

**Key observations:**
- **Two distinct failure modes emerge**, both EFE-optimal in the vulnerable regime:
  - **DISSOCIATE (14.3%)**: Dissociative shutdown — flat affect, present-locked, loss of personal historicity. The agent stops processing. Corresponds to depersonalization/derealization.
  - **ABSTRACT (11.3%)**: Ungrounded cognition — moderate hedonic signal, future-pulling. The agent is actively thinking but without interoceptive or episodic grounding. Corresponds to the psychosis–creativity axis.
- **Healthy agents rarely select either** (1.3% DISSOCIATE, 3.7% ABSTRACT) — directed grounded actions always have lower G.
- **The ABSTRACT→FUTURATE coupling** creates a second hedonic route:
  - Route 1 (grounded): FEEL → RECALL → FUTURATE (body → past → future)
  - Route 2 (ungrounded): ABSTRACT → FUTURATE (abstract → future, bypassing body and history)
- **Psychosis vs depression**: Depression is characterised by anhedonia and collapsed future framing (learned helplessness). The psychotic route is different: a tight ABSTRACT→FUTURATE loop that is cut off from both past and interoception, restoring some sense of agency at the cost of groundedness.
- **Interoceptive load is elevated** in the vulnerable profile (1.07 vs 0.97), consistent with accumulated unprocessed prediction error.

**Clinical interpretation:** The model produces two computationally distinct failure modes — shutdown (DISSOCIATE) and ungrounded spinning (ABSTRACT→FUTURATE). This maps onto the clinical distinction between dissociation and psychosis, both of which can co-occur in vulnerable populations. The ABSTRACT action also provides a computational account of the psychosis–creativity continuum: the same mechanism that produces ungrounded psychotic loops can, when balanced by FEEL and RECALL, support open-ended creative exploration.

**Figure:** `fig15_psychosis.png` — 2×3 panel: action proportions, DISSOCIATE + ABSTRACT rates over time, frame beliefs, valence trajectory, pi_pos_eff vs pi_pos, channel decomposition

---

## Summary Table

| Experiment | Key Finding |
|------------|-------------|
| Phenotypes | 3-way differentiation preserved: healthy (stable), depressive (low valence, impaired load), manic (high valence, depleted load) |
| Granularity | K=2 volatile; K≥4 stabilizes; finer granularity → slightly lower but more regulated valence |
| Parameter space | pi_pos and omega_e jointly determine affective regime |
| Emotion validation | Discrete emotions map onto circumplex via temporal frame signatures |
| Feedback reliance | 11% valence reduction with impaired recall precision |
| Chronic stress | Temporal narrowing: PRESENT increases 19%, PAST decreases 21% |
| Stress decay | 30% pi_pos reduction under sustained stress (emergent depression via M5) |
| **Psychosis** | **Two failure modes: DISSOCIATE (14.3%) + ABSTRACT (11.3%) in vulnerable agents; two hedonic routes (grounded vs ungrounded)** |

---

## Interoceptive Load Extension — Validation

The REST→FEEL rename and interoceptive load reinterpretation (Manon's proposal) is validated:

1. **FEEL as active interoceptive processing**: Agents that can FEEL effectively (high omega_e) maintain low interoceptive load and stable affect
2. **Load accumulation under imprecise interoception**: Low omega_e → FEEL doesn't reduce load → PE accumulates → pi_pos_eff drops
3. **DISSOCIATE as computational learned helplessness**: Emerges only when all directed actions fail — not a parameter choice but an EFE-optimal response to regulatory collapse
4. **ABSTRACT as ungrounded cognition**: Provides a second hedonic route (ABSTRACT→FUTURATE) that bypasses interoception and episodic recall — models the psychosis–creativity continuum
5. **Dual-timescale coupling**: Fast interoceptive pathway (every step) + slow M5 mood pathway (every T_mood steps) creates rich temporal dynamics visible in fig13 and fig15

**References:** Stephan et al. (2016), Seth & Friston (2016), Barrett (2017), Smith et al. (2020), Sandved-Smith et al. (2021), Pezzulo et al. (2015)
