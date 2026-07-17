> **⚠️ SUPERSEDED (2026-07-15).** The headline claims below (esp. the held-out
> prediction "win" and the named-baseline comparison) were later found to be
> circular or based on a weak baseline. See `EMPIRICAL_RECORD.md` for the
> authoritative, corrected record. Kept for provenance only.

# Empirical Validation Results And Status

Generated during the 2026-07-15 validation pass. This document records what has been done so far, what the results mean, and what should not yet be claimed. It is separate from the manuscript on purpose: no manuscript changes should be made until the empirical story is stable.

## Bottom Line

The project has moved from dataset discovery to three levels of empirical testing:

1. Dataset verification and descriptive targets.
2. Summary-statistic data-model checks.
3. A first held-out sequence-prediction test on real ESM observations.

The strongest result so far is that an inertial active-inference version of the temporal-framing model predicts held-out next-beep valence in the Geschwind residual-depression ESM data better than the named single-readout baselines and better than a simple current-valence persistence baseline.

This is promising, but still not the final validation. We have not yet fit participant-level parameters, tested a direct temporal-orientation dataset, or completed OpenNeuro group-level reward/punishment modelling.

## What “Testing On The Data” Means Right Now

There are two distinct testing modes.

### 1. Summary-Statistic Validation

We extract empirical signatures from datasets and compare them to analogous model signatures.

Examples:

- Empirical: event pleasantness predicts current valence.
- Model: `v_reward` predicts composite model valence.
- Empirical: worry persists across beeps.
- Model: future-frame belief persists under stress.
- Empirical: depressed groups show lower autobiographical specificity.
- Model: low `pi_pos` suppresses effective RECALL.

This does not mean the model has explained each participant’s time series. It tests whether the same dynamical patterns appear in data and model.

### 2. Held-Out Sequence Prediction

For the Geschwind ESM dataset, we now also drive the model with real observation sequences:

1. For each participant, each beep is converted into model observations.
2. Event pleasantness is mapped to the external observation channel.
3. Reported affect is mapped to the valence observation channel.
4. The model updates beliefs step by step.
5. The model predicts next-beep valence from its latent predicted state.
6. A single affine calibration is fit on the first 70% of each participant’s sequence.
7. Prediction is tested on the held-out final 30%.

This is a stronger test than the summary-statistic checks, though it is still not full participant-level parameter fitting.

## Datasets Currently Used

### OSF `83cfk` Emotion ESM

Local file:

- `data_raw/osf_83cfk_emotions_data.csv`

Use:

- Measurement reliability.
- Positive/negative affect dynamics.

Current result:

- Rows: 6,321.
- Participants: 91.
- Repeated-item reliability: `r = 0.922`.

Interpretation:

- Useful for showing that emotion-slider measures are reliable enough to support momentary affect dynamics.
- Does not contain direct temporal-orientation labels.

### Geschwind / Bringmann Residual-Depression ESM

Local file:

- `data_raw/geschwind_2013_s004.csv`

Use:

- Event pleasantness.
- Positive/negative affect.
- Worry persistence.
- Held-out next-beep valence prediction.

Current descriptive results:

- Rows: 28,600.
- Participants: 130.
- Valid affect rows: 11,734.
- Mean positive-minus-negative valence: `1.890`.
- Event pleasantness vs current valence: pooled `r = 0.371`.
- Event pleasantness vs next-beep valence: pooled `r = 0.234`.
- Worry lag-1 autocorrelation: pooled `r = 0.598`.

Participant-level targets:

- Event pleasantness vs current valence: `0.360 [0.335, 0.385]`, `n = 129`.
- Event pleasantness vs current worry: `-0.265 [-0.291, -0.239]`, `n = 129`.
- Event pleasantness at `t` vs valence at `t+1`: `0.189 [0.163, 0.215]`, `n = 129`.
- Worry lag-1 autocorrelation: `0.374 [0.336, 0.412]`, `n = 129`.

Held-out sequence prediction:

Test set:

- 129 participants.
- 2,801 held-out next-beep prediction pairs.
- First 70% of each participant’s sequence used for affine calibration.
- Final 30% held out for evaluation.

Results, lower RMSE is better:

| Predictor | Held-Out RMSE | Held-Out r | Notes |
|---|---:|---:|---|
| full active inference with valence inertia | 0.1356 | 0.7043 | best current result |
| current-valence persistence baseline | 0.1392 | 0.6870 | strong simple baseline |
| full active inference without valence inertia | 0.1468 | 0.6417 | improves over named readouts, loses to persistence |
| Pattisapu reward readout | 0.1846 | 0.2735 | captures event/reward alignment but not full sequence dynamics |
| event pleasantness alone | 0.1872 | 0.2205 | direct event predictor only |
| Joffily VFE-derivative readout | 0.1925 | 0.0799 | weak on this target |
| Hesp affective-charge readout | 0.1927 | -0.0496 | weak on immediate next-valence prediction |

Interpretation:

- The original full model was not enough to beat simple affect persistence.
- Adding latent valence inertia, a principled transition prior, improved prediction and beat the persistence baseline.
- This is not hard-coded to the target: valence inertia is a general state-transition prior, not a fitted participant-specific parameter.

### OpenNeuro `ds005356`

Source:

- `https://openneuro.org/datasets/ds005356`

Use:

- Reward/punishment learning substrate.
- Candidate for testing asymmetric hedonic sensitivity.

Current parsed result:

- Subjects listed: 85.
- Subjects with parsed event files: 84.
- Event rows: 39,045.
- Cue rows: 19,908.
- Feedback rows: 19,056.
- Wins: 10,870.
- Losses: 8,119.
- Win rate among feedback rows: `0.570`.

Important limitation:

- The MDD/control phenotype key appears to be stored in a DataLad-annexed Excel file, not directly available from the GitHub raw view.
- We should not make group-level reward/punishment claims until that key is recovered.

### Autobiographical-Memory Specificity Meta-Analysis

Local file:

- `data_raw/autobiographical_memory_Final_AutoData.csv`

Use:

- Group-level constraint on recall specificity and the positive-recall mechanism.

Current result:

- Effect sizes: 181.
- AMT specificity effect sizes: 117.
- Mean Hedges `g`, all rows: `-0.385`.
- Mean Hedges `g`, AMT specificity: `-0.747`.
- Positive-specific cues: `g = -0.787`.
- Negative-specific cues: `g = -0.679`.

Interpretation:

- Supports the direction of the recall-specificity claim.
- This is group-level meta-analytic evidence, not within-person time-series validation.

## Named Cited Baselines

The named cited models are currently operationalised as readout baselines:

- Joffily-Coricelli: `v_model`, the VFE-derivative readout.
- Pattisapu: `v_reward` and arousal.
- Hesp: `v_action`, affective charge / policy revision.
- Smith-Ellsworth: qualitative appraisal taxonomy, not a runnable time-series model unless separately implemented.

Latest named-baseline summary:

| Baseline | Event-Readout r | Lagged Event-Readout r | Persistence Proxy r | Main Limitation |
|---|---:|---:|---:|---|
| Full temporal-framing model | 0.833 | 0.035 | 0.276 | reference |
| Joffily-Coricelli VFE derivative | 0.028 | -0.046 | -0.484 | no present reward channel; no temporal frame |
| Pattisapu reward/arousal | 1.000 | 0.038 | -0.110 | captures event reward but not temporal direction or recall |
| Hesp affective charge | -0.159 | 0.181 | -0.095 | captures some policy revision but not immediate event valence |
| Smith-Ellsworth appraisal | n/a | n/a | n/a | qualitative taxonomy, not a generative time-series model |

Interpretation:

- Pattisapu-style reward valence is very strong for immediate event/reward coupling.
- Hesp-style affective charge is more relevant to policy revision and lagged structure.
- Joffily-style VFE derivative is weak for event-valence prediction in the present ESM target.
- The full model is better as a combined architecture, but claims should specify which target it improves.

## Formal Ablations

Current variants:

- Full adaptive temporal-framing model.
- Full fixed-depth temporal-framing model.
- One-step EFE.
- No habit prior.
- Symmetric hedonic sensitivity.
- No positive-recall gate.

Current broad target results:

- Full adaptive model: `7/7`.
- Full fixed-depth model: `7/7`.
- One-step EFE: `7/7`.
- No habit prior: `6/7`.
- Symmetric hedonic sensitivity: `7/7`.
- No positive-recall gate: `6/7`.

Interpretation:

- The current broad targets separate the positive-recall gate and habit-prior mechanism.
- They do not yet separate adaptive counterfactual depth from fixed-depth or one-step EFE.
- They do not yet separate hedonic asymmetry because the OpenNeuro group/choice target is incomplete.

## Model Improvements Made So Far

### Counterfactual Rollout

Implemented:

- Short rollout over imagined latent states.
- Stored `G_one_step` and `counterfactual_regret`.

Interpretation:

- This makes the counterfactual-emotion section more faithful to the code.
- Current broad empirical targets do not yet prove rollout is necessary.

### Adaptive Counterfactual Horizon

Implemented:

- Horizon increases when current VFE is high relative to the agent’s own running VFE scale.
- This is a state-dependent active-inference mechanism, not target-specific fitting.

Interpretation:

- Mechanistically plausible.
- Not yet shown to beat fixed-depth on the current empirical targets.

### Affective Precision Feedback

Implemented:

- Previous action valence modulates effective policy precision `gamma_eff`.
- Positive `v_action` sharpens policy selection.
- Negative `v_action` relaxes policy selection.
- The update is bounded to avoid runaway precision.

Interpretation:

- This matches the paper’s stated loop better than the earlier readout-only implementation.
- Needs more targeted validation.

### Valence Inertia

Implemented:

- Optional latent-state persistence prior in valence transitions.
- Used in the held-out Geschwind sequence-prediction test.

Result:

- Full active inference without inertia RMSE: `0.1468`.
- Full active inference with inertia RMSE: `0.1356`.
- Current-valence baseline RMSE: `0.1392`.

Interpretation:

- This is the first result where the model beats a strong simple baseline on held-out ESM prediction.
- It is principled: affective states are temporally persistent, and active-inference transition matrices should encode that persistence.

## What We Can Claim Now

Safe claims:

- The model is now tested across multiple datasets rather than one dataset.
- The full temporal-framing architecture covers more target types than the named single-readout baselines.
- The positive-recall gate and habit prior matter for recall-related targets.
- Valence inertia improves held-out next-beep prediction and allows the active-inference model to beat a current-valence persistence baseline on Geschwind ESM.
- Pattisapu’s reward readout is strong for immediate event-reward coupling but does not explain temporal-frame persistence or recall specificity by itself.

Claims to avoid for now:

- “The full model is fully validated.”
- “Adaptive counterfactual depth is empirically proven.”
- “Hedonic asymmetry is empirically proven.”
- “OpenNeuro confirms MDD/control reward asymmetry.”
- “The model predicts temporal orientation in real ESM data.”
- “All cited models fail.” The fair claim is narrower: single-readout operationalisations fail to cover the combined target set.

## Remaining Empirical Work

Highest priority:

1. Recover the OpenNeuro `ds005356` phenotype/group key and task-choice variables.
2. Fit reward/punishment learning or at least per-subject win/loss response structure.
3. Add a direct temporal-orientation ESM dataset with past/present/future thought labels.
4. Add participant-level parameter fitting, then held-out validation.
5. Add stronger tests for adaptive counterfactual depth, such as high-VFE periods requiring deeper rollout than low-VFE periods.

## Manuscript Implications Later

Do not update the manuscript yet unless asked. When ready, the text should:

- Replace generic “validation” with “multi-dataset empirical anchoring” unless referring to the held-out Geschwind prediction.
- Report the held-out sequence-prediction result clearly.
- State that valence inertia was needed for the model to beat a persistence baseline.
- Present named cited models as partial readout baselines, not full reimplementations.
- State that OpenNeuro group-level reward/punishment validation remains pending until the phenotype key is linked.
- Keep the missing temporal-orientation dataset as the main limitation.
