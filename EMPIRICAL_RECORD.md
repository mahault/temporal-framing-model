# Empirical Record — Temporal Framing Model

**Authoritative, honest record of the empirical validation work.** Compiled
2026-07-15. This supersedes the earlier optimistic summaries
(`EMPIRICAL_VALIDATION_RESULTS.md`, `empirical_validation_report.md`), whose
headline claims were later found to be circular or based on weak baselines.
Every result below is reported regardless of whether it favours the model.

Analysis scripts: `empirical_rebuild.py`, `fit_params.py`, `eval_fitted_cv.py`,
`diagnose_mechanisms.py`, `reward_task_analysis.py`, `validate_alt.py`.

---

## 1. Datasets

| Dataset | Source | N | License | Local |
|---|---|---|---|---|
| Geschwind/Bringmann residual-depression ESM | PLOS `10.1371/journal.pone.0060188.s004` (via openESM `0010_geschwind`) | 130 | CC BY-NC 4.0 | `data_raw/geschwind_2013_s004.csv` |
| OSF emotion-reliability ESM | osf.io/83cfk | 91 | public | `data_raw/osf_83cfk_emotions_data.csv` |
| OpenNeuro MEG PST (MDD/CTL) | openneuro.org/datasets/ds005356 | 52 MDD / 38 CTL | CC0 | `data_raw/ds005356/` |
| Probabilistic Reward Task (Pizzagalli) | osf.io/347rm (cpsy.108) | 49–59 MDD | public | `data_raw/prt_347rm/` |
| Reward+punishment reversal learning | osf.io/2gq96 | 64 MDD / 64 HC | public | `data_raw/revlearn_2gq96/` |
| Autobiographical-memory specificity meta-analysis | recovered CSV (source URL unpinned) | 181 effect sizes | — | `data_raw/autobiographical_memory_Final_AutoData.csv` |

The ds005356 MDD/CTL group key + reward/anhedonia phenotype (SHAPS, TEPS, DARS,
BAS/BIS, MASQ, BDI) was recovered from a git-annexed file via the OpenNeuro CRN
API and saved to `data_raw/ds005356/phenotype.csv`.

---

## 2. What holds up: affect-dynamics prediction (the empirical backbone)

**Claim:** the generative temporal-framing model predicts within-person affect
dynamics better than linear baselines out-of-sample.

**Method:** drive the model through each participant's real ESM observation
sequence (event pleasantness + reported valence), read out the policy-averaged
predictive transition, and predict valence *h* steps ahead. Each predictor
(incl. the model) gets an optimal train-fit linear calibration. Global model
parameters (`pi_pos`, `valence_inertia`, `omega_e`) were fit on training
participants and evaluated on held-out participants (nested); the reported
figures are 5-fold CV with **whole participants held out**, 129 participants,
11,734 prediction records. Baselines: naive persistence, AR(1), linear+event,
linear+asymmetric-event, and **direct h-step regression** (the strongest simple
predictor).

**Result (fitted params `pi_pos=2, valence_inertia=0.5, omega_e=5`):**

| Horizon | Model R² | Best simple baseline R² | Skill vs best baseline |
|---|---:|---:|---:|
| h=1 | 0.193 | 0.089 | **+5.8% ± 0.5** |
| h=2 | 0.302 | 0.274 | +1.9% ± 1.0 |
| h=3 | 0.170 | 0.097 | +4.1% ± 0.5 |

- At **h=1 the model roughly doubles the explained variance** of the best
  linear model using the *same inputs* — genuine nonlinear generative structure,
  not the persistence prior (removing inertia collapses it to −13% at h=2).
- Honest caveats: margins are modest; against a *direct 2-step regression* the
  h=2 advantage is small (+1.9%); h=3 gains are on a low base (R²≈0.1). The
  earlier "+12.7% at h=2" figure compared to an *iterated* AR(1), which is a
  weak multi-step forecaster — not reported as headline.

**Readout baselines reframed honestly.** Joffily (`v_model`, a VFE *derivative*)
and Hesp (`v_action`, policy-revision charge) are change signals, not
valence-*level* predictors; scoring them on next-valence-level (r≈0.05, −0.05)
is a category mismatch, not a defeat. They are components of the architecture,
not competing level-predictors.

### 2a. Second-sample replication + decomposition (2026-07-17, `esm_replication.py`, `esm_dig.py`)

Replicated on the independent **osf.io/83cfk reliability ESM** sample (n=91, ~71
beeps each; 12 emotion sliders; NO event/worry items, so the model runs on valence
alone). Same fitted global params, same 5-fold participant CV:

| Sample | config | h1 model/base | h2 | h3 |
|---|---|---|---|---|
| Geschwind (n=129) | full | 0.193/0.090 | 0.302/0.274 | 0.170/0.097 |
| Geschwind | valence-only | 0.202/0.089 | 0.311/0.274 | 0.179/0.097 |
| Geschwind | valence-only, inertia=0 | 0.128/0.089 | 0.007/0.274 | 0.049/0.097 |
| osf_83cfk (n=91) | full (=valence-only) | 0.486/0.475 | 0.377/0.338 | 0.322/0.276 |
| osf_83cfk | inertia=0 | 0.451/0.475 | 0.107/0.338 | 0.070/0.276 |

Findings (now the paper's framing):
- The model **leads at every horizon on both samples**, but margins are
  sample-dependent: ~2.2x at h=1 on the clinical sample, near parity (1.02x) at
  h=1 on the reliability sample where affect is already highly persistent
  (baseline R² 0.475 vs 0.089).
- The Geschwind h=1 win is **not the event channel** (valence-only: 0.202) and
  **not solely persistence** (inertia=0 still beats baseline, 0.128 vs 0.089).
- Multi-step margins (both samples) lean on the persistence-like inertia term.
- Claim wording: "outpredicts baselines on two ESM samples, ~2x at one step
  where affect is volatile; multi-step component leans on persistence." Do NOT
  claim an unqualified 2x.

### 2b. Non-circular latent-state validation

Driven only by valence + event, the model's **future-frame belief tracks the
independently measured worry item** (never given to the model):
**r = 0.166** (n = 11,712). Modest but genuinely out-of-model.

---

## 3. Why the distinctive mechanisms do not show on ESM (diagnosis, not failure)

Asymmetric hedonic sensitivity (`c_pos≠c_neg`) and counterfactual rollout depth
have **no measurable effect** on the ESM prediction. Diagnosed directly
(`diagnose_mechanisms.py`):

- Under passive, observation-driven filtering the agent parks in **FEEL+ENGAGE
  (~82%)**; RECALL (0.5%) and FUTURATE (3.5%) are barely used.
- Both mechanisms act on the *choice among the temporal actions* — precisely the
  actions this regime does not exercise. So `full` vs `symmetric` policies are
  near-identical (mean-policy L1 = 0.066) and their valence predictions
  correlate **r = 0.999**; `full` vs `one_step` correlate **r = 0.998**.
- The counterfactual machinery *is* live (adaptive horizon reaches depth 3 on
  37% of steps); it simply has no leverage on passive affect tracking.

**Conclusion:** ESM valence-level prediction is the wrong probe for these
mechanisms — they require choice / approach–avoidance settings.

---

## 4. The asymmetry / `c_pos` prediction: five tests, not supported behaviourally

**Prediction under test:** depression/anhedonia = **reward-specific** blunting
(`c_pos↓`, `c_neg` preserved).

| # | Test | Data | Result |
|---|---|---|---|
| 0 | Cross-sectional reward-learning vs anhedonia | ds005356 (win-rate; no choices) | MDD worse d≈−0.44 but **uncorrelated with anhedonia** (SHAPS/TEPS r≈0), weak BDI −0.22 |
| 0b | Cross-sectional reward sensitivity vs anhedonia | PRT 347rm | `beta` vs HAMD r=−0.11, vs TEPS r=−0.06; response-bias vs HAMD +0.07; the two reward indices barely agree (r=−0.18) → **null / unreliable** |
| A | Within-person reward reactivity in low vs high mood | Geschwind ESM | slope 0.0385 (high) vs 0.0374 (low), diff +0.001, 50% of people in predicted direction → **null** |
| B | Prognostic: baseline reward sensitivity → recovery | PRT 347rm | `beta` vs %improvement r=−0.35 (**wrong sign**), contradicts response-bias (+0.14); placebo arm + regression-to-mean confounds → **not credible** |
| C | **Direct**: reward vs punishment learning, MDD vs HC | 2gq96 | learning rate reduced in **both** conditions (punish d=−0.58, reward d=−0.50) → **general deficit, not reward-specific — against the prediction** |
| D | Neural Reward Positivity | Pirrung et al. 2025 (same ds005356 sample) | reward-specific vmPFC hypoactivation in MDD → **supported (group-level, neural)** |

**Group-level clinical context (ds005356 questionnaires):** MDD show large
reward/pleasure deficits (TEPS d≈−1.0, BAS d≈−0.96, SHAPS d=+1.66) — but anxiety
is *also* elevated (MASQ anxious-arousal d=+0.95, BIS d=+0.55) and general
severity is huge (BDI d=+3.1), so this is *not* a clean reward-selective
dissociation.

**Verdict (multi-level, and D and C do not actually conflict).** The results
separate cleanly by *level of description*:

- **Reward reactivity / valuation** — supported and reward-specific: the Reward
  Positivity (D) is an immediate neural response to reward receipt and is
  blunted in a reward-specific way in MDD (Pirrung 2025); self-reported hedonics
  (anhedonia questionnaires) show large reward/pleasure deficits. Our `c_pos`
  scales the *reward value* in the preference vector `C` — i.e. it is a
  reactivity/valuation parameter, which is exactly the level RewP indexes. So
  `c_pos↓` is supported where a reward-sensitivity parameter should show up.
- **Reinforcement learning rate** — a *general*, valence-nonspecific reduction:
  the direct reward-vs-punishment test (C) shows learning rate reduced in both
  conditions (punish d=−0.58, reward d=−0.50). This maps to a *different* model
  quantity (overall precision / learning rate), not to `c_pos`.
- **Behavioural choice individual differences** — not resolvable: tests 0, 0b,
  A, B are null, reflecting the well-known weak/noisy link between behavioural
  reward-sensitivity estimates and clinical self-report, plus restricted range.

Reactivity and learning are dissociable, so a reward-specific *reactivity*
deficit (D, `c_pos↓`) and a general *learning-rate* deficit (C) can both be
true. What is *not* supported is `c_pos↓` as something recoverable from
individual differences in behavioural choice. Counterfactual depth remains
unsupported (§3).

---

## 4b. Counterfactual depth: the right datasets, a robust behavioural null

Counterfactual emotions/depth require a choice with a **foregone outcome that is
revealed**. ESM (no choice) and partial-feedback reward tasks (PRT, reversal)
structurally cannot test it. We obtained the correct paradigm — complete-feedback
bandits that store the foregone outcome — from hrl-team/decay_1:

- **Sugawara & Katahira 2021** complete feedback (`S2021c.mat`, n=143, 192 trials)
- **Palminteri 2017** complete feedback (`P2017b.mat`, n=20)

Both store per trial: state (pair), choice, obtained outcome, and **foregone
outcome**. Fair nested model comparison (`reward_cf_fit.py`): FACTUAL (α, β,
update chosen only) vs COUNTERFACTUAL (α, α_c, β, also update the unchosen option
from the foregone outcome), compared on **held-out log-likelihood and BIC**.

| Dataset | Held-out NLL factual | NLL counterfactual | BIC favored | cf-better subjects |
|---|---:|---:|---|---:|
| Sugawara & Katahira (n=143) | 45.90 | 46.68 | factual (16/143) | 47% |
| Palminteri (n=20) | 27.26 | 28.68 | factual (7/20) | 35% |

**Counterfactual updating does not improve choice prediction** — a robust,
well-powered null on the correct paradigm. This is expected: choice is driven by
the chosen option's value, so foregone outcomes barely move the decision variable.
The literature's evidence for counterfactual processing is the **confirmation-bias
asymmetry** (a learning-rate signature, genuine per Cecchi & Palminteri), **neural**
counterfactual-PE signals, and **regret/relief affect** — none of which is a
choice-prediction improvement.

**Conclusion.** Counterfactual depth cannot be validated as a behavioural-fit
improvement. Its honest home (Paper 2) is as the **generator of counterfactual
emotions** (regret/relief) — validated by simulation + the neural/affect
literature — not by choice prediction.

## 4c. Affect prediction at scale (Rutledge GBE) — integration wins, counterfactual doesn't

Rutledge "Great Brain Experiment" happiness data (Dryad, CC0; `data_raw/rutledge_gbe/`):
47,067 participants, ~1.1M momentary-happiness ratings on a safe-vs-gamble task.
Happiness-equation regression (`rutledge_affect_fit.py`), held-out across subjects
(14,803 subjects; 96,624 held-out ratings), forgetting gamma=0.6:

| Model (predict momentary happiness) | Held-out R² |
|---|---:|
| RPE only (single channel) | 0.111 |
| CR+EV+RPE (full reward model) | 0.146 |
| + counterfactual/regret (outcome - foregone safe) | 0.147 |
| CF alone | 0.113 |

- **Integration beats single channels: +31.7% R²** (full vs RPE-only). The
  "insufficient, not wrong" claim, validated on affect at scale.
- **Counterfactual adds +0.2%** to affect prediction — negligible, because the
  regret term (outcome-foregone) is redundant with the reward-prediction terms
  (both outcome-driven).

**Combined with §4b:** counterfactual does not improve prediction of choice
(+0.7%) OR affect (+0.2%), on ~5 datasets. It is a **generative** mechanism
(produces regret/relief; behavioural signature regret->switch t=10.4), NOT a
predictive one. The model's predictive value comes from **multi-channel
integration**, not counterfactual depth.

## 4d. Generality: affect as a readout layer subsumes the happiness equation

The three channels are general operators on any active-inference agent's inference
dynamics; affect sits *on top of* a task-specific generative model. Instantiating
them on a gamble task-model for the Rutledge GBE data (`rutledge_affect_layer.py`;
14,803 subj, 96,624 held-out ratings):

| Predictor (momentary happiness, held out) | R² |
|---|---:|
| present channel only (RPE) | 0.111 |
| happiness equation (CR+EV+RPE) | 0.146 |
| affect layer: forward + present | 0.144 |
| affect layer: forward + present + backward | 0.144 |

- The **forward/EFE channel** natively equals the anticipated value of the chosen
  option (EV for gamble, CR for safe); the **present channel** is the RPE. So
  forward+present (0.144) **recovers the happiness equation (0.146)** with no EV
  injected --- the happiness equation is a *special case* of the framework.
- Forward channel validated non-circularly: present-only 0.111 -> +forward 0.144
  (+30%), forward derived from the task-model's EFE, not injected.
- On a single-shot gamble (no temporal structure) the backward channel adds 0.000,
  honestly; the framework *recovers* but does not *beat* the happiness equation here.
- Where temporal structure exists (ESM), the full model *beats* baselines ~2x (§2).

**Net:** the framework subsumes standard reward-affect models as special cases and
extends them where temporal/framing structure is present. Rutledge = recover;
ESM = beat.

## 5. What can and cannot be claimed

**Can claim:**
- The model predicts short-horizon within-person affect dynamics better than
  linear autoregressive/event baselines out-of-sample (h=1 R² ~2× baseline).
- Its latent temporal-frame state tracks an independently measured symptom (worry).
- Reduced *reward reactivity/valuation* in MDD (the level `c_pos` represents) is
  supported by reward-specific neural blunting (RewP; Pirrung 2025) and by
  self-reported anhedonia.
- Depression also involves a *general* (valence-nonspecific) reduction in
  reinforcement-learning rate/precision (2gq96) — a distinct mechanism.

**Cannot claim:**
- That `c_pos ≠ c_neg` is recoverable from behavioural *choice* individual
  differences — five tests could not detect it (measurement/level gap), and the
  one direct behavioural learning contrast (C) is general, not reward-specific.
- That the general behavioural learning deficit is itself evidence *for* the
  reward-specific asymmetry — it is a separate finding.
- That counterfactual rollout depth improves behaviour/choice prediction — a
  robust well-powered null on the correct complete-feedback paradigm (§4b). Its
  validation is generative (regret/relief) + neural, not behavioural-fit.
- That the PAD emotion-profile figure is external validation (it is simulation
  calibration — parameters were chosen to target the profiles).
- Any MDD/CTL reward *asymmetry* from ds005356 choices (choices were deleted
  from that dataset; only cue+feedback remain).

---

## 6. Implications for the manuscripts

1. **Empirical section (Paper 1):** headline the affect-dynamics prediction +
   frame→worry. Report baselines fairly (direct h-step regression, not iterated
   AR(1)). State margins honestly.
2. **Clinical mechanism (Paper 1 asymmetry extension + Paper 2 bipolar):**
   present a **two-level** depressive account — (i) reduced reward
   reactivity/valuation (`c_pos↓`), supported at the neural (RewP) and
   self-report levels but explicitly *not* validated from behavioural choice;
   and (ii) a **general precision / learning-rate reduction**, supported by the
   direct behavioural learning contrast. Do not conflate the two, and do not
   claim behavioural-choice validation of the asymmetry. Draft text:
   `clinical_mechanism_reframe.md`.
3. **Mechanism probes:** note explicitly that asymmetry and counterfactual depth
   are not testable by passive ESM valence prediction (probe–mechanism mismatch),
   motivating future choice-based tasks.

## 2a-bis. Direct temporal-orientation test + model revision (2026-07-17)

**Dataset:** Mulholland et al. 2023, Consciousness & Cognition (Mendeley
10.17632/zpmm72bg6s.1, CC BY-NC). N=101, 1,458 per-probe mDES rows with per-beep
past/future thought intensity + valence. `temporal_orientation_test.py`.

**Findings (current model):**
- RECALL/rumination branch VALIDATED: reported past-orientation ~ valence
  r=-0.224 pooled, -0.234 within-person. Interaction with trait positivity (pi_pos
  proxy) directionally correct (b=+0.06) but weak; low-mood past-slope -0.224,
  high-mood -0.144 -> attenuation, NOT sign-flip.
- FUTURATE default optimism NOT supported: future-orientation ~ valence r=-0.065
  (mildly negative in unselected sample).
- Frame INVERSION fails: model's latent frame belief driven on valence does NOT
  recover reported orientation (r~=0). Filtered and predicted frame both null.
  Orientation is exogenous context the model isn't given; validated claim is
  frame->affect COUPLING, not affect->frame recovery.

**Model revision (committed):** precision-gate FUTURATE and ABSTRACT symmetrically
with RECALL (v_fut = 0.2 + 0.6*alpha). Motivated by future-negative finding.

**Effect (held out, branch merged to master):**
- Rutledge subsumption unchanged (0.144).
- ESM full model ~unchanged: Geschwind 0.194/0.300/0.168, osf_83cfk 0.485/0.374/0.317.
- ESM NO-INERTIA ablation greatly improved (model's own forward dynamics now carry
  multi-step, not persistence): Geschwind h1/h2 0.14/0.29 (was 0.13/0.007),
  osf h1/h2 0.46/0.34 (was 0.45/0.107). This retires the "multi-step leans on
  persistence" caveat.
- PAD circumplex still 10/10; anger/fear dominance split preserved.
- Clinical numbers shifted: RECALL 29%->26%; chronic-stress future 0.77->0.66,
  present 0.17->0.25 (paper updated, figures regenerated).

## 7. Integrity pass (2026-07-17): removed hand-tuned PAD figure

The PAD/circumplex "emotion-space calibration" figure was REMOVED from the paper.
Reason: the readout centering constants could not be justified. With a principled
center (pleasure at the sigmoid knee pi_pos=2.0; arousal/dominance mean-centered) the
ten profiles separate only 7/10 into correct quadrants; the previously-reported 10/10
required pushing the pleasure center to 1.75 (below the knee, to move 'calm' across)
and the arousal center to 10.5 (below the cross-profile mean 13.32, to fix
'happy'/'alert'). It was hand-tuned to produce a clean result and is non-load-bearing,
so it was cut rather than dressed with a disclaimer. `make_pad_figure.py` retained as a
record of the check; `fig_pad_circumplex.png` no longer referenced.

Also fixed in this pass: chronic-stress and mania descriptions attributed forward
projection to FUTURATE, but on the (precision-gated) model ABSTRACT is the dominant
forward operator (stressed 79% ABSTRACT, FUTURATE ~2%); FUTURATE is a rare high-cost
action (~0-2% across phenotypes) by design. Paper + fig11 panel (f) updated to
forward-framing = FUTURATE+ABSTRACT. Added real citations for Sugawara & Katahira 2021
and Palminteri et al. 2017; corrected two Mulholland author first names.
