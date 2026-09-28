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

## 8. Mood-layer bug fix: valence-based mood -> robust diathesis-stress (2026-07-17)

**Bug (found during phenotype scrutiny):** the M5 mood layer observed mean VFE.
VFE is (a) nearly flat across pi_pos (range 0.17) and (b) adapts away under chronic
stress (stressed mean VFE 4.76 vs stable 4.69) -- a well-adapted depressed agent has
NORMAL VFE. So VFE structurally cannot detect depression; the mood did a sticky random
walk -> stochastic basin coin-flip. The old "emergent depression under chronic stress"
claim was contradicted across seeds (stressed ended lower pi_pos 0/10; higher 6/10),
on BOTH the old fixed-optimism and current models (not caused by the FUTURATE gating).

**Fix (agent.py):** the mood observes believed-valence LEVEL, not VFE -- the one
signal that stays low in depression (Beck's negative schema; Eldar & Niv 2016
mood-as-reward-level). Likelihood anchors neutral valence (0.5) to the sigmoid knee
(pi_pos=2, the rumination threshold); slope spreads the observed valence range across
the pi_pos axis. Above-neutral valence -> high pi_pos (resilience); persistently
below-neutral -> below the knee, where RECALL rumination sustains a low-mood attractor.

**Result -- robust diathesis-stress (Monroe & Simons 1991), 8 seeds, T=1500:**
| condition | final pi_pos | depressed(<2) |
|---|---|---|
| healthy + calm | 7.43 +/- 0.02 | 0/8 |
| healthy + stress | 5.16 +/- 0.69 | 0/8 (resilient -- stress alone insufficient) |
| vulnerable + calm | 4.22 +/- 1.04 | 0/8 (vulnerability alone insufficient) |
| vulnerable + stress | 0.76 +/- 0.17 | 8/8 (self-sustaining depressive collapse) |

Only the conjunction produces depression. Vulnerability = low pi_pos + blunted reward
sensitivity (c_scale, anhedonia) + weak interoception; stress = high volatility.

**Verified the fix breaks nothing:** ESM Geschwind 0.194/0.301/0.171 & osf
0.485/0.375/0.319 (unchanged); Rutledge head-to-head 0.144 (unchanged); Mulholland
orientation r=-0.234 (data-side); feedback-reliance RECALL 26%->0% (unchanged);
phenotypes distinct. Shifted (chronic-stress DEMO, seed 42): stressed future-frame
0.66->0.57, present 0.25->0.32, reward valence -0.42->-0.35, ABSTRACT 79%->66%
(qualitative story unchanged). Experiment redesigned as a 2x2 diathesis-stress;
fig14 rebuilt (STRESS_DECAY_PROFILES, plot_stress_decay).

## 9. Diathesis-stress prediction CONFIRMED on real ESM (2026-07-19, diathesis_stress_test.py)

The mood mechanism (§8) predicts vulnerability (a) lowers baseline mood and (b)
amplifies affective reactivity to stress -- a vulnerability x stress interaction, not
either main effect. Tested on the Geschwind ESM (n=128), whose baseline NEUROTICISM
score (col 12; person-constant) is NEVER used in the affect-dynamics fitting, so this
is a held-out qualitative prediction. Momentary valence = (cheerful+relaxed) -
(worried+fearful+sad); stress signal = per-beep event (un)pleasantness.

Result:
- corr(neuroticism, mean valence) = -0.53  (vulnerability -> lower baseline mood) ✓
- corr(neuroticism, event->valence reactivity slope) = +0.31, n=128  (amplified
  reactivity: low-neuroticism slope +1.02 vs high +1.27) ✓
- pooled within-person event x neuroticism interaction: b=+0.016, SE=0.0038, t=+4.2
  (11,315 beeps) ✓

Both signatures match the mood layer's simulation (lower baseline + steeper
stress-reactivity; only the conjunction collapses). Reproduces the established
neuroticism x stress reactivity effect (Geschwind/Wichers). NOT tested (Geschwind is
20 days, already-remitted): the months-scale transition into sustained low mood, which
the mechanism additionally predicts -- awaits longer longitudinal data (openESM
candidates: Nepal 2024 #0040 441 days; Jang 2024 #0017 402 days; Leuven 3-wave on
request). Paper upgraded: diathesis-stress is now a confirmed core prediction, not a
demonstration-awaiting-data.

---

## 10. AAAI-27 revision (2026-09-27): frame gating, honest worry test, frame ablation, regret rework, sensitivity

Model change: `agent.py` frame-gated precision, `frame_gain=1.0` (experiments.FRAME_GAIN);
`frame_gain=0` reproduces every pre-revision number exactly (tests/test_frame_gating.py).

### 10a. Frame-worry association, worry removed from input (`frame_worry_multilevel.py`)
Geschwind only (osf_83cfk has no worry item). 11,712 beeps, 129 participants.

| model | worry in input | raw r | partial r (v_t, e_t) | within-person r | z raw | z adjusted | z within |
|---|---|---:|---:|---:|---:|---:|---:|
| gated (paper) | yes | 0.078 | 0.006 | 0.008 | 6.3 | 0.5 | 0.7 |
| gated (paper) | no | 0.067 | 0.007 | 0.009 | 5.9 | 0.7 | 0.9 |
| ungated (old) | yes | 0.142 | -0.008 | 0.028 | 7.7 | -0.6 | 2.4 |
| ungated (old) | no | 0.121 | -0.003 | 0.024 | 6.6 | -0.2 | 2.3 |

Verdict: null after controls. The earlier r = 0.166 claim is withdrawn.

### 10b. Frame ablation, paper pipeline and FITTED params (`esm_frame_ablation.py`)
Held-out R2 (5-fold participant CV, fold SD), see reviews/esm_frame_ablation.md for all rows and CIs.

| sample | h | best baseline | full (gated) | ungated | clamp PRESENT | clamp FUTURE | channels only |
|---|---|---:|---:|---:|---:|---:|---:|
| Geschwind | 1 | 0.090 | 0.193 (.03) | 0.194 | 0.194 | 0.195 | 0.119 |
| Geschwind | 2 | 0.274 | 0.299 (.03) | 0.301 | 0.300 | 0.304 | 0.281 |
| Geschwind | 3 | 0.097 | 0.167 (.03) | 0.171 | 0.168 | 0.177 | 0.114 |
| osf_83cfk | 1 | 0.475 | 0.485 (.07) | 0.485 | 0.485 | 0.487 | 0.501 |
| osf_83cfk | 2 | 0.335 | 0.373 (.08) | 0.375 | 0.373 | 0.376 | 0.369 |
| osf_83cfk | 3 | 0.274 | 0.316 (.08) | 0.319 | 0.316 | 0.321 | 0.311 |

Participant-bootstrap 95% CI, full minus best baseline: Geschwind h1 [+.089,+.118], h2 [+.011,+.038],
h3 [+.056,+.084]; osf h1 [-.001,+.023], h2 [+.025,+.051], h3 [+.027,+.056].
Full minus ungated: within 0.005 everywhere. The frame does not carry the prediction.

### 10c. Regret rework (`regret_model_fit.py`)
After-loss switching, foregone better vs same (paired t over subjects); held-out NLL/trial.

| dataset | human | factual | cf-value | regret bias |
|---|---|---|---|---|
| Sugawara n=143 | .458/.325 (+.133, t 10.4) | .479/.426 (+.053) NLL .602 | .511/.418 (+.093) NLL .617 | .501/.425 (+.077) NLL .603 |
| Palminteri n=20 | .429/.161 (+.268, t 7.3) | .394/.301 (+.093) NLL .354 | .444/.281 (+.163) NLL .368 | .467/.297 (+.170) NLL .365 |

### 10d. Simulation headline numbers under gating (`run.py`, seed 42; `diathesis_seeds.py`, 8 seeds)
RECALL healthy 0.44 vs impaired 0.00 (FEEL 0.18 vs 0.68); policy entropy impaired 0.255 vs healthy 0.217.
Stressed frame (P, Pr, F) = (0.16, 0.40, 0.44) vs healthy (0.28, 0.51, 0.20); ABSTRACT 0.40 vs 0.03; v_reward -0.15 vs +0.04.
Diathesis: healthy_calm 7.48 +/- 0.00, healthy_stress 7.36 +/- 0.09 (0/8 below knee), vulnerable_calm 5.13 +/- 0.56 (0/8), vulnerable_stress 1.33 +/- 0.33 (8/8).

### 10e. Sensitivity (`sensitivity_bframe.py`)
See reviews/sensitivity_results.md (appended below when the sweep finished).
# Sensitivity of the qualitative results (2026-09-27)

Script: `sensitivity_bframe.py`, 4 seeds per cell, T=300, T_diathesis=3000. Paper defaults: stickiness 0.70/0.75/0.90/0.80, pull scale 1.0, gain 1.0.

## recall_collapse (rows: stickiness s; cols: pull scale k)

| s \ k | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 |
|---|---:|---:|---:|---:|---:|
| 0.5 | 0.387 | 0.374 | 0.391 | 0.368 | 0.245 |
| 0.6 | 0.360 | 0.380 | 0.331 | 0.358 | 0.281 |
| 0.7 | 0.361 | 0.390 | 0.358 | 0.360 | 0.266 |
| 0.8 | 0.370 | 0.377 | 0.323 | 0.321 | 0.220 |
| 0.9 | 0.386 | 0.393 | 0.337 | 0.322 | 0.227 |
| 0.95 | 0.367 | 0.362 | 0.350 | 0.319 | 0.213 |

## future_fixation (rows: stickiness s; cols: pull scale k)

| s \ k | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 |
|---|---:|---:|---:|---:|---:|
| 0.5 | 0.002 | 0.084 | 0.190 | 0.209 | 0.159 |
| 0.6 | 0.015 | 0.068 | 0.175 | 0.240 | 0.182 |
| 0.7 | 0.020 | 0.087 | 0.199 | 0.275 | 0.232 |
| 0.8 | 0.033 | 0.089 | 0.256 | 0.294 | 0.236 |
| 0.9 | 0.042 | 0.091 | 0.284 | 0.290 | 0.260 |
| 0.95 | 0.040 | 0.121 | 0.310 | 0.318 | 0.241 |

## diathesis (rows: stickiness s; cols: pull scale k)

| s \ k | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 |
|---|---:|---:|---:|---:|---:|
| 0.5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 |
| 0.6 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 |
| 0.7 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 |
| 0.8 | 1.000 | 0.750 | 1.000 | 1.000 | 0.750 |
| 0.9 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 |
| 0.95 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 |

## entropy_diff (rows: stickiness s; cols: pull scale k)

| s \ k | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 |
|---|---:|---:|---:|---:|---:|
| 0.5 | 0.011 | 0.021 | 0.033 | 0.016 | -0.022 |
| 0.6 | 0.016 | 0.039 | 0.031 | 0.036 | -0.026 |
| 0.7 | 0.010 | 0.035 | 0.030 | 0.028 | -0.028 |
| 0.8 | 0.014 | 0.023 | 0.013 | 0.013 | -0.045 |
| 0.9 | 0.001 | 0.014 | 0.012 | 0.006 | -0.050 |
| 0.95 | -0.023 | -0.022 | -0.004 | -0.029 | -0.059 |

## Frame gain g (default s, k)

| g | recall_collapse | future_fixation | diathesis | entropy_diff (impaired - healthy) | q(PAST) stressed |
|---|---:|---:|---:|---:|---:|
| 0.0 | 0.258 (0.013) | 0.265 (0.015) | 1.00 | -0.070 (0.010) | 0.108 |
| 0.25 | 0.365 (0.024) | 0.114 (0.039) | 1.00 | +0.035 (0.022) | 0.241 |
| 0.5 | 0.380 (0.023) | 0.140 (0.060) | 1.00 | +0.043 (0.008) | 0.220 |
| 0.75 | 0.407 (0.014) | 0.173 (0.045) | 1.00 | +0.058 (0.011) | 0.210 |
| 1.0 | 0.386 (0.039) | 0.245 (0.050) | 1.00 | +0.037 (0.013) | 0.168 |

Fraction of the s x k grid with recall_collapse > 0: 1.00; future_fixation > 0: 1.00; diathesis holding in all seeds: 0.77; in a majority of seeds: 0.87.

## 11. Round 2 (2026-09-27/28): loader bug, unified ESM re-evaluation, participant-holdout regret, choice task, ablations

**Loader bug (affects every earlier Geschwind ESM number).** `data_raw/geschwind_2013_s004.csv` holds one
row per (participant, day, beep) for EACH of two six-day periods (`st_period` 0/1, eight weeks apart).
`empirical_rebuild.load_participants` sorted by (day, beep) only, interleaving the periods, so "lag 1" was
the same beep slot two months earlier and the true previous beep sat at lag 2 (pooled lag-1 r 0.31,
lag-2 r 0.53). Fixed: sequences ordered by (period, day, beep), agent reset at the boundary, targets
never cross it (`same_segment`, `target`). Correct lag-1 r = 0.614. The earlier "2x over baselines at
one step" (0.193 vs 0.090) was an artefact of this.

**Descriptives (correct).** Geschwind: 129 participants, 11,734 records, 248 segments, median 12 sampling
days (5 to 14), median 96 beeps; 11,712 with worry, 11,452 with event, 128 with neuroticism; one-step
targets 11,486. osf_83cfk: 91 participants, 6,321 records, median 71 beeps.

### 11a. Unified ESM evaluation (`esm_eval_v2.py --workers 20`, reviews/esm_eval_v2.md, .json)
Nested selection of (rho_pos, inertia, omega_e) on training participants per fold from the 18-point grid
(selected: inertia 0.5 every fold; rho_pos 2 or 3; omega_e 3 on Geschwind, 3 or 5 on osf).

| sample | h | persistence | direct v_t | direct (v_t, v_t-1) | channels only | full (g=1) | inert (g=0) | clamp FUTURE | trans g_B=4 | no inertia |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Geschwind | 1 | .220 | .372 | .407 | .382 | .398 | .399 | .399 | .395 | .363 |
| Geschwind | 2 | .033 | .268 | .306 | .280 | .299 | .303 | .303 | .293 | .225 |
| Geschwind | 3 | -.069 | .219 | .253 | .230 | .248 | .253 | .255 | .238 | .132 |
| osf | 1 | .383 | .475 | .498 | .500 | .484 | .485 | .485 | .482 | .450 |
| osf | 2 | .168 | .335 | .368 | .368 | .373 | .375 | .375 | .365 | .302 |
| osf | 3 | .062 | .274 | .307 | .309 | .315 | .320 | .319 | .304 | .171 |

CIs (participant bootstrap): full - two-lag regression: Geschwind h1 [-.015,-.002], h2 [-.013,.000],
h3 [-.011,.002]; osf h1 [-.023,-.004], h2 [-.008,.016], h3 [-.004,.020]. full - inert: negative with CIs
excluding zero everywhere (-.001 to -.005). Verdict: the model matches a two-lag linear regression and does
not exceed it; the frame (gated, clamped, transition-gated) changes nothing; inertia carries the memory.

### 11b. Frame-worry, worry removed from input, correct ordering (same script)
| model | raw r | partial r (v_t, e_t) | within r | z raw | z adj | z within |
|---|---:|---:|---:|---:|---:|---:|
| gated g=1 | 0.018 | -0.014 | 0.004 | 1.9 | -1.5 | 0.3 |
| inert g=0 | 0.093 | 0.006 | 0.067 | 6.9 | 0.5 | 4.6 |
| trans g_B=4 | 0.026 | -0.009 | 0.011 | 2.5 | -1.0 | 0.8 |
| clamp FUTURE | 0.001 | 0.001 | -0.004 | 0.1 | 0.1 | -0.4 |
The inert model's within-person association is explained by concurrent valence (partial z 0.5).

### 11c. Regret, participant-level holdout (`regret_participant_holdout.py --workers 6`, reviews/regret_participant_holdout.md)
Group-level fits, 5 participant folds; generated contrasts from 20 simulated runs per held-out participant.
Sugawara (n=143): human +0.133 [0.108, 0.159]; factual +0.063, NLL/trial 0.6182; cfvalue collapses to factual
(alpha_cf -> 0); regret bias +0.085 [0.081, 0.090], NLL gain 0.0005 [-0.0001, 0.0011], 57% better;
regret frame-gated identical (w_past SD 0.07); salience-gated +0.087, gain 0.0006 [-0.0001, 0.0012].
Palminteri (n=20): human +0.268 [0.201, 0.342]; factual +0.115; regret +0.177 [0.160, 0.194], gain 0.0030
[-0.0001, 0.0060], 70% better; frame-gated +0.175, gain 0.0027 [-0.0005, 0.0058].

### 11d. Frame in a choice task (`frame_choice_task.py`, reviews/frame_choice_results.md)
Implied discount (delayed:immediate ratio at G tie): g=0 5.03; g=1 clamped FUTURE 1.00, PAST/PRESENT > 50;
inferred after RECALL (q=.44,.39,.17) 12.63, after ENGAGE (.12,.70,.18) 7.65, after FUTURATE (.04,.18,.78) 1.00.

### 11e. rho_pos decoupling (`rho_pos_decoupling.py`, 4 seeds) and one-factor stress (`stress_one_factor.py`)
RECALL collapse: coupled 0.386 (0.039); D only 0.057 (0.038); targets only 0.386 (0.039).
Diathesis final rho_pos vulnerable+stress: coupled 1.08 (0.20) 4/4 below knee; mood cut 1.33 (0.49) 3/4;
D healthy 1.43 (0.30) 4/4. Healthy+stress 7.3-7.4 throughout.
One factor: q(FUTURE) healthy .20; rho_pos 2.5 .28 (ABSTRACT .15); omega_e 0.5 .22; c 2.0 .22; volatility 0.9
.19 (v_reward .03); all four .44 (ABSTRACT .42, v_reward -.19).

### 11f. Diathesis and orientation statistics (`diathesis_stats_v2.py`, `mulholland_stats.py`)
Geschwind: n=128, 11,409 beeps; r(neur, mean valence) -0.526 [-0.632,-0.412]; r(neur, slope) +0.289
[0.145, 0.416]; two-stage slope on neuroticism t=+3.62; pooled interaction b=+0.0005, cluster-robust z=+2.60
(naive t=+3.90). Mulholland: 91 participants, 1,442 probes; within r(past, valence) -0.234, r(future,
valence) -0.069; past slope z=-5.29; past x trait interaction b=+0.064, SE 0.058, z=1.11; future slope z=-1.35;
r(trait, per-person past slope) +0.013 [-0.161, 0.208].

### 11g. Sensitivity counts corrected from reviews/sensitivity_results.md
RECALL collapse range .213-.393; future fixation range .002-.318 (all > 0); diathesis all-seeds in 23/30
(failures: the six k=1.4 cells and (0.8,0.8)), majority in 26/30; entropy diff positive in 20/30 (negative
in the k=1.4 column and the s=0.95 row). Gain sweep: RECALL collapse .26 (g=0) -> .39 (g=1); future
fixation .27 at g=0; diathesis 1.00 at every g.

## 12. Round 3 (2026-09-28): can the model predict better? No.

Directive: find a real participant-held-out predictive gain over STRONG baselines or establish there is
none. Scripts: `esm_eval_v3.py --workers 20` (reviews/esm_eval_v3.md, .json, .log), `plot_esm_v3.py`
(figures/fig_esm_v3.png), `esm_worry_pred_v3.py --workers 6` (reviews/esm_worry_pred_v3.md). Same
participant folds as section 11 (seed-0 shuffle, 5 folds), same preprocessing, horizons 1 to 6, participant
bootstrap CIs (1000 resamples) on pooled R2 differences. Ridge penalties by inner participant-grouped CV
on training participants (chosen 1 to 100); adaptation penalty chosen on 30 training participants by
simulating the protocol (chosen 300 at every fold and horizon, heavy shrinkage toward the pooled fit).
Pooled generative parameters selected per fold from a 36-point grid (rho_pos, inertia, omega_e, asymmetry):
Geschwind rho_pos 3, inertia 0.65, omega_e 3, (c_pos, c_neg) = (0.6, 1.6) in every fold; osf rho_pos 2,
inertia 0.65, omega_e 3 or 5, same asymmetry. Descriptives as section 11 (129 / 91 participants,
11,734 / 6,321 records; 11,486 / 6,230 one-step targets).

### 12a. Valence level, pooled protocol (parameters and coefficients from other participants only)
| predictor | G h1 | G h2 | G h3 | G h4 | G h5 | G h6 | osf h1 | osf h2 | osf h3 | osf h4 | osf h5 | osf h6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| persistence | .224 | .039 | -.062 | -.139 | -.163 | -.204 | .402 | .195 | .091 | .021 | .000 | -.032 |
| two-lag regression (ar2) | .410 | .309 | .257 | .229 | .215 | .202 | .512 | .387 | .326 | .295 | .281 | .268 |
| six-lag ridge (ar6) | .432 | .344 | .300 | .274 | .261 | .250 | .532 | .420 | .371 | .341 | .329 | .320 |
| six-lag ridge + events (ar6e) | .433 | .343 | .300 | .274 | .262 | .250 | | | | | | |
| kitchen ridge (lags, events, time of day) | .438 | .348 | .303 | .277 | .265 | .251 | .533 | .421 | .371 | .342 | .330 | .320 |
| two-regime switching AR (msar2) | .410 | .312 | .264 | .236 | .223 | .211 | .514 | .392 | .335 | .305 | .297 | .285 |
| frame model, gated (g=1) | .408 | .317 | .268 | .240 | .227 | .214 | .506 | .405 | .353 | .322 | .308 | .294 |
| frame model, inert (g=0) | .409 | .318 | .271 | .244 | .233 | .222 | .506 | .406 | .354 | .324 | .312 | .298 |
| transition-gated (g_B=4) | .405 | .309 | .255 | .221 | .202 | .182 | .504 | .399 | .346 | .311 | .294 | .277 |
| channels only | .386 | .287 | .238 | .206 | .198 | .180 | .520 | .396 | .345 | .316 | .306 | .289 |
| kitchen + model state (aug) | .437 | .347 | .302 | .277 | .265 | .252 | .534 | .425 | .377 | .348 | .337 | .323 |
| aug without frame features | .438 | .347 | .303 | .276 | .264 | .251 | .535 | .425 | .377 | .349 | .337 | .324 |
| aug without channels | .437 | .348 | .302 | .277 | .265 | .253 | .533 | .425 | .375 | .346 | .334 | .323 |
| kitchen + model expectation only | .438 | .348 | .303 | .277 | .265 | .251 | .533 | .422 | .373 | .343 | .331 | .320 |

CIs (Geschwind / osf): gated model minus kitchen h1 -.030 [-.036,-.023] / -.027 [-.036,-.019]; h3 -.035
[-.047,-.024] / -.019 [-.033,-.004]; h6 -.036 [-.048,-.025] / -.026 [-.043,-.010]. aug minus kitchen h1
-.000 [-.001,+.000] / +.002 [-.000,+.004]; h3 -.001 [-.003,+.000] / +.005 [+.001,+.009]; h6 +.001
[-.002,+.004] / +.003 [-.004,+.010]. aug minus aug-without-frame: .000 at every horizon in both samples
(largest |point| .001). gated minus inert: -.001 [-.002,-.000] (G h1) to -.008 [-.011,-.004] (G h6);
-.000 to -.004 on osf, CIs exclude zero at every horizon. transition-gated minus inert: -.004 (G h1) to
-.040 [-.053,-.026] (G h6); -.002 to -.021 on osf. kitchen minus ar2: +.028 [+.022,+.034] (G h1) to +.049
(G h6); +.020 to +.052 on osf. msar2 minus ar2: +.000 [-.005,+.006] (G h1), +.009 [+.004,+.015] (G h6);
+.002 to +.017 on osf.

### 12b. Valence level, adapted protocol (first period / first half of each held-out participant used to
adapt; scored on the rest; rows at h1: 5,655 Geschwind, 3,113 osf)
| predictor | G h1 | G h3 | G h6 | osf h1 | osf h3 | osf h6 |
|---|---:|---:|---:|---:|---:|---:|
| pooled kitchen ridge | .474 | .354 | .306 | .573 | .394 | .314 |
| adapted kitchen ridge | .475 | .357 | .310 | .580 | .408 | .331 |
| per-participant two-lag (no pooling) | .368 | .116 | -.018 | .504 | .349 | .284 |
| switching AR, pooled, regime filtered through the adaptation segment | .443 | .311 | .259 | .557 | .362 | .283 |
| frame model, pooled parameters and calibration | .441 | .315 | .256 | .540 | .378 | .290 |
| frame model, per-participant grid selection + shrunk calibration | .420 | .263 | .156 | .542 | .382 | .345 |
| frame model inert, shrunk calibration | .450 | .333 | .281 | .548 | .401 | .345 |
| adapted kitchen + model state | .473 | .349 | .310 | .580 | .411 | .332 |

CIs: adapted kitchen minus pooled kitchen G h1 +.002 [-.003,+.006], osf h1 +.007 [+.004,+.011], osf h3
+.015 [+.006,+.024]. Adapted model minus adapted kitchen G h1 -.056 [-.081,-.035], G h6 -.154
[-.229,-.089]; osf h1 -.038 [-.055,-.024], osf h3 -.026 [-.069,+.014], osf h6 +.014 [-.028,+.056].
Adapted model minus pooled model G h1 -.021 [-.045,-.001] (per-participant selection overfits), osf h6
+.055 [+.013,+.096]. Adapted aug minus adapted kitchen: -.002 to +.002, CIs include zero everywhere.

### 12c. Held-out Gaussian NLL per row, pooled protocol, valence level (sigma^2 from training residuals)
| predictor | G h1 | G h3 | G h6 | osf h1 | osf h3 | osf h6 |
|---|---:|---:|---:|---:|---:|---:|
| ar2 | -.451 | -.340 | -.309 | -.843 | -.679 | -.638 |
| kitchen | -.475 | -.373 | -.340 | -.864 | -.714 | -.674 |
| msar2 | -.452 | -.345 | -.315 | -.844 | -.686 | -.649 |
| frame model gated | -.450 | -.348 | -.317 | -.836 | -.699 | -.656 |
| frame model inert | -.450 | -.350 | -.321 | -.836 | -.701 | -.659 |
| kitchen + model state | -.475 | -.372 | -.341 | -.865 | -.718 | -.677 |
Same ordering as R2.

### 12d. Change in valence after events (Geschwind, pooled protocol)
After any reported event: kitchen .270/.339/.375 at h1/h3/h6, gated model .230/.304/.344, inert
.231/.306/.350, aug .270/.338/.376; gated minus kitchen h1 -.041 [-.050,-.032]; aug minus kitchen -.001
[-.002,+.001]. After the largest quartile of |event| (q75 over event beeps): kitchen .278/.333/.396,
gated .233/.297/.369, aug .279/.332/.396; gated minus kitchen h1 -.045 [-.060,-.032]; aug minus kitchen
+.000 [-.001,+.002]. Level targets on the same subsets show the same pattern (reviews/esm_eval_v3.md).

### 12e. Worry as target (Geschwind; worry never fed to the model; 11,442 rows at h1)
Ridge on six worry lags, six valence lags, events, time of day: R2 .406/.340/.311/.289/.276/.263 at h1..h6.
Plus the model's state features: .405/.339/.310/.288/.275/.263. Plus frame posterior only:
.406/.340/.310/.288/.276/.263. Every CI on the difference includes zero or is negative (largest |point|
.001). Persistence .138 at h1.

### 12f. Verdict
No predictor built on the model beats the strongest baseline at any horizon, on any target, under either
protocol, in either sample. The one CI that excludes zero in the model's favour (aug minus kitchen, osf, h3,
+.005) does not appear at neighbouring horizons or in the other sample and comes from the h-step expectation,
not from the frame or the channels. Gating (readout or transition) costs prediction. Per-participant
selection of the model's generative parameters overfits on Geschwind and is neutral on osf. A two-regime
switching regression is no better than two lags. The paper's ESM predictive claim should be dropped.

## 13. Model v2.1 (2026-09-28): hierarchical continuous state space

Spec reviews/model_v2_spec.md; code model_v2.py, fit_v2.py (b18bec2 and the commit recording this
section). Full record reviews/MODEL_V2_RESULTS.md; raw tables reviews/model_v2_forecast.md/.json.

### 13a. Forecasting, pooled protocol (`python fit_v2.py --workers 12`)
Held-out R2, h1 / h6. Geschwind: ridge (kitchen_tt) .439/.255, v1 .408/.214, v2 gated .448/.287,
v2 inert .448/.287. osf: ridge .533/.322, v1 .506/.294, v2 gated .542/.350, v2 inert .544/.358.
v2 gated minus ridge: Geschwind h1 +.009 [+.005,+.014], h6 +.032 [+.022,+.043]; osf h1 +.009
[+.004,+.015], h6 +.027 [+.011,+.043]. Positive with CIs excluding zero at every horizon 1..6 in both
samples. NLL same ordering. Adapted protocol no better than pooled.

### 13b. Ablations (difference from v2 gated, h1)
No slow mood -.038 (Geschwind) / -.074 (osf); frame clamped PRESENT -.038 / -.059 (the PAST weight gates
the level-2 pull); no time of day -.006 / -.001; no channels +.001 / +.002 (osf h6 +.016); inert vs gated
.000 / +.002 (osf h6 +.008 [+.001,+.020]). The gain is the slow mood level; channels and gating add
nothing to forecasting.

### 13c. Frame-sensitive tests
Variance after events (`python variance_v2.py`): predicted variance calibrated (ratio .96-1.05); gated vs
inert NLL +.0004 [-.0001,+.0009]. Change after events: v2 minus ridge h1 +.012 [+.005,+.018], h6 +.025
[+.016,+.036]. Regret (`python regret_v2.py --workers 6`): v2 q(PAST) weight mean .93 SD .01; regret_frame
= regret in held-out NLL. Choice (`python choice_v2.py`): affect histories leave q(f) near uniform
(.29-.41); implied discount 4.92-5.52 vs 5.03 inert. Worry target (`python worry_v2.py --workers 5`): v2
state adds h3 +.009 [+.002,+.015], h6 +.016 [+.009,+.024] to a worry ridge; frame posterior alone adds
.000.

### 13d. Verdict
v2.1 is the first version of the model to beat the strongest regression baseline, in both samples and
at every horizon, and to add information about future worry. The gain is from the slow mood level and the
online participant baseline. The frame, recognised from affect alone, stays near uniform and contributes
nothing to prediction; its behavioural effects (round 2) require a frame set by framing actions or clamps.
