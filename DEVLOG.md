# Devlog

## 2026-06-26
- Updated the temporal-framing and bipolar-disorder modelling papers (uncommitted, work in progress)
- Added mood-calibration, metastability-diagnostic, final-sweep and verification/audit scripts (uncommitted, work in progress)

## 2026-06-27
- (uncommitted, work in progress) Continued revising the affective-valence temporal-framing and bipolar-disorder modelling papers
- Added mood-calibration, metastability diagnostics, final parameter sweeps, and full verification/audit scripts

## 2026-06-29
- Updated the affective-valence temporal-framing and bipolar-disorder modelling papers.
- Added mood-calibration, metastability diagnostics, final sweeps and verification scripts (audit/verify_all/verify_state). (uncommitted)

## 2026-06-30
- Revised both papers (affective-valence temporal framing; computational modelling of bipolar disorder).
- Added mood calibration, metastability diagnostics, final parameter sweeps, and verification scripts (verify_all, verify_state, audit_paper1). (uncommitted, work in progress)

## 2026-07-15
- (uncommitted, work in progress) Empirical validation pass: dataset candidates surveyed and downloaded (research_dataset_candidates.md, data_raw/), and a full validation pipeline built — empirical_validation.py, empirical_rebuild.py, fit_params.py, eval_fitted_cv.py, diagnose_mechanisms.py, with a workplan (EMPIRICAL_VALIDATION_WORKPLAN.md).
- Headline result (EMPIRICAL_VALIDATION_RESULTS.md, empirical_rebuild_report.md): on the Geschwind residual-depression ESM dataset (129 participants, ~11.7k prediction records, whole-participant CV), the inertial active-inference temporal-framing model predicts held-out next-beep valence better than AR(1), event-linear, and the Joffily/Pattisapu/Hesp single-readout baselines (~+3% skill vs AR(1), R2 ~0.145 at h=1). Explicitly flagged as promising-but-not-final: no participant-level parameter fits yet, no direct temporal-orientation dataset, OpenNeuro group-level modelling incomplete.
- Model and figures refreshed alongside: agent.py, generative_model.py, experiments.py, plotting.py edited and all paper figures regenerated; the affective-valence temporal-framing tex updated.

## 2026-07-16
- (uncommitted, work in progress) Unified single-document version of the paper assembled and compiled: affective_valence_temporal_framing_unified.tex → .pdf (with bibliography).
- Raw behavioral datasets pulled into data_raw/ for the empirical-validation push: hrl_decay1 (P2017b/S2021c .mat + original decay-anneal model code) and palminteri_cf (counterfactual-learning behavioral data, exp1 subject files).

## 2026-07-17
- Unified paper finalized (affective_valence_temporal_framing_unified.tex, 22 pp): merged temporal-framing + taxonomy material with Manon's mental-time-travel/depth-regulator theory; every design choice literature-grounded (tab:justification); all generative models shown (full factor graph, three predecessor sub-graphs, gamble task-model, layered-architecture schematic).
- Honest empirical case, all held out across participants (EMPIRICAL_RECORD.md is the record):
  - Subsumption: three channels over a gamble task-model recover the Rutledge happiness equation (ours R2=0.144 vs eq 0.146; single-channel reconstructions 0.000/0.111/0.034; no EV regressor), 14,803 held-out subjects.
  - Extension REPLICATED on a second ESM sample (esm_replication.py, esm_dig.py): model leads at every horizon on both Geschwind (n=129, 2.2x at h=1) and osf_83cfk (n=91, near parity at h=1 where affect is highly persistent). Ablations: gain is not the event channel (valence-only reproduces it) and not solely persistence (inertia=0 still beats baseline at h=1 on Geschwind); multi-step margins lean on inertia. Paper reframed to the sample-dependent claim.
  - Counterfactual regret->switch (t=10.4, n=143) promoted to a behavioral prediction a reward-only model cannot produce; hedonic asymmetry scoped as generative/neural (RewP) conjecture.
- Three adversarial audit rounds (citations/hallucinations, numbers-vs-record, coherence+LLM-tells, reviewer-strength): fixed a taxonomy channel misassignment (-dF/dt is backward, RPE present), scoped the 2x claim, fixed Treadway/Hesp2020 misattributions and two bib author names (Singh Garima, Hermans Dirk), removed comma-splice seams, arrows, em-dashes, "fair", honesty-badging; PAD figure replaced by a recalibrated circumplex figure (10/10 correct quadrants, anger/fear dominance split).
- Deliverables: UNIFIED_PAPER_CURRENT.pdf and temporal_framing_paper_overleaf.zip (tex + references.bib + 13 figures) in Downloads.
- data_raw/ (184MB third-party datasets) and ad-hoc repo zips gitignored; download provenance documented in EMPIRICAL_RECORD.md.

## 2026-07-19
- Phenotype-figure scrutiny (PAD-style, "just to be certain"): found the M5 mood layer observed mean VFE, which is flat across pi_pos and adapts away under stress, so it could not detect depression -> the old "emergent depression under chronic stress" claim was contradicted 0/10 seeds (on both the old and current models). Diagnosed and fixed: the mood now observes believed-valence level (Beck's schema; Eldar & Niv 2016), neutral anchored to the sigmoid knee. Produces a ROBUST diathesis-stress result (Monroe & Simons 1991): only vulnerability x stress -> depression (vuln+stress 0.76 8/8; healthy+stress 5.16 resilient; vuln+calm 4.22 recovers; healthy+calm 7.43). Verified breaks nothing (ESM/Rutledge/orientation/RECALL% all unchanged). Redesigned Experiment 7 as a 2x2 diathesis-stress; fig14 rebuilt (agent.py, experiments.py, plotting.py). Removed the hand-tuned PAD/circumplex figure (constants gave 7/10 at a principled center; 10/10 required tuning).
- Tested the diathesis-stress prediction on REAL data (diathesis_stress_test.py, EMPIRICAL_RECORD §9): Geschwind baseline neuroticism (held-out) -> lower baseline mood (r=-0.53) and steeper event->affect reactivity (interaction t=4.2, 11,315 beeps). Upgraded from demonstration to a confirmed core prediction. Frame recovery confirmed a genuine identifiability limit (felt-valence likelihood frame-independent by design; affect explains <5% of orientation variance).
- Superiority reframe answering the original rejection: abstract now leads with the hard predictive win (ESM out-of-sample doubling + inertia ablation + second-sample lead), demotes Rutledge to "recovers the reference ceiling," and neutralizes the Joffily/Hesp strawman (channel-decomposition framing; category-mismatched not defeated). Cleared the writing-blocker backlog B1-B6 (AUDIT_FINDINGS_2026-07-17.md). Multiple adversarial-agent rounds (superiority, correctness, LLM-tells) verified; final de-slop pass ("reads human"). Paper 24pp, compiles clean (0 undefined, 0 bibtex warnings). Deliverables in Downloads (UNIFIED_PAPER_CURRENT.pdf, temporal_framing_paper_overleaf.zip). Commits 6ae6069..12af9de.

## 2026-09-27
- AAAI-27 #39504 rejected in Phase 1 (24 Sept). Reviews pulled from OpenReview into reviews/AAAI27_reviews_39504.md, triaged in reviews/REVISION_TODO_39504.md, and every point checked against the code. Four reviewer claims verified as correct: the temporal frame gated nothing (A and C frame-independent, B_frame action-only), the worry item WAS in the valence composite, Fig 2(c)'s caption contradicted its plot, and Fig 3 showed no transition to past framing.
- Model revision: frame-gated precision (agent.py `frame_gain`, `frame_clamp`). q(f) now sets horizon weights w_c = 1 + g(3 q(f=c) - 1) that scale the three affective channels in the readout and the present / future / retrospective terms of the EFE (new `_efe_gated`, `_efe_retro` = KL[q(s'|a) || D]). g = 0 reproduces the old model exactly (tests/test_frame_gating.py, 5 tests pass). experiments.FRAME_GAIN = 1.0 is now the paper model. Qualitative results survive gating: RECALL 44 percent -> 0 under impairment (FEEL 18 -> 68), stressed q(FUTURE) 0.44 vs 0.20, diathesis 8/8 (healthy+stress 7.4 +/- 0.1, vulnerable+calm 5.1 +/- 0.6, vulnerable+stress 1.3 +/- 0.3). Under gating the impaired agent's policy entropy is slightly HIGHER than healthy (the original caption's direction); the ungated model had it lower.
- Honest worry analysis (frame_worry_multilevel.py, reviews/frame_worry_multilevel_results.md): worry removed from input, raw r = 0.067, partial r after v_t and event = 0.007, within-person r = 0.009, cluster-robust z adjusted = 0.7. Claim withdrawn from abstract, Table 2 and section (now sec:worry).
- Frame ablation on both ESM samples with the paper's fitted pipeline (esm_frame_ablation.py, reviews/esm_frame_ablation.md): gated, inert and clamped models within 0.005 R2 at every horizon on both samples; gain over baselines comes from the valence transition dynamics (channels-only regression trails by 0.08 at h=1 on Geschwind). Participant-bootstrap CIs added; replication-sample h=1 lead is +0.011 [-0.001, +0.023] so "replicates" replaced by "does not reverse".
- Regret rework (regret_model_fit.py, reviews/regret_model_results.md): outcomes are binary so regret/relief was confounded with obtained outcome; the honest contrast is after-loss foregone-better vs foregone-same (human +0.13, t = 10.4; Palminteri +0.27). A factual Q-learner generates +0.05 of it, regret bias +0.08, counterfactual value learning +0.09; neither improves held-out likelihood. "A signature a factual model cannot produce" withdrawn; Fig 4 now shows human and three model-generated contrasts; new Table tab:regret.
- Sensitivity sweep (sensitivity_bframe.py -> figures/fig_sensitivity.png, reviews/sensitivity_results.md): frame stickiness 0.5..0.95 x pull-weight scale 0.6..1.4 x 4 seeds, plus frame gain 0..1. RECALL collapse and future fixation positive in 30/30 cells; diathesis holds in all seeds in 77 percent of cells (failures only at pull scale 1.4); all three hold at every gain. Policy-entropy sign is constant-dependent, so the paper no longer builds on it. Recorded in EMPIRICAL_RECORD section 10e.
- Paper (affective_valence_temporal_framing_unified.tex): new sec:fullmodel with the factorisation, Eq. frameweights and Eq. gatedefe (fixes the missing discount/rollout equation), new sec:worry with tab:worry, new sec:esm-ablation with tab:frameablation, new sec:sensitivity with fig:sensitivity, tab:regret, taxonomy rewritten as discriminating predictions, "unified" defined as structural in abstract, pi_pos renamed rho_pos (pi_pos in code), "reference ceiling" -> "reference model", baselines and grid listed in the main text, Table 2 rows revised, limitations extended (channel scale, rho_pos triple duty, FUTURATE post-hoc). All figures regenerated with the gated model (run.py); fig_model_advantage numbers updated (2.1x). Compiles clean with latexmk (28 pp, 0 undefined refs).
- Not done: the 7-page AAAI-format tex lives on Overleaf only; these edits are in the long-form tex and must be ported. rho_pos decoupling ablation not run. Target venue per REVISION_TODO: Computational Psychiatry.
- Nightly sweep (21:00): the above landed as two commits on master, cd8128a (the revision: frame-gated precision, honest worry test, frame ablation, regret rework, sensitivity sweep, paper edits, 39 files) and 8b2b55f (sensitivity results into the paper and EMPIRICAL_RECORD, `figures/fig_sensitivity.png`, PDF rebuilt and compiling clean). Master is 2 ahead of origin, unpushed. No CHANGELOG in this repo.

## 2026-09-28
- Round 2 on AAAI #39504 against reviews/INTERNAL_REVIEW_2026-09-27.md. Found and fixed the Geschwind
  loader bug (two sampling periods interleaved; see EMPIRICAL_RECORD §11): every earlier Geschwind ESM
  number was computed on scrambled sequences and the "2x at one step" headline was an artefact. New
  unified evaluation (esm_eval_v2.py): the model matches a two-lag linear regression on both samples and
  does not exceed it; frame variants (inert, clamped, transition-gated: new Agent.frame_transition_gain)
  change nothing. Frame-worry redone on correct sequences: null after controls for every variant.
- New: regret_participant_holdout.py (group-level fits, participant folds, simulated contrasts with CIs;
  regret bias reproduces most of the human contrast, held-out gain CI includes zero; frame gating of the
  bias changes nothing); frame_choice_task.py (intertemporal choice under the gated EFE: inferred frame
  moves the implied discount from 5:1 to 12.6:1 after RECALL and parity after FUTURATE);
  rho_pos_decoupling.py (targets carry the RECALL collapse; the mood loop deepens but does not create the
  attractor); stress_one_factor.py; diathesis_stats_v2.py and mulholland_stats.py (clustered SEs, CIs).
- Model code: generative_model TARGET_ALPHA_OVERRIDE / D_PI_POS_OVERRIDE; agent frame_transition_gain;
  plotting labels rho_pos; fig10 now two panels (entropy panel dropped); fig_model_advantage and
  fig_counterfactual_signature regenerated; new figures/fig_frame_choice.png.
- Paper: round-2 edits staged as exact replacements (reviews/round2_edits_part1.py, part2.py,
  round2_edits_part3.json), verified on a local AAAI build (reviews/round2_build, 12 pp main + 4 pp
  appendix, 0 errors) before being pushed to Overleaf.
- Round 3, "predict better": esm_eval_v3.py (strong baselines: six-lag ridge with missing-lag
  indicators, events, time of day; two-regime Markov-switching AR by EM with causal regime posterior;
  hierarchical adaptation of coefficients and per-participant grid selection of generative parameters on
  the first period; horizons 1 to 6; change-after-event targets; ridge augmented with the model's state
  features and ablations; held-out Gaussian NLL) and esm_worry_pred_v3.py (worry as a second target).
  Result: no gain. The six-lag ridge beats the model by 0.03 R2 at every horizon in both samples; the
  model's state adds 0.000 to the ridge (0.005 at h3 on osf only); gating and adaptation of the model's
  parameters cost prediction. Recorded in EMPIRICAL_RECORD §12 and reviews/PREDICTION_ROUND3.md; figure
  figures/fig_esm_v3.png. No paper fragment written; the predictive claim should be dropped.
- Model v2 (hierarchical continuous state space; model_v2.py, fit_v2.py, spec reviews/model_v2_spec.md).
  v2.0 (d8868d6) lost to the ridge because per-participant offsets overfit; v2.1 (b18bec2) moved the
  participant baseline into the filter state. Full run (`python fit_v2.py --workers 12`, both samples,
  9 variants x 5 folds): v2.1 beats the six-lag ridge with events and time of day at every horizon in
  both samples (h1 +0.009, h6 +0.03, CIs exclude zero) and beats the discrete model by 0.04 to 0.07.
  Ablations: the gain is the slow mood level; channels and frame gating add nothing; clamping PRESENT
  hurts because the PAST weight gates the mood pull. Frame-sensitive reruns on v2.1 (variance_v2,
  regret_v2, choice_v2, worry_v2): variance calibrated, gating neutral; frame posterior from affect stays
  near uniform; v2 state adds to a worry ridge at h3 to h6 (+0.016 at h6), the frame does not. New
  plot_v2.py and figures/fig_forecast_v2.png. EMPIRICAL_RECORD section 13, reviews/MODEL_V2_RESULTS.md,
  merge-ready reviews/model_v2.tex.

## 2026-09-30

- Fairer test of active frame selection (analysis/orientation, reviews/FRAME_ORIENTATION_MODEL.md
  section of 2026-09-30). All competitors refit with one valence mean shared across frames, so frames
  are defined by orientation alone (past-frame emission 0.97 to 1.00). The active-inference framing
  agent still predicts the next reported orientation worse than a Markov chain with valence:
  Baumeister S1 -0.028 [-0.039, -0.018], Bayer -0.009 [-0.015, -0.003], Bayer with 20 or more labels
  -0.006 [-0.011, -0.002], pooled over 617 participants -0.023 [-0.030, -0.015] nats per signal.
  With a shared mean the risk term cannot separate actions, so the epistemic term is the only part of
  expected free energy that acts, and removing it improves the fit by +0.023 pooled. Verdict: active
  selection of orientation by expected free energy is rejected on these data.
- Baumeister Study 2 is not usable for sequences: no clock time, and the file is sorted by the
  orientation label within participant. Kept as a concurrent-valence replication only.
- frame_orientation_model.py: new shared-mean specs, fit1 skips saved parts, chain/eval1/pool stages,
  dense-label restriction and per-participant differences in eval. New
  frame_orientation_fairtest_fig.py and figures/frame_orientation_fairtest.png. reviews/frame_orientation.tex
  rewritten around the fair test; reviews/frame_orientation.bib with baumeister2020everyday,
  mao2023mcog and siepe2026openesm (checked against Crossref). EMPIRICAL_RECORD section 14.

## 2026-09-30 (channel-specific test)

- Preregistered four predictions for the three valence channels in reviews/CHANNEL_TEST.md section 0
  and committed them (9d9b048) before relating any channel to any item.
- analysis/channels/channel_test.py: Baumeister et al. (2020) Study 1, 453 participants, 6,544 signals.
  v2.1 channels (model_v2.py unmodified, g = 1) fitted on four participant folds and run on the held-out
  fold; valence-only input is primary, thought pleasantness as event is secondary, cached v1 drives are
  the robustness check. Cluster-robust logistic and linear regressions of regret, replaying,
  past-disappointed, what-might-have-been, worry, fear, planning, hoping and the disappointed, anxious
  and angry ratings on the three channels plus valence, time of day and orientation; held-out
  discriminant and incremental log-likelihood tests with a participant bootstrap.
- Verdicts: P1 inconclusive, P2 inconclusive (signs as predicted for worry, fear, planning, anxiety, CIs
  include zero), P3 supported in the primary variant, P4 not supported. The one specific association is
  backward channel with "what might have been" thought, -0.16 [-0.29, -0.03], stable across variants.
  Channels add nothing to valence, time of day and orientation in held-out prediction. With valence as
  the only input the present and forward channels correlate 0.89 and -0.87 with valence.
- Common-scale check: analysis/channels/rutledge_scale.py refits the Rutledge affect layer (3,000
  subjects, 326,340 ratings); per-SD weights present 0.33, forward 0.19, backward 0.02. Fitted ESM drive
  weights depend on the input variant, so no single scaling transfers.
- New files: analysis/channels/channel_test.py, rutledge_scale.py, .gitignore; reviews/CHANNEL_TEST.md,
  reviews/channel_test.tex (compiles in a two-column scratch wrapper, 0 errors); figures
  channel_test_coefficients.png, channel_test_discriminant.png. EMPIRICAL_RECORD section 15.
- (uncommitted, work in progress) Clinical predictions on real people: `reviews/DIATHESIS_DATA_TEST.md` preregisters five predictions (mood level, inertia, stress reactivity, persistent negative thought, diathesis-stress interaction) at 18:24 EDT before analysis; `analysis/diathesis/diathesis_data.py` and `report.py` fit v2.1 per dataset with per-person deviations by empirical Bayes; runs on Gainey, Geschwind and Kane finished and `out/summary.json` is written, verdicts not yet recorded.
- (uncommitted, work in progress) `analysis/joint_gamble/`: a joint model of momentary happiness and risky choice on the Rutledge GBE task (three affect channels plus a two-timescale slow mood), five cross-validation folds run into `out/`.
- Branch is 11 commits ahead of origin, unpushed.

## 2026-10-01
- Depression test on the Rutledge gamble data (1,838 BDI participants, model J, predictions committed first in fa830ae). Baseline mood strongly lower with BDI (-0.057 per SD, LR chi² 222), signed RPE weight unrelated (replicates Rutledge 2017), backward (unsigned surprise) weight more negative with BDI (chi² 19.1), forward weight opposite to prediction, optimism coupling unchanged, persistence not identified. Model parameters predict held-out BDI better than model-free summaries (+0.039 R²). reviews/GAMBLE_BDI_TEST.md, reviews/gamble_bdi.tex, EMPIRICAL_RECORD section 16.
- Gamble round 5 (reviews/GAMBLE_R5.md, reviews/CORRECTIONS_R5.md, reviews/gamble_r5.tex, EMPIRICAL_RECORD section 18). BDI backward-channel sign was reversed in the draft: more depressed participants lose less happiness after surprise (smaller loss-over-win asymmetry); corrected in GAMBLE_BDI_TEST.md. Table 9 intervals replaced by a 200-replicate participant bootstrap that agrees with the likelihood ratios. BDI baseline and backward effects replicate on 929 second plays; forward does not. Person weights from play 1 predict play 2 better with the three channels than with the happiness equation (+0.0077 nats per rating), eleven times the shared-weight gap. Momentary affect improves choice prediction beyond perseveration and prospect theory (+0.0011 nats per choice, held out). FDR within seven families: the ESM channel-test association does not survive.
- Forecasting round 5 (reviews/FORECAST_R5.md, reviews/forecast_r5.tex, EMPIRICAL_RECORD section 17). Added the baselines the cold review asked for. A local level plus AR(1) Kalman filter is the best forecaster at every horizon in both ESM samples and beats v2.1 (h6 -0.016 and -0.029). Improved the model (person state carried across periods, fitted set-point, six-step objective, volatility state): +0.008 and +0.015 at h6 over v2.1, still below the filter. The model's two timescales written as the filter, with the volatility state and the three channels, ties the filter on R2. The volatility state gives the largest likelihood gain (0.05 to 0.09 nats per beep). Channels add 0.0015 R2 at h6 on Geschwind only.
