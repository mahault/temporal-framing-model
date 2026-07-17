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
