# Empirical Validation Workplan

This document tracks the remaining empirical work for the temporal framing model and the manuscript changes each step will require. The current state is useful but preliminary: we have dataset discovery, descriptive statistics, directional data-model checks, and named readout baselines. We do not yet have fitted predictive validation or formal model comparison.

## Current Empirical State

- `empirical_validation.py` parses the OSF emotion reliability file, the Geschwind/Bringmann residual-depression ESM supplement, OpenNeuro `ds005356` lightweight BIDS event files, and the recovered autobiographical-memory specificity meta-analysis.
- `empirical_validation_report.md` reports descriptive dataset statistics, directional data-model checks, and named cited readout baselines.
- `EMPIRICAL_VALIDATION_RESULTS.md` records the current results, including the held-out Geschwind sequence-prediction test where the inertial active-inference model beats both named single-readout baselines and the current-valence persistence baseline.
- The current checks are pooled or simulation-summary based. They are not participant-level fitted predictions.
- The strongest current claims are:
  - event pleasantness aligns with current and next-step valence in Geschwind ESM;
  - worry is persistent in Geschwind ESM;
  - the stressed model has stronger future-frame persistence than the matched healthy stress profile;
  - AMT specificity is lower in depressed groups in the meta-analysis, while the recall-impaired model suppresses RECALL;
  - single readouts associated with Joffily, Pattisapu, and Hesp capture parts of the pattern but not the whole temporal-framing profile.

## Remaining Empirical Steps

### 1. Data Provenance And Reproducibility

Tasks:
- Pin all dataset URLs, citations, licenses, and local file paths in one machine-readable manifest.
- Avoid hard-coded Claude temp paths where possible by copying small recovered CSVs into `data_raw/` or documenting that they must be restored.
- Add a script mode that can run without network if all local data files are present.

Text consequences:
- Methods section can say which datasets were actually parsed locally.
- Claims about OpenNeuro, OSF, and PLOS supplements can cite licenses and access dates.
- Any dataset without a stable source remains a lead, not validation evidence.

### 2. Participant-Level Empirical Targets

Tasks:
- Replace pooled correlations with participant-level summaries and confidence intervals.
- For Geschwind ESM, compute within-person centered:
  - event pleasantness to current valence;
  - event pleasantness to next-beep valence;
  - event pleasantness to worry;
  - worry lag-1 persistence.
- For OSF `83cfk`, compute reliability and within-person affect dynamics, with participants as the unit.
- For the autobiographical-memory meta-analysis, compute random-effects or at least study/effect-size grouped summaries for AMT specificity and cue valence.
- For OpenNeuro `ds005356`, reconstruct trial pairs from cue/feedback events and extract win/loss rates per subject. Then locate or derive the MDD/control group key before making group claims.

Text consequences:
- Replace single pooled `r` values in the empirical section with participant-level means and intervals.
- State when effects are descriptive, not inferential.
- Do not claim clinical group prediction from OpenNeuro until group labels are linked to subject IDs.

### 3. Data-Model Target Matching

Tasks:
- Define a small target vector that both data and model can produce:
  - reward/event to current valence;
  - reward/event to next-step valence;
  - worry/future persistence;
  - recall specificity or RECALL suppression;
  - reward versus punishment asymmetry where task data support it.
- Compute the same target vector for model profiles across seeds.
- Add uncertainty over seeds and compare model values against empirical intervals.
- Separate “directional match” from “numeric fit.”

Text consequences:
- Results section can report which targets the model matches directionally and which are only qualitatively aligned.
- Discussion must keep the missing temporal-orientation dataset as the main empirical gap.

### 4. Formal Ablations

Tasks:
- Add variant runners for:
  - full temporal-framing model;
  - one-step EFE without counterfactual rollout;
  - single-channel Joffily/VFE valence;
  - single-channel Pattisapu reward/arousal valence;
  - single-channel Hesp affective-charge valence;
  - no temporal-frame state;
  - no habit prior;
  - symmetric hedonic sensitivity (`c_pos = c_neg`);
  - no positive-recall gate;
  - fixed-depth Hesp-style rollout without state-dependent temporal policies.
- Score every variant against the target vector.
- Report distance from empirical target and pass/fail for directional signs.

Text consequences:
- The paper can claim “the full model covers more empirical targets than these ablations” only if the table supports it.
- If a simpler baseline matches a target, the text must say so.
- Smith and Ellsworth should be treated as a qualitative appraisal taxonomy unless we implement a computational classifier from its appraisal dimensions.

### 5. Predictive Or Fitted Validation

Tasks:
- Decide whether to fit parameters to empirical summaries or only score pre-registered profiles.
- If fitting, use held-out validation:
  - fit parameters on a subset of empirical targets or participants;
  - evaluate held-out targets or participants.
- For Geschwind ESM, start with summary-statistic fitting rather than participant-level hidden-state inference.
- For OpenNeuro, fit reward/punishment learning only after group labels and trial choices are clear.

Text consequences:
- If we do not fit, use “empirical anchoring” and “directional validation.”
- If we fit with held-out evaluation, we can use stronger language such as “predictive validation,” but only for the targets tested.

### 6. Figures And Tables

Tasks:
- Add a validation table comparing empirical targets, full model, and ablations.
- Add a named-baseline table for Joffily, Pattisapu, Hesp, and Smith.
- Add a provenance table for all datasets.
- Optionally add one figure showing empirical intervals versus model/ablation summaries.

Text consequences:
- The PAD figure remains simulation calibration, not external validation.
- The empirical validation section should cite the new tables and separate calibration, data anchoring, and model comparison.

### 7. Manuscript Revision

Tasks:
- Update abstract and introduction to avoid “validate” where only calibration or anchoring is meant.
- Replace `Empirical Validation Plan` with `Empirical Data Checks And Model Comparison`.
- Add a concise methods paragraph for dataset parsing and summary targets.
- Add a results paragraph for empirical targets and ablations.
- Update limitations:
  - no direct temporal-orientation ESM dataset yet;
  - no OpenNeuro group-level fitting until phenotype key is linked;
  - named predecessors are partial readout baselines unless fully reimplemented.
- Remove draft markers, stale line references, and LLM-tell language.

### 8. Verification

Tasks:
- Run `python -B empirical_validation.py`.
- Run a quick simulation suite with `python -B run.py --quick`.
- Run targeted smoke tests for `run_trial`, counterfactual rollout, and variant scoring.
- Search the manuscript for draft markers, overclaiming, and tell-list terms.
- If LaTeX is available, compile or at least check labels/references.

## Execution Order

1. Make data provenance local and reproducible.
2. Upgrade empirical target extraction to participant-level summaries.
3. Add formal model variants and scoring.
4. Run the variant comparison.
5. Update `empirical_validation_report.md`.
6. Revise the manuscript from the results.
7. Run verification and cleanup.
