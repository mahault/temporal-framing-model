# Empirical Dataset Candidates for Affective Temporal Framing

Status: recovered from Claude deep-research task `w71jl5x9q` and locally checked on 2026-07-15. The Claude verifier phase failed because the session hit its usage limit, so this file treats the run as a lead list, not as a fully verified evidence review.

## What The Research Was Trying To Find

The search targeted datasets for testing:

- three-channel valence dynamics: backward/model valence, present/reward valence, forward/action valence
- temporal orientation of thought: past, present, future
- rumination versus worry: backward- versus forward-directed repetitive negative thought
- asymmetric reward and punishment sensitivity, especially anhedonia
- impaired autobiographical memory specificity and positive recall
- chronic-stress or slow mood-attractor dynamics

## Domain A: ESM/EMA Affect Dynamics

### Best first pass: openESM database

- Host: https://openesmdata.org/
- Local evidence: `scratchpad/datasets_table.json` in Claude temp cache contains 61 dataset metadata rows.
- Access: intended as openly downloadable through the openESM R/Python packages.
- Why it matters: broadest route into repeated within-person affect dynamics. Several datasets have many repeated observations, positive/negative affect variables, stress, mood, cognition, or passive sensing.
- Model mapping: fit within-person affect dynamics and compare whether a three-channel latent model explains temporal autocorrelation and affect transitions better than one-channel valence baselines.
- Next action: install/use `openesm` package or direct dataset pages, then shortlist datasets with momentary positive affect, negative affect, stress, rumination, thought, and timestamps.

Strong candidates visible in cached metadata:

- `0010_geschwind`: depression, neuroticism, mood; 130 participants, 200 time points, 20 days, 10 beeps/day.
- `0006_rowland`: mindfulness, affect, network; 125 participants, 240 time points, 40 days, 6 beeps/day.
- `0003_hawks`: momentary cognition, context, stress; 122 participants, 30 time points, 10 days, 3 beeps/day.
- `0004_wang`: students, mental health, academic performance, stress, activity; 49 participants, 64 time points, passive sensor data.

### OSF 83cfk emotion reliability / momentary affect

- Host: https://osf.io/83cfk/
- Local files recovered: `emotions_data.csv`, `baseline_data.csv`, `Data_protocol_2.csv`.
- Local evidence: `emotions_data.csv` has timestamped rows with participant alias, scheduled beep id, and 12 momentary emotion slider variables, each with paired repeated columns for reliability checks.
- Measures seen locally: calm, anxious, dejected, relaxed, bored, happy, satisfied, stressed, frustrated, cheerful, sad, enthusiastic.
- Access: OSF public download appears to have been used by Claude; verify citation/license before publication use.
- Model mapping: useful for measurement-noise constraints and within-person affect dynamics, but it lacks explicit temporal orientation of thought.

### Mind-wandering / temporal orientation in depression

- Lead sources from Claude: https://pmc.ncbi.nlm.nih.gov/articles/PMC11826933/ and ScienceDirect `S0165032724013387`.
- Claimed design: MDD and healthy controls, ESM prompts measuring mind-wandering, thought valence, temporal orientation, positive/negative affect, and brooding.
- Access: article visible in search output, but raw data availability still needs direct verification.
- Model mapping: strongest direct test of whether depression changes temporal orientation itself, thought valence, or affect-to-thought coupling.
- Current status: not safe to cite as an open dataset until the data host is confirmed.

## Domain B: Reward / Punishment Sensitivity

### Best first pass: OpenNeuro ds005356

- Host: https://github.com/OpenNeuroDatasets/ds005356 and https://openneuro.org/datasets/ds005356
- Local/web evidence: GitHub README identifies the dataset as MEG data from a probabilistic selection / reinforcement-learning task with SCID-interviewed controls and MDD participants.
- Sample: README states CTL non-depressed `n=38`, MDD `n=52`.
- Task: MEG-compatible probabilistic selection task; reward positivity / reinforcement-learning context.
- Access: OpenNeuro/GitHub dataset; likely directly downloadable, but license should be checked in `dataset_description.json`.
- Model mapping: fit reward versus punishment learning parameters and test whether asymmetric hedonic sensitivity (`c_pos`, `c_neg`) maps onto MDD/anhedonia better than scalar reward sensitivity.

Other leads:

- Pizzagalli-style probabilistic reward task open data may exist, but Claude did not recover a verified direct download in the final output.
- Effort-cost / EEfRT datasets would be useful for the interoceptive-energy-cost side of FUTURATE/FEEL, but this was not yet verified.

## Domain C: Autobiographical Memory, Rumination, Worry

### Best first pass: Autobiographical memory specificity meta-analysis data

- Local files recovered: `Final_AutoData.csv`, `Data_protocol_2.csv`, `readme.txt`.
- Local evidence: `readme.txt` documents effect-size level variables for depression versus control comparisons, AMT versus non-AMT, stimulus valence, specificity, categoricity, symptom severity, medication, episode, and other moderators.
- Access: local recovered CSVs; source URL still needs to be pinned before citation.
- Model mapping: supports the paper's claim that depression is linked to overgeneral / less specific autobiographical retrieval, especially by stimulus valence. It is meta-analytic rather than per-trial within-person data, so it tests the recall-precision parameter at the group/effect-size level.

### FEST / future event specificity OSF lead

- Host from Claude: https://osf.io/8n2sq/
- Claimed content: Future Event Specificity Training, anhedonia, dampening, episodic future thinking specificity/detail/imagery, anticipated and anticipatory pleasure.
- Access: Claude output shows registration metadata, but `has_data` was false on the registration object it fetched. Need the linked project/node checked before treating it as downloadable.
- Model mapping: would test the future-oriented specificity side of mental time travel and positive future simulation, if actual data are available.

### EMA repetitive negative thinking lead

- Lead source from Claude: PLOS ONE `10.1371/journal.pone.0318453`, OSF `dm2ab`.
- Claimed content: EMA repetitive negative thought in adolescents/young adults, plus baseline brooding and worry scales.
- Access: needs verification; Claude's verifier failed.
- Model mapping: good for repetitive negative thought dynamics, but likely weak for rumination-versus-worry unless momentary items distinguish past from future orientation.

## Recommended First Empirical Validation Plan

1. Start with openESM for Domain A and select one high-frequency affect dataset with timestamps and positive/negative affect.
2. Use OpenNeuro `ds005356` for Domain B to validate reward/punishment sensitivity against MDD/control task behavior.
3. Use the autobiographical-memory meta-analysis CSVs for Domain C as a group-level validation of the recall-specificity parameter.
4. Treat temporal-orientation-of-thought datasets as the missing keystone: do not claim direct empirical support for past/present/future orientation until the MDD mind-wandering raw data or another open EMA temporal-orientation dataset is confirmed downloadable.

## How This Should Affect The Paper

- Replace "PAD validation" language with "expressivity and simulation calibration" unless parameters are fit to independent data.
- Add a concrete empirical validation section with three staged tests:
  - within-person EMA affect dynamics,
  - reward/punishment task parameter recovery,
  - autobiographical specificity / rumination validation.
- Add one caveat: temporal orientation of thought is the most direct test of the theory but currently the least verified open-data path.
- Keep the model-comparison promise, but frame it as planned ablations unless we actually implement and run the comparisons.

