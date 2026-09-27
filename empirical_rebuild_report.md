# Rebuilt Empirical Validation Report

- Model: frame-gated (g=1.0); `model_ungated` is the pre-revision frame-inert model; `model_frame_*` clamp the gating posterior; `channels_only` regresses on the three readout channels with no state rollout.
- Dataset: Geschwind/Bringmann residual-depression ESM (`data_raw/geschwind_2013_s004.csv`).
- Participants used: 129; prediction records: 11734.
- Cross-validation: whole participants held out per fold.
- Valence normalised to [0,1]; RMSE and R2 in that scale; each predictor gets an optimal train-fit linear calibration.

## Horizon h = 1 step(s) ahead

| Predictor | RMSE | R2 (fold SD) | r | skill vs AR(1) |
|---|---:|---:|---:|---:|
| persistence | 0.2359 | -0.393 (0.092) | 0.306 | -23.6% |
| mean | 0.2007 | -0.007 (0.009) | n/a | -5.1% |
| ar1 | 0.1909 | 0.089 (0.027) | 0.306 | +0.0% |
| direct_v | 0.1909 | 0.089 (0.027) | 0.306 | +0.0% |
| linear_event | 0.1908 | 0.089 (0.027) | 0.307 | +0.0% |
| linear_event_asym | 0.1908 | 0.090 (0.026) | 0.308 | +0.1% |
| model_full | 0.1842 | 0.152 (0.027) | 0.392 | +3.5% |
| model_ungated | 0.1841 | 0.153 (0.027) | 0.393 | +3.6% |
| model_frame_present | 0.1840 | 0.154 (0.027) | 0.395 | +3.6% |
| model_frame_past | 0.1844 | 0.150 (0.026) | 0.389 | +3.4% |
| model_frame_future | 0.1844 | 0.150 (0.027) | 0.389 | +3.4% |
| channels_only | 0.1860 | 0.135 (0.026) | 0.373 | +2.6% |
| model_symmetric | 0.1840 | 0.153 (0.027) | 0.393 | +3.6% |
| model_one_step | 0.1841 | 0.152 (0.026) | 0.392 | +3.5% |
| model_no_inertia | 0.1865 | 0.130 (0.027) | 0.364 | +2.3% |
| readout_joffily | 0.2004 | -0.004 (0.010) | 0.052 | -5.0% |
| readout_pattisapu | 0.2000 | 0.000 (0.007) | 0.078 | -4.8% |
| readout_hesp | 0.1962 | 0.038 (0.008) | 0.209 | -2.8% |

Participant-bootstrap 95% CI on R2 differences (2000 resamples of participants):

- full - best baseline (linear_event_asym): +0.062 [+0.052, +0.072], P(diff<=0)=0.000
- full - ungated (frame-inert): -0.001 [-0.002, +0.001], P(diff<=0)=0.756
- full - frame clamped to PRESENT: -0.002 [-0.004, +0.001], P(diff<=0)=0.903
- full - channels only: +0.017 [+0.004, +0.030], P(diff<=0)=0.006

## Horizon h = 2 step(s) ahead

| Predictor | RMSE | R2 (fold SD) | r | skill vs AR(1) |
|---|---:|---:|---:|---:|
| persistence | 0.1952 | 0.045 (0.061) | 0.525 | -9.9% |
| mean | 0.2005 | -0.007 (0.009) | n/a | -12.9% |
| ar1 | 0.1775 | 0.210 (0.015) | 0.525 | +0.0% |
| direct_v | 0.1702 | 0.274 (0.033) | 0.525 | +4.1% |
| linear_event | 0.1774 | 0.212 (0.015) | 0.525 | +0.1% |
| linear_event_asym | 0.1774 | 0.212 (0.014) | 0.526 | +0.1% |
| model_full | 0.1720 | 0.259 (0.022) | 0.510 | +3.1% |
| model_ungated | 0.1700 | 0.276 (0.025) | 0.527 | +4.3% |
| model_frame_present | 0.1740 | 0.242 (0.025) | 0.493 | +2.0% |
| model_frame_past | 0.1725 | 0.255 (0.022) | 0.506 | +2.9% |
| model_frame_future | 0.1685 | 0.289 (0.026) | 0.538 | +5.1% |
| channels_only | 0.1692 | 0.282 (0.031) | 0.533 | +4.7% |
| model_symmetric | 0.1713 | 0.265 (0.024) | 0.515 | +3.5% |
| model_one_step | 0.1720 | 0.259 (0.023) | 0.510 | +3.1% |
| model_no_inertia | 0.1878 | 0.117 (0.027) | 0.348 | -5.8% |

Participant-bootstrap 95% CI on R2 differences (2000 resamples of participants):

- full - best baseline (direct_v): -0.016 [-0.034, -0.001], P(diff<=0)=0.985
- full - ungated (frame-inert): -0.018 [-0.023, -0.012], P(diff<=0)=1.000
- full - frame clamped to PRESENT: +0.018 [+0.009, +0.026], P(diff<=0)=0.000
- full - channels only: -0.025 [-0.042, -0.010], P(diff<=0)=1.000

## Horizon h = 3 step(s) ahead

| Predictor | RMSE | R2 (fold SD) | r | skill vs AR(1) |
|---|---:|---:|---:|---:|
| persistence | 0.2337 | -0.374 (0.097) | 0.319 | -17.9% |
| mean | 0.2001 | -0.006 (0.009) | n/a | -1.0% |
| ar1 | 0.1981 | 0.013 (0.009) | 0.319 | +0.0% |
| direct_v | 0.1895 | 0.097 (0.030) | 0.319 | +4.4% |
| linear_event | 0.1981 | 0.014 (0.009) | 0.318 | +0.0% |
| linear_event_asym | 0.1982 | 0.013 (0.005) | 0.320 | -0.0% |
| model_full | 0.1904 | 0.088 (0.021) | 0.303 | +3.9% |
| model_ungated | 0.1927 | 0.067 (0.024) | 0.268 | +2.7% |
| model_frame_present | 0.1977 | 0.018 (0.020) | 0.159 | +0.2% |
| model_frame_past | 0.1905 | 0.088 (0.019) | 0.301 | +3.9% |
| model_frame_future | 0.1848 | 0.142 (0.030) | 0.379 | +6.7% |
| channels_only | 0.1870 | 0.120 (0.034) | 0.353 | +5.6% |
| model_symmetric | 0.1893 | 0.099 (0.023) | 0.320 | +4.5% |
| model_one_step | 0.1905 | 0.087 (0.020) | 0.302 | +3.8% |
| model_no_inertia | 0.2000 | -0.005 (0.011) | 0.040 | -0.9% |

Participant-bootstrap 95% CI on R2 differences (2000 resamples of participants):

- full - best baseline (direct_v): -0.008 [-0.029, +0.016], P(diff<=0)=0.746
- full - ungated (frame-inert): +0.022 [+0.011, +0.034], P(diff<=0)=0.000
- full - frame clamped to PRESENT: +0.070 [+0.041, +0.096], P(diff<=0)=0.000
- full - channels only: -0.031 [-0.053, -0.008], P(diff<=0)=0.995

## Transition asymmetry (effect of event on 1-step valence change)

Positive vs negative event sensitivity. A symmetric mechanism predicts |beta_pos| ~ |beta_neg|.

| Source | beta_pos | beta_neg | |neg|/|pos| |
|---|---:|---:|---:|
| empirical data | -0.0343 | -0.0297 | 0.86 |
| model (full, asymmetric) | -0.0314 | -0.0140 | 0.44 |
| model (symmetric ablation) | -0.0302 | -0.0138 | 0.46 |

## Raw future-frame belief vs measured worry (NOT non-circular)

The valence composite fed to the model includes the worried item. See frame_worry_multilevel.py for the analysis with worry removed from the input and covariates controlled.

- corr(future-frame belief, worry item) = 0.078  (n=11712)
- control corr(reward readout, worry item) = -0.394
