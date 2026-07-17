# Rebuilt Empirical Validation Report

- Dataset: Geschwind/Bringmann residual-depression ESM (`data_raw/geschwind_2013_s004.csv`).
- Participants used: 129; prediction records: 11734.
- Cross-validation: whole participants held out per fold.
- Valence normalised to [0,1]; RMSE and R2 in that scale; each predictor gets an optimal train-fit linear calibration.

## Horizon h = 1 step(s) ahead

| Predictor | RMSE | R2 | r | skill vs AR(1) |
|---|---:|---:|---:|---:|
| persistence | 0.2359 | -0.393 | 0.306 | -23.6% |
| mean | 0.2007 | -0.007 | n/a | -5.1% |
| ar1 | 0.1909 | 0.089 | 0.306 | +0.0% |
| linear_event | 0.1908 | 0.089 | 0.307 | +0.0% |
| linear_event_asym | 0.1908 | 0.090 | 0.308 | +0.1% |
| model_full | 0.1849 | 0.145 | 0.384 | +3.1% |
| model_symmetric | 0.1848 | 0.146 | 0.385 | +3.2% |
| model_one_step | 0.1851 | 0.144 | 0.382 | +3.0% |
| model_no_inertia | 0.1876 | 0.120 | 0.351 | +1.7% |
| readout_joffily | 0.2002 | -0.002 | 0.069 | -4.9% |
| readout_pattisapu | 0.1999 | 0.001 | 0.083 | -4.7% |
| readout_hesp | 0.1952 | 0.048 | 0.232 | -2.2% |

## Horizon h = 2 step(s) ahead

| Predictor | RMSE | R2 | r | skill vs AR(1) |
|---|---:|---:|---:|---:|
| persistence | 0.1952 | 0.045 | 0.525 | -9.9% |
| mean | 0.2005 | -0.007 | n/a | -12.9% |
| ar1 | 0.1775 | 0.210 | 0.525 | +0.0% |
| linear_event | 0.1774 | 0.212 | 0.525 | +0.1% |
| linear_event_asym | 0.1774 | 0.212 | 0.526 | +0.1% |
| model_full | 0.1716 | 0.263 | 0.515 | +3.4% |
| model_symmetric | 0.1715 | 0.263 | 0.515 | +3.4% |
| model_one_step | 0.1708 | 0.269 | 0.521 | +3.8% |
| model_no_inertia | 0.2003 | -0.004 | 0.080 | -12.8% |

## Horizon h = 3 step(s) ahead

| Predictor | RMSE | R2 | r | skill vs AR(1) |
|---|---:|---:|---:|---:|
| persistence | 0.2337 | -0.374 | 0.319 | -17.9% |
| mean | 0.2001 | -0.006 | n/a | -1.0% |
| ar1 | 0.1981 | 0.013 | 0.319 | +0.0% |
| linear_event | 0.1981 | 0.014 | 0.318 | +0.0% |
| linear_event_asym | 0.1982 | 0.013 | 0.320 | -0.0% |
| model_full | 0.1995 | 0.000 | 0.103 | -0.7% |
| model_symmetric | 0.1985 | 0.010 | 0.139 | -0.2% |
| model_one_step | 0.2001 | -0.006 | 0.067 | -1.0% |
| model_no_inertia | 0.1944 | 0.050 | 0.231 | +1.9% |

## Transition asymmetry (effect of event on 1-step valence change)

Positive vs negative event sensitivity. A symmetric mechanism predicts |beta_pos| ~ |beta_neg|.

| Source | beta_pos | beta_neg | |neg|/|pos| |
|---|---:|---:|---:|
| empirical data | -0.0343 | -0.0297 | 0.86 |
| model (full, asymmetric) | -0.0334 | -0.0141 | 0.42 |
| model (symmetric ablation) | -0.0324 | -0.0139 | 0.43 |

## Non-circular test: future-frame belief vs measured worry

Worry is never given to the model; the future-frame belief is driven only by valence+event observations.

- corr(future-frame belief, worry item) = 0.166  (n=11712)
- control corr(reward readout, worry item) = -0.399
