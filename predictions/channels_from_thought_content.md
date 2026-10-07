# Predictions: Channels from thought content (Baumeister et al. 2020, Study 1; Bayer replication)

- Original private commit: `fba8116aa7c4f6c2014a7c58453452209d5835a1`
- Author timestamp: 2026-10-01T15:20:05-04:00
- Committer timestamp: 2026-10-01T15:20:05-04:00
- Source file in that commit: `reviews/CHANNELS_CONTENT.md`

The text below is reproduced verbatim from the file as it stood in that commit, before the analysis was run. The commit sits in the project's private version-control history, which also holds confidential peer-review material, so the commit itself cannot be published. See README.md in this folder.

---

# Channels from thought content in daily life (Baumeister et al. 2020 Study 1; Bayer replication)

## 0. Predictions and plan, committed before analysis

Written 2026-10-01T19:19Z. At the time of writing no channel input defined below had been related to mood
in any analysis. The data files had been opened only to read item labels and counts.

Why this test. The earlier everyday channel test (reviews/CHANNEL_TEST.md) computed the three channels from
the valence sequence alone, so the present and forward channels correlated about 0.89 with valence and
could not be told apart. Here each channel gets its own input from what the person was thinking about.

Data. Baumeister, Hofmann, Summerville, Reiss & Vohs (2020) Study 1 (OSF 9uytp). Same filter as the
orientation and channel tests: valence present, at least 4 signals, ordered by day and signal
(453 participants, 6,544 signals). Mood is "How happy/sad do you feel right now?" (-3..+3). Thought
pleasantness is "Altogether, to what extent were your thoughts about something pleasant/unpleasant?"
(-3..+3). Orientation is the past/present/future checkbox set (onefactor: 1 past, 2 present, 3 future,
4 past+present, 5 past+future, 6 present+future, 7 all three, 8 no time aspect).

Channel inputs.
- Primary (split rule). For a signal with k checked orientations and thought pleasantness c, each checked
  orientation receives c/k and the others 0. Backward input u_B, present input u_P, forward input u_F.
  Signals with no time aspect have all three inputs at 0. Orientation indicators (past, present, future
  checked) enter as main effects, so each channel weight is the slope of mood on thought pleasantness
  within that orientation.
- Robustness 1 (copy rule). Each checked orientation receives the full c.
- Robustness 2 (single focus). Only signals with exactly one orientation checked or none.
- Signed content version. u_B = (happy + proud + relieved + nostalgia) minus (regret + sad +
  disappointed + angry + replaying + what might have been), on past-checked signals; u_F = (planning +
  what you hope to do + what you hope will happen) minus (worries + what you fear will happen), on
  future-checked signals; u_P as in the primary rule. Items come from the branched checklists, so these
  inputs exist only on signals where the branch was shown.

Model.
- Static. Mood_t on u_B, u_P, u_F, the three orientation indicators, the participant-mean-centred previous
  mood within day, and time of day ((hour-9)/9), with participant random intercepts (statsmodels MixedLM);
  random slopes for the three inputs if the model converges. Cluster-robust OLS on within-person centred
  variables as a fallback and as a check.
- Undifferentiated comparison. The same model with one thought-pleasantness term (u_B + u_P + u_F) in
  place of the three.
- Dynamic. A linear Gaussian state space with a slow person level (random walk) and a fast state that
  decays within day and is reset at day boundaries; each input drives the fast state with its own gain.
  Group parameters fitted by maximum likelihood on training participants, scored on held-out
  participants (5 participant folds), against the same model with one shared gain.

Predictions and decision rules.
- P1 (distinct weights). The three static weights differ. Supported if the Wald test of equality of the
  three weights (cluster-robust) gives p < .05 after Benjamini-Hochberg within this family.
- P2 (channels are real in daily life). The channel-specific model predicts held-out participants' mood
  better than the undifferentiated model. Supported if the participant-bootstrap 95% CI of the held-out
  Gaussian log-likelihood difference (per signal) lies above 0, for the static model and for the dynamic
  model separately.
- P3 (same ordering as the gamble task). w_P > w_F > w_B, as in the gamble task's group weights
  (present 0.097, forward 0.049, backward 0.017; ratios F/P 0.51, B/P 0.18). Supported if both ordered
  differences have 95% CIs above 0. The ratios are reported with bootstrap CIs and compared with the
  gamble ratios descriptively.
- P4 (carry-over). Channel inputs at t predict mood at t+1 within day beyond mood at t. Predicted: the
  forward input carries over more than the backward input (anticipation persists, past thoughts are
  absorbed at once). Supported for carry-over if at least one input has a 95% CI above 0; the B versus F
  difference is reported with its CI.
- Robustness rules and the signed content version are secondary and reported in full whatever they show.
- Benjamini-Hochberg is applied over every p-value in this file (primary, robustness, signed, Bayer).

Bayer replication (openESM 0076). The data have mood ("how positive or negative do you feel", 1..5),
single-choice orientation, and "were you thinking about a particular problem" (0 none, 1 unrelated
problem, 2 problem related to the activity), but no thought-pleasantness item. A full replication is not
possible. Planned partial replication: a signed binary content input (thinking about a problem = -1, not
= 0) routed by orientation into backward, present and forward inputs, with the same static model and
equality test. Interaction pleasantness (rated only when an interaction occurred) is a present-only input
and is used as a check on the present weight.
