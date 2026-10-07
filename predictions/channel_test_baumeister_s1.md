# Predictions: Channel test in daily life (Baumeister et al. 2020, Study 1)

- Original private commit: `9d9b04865c23baae3a5adf86601fc28e5c0ce57b`
- Author timestamp: 2026-09-30T18:22:06-04:00
- Committer timestamp: 2026-09-30T18:22:06-04:00
- Source file in that commit: `reviews/CHANNEL_TEST.md`

The text below is reproduced verbatim from the file as it stood in that commit, before the analysis was run. The commit sits in the project's private version-control history, which also holds confidential peer-review material, so the commit itself cannot be published. See README.md in this folder.

---

# Channel-specific test of the three valence channels in daily life (Baumeister et al. 2020, Study 1)

## 0. Preregistration of this analysis

Written 2026-09-30T22:21Z, before any channel value was related to any thought-content or emotion item.
Nothing below this section existed when it was written. Item names and coding were checked only for
counts and branching (see section 1), never against channel values.

Data: Baumeister, Hofmann, Summerville, Reiss & Vohs (2020) Study 1, OSF 9uytp, signal level.

Channels: the backward (B), present (P) and forward (F) channels of model v2.1 (`model_v2.py`,
unmodified), computed causally at each signal from the participant's valence sequence, with group
parameters fitted on the other participants (5 participant folds). Robustness: the same channels from the
v1 discrete model (`v_model`, `v_reward`, `v_action`, pooled round-3 parameters).

Sign conventions: v_B > 0 when surprise falls (positive backward valence), v_P > 0 when the present
input beats its running expectation, v_F > 0 when the expected next change in valence is positive.

Predictions.

- P1. The backward channel predicts regret and replaying (past-thought content) and disappointment.
  Predicted sign negative (regret goes with negative backward valence).
- P2. The forward channel predicts worry and fear with a negative sign and planning (and hoping) with a
  positive sign.
- P3. The present channel predicts neither regret nor worry once concurrent valence is controlled
  (coefficient CI includes zero).
- P4. Discriminant validity. Each channel predicts its own items better than either other channel does,
  in held-out log-likelihood with valence in every model. Own items: B for regret, replaying,
  past-disappointed, what-might-have-been, and the momentary disappointment rating. F for worry, fear,
  planning, hoping, and the momentary anxiety rating.

Decision rules. A prediction is supported when the stated coefficient has the stated sign with a 95%
cluster-robust CI excluding zero (P1, P2), or the held-out log-likelihood difference has a
participant-bootstrap 95% CI above zero (P4). Contradicted when the CI excludes zero in the opposite
direction. Otherwise inconclusive. P3 is supported when the CIs include zero for both regret and worry.

Primary models. Cluster-robust logistic regression (participant clusters) of each binary item on the
three standardized channels, concurrent valence, time of day and orientation dummies (past, present,
future checked), all signals. Continuous 0-4 ratings (disappointed, anxious) by cluster-robust linear
regression on the same predictors. Secondary: the same models within the branch that showed the item
(past-checked signals for past items, future-checked for future items), and within-person centred
channels. statsmodels is not installed in the project Python, so mixed models are replaced by
cluster-robust estimates, as the brief allows.
