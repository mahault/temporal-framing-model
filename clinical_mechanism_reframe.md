# Reframed clinical mechanism — paper-ready draft text

Reframes the depressive/anhedonic mechanism as a **two-level** account, honestly
scoped to what the data (see `EMPIRICAL_RECORD.md`) support. Replaces overclaims
that asymmetric hedonic sensitivity (`c_pos≠c_neg`) is behaviourally validated.

Key principle: `c_pos` is a **reward-valuation / reactivity** parameter (it
scales reward value in `C`), so it is validated at the level of reward
reactivity (neural RewP, self-reported anhedonia) — *not* at the level of
behavioural-choice individual differences, where it is not recoverable. A
*separate*, general reduction in learning rate/precision is what the behavioural
reinforcement-learning data show.

---

## For Paper 1 — replace/augment the "Asymmetric hedonic sensitivity" passage

> We model reduced positive-hedonic sensitivity as a lowered scaling `c_pos` of
> the positive entries of the preference vectors. This should be read as a
> **reward-valuation** parameter — the precision with which rewarding outcomes
> are treated as preferred — rather than a claim about reinforcement-learning
> rate. Interpreted at this level, the parameter is consistent with the two
> lines of evidence that bear on reward *reactivity* in depression: the
> reward-specific hypoactivation of ventromedial frontal cortex indexed by the
> Reward Positivity \citep{pirrung2025}, and the large group-level deficits in
> self-reported anticipatory and consummatory pleasure that characterise
> anhedonia \citep{...TEPS/SHAPS...}.
>
> We are deliberately careful about the scope of this claim. In our own
> analyses, individual differences in reward sensitivity estimated from
> behavioural choice did not track self-reported anhedonia, and a direct
> comparison of reward- versus punishment-learning in depression revealed a
> *general* reduction in learning rate across both valences rather than a
> reward-selective one. We therefore treat `c_pos↓` as a hypothesis about
> reward valuation/reactivity — supported at the neural and experiential levels
> — and **not** as a behaviourally-validated learning asymmetry. The
> valence-general component of the depressive deficit is better captured by a
> separate reduction in overall precision (below).

## New passage — general precision / learning-rate reduction (the empirically robust part)

> Independently of any hedonic asymmetry, depression is associated with a
> **general reduction in the rate at which outcomes update beliefs** — a lowered
> learning rate / precision that is not specific to reward. In a reward-and-
> punishment reversal-learning task, depressed participants showed reduced
> learning rates in *both* the reward and punishment conditions (Cohen's
> d ≈ −0.5 to −0.6), with no reward-selective advantage for controls. Within the
> present model this maps naturally onto reduced precision on prediction errors
> (equivalently, a flatter effective `c_scale` and lower policy precision),
> which blunts belief updating regardless of outcome valence. This general
> precision reduction, rather than a reward-specific asymmetry, is the component
> of the depressive phenotype for which we have direct behavioural support.

## Discussion — what the model predicts well, and its limits

> The strongest empirical support for the framework is dynamical rather than
> clinical-parametric: driven by real experience-sampling sequences, the model
> predicts short-horizon within-person affect better than linear autoregressive
> and event-regression baselines (out-of-sample, whole participants held out;
> roughly double the explained variance at one step), and its latent
> temporal-frame state tracks an independently measured worry item that is never
> supplied to the model. By contrast, the model's finer clinical parameters —
> the hedonic asymmetry `c_pos/c_neg` and the depth of counterfactual rollout —
> are not separable in passive affect-tracking data: under observation-driven
> filtering the agent occupies present-oriented modes almost exclusively, so
> these parameters, which act through the selection of past- and future-oriented
> policies, have little behavioural leverage there. Testing them requires
> choice- and approach/avoidance-based paradigms, which we identify as the
> appropriate next empirical step.

---

## For Paper 2 (bipolar) — depressive profile

The depressive phenotype currently leans on `c_pos ≪ c_neg`. Reframe as:

- **Primary, empirically supported:** reduced overall precision / reward
  sensitivity (general learning-rate reduction) → blunted belief updating,
  low mood-state precision, RECALL impairment via low `pi_pos`.
- **Secondary, neural/experiential:** reduced reward reactivity (`c_pos↓`) as a
  reward-valuation deficit consistent with RewP and anhedonia measures — flagged
  as such, not as a behaviourally-validated choice asymmetry.
- Remove any statement implying the pos/neg asymmetry is confirmed by task
  behaviour; cite the general-deficit finding for the valence-nonspecific part.

## Citations to add / verify
- `pirrung2025` — Pirrung, Singh, Hogeveen, Quinn & Cavanagh (2025), *Biol.
  Psychiatry: CNNI*, "Hypoactivation of the ventromedial frontal cortex in MDD:
  an MEG study of the Reward Positivity." (PubMed 39551134)
- Reward+punishment reversal-learning source for the general-deficit claim
  (osf.io/2gq96 dataset paper) — verify full citation before use.
- TEPS/SHAPS anhedonia group-difference support (from ds005356 phenotype, or a
  standard anhedonia reference).
