# Design justification: every model choice grounded in the literature

Each structural and parametric choice in the generative model, with its rationale
and citation. Keys marked ✓ exist in `references.bib`; keys marked **[ADD]** need a
bib entry (cited by name in the papers but not yet in the .bib).

## Core architecture

| Choice | Rationale | Citation |
|---|---|---|
| Active-inference / free-energy generative model | Perception, learning, action as free-energy minimisation | parr2022active ✓ |
| Attention = precision-weighting of prediction errors | Grounds "temporal framing = precision redistribution" | feldman2010attention ✓ |
| Metacognitive control of attention (covert mental action) | Frame selection is a self-directed, second-order inference | sandved-smith2021 **[ADD]** |
| Constructed-emotion / interoceptive-inference stance | Emotions as concepts over allostatic regulation | barrett2017theory ✓; seth2016interoceptive ✓ |
| Circumplex readout (valence × arousal) | Two-D affect space; arousal = posterior entropy | russell1980circumplex ✓; pattisapu2024free ✓; mehrabian1996pleasure ✓ |

## The three valence channels (the integration claim)

| Channel | What it is | Citation |
|---|---|---|
| Model valence $v_{\text{model}}$ (backward) | Rate of change of VFE, $-dF/dt$ | joffily2013emotional ✓ |
| Reward valence $v_{\text{reward}}$ (present) | Reward prediction error $U-\mathbb E[U]$ | pattisapu2024free ✓ |
| Action valence $v_{\text{action}}$ (forward) | Affective charge of policy revision | hesp2021deeply ✓ |
| Integration of all three | Each predecessor is *insufficient* alone; combined they span backward/present/forward | joffily2013emotional ✓; pattisapu2024free ✓; hesp2021deeply ✓ (empirical: this record) |

## Hidden-state factors

| Factor | Rationale | Citation |
|---|---|---|
| Valence $v$ at granularity $K$ | Emotional granularity is adaptive; coarse affect = vulnerability | barrett2017theory ✓; tugade2004 **[ADD]** |
| Energy / interoceptive $e$ | Allostatic body-budget; interoceptive precision | seth2016interoceptive ✓; stephan2016allostatic ✓; barrett-simmons2015 **[ADD]** |
| Temporal frame $f$ (past/present/future) | Mental time travel; remembering & imagining share one machinery | suddendorf-corballis1997 **[ADD]**; schacter-addis2007 **[ADD]** |

## The six framing actions

| Action | Rationale | Citation |
|---|---|---|
| RECALL (past, narrative stabilisation) | Self-memory system; working-self gates retrieval; positive bias in health | conway-pleydell-pearce2000 **[ADD]**; williams2007autobiographical ✓ |
| ENGAGE (present, perception–action coupling) | Flow-like present engagement | (Rietveld-Kiverstein2014 **[ADD]**) |
| FUTURATE (prospection) | Constructive episodic simulation; sophisticated inference | schacter-addis2007 **[ADD]**; friston2018deep ✓; hesp2020sophisticated ✓ |
| FEEL (interoceptive processing) | Allostatic self-efficacy; effort/energy restoration | stephan2016allostatic ✓ |
| DISSOCIATE (temporal unmooring) | Dissociation ↔ undirected past/future thought, low present awareness | vannikov2018dissociation ✓ |
| ABSTRACT (ungrounded future cognition) | Construal-level; overgeneral abstract rumination | williams2007autobiographical ✓; tropeliberman **[ADD]** |
| Energy costs (FUTURATE > RECALL; FEEL restores) | Deliberation is metabolically rationed; effort cost | treadway2009effort ✓; treadway2012effort ✓ |

## Extensions and the deep model

| Choice | Rationale | Citation |
|---|---|---|
| Habit priors (E-vector over policies) | EFE-suboptimal, habitual action tendencies; ruminative habit | dacosta2020active ✓; watkins2008constructive ✓; ehring2008repetitive ✓ |
| Asymmetric hedonic sensitivity ($c_{\text{pos}}\neq c_{\text{neg}}$) | Reward vs punishment sensitivity dissociate; anhedonia is reward-specific *at the reactivity/neural level* | eshel2010reward ✓; pizzagalli2014depression ✓; pirrung2025 **[ADD]** |
| Preserved interoceptive preference under low reward | Effort-cost computation intact under reduced reward | treadway2012effort ✓; stephan2016allostatic ✓ |
| Mood layer (M5 POMDP over $\pi_{\text{pos}}$) | Mood/emotion timescale separation; slow valence-state inference | hesp2021deeply ✓; eldar2016mood ✓ |
| Counterfactual emotions (regret/relief) [GENERATIVE, not predictive] | Regret = comparison of obtained vs foregone; sophisticated (counterfactual) affective inference | hesp2020sophisticated ✓; coricelli2005 **[ADD]**; camille2004 **[ADD]** |

## Emotion taxonomy grounding

| Choice | Rationale | Citation |
|---|---|---|
| VFE/EFE → temporal taxonomy of emotion | Extends appraisal theory by grounding the temporal axis in free energy | smith1985patterns ✓; ortony1988cognitive ✓ |
| Backward emotions (regret, guilt, pride) | VFE evaluation of past model fit | joffily2013emotional ✓ |
| Forward emotions (hope, fear, anxiety) | EFE evaluation of expected performance | parr2022active ✓; hesp2021deeply ✓ |

## Clinical mapping (two-level, honestly scoped)

| Claim | Support level | Citation |
|---|---|---|
| Reduced reward reactivity/valuation (`c_pos↓`) in depression | Neural (RewP) + self-report; NOT behavioural-choice | pirrung2025 **[ADD]**; pizzagalli2014depression ✓ |
| General precision/learning-rate reduction in depression | Behavioural (reward+punishment RL) | huys2013mapping ✓ |
| Rumination = past-directed / worry = future-directed repetitive thought | Clinical phenomenology | nolenhoeksema2008rethinking ✓; ehring2008repetitive ✓; berg2022rumination ✓ |
| Chronic-stress mood attractor | Momentary-assessment mood dynamics | wichers2010momentary ✓; mason2017mood ✓ |

## Empirical benchmarks (external datasets)

| Benchmark | Use | Source |
|---|---|---|
| Rutledge GBE happiness (47k) | Affect prediction; integration > single channel | rutledge2014 **[ADD]** (Dryad CC0) |
| Geschwind ESM (residual depression) | Affect-dynamics prediction; frame→worry | geschwind2013 **[ADD]** |
| Sugawara & Katahira / Palminteri (counterfactual RL) | Counterfactual signature (regret→switch) | sugawara2021 **[ADD]**; lefebvre-palminteri2017 **[ADD]** |
| Autobiographical-memory meta-analysis | RECALL/specificity direction | williams2007autobiographical ✓ |

---

### Bib entries to add (cited by name, not yet in references.bib)
sandved-smith2021, tugade2004, barrett-simmons2015, suddendorf-corballis1997,
schacter-addis2007, conway-pleydell-pearce2000, rietveld-kiverstein2014,
tropeliberman, pirrung2025, coricelli2005, camille2004, rutledge2014,
geschwind2013, sugawara2021, lefebvre-palminteri2017.
