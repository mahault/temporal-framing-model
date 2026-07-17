# Adversarial audit + phenotype scrutiny findings (2026-07-17)

Full re-audit of `affective_valence_temporal_framing_unified.tex` after the model
revision and PAD removal. Four parallel adversarial agents (superiority, writing,
coherence, citations) plus a physical phenotype-figure scrutiny (the PAD-level
"emergent vs hand-tuned" test). Findings below, prioritized. Nothing fixed yet — this
is the work plan.

---

## P0 — INTEGRITY: phenotype figures scrutinized like PAD

Method: re-run each cited phenotype claim across 6–10 seeds; a claim is *emergent* if
its direction holds across seeds, *hand-tuned/fragile* if it needs a specific seed.

### P0.1 Favorable-seed point numbers (mild cherry-pick; fixable, makes claim stronger)
The **directions are robustly emergent** but the **specific numbers reported are the
seed=42 values at the favorable tail** of the seed distribution:

| Claim (paper, seed=42) | Seed-mean ± sd (n=8) | Direction robust? |
|---|---|---|
| healthy RECALL 26% | 19.6% ± 4.7 (range 12–27) | impaired <1% in all seeds ✓ |
| stressed future-frame 0.66 | 0.56 ± 0.06 | stressed>healthy in all 8 ✓ |
| stressed reward-valence −0.42 | −0.31 ± 0.09 | stressed<healthy in all 8 ✓ |
| healthy reward-valence +0.10 | +0.09 ± 0.04 | — |

**Fix:** report seed-averaged mean ± sd with the "holds in all N seeds" robustness
statement instead of single-seed point values. This converts a favorable-seed number
into a robust statistical claim — stronger and more defensible.

### P0.2 "Emergent depression under chronic stress" (fig14, fig:mood) — CONTRADICTED, REMOVE
Claim (§dynamics line ~535 + fig14 title "Emergent Depression Under Chronic Stress"):
"two agents with identical healthy initial conditions but different volatility diverge,
one drifting to high pi_pos and the other crashing below the sigmoid knee"; the stressed
agent develops a "stable depressive attractor."

**10-seed test (T=3000) result — the model does NOT produce this:**
- Stressed agent ends LOWER pi_pos than stable (the "emergent depression" claim): **0/10**.
- Stressed ends HIGHER: **6/10**. Both converge to same basin: **4/10**.
- At the paper's own **seed=42**, both agents end HIGH (stressed 7.36, stable 7.48; the
  θ knee is 2.0) — no crash. fig14 panel (a) shows the stressed agent dips transiently
  (touches θ near t=900) then **recovers to ~7.4 and stays**. It does not end depressed.

**Verdict: this is worse than PAD** — PAD was hand-tuned but showed something; this
claims "emergent depression" that the model contradicts in 10/10 seeds (stressed never
ends lower). What the model actually does: chronic volatility yields a more volatile,
higher-entropy mood trajectory with transient depressive dips, but the agent RECOVERS to
the healthy attractor. No stable emergent depression from volatility alone.

**Fix: REMOVE fig14 + the "emergent mood divergence / stable depressive attractor"
claims** (the "two agents diverge / one crashes" sentences in the chronic-stress
section, and fig:mood). Keep the M5 mood layer as an architectural component (it is used
in the model), but drop the unsupported demonstration. Optionally report the honest
negative in Limitations: the model does not yet produce stable stress-induced depression;
volatility perturbs mood transiently but the agent recovers.
NOTE: fig12 (chronic-stress temporal *fixation*: future-dominance + negative valence) IS
robust (stressed>healthy future in 8/8 seeds; stressed<healthy valence in 8/8) — keep it,
with seed-averaged numbers (P0.1).

### P0.3 ESM params are genuinely fit (clean)
Confirmed earlier: pi_pos=2, inertia=0.5, omega_e=5 came from an actual grid search on
training participants scored on held-out folds (`fit_params.py`). Not hand-picked.

---

## P1 — SUPERIORITY over prior models (the rejection reason; strategic reframe)

Verdict from the superiority agent: superiority is **NOT airtightly demonstrated on
quantitative grounds, and the paper leads with its weakest card**.
- Rutledge head-to-head is a **tie/recovery** (0.144 vs 0.146), and two predecessor
  cells (Joffily 0.000, Hesp 0.034) are disclosed category mismatches. The only genuine
  out-of-sample **beat** (ESM ~2×) is vs AR(1)/regression baselines, not vs the three
  models on their own terrain.
- The paper **buries its strongest assets**: (i) the ESM no-inertia ablation showing
  the beat is real forward dynamics not persistence; (ii) categorical capabilities the
  predecessors structurally lack (counterfactual regret→switch t=10.4; latent frame
  tracking an unshown symptom; mood-timescale dynamics).

**Reframe (no new data):**
1. Lead the abstract's empirical sentence and §empirical with the **ESM beat + no-inertia
   ablation**, not the Rutledge recovery. Demote Rutledge to a **necessity/subsumption**
   result: "each predecessor channel is individually insufficient; only integration
   reaches the reference ceiling, recovering it without injecting EV."
2. Pull Joffily 0.000 / Hesp 0.034 OUT of a table titled "head-to-head" (implies fair
   fight) into a clearly-labelled **structural-reduction** panel, or state the reduction
   argument with no score. Own plainly that the forward-channel +30% **recovers the
   reference's EV term** (defuses the circularity charge).
3. Fix the **Hesp inconsistency**: text calls Hesp a "single-channel restriction" but the
   architecture caption says "forward channel + mood layer," and we ADOPT Hesp's M5 mood
   layer wholesale. State this as an explicit **design/integration** statement, not "we
   beat Hesp."
4. Make the superiority case **qualitative-first**: lead with the three capabilities the
   rivals cannot produce; let R² play the supporting "necessary and matches the gold
   standard" role. A reviewer forgives a tie on a static task; cannot dismiss a
   capability the rivals structurally lack.

---

## P2 — WRITING blockers (defensibility; local fixes)

- **B1 hope/fear contradiction:** §3.2 derives hope/fear from the *backward* VFE second
  derivative (line ~223); §5.4 taxonomy classifies hope/fear as *forward/EFE* (line ~426).
  Same words, incompatible generators. Reconcile (relabel §3.2 states, or add a
  distinct-constructs-sharing-a-folk-label sentence).
- **B2 no-inertia ablation overclaim (introduced this session):** "advantage largely
  intact at every horizon" is FALSE for the reliability sample (0.46<0.48 at h=1,
  0.34=0.34 at h=2). Reword: intact on clinical, parity on the already-persistent
  reliability sample.
- **B3 orphaned datasets:** Palminteri (n=20) and the 181-ES autobiographical-memory
  meta-analysis are in Table 2 as evidence but never reported in the body. Report each in
  one sentence or remove the rows.
- **B4 pi_pos undefined:** used ~15× before an implicit definition. Gloss at first use
  (line ~137): "positive-belief precision, the confidence of the slow mood belief that
  outcomes will be favourable; formalised in §4.3."
- **B5 reimplementation gaps:** θ and σ in α=σ(pi_pos−θ) (line ~310), exact B-matrix
  masses for FEEL/DISSOCIATE/ABSTRACT (ranges given), and τ_model/τ_reward/τ_action
  (line ~381) never assigned. Give values or a precise supplement pointer.
- **B6 "manic profile"/"profiles":** referenced only in fig11 caption (line ~508);
  this unified paper defines only healthy/recall-impaired/stressed. Introduce or excise.

### Clarity / precision (P2b)
- C1 split the two ~6-line abstract run-on sentences.
- C2 §3.2 verbless compressed sentence.
- C3 "The loop admits two regimes" — ambiguous antecedent; name it.
- C4 RECALL "self-transition to PAST" vs others "assigns X% to <frame>" — make parallel.
- C5 "narrative inertia" / "absorptive narrative inertia" undefined; gloss at first use.
- C6 subsection titles promise more than delivered ("Scope", "identity pole",
  "interoception"); trim to content.
- P1 standardize the 2.18× effect phrasing (currently "about twice"/"roughly double"/
  "2.2×"/"~2×") → one form, "2.2×".
- P2 regret ratio 0.45/0.26 = 1.7×, not 1.8× (fig:cf caption).
- P3 "+30%" → "30% relative increase in R² (0.111→0.144)".
- P4 t=10.4 report with df and p.
- S1 §6 (predecessors) separated from its proof (§9 head-to-head) by two sections.
- S2 §3.4 constructionism reads as digression; compress.
- S3 abstract omits the strongest-honesty result (orientation test: prediction held,
  default assumption falsified).
- Padding: line ~841 restates ~838 in same paragraph; consciousness disclaimer;
  over-explained tractability argument.

---

## P3 — COHERENCE re-check: clean

No broken/dangling items after PAD removal (0 circumplex/octant/PAD/fig:pad hits); all
`\ref`/`\label` resolve; LaTeX balanced; all 39 cite keys resolve; new citations
(sugawara2021dissociation, palminteri2017confirmation, mulholland2023 with corrected
author names) verified. russell/mehrabian orphaned in this paper but correctly retained
(cited by the sibling bipolar paper). Only nit: fig:cf "1.8×" → "1.7×" (= P2 P2).
Two figure-derived numbers (reward valence −0.42/+0.10, frame 0.32/0.46) not in
EMPIRICAL_RECORD — corroborate on regeneration (and see P0.1: use seed-means).

---

## Execution order
1. Finish the 10-seed mood-divergence test → decide fig14/mood fate (P0.2).
2. Phenotype numbers → seed-averaged + robustness (P0.1).
3. Superiority reframe: abstract + §empirical + predecessors + Hesp (P1).
4. Writing blockers B1–B6 (P2).
5. Clarity/precision C/P/S (P2b).
6. Recompile, re-verify, refresh deliverables, commit.
