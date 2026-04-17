# Proposal: REST → FEEL and Interoceptive Load Reinterpretation

## Summary for Manon

Hey Manon — following up on your comments about REST, energy, and dynamic coupling. I dug into the AIF literature and your three proposals — (1) REST → FEEL, (2) energy → interoceptive load, and (3) dynamic coupling between load and cognitive parameters — map cleanly onto existing active inference formalism. Below I lay out how to implement them in a principled way, what changes in the model and what stays the same, and what this buys us clinically.

---

## 1. The core reinterpretation

### REST → FEEL

You're right that REST is misleading. What the model currently calls REST isn't passive conservation — it does three things:

1. **Load reduction**: B_energy gives it a +1.2 shift toward higher energy (equivalently: reduces accumulated interoceptive prediction error)
2. **Present-focused attention**: B_frame assigns 50–60% probability of shifting to the PRESENT temporal frame
3. **Valence stabilisation**: B_valence pulls gently toward neutral (30% pull, 70% stay)

This is exactly what interoceptive processing does: you stop projecting forward or backward, attend to your body in the present, process accumulated signals, and the result is affective regulation. **FEEL** captures this better than REST.

### Energy → Interoceptive load

The key insight — and the formal justification — is that "accumulated unprocessed prediction errors" don't literally accumulate in variational free energy (VFE is computed fresh each timestep). Instead, they accumulate **implicitly through belief drift**:

When the agent selects RECALL, ENGAGE, or FUTURATE, it is effectively allocating precision away from the interoceptive channel. During those timesteps, the agent's beliefs about its bodily state stop updating while the true bodily state continues evolving (metabolism, fatigue, stress hormones, etc.). When the agent eventually selects FEEL and redirects precision to interoception, the mismatch between beliefs and reality surfaces as a burst of prediction error.

This is exactly the allostatic self-efficacy framework of **Stephan et al. (2016)**: interoceptive load = the accumulated divergence between the agent's body model and the body's actual state, measurable as the interoceptive component of VFE. Depression, in this framework, is the inference that one has lost the ability to regulate this quantity — learned interoceptive helplessness.

The mathematical structure of our model is already correct for this interpretation. Where the model currently says "FUTURATE costs energy (-1.2)" we reinterpret: "FUTURATE ignores the body, so interoceptive prediction errors accumulate." Where it says "REST recovers energy (+1.2)" we reinterpret: "FEEL processes bodily signals, reducing the accumulated PE." Same equations, richer interpretation.

---

## 2. What changes in the model

### Naming (cosmetic — no equations change)

| Before | After |
|--------|-------|
| REST | FEEL |
| Energy (M levels) | Interoceptive load (M levels) |
| omega_e: energy estimation precision | omega_i: interoceptive precision |
| "Low energy → valence drop" | "High load → affective dysregulation" |

### New mechanism: fast interoceptive coupling

Currently, the only dynamic parameter is pi_pos, which evolves on a slow timescale via the M5 mood layer (Bayesian update every 50 steps based on accumulated VFE). This captures **mood** — the slow drift of affective disposition.

I propose adding a **fast pathway**: the agent tracks the running interoceptive component of its VFE (exponential moving average). When this interoceptive surprise is high, it directly reduces the effective recall precision (pi_pos_eff). This creates a principled dual-timescale system:

- **Fast** (every step): high interoceptive load → impaired recall → shifted action selection
- **Slow** (every 50 steps): accumulated VFE → mood-level Bayesian inference → base pi_pos adjustment

Formally, let Phi(t) be the EMA of interoceptive accuracy:

```
Phi(t) = (1 - alpha) * Phi(t-1) + alpha * [-log P(o_int | s)]
```

Then:

```
pi_pos_eff(t) = pi_pos_mood(t) / (1 + beta * max(0, Phi(t) - threshold))
```

When Phi is below threshold (body well-predicted), pi_pos_eff equals the mood-derived value — no change from current behaviour. When Phi rises (body poorly predicted), effective recall precision drops. This is principled: if your body model is wrong, your self-model (which includes body awareness) should also be less reliable.

### What does NOT change

- The POMDP architecture (K × M × 3 states, 3 modalities, 4 actions)
- All A, B, C, D matrices (same numbers, same construction)
- The M5 hierarchical mood layer
- Three-channel valence (v_model, v_reward, v_action)
- EFE-based policy selection
- All existing experiments and plotting

---

## 3. Clinical payoff

### Depression

The depressive agent (pi_pos=0.2, omega_e=5.0, c_scale=0.1) already has:
- Intact interoceptive precision (omega_e=5.0) — they can *feel* their body
- Low recall precision — RECALL doesn't work
- Anhedonia (c_scale=0.1) — FUTURATE doesn't produce positive affect either

With the interoceptive load framing: the depressive agent *knows* their load is high (good omega_i) but *can't reduce it* because (a) FEEL only works if the agent's body model is reasonably accurate, which requires sustained attention, and (b) the anhedonia means the agent can't motivate itself to sustain any strategy. This is **Stephan et al.'s interoceptive learned helplessness**: the metacognitive inference that "I cannot regulate my own body" — which is precisely what depression feels like.

### Mania

The manic agent (pi_pos=1.5, omega_e=0.5, c_scale=2.0) has:
- Poor interoceptive precision (omega_e=0.5) — they *can't read* their body
- Low backward-grounded optimism — positivity is projected forward, not grounded in past evidence
- Reward hypersensitivity — amplified preferences drive aggressive FUTURATE selection

With the load framing: the manic agent ignores its body (low omega_i), so interoceptive load accumulates without the agent noticing. The agent keeps FUTURATE-ing because it can't feel the cost. When the load becomes extreme, the energy-valence coupling forces a crash — the "manic episode" collapses into low valence. After recovery (FEEL), the cycle can restart. This is the **bipolar oscillation as emergent interoceptive neglect**.

With the new fast coupling pathway: as interoceptive load rises, it further erodes the already-low pi_pos (1.5), making RECALL even weaker and reinforcing FUTURATE dominance — a positive feedback loop that accelerates the manic escalation phase.

### Healthy regulation

The healthy agent (pi_pos=5.0, omega_e=5.0, c_scale=1.0) naturally alternates between RECALL and FEEL, keeping interoceptive load low. The fast coupling pathway stays inactive (Phi below threshold). This is affective homeostasis: regular interoceptive processing prevents PE accumulation.

### Chronic stress → emergent depression

In the stress decay experiment (Experiment 7), both agents start with identical healthy parameters but different environmental volatility. Under chronic stress (volatility=0.9), the agent accumulates more interoceptive surprise because the volatile environment produces more bodily perturbation. With the fast coupling, this would accelerate the mood decay pathway — the stressed agent's effective pi_pos drops faster, weakening RECALL sooner, and the emergent depressive trajectory appears earlier. This strengthens the existing result.

### Dissociation (BLANK state)

I've also been thinking about a dissociative state — what I'm calling BLANK. This emerges naturally as an extreme case of the model: when interoceptive load is very high AND allostatic self-efficacy is low (FEEL has repeatedly failed to reduce load), the system enters a regime where all four actions have similar (high) EFE values. The policy becomes near-uniform — the agent effectively "gives up" on directed action. This is dissociation as **computational learned helplessness**: a fixed point of the belief dynamics where no action is expected to improve the situation.

This could be modelled either as:
- An emergent regime of the existing 4-action model (no code changes — just documentation of when it occurs)
- A 5th action BLANK = no precision allocation to any channel (future extension)

---

## 4. Literature grounding

The proposed changes connect to a well-established literature:

| Reference | Contribution to our model |
|-----------|---------------------------|
| Seth & Friston (2016) | Interoceptive inference framework — emotions arise from precision-weighted interoceptive PE |
| Stephan et al. (2016) | Allostatic self-efficacy — accumulated interoceptive surprise as depression mechanism |
| Barrett (2017) | Body budgeting — brain's primary job is allostatic regulation, not world-modelling |
| Smith et al. (2020) | Bayesian computational model showing transdiagnostic failure of interoceptive precision adaptation |
| Sandved-Smith et al. (2021) | Meditation as restoration of interoceptive precision — formalises FEEL |
| Pezzulo et al. (2015) | Active inference homeostatic regulation — interoceptive load = allostatic debt |
| Mirza et al. (2016) | Scene construction via active inference — attention as hidden state factor in discrete POMDPs |
| Parr & Friston (2017) | Working memory, attention, and salience in active inference — precision gating of likelihood mappings |

---

## 5. Future extension: attention-mode hidden state

Not for this round, but worth flagging: the most principled implementation of "FEEL allocates precision to interoception" is adding a hidden state factor s_mode in {EXTERNALISE, INTEROCEPTE} that gates the A-matrix precision — sharp for the attended modality, flat for the unattended one. This is the standard discrete POMDP attention mechanism from Mirza et al. (2016, "Scene Construction, Visual Foraging, and Active Inference," *Frontiers in Computational Neuroscience*) and Parr & Friston (2017, "Working Memory, Attention, and Salience in Active Inference," *Scientific Reports*), used in saccade and visual attention models.

This would make the load accumulation fully emergent: when s_mode = EXTERNALISE, the agent's interoceptive A matrix becomes flat (imprecise), so beliefs about bodily states drift. When s_mode = INTEROCEPTE (via FEEL action), precision sharpens, beliefs update, and the accumulated mismatch surfaces as PE. No explicit load variable needed — it emerges from the generative model's own dynamics.

State space would double (K × M × 3 × 2) but the math stays standard. This could be a follow-up paper.

---

## Summary

| Aspect | Status |
|--------|--------|
| REST → FEEL | Rename only — no equations change |
| Energy → interoceptive load | Reinterpretation only — same state space and dynamics |
| Fast interoceptive coupling | One new mechanism: EMA of interoceptive VFE modulates effective pi_pos |
| Existing results | All preserved — coupling adds a fast pathway alongside existing slow M5 mood |
| Clinical narratives | Strengthened — depression as interoceptive helplessness, mania as interoceptive neglect |
| Dissociation | Emerges naturally as a fixed point when all actions have similar high EFE |
| Mathematical rigour | Grounded in Stephan et al. 2016, Seth & Friston 2016, Parr & Friston 2017 |
