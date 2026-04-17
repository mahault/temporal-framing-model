# Counterfactual Temporal Framing Model

An active inference simulation of how counterfactual temporal framing — the tendency to mentally inhabit past, present, or future — interacts with affective granularity, interoceptive precision, and allostatic regulation to produce distinct clinical phenotypes (healthy, depressive, manic, psychotic).

Companion code for two papers:

1. **Computational Modelling of Bipolar Disorder** (`computational_modelling_bipolar_disorder.tex`)
2. **Affective Valence and Temporal Framing** (`affective_valence_temporal_framing.tex`)

## Model

The agent is a partially observable Markov decision process (POMDP) with factored hidden states:

| Factor | Dimension | Description |
|--------|-----------|-------------|
| **Valence** | K levels (2--8) | Subjective well-being |
| **Interoceptive load** | M levels (default 8) | Accumulated unprocessed bodily prediction error |
| **Frame** | 3 (Past / Present / Future) | Active temporal orientation |

The agent selects among five **temporal-framing policies**:

| Action | Description |
|--------|-------------|
| **RECALL** | Re-engage past events (positive bias gated by `pi_pos`) |
| **ENGAGE** | Attend to the present (valence-preserving) |
| **FUTURATE** | Simulate future scenarios (high load cost, high potential valence gain) |
| **FEEL** | Active interoceptive processing — reduces accumulated prediction error |
| **BLANK** | Null action (flat affect, present-locked) — emergent under regulatory collapse |

Policy selection follows expected free energy (EFE) minimisation with softmax action selection.

### Three-channel valence

Valence is computed as a weighted sum of three complementary channels:

| Channel | Timescale | Formulation | Source |
|---------|-----------|-------------|--------|
| v_model | Backward | -dF/dt (rate of VFE reduction) | Joffily & Coricelli (2013) |
| v_reward | Present | reward prediction error | Pattisapu et al. (2024) |
| v_action | Forward | -G(pi*) (negated EFE of selected policy) | Hesp et al. (2021) |

### Hierarchical mood layer (M5)

A slow Bayesian update over recall precision `pi_pos` operates every `T_mood` steps, creating dual-timescale dynamics: fast emotion (every step) and slow mood (every T_mood steps).

### Interoceptive load coupling

Interoceptive surprise is tracked via an exponential moving average (EMA). When accumulated load exceeds a threshold, it impairs recall precision on a fast timescale (every step), alongside the slow M5 mood pathway. This implements Stephan et al.'s (2016) allostatic self-efficacy: meta-inference over one's own regulatory capacity.

### Clinical parameters

| Parameter | Symbol | Mechanism |
|-----------|--------|-----------|
| Positive-belief precision | `pi_pos` | Controls D prior skew and RECALL pull strength |
| Affective granularity | `K` | Resolution of the valence state space |
| Interoceptive precision | `omega_e` | Accuracy of interoceptive load observation (A_int) |
| Reward sensitivity | `gamma` | Softmax temperature for policy selection |
| Reward scaling | `c_scale` | Amplitude of C vector preferences |

### Phenotypes as parameter regimes

| Phenotype | `pi_pos` | `K` | `omega_e` | `gamma` | `c_scale` | Description |
|-----------|----------|-----|-----------|---------|-----------|-------------|
| Healthy | 5.0 | 8 | 5.0 | 16.0 | 1.0 | Balanced precision, high granularity |
| Depressive | 1.0 | 4 | 1.0 | 4.0 | 0.3 | Low positive-belief precision, anhedonic |
| Manic | 2.0 | 4 | 0.5 | 64.0 | 3.0 | Overconfident, poor interoceptive reading |
| Vulnerable | 1.0 | 4 | 0.3 | 16.0 | 1.5 | Psychosis-prone, very low interoceptive precision |

## Experiments

The simulation suite runs 11 experiments generating 15 publication figures:

| # | Experiment | Figures | Key question |
|---|------------|---------|-------------|
| 1 | Phenotype comparison | fig1--fig2, fig5--fig6, fig8 | Do parameter regimes differentiate clinical profiles? |
| 2 | Granularity sweep | fig3 | How does K affect valence stability and action selection? |
| 3 | Parameter landscape | fig4 | How do pi_pos and omega_e jointly determine the affective regime? |
| 4 | Emotion validation | fig7 | Do discrete emotions map onto the circumplex via temporal frame? |
| 5 | Temporal aiming | fig9 | What temporal orientations characterise each emotion? |
| 6 | Feedback reliance | fig10 | How does recall impairment degrade valence? |
| 7 | Framing dynamics | fig11 | How do actions drive temporal frame transitions? |
| 8 | Chronic stress | fig12 | Does stress narrow temporal orientation toward the present? |
| 9 | pi_pos dynamics | fig13 | Does the M5 mood layer produce slow mood drift? |
| 10 | Stress decay | fig14 | Can sustained stress produce emergent depression via pi_pos decay? |
| 11 | Psychotic decompensation | fig15 | Does BLANK emerge when all directed actions fail? |

See `RESULTS.md` for full numerical results from the latest run.

## Usage

```bash
# Full run (~3 min, T=300, T_sd=3000)
python run.py

# Quick test (~30s, T=100, T_sd=300)
python run.py --quick
```

Figures are saved to `figures/`.

## Requirements

Python 3.8+

```bash
pip install numpy matplotlib
```

## Project structure

```
temporal-framing-model/
  generative_model.py    # POMDP specification (A, B, C, D matrices)
  agent.py               # Two-level hierarchical active inference agent
  environment.py         # Stochastic generative process
  experiments.py         # Experiment configurations and runners
  plotting.py            # Publication figure generation
  run.py                 # Entry point
  emotion_diagnostic.py  # Diagnostic 3D emotion visualisation
  references.bib         # Shared bibliography
  computational_modelling_bipolar_disorder.tex   # Paper 1
  affective_valence_temporal_framing.tex         # Paper 2
  RESULTS.md             # Full simulation results
  figures/               # Generated figures (fig1--fig15)
```

## References

- Pattisapu, C., Verbelen, T., Pitliya, R. J., Kiefer, A. B., & Albarracin, M. (2025). Free Energy in a Circumplex Model of Emotion. *IWAI 2024, CCIS* 2193, 34--46.
- Joffily, M. & Coricelli, G. (2013). Emotional valence and the free-energy principle. *PLoS Computational Biology*, 9(6), e1003094.
- Hesp, C., Smith, R., Parr, T., Allen, M., Friston, K. J., & Ramstead, M. J. D. (2021). Deeply felt affect: The emergence of valence in deep active inference. *Neural Computation*, 33(2), 398--446.
- Barrett, L. F. (2017). The theory of constructed emotion. *Social Cognitive and Affective Neuroscience*, 12(1), 1--23.
- Stephan, K. E., et al. (2016). Allostatic self-efficacy: A metacognitive theory of dyshomeostasis-induced fatigue and depression. *Frontiers in Human Neuroscience*, 10, 550.
- Seth, A. K. & Friston, K. J. (2016). Active interoceptive inference and the emotional brain. *Phil. Trans. R. Soc. B*, 371(1708), 20160007.
- Parr, T., Pezzulo, G., & Friston, K. J. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.
