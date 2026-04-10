# Counterfactual Temporal Framing in Bipolar Disorder

An active inference simulation of how counterfactual temporal framing — the tendency to mentally inhabit past, present, or future — interacts with affective granularity and interoceptive precision to produce distinct clinical phenotypes (healthy, depressive, manic).

## Model

The agent is a partially observable Markov decision process (POMDP) with factored hidden states:

| Factor | Dimension | Description |
|--------|-----------|-------------|
| **Valence** | K levels (2–8) | Subjective well-being |
| **Energy** | M levels (default 5) | Metabolic resource budget |
| **Frame** | 3 (Past/Present/Future) | Active temporal orientation |

The agent selects among four **temporal-framing policies**:

- **RECALL** — re-engage past events (positive bias gated by `pi_pos`)
- **ENGAGE** — attend to the present (valence-preserving)
- **FUTURATE** — simulate future scenarios (high energy cost, high potential valence gain)
- **REST** — conserve energy (valence drifts to neutral)

Policy selection follows expected free energy (EFE) minimisation with softmax action selection.

### Clinical parameters

| Parameter | Symbol | Mechanism |
|-----------|--------|-----------|
| Positive-belief precision | `pi_pos` | Controls D prior skew and RECALL pull strength |
| Affective granularity | `K` | Resolution of the valence state space |
| Interoceptive precision | `omega_e` | Accuracy of energy observation (A_int) |

### Phenotypes as parameter regimes

| Phenotype | `pi_pos` | `K` | `omega_e` | Description |
|-----------|----------|-----|-----------|-------------|
| Healthy | 5.0 | 8 | 5.0 | Balanced precision, high granularity |
| Depressive | 0.5 | 2 | 5.0 | Low positive-belief precision, coarse affect |
| Manic | 4.0 | 2 | 0.5 | Overconfident, poor energy estimation |

## Affect readout

Two complementary formulations are computed at every timestep:

**Pattisapu et al. (2024)** — Circumplex model:
- Valence: `V = U - EU` (reward prediction error)
- Arousal: `A = H[Q(s|o)]` (posterior entropy)

**Joffily & Coricelli (2013)** — Free-energy dynamics:
- Valence: `-dF/dt` (rate of free energy reduction)
- Anticipation: `-d²F/dt²` (acceleration → hope/fear)

## Figures

The simulation generates six publication figures:

1. **Phenotype comparison** — valence, energy, policy, and VFE traces
2. **Joffily-Coricelli dynamics** — F(t), valence (-dF/dt), anticipation (-d²F/dt²)
3. **Granularity effect** — trajectories, variance, jump-size distribution, action mix
4. **Parameter landscape** — pi_pos × omega_e heatmaps (mean valence, variance, energy, futurate fraction)
5. **Phase portrait** — valence × energy trajectories per phenotype
6. **Circumplex trajectories** — Pattisapu et al. polar emotional space

## Usage

```bash
# Full run (~2 min, T=300)
python run.py

# Quick test (~20s, T=100)
python run.py --quick
```

Figures are saved to `figures/`.

**Requirements:** Python 3.8+, NumPy, Matplotlib.

## References

- Pattisapu, A., Albarracin, M., et al. (2024). Free Energy in a Circumplex Model of Emotion.
- Joffily, M. & Coricelli, G. (2013). Emotional valence and the free-energy principle. *PLoS Computational Biology*, 9(6).
- Hesp, C., Smith, R., Parr, T., Allen, M., Friston, K. J., & Ramstead, M. J. (2021). Deeply felt affect: The emergence of valence in deep active inference. *Neural Computation*, 33(2).
- Barrett, L. F. (2017). The theory of constructed emotion. *Trends in Cognitive Sciences*, 21(12).
