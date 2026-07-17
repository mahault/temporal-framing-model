"""
Calibrate M5 mood layer to the new model VFE levels.

Runs short trials at different pi_pos values and measures mean VFE.
This gives us the function hat_F(pi_pos) needed for A_mood.
"""

import numpy as np
from experiments import run_trial

PI_POS_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5]
T = 200
SEED = 42

print("VFE calibration across pi_pos (new model)")
print("=" * 60)
print(f"{'pi_pos':>8s}  {'mean_VFE':>10s}  {'std_VFE':>10s}  {'alpha':>8s}")
print("-" * 60)

results = []
for pp in PI_POS_VALUES:
    # Run 3 seeds and average
    vfes = []
    for s in range(3):
        h = run_trial(K=8, M=8, pi_pos=pp, omega_e=5.0, gamma=16.0,
                      c_scale=1.0, T=T, seed=SEED + s, volatility=0.3)
        vfes.append(np.mean(h['vfe']))
    mean_vfe = np.mean(vfes)
    std_vfe = np.std(vfes)
    alpha = 1.0 / (1.0 + np.exp(-(pp - 2.0)))
    results.append((pp, mean_vfe, std_vfe, alpha))
    print(f"{pp:8.1f}  {mean_vfe:10.4f}  {std_vfe:10.4f}  {alpha:8.4f}")

print("\n" + "=" * 60)
print("Fit: VFE = a + b * (1 - alpha)")
print("=" * 60)

# Fit linear model: VFE = a + b * (1 - alpha)
alphas = np.array([r[3] for r in results])
vfes = np.array([r[1] for r in results])
one_minus_alpha = 1.0 - alphas

# Least squares
A_mat = np.column_stack([np.ones_like(one_minus_alpha), one_minus_alpha])
coeffs, residuals, _, _ = np.linalg.lstsq(A_mat, vfes, rcond=None)
a, b = coeffs

print(f"  VFE = {a:.4f} + {b:.4f} * (1 - alpha)")
print(f"  At pi_pos=0.5 (alpha=0.18): VFE = {a + b*(1-0.18):.4f}")
print(f"  At pi_pos=5.0 (alpha=0.95): VFE = {a + b*(1-0.95):.4f}")
print(f"  At pi_pos=7.5 (alpha=1.00): VFE = {a + b*(1-1.00):.4f}")

# Check VFE range for observation binning
vfe_low = a + b * (1 - 1.0)   # high pi_pos
vfe_high = a + b * (1 - 0.18)  # low pi_pos
print(f"\n  VFE range to discriminate: [{vfe_low:.4f}, {vfe_high:.4f}]")
print(f"  Range width: {vfe_high - vfe_low:.4f}")

# Suggest observation bin edges
mid = (vfe_low + vfe_high) / 2
step = (vfe_high - vfe_low) / 3
print(f"\n  Suggested obs_centers for A_mood (5 bins):")
print(f"    Far below range:  {vfe_low - 2.0:.1f}")
print(f"    Low VFE:          {vfe_low + step*0.5:.2f}")
print(f"    Mid VFE:          {mid:.2f}")
print(f"    High VFE:         {vfe_high - step*0.5:.2f}")
print(f"    Far above range:  {vfe_high + 2.0:.1f}")

print(f"\n  Suggested sigma_A: {(vfe_high - vfe_low) / 6:.3f}")
print(f"  (should be ~1/6 of range to discriminate 3 levels within range)")

# Also check stressed environment
print("\n" + "=" * 60)
print("VFE under stress (volatility=0.9) across pi_pos")
print("=" * 60)
for pp in [1.0, 3.0, 5.0, 7.0]:
    vfes = []
    for s in range(3):
        h = run_trial(K=8, M=8, pi_pos=pp, omega_e=5.0, gamma=16.0,
                      c_scale=1.0, T=T, seed=SEED + s, volatility=0.9)
        vfes.append(np.mean(h['vfe']))
    mean_vfe = np.mean(vfes)
    alpha = 1.0 / (1.0 + np.exp(-(pp - 2.0)))
    print(f"  pi_pos={pp:.1f}  alpha={alpha:.3f}  VFE={mean_vfe:.4f}")
