"""
Comprehensive emotion validation in 3D PAD space:
  X = Valence (pleasure)
  Y = Dominance (policy precision)
  Z = Arousal (state entropy or EFE range)

Tests whether parameter configurations produce agents in distinct
regions of the full Pleasure-Arousal-Dominance space.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from experiments import run_trial
from plotting import _ema

# ── Emotion target profiles ──────────────────────────────────
EMOTION_PROFILES = {
    # (+V, +D, low A): positive, decisive, calm
    'happy': dict(
        K=8, M=8, pi_pos=5.0, omega_e=5.0, gamma=16.0, c_scale=1.5,
        volatility=0.3,
    ),
    # (+V, moderate D, low A): positive, relaxed
    'content': dict(
        K=8, M=8, pi_pos=5.0, omega_e=5.0, gamma=16.0, c_scale=0.8,
        volatility=0.3,
    ),
    # (near 0 V, moderate D, low A): neutral, calm
    'calm': dict(
        K=8, M=8, pi_pos=2.0, omega_e=3.0, gamma=16.0, c_scale=0.7,
        volatility=0.3,
    ),
    # (+V, -D, high A): positive, scattered, activated — elated
    # High c_scale → high arousal; LOW gamma → flat policy (low D)
    # High pi_pos → positive valence
    'excited': dict(
        K=4, M=8, pi_pos=5.0, omega_e=0.5, gamma=4.0, c_scale=2.5,
        volatility=0.45,
    ),
    # (near 0 V, +D, high A): vigilant, uncertain environment
    'alert': dict(
        K=4, M=8, pi_pos=2.5, omega_e=2.0, gamma=16.0, c_scale=2.0,
        volatility=0.6,
    ),
    # (-V, +D, high A): negative, decisive, activated — frustrated
    # Very high c_scale + high gamma → high arousal + sharp policy (high D)
    # Low pi_pos + high volatility → negative valence
    'angry': dict(
        K=8, M=8, pi_pos=0.1, omega_e=0.2, gamma=16.0, c_scale=5.0,
        volatility=0.8,
    ),
    # (-V, -D, high A): negative, helpless, activated — threat
    # High c_scale → high mean G (high arousal, agent cares about outcomes)
    # LOW gamma → can't decide what to do (low dominance, indecisive under threat)
    'fearful': dict(
        K=4, M=8, pi_pos=0.3, omega_e=0.3, gamma=4.0, c_scale=2.5,
        volatility=0.8,
    ),
    # (-V, moderate-low D, low A): negative, low energy, passive
    'sad': dict(
        K=4, M=8, pi_pos=0.1, omega_e=3.0, gamma=16.0, c_scale=0.25,
        volatility=0.6,
    ),
    # (-V, -D, low A): negative, helpless, deactivated
    'depressed': dict(
        K=4, M=8, pi_pos=0.2, omega_e=5.0, gamma=16.0, c_scale=0.1,
        volatility=0.45,
    ),
    # (near 0 V, -D, low A): flat, unmotivated
    'bored': dict(
        K=4, M=8, pi_pos=1.5, omega_e=5.0, gamma=16.0, c_scale=0.15,
        volatility=0.3,
    ),
}

# Colour palette
ECOL = {
    'happy':     '#f1c40f',
    'content':   '#2ecc71',
    'calm':      '#1abc9c',
    'excited':   '#e67e22',
    'alert':     '#e74c3c',
    'angry':     '#c0392b',
    'fearful':   '#8e44ad',
    'sad':       '#3498db',
    'depressed': '#2c3e50',
    'bored':     '#95a5a6',
}


def compute_pad(h):
    """Compute smoothed Pleasure, Arousal, Dominance from history."""
    # Pleasure = composite valence
    V = _ema(h['valence'], alpha=0.1)

    # Dominance = policy precision, centred [-1, 1]
    D_raw = 1.0 - h['policy_entropy_norm']
    D = 2.0 * D_raw - 1.0
    D = _ema(D, alpha=0.1)

    # Arousal = mean EFE across policies (expected threat level)
    # High mean G = all policies lead to bad outcomes = threatening = high arousal
    # Low mean G  = agent expects manageable outcomes = calm = low arousal
    # This is principled: G = E[-ln P(o|C)] + ambiguity, so mean G captures
    # how far the agent expects observations to deviate from preferences.
    # High c_scale amplifies G → high arousal (angry/fearful),
    # Low c_scale compresses G → low arousal (depressed).
    G = h['G']                              # (T, N_ACTIONS)
    mean_G = G.mean(axis=1)                 # mean across policies
    A = np.tanh(mean_G / 8.0)              # normalise [0, ~1]
    A = 2.0 * A - 1.0                       # centre to [-1, 1]
    A = _ema(A, alpha=0.1)

    return V, A, D


def run_and_plot(save_path=None):
    # ── Collect data ──────────────────────────────────────
    results = {}
    print(f"{'Profile':<12} {'mean_V':>8} {'mean_A':>8} {'mean_D':>8}")
    print("-" * 42)

    for name, params in EMOTION_PROFILES.items():
        h = run_trial(**params, T=300, seed=42)
        V, A, D = compute_pad(h)
        results[name] = (V, A, D)
        # Also show raw mean G for calibration
        raw_mG = np.mean(h['G'].mean(axis=1))
        print(f"{name:<12} {np.mean(V):>8.3f} {np.mean(A):>8.3f} {np.mean(D):>8.3f}  raw_mG={raw_mG:.2f}")

    # ── 3D PAD plot ───────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    for name, (V, A, D) in results.items():
        col = ECOL[name]
        mv, ma, md = np.mean(V), np.mean(A), np.mean(D)

        # Trajectory (faded)
        ax.plot(V, A, D, color=col, alpha=0.1, lw=0.3)

        # Mean marker (bold)
        ax.scatter(mv, ma, md, color=col, s=200, edgecolors='k',
                   linewidths=1.5, zorder=10, label=name.capitalize(),
                   depthshade=False)

    ax.set_xlabel('Valence (Pleasure)', fontsize=11, labelpad=10)
    ax.set_ylabel('Arousal (mean EFE)', fontsize=11, labelpad=10)
    ax.set_zlabel('Dominance (policy precision)', fontsize=11, labelpad=10)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)

    ax.set_title('3D PAD Emotion Validation\n(Mehrabian Pleasure-Arousal-Dominance)',
                 fontsize=13, pad=20)
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=9)

    # Set viewing angle for best separation
    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved to {save_path}")
    plt.close(fig)

    # ── Also save 2D projections ──────────────────────────
    fig2, axes = plt.subplots(1, 3, figsize=(18, 6))
    projections = [
        ('Valence', 'Dominance', lambda v, a, d: (v, d)),
        ('Valence', 'Arousal', lambda v, a, d: (v, a)),
        ('Arousal', 'Dominance', lambda v, a, d: (a, d)),
    ]

    for idx, (xlabel, ylabel, proj_fn) in enumerate(projections):
        ax = axes[idx]
        for name, (V, A, D) in results.items():
            col = ECOL[name]
            x, y = proj_fn(V, A, D)
            mx, my = np.mean(x), np.mean(y)
            ax.scatter(x, y, color=col, s=8, alpha=0.2, zorder=3)
            ax.scatter(mx, my, color=col, s=150, edgecolors='k',
                       linewidths=1.2, zorder=10,
                       label=name.capitalize() if idx == 2 else None)

        ax.axhline(0, color='gray', lw=0.5, ls=':')
        ax.axvline(0, color='gray', lw=0.5, ls=':')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{xlabel} vs {ylabel}', fontsize=12)

    axes[2].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig2.suptitle('PAD Projections: Emotion Validation', fontsize=13, y=1.02)
    plt.tight_layout()
    proj_path = save_path.replace('.png', '_projections.png') if save_path else None
    if proj_path:
        fig2.savefig(proj_path, dpi=300, bbox_inches='tight')
        print(f"Saved projections to {proj_path}")
    plt.close(fig2)


if __name__ == '__main__':
    run_and_plot(save_path='figures/fig7_emotion_validation.png')
