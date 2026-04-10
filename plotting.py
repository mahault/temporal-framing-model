"""
Publication figures for the temporal framing model.

Figure 1 – Phenotype comparison  (valence, energy, policy, VFE)
Figure 2 – Joffily-Coricelli     (F, dF/dt, d²F/dt²)
Figure 3 – Granularity effect    (trajectories, variance, jumps, action mix)
Figure 4 – Parameter landscape   (pi_pos × omega_e heatmaps)
Figure 5 – Phase portrait        (valence × energy trajectories)
Figure 6 – Circumplex            (Pattisapu et al. polar trajectories)
"""

import numpy as np
import matplotlib.pyplot as plt
from generative_model import ACTION_NAMES, N_ACTIONS, FUTURATE

# ── Style ──────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

PCOL = {'healthy': '#2ecc71', 'depressive': '#3498db', 'manic': '#e74c3c'}
ACOL = ['#8e44ad', '#2ecc71', '#e74c3c', '#95a5a6']   # RECALL ENGAGE FUTURATE REST


# ── Smoothing helper ───────────────────────────────────────
def _ema(x, alpha=0.08):
    """Exponential moving average for display smoothing."""
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


# ── Figure 1: Phenotype comparison ─────────────────────────
def plot_phenotypes(results, save_path=None):
    names = ['healthy', 'depressive', 'manic']
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True)

    for row, name in enumerate(names):
        h = results[name]
        T = len(h['valence_true'])
        t = np.arange(T)
        c = PCOL[name]

        # valence (smoothed)
        ax = axes[row, 0]
        ax.plot(t, _ema(h['valence_true'], 0.15), alpha=0.4, color='gray',
                lw=0.8, label='True')
        ax.plot(t, _ema(h['valence_belief'], 0.15), color=c, lw=1.5,
                label='Believed')
        ax.set_ylim(-0.05, 1.05)
        if row == 0:
            ax.set_title('Valence')
            ax.legend(fontsize=7)
        ax.set_ylabel(name.capitalize())

        # energy (smoothed)
        ax = axes[row, 1]
        ax.plot(t, _ema(h['energy_true'], 0.15), color='gray', alpha=0.5,
                lw=0.8, label='True')
        ax.plot(t, _ema(h['energy_belief'], 0.15), color=c, lw=1.5,
                label='Believed')
        ax.set_ylim(-0.05, 1.05)
        if row == 0:
            ax.set_title('Energy')
            ax.legend(fontsize=7)

        # rolling policy probabilities
        ax = axes[row, 2]
        win = 15
        for a in range(N_ACTIONS):
            rolling = np.convolve(h['pi'][:, a], np.ones(win) / win,
                                  mode='same')
            ax.plot(t, rolling, color=ACOL[a], lw=1.2,
                    label=ACTION_NAMES[a] if row == 0 else None)
        ax.set_ylim(0, 1)
        if row == 0:
            ax.set_title('Policy  $\\pi(a)$')
            ax.legend(fontsize=7, ncol=2)

        # VFE + J-C valence (both smoothed)
        ax = axes[row, 3]
        ax.plot(t, _ema(h['vfe'], 0.05), color=c, alpha=0.5, lw=0.8,
                label='VFE')
        ax2 = ax.twinx()
        ax2.plot(t, _ema(h['valence_jc'], 0.05), color=c, lw=1.5,
                 ls='--', label='Valence (J-C)')
        ax2.set_ylim(-1.1, 1.1)
        ax2.axhline(0, color='gray', lw=0.5, ls=':')
        if row == 0:
            ax.set_title('VFE  &  Valence')
            ax.legend(loc='upper left', fontsize=7)
            ax2.legend(loc='upper right', fontsize=7)

    for j in range(4):
        axes[-1, j].set_xlabel('Timestep')
    fig.suptitle('Counterfactual Temporal Framing: Clinical Phenotypes',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ── Figure 2: Joffily-Coricelli dynamics ───────────────────
def plot_joffily(results, save_path=None):
    names = ['healthy', 'depressive', 'manic']
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)

    # Compute shared y-limits per row
    ylims = [None, None, None]
    for name in names:
        h = results[name]
        vfe_sm = _ema(h['vfe'], 0.05)
        v_sm = _ema(h['valence_jc'], 0.05)
        a_sm = _ema(h['anticipation'], 0.05)
        yl0 = (vfe_sm.min(), vfe_sm.max())
        yl1 = (v_sm.min(), v_sm.max())
        yl2 = (a_sm.min(), a_sm.max())
        if ylims[0] is None:
            ylims[0] = yl0
            ylims[1] = yl1
            ylims[2] = yl2
        else:
            ylims[0] = (min(ylims[0][0], yl0[0]), max(ylims[0][1], yl0[1]))
            ylims[1] = (min(ylims[1][0], yl1[0]), max(ylims[1][1], yl1[1]))
            ylims[2] = (min(ylims[2][0], yl2[0]), max(ylims[2][1], yl2[1]))

    # Add padding
    for i in range(3):
        span = ylims[i][1] - ylims[i][0]
        ylims[i] = (ylims[i][0] - 0.05 * span, ylims[i][1] + 0.05 * span)

    for col, name in enumerate(names):
        h = results[name]
        T = len(h['vfe'])
        t = np.arange(T)
        c = PCOL[name]

        # F(t) — smoothed
        ax = axes[0, col]
        ax.plot(t, _ema(h['vfe'], 0.05), color=c, lw=1.2)
        ax.set_title(name.capitalize())
        ax.set_ylim(ylims[0])
        if col == 0:
            ax.set_ylabel('$F(t)$')

        # dF/dt → valence — smoothed
        ax = axes[1, col]
        v = _ema(h['valence_jc'], 0.05)
        ax.fill_between(t, 0, v, where=v >= 0, alpha=.3, color='#2ecc71')
        ax.fill_between(t, 0, v, where=v < 0, alpha=.3, color='#e74c3c')
        ax.plot(t, v, color=c, lw=1.0)
        ax.axhline(0, color='gray', lw=.5)
        ax.set_ylim(ylims[1])
        if col == 0:
            ax.set_ylabel('$-\\dot{F}$  (valence)')

        # d²F/dt² → anticipation — smoothed
        ax = axes[2, col]
        a_sm = _ema(h['anticipation'], 0.05)
        ax.fill_between(t, 0, a_sm, where=a_sm >= 0, alpha=.2,
                        color='#f39c12', label='Hope' if col == 0 else None)
        ax.fill_between(t, 0, a_sm, where=a_sm < 0, alpha=.2,
                        color='#8e44ad', label='Fear' if col == 0 else None)
        ax.plot(t, a_sm, color=c, lw=1.0)
        ax.axhline(0, color='gray', lw=.5)
        ax.set_ylim(ylims[2])
        if col == 0:
            ax.set_ylabel('$-\\ddot{F}$  (anticipation)')
            ax.legend(fontsize=8)
        ax.set_xlabel('Timestep')

    fig.suptitle('Free-Energy Dynamics: Valence and Anticipation',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ── Figure 3: Granularity sweep ────────────────────────────
def plot_granularity(results, save_path=None):
    Ks = sorted(results.keys())
    cols = plt.cm.viridis(np.linspace(0.2, 0.8, len(Ks)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) trajectories — smoothed
    ax = axes[0, 0]
    for i, K in enumerate(Ks):
        ax.plot(_ema(results[K][0]['valence_belief'], 0.1),
                color=cols[i], lw=1.2, label=f'K={K}')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Believed Valence')
    ax.set_title('(a) Valence Trajectories')
    ax.legend()

    # (b) variance box
    ax = axes[0, 1]
    vv = [[np.var(r['valence_belief']) for r in results[K]] for K in Ks]
    ax.boxplot(vv, labels=[str(K) for K in Ks])
    ax.set_xlabel('Granularity K')
    ax.set_ylabel('Valence Variance')
    ax.set_title('(b) Oscillation Amplitude')

    # (c) jump-size distribution
    ax = axes[1, 0]
    for i, K in enumerate(Ks):
        jumps = np.concatenate([np.abs(np.diff(r['valence_belief']))
                                for r in results[K]])
        ax.hist(jumps, bins=30, alpha=.45, color=cols[i],
                label=f'K={K}', density=True)
    ax.set_xlabel('$|\\Delta v|$')
    ax.set_ylabel('Density')
    ax.set_title('(c) Jump-Size Distribution')
    ax.legend()

    # (d) action mix
    ax = axes[1, 1]
    x = np.arange(len(Ks))
    w = 0.18
    for a in range(N_ACTIONS):
        means = [np.mean([np.mean(r['action'] == a) for r in results[K]])
                 for K in Ks]
        ax.bar(x + a * w, means, w, label=ACTION_NAMES[a],
               color=ACOL[a], alpha=.85)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels([str(K) for K in Ks])
    ax.set_xlabel('Granularity K')
    ax.set_ylabel('Fraction')
    ax.set_title('(d) Action Selection')
    ax.legend(fontsize=8)

    fig.suptitle('Effect of Affective Granularity on Oscillation Dynamics',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ── Figure 4: Parameter landscape ──────────────────────────
def plot_parameter_space(sw, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    pi, om = sw['pi_values'], sw['omega_values']
    ext = [om[0], om[-1], pi[0], pi[-1]]

    panels = [
        ('mean_valence',     'Mean Believed Valence',  'RdYlGn'),
        ('valence_variance', 'Valence Variance',       'hot_r'),
        ('mean_energy',      'Mean Believed Energy',   'YlOrRd'),
        ('action_entropy',   'Action Entropy',         'viridis'),
    ]
    markers = {'H': (5.0, 5.0), 'D': (5.0, 0.5), 'M': (0.5, 4.0)}

    for idx, (key, title, cmap) in enumerate(panels):
        ax = axes[idx // 2, idx % 2]
        im = ax.imshow(sw[key], origin='lower', aspect='auto',
                       extent=ext, cmap=cmap, interpolation='bilinear')
        ax.set_xlabel('$\\omega_e$ (energy precision)')
        ax.set_ylabel('$\\pi_{pos}$ (positive-belief precision)')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        for lab, (oe, pp) in markers.items():
            ax.plot(oe, pp, 'k*', ms=12)
            ax.annotate(lab, (oe, pp), textcoords='offset points',
                        xytext=(5, 5), fontsize=9, fontweight='bold')

    fig.suptitle('Clinical Parameter Landscape', fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ── Figure 5: Phase portrait ──────────────────────────────
def plot_phase_portrait(results, save_path=None):
    names = ['healthy', 'depressive', 'manic']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    for i, name in enumerate(names):
        ax = axes[i]
        h = results[name]
        v = _ema(h['valence_belief'], 0.12)
        e = _ema(h['energy_true'], 0.12)
        T = len(v)

        sc = ax.scatter(v, e, c=np.arange(T), cmap='viridis',
                        s=12, alpha=.6, zorder=3)
        ax.plot(v, e, color=PCOL[name], alpha=.2, lw=.6)
        ax.plot(v[0], e[0], 'go', ms=10, label='Start', zorder=5)
        ax.plot(v[-1], e[-1], 'rs', ms=10, label='End', zorder=5)
        ax.set_xlabel('Believed Valence')
        ax.set_ylabel('True Energy')
        ax.set_title(name.capitalize())
        ax.set_xlim(-.05, 1.05)
        ax.set_ylim(-.05, 1.05)
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        if i == 2:
            plt.colorbar(sc, ax=ax, label='Timestep', shrink=0.8)

    fig.suptitle('Phase Portraits: Valence $\\times$ Energy',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ── Figure 6: Valence × Dominance affect space (PAD model) ──
def plot_circumplex(results, save_path=None):
    """
    Figure 6: Valence × Dominance trajectories per phenotype.

    Following Mehrabian's PAD model and the precision-as-dominance
    framework (empathy.md §11.2):

    Valence (x): EMA-smoothed three-channel composite
      V = EMA(tanh(v_model + v_reward + v_action))

    Dominance (y): Policy precision = 1 − H[q(π)] / H_max
      High dominance = sharp policy = decisive / in control
      Low dominance  = flat policy  = indecisive / helpless
      Centred to [-1, 1]: D = 2·(1 − H_norm) − 1

    Quadrants:
      (+V, +D)  Happy / Content     — positive, decisive
      (−V, +D)  Angry / Frustrated  — negative, decisive
      (+V, −D)  Excited / Manic     — positive, scattered
      (−V, −D)  Depressed / Fearful — negative, helpless
    """
    names = ['healthy', 'depressive', 'manic']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for idx, name in enumerate(names):
        ax = axes[idx]
        h = results[name]

        # Valence: EMA-smoothed composite
        V = _ema(h['valence'], alpha=0.1)

        # Dominance: policy precision, centred to [-1, 1]
        D_raw = 1.0 - h['policy_entropy_norm']   # [0, 1]
        D = 2.0 * D_raw - 1.0                     # [-1, 1]
        D = _ema(D, alpha=0.1)

        T = len(V)
        sc = ax.scatter(V, D, c=np.arange(T), cmap='viridis',
                        s=18, alpha=0.7, zorder=5)
        ax.plot(V, D, color=PCOL[name], alpha=0.15, lw=0.5)

        # Start / end markers
        ax.scatter(V[0], D[0], marker='o', s=80, c='green',
                   edgecolors='k', zorder=10, label='Start')
        ax.scatter(V[-1], D[-1], marker='s', s=80, c='red',
                   edgecolors='k', zorder=10, label='End')

        # Cross-hairs
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        ax.axvline(0, color='gray', lw=0.5, ls=':')

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.set_xlabel('Valence')
        if idx == 0:
            ax.set_ylabel('Dominance  (policy precision)')
        ax.set_title(f'{name.capitalize()}', fontsize=12)
        if idx == 0:
            ax.legend(fontsize=7, loc='lower left')

        # Quadrant emotion labels (inset from corners to avoid legend)
        ax.text( 0.65,  0.95, 'Happy',      ha='center', fontsize=8,
                 color='#27ae60', fontstyle='italic')
        ax.text(-0.65,  0.95, 'Angry',       ha='center', fontsize=8,
                 color='#c0392b', fontstyle='italic')
        ax.text( 0.65, -0.95, 'Excited',     ha='center', fontsize=8,
                 color='#f39c12', fontstyle='italic')
        ax.text(-0.65, -0.95, 'Depressed',   ha='center', fontsize=8,
                 color='#2980b9', fontstyle='italic')

    plt.colorbar(sc, ax=axes[-1], label='Timestep', shrink=0.65, pad=0.1)
    fig.suptitle('Affect Space: Valence $\\times$ Dominance  (Three-Channel Model)',
                 fontsize=13, y=1.0)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ── Figure 7: PAD emotion validation ─────────────────────
_ECOL = {
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


def _compute_pad(h):
    """Compute smoothed Pleasure, Arousal, Dominance from trial history."""
    V = _ema(h['valence'], alpha=0.1)

    D_raw = 1.0 - h['policy_entropy_norm']
    D = 2.0 * D_raw - 1.0
    D = _ema(D, alpha=0.1)

    G = h['G']
    mean_G = G.mean(axis=1)
    A = np.tanh(mean_G / 8.0)
    A = 2.0 * A - 1.0
    A = _ema(A, alpha=0.1)

    return V, A, D


def plot_emotion_validation(results, save_path=None):
    """
    Figure 7: 3D PAD emotion validation.

    Tests whether targeted parameter configurations produce agents
    in distinct regions of the full Pleasure-Arousal-Dominance space.

    Arousal = mean EFE across policies (expected threat level).
    Dominance = policy precision (1 - normalised policy entropy).
    Valence = EMA-smoothed three-channel composite.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Collect PAD coordinates
    pad = {}
    for name, h in results.items():
        pad[name] = _compute_pad(h)

    # ── 3D plot ───────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    for name, (V, A, D) in pad.items():
        col = _ECOL.get(name, '#999')
        mv, ma, md = np.mean(V), np.mean(A), np.mean(D)
        ax.plot(V, A, D, color=col, alpha=0.1, lw=0.3)
        ax.scatter(mv, ma, md, color=col, s=200, edgecolors='k',
                   linewidths=1.5, zorder=10, label=name.capitalize(),
                   depthshade=False)

    ax.set_xlabel('Valence (Pleasure)', fontsize=11, labelpad=10)
    ax.set_ylabel('Arousal (mean EFE)', fontsize=11, labelpad=10)
    ax.set_zlabel('Dominance (policy precision)', fontsize=11, labelpad=10)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    ax.set_title('3D PAD Emotion Validation', fontsize=13, pad=20)
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=9)
    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ── 2D projections ────────────────────────────────────
    fig2, axes = plt.subplots(1, 3, figsize=(18, 6))
    projections = [
        ('Valence', 'Dominance', lambda v, a, d: (v, d)),
        ('Valence', 'Arousal',   lambda v, a, d: (v, a)),
        ('Arousal', 'Dominance', lambda v, a, d: (a, d)),
    ]

    for idx, (xlabel, ylabel, proj_fn) in enumerate(projections):
        ax = axes[idx]
        for name, (V, A, D) in pad.items():
            col = _ECOL.get(name, '#999')
            x, y = proj_fn(V, A, D)
            ax.scatter(x, y, color=col, s=8, alpha=0.2, zorder=3)
            ax.scatter(np.mean(x), np.mean(y), color=col, s=150,
                       edgecolors='k', linewidths=1.2, zorder=10,
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
    plt.close(fig2)

    return fig
