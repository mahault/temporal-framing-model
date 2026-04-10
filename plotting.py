"""
Publication figures for the temporal framing model.

Figure 1 – Phenotype comparison  (valence, energy, policy, VFE)
Figure 2 – Joffily-Coricelli     (F, dF/dt, d²F/dt²)
Figure 3 – Granularity effect    (trajectories, variance, jumps, action mix)
Figure 4 – Parameter landscape   (pi_pos × omega_e heatmaps)
Figure 5 – Phase portrait        (valence × energy trajectories)
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


# ── Figure 1: Phenotype comparison ─────────────────────────
def plot_phenotypes(results, save_path=None):
    names = ['healthy', 'depressive', 'manic']
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True)

    for row, name in enumerate(names):
        h = results[name]
        T = len(h['valence_true'])
        t = np.arange(T)
        c = PCOL[name]

        # valence
        ax = axes[row, 0]
        ax.plot(t, h['valence_true'], alpha=0.3, color='gray', lw=0.8, label='True')
        ax.plot(t, h['valence_belief'], color=c, lw=1.5, label='Believed')
        ax.set_ylim(-0.05, 1.05)
        if row == 0: ax.set_title('Valence'); ax.legend(fontsize=7)
        ax.set_ylabel(name.capitalize())

        # energy
        ax = axes[row, 1]
        ax.plot(t, h['energy_true'], color='gray', alpha=0.5, lw=0.8, label='True')
        ax.plot(t, h['energy_belief'], color=c, lw=1.5, label='Believed')
        ax.set_ylim(-0.05, 1.05)
        if row == 0: ax.set_title('Energy'); ax.legend(fontsize=7)

        # rolling policy probabilities
        ax = axes[row, 2]
        win = 10
        for a in range(N_ACTIONS):
            rolling = np.convolve(h['pi'][:, a], np.ones(win)/win, mode='same')
            ax.plot(t, rolling, color=ACOL[a], lw=1.2,
                    label=ACTION_NAMES[a] if row == 0 else None)
        ax.set_ylim(0, 1)
        if row == 0: ax.set_title('Policy  $\\pi(a)$'); ax.legend(fontsize=7, ncol=2)

        # VFE + J-C valence
        ax = axes[row, 3]
        ax.plot(t, h['vfe'], color=c, alpha=0.4, lw=0.8, label='VFE')
        ax2 = ax.twinx()
        ax2.plot(t, h['valence_jc'], color=c, lw=1.5, ls='--', label='Valence (J-C)')
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
    if save_path: fig.savefig(save_path)
    return fig


# ── Figure 2: Joffily-Coricelli dynamics ───────────────────
def plot_joffily(results, save_path=None):
    names = ['healthy', 'depressive', 'manic']
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)

    for col, name in enumerate(names):
        h = results[name]
        T = len(h['vfe'])
        t = np.arange(T)
        c = PCOL[name]

        # F(t)
        ax = axes[0, col]
        ax.plot(t, h['vfe'], color=c, lw=1.2)
        ax.set_title(name.capitalize())
        if col == 0: ax.set_ylabel('$F(t)$')

        # dF/dt → valence
        ax = axes[1, col]
        v = h['valence_jc']
        ax.fill_between(t, 0, v, where=v >= 0, alpha=.3, color='#2ecc71')
        ax.fill_between(t, 0, v, where=v < 0,  alpha=.3, color='#e74c3c')
        ax.plot(t, v, color=c, lw=1.0)
        ax.axhline(0, color='gray', lw=.5)
        ax.set_ylim(-1.1, 1.1)
        if col == 0: ax.set_ylabel('$-\\dot{F}$  (valence)')

        # d²F/dt² → anticipation
        ax = axes[2, col]
        win = 5
        a_sm = np.convolve(h['anticipation'], np.ones(win)/win, mode='same')
        ax.fill_between(t, 0, a_sm, where=a_sm >= 0, alpha=.2, color='#f39c12', label='Hope')
        ax.fill_between(t, 0, a_sm, where=a_sm < 0,  alpha=.2, color='#8e44ad', label='Fear')
        ax.plot(t, a_sm, color=c, lw=1.0)
        ax.axhline(0, color='gray', lw=.5)
        if col == 0: ax.set_ylabel('$-\\ddot{F}$  (anticipation)'); ax.legend(fontsize=8)
        ax.set_xlabel('Timestep')

    fig.suptitle('Free-Energy Dynamics: Valence and Anticipation',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path: fig.savefig(save_path)
    return fig


# ── Figure 3: Granularity sweep ────────────────────────────
def plot_granularity(results, save_path=None):
    Ks = sorted(results.keys())
    cols = plt.cm.viridis(np.linspace(0.2, 0.8, len(Ks)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) trajectories
    ax = axes[0, 0]
    for i, K in enumerate(Ks):
        ax.plot(results[K][0]['valence_belief'], color=cols[i], lw=1., label=f'K={K}')
    ax.set_xlabel('Timestep'); ax.set_ylabel('Believed Valence')
    ax.set_title('(a) Valence Trajectories'); ax.legend()

    # (b) variance box
    ax = axes[0, 1]
    vv = [[np.var(r['valence_belief']) for r in results[K]] for K in Ks]
    ax.boxplot(vv, labels=[str(K) for K in Ks])
    ax.set_xlabel('Granularity K'); ax.set_ylabel('Valence Variance')
    ax.set_title('(b) Oscillation Amplitude')

    # (c) jump-size distribution
    ax = axes[1, 0]
    for i, K in enumerate(Ks):
        jumps = np.concatenate([np.abs(np.diff(r['valence_belief'])) for r in results[K]])
        ax.hist(jumps, bins=30, alpha=.45, color=cols[i], label=f'K={K}', density=True)
    ax.set_xlabel('$|\\Delta v|$'); ax.set_ylabel('Density')
    ax.set_title('(c) Jump-Size Distribution'); ax.legend()

    # (d) action mix
    ax = axes[1, 1]
    x = np.arange(len(Ks)); w = 0.18
    for a in range(N_ACTIONS):
        means = [np.mean([np.mean(r['action'] == a) for r in results[K]]) for K in Ks]
        ax.bar(x + a * w, means, w, label=ACTION_NAMES[a], color=ACOL[a], alpha=.85)
    ax.set_xticks(x + 1.5 * w); ax.set_xticklabels([str(K) for K in Ks])
    ax.set_xlabel('Granularity K'); ax.set_ylabel('Fraction')
    ax.set_title('(d) Action Selection'); ax.legend(fontsize=8)

    fig.suptitle('Effect of Affective Granularity on Oscillation Dynamics',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path: fig.savefig(save_path)
    return fig


# ── Figure 4: Parameter landscape ──────────────────────────
def plot_parameter_space(sw, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    pi, om = sw['pi_values'], sw['omega_values']
    ext = [om[0], om[-1], pi[0], pi[-1]]

    panels = [
        ('mean_valence',     'Mean Believed Valence',  'RdYlGn'),
        ('valence_variance', 'Valence Variance',       'hot_r'),
        ('mean_energy',      'Mean True Energy',       'YlOrRd'),
        ('future_fraction',  'FUTURATE Fraction',      'Reds'),
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
    if save_path: fig.savefig(save_path)
    return fig


# ── Figure 5: Phase portrait ──────────────────────────────
def plot_phase_portrait(results, save_path=None):
    names = ['healthy', 'depressive', 'manic']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, name in enumerate(names):
        ax = axes[i]
        h = results[name]
        v, e = h['valence_belief'], h['energy_true']
        sc = ax.scatter(v, e, c=np.arange(len(v)), cmap='viridis', s=8, alpha=.6)
        ax.plot(v, e, color=PCOL[name], alpha=.15, lw=.5)
        ax.plot(v[0], e[0], 'go', ms=10, label='Start')
        ax.plot(v[-1], e[-1], 'rs', ms=10, label='End')
        ax.set_xlabel('Believed Valence'); ax.set_ylabel('True Energy')
        ax.set_title(name.capitalize()); ax.set_xlim(-.05, 1.05); ax.set_ylim(-.05, 1.05)
        ax.legend(fontsize=8)
        if i == 2: plt.colorbar(sc, ax=ax, label='Timestep')

    fig.suptitle('Phase Portraits: Valence $\\times$ Energy', fontsize=13, y=1.05)
    plt.tight_layout()
    if save_path: fig.savefig(save_path)
    return fig


# ── Figure 6: Circumplex (Pattisapu et al.) ────────────────
_CIRC_LABELS = [
    (  0, 'Happy'),   ( 45, 'Excited'), ( 90, 'Alert'),
    (135, 'Angry'),   (180, 'Sad'),     (225, 'Depressed'),
    (270, 'Calm'),    (315, 'Relaxed'),
]

def plot_circumplex(results, save_path=None):
    """
    Figure 6: Circumplex trajectories per phenotype.

    V = U − EU  (horizontal: positive = happy, negative = sad).
    A = 2·H_norm − 1  (vertical: +1 = alert/uncertain, −1 = calm/certain).
    Polar transform: θ = atan2(A, V),  r = √(V² + A²)  = intensity.
    """
    names = ['healthy', 'depressive', 'manic']
    fig, axes = plt.subplots(1, 3, figsize=(17, 6),
                             subplot_kw=dict(projection='polar'))

    for idx, name in enumerate(names):
        ax = axes[idx]
        h = results[name]
        V = h['valence_p'].copy()
        A_raw = h['arousal_norm'].copy()

        # Centre arousal:  0 → −1 (calm/certain),  1 → +1 (alert/uncertain)
        A = 2.0 * A_raw - 1.0

        # Scale valence to roughly ±1 for visual balance
        v_scale = max(np.abs(V).max(), 0.01)
        V_s = V / v_scale

        theta = np.arctan2(A, V_s)
        r = np.sqrt(V_s**2 + A**2)

        sc = ax.scatter(theta, r, c=np.arange(len(V)), cmap='viridis',
                        s=18, alpha=0.7, zorder=5)
        ax.plot(theta, r, color=PCOL[name], alpha=0.15, lw=0.5)

        # Start / end markers
        ax.scatter(theta[0], r[0], marker='o', s=80, c='green',
                   edgecolors='k', zorder=10, label='Start')
        ax.scatter(theta[-1], r[-1], marker='s', s=80, c='red',
                   edgecolors='k', zorder=10, label='End')

        # Emotion labels around the perimeter
        rmax = max(r.max() * 1.25, 0.3)
        ax.set_rlim(0, rmax)
        for deg, label in _CIRC_LABELS:
            rad = np.deg2rad(deg)
            ax.text(rad, rmax * 0.92, label, ha='center', va='center',
                    fontsize=8, fontweight='bold', color='#444')

        # Light reference circles
        for frac in [0.33, 0.66]:
            circle_r = rmax * frac
            th = np.linspace(0, 2 * np.pi, 100)
            ax.plot(th, np.full_like(th, circle_r),
                    color='gray', lw=0.3, alpha=0.4)

        ax.set_title(f'{name.capitalize()}', pad=22, fontsize=12)
        ax.set_rticks([])
        ax.set_thetagrids([])
        ax.legend(fontsize=7, loc='lower right',
                  bbox_to_anchor=(1.15, -0.05))

    plt.colorbar(sc, ax=axes[-1], label='Timestep', shrink=0.65, pad=0.15)
    fig.suptitle('Circumplex Emotional Trajectories  (Pattisapu et al. 2024)',
                 fontsize=13, y=1.0)
    plt.tight_layout()
    if save_path: fig.savefig(save_path)
    return fig
