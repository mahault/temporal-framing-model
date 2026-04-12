"""
Publication figures for the temporal framing model.

Figure 1  – Phenotype comparison  (valence, energy, policy, VFE)
Figure 2  – Joffily-Coricelli     (F, dF/dt, d²F/dt²)
Figure 3  – Granularity effect    (trajectories, variance, jumps, action mix,
                                    FUTURATE effectiveness, elaboration/nuance)
Figure 4  – Parameter landscape   (pi_pos × omega_e heatmaps)
Figure 5  – Phase portrait        (valence × energy trajectories)
Figure 6  – Circumplex            (Pattisapu et al. polar trajectories)
Figure 8  – Temporal aiming       (three-channel temporal decomposition)
Figure 10 – Feedback reliance     (healthy vs recall-impaired)
Figure 11 – Framing dynamics      (frame beliefs, VFE|action, run lengths,
                                    transition matrices, valence stability,
                                    VFE–FUTURATE correlation)
Figure 12 – Chronic stress        (stressed vs healthy: frame beliefs,
                                    backward valence, channel decomposition)
"""

import numpy as np
import matplotlib.pyplot as plt
from generative_model import (ACTION_NAMES, N_ACTIONS, FUTURATE, RECALL,
                              ENGAGE, REST, FRAME_NAMES)

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
    fig, axes = plt.subplots(3, 2, figsize=(12, 13))

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

    # (e) FUTURATE effectiveness per K (Gap 3)
    ax = axes[2, 0]
    fut_abs_dv_means = []
    fut_abs_dv_stds = []
    fut_props = []
    for K in Ks:
        # Collect |Δv| at FUTURATE timesteps across runs
        run_means = []
        fut_count = 0
        total_count = 0
        for r in results[K]:
            actions = r['action']
            vb = r['valence_belief']
            dv = np.diff(vb, prepend=vb[0])
            fut_mask = actions == FUTURATE
            if fut_mask.sum() > 0:
                run_means.append(np.mean(np.abs(dv[fut_mask])))
            fut_count += fut_mask.sum()
            total_count += len(actions)
        fut_abs_dv_means.append(np.mean(run_means) if run_means else 0.0)
        fut_abs_dv_stds.append(np.std(run_means) if len(run_means) > 1 else 0.0)
        fut_props.append(fut_count / max(total_count, 1))

    x = np.arange(len(Ks))
    ax.bar(x, fut_abs_dv_means, 0.5, yerr=fut_abs_dv_stds, capsize=4,
           color=ACOL[2], alpha=0.85, label='$|\\Delta v|$ | FUTURATE')
    ax2e = ax.twinx()
    ax2e.plot(x, fut_props, 'o--', color='#2c3e50', lw=1.5, ms=7,
              label='FUTURATE proportion')
    ax.set_xticks(x)
    ax.set_xticklabels([str(K) for K in Ks])
    ax.set_xlabel('Granularity K')
    ax.set_ylabel('Mean $|\\Delta v_{belief}|$')
    ax2e.set_ylabel('Selection proportion')
    ax2e.set_ylim(0, max(fut_props) * 1.8 if max(fut_props) > 0 else 0.3)
    ax.set_title('(e) FUTURATE Effectiveness')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2e.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

    # (f) Elaboration vs Nuance per K (Gap 4)
    ax = axes[2, 1]
    elaborations = []
    nuances = []
    for K in Ks:
        switch_rates = []
        resolutions = []
        for r in results[K]:
            actions = r['action']
            vb = r['valence_belief']
            # Elaboration: action switching rate
            switches = np.sum(actions[1:] != actions[:-1])
            switch_rates.append(switches / max(len(actions) - 1, 1))
            # Nuance: unique valence states visited, quantised to K bins
            bins = np.linspace(0, 1 + 1e-10, K + 1)
            digitized = np.digitize(vb, bins)
            unique_states = len(np.unique(digitized))
            resolutions.append(unique_states)
        elaborations.append(np.mean(switch_rates))
        nuances.append(np.mean(resolutions))

    x = np.arange(len(Ks))
    ax.bar(x, elaborations, 0.5, color='#3498db', alpha=0.85,
           label='Elaboration (switch rate)')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Action switching rate')
    ax2f = ax.twinx()
    ax2f.plot(x, nuances, 's-', color='#e74c3c', lw=2, ms=8, mfc='white',
              mew=2, label='Nuance (distinct states)')
    ax2f.set_ylabel('Distinct valence states visited')
    ax2f.set_ylim(0, max(Ks) + 1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(K) for K in Ks])
    ax.set_xlabel('Granularity K')
    ax.set_title('(f) Elaboration vs Nuance')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2f.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

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


# ── Figure 8: Temporal aiming ────────────────────────────
_TCOL = {'backward': '#3498db', 'present': '#2ecc71', 'forward': '#e67e22'}


def plot_temporal_aiming(results, save_path=None):
    """
    Figure 8: Temporal aiming of affective channels.

    Shows which temporal direction dominates the agent's affect at
    each timestep, following the three-channel architecture:

        v_model  (backward) -- Joffily & Coricelli 2013: -dF
        v_reward (present)  -- Pattisapu et al. 2024:    U - EU
        v_action (forward)  -- Hesp et al. 2021:         AC

    Row 1: Channel traces overlaid per phenotype
    Row 2: Normalised temporal orientation (stacked area)
    Row 3: Summary statistics -- mean channel magnitudes + disagreement
    """
    names = ['healthy', 'depressive', 'manic']
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    sm_alpha = 0.08

    for col, name in enumerate(names):
        h = results[name]
        T = len(h['v_model'])
        t = np.arange(T)

        vm = _ema(h['v_model'], sm_alpha)
        vr = _ema(h['v_reward'], sm_alpha)
        va = _ema(h['v_action'], sm_alpha)
        vc = _ema(h['valence'], sm_alpha)

        # ── Row 1: Channel traces ──────────────────────
        ax = axes[0, col]
        ax.plot(t, vm, color=_TCOL['backward'], lw=1.2,
                label='Backward ($v_{model}$)')
        ax.plot(t, vr, color=_TCOL['present'],  lw=1.2,
                label='Present ($v_{reward}$)')
        ax.plot(t, va, color=_TCOL['forward'],  lw=1.2,
                label='Forward ($v_{action}$)')
        ax.plot(t, vc, color='k', lw=1.5, ls='--', alpha=0.6,
                label='Composite')
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(name.capitalize(), fontsize=12)
        if col == 0:
            ax.set_ylabel('Channel value')
            ax.legend(fontsize=7, loc='lower left')

        # ── Row 2: Normalised temporal orientation ─────
        ax = axes[1, col]
        abs_vm = np.abs(vm)
        abs_vr = np.abs(vr)
        abs_va = np.abs(va)
        total = abs_vm + abs_vr + abs_va + 1e-10
        frac_back = abs_vm / total
        frac_pres = abs_vr / total
        frac_fwd  = abs_va / total

        ax.fill_between(t, 0, frac_back,
                         color=_TCOL['backward'], alpha=0.7, label='Backward')
        ax.fill_between(t, frac_back, frac_back + frac_pres,
                         color=_TCOL['present'], alpha=0.7, label='Present')
        ax.fill_between(t, frac_back + frac_pres, 1.0,
                         color=_TCOL['forward'], alpha=0.7, label='Forward')
        ax.set_ylim(0, 1)
        if col == 0:
            ax.set_ylabel('Temporal orientation')
            ax.legend(fontsize=7, loc='lower left')

        # ── Row 3: Summary bars + disagreement ────────
        ax = axes[2, col]
        means = [np.mean(np.abs(vm)), np.mean(np.abs(vr)),
                 np.mean(np.abs(va))]
        labels = ['Backward', 'Present', 'Forward']
        colors = [_TCOL['backward'], _TCOL['present'], _TCOL['forward']]
        ax.bar(labels, means, color=colors, alpha=0.8, width=0.5)

        # Disagreement: fraction of timesteps where backward and forward
        # channels have opposite signs (temporal conflict)
        sign_back = np.sign(vm)
        sign_fwd  = np.sign(va)
        disagree_frac = np.mean(sign_back != sign_fwd)

        ax.text(0.97, 0.95, f'B-F conflict: {disagree_frac:.0%}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5',
                          edgecolor='gray', alpha=0.9))

        if col == 0:
            ax.set_ylabel('Mean |channel|')

    fig.suptitle('Temporal Aiming of Affective Channels',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_temporal_summary(results, save_path=None):
    """
    Figure 9: Temporal orientation summary across all emotion profiles.

    Left: Stacked horizontal bars showing the proportion of affect
          carried by each temporal channel (backward/present/forward).
    Right: Backward-forward conflict rate for each emotion.
    """
    names = list(results.keys())
    sm_alpha = 0.08

    fracs = {'backward': [], 'present': [], 'forward': []}
    conflicts = []

    for name in names:
        h = results[name]
        vm = _ema(h['v_model'], sm_alpha)
        vr = _ema(h['v_reward'], sm_alpha)
        va = _ema(h['v_action'], sm_alpha)

        abs_vm = np.mean(np.abs(vm))
        abs_vr = np.mean(np.abs(vr))
        abs_va = np.mean(np.abs(va))
        total = abs_vm + abs_vr + abs_va + 1e-10

        fracs['backward'].append(abs_vm / total)
        fracs['present'].append(abs_vr / total)
        fracs['forward'].append(abs_va / total)

        sign_back = np.sign(vm)
        sign_fwd  = np.sign(va)
        conflicts.append(np.mean(sign_back != sign_fwd))

    # Sort by backward fraction (most backward-oriented at top)
    order = np.argsort(fracs['backward'])[::-1]
    names_sorted = [names[i] for i in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    gridspec_kw={'width_ratios': [3, 1]})

    y = np.arange(len(names_sorted))
    b = np.array([fracs['backward'][i] for i in order])
    p = np.array([fracs['present'][i] for i in order])
    f = np.array([fracs['forward'][i] for i in order])

    ax1.barh(y, b, color=_TCOL['backward'], alpha=0.8, label='Backward')
    ax1.barh(y, p, left=b, color=_TCOL['present'], alpha=0.8,
             label='Present')
    ax1.barh(y, f, left=b + p, color=_TCOL['forward'], alpha=0.8,
             label='Forward')
    ax1.set_yticks(y)
    ax1.set_yticklabels([n.capitalize() for n in names_sorted])
    ax1.set_xlabel('Temporal orientation fraction')
    ax1.set_xlim(0, 1)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.set_title('Temporal Orientation by Emotion')

    # Right: conflict bars
    c_sorted = [conflicts[i] for i in order]
    cols = [_ECOL.get(n, '#999') for n in names_sorted]
    ax2.barh(y, c_sorted, color=cols, alpha=0.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel('B-F conflict rate')
    ax2.set_xlim(0, 1)
    ax2.set_title('Temporal Conflict')

    fig.suptitle('Temporal Aiming Across Emotion Profiles',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# ── Figure 10: Feedback reliance (Gap 2) ─────────────────
_FRCOL = {'healthy': '#2ecc71', 'recall_impaired': '#e67e22'}


def plot_feedback_reliance(results, save_path=None):
    """
    Figure 10: Feedback reliance when RECALL is impaired.

    Shows that with identical c_scale and K, low pi_pos shifts the agent
    toward ENGAGE (external feedback) because RECALL is biologically
    ineffective.

    1×3 layout:
      (a) Action proportions comparison (grouped bars)
      (b) Valence trajectories overlay
      (c) Policy entropy over time
    """
    names = ['healthy', 'recall_impaired']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Action proportions — grouped bars
    ax = axes[0]
    x = np.arange(N_ACTIONS)
    w = 0.3
    for i, name in enumerate(names):
        h = results[name]
        props = [np.mean(h['action'] == a) for a in range(N_ACTIONS)]
        bars = ax.bar(x + i * w, props, w, label=name.replace('_', ' ').title(),
                      color=_FRCOL[name], alpha=0.85)
        for bar, p in zip(bars, props):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{p:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(ACTION_NAMES)
    ax.set_ylabel('Proportion')
    ax.set_title('(a) Action Selection')
    ax.legend(fontsize=9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    # (b) Valence trajectories
    ax = axes[1]
    for name in names:
        h = results[name]
        T = len(h['valence_belief'])
        ax.plot(np.arange(T), _ema(h['valence_belief'], 0.1),
                color=_FRCOL[name], lw=1.5,
                label=name.replace('_', ' ').title())
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Believed Valence')
    ax.set_title('(b) Valence Trajectories')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)

    # (c) Policy entropy over time
    ax = axes[2]
    for name in names:
        h = results[name]
        ax.plot(np.arange(len(h['policy_entropy_norm'])),
                _ema(h['policy_entropy_norm'], 0.08),
                color=_FRCOL[name], lw=1.5,
                label=name.replace('_', ' ').title())
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Normalised Policy Entropy')
    ax.set_title('(c) Policy Entropy')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)

    fig.suptitle('Feedback Reliance: Effect of Recall Impairment ($\\pi_{pos}$)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# ── Figure 11: Framing dynamics (Gaps A, B, C) ──────────
_FCOL = ['#3498db', '#2ecc71', '#e67e22']   # PAST, PRESENT, FUTURE


def plot_framing_dynamics(results, save_path=None):
    """
    Figure 11: Temporal Framing Dynamics (2x3).

    All analyses derive from the posterior frame beliefs q(f|o) and policy
    distributions already tracked in the phenotype experiment — no new
    generative-model quantities needed, only post-hoc statistics.

    (a) Frame beliefs over time — stacked area of q(PAST), q(PRESENT), q(FUTURE)
    (b) VFE conditioned on action — grouped bars: E[F|a] per phenotype (Gap B)
    (c) Action run lengths — grouped bars per action, per phenotype (Gap C)
    (d) Action transition matrix — heatmaps P(a_t | a_{t-1}) (Gap A)
    (e) Post-action valence stability — rolling variance after each action (Gap A)
    (f) VFE–FUTURATE correlation — P(FUTURATE) from policy vs VFE (Gap B)
    """
    names = ['healthy', 'depressive', 'manic']
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # ── (a) Frame beliefs over time ──────────────────────
    for i, name in enumerate(names):
        ax = axes[0, 0] if i == 0 else (axes[0, 0] if False else None)
    # Use three sub-axes inside (a) via inset — simpler: one panel per phenotype
    # Actually, overlay all three as separate subplots would crowd. Use a single
    # panel with the healthy agent's frame beliefs as primary example.
    ax = axes[0, 0]
    for i, name in enumerate(names):
        h = results[name]
        fb = h['frame_belief']   # (T, 3)
        T = len(fb)
        t = np.arange(T)
        # Plot the FUTURE belief as a line per phenotype
        fut_sm = _ema(fb[:, 2], 0.08)
        ax.plot(t, fut_sm, color=PCOL[name], lw=1.5, label=f'{name.capitalize()}')
    ax.set_ylabel('$q(f{=}\\mathrm{FUTURE} \\mid o)$')
    ax.set_xlabel('Timestep')
    ax.set_title('(a) Future Frame Belief')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    # ── (b) VFE conditioned on action (Gap B) ────────────
    ax = axes[0, 1]
    x = np.arange(N_ACTIONS)
    w = 0.22
    for i, name in enumerate(names):
        h = results[name]
        vfe_by_action = []
        for a in range(N_ACTIONS):
            mask = h['action'] == a
            if mask.sum() > 0:
                vfe_by_action.append(np.mean(h['vfe'][mask]))
            else:
                vfe_by_action.append(0.0)
        ax.bar(x + i * w, vfe_by_action, w, label=name.capitalize(),
               color=PCOL[name], alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels(ACTION_NAMES, fontsize=9)
    ax.set_ylabel('$\\mathbb{E}[F \\mid a]$')
    ax.set_title('(b) VFE Conditioned on Action')
    ax.legend(fontsize=8)

    # ── (c) Action run lengths (Gap C) ───────────────────
    ax = axes[0, 2]
    x = np.arange(N_ACTIONS)
    w = 0.22
    for i, name in enumerate(names):
        h = results[name]
        actions = h['action']
        run_lengths_by_action = []
        for a in range(N_ACTIONS):
            runs = []
            count = 0
            for t_idx in range(len(actions)):
                if actions[t_idx] == a:
                    count += 1
                else:
                    if count > 0:
                        runs.append(count)
                    count = 0
            if count > 0:
                runs.append(count)
            run_lengths_by_action.append(np.mean(runs) if runs else 0.0)
        ax.bar(x + i * w, run_lengths_by_action, w,
               label=name.capitalize(), color=PCOL[name], alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels(ACTION_NAMES, fontsize=9)
    ax.set_ylabel('Mean consecutive run length')
    ax.set_title('(c) Action Run Lengths')
    ax.legend(fontsize=8)

    # ── (d) Action transition matrices (Gap A) ──────────
    ax = axes[1, 0]
    trans_mats = []
    for name in names:
        h = results[name]
        actions = h['action']
        T_mat = np.zeros((N_ACTIONS, N_ACTIONS))
        for t_idx in range(1, len(actions)):
            T_mat[actions[t_idx], actions[t_idx - 1]] += 1
        # Normalise columns: P(a_t | a_{t-1})
        col_sums = T_mat.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        T_mat /= col_sums
        trans_mats.append(T_mat)

    # Display as 3 side-by-side heatmaps within the single axes
    combined = np.hstack([tm for tm in trans_mats])
    # Add thin separator columns
    sep = np.full((N_ACTIONS, 1), np.nan)
    display = trans_mats[0]
    for tm in trans_mats[1:]:
        display = np.hstack([display, sep, tm])

    im = ax.imshow(display, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    # Label ticks
    n_cols = display.shape[1]
    # Phenotype labels above
    for pi, name in enumerate(names):
        center_x = pi * (N_ACTIONS + 1) + N_ACTIONS / 2 - 0.5
        ax.text(center_x, -0.8, name.capitalize(), ha='center', fontsize=9,
                fontweight='bold')
    ax.set_yticks(range(N_ACTIONS))
    ax.set_yticklabels(ACTION_NAMES, fontsize=8)
    # X-tick: action labels repeated per phenotype
    xtick_pos = []
    xtick_lab = []
    for pi in range(len(names)):
        offset = pi * (N_ACTIONS + 1)
        for a in range(N_ACTIONS):
            xtick_pos.append(offset + a)
            xtick_lab.append(ACTION_NAMES[a][0])  # first letter
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab, fontsize=7)
    ax.set_ylabel('$a_t$ (to)')
    ax.set_xlabel('$a_{t-1}$ (from)')
    ax.set_title('(d) Action Transition $P(a_t \\mid a_{t-1})$')

    # Annotate cells with values
    for pi, tm in enumerate(trans_mats):
        offset = pi * (N_ACTIONS + 1)
        for row in range(N_ACTIONS):
            for col in range(N_ACTIONS):
                val = tm[row, col]
                color = 'white' if val > 0.5 else 'black'
                ax.text(offset + col, row, f'{val:.2f}', ha='center',
                        va='center', fontsize=6, color=color)

    # ── (e) Post-action valence stability (Gap A) ────────
    ax = axes[1, 1]
    win = 5
    action_subset = [RECALL, ENGAGE, FUTURATE, REST]
    x = np.arange(len(action_subset))
    w = 0.22
    for i, name in enumerate(names):
        h = results[name]
        actions = h['action']
        valence = h['valence']
        vars_by_action = []
        for a in action_subset:
            # Find timesteps where this action was selected
            mask_indices = np.where(actions == a)[0]
            rolling_vars = []
            for idx in mask_indices:
                if idx + win <= len(valence):
                    window = valence[idx:idx + win]
                    rolling_vars.append(np.var(window))
            vars_by_action.append(np.mean(rolling_vars) if rolling_vars else 0.0)
        ax.bar(x + i * w, vars_by_action, w, label=name.capitalize(),
               color=PCOL[name], alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels(ACTION_NAMES, fontsize=9)
    ax.set_ylabel('Mean post-action Var($V$, 5-step)')
    ax.set_title('(e) Post-Action Valence Stability')
    ax.legend(fontsize=8)

    # ── (f) VFE–FUTURATE correlation (Gap B) ─────────────
    ax = axes[1, 2]
    for name in names:
        h = results[name]
        vfe = h['vfe']
        p_fut = h['pi'][:, FUTURATE]
        # Bin VFE into quantiles and compute mean P(FUTURATE) per bin
        n_bins = 8
        vfe_sorted_idx = np.argsort(vfe)
        bin_size = max(1, len(vfe) // n_bins)
        bin_centers = []
        bin_p_fut = []
        for b in range(n_bins):
            start = b * bin_size
            end = min(start + bin_size, len(vfe))
            if start >= len(vfe):
                break
            idx = vfe_sorted_idx[start:end]
            bin_centers.append(np.mean(vfe[idx]))
            bin_p_fut.append(np.mean(p_fut[idx]))
        ax.plot(bin_centers, bin_p_fut, 'o-', color=PCOL[name], lw=1.5,
                ms=5, label=name.capitalize())

    ax.set_xlabel('VFE (binned)')
    ax.set_ylabel('$\\pi(\\mathrm{FUTURATE})$')
    ax.set_title('(f) VFE–FUTURATE Correlation')
    ax.legend(fontsize=8)

    fig.suptitle('Temporal Framing Dynamics', fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# ── Figure 12: Chronic stress (Gap D) ───────────────────

def plot_chronic_stress(results, save_path=None):
    """
    Figure 12: Chronic Stress as Maladaptive Future Stabilisation (1x3).

    Compares stressed vs healthy agent to demonstrate that chronic
    environmental volatility + reward hypersensitivity locks the agent
    into FUTURATE, producing:
      (a) Frame belief dominated by FUTURE
      (b) Persistently negative v_model (backward valence)
      (c) Valence channel decomposition showing backward channel dominance

    All quantities are standard active inference readouts — no new
    generative-model parameters, only a different environment volatility
    and C-vector scaling.
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    _SCOL = {'healthy': '#2ecc71', 'stressed': '#e74c3c'}
    sm_alpha = 0.08

    # ── (a) Frame beliefs: stacked area ──────────────────
    ax = axes[0]
    for name in ['healthy', 'stressed']:
        h = results[name]
        fb = h['frame_belief']
        T = len(fb)
        t = np.arange(T)
        fut_sm = _ema(fb[:, 2], sm_alpha)
        ax.plot(t, fut_sm, color=_SCOL[name], lw=1.5,
                label=f'{name.capitalize()} $q(f{{=}}\\mathrm{{FUTURE}})$')
        past_sm = _ema(fb[:, 0], sm_alpha)
        ax.plot(t, past_sm, color=_SCOL[name], lw=1.0, ls='--', alpha=0.6,
                label=f'{name.capitalize()} $q(f{{=}}\\mathrm{{PAST}})$')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Frame belief')
    ax.set_ylim(0, 1)
    ax.set_title('(a) Temporal Frame Beliefs')
    ax.legend(fontsize=7, loc='upper right')

    # ── (b) Backward valence v_model ─────────────────────
    ax = axes[1]
    for name in ['healthy', 'stressed']:
        h = results[name]
        T = len(h['v_model'])
        t = np.arange(T)
        vm = _ema(h['v_model'], sm_alpha)
        ax.plot(t, vm, color=_SCOL[name], lw=1.5, label=name.capitalize())
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('$v_{\\mathrm{model}}$ (backward valence)')
    ax.set_ylim(-1.1, 1.1)
    ax.set_title('(b) Backward Valence $v_{\\mathrm{model}}$')
    ax.legend(fontsize=9)

    # ── (c) Valence channels decomposition (stressed) ────
    ax = axes[2]
    h = results['stressed']
    T = len(h['v_model'])
    t = np.arange(T)
    vm = _ema(h['v_model'], sm_alpha)
    vr = _ema(h['v_reward'], sm_alpha)
    va = _ema(h['v_action'], sm_alpha)

    ax.plot(t, vm, color=_TCOL['backward'], lw=1.2,
            label='Backward ($v_{\\mathrm{model}}$)')
    ax.plot(t, vr, color=_TCOL['present'], lw=1.2,
            label='Present ($v_{\\mathrm{reward}}$)')
    ax.plot(t, va, color=_TCOL['forward'], lw=1.2,
            label='Forward ($v_{\\mathrm{action}}$)')
    vc = _ema(h['valence'], sm_alpha)
    ax.plot(t, vc, color='k', lw=1.5, ls='--', alpha=0.6, label='Composite')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Channel value')
    ax.set_ylim(-1.1, 1.1)
    ax.set_title('(c) Stressed Agent: Valence Channels')
    ax.legend(fontsize=7)

    fig.suptitle('Chronic Stress: Maladaptive Future Stabilisation',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
