"""
Experiment configurations and runners.

Experiments
-----------
1. Phenotype comparison   – healthy / depressive / manic
2. Granularity sweep      – K = 2, 4, 6, 8
3. Parameter landscape    – pi_pos x omega_e heatmap
4. Emotion validation     – 10 profiles targeting PAD regions
"""

import numpy as np
from generative_model import build_model, N_ACTIONS
from agent import Agent
from environment import Environment


# ── Clinical profiles ──────────────────────────────────────
PROFILES = {
    'healthy': dict(
        K=8, M=8, pi_pos=5.0, omega_e=5.0, gamma=16.0, c_scale=1.0,
        desc='Balanced precision, high granularity, normal reward sensitivity',
    ),
    'depressive': dict(
        K=4, M=8, pi_pos=0.2, omega_e=5.0, gamma=16.0, c_scale=0.1,
        desc='Low positive-belief precision, low granularity, anhedonic',
    ),
    'manic': dict(
        K=4, M=8, pi_pos=5.0, omega_e=0.5, gamma=16.0, c_scale=2.0,
        desc='Overconfident, poor energy estimation, hypersensitive reward',
    ),
}


# ── Single trial ───────────────────────────────────────────
def run_trial(K=8, M=5, pi_pos=5.0, omega_e=5.0, gamma=16.0, c_scale=1.0,
              T=300, volatility=0.45, seed=42, **_ignored):
    """
    Run one agent–environment trial.

    Returns dict of (T,)-shaped arrays for every tracked variable.
    """
    model = build_model(K=K, M=M, pi_pos=pi_pos, omega_e=omega_e,
                        gamma=gamma, c_scale=c_scale)
    agent = Agent(model, gamma=gamma)
    env = Environment(K=K, M=M, volatility=volatility, seed=seed,
                      pi_pos=pi_pos, c_scale=c_scale)

    hist = _make_history(T, K)
    obs = env.reset()

    for t in range(T):
        action, info = agent.step(obs)
        obs, env_info = env.step(action)

        # Marginal beliefs
        joint = info['beliefs'].reshape(K, M, 3)
        v_marg = joint.sum(axis=(1, 2))
        e_marg = joint.sum(axis=(0, 2))

        hist['valence_true'][t]  = env_info['true_v'] / max(K - 1, 1)
        hist['energy_true'][t]   = env_info['true_e'] / max(M - 1, 1)
        hist['frame_true'][t]    = env_info['true_f']
        hist['valence_belief'][t] = np.dot(np.arange(K), v_marg) / max(K - 1, 1)
        hist['energy_belief'][t]  = np.dot(np.arange(M), e_marg) / max(M - 1, 1)
        hist['action'][t]  = action
        hist['vfe'][t]     = info['vfe']
        hist['dF'][t]      = info['dF']
        hist['valence_jc'][t] = info['valence_jc']
        hist['G'][t]       = info['G']
        hist['pi'][t]      = info['pi']
        # Three-channel valence
        hist['v_model'][t]   = info['v_model']
        hist['v_reward'][t]  = info['v_reward']
        hist['v_action'][t]  = info['v_action']
        hist['valence'][t]   = info['valence']
        # Pattisapu et al. (legacy)
        hist['valence_p'][t] = info['valence_p']
        hist['arousal_p'][t] = info['arousal_p']
        hist['arousal_norm'][t] = info['arousal_norm']
        hist['policy_entropy_norm'][t] = info['policy_entropy_norm']

    # Second derivative → anticipation
    hist['d2F'] = np.diff(hist['dF'], prepend=hist['dF'][0])
    hist['anticipation'] = -hist['d2F']

    return hist


def _make_history(T, K):
    return dict(
        valence_true   = np.zeros(T),
        energy_true    = np.zeros(T),
        frame_true     = np.zeros(T, dtype=int),
        valence_belief = np.zeros(T),
        energy_belief  = np.zeros(T),
        action         = np.zeros(T, dtype=int),
        vfe            = np.zeros(T),
        dF             = np.zeros(T),
        valence_jc     = np.zeros(T),
        G              = np.zeros((T, N_ACTIONS)),
        pi             = np.zeros((T, N_ACTIONS)),
        d2F            = np.zeros(T),
        anticipation   = np.zeros(T),
        # Three-channel valence
        v_model        = np.zeros(T),
        v_reward       = np.zeros(T),
        v_action       = np.zeros(T),
        valence        = np.zeros(T),
        # Pattisapu et al. (legacy)
        valence_p      = np.zeros(T),
        arousal_p      = np.zeros(T),
        arousal_norm   = np.zeros(T),
        policy_entropy_norm = np.zeros(T),
    )


# ── Experiment 1: phenotype comparison ─────────────────────
def run_phenotype_experiment(T=300, seed=42):
    results = {}
    for name, prof in PROFILES.items():
        print(f"    {name} …", end=" ", flush=True)
        results[name] = run_trial(**prof, T=T, seed=seed)
        mv = np.mean(results[name]['valence_belief'])
        me = np.mean(results[name]['energy_true'])
        print(f"mean_v={mv:.3f}  mean_e={me:.3f}")
    return results


# ── Experiment 2: granularity sweep ────────────────────────
def run_granularity_experiment(T=300, n_runs=5, base_seed=42):
    results = {}
    for K in [2, 4, 6, 8]:
        runs = []
        for r in range(n_runs):
            h = run_trial(K=K, M=8, pi_pos=3.0, omega_e=0.8, gamma=16.0,
                          T=T, seed=base_seed + r)
            runs.append(h)
        results[K] = runs
        mv = np.mean([np.mean(r['valence_belief']) for r in runs])
        vv = np.mean([np.var(r['valence_belief']) for r in runs])
        print(f"    K={K}  mean_v={mv:.3f}  var_v={vv:.4f}")
    return results


# ── Experiment 3: parameter landscape ──────────────────────
def run_parameter_sweep(T=200, seed=42, n_pi=10, n_omega=10):
    pi_vals = np.linspace(0.1, 8.0, n_pi)
    om_vals = np.linspace(0.1, 8.0, n_omega)

    mean_valence     = np.zeros((n_pi, n_omega))
    valence_variance = np.zeros((n_pi, n_omega))
    mean_energy      = np.zeros((n_pi, n_omega))
    action_entropy   = np.zeros((n_pi, n_omega))

    total = n_pi * n_omega
    done = 0
    for i, pi in enumerate(pi_vals):
        for j, om in enumerate(om_vals):
            h = run_trial(K=4, M=8, pi_pos=pi, omega_e=om, gamma=16.0,
                          T=T, seed=seed)
            mean_valence[i, j]     = np.mean(h['valence_belief'])
            valence_variance[i, j] = np.var(h['valence_belief'])
            mean_energy[i, j]      = np.mean(h['energy_belief'])
            # Action entropy — higher = more diverse policy
            act_counts = np.array([np.mean(h['action'] == a)
                                   for a in range(N_ACTIONS)])
            act_counts = np.maximum(act_counts, 1e-10)
            action_entropy[i, j] = float(-np.dot(act_counts,
                                                  np.log(act_counts)))
            done += 1
        print(f"    sweep {done}/{total}", flush=True)

    return dict(
        pi_values=pi_vals, omega_values=om_vals,
        mean_valence=mean_valence, valence_variance=valence_variance,
        mean_energy=mean_energy, action_entropy=action_entropy,
    )


# ── Experiment 4: PAD emotion validation ─────────────────
# Each profile targets a region of the Pleasure-Arousal-Dominance space.
# Key decoupling: c_scale controls arousal (mean G magnitude),
# gamma controls dominance (policy precision) independently.
EMOTION_PROFILES = {
    'happy': dict(
        K=8, M=8, pi_pos=5.0, omega_e=5.0, gamma=16.0, c_scale=1.5,
        volatility=0.3,
    ),
    'content': dict(
        K=8, M=8, pi_pos=5.0, omega_e=5.0, gamma=16.0, c_scale=0.8,
        volatility=0.3,
    ),
    'calm': dict(
        K=8, M=8, pi_pos=2.0, omega_e=3.0, gamma=16.0, c_scale=0.7,
        volatility=0.3,
    ),
    'excited': dict(
        K=4, M=8, pi_pos=5.0, omega_e=0.5, gamma=4.0, c_scale=2.5,
        volatility=0.45,
    ),
    'alert': dict(
        K=4, M=8, pi_pos=2.5, omega_e=2.0, gamma=16.0, c_scale=2.0,
        volatility=0.6,
    ),
    'angry': dict(
        K=8, M=8, pi_pos=0.1, omega_e=0.2, gamma=16.0, c_scale=5.0,
        volatility=0.8,
    ),
    'fearful': dict(
        K=4, M=8, pi_pos=0.3, omega_e=0.3, gamma=4.0, c_scale=2.5,
        volatility=0.8,
    ),
    'sad': dict(
        K=4, M=8, pi_pos=0.1, omega_e=3.0, gamma=16.0, c_scale=0.25,
        volatility=0.6,
    ),
    'depressed': dict(
        K=4, M=8, pi_pos=0.2, omega_e=5.0, gamma=16.0, c_scale=0.1,
        volatility=0.45,
    ),
    'bored': dict(
        K=4, M=8, pi_pos=1.5, omega_e=5.0, gamma=16.0, c_scale=0.15,
        volatility=0.3,
    ),
}


def run_emotion_validation(T=300, seed=42):
    results = {}
    for name, prof in EMOTION_PROFILES.items():
        h = run_trial(**prof, T=T, seed=seed)
        results[name] = h
    return results
