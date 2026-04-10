"""
Active inference agent for the temporal framing model.

Performs:
  1. Bayesian belief update  (predict → observe → posterior)
  2. EFE-based policy evaluation  (risk + ambiguity per modality)
  3. Softmax policy selection
  4. Three-channel affective readout:
     a) v_model  = tanh(−ΔF/τ)     — backward (Joffily & Coricelli 2013)
     b) v_reward = tanh((U−EU)/τ)   — present  (Pattisapu et al. 2024)
     c) v_action = tanh(AC/τ)       — forward  (Hesp et al. 2021)
     Composite: valence = tanh(v_model + v_reward + v_action)
     Arousal: H[Q(s|o)] normalised to [−1, 1]
"""

import numpy as np
from generative_model import ModelSpec, N_ACTIONS, EPS


class Agent:
    def __init__(self, model: ModelSpec, gamma: float = 16.0,
                 tau_model: float = 0.5, tau_reward: float = 2.0,
                 tau_action: float = 1.0):
        self.model = model
        self.gamma = gamma
        self.tau_model = tau_model
        self.tau_reward = tau_reward
        self.tau_action = tau_action
        self.beliefs = model.D.copy()
        self.prev_action = None
        self._vfe_prev = None
        self._pi_prev = None    # previous policy for affective charge

        # Pre-compute normalised preference distributions  σ(C_m)
        self._log_pref = []
        for Cm in model.C:
            p = np.exp(Cm - Cm.max())
            p /= (p.sum() + EPS)
            self._log_pref.append(np.log(p + EPS))

    def reset(self):
        self.beliefs = self.model.D.copy()
        self.prev_action = None
        self._vfe_prev = None
        self._pi_prev = None

    # ── Main loop ──────────────────────────────────────────
    def step(self, obs):
        """
        Full inference-action cycle.

        Parameters
        ----------
        obs : list[int]
            Observation indices [o_ext, o_int, o_val].

        Returns
        -------
        action : int
        info   : dict
        """
        # 1. Predict
        if self.prev_action is not None:
            q_pred = self.model.B[self.prev_action] @ self.beliefs
        else:
            q_pred = self.model.D.copy()
        q_pred = np.maximum(q_pred, EPS)
        q_pred /= q_pred.sum()

        # 2. Observe — multiply likelihoods across modalities
        log_lik = np.zeros(self.model.n_states)
        for m, o_m in enumerate(obs):
            log_lik += np.log(self.model.A[m][o_m, :] + EPS)

        log_post = np.log(q_pred + EPS) + log_lik
        log_post -= log_post.max()
        q_post = np.exp(log_post)
        q_post /= (q_post.sum() + EPS)
        self.beliefs = q_post

        # ── VFE ──────────────────────────────────────────────
        accuracy = sum(
            np.dot(q_post, np.log(self.model.A[m][obs[m], :] + EPS))
            for m in range(len(obs))
        )
        complexity = np.dot(q_post,
                            np.log(q_post + EPS) - np.log(q_pred + EPS))
        vfe = -accuracy + complexity

        if self._vfe_prev is not None:
            dF = vfe - self._vfe_prev
        else:
            dF = 0.0
        self._vfe_prev = vfe

        # ── Three-channel valence ────────────────────────────
        # Channel 1: v_model — backward (Joffily & Coricelli 2013)
        v_model = float(np.tanh(-dF / self.tau_model))

        # Channel 2: v_reward — present (Pattisapu et al. 2024)
        U = sum(self._log_pref[m][obs[m]] for m in range(len(obs)))
        EU = 0.0
        for m in range(len(self.model.A)):
            q_o = self.model.A[m] @ q_pred
            q_o = np.maximum(q_o, EPS)
            q_o /= q_o.sum()
            EU += float(np.dot(q_o, self._log_pref[m]))
        v_reward = float(np.tanh((U - EU) / self.tau_reward))

        # Arousal = posterior state entropy (Pattisapu)
        arousal_p = float(-np.dot(q_post, np.log(q_post + EPS)))
        max_H = np.log(self.model.n_states)
        arousal_norm = arousal_p / max_H if max_H > 0 else 0.0

        # ── EFE & policy selection ─────────────────────────
        G = np.array([self._efe(a) for a in range(N_ACTIONS)])

        log_pi = -self.gamma * G
        log_pi -= log_pi.max()
        pi = np.exp(log_pi)
        pi /= (pi.sum() + EPS)

        # Channel 3: v_action — forward (Hesp et al. 2021)
        # Affective charge: policy improvement weighted by EFE
        if self._pi_prev is not None:
            AC = float(np.dot(self._pi_prev - pi, G))
            v_action = float(np.tanh(AC / self.tau_action))
        else:
            AC = 0.0
            v_action = 0.0

        # Composite valence
        valence = float(np.tanh(v_model + v_reward + v_action))

        action = int(np.random.choice(N_ACTIONS, p=pi))
        self.prev_action = action
        self._pi_prev = pi.copy()

        # Policy entropy (decision uncertainty)
        pi_safe = np.maximum(pi, EPS)
        policy_entropy = float(-np.dot(pi_safe, np.log(pi_safe)))
        max_H_pi = np.log(N_ACTIONS)
        policy_entropy_norm = policy_entropy / max_H_pi if max_H_pi > 0 else 0.0

        info = dict(
            beliefs=q_post.copy(),
            q_pred=q_pred.copy(),
            G=G.copy(),
            pi=pi.copy(),
            # Three-channel valence (all tanh-bounded to [-1, 1])
            v_model=v_model,
            v_reward=v_reward,
            v_action=v_action,
            valence=valence,
            # Legacy aliases
            vfe=float(vfe),
            dF=float(dF),
            valence_jc=v_model,
            valence_p=float(U - EU),
            # Arousal (two measures)
            arousal_p=arousal_p,
            arousal_norm=arousal_norm,           # state entropy
            policy_entropy_norm=policy_entropy_norm,  # policy entropy
            utility=float(U),
            expected_utility=float(EU),
        )
        return action, info

    # ── EFE for a single action ────────────────────────────
    def _efe(self, action):
        """G(a) = Σ_m [ risk_m + ambiguity_m ]."""
        q_pred = self.model.B[action] @ self.beliefs
        q_pred = np.maximum(q_pred, EPS)
        q_pred /= q_pred.sum()

        G = 0.0
        for m in range(len(self.model.A)):
            Am = self.model.A[m]
            Cm = self.model.C[m]

            q_o = Am @ q_pred
            q_o = np.maximum(q_o, EPS)
            q_o /= q_o.sum()

            p_pref = np.exp(Cm - Cm.max())
            p_pref /= (p_pref.sum() + EPS)

            risk = float(np.dot(q_o, np.log(q_o + EPS) - np.log(p_pref + EPS)))
            H_cols = -np.sum(Am * np.log(Am + EPS), axis=0)
            ambiguity = float(np.dot(q_pred, H_cols))

            G += risk + ambiguity

        return G
