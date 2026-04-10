"""
Active inference agent for the temporal framing model.

Performs:
  1. Bayesian belief update  (predict → observe → posterior)
  2. EFE-based policy evaluation  (risk + ambiguity per modality)
  3. Softmax policy selection
  4. Dual affect readout:
     a) Pattisapu et al. (2024):  V = U − EU,  A = H[Q(s|o)]
     b) Joffily & Coricelli (2013):  valence = −dF/dt
"""

import numpy as np
from generative_model import ModelSpec, N_ACTIONS, EPS


class Agent:
    def __init__(self, model: ModelSpec, gamma: float = 16.0):
        self.model = model
        self.gamma = gamma
        self.beliefs = model.D.copy()
        self.prev_action = None
        self._vfe_prev = None

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

        # ── VFE (for Joffily-Coricelli) ───────────────────
        accuracy = sum(
            np.dot(q_post, np.log(self.model.A[m][obs[m], :] + EPS))
            for m in range(len(obs))
        )
        complexity = np.dot(q_post,
                            np.log(q_post + EPS) - np.log(q_pred + EPS))
        vfe = -accuracy + complexity

        if self._vfe_prev is not None:
            dF = vfe - self._vfe_prev
            valence_jc = float(np.tanh(-dF / 0.5))
        else:
            dF = 0.0
            valence_jc = 0.0
        self._vfe_prev = vfe

        # ── Pattisapu et al.:  V = U − EU,  A = H[q(s|o)] ─
        # Utility of actual observation
        U = sum(self._log_pref[m][obs[m]] for m in range(len(obs)))

        # Expected utility under predicted observations
        EU = 0.0
        for m in range(len(self.model.A)):
            q_o = self.model.A[m] @ q_pred          # predicted obs
            q_o = np.maximum(q_o, EPS)
            q_o /= q_o.sum()
            EU += float(np.dot(q_o, self._log_pref[m]))

        valence_p = float(U - EU)   # reward prediction error

        # Arousal = posterior entropy
        arousal_p = float(-np.dot(q_post, np.log(q_post + EPS)))

        # Normalised arousal  ∈ [0, 1]
        max_H = np.log(self.model.n_states)
        arousal_norm = arousal_p / max_H if max_H > 0 else 0.0

        # ── EFE & policy selection ─────────────────────────
        G = np.array([self._efe(a) for a in range(N_ACTIONS)])

        log_pi = -self.gamma * G
        log_pi -= log_pi.max()
        pi = np.exp(log_pi)
        pi /= (pi.sum() + EPS)
        action = int(np.random.choice(N_ACTIONS, p=pi))
        self.prev_action = action

        info = dict(
            beliefs=q_post.copy(),
            q_pred=q_pred.copy(),
            G=G.copy(),
            pi=pi.copy(),
            # Joffily-Coricelli
            vfe=float(vfe),
            dF=float(dF),
            valence_jc=valence_jc,
            # Pattisapu et al.
            valence_p=valence_p,
            arousal_p=arousal_p,
            arousal_norm=arousal_norm,
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
