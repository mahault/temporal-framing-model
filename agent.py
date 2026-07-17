"""
Active inference agent for the temporal framing model.

Two-level hierarchical POMDP:
  M4 (emotion, every step):
    1. Bayesian belief update  (predict → observe → posterior)
    2. EFE-based policy evaluation  (risk + ambiguity per modality)
    3. Softmax policy selection
    4. Three-channel affective readout:
       a) v_model  = tanh(−ΔF/τ)     — backward (Joffily & Coricelli 2013)
       b) v_reward = tanh((U−EU)/τ)   — present  (Pattisapu et al. 2024)
       c) v_action = tanh(AC/τ)       — forward  (Hesp et al. 2021)
       Composite: valence = tanh(v_model + v_reward + v_action)
       Arousal: H[Q(s|o)] normalised to [−1, 1]

  M5 (mood, every T_mood steps):
    Hierarchical POMDP over discretised pi_pos.
    Observes mean VFE over the emotion-level window.
    Bayesian filtering: predict (B_mood) → update (A_mood likelihood).
    E[pi_pos] from mood posterior parameterises emotion-level B matrices.
"""

import numpy as np
from generative_model import (ModelSpec, N_ACTIONS, EPS,
                              B_frame, rebuild_B_single, rebuild_B_with_frame,
                              N_MOOD, MOOD_BIN_CENTERS, N_OBS_MOOD,
                              _compute_vfe_curve, _build_A_mood_from_vfe,
                              build_B_mood, build_D_mood)


class Agent:
    def __init__(self, model: ModelSpec, gamma: float = 16.0,
                 tau_model: float = 0.5, tau_reward: float = 2.0,
                 tau_action: float = 1.0,
                 pi_pos: float = 5.0, T_mood: int = 50,
                 omega_e: float = 5.0, c_scale: float = 1.0,
                 c_pos: float = None, c_neg: float = None,
                 neg_val_precision: float = 1.0,
                 valence_inertia: float = 0.0,
                 habit_E: np.ndarray = None,
                 affect_precision_gain: float = 0.5,
                 learn_B_frame: bool = False, frame_concentration: float = 50.0,
                 counterfactual_horizon: int = 2,
                 counterfactual_discount: float = 0.75,
                 adaptive_counterfactual_horizon: bool = False,
                 max_counterfactual_horizon: int = 4,
                 seed: int = 0):
        self.model = model
        self.gamma = gamma
        self.rng = np.random.RandomState(seed)
        self.tau_model = tau_model
        self.tau_reward = tau_reward
        self.tau_action = tau_action
        self.beliefs = model.D.copy()
        self.valence_inertia = float(valence_inertia)
        self.prev_action = None
        self._vfe_prev = None
        # E vector: log-prior over policies (habit prior, Da Costa et al. 2020)
        # Additive in log-space: log_pi = -gamma*G + E
        # None → uniform prior (no habit bias), backward compatible
        self._habit_E = habit_E
        self._pi_prev = None    # previous policy for affective charge
        self._v_action_prev = 0.0
        self.affect_precision_gain = float(affect_precision_gain)
        self.counterfactual_horizon = max(1, int(counterfactual_horizon))
        self.counterfactual_discount = float(counterfactual_discount)
        self.adaptive_counterfactual_horizon = bool(adaptive_counterfactual_horizon)
        self.max_counterfactual_horizon = max(
            self.counterfactual_horizon, int(max_counterfactual_horizon)
        )
        self._vfe_ema = None
        self._vfe_var_ema = None
        self._vfe_alpha = 0.05
        self._last_counterfactual_horizon = self.counterfactual_horizon

        # ── M5 mood layer: per-model calibrated POMDP over pi_pos ──
        self.pi_pos = float(pi_pos)
        self._initial_pi_pos = float(pi_pos)
        self.T_mood = T_mood
        # Analytical VFE curve for this model's (K, M, omega_e, c_scale)
        self._vfe_curve = _compute_vfe_curve(model.K, model.M, omega_e, c_scale,
                                             c_pos=c_pos, c_neg=c_neg,
                                             neg_val_precision=neg_val_precision)
        self.A_mood, self._mood_obs_edges = _build_A_mood_from_vfe(self._vfe_curve)
        self.B_mood = build_B_mood()
        self.mood_beliefs = build_D_mood(pi_pos)
        self._vfe_buffer = []
        self._val_buffer = []           # believed-valence level (mood evidence)
        self._step_count = 0
        self._mood_calibrated = False   # online offset not yet applied

        # ── Interoceptive load coupling (Stephan et al. 2016) ──
        self._intero_vfe_ema = 0.0       # running interoceptive surprise
        self._alpha_intero = 0.1          # EMA decay rate
        self._load_threshold = 1.5        # intero VFE above this impairs pi_pos

        # ── Learned B_frame (Dirichlet) ──
        self.learn_B_frame = learn_B_frame
        if learn_B_frame:
            self._bf_counts = {a: B_frame(a) * frame_concentration
                               for a in range(N_ACTIONS)}

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
        self._v_action_prev = 0.0
        self.pi_pos = self._initial_pi_pos
        self.mood_beliefs = build_D_mood(self._initial_pi_pos)
        self._vfe_buffer = []
        self._val_buffer = []
        self._step_count = 0
        self._intero_vfe_ema = 0.0
        self._vfe_ema = None
        self._vfe_var_ema = None
        self._last_counterfactual_horizon = self.counterfactual_horizon
        # Reset mood calibration (re-use stored analytical curve)
        self._mood_calibrated = False
        self.A_mood, self._mood_obs_edges = _build_A_mood_from_vfe(self._vfe_curve)

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
        self._update_vfe_scale(float(vfe))

        # ── Interoceptive surprise tracking (Stephan et al. 2016) ──
        # Accuracy on the o_int modality (index 1) = interoceptive channel
        intero_acc = float(np.dot(q_post, np.log(self.model.A[1][obs[1], :] + EPS)))
        self._intero_vfe_ema = ((1.0 - self._alpha_intero) * self._intero_vfe_ema
                                + self._alpha_intero * (-intero_acc))

        # Fast interoceptive modulation of pi_pos (alongside slow M5 mood)
        load_penalty = max(0.0, self._intero_vfe_ema - self._load_threshold)
        pi_pos_eff = self.pi_pos / (1.0 + 0.5 * load_penalty)

        # ── Three-channel valence ────────────────────────────
        # Channel 1: v_model — backward (Joffily & Coricelli 2013)
        v_model = float(np.tanh(-dF / self.tau_model))

        # ── M5 mood layer: hierarchical Bayesian inference ──
        self._vfe_buffer.append(float(vfe))
        # Believed-valence level (0..1): the mood evidence. Unlike VFE (which
        # adapts away) valence level stays low in depression, so it is what lets
        # chronic stress accumulate into a persistent low-mood state.
        K_ = self.model.K
        v_lvl = q_post.reshape(K_, self.model.M, 3).sum(axis=(1, 2))
        self._val_buffer.append(float(v_lvl @ np.arange(K_) / max(K_ - 1, 1)))
        self._step_count += 1
        if self._step_count % self.T_mood == 0 and \
                len(self._vfe_buffer) >= self.T_mood:
            self._mood_update()

        # ── Learned B_frame: Dirichlet update ──
        K, M = self.model.K, self.model.M
        if self.learn_B_frame and self.prev_action is not None:
            f_post = q_post.reshape(K, M, 3).sum(axis=(0, 1))
            f_pred = q_pred.reshape(K, M, 3).sum(axis=(0, 1))
            self._bf_counts[self.prev_action] += np.outer(f_post, f_pred)
            self._rebuild_B()   # only needed when B_frame changes

        # Channel 2: v_reward — present (Pattisapu et al. 2024)
        # Hedonic modalities only (ext=0, val=2) for FELT valence.
        # Interoceptive modality (1) drives EFE/policy (allostatic regulation)
        # but does not produce conscious hedonic experience — body-budget
        # maintenance is largely preconscious (Seth 2013, Barrett 2017).
        _HEDONIC = [0, 2]   # o_ext, o_val
        U = sum(self._log_pref[m][obs[m]] for m in _HEDONIC)
        EU = 0.0
        for m in _HEDONIC:
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
        # Counterfactual EFE: evaluate each action by rolling the generative
        # model forward over short roads-not-taken.
        rollout_horizon = self._current_counterfactual_horizon(float(vfe))
        self._last_counterfactual_horizon = rollout_horizon
        G = np.array([self._efe_rollout(a, rollout_horizon)
                      for a in range(N_ACTIONS)])
        G_one_step = np.array([self._efe(a) for a in range(N_ACTIONS)])

        gamma_eff = self.gamma * np.exp(
            np.clip(self.affect_precision_gain * self._v_action_prev, -1.0, 1.0)
        )
        log_pi = -gamma_eff * G
        if self._habit_E is not None:
            log_pi = log_pi + self._habit_E
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

        action = int(self.rng.choice(N_ACTIONS, p=pi))
        counterfactual_regret = float(G[action] - np.min(G))
        self.prev_action = action
        self._pi_prev = pi.copy()
        self._v_action_prev = v_action

        # Policy entropy (decision uncertainty)
        pi_safe = np.maximum(pi, EPS)
        policy_entropy = float(-np.dot(pi_safe, np.log(pi_safe)))
        max_H_pi = np.log(N_ACTIONS)
        policy_entropy_norm = policy_entropy / max_H_pi if max_H_pi > 0 else 0.0

        info = dict(
            beliefs=q_post.copy(),
            q_pred=q_pred.copy(),
            G=G.copy(),
            G_one_step=G_one_step.copy(),
            pi=pi.copy(),
            counterfactual_regret=counterfactual_regret,
            counterfactual_horizon=float(rollout_horizon),
            gamma_eff=float(gamma_eff),
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
            pi_pos=self.pi_pos,
            pi_pos_eff=pi_pos_eff,
            intero_load=self._intero_vfe_ema,
            mood_beliefs=self.mood_beliefs.copy(),
        )
        return action, info

    def _update_vfe_scale(self, vfe):
        if self._vfe_ema is None:
            self._vfe_ema = vfe
            self._vfe_var_ema = 0.0
            return
        delta = vfe - self._vfe_ema
        self._vfe_ema += self._vfe_alpha * delta
        self._vfe_var_ema = (
            (1.0 - self._vfe_alpha) * self._vfe_var_ema
            + self._vfe_alpha * delta * delta
        )

    def _current_counterfactual_horizon(self, vfe):
        if not self.adaptive_counterfactual_horizon:
            return self.counterfactual_horizon
        scale = np.sqrt(max(self._vfe_var_ema or 0.0, EPS))
        z = (vfe - (self._vfe_ema if self._vfe_ema is not None else vfe)) / scale
        extra = int(np.clip(np.ceil(max(0.0, z)), 0, self.max_counterfactual_horizon))
        return int(min(self.max_counterfactual_horizon,
                       self.counterfactual_horizon + extra))

    # ── M5 hierarchical mood update ─────────────────────────
    def _mood_update(self):
        """Bayesian belief update at the mood (M5) level.

        Aggregates VFE over the last T_mood steps, bins the mean,
        and performs one step of prediction-update on the mood posterior.
        The A_mood matrix is calibrated per-model (K, M, omega_e) so that
        VFE observations are correctly mapped to mood states regardless
        of absolute VFE level.

        On the first mood window, an online offset correction aligns
        the analytical VFE curve with the agent's actual VFE.
        """
        mean_val = float(np.mean(self._val_buffer[-self.T_mood:]))

        # Predict (slow mood transition)
        q_pred = self.B_mood @ self.mood_beliefs
        q_pred = np.maximum(q_pred, EPS)
        q_pred /= q_pred.sum()

        # Update (Bayesian): the mood infers pi_pos from believed-valence level,
        # relative to neutral (0.5). Above neutral -> high pi_pos; below -> low.
        # Each pi_pos bin predicts a valence level spanning the plausible range,
        # so persistently low valence (chronic stress / anhedonia) drives the
        # posterior toward low pi_pos, which in turn biases RECALL/FUTURATE toward
        # negative targets -> a self-sustaining low-mood attractor (Beck's schema;
        # Eldar & Niv 2016 mood-as-reward-level).
        # The mood infers pi_pos from believed-valence level. Anchor: neutral
        # valence (0.5) corresponds to the sigmoid knee (pi_pos = 2), the point
        # below which RECALL turns to rumination; the slope spreads the observed
        # valence range across the pi_pos axis. So above-neutral valence -> high
        # pi_pos (resilience), and persistently below-neutral valence (chronic
        # stress in a vulnerable agent) -> pi_pos below the knee, where rumination
        # sustains a low-mood attractor (Beck's schema; Eldar & Niv 2016).
        pred_val = np.clip(0.5 + 0.02 * (MOOD_BIN_CENTERS - 2.0), 0.05, 0.95)
        lik = np.exp(-0.5 * ((mean_val - pred_val) / 0.06) ** 2)
        q_post = lik * q_pred
        q_post = np.maximum(q_post, EPS)
        q_post /= q_post.sum()
        self.mood_beliefs = q_post

        # Extract expected pi_pos
        self.pi_pos = float(np.dot(self.mood_beliefs, MOOD_BIN_CENTERS))

        # Rebuild lower-level B matrices with updated pi_pos
        self._rebuild_B()

        # Clear buffers for next cycle
        self._vfe_buffer = []
        self._val_buffer = []

    # ── Rebuild B matrices with current pi_pos ─────────────
    def _rebuild_B(self):
        for a in range(N_ACTIONS):
            if self.learn_B_frame:
                counts = self._bf_counts[a]
                Bf = counts / (counts.sum(axis=0, keepdims=True) + EPS)
                self.model.B[a] = rebuild_B_with_frame(
                    self.model, a, self.pi_pos, Bf,
                    valence_inertia=self.valence_inertia)
            else:
                self.model.B[a] = rebuild_B_single(
                    self.model, a, self.pi_pos,
                    valence_inertia=self.valence_inertia)

    # ── EFE for a single action ────────────────────────────
    def _efe(self, action):
        return self._efe_from_belief(action, self.beliefs)

    def _efe_one_step_legacy(self, action):
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

    def _efe_from_belief(self, action, belief):
        """G(a | q) = sum_m [risk_m + ambiguity_m] after one predicted step."""
        q_pred = self.model.B[action] @ belief
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

    def _efe_rollout(self, action, horizon, belief=None):
        """Counterfactual expected free energy over a short action rollout.

        This implements bounded mental time travel: each candidate action
        predicts the state it would lead to, then recursively evaluates the
        expected cost of subsequent actions from that imagined state.
        """
        if belief is None:
            belief = self.beliefs

        immediate = self._efe_from_belief(action, belief)
        if horizon <= 1:
            return immediate

        q_next = self.model.B[action] @ belief
        q_next = np.maximum(q_next, EPS)
        q_next /= q_next.sum()

        future_G = np.array([
            self._efe_rollout(a, horizon - 1, q_next)
            for a in range(N_ACTIONS)
        ])
        log_pi = -self.gamma * future_G
        if self._habit_E is not None:
            log_pi = log_pi + self._habit_E
        log_pi -= log_pi.max()
        future_pi = np.exp(log_pi)
        future_pi /= (future_pi.sum() + EPS)

        return float(immediate + self.counterfactual_discount *
                     np.dot(future_pi, future_G))
