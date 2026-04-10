"""
Stochastic environment (true generative process).

The environment maintains a *true* hidden state and generates observations.
Clinical parameters modulate action effectiveness:
  - pi_pos → recall effectiveness (low = impaired positive memory access)
  - c_scale → reward sensitivity (low = anhedonia, even imagining future
    doesn't produce positive affect)
  - omega_e only affects the AGENT's model, not the environment
    (energy depletion is objectively real; the manic agent just can't read it)

Key mechanism:
  Depressive trap: low pi_pos → RECALL fails → low c_scale → FUTURATE
  doesn't help either → valence drifts down from volatility → stuck.
  Manic cycle: high c_scale → aggressive FUTURATE → energy depletes →
  homeostatic recovery → cycle.
"""

import numpy as np
from generative_model import (
    flat_idx, N_EXT, N_INT, N_FRAMES,
    RECALL, ENGAGE, FUTURATE, REST,
    PAST, PRESENT, FUTURE,
    _gaussian_col,
)


class Environment:
    def __init__(self, K=8, M=8, volatility=0.3, seed=None,
                 pi_pos=5.0, c_scale=1.0, **_ignored):
        self.K = K
        self.M = M
        self.volatility = volatility
        self.rng = np.random.RandomState(seed)
        self.true_v = 0
        self.true_e = 0
        self.true_f = PRESENT

        # Biologically-grounded action effectiveness
        self.recall_eff = 1.0 / (1.0 + np.exp(-(pi_pos - 2.0)))
        self.futurate_eff = min(1.0, c_scale)

    def reset(self):
        """Neutral start — pathology emerges from agent parameters."""
        self.true_v = (self.K - 1) // 2    # neutral valence
        self.true_e = self.M - 2            # high energy
        self.true_f = PRESENT
        return self._observe()

    # ── Step ───────────────────────────────────────────────
    def step(self, action):
        self._transition(action)
        self._apply_volatility()
        self._energy_consequences()
        obs = self._observe()
        info = dict(true_v=self.true_v, true_e=self.true_e, true_f=self.true_f)
        return obs, info

    # ── True transitions ───────────────────────────────────
    def _transition(self, action):
        K, M = self.K, self.M

        # --- valence (effectiveness modulated by clinical parameters) ---
        if action == RECALL:
            if self.rng.random() < self.recall_eff:
                target = self.true_v + 1
            else:
                target = self.true_v  # recall failed
        elif action == FUTURATE:
            if self.rng.random() < self.futurate_eff:
                target = self.true_v + 1
            else:
                target = self.true_v  # anhedonic — can't feel future reward
        elif action == REST:
            neutral = K // 2
            target = self.true_v + int(np.sign(neutral - self.true_v))
        else:  # ENGAGE
            target = self.true_v
        target = int(np.clip(target, 0, K - 1))
        self.true_v = self._noisy_step(self.true_v, target, K)

        # --- energy ---
        e_deltas = {RECALL: 0, ENGAGE: 0, FUTURATE: -1, REST: +2}
        e_target = int(np.clip(self.true_e + e_deltas[action], 0, M - 1))

        # Background metabolic cost: all non-REST actions slowly drain
        if action != REST and self.rng.random() < 0.2:
            e_target = max(0, e_target - 1)

        # Homeostatic recovery: energy drifts toward baseline when very low
        baseline = M // 3
        if self.true_e < baseline and e_target <= self.true_e:
            e_target = min(self.true_e + 1, M - 1)

        self.true_e = self._noisy_step(self.true_e, e_target, M, noise=0.2)

        # --- frame ---
        frame_targets = {RECALL: PAST, ENGAGE: PRESENT,
                         FUTURATE: FUTURE, REST: PRESENT}
        if self.rng.random() < 0.7:
            self.true_f = frame_targets[action]

    def _apply_volatility(self):
        """Environmental volatility with mild negative bias (life is hard)."""
        if self.rng.random() < self.volatility:
            delta = self.rng.choice([-1, 0, 1], p=[0.50, 0.25, 0.25])
            self.true_v = int(np.clip(self.true_v + delta, 0, self.K - 1))

    def _energy_consequences(self):
        """Bidirectional coupling: low energy → valence drop;
        low valence → energy drain (motivational fatigue / anhedonia)."""
        if self.true_e <= 1 and self.rng.random() < 0.5:
            self.true_v = max(0, self.true_v - 1)
        if self.true_v <= 1 and self.rng.random() < 0.25:
            self.true_e = max(0, self.true_e - 1)

    def _noisy_step(self, current, target, n, noise=0.3):
        if self.rng.random() < noise:
            delta = self.rng.choice([-1, 0, 1])
            return int(np.clip(current + delta, 0, n - 1))
        return int(np.clip(target, 0, n - 1))

    # ── Observation generation ─────────────────────────────
    def _observe(self):
        v, e = self.true_v, self.true_e
        K, M = self.K, self.M

        # o_ext — depends on valence + energy
        v_n = v / max(K - 1, 1)
        e_n = e / max(M - 1, 1)
        pos = 0.5 * v_n + 0.3 * e_n
        p_ext = np.array([max(0.05, 0.6 - 0.5 * pos),
                          0.3,
                          max(0.05, 0.1 + 0.5 * pos)])
        p_ext /= p_ext.sum()
        o_ext = int(self.rng.choice(N_EXT, p=p_ext))

        # o_int — high precision (truth is available)
        if e <= 1:
            p_int = np.array([0.8, 0.15, 0.05])
        elif e >= M - 2:
            p_int = np.array([0.05, 0.15, 0.8])
        else:
            p_int = np.array([0.15, 0.7, 0.15])
        o_int = int(self.rng.choice(N_INT, p=p_int))

        # o_val — fairly accurate
        p_val = _gaussian_col(K, v, 6.0)
        o_val = int(self.rng.choice(K, p=p_val))

        return [o_ext, o_int, o_val]
