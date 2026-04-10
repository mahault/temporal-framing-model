"""
Stochastic environment (true generative process).

The environment maintains a *true* hidden state and generates observations.
Its parameters are well-calibrated; mismatches between the agent's model
and the environment drive clinical pathology.

Key mechanism for mania:
  The environment depletes energy when the agent FUTURATEs.
  If the agent's A_int (interoceptive precision) is low, it cannot read
  the depletion signal, so it keeps FUTURATing until crash.
"""

import numpy as np
from generative_model import (
    flat_idx, N_EXT, N_INT, N_FRAMES,
    RECALL, ENGAGE, FUTURATE, REST,
    PAST, PRESENT, FUTURE,
    _gaussian_col,
)


class Environment:
    def __init__(self, K=8, M=5, volatility=0.3, seed=None):
        self.K = K
        self.M = M
        self.volatility = volatility
        self.rng = np.random.RandomState(seed)
        self.true_v = 0
        self.true_e = 0
        self.true_f = PRESENT

    def reset(self):
        """Sample a mildly positive, high-energy, present-oriented start."""
        self.true_v = int(self.K * 0.6)
        self.true_e = self.M - 2
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

        # --- valence ---
        v_deltas = {RECALL: +1, ENGAGE: 0, FUTURATE: +1, REST: 0}
        if action == REST:
            neutral = K // 2
            target = self.true_v + int(np.sign(neutral - self.true_v))
        else:
            target = self.true_v + v_deltas[action]
        target = int(np.clip(target, 0, K - 1))
        self.true_v = self._noisy_step(self.true_v, target, K)

        # --- energy (calibrated depletion / recovery) ---
        e_deltas = {RECALL: 0, ENGAGE: -1, FUTURATE: -1, REST: +1}
        e_target = int(np.clip(self.true_e + e_deltas[action], 0, M - 1))
        self.true_e = self._noisy_step(self.true_e, e_target, M, noise=0.2)

        # --- frame ---
        frame_targets = {RECALL: PAST, ENGAGE: PRESENT, FUTURATE: FUTURE, REST: PRESENT}
        if self.rng.random() < 0.7:
            self.true_f = frame_targets[action]

    def _apply_volatility(self):
        if self.rng.random() < self.volatility:
            delta = self.rng.choice([-1, 0, 1], p=[0.4, 0.3, 0.3])
            self.true_v = int(np.clip(self.true_v + delta, 0, self.K - 1))

    def _energy_consequences(self):
        """Very low energy drags valence down (the crash)."""
        if self.true_e <= 1:
            self.true_v = max(0, self.true_v - 1)

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
