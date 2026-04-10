"""
Generative model for counterfactual temporal framing in bipolar disorder.

Factored hidden states:
  v: valence (K levels), e: energy (M levels), f: frame (3: PAST/PRESENT/FUTURE)

Observations:
  o_ext (3): environmental feedback
  o_int (3): interoceptive energy signal
  o_val (K): felt valence

Actions:
  RECALL(0), ENGAGE(1), FUTURATE(2), REST(3)

Clinical parameters:
  pi_pos:  precision on positive self-beliefs (controls D prior + RECALL pull)
  K:       valence granularity (number of discrete valence states)
  omega_e: energy estimation precision (controls A_int accuracy)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List

# ── Constants ──────────────────────────────────────────────
RECALL, ENGAGE, FUTURATE, REST = 0, 1, 2, 3
PAST, PRESENT, FUTURE = 0, 1, 2
N_ACTIONS = 4
N_FRAMES = 3
N_EXT = 3   # neg / neutral / pos
N_INT = 3   # depleted / neutral / energised
ACTION_NAMES = ['RECALL', 'ENGAGE', 'FUTURATE', 'REST']
FRAME_NAMES = ['PAST', 'PRESENT', 'FUTURE']
EPS = 1e-16


# ── Data container ─────────────────────────────────────────
@dataclass
class ModelSpec:
    A: List[np.ndarray]      # likelihood matrices per modality
    B: List[np.ndarray]      # transition matrices per action
    C: List[np.ndarray]      # preference vectors per modality
    D: np.ndarray            # prior over states
    K: int = 8
    M: int = 5
    n_states: int = 0
    n_obs: List[int] = field(default_factory=list)


# ── Index helpers ──────────────────────────────────────────
def flat_idx(v, e, f, M, F=N_FRAMES):
    """Flat index from factored (v, e, f)."""
    return v * (M * F) + e * F + f


def unflatten(idx, K, M, F=N_FRAMES):
    """Factored (v, e, f) from flat index."""
    f = idx % F
    remainder = idx // F
    e = remainder % M
    v = remainder // M
    return v, e, f


# ── Utility distributions ─────────────────────────────────
def _softmax(logits):
    x = logits - logits.max()
    p = np.exp(x)
    return p / (p.sum() + EPS)


def _gaussian_col(n, center, precision):
    """Discrete Gaussian-like distribution over n bins."""
    x = np.arange(n, dtype=float)
    logits = -precision * ((x - center) / max(n - 1, 1)) ** 2
    return _softmax(logits)


# ── Model builder ──────────────────────────────────────────
def build_model(K=8, M=5, pi_pos=5.0, omega_e=5.0, gamma=16.0):
    """
    Construct a full POMDP generative model.

    Parameters
    ----------
    K : int          – valence granularity (2–8)
    M : int          – energy levels
    pi_pos : float   – precision on positive self-beliefs
    omega_e : float  – interoceptive precision for energy
    gamma : float    – policy precision (inverse temperature)

    Returns
    -------
    ModelSpec
    """
    F = N_FRAMES
    n_s = K * M * F

    A = _build_A(K, M, F, n_s, omega_e)
    B = _build_B(K, M, F, n_s, pi_pos)
    C = _build_C(K)
    D = _build_D(K, M, F, n_s, pi_pos)

    return ModelSpec(A=A, B=B, C=C, D=D, K=K, M=M,
                     n_states=n_s, n_obs=[N_EXT, N_INT, K])


# ── A matrices (likelihood) ───────────────────────────────
def _build_A(K, M, F, n_s, omega_e):
    A = []

    # --- A_ext: P(o_ext | v, e) ---
    A_ext = np.zeros((N_EXT, n_s))
    for v in range(K):
        for e in range(M):
            v_n = v / max(K - 1, 1)
            e_n = e / max(M - 1, 1)
            positivity = 0.5 * v_n + 0.3 * e_n
            col = np.array([
                max(0.05, 0.6 - 0.5 * positivity),   # neg
                0.3,                                    # neutral
                max(0.05, 0.1 + 0.5 * positivity),    # pos
            ])
            col /= col.sum()
            for f in range(F):
                A_ext[:, flat_idx(v, e, f, M)] = col
    A.append(A_ext)

    # --- A_int: P(o_int | e) with precision omega_e ---
    A_int = np.zeros((N_INT, n_s))
    for e in range(M):
        e_n = e / max(M - 1, 1)
        # Three observations: depleted (0), neutral (1), energised (2)
        # High omega_e → peaked on correct bin; low → uniform
        logits = omega_e * np.array([
            (1.0 - e_n),      # depleted signal strength
            -abs(e_n - 0.5),   # neutral peaks at mid-energy
            e_n,               # energised signal strength
        ])
        col = _softmax(logits)
        for v in range(K):
            for f in range(F):
                A_int[:, flat_idx(v, e, f, M)] = col
    A.append(A_int)

    # --- A_val: P(o_val | v), near-identity ---
    A_val = np.zeros((K, n_s))
    for v in range(K):
        col = _gaussian_col(K, v, 8.0)
        for e in range(M):
            for f in range(F):
                A_val[:, flat_idx(v, e, f, M)] = col
    A.append(A_val)

    return A


# ── B matrices (transitions) ──────────────────────────────
def _build_B(K, M, F, n_s, pi_pos):
    B = []
    for a in range(N_ACTIONS):
        Bv = _B_valence(K, a, pi_pos)
        Be = _B_energy(M, a)
        Bf = _B_frame(a)
        B_full = np.kron(Bv, np.kron(Be, Bf))
        # Normalise columns
        B_full /= (B_full.sum(axis=0, keepdims=True) + EPS)
        B.append(B_full)
    return B


def _B_valence(K, action, pi_pos):
    """K x K valence transition matrix (cols = from, rows = to)."""
    B = np.zeros((K, K))
    alpha_recall = 1.0 / (1.0 + np.exp(-(pi_pos - 2.0)))  # sigmoid

    for v in range(K):
        if action == RECALL:
            target = (K - 1) * 0.8
            pull = _gaussian_col(K, target, 3.0)
            stay = _gaussian_col(K, v, 4.0)
            B[:, v] = alpha_recall * pull + (1 - alpha_recall) * stay

        elif action == ENGAGE:
            B[:, v] = _gaussian_col(K, v, 3.0)

        elif action == FUTURATE:
            solution = _gaussian_col(K, K - 1, 2.5)
            stay = _gaussian_col(K, v, 3.0)
            B[:, v] = 0.5 * solution + 0.5 * stay

        elif action == REST:
            neutral = (K - 1) / 2.0
            toward = _gaussian_col(K, neutral, 2.0)
            stay = _gaussian_col(K, v, 4.0)
            B[:, v] = 0.3 * toward + 0.7 * stay

    B /= (B.sum(axis=0, keepdims=True) + EPS)
    return B


def _B_energy(M, action):
    """M x M energy transition matrix."""
    # Slower depletion so manic phase lasts ~40-60 steps before crash
    deltas = {RECALL: 0.0, ENGAGE: -0.4, FUTURATE: -0.8, REST: 0.8}
    delta = deltas[action]
    B = np.zeros((M, M))
    for e in range(M):
        target = np.clip(e + delta, 0, M - 1)
        B[:, e] = _gaussian_col(M, target, 4.0)
    B /= (B.sum(axis=0, keepdims=True) + EPS)
    return B


def _B_frame(action):
    """3 x 3 temporal-frame transition matrix."""
    matrices = {
        RECALL: np.array([
            [0.70, 0.40, 0.30],   # → PAST
            [0.20, 0.45, 0.40],   # → PRESENT
            [0.10, 0.15, 0.30],   # → FUTURE
        ]),
        ENGAGE: np.array([
            [0.20, 0.10, 0.10],
            [0.65, 0.75, 0.60],
            [0.15, 0.15, 0.30],
        ]),
        FUTURATE: np.array([
            [0.05, 0.05, 0.02],
            [0.20, 0.20, 0.08],
            [0.75, 0.75, 0.90],
        ]),
        REST: np.array([
            [0.30, 0.20, 0.20],
            [0.50, 0.60, 0.50],
            [0.20, 0.20, 0.30],
        ]),
    }
    B = matrices[action].copy()
    B /= (B.sum(axis=0, keepdims=True) + EPS)
    return B


# ── C vectors (preferences) ───────────────────────────────
def _build_C(K):
    return [
        np.array([-2.0, 0.0, 1.5]),           # C_ext: prefer positive feedback
        np.array([-2.0, 0.0, 1.5]),           # C_int: prefer feeling energised
        np.linspace(-2.0, 1.5, K),            # C_val: prefer positive valence
    ]


# ── D vector (prior) ──────────────────────────────────────
def _build_D(K, M, F, n_s, pi_pos):
    D = np.zeros(n_s)
    for v in range(K):
        for e in range(M):
            for f in range(F):
                v_n = v / max(K - 1, 1)
                e_n = e / max(M - 1, 1)
                p_v = np.exp(pi_pos * (v_n - 0.5))
                p_e = np.exp(-3.0 * (e_n - 0.7) ** 2)
                p_f = [0.2, 0.6, 0.2][f]
                D[flat_idx(v, e, f, M)] = p_v * p_e * p_f
    D /= (D.sum() + EPS)
    return D
