"""
Generative model for counterfactual temporal framing in bipolar disorder.

Factored hidden states:
  v: valence (K levels), e: interoceptive load (M levels), f: frame (3: PAST/PRESENT/FUTURE)

Observations:
  o_ext (3): environmental feedback
  o_int (3): interoceptive load signal
  o_val (K): felt valence

Actions:
  RECALL(0), ENGAGE(1), FUTURATE(2), FEEL(3), BLANK(4)

  FEEL: active interoceptive processing — reduces accumulated prediction error
        (reframed from REST; Sandved-Smith et al. 2021, Stephan et al. 2016)
  BLANK: psychotic/dissociative null action — flat affect, present-locked,
         loss of personal historicity (Sterzer et al. 2018)

Clinical parameters:
  pi_pos:  precision on positive self-beliefs (controls D prior + RECALL pull)
  K:       valence granularity (number of discrete valence states)
  omega_e: interoceptive precision (controls A_int accuracy)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List

# ── Constants ──────────────────────────────────────────────
RECALL, ENGAGE, FUTURATE, FEEL, BLANK = 0, 1, 2, 3, 4
REST = FEEL   # backward compatibility alias
PAST, PRESENT, FUTURE = 0, 1, 2
N_ACTIONS = 5
N_FRAMES = 3
N_EXT = 3   # neg / neutral / pos
N_INT = 3   # depleted / neutral / energised
ACTION_NAMES = ['RECALL', 'ENGAGE', 'FUTURATE', 'FEEL', 'BLANK']
FRAME_NAMES = ['PAST', 'PRESENT', 'FUTURE']
EPS = 1e-16

# ── M5 mood-level constants (hierarchical POMDP) ──────────
N_MOOD = 8                                          # discretised pi_pos bins
MOOD_BIN_CENTERS = np.linspace(0.5, 7.5, N_MOOD)   # [0.5, 1.5, ..., 7.5]
N_OBS_MOOD = 5                                      # binned mean VFE
MOOD_OBS_EDGES = np.array([0.0, 3.6, 4.0, 4.3, 4.6, 20.0])


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
def build_model(K=8, M=5, pi_pos=5.0, omega_e=5.0, gamma=16.0, c_scale=1.0):
    """
    Construct a full POMDP generative model.

    Parameters
    ----------
    K : int          – valence granularity (2–8)
    M : int          – energy levels
    pi_pos : float   – precision on positive self-beliefs
    omega_e : float  – interoceptive precision for energy
    gamma : float    – policy precision (inverse temperature)
    c_scale : float  – reward sensitivity (1.0=normal, <1=anhedonic, >1=hypersensitive)

    Returns
    -------
    ModelSpec
    """
    F = N_FRAMES
    n_s = K * M * F

    A = _build_A(K, M, F, n_s, omega_e)
    B = _build_B(K, M, F, n_s, pi_pos)
    C = _build_C(K, c_scale)
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

    # --- A_val: P(o_val | v), precision scales with granularity ---
    A_val = np.zeros((K, n_s))
    val_precision = max(2.0, float(K))   # low K → blurred self-observation
    for v in range(K):
        col = _gaussian_col(K, v, val_precision)
        for e in range(M):
            for f in range(F):
                A_val[:, flat_idx(v, e, f, M)] = col
    A.append(A_val)

    return A


# ── B matrices (transitions) ──────────────────────────────
def _build_B(K, M, F, n_s, pi_pos):
    B = []
    for a in range(N_ACTIONS):
        Bv = B_valence(K, a, pi_pos)
        Be = B_energy(M, a)
        Bf = B_frame(a)
        B_full = np.kron(Bv, np.kron(Be, Bf))
        # Normalise columns
        B_full /= (B_full.sum(axis=0, keepdims=True) + EPS)
        B.append(B_full)
    return B


# ── Rebuild helpers ──────────────────────────────────────
def recall_alpha(pi_pos):
    """Sigmoid gating for RECALL effectiveness."""
    return 1.0 / (1.0 + np.exp(-(pi_pos - 2.0)))


def rebuild_B_single(model, action, pi_pos):
    """Rebuild one action's full B matrix with current pi_pos."""
    Bv = B_valence(model.K, action, pi_pos)
    Be = B_energy(model.M, action)
    Bf = B_frame(action)
    B_full = np.kron(Bv, np.kron(Be, Bf))
    B_full /= (B_full.sum(axis=0, keepdims=True) + EPS)
    return B_full


def rebuild_B_with_frame(model, action, pi_pos, Bf_learned):
    """Rebuild one action's B matrix with current pi_pos AND learned B_frame."""
    Bv = B_valence(model.K, action, pi_pos)
    Be = B_energy(model.M, action)
    B_full = np.kron(Bv, np.kron(Be, Bf_learned))
    B_full /= (B_full.sum(axis=0, keepdims=True) + EPS)
    return B_full


def B_valence(K, action, pi_pos):
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

        elif action == FEEL:
            neutral = (K - 1) / 2.0
            toward = _gaussian_col(K, neutral, 2.0)
            stay = _gaussian_col(K, v, 4.0)
            B[:, v] = 0.3 * toward + 0.7 * stay

        elif action == BLANK:
            # Flat affect: mostly stay, slight drift toward neutral
            stay = _gaussian_col(K, v, 5.0)
            neutral = _gaussian_col(K, (K - 1) / 2.0, 1.5)
            B[:, v] = 0.9 * stay + 0.1 * neutral

    B /= (B.sum(axis=0, keepdims=True) + EPS)
    return B


def B_energy(M, action):
    """M x M interoceptive load transition matrix.

    Reinterpreted as load accumulation: positive delta = load reduction (FEEL),
    negative delta = load increase (FUTURATE ignores body signals).
    """
    deltas = {RECALL: 0.0, ENGAGE: -0.5, FUTURATE: -1.2, FEEL: 1.2, BLANK: -0.3}
    delta = deltas[action]
    B = np.zeros((M, M))
    for e in range(M):
        target = np.clip(e + delta, 0, M - 1)
        B[:, e] = _gaussian_col(M, target, 4.0)
    B /= (B.sum(axis=0, keepdims=True) + EPS)
    return B


def B_frame(action):
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
        FEEL: np.array([
            [0.30, 0.20, 0.20],
            [0.50, 0.60, 0.50],
            [0.20, 0.20, 0.30],
        ]),
        BLANK: np.array([
            [0.05, 0.05, 0.05],   # → PAST  (historicity cut off)
            [0.90, 0.90, 0.85],   # → PRESENT (locked in)
            [0.05, 0.05, 0.10],   # → FUTURE (historicity cut off)
        ]),
    }
    B = matrices[action].copy()
    B /= (B.sum(axis=0, keepdims=True) + EPS)
    return B


# ── C vectors (preferences) ───────────────────────────────
def _build_C(K, c_scale=1.0):
    """Preference vectors scaled by reward sensitivity (c_scale).

    c_scale < 1 → anhedonia (flattened preferences, weak reward pursuit).
    c_scale > 1 → hypersensitive (exaggerated reward/punishment signals).
    """
    return [
        c_scale * np.array([-2.0, 0.0, 1.5]),      # C_ext
        c_scale * np.array([-2.0, 0.0, 1.5]),      # C_int
        c_scale * np.linspace(-2.0, 1.5, K),       # C_val
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


# ── M5 mood-level generative model (hierarchical POMDP) ──

def _mood_expected_vfe(pi_pos):
    """Expected mean VFE given mood state pi_pos.

    Higher pi_pos → better recall effectiveness → better model fit → lower VFE.
    Calibrated from simulation: VFE ≈ 4.14 + 0.66 * (1 - recall_alpha).
    """
    alpha = 1.0 / (1.0 + np.exp(-(pi_pos - 2.0)))
    return 4.14 + 0.66 * (1.0 - alpha)


def build_A_mood():
    """P(o_mood | mood_state): observation model for mood level.

    Maps mood states (discretised pi_pos) to expected binned VFE.
    Higher mood → expect LOWER VFE (better model fit).
    Lower mood → expect HIGHER VFE (more prediction errors).
    """
    obs_centers = np.array([1.8, 3.8, 4.15, 4.45, 12.3])  # bin centers
    sigma_A = 0.2  # tight to discriminate VFE range [4.14, 4.68]
    A = np.zeros((N_OBS_MOOD, N_MOOD))
    for j in range(N_MOOD):
        mu = _mood_expected_vfe(MOOD_BIN_CENTERS[j])
        for i in range(N_OBS_MOOD):
            A[i, j] = np.exp(-((obs_centers[i] - mu) ** 2) / (2 * sigma_A ** 2))
        A[:, j] = np.maximum(A[:, j], 0.01)
        A[:, j] /= A[:, j].sum()
    return A


def build_B_mood():
    """P(mood' | mood): slow transition dynamics for mood states.

    Very sticky (0.97 self-transition). Symmetric ±1 transitions — all
    asymmetry in mood drift comes from the VFE evidence, not built-in bias.
    """
    B = np.zeros((N_MOOD, N_MOOD))
    for j in range(N_MOOD):
        B[j, j] = 0.97
        if j > 0:
            B[j - 1, j] = 0.015
        if j < N_MOOD - 1:
            B[j + 1, j] = 0.015
    for j in range(N_MOOD):
        B[:, j] /= (B[:, j].sum() + EPS)
    return B


def build_D_mood(initial_pi_pos=5.0):
    """Prior over mood states, concentrated around initial pi_pos."""
    D = np.zeros(N_MOOD)
    for i in range(N_MOOD):
        D[i] = np.exp(-((MOOD_BIN_CENTERS[i] - initial_pi_pos) ** 2)
                       / (2 * 0.8 ** 2))
    D = np.maximum(D, EPS)
    D /= D.sum()
    return D
