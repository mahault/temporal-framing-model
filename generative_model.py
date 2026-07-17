"""
Generative model for counterfactual temporal framing in bipolar disorder.

Factored hidden states:
  v: valence (K levels), e: interoceptive load (M levels), f: frame (3: PAST/PRESENT/FUTURE)

Observations:
  o_ext (3): environmental feedback
  o_int (3): interoceptive load signal
  o_val (K): felt valence

Actions:
  RECALL(0), ENGAGE(1), FUTURATE(2), FEEL(3), DISSOCIATE(4), ABSTRACT(5)

  FEEL: active interoceptive processing — reduces accumulated prediction error
        (reframed from REST; Sandved-Smith et al. 2021, Stephan et al. 2016)
  DISSOCIATE: dissociative null action — flat affect, temporal drifting,
              unmoored from directed temporal processing (Vannikov-Lugassi
              & Soffer-Dudek 2018; Sterzer et al. 2018)
  ABSTRACT: effortful ungrounded cognition — coarse hedonic signal,
            future-pulling, couples with FUTURATE to form a second
            hedonic route bypassing interoception and episodic recall

Clinical parameters:
  pi_pos:  precision on positive self-beliefs (controls D prior + RECALL pull + recall target)
  K:       valence granularity (number of discrete valence states)
  omega_e: interoceptive precision (controls A_int accuracy)

RECALL is bidirectional: the pull target = (K-1) * (0.2 + 0.6*alpha), where
alpha = sigma(pi_pos - 2). High pi_pos -> positive past (narrative stabilisation);
low pi_pos -> negative past (rumination). Pull strength is also gated by alpha,
so rumination is weak at low pi_pos (EFE-optimal agents rarely RECALL when
depressive; habitual rumination requires an E vector, noted as future work).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List

# ── Constants ──────────────────────────────────────────────
RECALL, ENGAGE, FUTURATE, FEEL, DISSOCIATE, ABSTRACT = 0, 1, 2, 3, 4, 5
REST = FEEL        # backward compatibility alias
BLANK = DISSOCIATE  # backward compatibility alias
PAST, PRESENT, FUTURE = 0, 1, 2
N_ACTIONS = 6
N_FRAMES = 3
N_EXT = 3   # neg / neutral / pos
N_INT = 3   # depleted / neutral / energised
ACTION_NAMES = ['RECALL', 'ENGAGE', 'FUTURATE', 'FEEL', 'DISSOCIATE', 'ABSTRACT']
FRAME_NAMES = ['PAST', 'PRESENT', 'FUTURE']
EPS = 1e-16

# ── M5 mood-level constants (hierarchical POMDP) ──────────
N_MOOD = 8                                          # discretised pi_pos bins
MOOD_BIN_CENTERS = np.linspace(0.5, 7.5, N_MOOD)   # [0.5, 1.5, ..., 7.5]
N_OBS_MOOD = 5                                      # binned mean VFE
MOOD_OBS_EDGES = np.array([0.0, 4.80, 4.97, 5.13, 5.29, 20.0])


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
def build_model(K=8, M=5, pi_pos=5.0, omega_e=5.0, gamma=16.0, c_scale=1.0,
                c_pos=None, c_neg=None, neg_val_precision=1.0,
                valence_inertia=0.0):
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
    c_pos : float or None – positive-entry scaling for C vectors (defaults to c_scale)
    c_neg : float or None – negative-entry scaling for C vectors (defaults to c_scale)
    neg_val_precision : float – asymmetric A_val precision multiplier (1.0=symmetric)

    Returns
    -------
    ModelSpec
    """
    F = N_FRAMES
    n_s = K * M * F

    A = _build_A(K, M, F, n_s, omega_e, neg_val_precision=neg_val_precision)
    B = _build_B(K, M, F, n_s, pi_pos, valence_inertia=valence_inertia)
    C = _build_C(K, c_scale, c_pos=c_pos, c_neg=c_neg)
    D = _build_D(K, M, F, n_s, pi_pos)

    return ModelSpec(A=A, B=B, C=C, D=D, K=K, M=M,
                     n_states=n_s, n_obs=[N_EXT, N_INT, K])


# ── A matrices (likelihood) ───────────────────────────────
def _build_A(K, M, F, n_s, omega_e, neg_val_precision=1.0):
    """Build likelihood matrices.

    Parameters
    ----------
    neg_val_precision : float
        Asymmetric precision multiplier for A_val. Default 1.0 (symmetric).
        Values > 1 make negative-valence states sharper and positive-valence
        states blurrier, so RECALL toward negative past has lower ambiguity
        than FUTURATE toward positive future → rumination emerges from EFE.

        Interpolation across v ∈ [0, K-1]:
          midpoint = (K-1)/2
          v ≤ midpoint: eff = base_prec × neg_val_precision^(1 - v/midpoint)
          v > midpoint: eff = base_prec / neg_val_precision^((v-mid)/(K-1-mid))
    """
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

    # --- A_val: P(o_val | v), with asymmetric precision ---
    A_val = np.zeros((K, n_s))
    base_precision = max(2.0, float(K))   # low K → blurred self-observation
    midpoint = (K - 1) / 2.0

    for v in range(K):
        if neg_val_precision == 1.0:
            eff_precision = base_precision
        elif midpoint == 0:
            eff_precision = base_precision
        elif v <= midpoint:
            # Negative valence: sharper (higher precision)
            t = v / midpoint  # 0 at v=0, 1 at midpoint
            eff_precision = base_precision * (neg_val_precision ** (1.0 - t))
        else:
            # Positive valence: blurrier (lower precision)
            t = (v - midpoint) / (K - 1 - midpoint)  # 0 at midpoint, 1 at v=K-1
            eff_precision = base_precision / (neg_val_precision ** t)

        col = _gaussian_col(K, v, eff_precision)
        for e in range(M):
            for f in range(F):
                A_val[:, flat_idx(v, e, f, M)] = col
    A.append(A_val)

    return A


# ── B matrices (transitions) ──────────────────────────────
def _build_B(K, M, F, n_s, pi_pos, valence_inertia=0.0):
    B = []
    for a in range(N_ACTIONS):
        Bv = B_valence(K, a, pi_pos, valence_inertia=valence_inertia)
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


def rebuild_B_single(model, action, pi_pos, valence_inertia=0.0):
    """Rebuild one action's full B matrix with current pi_pos."""
    Bv = B_valence(model.K, action, pi_pos, valence_inertia=valence_inertia)
    Be = B_energy(model.M, action)
    Bf = B_frame(action)
    B_full = np.kron(Bv, np.kron(Be, Bf))
    B_full /= (B_full.sum(axis=0, keepdims=True) + EPS)
    return B_full


def rebuild_B_with_frame(model, action, pi_pos, Bf_learned, valence_inertia=0.0):
    """Rebuild one action's B matrix with current pi_pos AND learned B_frame."""
    Bv = B_valence(model.K, action, pi_pos, valence_inertia=valence_inertia)
    Be = B_energy(model.M, action)
    B_full = np.kron(Bv, np.kron(Be, Bf_learned))
    B_full /= (B_full.sum(axis=0, keepdims=True) + EPS)
    return B_full


def B_valence(K, action, pi_pos, valence_inertia=0.0):
    """K x K valence transition matrix (cols = from, rows = to)."""
    B = np.zeros((K, K))
    alpha_recall = 1.0 / (1.0 + np.exp(-(pi_pos - 2.0)))  # sigmoid

    for v in range(K):
        if action == RECALL:
            # Bidirectional RECALL: pull target depends on pi_pos via alpha
            #   High pi_pos (healthy): alpha~1 -> target~80% max (positive past)
            #   Low pi_pos (depressive): alpha~0.14 -> target~28% max (rumination)
            recall_valence = 0.2 + 0.6 * alpha_recall
            target = (K - 1) * recall_valence
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

        elif action == DISSOCIATE:
            # Flat affect: mostly stay, slight drift toward neutral
            stay = _gaussian_col(K, v, 5.0)
            neutral = _gaussian_col(K, (K - 1) / 2.0, 1.5)
            B[:, v] = 0.9 * stay + 0.1 * neutral

        elif action == ABSTRACT:
            # Moderate positive pull — "budget FUTURATE"
            # Pulls toward ~70% max valence (vs FUTURATE's 100%)
            # Tighter prediction than "coarse" would suggest — the agent
            # BELIEVES abstract thinking will moderately improve valence
            target_abstract = (K - 1) * 0.7
            pull = _gaussian_col(K, target_abstract, 3.0)
            stay = _gaussian_col(K, v, 5.0)
            B[:, v] = 0.35 * stay + 0.65 * pull

    inertia = float(np.clip(valence_inertia, 0.0, 0.95))
    if inertia > 0:
        B = inertia * np.eye(K) + (1.0 - inertia) * B
    B /= (B.sum(axis=0, keepdims=True) + EPS)
    return B


def B_energy(M, action):
    """M x M interoceptive load transition matrix.

    Reinterpreted as load accumulation: positive delta = load reduction (FEEL),
    negative delta = load increase (FUTURATE ignores body signals).

    FEEL has saturating returns: interoceptive processing reduces accumulated
    prediction error, but the effect diminishes as load approaches the
    homeostatic setpoint (allostatic regulation; Stephan et al. 2016).
    At high energy (low load), FEEL is effectively a no-op.

    RECALL has a small energy cost (-0.3): episodic reconstruction is
    effortful (Conway & Pleydell-Pearce 2000).

    ENGAGE is mildly restorative (+0.3): present-moment attention requires
    less model-maintenance than constructive temporal projection
    (Sandved-Smith et al. 2021).

    FUTURATE has moderate energy cost (-0.5): prospective simulation is
    metabolically expensive but produces short reactive planning bursts
    rather than sustained depletion at this cost level.
    """
    deltas = {RECALL: -0.3, ENGAGE: 0.3, FUTURATE: -0.5,
              DISSOCIATE: -0.3, ABSTRACT: 0.0}
    B = np.zeros((M, M))
    for e in range(M):
        if action == FEEL:
            # Saturating FEEL: effect proportional to available load
            # e=0 (max load) -> delta=+1.2; e=M-1 (min load) -> delta~0
            e_norm = e / max(M - 1, 1)
            delta = 1.2 * (1.0 - e_norm)
        else:
            delta = deltas[action]
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
        DISSOCIATE: np.array([
            [0.35, 0.30, 0.25],   # → PAST  (temporal drifting)
            [0.35, 0.40, 0.35],   # → PRESENT (slight default)
            [0.30, 0.30, 0.40],   # → FUTURE (temporal drifting)
        ]),
        ABSTRACT: np.array([
            [0.05, 0.05, 0.05],   # → PAST  (cut off from episodic grounding)
            [0.30, 0.35, 0.15],   # → PRESENT
            [0.65, 0.60, 0.80],   # → FUTURE (couples with FUTURATE)
        ]),
    }
    B = matrices[action].copy()
    B /= (B.sum(axis=0, keepdims=True) + EPS)
    return B


# ── C vectors (preferences) ───────────────────────────────
def _build_C(K, c_scale=1.0, c_pos=None, c_neg=None):
    """Preference vectors scaled by reward sensitivity.

    c_scale : float – symmetric reward sensitivity (1.0=normal).
    c_pos   : float or None – scaling for positive C entries (reward sensitivity).
              Defaults to c_scale if None.
    c_neg   : float or None – scaling for negative C entries (punishment sensitivity).
              Defaults to c_scale if None.

    When c_pos != c_neg the agent has asymmetric hedonic sensitivity:
      - Depression: c_pos=0.1 (anhedonia), c_neg=1.0 (preserved negative sensitivity)
        → agent feels bad observations but not good ones.
      - Mania: c_pos>1, c_neg<1 (opposite asymmetry, future work).

    C_int is NOT scaled: interoceptive preferences (body-budget maintenance)
    remain intact even in anhedonia. Supported by:
      - Treadway et al. (2009, 2012): effort-cost computation preserved in depression
      - Stephan et al. (2016): allostatic self-efficacy as separate system
      - Barrett (2017): body-budgeting independent of hedonic reward
    """
    cp = c_pos if c_pos is not None else c_scale
    cn = c_neg if c_neg is not None else c_scale

    def _asymmetric_scale(arr):
        """Element-wise: positive entries × c_pos, negative × c_neg, zero unchanged."""
        out = np.empty_like(arr)
        out[arr > 0] = arr[arr > 0] * cp
        out[arr < 0] = arr[arr < 0] * cn
        out[arr == 0] = 0.0
        return out

    C_ext_raw = np.array([-2.0, 0.0, 1.5])
    C_val_raw = np.linspace(-2.0, 1.5, K)

    return [
        _asymmetric_scale(C_ext_raw),                # C_ext (asymmetric scaling)
        np.array([-2.0, 0.0, 1.5]),                  # C_int (preserved: interoceptive
                                                      #   preferences intact even in anhedonia)
        _asymmetric_scale(C_val_raw),                 # C_val (asymmetric scaling)
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

def _compute_vfe_curve(K, M, omega_e, c_scale,
                       c_pos=None, c_neg=None, neg_val_precision=1.0):
    """Analytical VFE estimate at each mood bin center.

    Computes the prior-predictive ambiguity: E_D[Σ_m H[P(o|s)]].
    This captures K, omega_e, and pi_pos dependencies without simulation.
    The estimate represents the "irreducible" prediction error from
    observation noise given the model's structure and prior beliefs.

    Parameters
    ----------
    K, M : int
        Valence granularity and energy levels.
    omega_e : float
        Interoceptive precision.
    c_scale : float
        Reward sensitivity (affects policy → indirect VFE effect,
        captured by online calibration offset, not here).
    c_pos, c_neg : float or None
        Asymmetric C-vector scaling (forwarded to build_model).
    neg_val_precision : float
        Asymmetric A_val precision (forwarded to build_model).

    Returns
    -------
    vfe_curve : ndarray (N_MOOD,) — analytical VFE at each mood bin center
    """
    vfe_estimates = []
    for pp in MOOD_BIN_CENTERS:
        model = build_model(K=K, M=M, pi_pos=pp, omega_e=omega_e,
                            gamma=16.0, c_scale=c_scale,
                            c_pos=c_pos, c_neg=c_neg,
                            neg_val_precision=neg_val_precision)
        total = 0.0
        for Am in model.A:
            H_cols = -np.sum(Am * np.log(Am + EPS), axis=0)
            total += float(np.dot(model.D, H_cols))
        vfe_estimates.append(total)
    return np.array(vfe_estimates)


def _build_A_mood_from_vfe(vfe_at_mood):
    """Build A_mood observation matrix from expected VFE at each mood bin.

    Handles three regimes:
    - Normal (healthy): VFE decreases with pi_pos → standard mood inference
    - Flat (manic): VFE independent of pi_pos → mood layer inert
    - Inverted (depressive): VFE increases with pi_pos → reversed mapping

    Parameters
    ----------
    vfe_at_mood : ndarray (N_MOOD,) — expected VFE at each mood bin

    Returns
    -------
    A : ndarray (N_OBS_MOOD, N_MOOD) — mood observation likelihood
    obs_edges : ndarray (N_OBS_MOOD + 1,) — binning edges for VFE
    """
    vfe_min = vfe_at_mood.min()
    vfe_max = vfe_at_mood.max()
    vfe_range = vfe_max - vfe_min

    if vfe_range < 0.05:
        # VFE flat w.r.t. pi_pos — mood layer should be inert
        # Uniform A means all mood states equally likely given any VFE
        A = np.ones((N_OBS_MOOD, N_MOOD)) / N_OBS_MOOD
        mid = (vfe_min + vfe_max) / 2
        obs_edges = np.array([
            0.0, mid - 1.0, mid - 0.3, mid + 0.3, mid + 1.0, 20.0])
        return A, obs_edges

    # Observation centers spanning VFE range with 30% margin
    margin = vfe_range * 0.3
    obs_centers = np.linspace(vfe_min - margin, vfe_max + margin, N_OBS_MOOD)
    sigma_A = vfe_range / 4.0
    sigma_A = max(sigma_A, 0.05)

    A = np.zeros((N_OBS_MOOD, N_MOOD))
    for j in range(N_MOOD):
        mu = vfe_at_mood[j]
        for i in range(N_OBS_MOOD):
            A[i, j] = np.exp(-((obs_centers[i] - mu) ** 2)
                             / (2 * sigma_A ** 2))
        A[:, j] = np.maximum(A[:, j], 0.01)
        A[:, j] /= A[:, j].sum()

    # Observation bin edges (midpoints between consecutive centers)
    obs_edges = np.zeros(N_OBS_MOOD + 1)
    obs_edges[0] = 0.0
    obs_edges[-1] = 20.0
    for i in range(1, N_OBS_MOOD):
        obs_edges[i] = (obs_centers[i - 1] + obs_centers[i]) / 2.0

    return A, obs_edges


def build_A_mood():
    """Legacy A_mood for backward compatibility (K=8, omega_e=5.0)."""
    vfe_curve = _compute_vfe_curve(K=8, M=8, omega_e=5.0, c_scale=1.0)
    A, _ = _build_A_mood_from_vfe(vfe_curve)
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
