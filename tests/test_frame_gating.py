"""Unit tests for frame-gated precision (revision 2026-09-27).

Run:  python -m pytest tests/ -q   or   python tests/test_frame_gating.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import Agent                                   # noqa: E402
from generative_model import build_model, PAST, PRESENT, FUTURE, N_ACTIONS  # noqa: E402

OBS = [[1, 1, 4], [2, 2, 5], [0, 1, 3], [1, 1, 4], [2, 1, 6], [0, 0, 2]]


def _agent(frame_gain, frame_clamp=None, seed=0):
    model = build_model(K=8, M=5, pi_pos=3.0, omega_e=3.0)
    return Agent(model, pi_pos=3.0, omega_e=3.0, frame_gain=frame_gain,
                 frame_clamp=frame_clamp, counterfactual_horizon=2, seed=seed)


def test_gain_zero_matches_legacy_efe():
    """frame_gain = 0 must reproduce the pre-revision EFE exactly."""
    ag = _agent(0.0)
    for o in OBS:
        _, info = ag.step(o)
        legacy = np.array([ag._efe_rollout(a, ag._last_counterfactual_horizon)
                           for a in range(N_ACTIONS)])
        assert np.allclose(info["G"], legacy), (info["G"], legacy)
        assert np.allclose(info["frame_weights"], 1.0)


def test_gain_zero_valence_is_plain_sum():
    ag = _agent(0.0)
    for o in OBS:
        _, info = ag.step(o)
        expect = np.tanh(info["v_model"] + info["v_reward"] + info["v_action"])
        assert abs(info["valence"] - expect) < 1e-12


def test_frame_changes_efe_when_gated():
    """With gating on, the frame posterior alone changes G (same beliefs)."""
    a_past = _agent(1.0, frame_clamp=PAST)
    a_fut = _agent(1.0, frame_clamp=FUTURE)
    a_pres = _agent(1.0, frame_clamp=PRESENT)
    diffs = []
    for o in OBS:
        _, ip = a_past.step(o)
        _, ifu = a_fut.step(o)
        _, ipr = a_pres.step(o)
        diffs.append(np.abs(ip["G"] - ifu["G"]).max())
        assert np.allclose(ip["frame_weights"], [3, 0, 0])
        assert np.allclose(ifu["frame_weights"], [0, 0, 3])
        assert np.allclose(ipr["frame_weights"], [0, 3, 0])
    assert max(diffs) > 1e-3, diffs


def test_frame_changes_efe_without_clamp():
    """Ungated vs gated agent on identical observations: G must differ, and
    the gated agent's weights must track its own q(f)."""
    a0 = _agent(0.0)
    a1 = _agent(1.0)
    differs = False
    for o in OBS:
        _, i0 = a0.step(o)
        _, i1 = a1.step(o)
        qf = i1["beliefs"].reshape(8, 5, 3).sum(axis=(0, 1))
        w = 1.0 + (3.0 * qf - 1.0)
        assert np.allclose(i1["frame_weights"], w, atol=1e-9)
        if np.abs(i0["G"] - i1["G"]).max() > 1e-6:
            differs = True
    assert differs


def test_weights_sum_to_three():
    ag = _agent(0.7)
    for o in OBS:
        _, info = ag.step(o)
        assert abs(info["frame_weights"].sum() - 3.0) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
