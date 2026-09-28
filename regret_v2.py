"""Frame-sensitive test (b): regret switching with the retrospective channel gated
by q(PAST) from the v2 recogniser (2026-09-28).

Same protocol, data, models and folds as regret_participant_holdout.py (round 2),
restricted to factual, regret and regret_frame, with the frame weights
w_past[t] = 3 q_t(PAST) produced by model v2 (gated, Geschwind group parameters
from fit_v2.py) driven by the trial outcomes: y_t = 0.25 for a loss and 0.75
for a gain, e_t = the outcome (-1 / +1), as in the round-2 driver.

Writes reviews/regret_v2.md and .json.  Run: python regret_v2.py --workers 6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import regret_participant_holdout as rph
from model_v2 import ModelV2, drive_sequence, load_group

ROOT = Path(__file__).resolve().parent


def frame_weights_v2(d, workers=1):
    gp = json.loads((ROOT / "reviews" / "model_v2_forecast.json").read_text(encoding="utf-8"))["Geschwind"]["group"]["gated"]
    model = load_group(ModelV2(1, True, g=1.0), gp)
    n, T = d["n"], d["T"]
    W = np.ones((n, T))
    for i in range(n):
        valid = d["valid"][i]
        o = d["out"][i]
        y = np.where(o < 0, 0.25, 0.75).astype(np.float32)
        e = o.astype(np.float32)
        # invalid trials: hold the observation (no update would need a mask; keep it simple and rare)
        res = drive_sequence(model, y, e=e)
        q_past = res["q"][:, 0]
        for t in range(T - 1):
            W[i, t + 1] = 3.0 * q_past[t] if valid[t] else W[i, t]
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    rph.frame_weights = frame_weights_v2
    rph.MODELS = ("factual", "regret", "regret_frame")
    lines = ["# Regret mechanism with v2 frame weights, participant-level holdout (2026-09-28)", "",
             "Script: `regret_v2.py`. Identical to round 2 (`regret_participant_holdout.py`) except that "
             "w_past = 3 q(PAST) comes from model v2 (gated, Geschwind group parameters) driven by the outcomes.", ""]
    results = []
    for name, mf, keys in rph.DATASETS:
        o = rph.analyse(name, mf, keys, args.workers)
        results.append(o)
        lines += rph.fmt(o)
        print("\n".join(rph.fmt(o)))
    (ROOT / "reviews" / "regret_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "reviews" / "regret_v2.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
