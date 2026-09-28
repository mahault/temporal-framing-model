"""Does frame gating change behaviour? An intertemporal-choice task under the
gated expected free energy of the paper (round 2, 2026-09-27).

The ESM analyses cannot show a behavioural role for the frame because the
observation stream carries no orientation information and the agent only
tracks affect. This task can. At the choice point the agent picks IMMEDIATE
(a small reward now, nothing after) or DELAYED (nothing now, a larger reward
one step later). The gated EFE of the paper (main text Eq. 3) scores each
option as

    G(a) = w_present G_1(a) + w_future delta E_pi'[G_1(a' | a)] + w_past KL[q(s'|a) || D],

with the horizon precisions w_c = 1 + g (3 q(f = c) - 1) set by the frame
posterior. A future-dominant frame upweights the rollout term, where the
delayed reward lives; a present-dominant frame weights the immediate risk; a
past-dominant frame weights consistency with the identity prior D, which is
placed on the familiar (immediate) outcome. The script reports P(DELAYED) as a
function of the gain g under each clamped frame and under frames inferred from
the paper's own frame transitions after a RECALL, ENGAGE or FUTURATE step, and
the delayed/immediate reward ratio at which the agent is indifferent (the
implied discount) for each frame. Writes figures/fig_frame_choice.png and
reviews/frame_choice_results.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from generative_model import B_frame, RECALL, ENGAGE, FUTURATE

ROOT = Path(__file__).resolve().parent
EPS = 1e-16
GAMMA = 4.0   # softer than the paper's 16 so the curves are not step functions; the crossing ratios below do not depend on gamma
DELTA = 0.75
# observations: 0 none, 1 small reward, 2 large reward
# states: 0 start, 1 paid-small, 2 waiting, 3 paid-large, 4 done
IMMEDIATE, DELAYED = 0, 1


def prefs(c_small, c_large):
    C = np.array([0.0, c_small, c_large])
    p = np.exp(C - C.max())
    return p / p.sum()


def A():
    A_ = np.zeros((3, 5))
    A_[0, 0] = A_[0, 2] = A_[0, 4] = 1.0
    A_[1, 1] = 1.0
    A_[2, 3] = 1.0
    return A_


def B(a):
    B_ = np.zeros((5, 5))
    if a == IMMEDIATE:
        B_[1, 0] = 1.0      # start -> paid-small
    else:
        B_[2, 0] = 1.0      # start -> waiting
    B_[4, 1] = 1.0          # paid-small -> done
    B_[3, 2] = 1.0          # waiting -> paid-large
    B_[4, 3] = 1.0
    B_[4, 4] = 1.0
    return B_


def G1(a, q, p_pref):
    """One-step risk plus ambiguity (ambiguity is zero: deterministic A)."""
    qs = B(a) @ q
    qo = A() @ qs
    qo = np.maximum(qo, EPS)
    return float(np.dot(qo, np.log(qo) - np.log(p_pref + EPS)))


def gated_G(a, q, qf, g, p_pref, D):
    w = 1.0 + g * (3.0 * qf - 1.0)
    w = np.maximum(w, 0.0)
    imm = G1(a, q, p_pref)
    qs = B(a) @ q
    # one-step counterfactual rollout: both actions from the next state
    fut = np.array([G1(b, qs, p_pref) for b in (IMMEDIATE, DELAYED)])
    pi_f = np.exp(-GAMMA * (fut - fut.min()))
    pi_f /= pi_f.sum()
    future = DELTA * float(pi_f @ fut)
    qs_ = np.maximum(qs, EPS)
    retro = float(np.dot(qs_, np.log(qs_) - np.log(np.maximum(D, EPS))))
    return w[1] * imm + w[2] * future + w[0] * retro


def p_delayed(qf, g, c_small=1.0, c_large=2.0):
    q = np.array([1.0, 0, 0, 0, 0])
    D = np.array([0.05, 0.80, 0.05, 0.05, 0.05])     # identity prior on the familiar outcome
    p_pref = prefs(c_small, c_large)
    Gs = np.array([gated_G(a, q, qf, g, p_pref, D) for a in (IMMEDIATE, DELAYED)])
    pi = np.exp(-GAMMA * (Gs - Gs.min()))
    pi /= pi.sum()
    return float(pi[DELAYED])


def inferred_frames():
    """Frame posteriors after one framing step from the paper's prior [0.2, 0.6, 0.2]."""
    prior = np.array([0.2, 0.6, 0.2])
    return {"after RECALL": B_frame(RECALL) @ prior,
            "after ENGAGE": B_frame(ENGAGE) @ prior,
            "after FUTURATE": B_frame(FUTURATE) @ prior}


def indifference_ratio(qf, g):
    """Large/small reward ratio at which G(DELAYED) = G(IMMEDIATE), by bisection
    on c_large (independent of gamma). inf when the delayed option never wins
    below a 50:1 ratio, 1.0 when it already wins at parity."""
    lo, hi = 1.0, 50.0
    if p_delayed(qf, g, 1.0, hi) < 0.5:
        return float("inf")
    if p_delayed(qf, g, 1.0, lo) > 0.5:
        return 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if p_delayed(qf, g, 1.0, mid) < 0.5:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    gains = np.linspace(0, 1, 11)
    clamped = {"PAST": np.array([1.0, 0, 0]), "PRESENT": np.array([0, 1.0, 0]),
               "FUTURE": np.array([0, 0, 1.0])}
    inf = inferred_frames()
    res = {"gains": gains.tolist(), "clamped": {}, "inferred": {}, "indifference": {}}
    for nm, qf in clamped.items():
        res["clamped"][nm] = [p_delayed(qf, g) for g in gains]
        res["indifference"][nm] = {"g=0": indifference_ratio(qf, 0.0), "g=1": indifference_ratio(qf, 1.0)}
    for nm, qf in inf.items():
        res["inferred"][nm] = [p_delayed(qf, g) for g in gains]
        res["inferred"][nm + " q(f)"] = qf.tolist()
        res["indifference"][nm] = {"g=0": indifference_ratio(qf, 0.0), "g=1": indifference_ratio(qf, 1.0)}
    ratios = np.linspace(1.0, 8.0, 36)
    res["ratios"] = ratios.tolist()
    res["by_ratio"] = {nm: [p_delayed(qf, 1.0, 1.0, r) for r in ratios] for nm, qf in clamped.items()}
    res["by_ratio"]["g=0 (any frame)"] = [p_delayed(clamped["PRESENT"], 0.0, 1.0, r) for r in ratios]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    cols = {"PAST": "#D55E00", "PRESENT": "#009E73", "FUTURE": "#0072B2"}
    for nm, ys in res["clamped"].items():
        ax.plot(gains, ys, marker="o", color=cols[nm], label=f"frame clamped {nm}")
    for nm, ls in (("after RECALL", ":"), ("after ENGAGE", "-."), ("after FUTURATE", "--")):
        ax.plot(gains, res["inferred"][nm], ls=ls, color="gray", label=f"inferred q(f) {nm}")
    ax.set_xlabel("frame gain g")
    ax.set_ylabel("P(choose DELAYED)")
    ax.set_title("Reward ratio 2:1, delay one step", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    for nm in ("PAST", "PRESENT", "FUTURE"):
        ax.plot(ratios, res["by_ratio"][nm], color=cols[nm], label=f"g=1, {nm}")
    ax.plot(ratios, res["by_ratio"]["g=0 (any frame)"], color="k", ls="--", label="g=0 (frame inert)")
    ax.axhline(0.5, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("delayed / immediate reward ratio")
    ax.set_ylabel("P(choose DELAYED)")
    ax.set_title("Implied discounting by frame", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "fig_frame_choice.png", dpi=200, bbox_inches="tight")

    L = ["# Frame gating and intertemporal choice (2026-09-27)", "",
         "Script: `frame_choice_task.py`. Gated EFE of the main text, policy precision gamma 4 (the crossing ratios do not depend on it), delta 0.75, "
         "preferences (none 0, small 1, large 2 in log-preference units), identity prior D on the "
         "familiar immediate outcome.", "",
         "| frame | P(DELAYED) at g=0 | P(DELAYED) at g=1 | indifference ratio g=0 | indifference ratio g=1 |",
         "|---|---:|---:|---:|---:|"]
    for nm in ("PAST", "PRESENT", "FUTURE"):
        L.append(f"| clamped {nm} | {res['clamped'][nm][0]:.3f} | {res['clamped'][nm][-1]:.3f} | "
                 f"{res['indifference'][nm]['g=0']:.2f} | {res['indifference'][nm]['g=1']:.2f} |")
    for nm in ("after RECALL", "after ENGAGE", "after FUTURATE"):
        qf = res["inferred"][nm + " q(f)"]
        L.append(f"| inferred {nm} (q(f) = {qf[0]:.2f}, {qf[1]:.2f}, {qf[2]:.2f}) | "
                 f"{res['inferred'][nm][0]:.3f} | {res['inferred'][nm][-1]:.3f} | "
                 f"{res['indifference'][nm]['g=0']:.2f} | {res['indifference'][nm]['g=1']:.2f} |")
    L.append("")
    (ROOT / "reviews" / "frame_choice_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (ROOT / "reviews" / "frame_choice_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
