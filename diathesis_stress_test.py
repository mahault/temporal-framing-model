"""Test the model's diathesis-stress prediction on real ESM data.

The mood layer predicts (a) vulnerability lowers baseline mood and (b) vulnerability
AMPLIFIES affective reactivity to stress -- a vulnerability x stress interaction, not
either main effect alone. We test this on the Geschwind-Bringmann ESM
(data_raw/geschwind_2013_s004.csv), which carries a baseline NEUROTICISM score
(vulnerability) alongside per-beep event pleasantness (stress) and momentary affect.
Neuroticism is never used in the model's affect-dynamics fitting, so this is a
held-out qualitative prediction.

Column layout (the file has a leading row-counter and an unlabelled trailing field):
  idx1 participant | idx6 cheerful | idx7 event pleasantness | idx8 worried
  idx9 fearful | idx10 sad | idx11 relaxed | idx12 neuroticism (person-constant)
"""
import csv
import numpy as np
from collections import defaultdict

PATH = "data_raw/geschwind_2013_s004.csv"


def _f(x):
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def main():
    data = list(csv.reader(open(PATH, encoding="latin-1")))[1:]
    P = defaultdict(list); neur = {}
    for r in data:
        pid = r[1]
        ch, ev, wo, fe, sa, rel, ne = (_f(r[6]), _f(r[7]), _f(r[8]),
                                       _f(r[9]), _f(r[10]), _f(r[11]), _f(r[12]))
        if None in (ch, ev, wo, fe, sa, rel):
            continue
        val = (ch + rel) - (wo + fe + sa)          # momentary valence composite
        P[pid].append((ev, val))                   # ev = event pleasantness (high = pleasant)
        if ne is not None:
            neur[pid] = ne

    slopes, ns, mv = [], [], []
    for pid, seq in P.items():
        if pid not in neur or len(seq) < 10:
            continue
        A = np.array(seq, float)
        x = A[:, 0] - A[:, 0].mean(); y = A[:, 1] - A[:, 1].mean()
        if x.std() < 1e-6:
            continue
        slopes.append(np.polyfit(x, y, 1)[0]); ns.append(neur[pid]); mv.append(A[:, 1].mean())
    slopes, ns, mv = np.array(slopes), np.array(ns), np.array(mv)

    print(f"n participants: {len(slopes)}")
    print(f"mean event->valence reactivity slope: {slopes.mean():+.3f}")
    print(f"corr(neuroticism, reactivity slope) = {np.corrcoef(ns, slopes)[0,1]:+.3f}  "
          f"(model predicts positive: vulnerability amplifies stress-reactivity)")
    med = np.median(ns)
    print(f"  low-neuroticism slope {slopes[ns<=med].mean():+.3f} vs high {slopes[ns>med].mean():+.3f}")
    print(f"corr(neuroticism, baseline mood) = {np.corrcoef(ns, mv)[0,1]:+.3f}  "
          f"(model predicts negative: vulnerability lowers baseline mood)")

    # pooled within-person cross-level interaction: valence_c ~ event_c + event_c:neuroticism
    gm = np.mean([neur[p] for p in P if p in neur])
    X, Y = [], []
    for pid, seq in P.items():
        if pid not in neur or len(seq) < 10:
            continue
        A = np.array(seq, float); xc = A[:, 0] - A[:, 0].mean(); yc = A[:, 1] - A[:, 1].mean()
        ni = neur[pid] - gm
        for k in range(len(A)):
            X.append([xc[k], xc[k] * ni]); Y.append(yc[k])
    X = np.c_[np.array(X), np.ones(len(X))]; Y = np.array(Y)
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ b; s2 = resid @ resid / (len(Y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    print(f"pooled interaction (event_c x neuroticism): b={b[1]:+.4f}, SE={se[1]:.4f}, "
          f"t={b[1]/se[1]:+.1f}  ({len(Y)} beeps)")


if __name__ == "__main__":
    main()
