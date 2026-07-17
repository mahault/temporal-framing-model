"""Alternative validations of the model's positive-hedonic (c_pos) prediction.

These designs avoid what killed the cross-sectional test (noisy behaviour vs
noisy self-report, across people, restricted range):

A. WITHIN-PERSON reward reactivity (Geschwind ESM):
   The model's c_pos is mood-coupled, so within one person a pleasant event
   should lift valence LESS during low-mood stretches than high-mood stretches.
   Per person we fit the event->valence slope separately in their own
   low-mood vs high-mood beeps (mood = causal trailing average of valence, so
   the moderator is lagged, not the outcome). Prediction: slope_low < slope_high.

B. PROGNOSTIC (PRT osf.io/347rm):
   Does BASELINE reward sensitivity predict subsequent symptom improvement?
   (A future-outcome test is less circular than a same-time correlation, and
   PRT reward bias predicting antidepressant response is an established effect.)
   NOTE: this is the PLACEBO arm, so it tests placebo-response prediction.

All results are reported regardless of outcome.
"""
from __future__ import annotations
import csv, re, math
import numpy as np
from pathlib import Path
from empirical_rebuild import load_participants

D = Path("data_raw/prt_347rm")


def ols_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 5 or x.std() < 1e-9:
        return None
    return float(np.polyfit(x, y, 1)[0])


def pearson(pairs):
    n = len(pairs)
    if n < 5:
        return float("nan"), n
    xs = np.array([p[0] for p in pairs]); ys = np.array([p[1] for p in pairs])
    if xs.std() < 1e-9 or ys.std() < 1e-9:
        return float("nan"), n
    return float(np.corrcoef(xs, ys)[0, 1]), n


# ================= A: within-person reward reactivity =================
def validate_A():
    parts = load_participants()
    hi, lo = [], []
    n_dir = 0; n_used = 0
    for pid, seq in parts.items():
        idx = [i for i, b in enumerate(seq) if b["e"] is not None]
        if len(idx) < 24:
            continue
        v = np.array([seq[i]["v"] for i in idx])
        e = np.array([seq[i]["e"] for i in idx])
        # causal trailing mood = mean of previous up-to-5 valence values
        mood = np.full(len(idx), np.nan)
        for k in range(1, len(idx)):
            w = v[max(0, k - 5):k]
            mood[k] = w.mean()
        valid = ~np.isnan(mood)
        v, e, mood = v[valid], e[valid], mood[valid]
        if len(v) < 20:
            continue
        med = np.median(mood)
        himask = mood >= med; lomask = mood < med
        if himask.sum() < 6 or lomask.sum() < 6:
            continue
        sh = ols_slope(e[himask], v[himask]); sl = ols_slope(e[lomask], v[lomask])
        if sh is None or sl is None:
            continue
        hi.append(sh); lo.append(sl); n_used += 1
        if sl < sh:
            n_dir += 1
    hi = np.array(hi); lo = np.array(lo)
    diff = hi - lo
    # paired: mean diff, sd, t, and sign-test fraction
    t = diff.mean() / (diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    print("=== A. Within-person reward reactivity (Geschwind ESM) ===")
    print(f"  participants used: {n_used}")
    print(f"  mean event->valence slope, HIGH mood: {hi.mean():.4f}")
    print(f"  mean event->valence slope, LOW  mood: {lo.mean():.4f}")
    print(f"  mean(high-low) = {diff.mean():+.4f}  (paired t={t:.2f})")
    print(f"  predicted direction (slope_low < slope_high): {n_dir}/{n_used} "
          f"= {100*n_dir/n_used:.0f}%  (chance=50%)")


# ================= B: prognostic (PRT) =================
def _load_prt_reward():
    import openpyxl
    wb = openpyxl.load_workbook(D / "CompMod_FinalPlaceboDataset_1april23.xlsx",
                                read_only=True, data_only=True)
    beta = {}
    for r in wb.active.iter_rows(values_only=True):
        if r and isinstance(r[0], str):
            m = re.match(r"sigdet_REW1-(\d+)-(\d+)-output", r[0])
            if m and isinstance(r[1], (int, float)):
                beta[(int(m.group(1)), int(m.group(2)))] = float(r[1])
    rbias = {}
    with (D / "PRT_sumdata_10_06_2022.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                s = int(row["subject"]); sess = int(row["session"])
            except (ValueError, TypeError):
                continue
            vals = [float(row[f"b{b}_rbias"]) for b in (1, 2, 3)
                    if row.get(f"b{b}_rbias") not in (None, "", "NA")]
            if vals:
                rbias[(s, sess)] = sum(vals) / len(vals)
    return beta, rbias


def validate_B():
    beta, rbias = _load_prt_reward()
    with (D / "Placebo_ClinicalData_Final_5.9.23.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    prtid = {}; change = {}; responder = {}
    for r in rows:
        sid = r.get("subject_id"); p = r.get("PRT_SubjectID", "").strip()
        if p not in ("", "XXXXX"):
            try:
                prtid[sid] = int(float(p))
            except ValueError:
                pass
        for col, store in (("changehamd_pct_phase1", change),
                           ("responder_status_phase1", responder)):
            v = r.get(col, "").strip()
            if v not in ("",):
                store[sid] = v
    num_change = {}
    for sid, p in prtid.items():
        if sid in change:
            try:
                num_change[p] = float(change[sid])
            except ValueError:
                pass
    num_resp = {prtid[sid]: responder[sid] for sid in prtid if sid in responder}
    print("\n=== B. Prognostic: baseline reward sensitivity -> symptom change (PRT, placebo arm) ===")
    for label, sens in (("fitted beta", beta), ("response bias", rbias)):
        pairs = [(sens[(s, 1)], num_change[s]) for s in num_change if (s, 1) in sens]
        r, n = pearson(pairs)
        print(f"  {label} vs HAMD %change (phase1): r={r:+.3f}  n={n}")
        # responder vs non
        resp = [sens[(s, 1)] for s in num_resp if (s, 1) in sens and num_resp[s].lower().startswith(("r", "1", "y"))]
        nonr = [sens[(s, 1)] for s in num_resp if (s, 1) in sens and not num_resp[s].lower().startswith(("r", "1", "y"))]
        if len(resp) >= 4 and len(nonr) >= 4:
            sp = math.sqrt((np.std(resp) ** 2 + np.std(nonr) ** 2) / 2) or 1e-9
            d = (np.mean(resp) - np.mean(nonr)) / sp
            print(f"      responders n={len(resp)} mean={np.mean(resp):.3f} | "
                  f"non-responders n={len(nonr)} mean={np.mean(nonr):.3f} | d={d:+.2f}")
        # show distinct responder labels for transparency
    print("  responder label values:", sorted(set(num_resp.values())))


if __name__ == "__main__":
    validate_A()
    validate_B()
