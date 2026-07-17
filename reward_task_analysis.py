"""Reward-learning analysis of OpenNeuro ds005356 (MEG PST, MDD vs CTL).

NOTE ON LIMITATION: the dataset's behavioural logs were removed (CHANGES
v1.1.0), so the events encode only cue (which pair) + feedback (win/loss),
NOT the subject's choice. We therefore cannot fit choice-level asymmetric
reinforcement learning here. What we CAN measure is reward-learning
PERFORMANCE: on the probabilistic selection task, a subject who learns to
select the higher-reward stimulus wins more often, so per-subject win rate
(especially on the 80/20 'AB' pair and in the late phase) is a behavioural
index of reward sensitivity. We test whether this is blunted in MDD and
tracks clinical anhedonia (SHAPS/TEPS) -- the behavioural correlate of the
model's positive-hedonic sensitivity c_pos.

Fetches events via raw.githubusercontent (no API rate limit), caches locally.
"""
from __future__ import annotations

import csv, ssl, urllib.request, math
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parent
DS = ROOT / "data_raw" / "ds005356"
EVDIR = DS / "events"; EVDIR.mkdir(parents=True, exist_ok=True)
RAW = "https://raw.githubusercontent.com/OpenNeuroDatasets/ds005356/main"
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
        return r.read().decode("utf-8", "replace")


def get_events(bids_id):
    """Return list of (trial_type, value) across all runs; cache locally."""
    rows = []
    for run in (1, 2, 3):
        cache = EVDIR / f"{bids_id}_run{run}.tsv"
        if cache.exists():
            txt = cache.read_text(encoding="utf-8")
        else:
            url = f"{RAW}/{bids_id}/ses-01/meg/{bids_id}_ses-01_task-pst_run-{run}_events.tsv"
            try:
                txt = fetch(url)
            except Exception:
                continue
            if "\t" not in txt:
                continue
            cache.write_text(txt, encoding="utf-8")
        lines = [l.split("\t") for l in txt.splitlines() if l.strip()]
        for r in lines[1:]:
            if len(r) >= 3:
                rows.append(r[2])
    return rows


PAIR = {"AB": "AB", "BA": "AB", "CD": "CD", "DC": "CD", "EF": "EF", "FE": "EF"}


def subject_indices(trial_types):
    """Pair each feedback with the preceding cue; compute win-rate indices."""
    trials = []          # (pair, win?)
    cur_pair = None
    for tt in trial_types:
        if tt.startswith("cue/"):
            cur_pair = PAIR.get(tt.split("/")[1])
        elif tt.startswith("FB/") and cur_pair is not None:
            trials.append((cur_pair, 1 if tt.endswith("win") else 0))
            cur_pair = None
    if len(trials) < 30:
        return None
    wins = [w for _, w in trials]
    ab = [w for p, w in trials if p == "AB"]
    n = len(trials)
    early = wins[: n // 3]; late = wins[-(n // 3):]
    return dict(
        n_trials=n,
        win_overall=mean(wins),
        win_AB=mean(ab) if ab else float("nan"),
        win_late=mean(late),
        learning_gain=mean(late) - mean(early),
    )


def cohen_d(a, b):
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    sp = math.sqrt((pstdev(a) ** 2 + pstdev(b) ** 2) / 2) or 1e-9
    return (mean(a) - mean(b)) / sp


def pearson(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys)
           if isinstance(x, (int, float)) and isinstance(y, (int, float))
           and not (isinstance(x, float) and math.isnan(x))
           and not (isinstance(y, float) and math.isnan(y))]
    if len(pts) < 5:
        return float("nan"), 0
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return (num / den if den else float("nan")), len(pts)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    pheno = list(csv.DictReader((DS / "phenotype.csv").open(encoding="utf-8")))
    print(f"{len(pheno)} subjects in phenotype")
    recs = []
    for i, p in enumerate(pheno):
        idx = subject_indices(get_events(p["bids_id"]))
        if idx is None:
            continue
        rec = dict(p); rec.update(idx)
        recs.append(rec)
        if (i + 1) % 20 == 0:
            print(f"  fetched {i+1}/{len(pheno)}")
    print(f"{len(recs)} subjects with usable task data")

    ctl = [r for r in recs if r["Group"] == "CTL"]
    mdd = [r for r in recs if r["Group"] == "MDD"]
    print(f"\nCTL={len(ctl)}  MDD={len(mdd)}")

    print("\n=== Reward-learning performance by group (Cohen d = MDD-CTL) ===")
    print(f"{'index':<16}{'CTL':>8}{'MDD':>8}{'d':>8}")
    for k in ("win_overall", "win_AB", "win_late", "learning_gain"):
        c = [r[k] for r in ctl if not math.isnan(r[k])]
        m = [r[k] for r in mdd if not math.isnan(r[k])]
        print(f"{k:<16}{mean(c):>8.3f}{mean(m):>8.3f}{cohen_d(m, c):>8.2f}")

    print("\n=== Correlation: reward-learning vs clinical measures (all subj) ===")
    print("(negative r with SHAPS/anhedonia = worse learning -> more anhedonic)")
    print(f"{'behav x clinical':<34}{'r':>8}{'n':>5}")
    for bk in ("win_AB", "win_late", "win_overall"):
        for ck in ("SHAPS", "TEPS_Total", "TEPS_anticipatory", "BAS", "BDI"):
            r, n = pearson([rr[bk] for rr in recs], [_f(rr[ck]) for rr in recs])
            print(f"{bk+' x '+ck:<34}{r:>8.3f}{n:>5}")

    # save merged table
    out = DS / "reward_learning_merged.csv"
    keys = ["bids_id", "Group", "n_trials", "win_overall", "win_AB", "win_late",
            "learning_gain", "SHAPS", "TEPS_Total", "TEPS_anticipatory", "BAS", "BDI"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(keys)
        for r in recs:
            w.writerow([r.get(k) for k in keys])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
