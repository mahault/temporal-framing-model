"""Benjamini-Hochberg over every bootstrap contrast computed in the round-5 forecasting analysis.

Family: every line of out/score.md and out/contrasts_r6.md that reports a paired difference with a
95% participant-bootstrap interval (R2 and log-likelihood, every horizon, both samples). Two-sided p
values are approximated from the interval by a normal approximation, se = (hi - lo) / 3.92.
Writes out/bh_forecast.json and prints the reported contrasts.
"""
import json
import re
from math import erf, sqrt
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
pat = re.compile(r"^- (.*?): ([+-]?\d+\.\d+) \[([+-]?\d+\.\d+), ([+-]?\d+\.\d+)\]")
tests, seen = [], set()
for fn in ("score.md", "contrasts_r6.md", "contrasts_r6_kfvolh6.md"):
    sample = None
    for line in (OUT / fn).read_text().splitlines():
        if line.startswith("## "):
            sample = line[3:].strip()
        m = pat.match(line.strip())
        if not m:
            continue
        label, est, lo, hi = m.group(1), *map(float, m.groups()[1:])
        key = (sample, label)
        if key in seen:
            continue
        seen.add(key)
        se = (hi - lo) / 3.92
        if se <= 0:
            p = 0.0 if est != 0 else 1.0
        else:
            z = abs(est) / se
            p = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
        tests.append(dict(file=fn, sample=sample, label=label, est=est, lo=lo, hi=hi, p=p))
p = np.array([t["p"] for t in tests])
n = len(p)
order = np.argsort(p)
adj = np.empty(n)
prev = 1.0
for rank in range(n, 0, -1):
    i = order[rank - 1]
    prev = min(prev, p[i] * n / rank)
    adj[i] = prev
for t, a in zip(tests, adj):
    t["p_adj"] = float(a)
excl = [t for t in tests if t["lo"] > 0 or t["hi"] < 0]
lost = [t for t in excl if t["p_adj"] >= 0.05]
json.dump(dict(n_tests=n, interval_excludes_zero=len(excl), lost_after_bh=len(lost), tests=tests),
          open(OUT / "bh_forecast.json", "w"), indent=1)
print("n_tests", n, "excluding zero", len(excl), "lost after BH", len(lost))
for t in lost:
    print("LOST", t["sample"], t["label"], t["est"], t["lo"], t["hi"], round(t["p_adj"], 3))
