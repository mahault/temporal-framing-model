"""Benjamini-Hochberg false discovery rate within each family of tests reported in the Results.

Every estimate reported with a 95% interval in the Results sections and the appendix tables of
reviews/restructured/TF_restructured.tex is extracted. Duplicates within a family (the same triple
reported in the text and a table) are counted once. A normal approximation gives the two-sided p
value against zero: SE = (hi - lo) / 3.92, z = estimate / SE. Corrections listed in
reviews/CORRECTIONS_R5.md (the BDI table) are applied before testing. Benjamini-Hochberg at
q = 0.05 within each family. Estimates of levels that are not tests against zero (R2 values,
correlations of reliability, raw means) are excluded by the rule below and listed.

Usage: python analysis/gamble_r5/fdr.py
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
from math import erf, sqrt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEX = ROOT / "reviews/restructured/TF_restructured.tex"

FAMILIES = [  # (family, first line, last line), 1-based inclusive, from the section structure
    ("forecasting", 357, 425), ("orientation", 426, 495),
    ("channels: gamble and counterfactual", 496, 578), ("channels: ESM channel test", 579, 631),
    ("vulnerability: ESM", 632, 693), ("vulnerability: depression in the gamble", 694, 744),
    ("framing", 745, 842),
    ("forecasting", 1028, 1066), ("vulnerability: ESM", 1068, 1091),
    ("channels: gamble and counterfactual", 1093, 1118), ("framing", 1120, 1140),
]
# rows of the channel-test table: the fourth interval on each row is the valence covariate, not a test
COVARIATE_FOURTH = range(598, 606)
NUM = r"[+\-−]?\d*\.?\d+"
PAT = re.compile(rf"({NUM})\s*\[\s*({NUM})\s*,\s*({NUM})\s*\]")
PAT_SE = re.compile(rf"({NUM})\\?\s*\(\s*({NUM})\s*\)")
# results reported in prose as estimate and SE (or z), added by hand: (family, line, est, se, label)
MANUAL_SE = [
    ("orientation", 442, -0.06, 0.02, "valence per temporal-distance bin, future thought, Study 1"),
    ("orientation", 446, 0.07, 0.015, "orientation persistence, mCog"),
    ("orientation", 448, -0.03, 0.017, "orientation persistence, Study 1"),
    ("orientation", 435, -5.29, 1.0, "Mulholland within-person past slope (z)"),
    ("orientation", 436, -1.35, 1.0, "Mulholland within-person future slope (z)"),
]


def clean(s):
    s = s.replace("$", "").replace("\\,", "").replace("−", "-").replace("\\textbf{", "").replace("\\mathbf{", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    return s.replace("}", "").replace("{", "")


def pnorm2(z):
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def bh(p, q=0.05):
    p = np.asarray(p); n = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1)) / n
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    sig = np.zeros(n, bool); sig[order[:k]] = True
    adj = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    padj = np.empty(n); padj[order] = np.minimum(adj, 1)
    return sig, padj


def main():
    lines = TEX.read_text(encoding="utf8").splitlines()
    corr = json.loads((HERE / "out/fdr_overrides.json").read_text()) if (HERE / "out/fdr_overrides.json").exists() else {}
    fam_items = {}
    excluded = []
    for fam, a, b in FAMILIES:
        # join the block so intervals split across source lines are found; keep a char->line map
        text = ""; owner = []
        for ln in range(a, b + 1):
            c = clean(lines[ln - 1]) + " "
            text += c; owner += [ln] * len(c)
        seen_on_line = {}
        for m in PAT.finditer(text):
            ln = owner[m.start()]
            raw = lines[ln - 1]
            seen_on_line[ln] = seen_on_line.get(ln, 0) + 1
            if ln in COVARIATE_FOURTH and seen_on_line[ln] == 4:
                continue
            est, lo, hi = (float(x) for x in m.groups())
            key = f"{est}|{lo}|{hi}"
            if key in corr:
                est, lo, hi = corr[key]
            if hi <= lo:
                continue
            level = re.search(r"R\^?2|R2|reliab|split-half|Spearman|human|proportion|rate", raw) and lo > 0.1
            item = {"line": ln, "est": est, "lo": lo, "hi": hi, "text": raw.strip()[:110]}
            if level:
                excluded.append(item); continue
            se = (hi - lo) / 3.92
            item["p"] = pnorm2(est / se) if se > 0 else 1.0
            fam_items.setdefault(fam, {})
            fam_items[fam].setdefault(key, item)
        for m in PAT_SE.finditer(text):
            ln = owner[m.start()]
            est, se = (float(x) for x in m.groups())
            if se <= 0 or "&" not in lines[ln - 1]:
                continue
            key = f"{est}|se{se}"
            fam_items.setdefault(fam, {}).setdefault(key, {"line": ln, "est": est, "se": se, "text": lines[ln - 1].strip()[:110],
                                                             "p": pnorm2(est / se)})
    for fam, ln, est, se, lab in MANUAL_SE:
        fam_items.setdefault(fam, {}).setdefault(f"{est}|se{se}", {"line": ln, "est": est, "se": se, "text": lab, "p": pnorm2(est / se)})
    out = {"q": 0.05, "families": {}, "excluded_levels": excluded}
    for fam, d in fam_items.items():
        items = list(d.values())
        sig, padj = bh([i["p"] for i in items])
        raw_sig = sum(i["p"] < 0.05 for i in items)
        for i, s, pa in zip(items, sig, padj):
            i["bh_significant"] = bool(s); i["p_adj"] = float(pa)
        out["families"][fam] = {"n_tests": len(items), "nominal_p_below_05": int(raw_sig),
                                "survive_bh": int(sig.sum()),
                                "lost_after_bh": [i for i in items if i["p"] < 0.05 and not i["bh_significant"]],
                                "tests": items}
        print(f"{fam:40s} tests {len(items):3d}  nominal {raw_sig:3d}  survive BH {int(sig.sum()):3d}")
    (HERE / "out/fdr.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
