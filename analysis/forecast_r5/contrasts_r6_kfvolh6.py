"""Extra contrasts for the paper (round 6 integration).

Same rows, folds and bootstrap as score.py. For the reported forecaster (two-timescale filter with
precision state and channels, kfx_all_h6) against every competitor: R2 differences at h = 1..6, and
held-out log-likelihood differences (own predictive variance for the filters, leave-fold-out residual
variance for regressions and the running mean). Writes out/contrasts_r6_kfvolh6.json and out/contrasts_r6_kfvolh6.md.
"""
import json
from collections import defaultdict

import numpy as np

from common import (HORIZONS, OUT, SAMPLES, boot_mean_diff, boot_r2_diff, common_keys, folds_of, lfo_var,
                    load, load_sample, load_v21_rows, nll_rows, r2_of, records)
from run_r5 import CANDIDATES
from score import BASELINES, V21_VARIANTS, load_r5

MODEL = "kfx_all_h6"
COMPS = ("kfvol_all_h6",)


def main():
    rep, md = {}, ["# Round-6 integration contrasts (contrasts_r6.py)", ""]
    for sample in SAMPLES:
        parts, has_event = load_sample(sample)
        pids = sorted(parts)
        fold = folds_of(pids)
        recs = records(parts, has_event)
        base = load(f"baselines_{sample}.pkl")
        oof, own = {}, {}
        for b in BASELINES + ("kfx_all_h1", "kfx_all_h6", "kfx0_all_h1"):
            if b in base["oof"]:
                oof[b] = base["oof"][b]
                if b in base["S"]:
                    own[b] = base["S"][b]
        for v in V21_VARIANTS:
            rows = load_v21_rows(sample, v, pids)
            if rows is not None:
                oof[f"v21_{v}"] = {h: {k: float(val[0][h - 1]) for k, val in rows.items()} for h in HORIZONS}
        for cfg in CANDIDATES:
            o = load_r5(sample, cfg, "outer")
            if o is not None:
                oof[f"r5_{cfg}"] = o["pred"]
        keys = {h: common_keys(recs, [oof[n] for n in oof], h) for h in HORIZONS}
        out = dict(r2={}, r2_diff=[], ll_diff=[])
        for n in (MODEL,) + COMPS:
            out["r2"][n] = {h: r2_of(recs, oof[n], h, keys[h])[0] for h in HORIZONS}
        for b in COMPS:
            for h in HORIZONS:
                pt, lo, hi = boot_r2_diff(recs, oof[MODEL], oof[b], h, keys[h])
                out["r2_diff"].append(dict(b=b, h=h, point=pt, lo=lo, hi=hi))
        # log-likelihood differences: nll_rows returns negative log-likelihood per row
        def rows_for(n):
            var = own.get(n) if n in own else {h: lfo_var(recs, oof[n], h, fold) for h in HORIZONS}
            kind = "own" if n in own else "lfo"
            return kind, {h: nll_rows(recs, oof[n], var, h, keys[h]) for h in HORIZONS}
        km, rm = rows_for(MODEL)
        for b in COMPS:
            kb, rb = rows_for(b)
            for h in HORIZONS:
                pt, lo, hi = boot_mean_diff(rb[h], rm[h])  # NLL_b - NLL_model = loglik_model - loglik_b
                out["ll_diff"].append(dict(b=f"{b}/{kb}", h=h, point=pt, lo=lo, hi=hi))
        rep[sample] = out
        md.append(f"## {sample}")
        md.append("R2: " + "; ".join(f"{n} h1 {out['r2'][n][1]:.3f} h6 {out['r2'][n][6]:.3f}" for n in out["r2"]))
        for c in out["r2_diff"]:
            md.append(f"- R2 h={c['h']} {MODEL} - {c['b']}: {c['point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]")
        for c in out["ll_diff"]:
            md.append(f"- loglik h={c['h']} {MODEL}(own) - {c['b']}: {c['point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]")
        md.append("")
    json.dump(rep, open(OUT / "contrasts_r6_kfvolh6.json", "w"), indent=1, default=float)
    open(OUT / "contrasts_r6_kfvolh6.md", "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
