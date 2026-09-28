"""Hierarchical fitting and forecasting evaluation of model v2 (2026-09-28).

Same participant folds, preprocessing, horizons, ridge baselines and bootstrap
as esm_eval_v3.py (round 3). Adds a kitchen ridge that also sees the target
beep's time of day (kitchen_tt), because v2's readout does.

Variants of v2 fitted per fold: gated (g=1), inert (g=0), clamped PAST /
PRESENT / FUTURE, no_level2, no_tod, no_hier, no_channels. Protocol P (pooled:
group parameters only on held-out participants) for every variant; protocol A
(delta_i adapted on the first segment of each held-out participant, scored on
the rest) for gated and inert, with the adapted kitchen ridge of round 3 as the
comparison.

Writes reviews/model_v2_forecast.md, .json, and per-row held-out arrays in
reviews/v2_rows_<sample>_<variant>.npz (used by variance_v2.py and worry_v2.py).
Run:  python fit_v2.py --workers 12 [--quick] [--iters 300]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import empirical_rebuild as er
from esm_eval_v3 import (HORIZONS, Ridge, add_time_features, bootstrap_diff, build_records, feats,
                         load_osf, pick_lambda, r2)
from model_v2 import ModelV2, adapt, fit_hierarchical, make_tensors

ROOT = Path(__file__).resolve().parent
VARIANTS = {
    "gated": dict(g=1.0), "inert": dict(g=0.0),
    "clampPAST": dict(g=1.0, clamp=0), "clampPRESENT": dict(g=1.0, clamp=1), "clampFUTURE": dict(g=1.0, clamp=2),
    "no_level2": dict(g=1.0, no_level2=True), "no_tod": dict(g=1.0, no_tod=True),
    "no_baseline": dict(g=1.0, no_baseline=True), "no_channels": dict(g=1.0, no_channels=True),
}
ADAPT_VARIANTS = ("gated", "inert")


def folds_of(pids):
    rng = np.random.RandomState(0)
    order = list(pids)
    rng.shuffle(order)
    return {p: k % 5 for k, p in enumerate(order)}


def split_adapt_mask(parts, pids, has_event, dt):
    """Target mask restricted to the adaptation segment (first period on
    Geschwind, first half of beeps on osf); eval mask for the rest."""
    N, T = dt["y"].shape
    ad = np.zeros((N, T), np.float32)
    for n, p in enumerate(pids):
        seq = parts[p]
        if has_event:
            segs = sorted({b.get("p") for b in seq})
            cut = next((i for i, b in enumerate(seq) if b.get("p") == segs[-1]), len(seq) // 2) if len(segs) == 2 else len(seq) // 2
        else:
            cut = len(seq) // 2
        ad[n, :cut] = 1.0
    return torch.tensor(ad)


def _fit_job(a):
    sample, k, variant, parts, pids, fold, has_event, iters, quick = a
    torch.set_num_threads(2)
    t0 = time.time()
    dt = make_tensors(parts, pids, has_event)
    tr = np.array([i for i, p in enumerate(pids) if fold[p] != k])
    te = np.array([i for i, p in enumerate(pids) if fold[p] == k])
    spec = VARIANTS[variant]
    model = ModelV2(len(pids), has_event, **spec)
    dtr = _sub(dt, tr)
    lam = fit_hierarchical(model, dtr, tr, iters=iters, seed=k)
    # pooled predictions on held-out participants (delta = 0 there)
    model.eval()
    with torch.no_grad():
        res = model(_sub(dt, te), torch.as_tensor(te))
    rows = _collect(res, _sub(dt, te), [pids[i] for i in te])
    out = dict(sample=sample, fold=k, variant=variant, lam=lam, secs=time.time() - t0,
               group={n: (float(v) if v.dim() == 0 else [float(x) for x in v]) for n, v in model.named_parameters() if n != "delta"}, rows=rows)
    if variant in ADAPT_VARIANTS and lam is not None:
        # protocol A: adapt delta on the first segment of each held-out participant
        dte = _sub(dt, te)
        ad = split_adapt_mask(parts, [pids[i] for i in te], has_event, dte)
        dad = dict(dte)
        dad["mt"] = {h: dte["mt"][h] * ad for h in dte["mt"]}
        adapt(model, dad, te, lam, iters=max(iters // 2, 50), seed=k + 200)
        with torch.no_grad():
            res_a = model(dte, torch.as_tensor(te))
        out["rows_A"] = _collect(res_a, dte, [pids[i] for i in te], eval_mask=(1 - ad))
    return out


def _sub(d, rows):
    from model_v2 import subset
    # keep participant indexing global: subset only slices tensors; idx passed separately
    return subset(d, rows)


def _collect(res, dte, pids_te, eval_mask=None):
    """Per-row held-out arrays: (pid, i, h) -> yhat, S, plus state features."""
    rows = {}
    mask = dte["mask"].numpy()
    em = eval_mask.numpy() if eval_mask is not None else np.ones_like(mask)
    for n, p in enumerate(pids_te):
        T = int(mask[n].sum())
        rows[p] = dict(
            i=np.arange(T), keep=em[n, :T].astype(bool),
            yhat=res["yhat"][n, :T].numpy(), S=res["S"][n, :T].numpy(), S1=res["S1"][n, :T].numpy(),
            eps=res["eps"][n, :T].numpy(), q=res["q"][n, :T].numpy(), x=res["x"][n, :T].numpy(),
            m=res["m"][n, :T].numpy(), b=res["b"][n, :T].numpy(), vB=res["vB"][n, :T].numpy(), vP=res["vP"][n, :T].numpy(),
            vF=res["vF"][n, :T].numpy(), rho=res["rho"][n, :T].numpy(), u=res["u"][n, :T].numpy())
    return rows


def ridge_oof(recs, fold, has_event, log):
    """Kitchen ridge as in round 3, plus kitchen_tt (target beep's time of day)."""
    oof = defaultdict(dict)
    train_var = defaultdict(dict)
    first = {}
    for k_, r in enumerate(recs):
        first.setdefault(r["pid"], k_)
    for k in range(5):
        tr = [r for r in recs if fold[r["pid"]] != k]
        te = [r for r in recs if fold[r["pid"]] == k]
        for h in HORIZONS:
            trh = [r for r in tr if r[f"y{h}"] is not None]
            teh = [r for r in te if r[f"y{h}"] is not None]
            ytr = np.array([r[f"y{h}"] for r in trh])
            gtr = [r["pid"] for r in trh]
            for kind in ("ar2", "kitchen", "kitchen_tt"):
                ff = (lambda r: feats(r, "kitchen") + [r[f"tt{h}"], r[f"tt{h}"] ** 2, r[f"ft{h}"]]) if kind == "kitchen_tt" \
                    else (lambda r: feats(r, kind))
                Xtr = np.array([ff(r) for r in trh])
                Xte = np.array([ff(r) for r in teh])
                lam = 0.0 if kind == "ar2" else pick_lambda(Xtr, ytr, gtr)
                mdl = Ridge(max(lam, 1e-6)).fit(Xtr, ytr)
                p = mdl.predict(Xte)
                train_var[(kind, h)][k] = float(np.mean((ytr - mdl.predict(Xtr)) ** 2)) + 1e-9
                for r, pv in zip(teh, p):
                    oof[(kind, h)][(r["pid"], r["i"])] = float(pv)
        log(f"  ridge fold {k} done")
    return oof, train_var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--samples", default="Geschwind,osf_83cfk")
    args = ap.parse_args()
    logf = open(ROOT / "reviews" / "fit_v2.log", "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    variants = args.variants.split(",")
    v1 = json.loads((ROOT / "reviews" / "esm_eval_v3.json").read_text(encoding="utf-8"))
    samples = []
    want = args.samples.split(",")
    g = er.load_participants()
    add_time_features(g, True)
    if args.quick:
        g = {p: g[p] for p in sorted(g)[:30]}
    if "Geschwind" in want:
        samples.append(("Geschwind", g, True))
    osf = load_osf()
    add_time_features(osf, False)
    if args.quick:
        osf = {p: osf[p] for p in sorted(osf)[:30]}
    if "osf_83cfk" in want:
        samples.append(("osf_83cfk", osf, False))

    out = {}
    lines = ["# Model v2 forecasting evaluation (2026-09-28)", "",
             "Script: `fit_v2.py`. Same participant folds as rounds 2 and 3 (seed-0 shuffle, five folds), "
             "horizons 1 to 6, participant bootstrap CIs (1000 resamples) on pooled R2 differences. "
             "Protocol P: group parameters only on held-out participants (the participant baseline is a latent state "
             "inferred online by the filter). Protocol A: participant deviations of the dynamics parameters adapted on "
             "the first segment of each held-out participant under the empirical-Bayes prior, scored on the rest. "
             "kitchen = round-3 ridge (six lags, events, time of day); kitchen_tt = kitchen plus the "
             "target beep's time of day, which v2's readout also uses. v1 = the discrete model of "
             "round 3 (numbers from esm_eval_v3.json, no CI against v2).", ""]
    for name, parts, has_event in samples:
        t0 = time.time()
        pids = sorted(parts)
        fold = folds_of(pids)
        recs = build_records(parts, has_event)
        for r in recs:      # target beep's time features
            seq = parts[r["pid"]]
            for h in HORIZONS:
                j = r["i"] + h
                ok = j < len(seq) and seq[j].get("p") == seq[r["i"]].get("p")
                r[f"tt{h}"] = seq[j].get("tod", 0.5) if ok else 0.5
                r[f"ft{h}"] = seq[j].get("first", 0.0) if ok else 0.0
        log(f"[{name}] {len(pids)} participants, {len(recs)} records; ridge baselines")
        oof, train_var = ridge_oof(recs, fold, has_event, log)
        jobs = [(name, k, v, parts, pids, fold, has_event, args.iters, args.quick) for k in range(5) for v in variants]
        log(f"[{name}] {len(jobs)} v2 fits")
        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                results = list(ex.map(_fit_job, jobs))
        else:
            results = [_fit_job(j) for j in jobs]
        log(f"[{name}] fits done ({time.time() - t0:.0f}s); mean fit {np.mean([r['secs'] for r in results]):.0f}s")
        # assemble OOF predictions
        rec_by = {(r["pid"], r["i"]): r for r in recs}
        for res in results:
            v = res["variant"]
            for proto, key in (("P", "rows"), ("A", "rows_A")):
                if key not in res:
                    continue
                for p, rw in res[key].items():
                    for i in rw["i"]:
                        if not rw["keep"][i]:
                            continue
                        for h in HORIZONS:
                            r = rec_by[(p, int(i))]
                            if r[f"y{h}"] is None:
                                continue
                            oof[(f"{proto}:v2_{v}", h)][(p, int(i))] = float(rw["yhat"][i, h - 1])
                            oof[(f"{proto}:v2_{v}_S", h)][(p, int(i))] = float(rw["S"][i, h - 1])
            # per-row arrays for later scripts (protocol P only)
            if res["fold"] == 0 or True:
                np.savez_compressed(ROOT / "reviews" / f"v2_rows_{name}_{v}_fold{res['fold']}.npz",
                                    **{f"{p}__{k2}": val for p, rw in res["rows"].items() for k2, val in rw.items()})
        # training-residual variance for the v2 variants (for the round-3 style NLL): from training folds' own rows
        # (approximate: use held-out residual variance pooled, reported separately as model-own S)

        def rows_for(nm, h, subset=None, change=False):
            by = defaultdict(list)
            d = oof.get((nm, h), {})
            for (p, i), pv in d.items():
                r = rec_by[(p, i)]
                if r[f"y{h}"] is None or (subset is not None and not subset(r)):
                    continue
                y, pr = r[f"y{h}"], pv
                if change:
                    y, pr = y - r["v"], pr - r["v"]
                by[p].append((y, pr))
            return {p: np.array(v, float) for p, v in by.items()}

        def score(nm, h, subset=None, change=False):
            by = rows_for(nm, h, subset, change)
            if not by:
                return float("nan"), 0
            arr = np.concatenate(list(by.values()))
            return r2(arr[:, 1], arr[:, 0]), len(arr)

        def ci(a, b, h, subset=None, change=False):
            A_, B_ = rows_for(a, h, subset, change), rows_for(b, h, subset, change)
            both = {p: np.column_stack([A_[p][:, 0], A_[p][:, 1], B_[p][:, 1]]) for p in A_ if p in B_ and len(A_[p]) == len(B_[p])}
            return bootstrap_diff(both, a, b)

        def nll_own(v, h):
            """Gaussian NLL with the model's own predictive variance."""
            d, dS = oof.get((f"P:v2_{v}", h), {}), oof.get((f"P:v2_{v}_S", h), {})
            vals = []
            for key, pv in d.items():
                r = rec_by[key]
                S = dS[key]
                vals.append(0.5 * np.log(2 * np.pi * S) + (r[f"y{h}"] - pv) ** 2 / (2 * S))
            return float(np.mean(vals)) if vals else float("nan")

        def nll_train_var(nm, h):
            """Round-3 style NLL: sigma^2 = pooled training residual variance of the same predictor
            (for v2 variants, the variance of the held-out residuals of the other folds is used as the
            stand-in, which is the closest available)."""
            d = oof.get((nm, h), {})
            if not d:
                return float("nan")
            res_ = np.array([rec_by[k][f"y{h}"] - pv for k, pv in d.items()])
            s2 = float(np.var(res_)) + 1e-9
            return float(np.mean(0.5 * np.log(2 * np.pi * s2) + res_ ** 2 / (2 * s2)))

        names = ["ar2", "kitchen", "kitchen_tt"] + [f"P:v2_{v}" for v in variants]
        v1t = v1[name]["tables"]["P/level/level"]
        tab = {}
        lines.append(f"## {name} (n={len(pids)} participants, {len(recs)} records)")
        lines.append("")
        lines.append("### Protocol P, valence level, held-out R2")
        lines.append("")
        lines.append("| predictor | " + " | ".join(f"h={h}" for h in HORIZONS) + " |")
        lines.append("|---|" + "---:|" * len(HORIZONS))
        for nm in ["v1 model_full", "v1 model_inert"]:
            key = nm.split()[1]
            lines.append(f"| {nm} (round 3) | " + " | ".join(f"{v1t[key][str(h)][0]:.3f}" for h in HORIZONS) + " |")
            tab[nm] = {h: v1t[key][str(h)][0] for h in HORIZONS}
        for nm in names:
            sc = {h: score(nm, h) for h in HORIZONS}
            tab[nm] = {h: sc[h][0] for h in HORIZONS}
            lines.append(f"| {nm} | " + " | ".join(f"{sc[h][0]:.3f}" for h in HORIZONS) + " |")
        lines.append(f"\nrows at h=1: {score('kitchen', 1)[1]} (ridge), {score('P:v2_gated', 1)[1]} (v2)\n")
        cis = []
        pairs = []
        for h in HORIZONS:
            pairs += [("P:v2_gated", "kitchen"), ("P:v2_gated", "kitchen_tt"), ("P:v2_inert", "kitchen_tt"),
                      ("P:v2_gated", "P:v2_inert"), ("P:v2_gated", "ar2"), ("kitchen_tt", "kitchen")]
            for v in variants:
                if v not in ("gated", "inert"):
                    pairs.append((f"P:v2_{v}", "P:v2_gated"))
                    pairs.append((f"P:v2_{v}", "kitchen_tt"))
            for a, b in pairs[-(6 + 2 * (len(variants) - 2)):]:
                if (a, h) not in oof or (b, h) not in oof:
                    continue
                d = ci(a, b, h)
                d.update(h=h, a=a, b=b)
                cis.append(d)
        lines.append("CIs (participant bootstrap on pooled R2 difference):")
        lines.append("")
        for d in cis:
            lines.append(f"- h={d['h']}: {d['a']} - {d['b']}: {d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
        lines.append("")
        lines.append("### Protocol P, held-out Gaussian NLL per row (lower is better)")
        lines.append("")
        lines.append("| predictor | variance | " + " | ".join(f"h={h}" for h in HORIZONS) + " |")
        lines.append("|---|---|" + "---:|" * len(HORIZONS))
        nll = {}
        for nm in ["kitchen", "kitchen_tt", "P:v2_gated", "P:v2_inert"]:
            nll[nm + " (pooled residual var)"] = {h: nll_train_var(nm, h) for h in HORIZONS}
            lines.append(f"| {nm} | pooled residual | " + " | ".join(f"{nll[nm + ' (pooled residual var)'][h]:.4f}" for h in HORIZONS) + " |")
        for v in ("gated", "inert"):
            nll[f"v2_{v} (own S)"] = {h: nll_own(v, h) for h in HORIZONS}
            lines.append(f"| v2_{v} | model's own predictive S | " + " | ".join(f"{nll[f'v2_{v} (own S)'][h]:.4f}" for h in HORIZONS) + " |")
        lines.append("")
        # protocol A
        if any((f"A:v2_{v}", 1) in oof for v in ADAPT_VARIANTS):
            lines.append("### Protocol A (adapted on the first segment, scored on the rest), held-out R2")
            lines.append("")
            lines.append("| predictor | " + " | ".join(f"h={h}" for h in HORIZONS) + " |")
            lines.append("|---|" + "---:|" * len(HORIZONS))
            v1a = v1[name]["tables"]["A/level/level"]
            for nm in ("A_kitchen_pooled", "A_kitchen", "A_model", "A_model_inert"):
                lines.append(f"| {nm} (round 3) | " + " | ".join(f"{v1a[nm][str(h)][0]:.3f}" for h in HORIZONS) + " |")
                tab[nm + " (round 3)"] = {h: v1a[nm][str(h)][0] for h in HORIZONS}
            # kitchen scored on the same eval rows as protocol A of v2
            keep_rows = set(oof[("A:v2_gated", 1)].keys()) if ("A:v2_gated", 1) in oof else set()
            for nm in ("kitchen", "kitchen_tt"):
                sub = lambda r, kr=keep_rows: (r["pid"], r["i"]) in kr
                sc = {h: score(nm, h, sub) for h in HORIZONS}
                tab[f"A-rows:{nm}"] = {h: sc[h][0] for h in HORIZONS}
                lines.append(f"| {nm} pooled, same eval rows | " + " | ".join(f"{sc[h][0]:.3f}" for h in HORIZONS) + " |")
            for v in ADAPT_VARIANTS:
                sc = {h: score(f"A:v2_{v}", h) for h in HORIZONS}
                tab[f"A:v2_{v}"] = {h: sc[h][0] for h in HORIZONS}
                lines.append(f"| A:v2_{v} | " + " | ".join(f"{sc[h][0]:.3f}" for h in HORIZONS) + " |")
            lines.append("")
            for h in HORIZONS:
                for a, b in (("A:v2_gated", "kitchen_tt"), ("A:v2_gated", "P:v2_gated"), ("A:v2_gated", "A:v2_inert")):
                    sub = lambda r, kr=keep_rows: (r["pid"], r["i"]) in kr
                    d = ci(a, b, h, sub)
                    d.update(h=h, a=a, b=b, proto="A")
                    cis.append(d)
                    lines.append(f"- h={h}: {a} - {b} (A rows): {d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
            lines.append("")
        if has_event:
            e_abs = np.array([abs(r["e"]) for r in recs if r["e"] != 0.0])
            q75 = float(np.percentile(e_abs, 75))
            lines.append("### After events (Geschwind), change in valence, protocol P")
            lines.append("")
            lines.append("| predictor | subset | " + " | ".join(f"h={h}" for h in HORIZONS) + " |")
            lines.append("|---|---|" + "---:|" * len(HORIZONS))
            for sname, sub in (("after_event", lambda r: r["e"] != 0.0), ("after_big_event", lambda r: abs(r["e"]) >= q75)):
                for nm in ("kitchen_tt", "P:v2_gated", "P:v2_inert"):
                    sc = {h: score(nm, h, sub, True) for h in HORIZONS}
                    tab[f"{sname}:{nm}"] = {h: sc[h][0] for h in HORIZONS}
                    lines.append(f"| {nm} | {sname} | " + " | ".join(f"{sc[h][0]:.3f}" for h in HORIZONS) + " |")
                for h in (1, 3, 6):
                    d = ci("P:v2_gated", "kitchen_tt", h, sub, True)
                    d.update(h=h, a="P:v2_gated", b="kitchen_tt", subset=sname, change=True)
                    cis.append(d)
                    lines.append(f"- h={h} {sname} change: v2_gated - kitchen_tt {d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
            lines.append("")
        groups = defaultdict(list)
        for res in results:
            groups[res["variant"]].append(res["group"])
        gp = {v: {k2: (float(np.mean([gg[k2] for gg in gl])) if not isinstance(gl[0][k2], list) else [float(x) for x in np.mean([gg[k2] for gg in gl], axis=0)]) for k2 in gl[0]} for v, gl in groups.items()}
        lines.append("Group parameters (unconstrained, mean over folds), gated: " +
                     ", ".join(f"{k2}={v2:.3f}" if isinstance(v2, float) else f"{k2}={v2}" for k2, v2 in gp.get("gated", {}).items()))
        lines.append("Empirical-Bayes lambda per fold (gated): " + ", ".join(f"{r['lam']:.1f}" for r in results if r["variant"] == "gated" and r["lam"] is not None))
        lines.append("")
        out[name] = dict(tables=tab, cis=cis, nll=nll, group=gp, n=len(pids), records=len(recs))
        log(f"[{name}] scoring done ({time.time() - t0:.0f}s)")
    suffix = ("_quick" if args.quick else "") + ("" if len(want) == 2 else "_" + want[0])
    (ROOT / "reviews" / f"model_v2_forecast{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "reviews" / f"model_v2_forecast{suffix}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    log(f"written reviews/model_v2_forecast{suffix}.md")


if __name__ == "__main__":
    main()
