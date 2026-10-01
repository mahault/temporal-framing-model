"""Clinical predictions of model v2.1 tested on existing ESM datasets (2026-09-30).

Predictions are fixed in reviews/DIATHESIS_DATA_TEST.md (written before this ran).

Datasets with a person-level vulnerability measure and ESM affect sequences:
  Geschwind  neuroticism; events (pleasantness); worry item          (repo loader)
  Kane 2017  NEO-FFI neuroticism; situation stressful/positive; MW-worry item
  Gainey     BFI neuroticism, RRS brooding, IDAS dysphoria; brooding item; no events

For each dataset: fit v2.1 (gated, the paper's model) to all participants (group
parameters, then per-person deviations by empirical Bayes with the group frozen),
extract per-person parameters with Laplace posterior SDs, compute the model-free
equivalents, relate both to the trait scores, estimate split-half reliability
(first vs second half of each person's signals, refitting the per-person
deviations only), test the diathesis-stress interaction, and pool neuroticism
effects across datasets by DerSimonian-Laird random effects.

Run (from the repo root):  python analysis/diathesis/diathesis_data.py [--iters 300] [--quick]
Outputs: analysis/diathesis/out/*.json (participant level, git-ignored),
         analysis/diathesis/out/summary.json (aggregate), figures/diathesis_*.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import empirical_rebuild as er                     # noqa: E402
from esm_eval_v3 import add_time_features          # noqa: E402
from model_v2 import HIER, ModelV2, fit, make_tensors, subset  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
CAND = ROOT / "data_raw" / "candidates"
MIN_BEEPS = 12
H = 3
LOGF = open(OUT / "run.log", "a", encoding="utf-8")


def log(s):
    print(s, flush=True)
    LOGF.write(s + "\n")
    LOGF.flush()


def _num(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


# ── loaders: {pid: [beep dicts]}, {pid: {trait: value}} ────────────────
def load_geschwind():
    parts = er.load_participants()
    add_time_features(parts, True)
    traits = {}
    for p, seq in parts.items():
        for b in seq:
            e = b.get("e")
            b["neg"] = max(0.0, -e) if e is not None else None
            b["seg"] = (b.get("p"), b.get("d"))          # same-day pairs for model-free lags
        n = seq[0].get("n")
        if n is not None:
            traits[p] = dict(neuroticism=n)
    return parts, traits, True, "worry item (1-7)"


def load_kane():
    k = pd.read_csv(CAND / "kane2017_esm" / "Kane_ESM_L1.csv", na_values=[" ", ""], skipinitialspace=True)
    L2 = pd.read_csv(CAND / "kane2017_esm" / "Kane_ESM_and_NEO_L2.csv", na_values=[" ", ""], skipinitialspace=True)
    traits = {}
    for _, r in L2.iterrows():
        n = _num(r["N"])
        if n is not None:
            traits[str(int(r["subjnumb"]))] = dict(neuroticism=n)
    parts = {}
    for pid, d in k.groupby("subjnumb", sort=False):
        seq = []
        for _, r in d.iterrows():
            hap = _num(r["esm16"])
            neg = [_num(r[c]) for c in ("esm18", "esm20", "esm22")]
            neg = [x for x in neg if x is not None]
            if hap is None or not neg:
                continue
            v = float(np.clip((hap - np.mean(neg) + 6.0) / 12.0, 0.0, 1.0))
            st, po = _num(r["esm34"]), _num(r["esm35"])
            e = (po - st) if (st is not None and po is not None) else None
            mw = _num(r["esm01"])
            w = _num(r["esm04"]) if mw == 1 else (1.0 if mw == 2 else None)
            seq.append(dict(v=v, e=e, neg=(st - 1.0) if st is not None else None, w=w, p=0, d=0,
                            tod=0.5, first=0.0, seg=0))
        if len(seq) >= MIN_BEEPS:
            parts[str(int(pid))] = seq
    return parts, traits, True, "mind-wandering to worries (1-7; 1 when the mind had not wandered)"


def load_gainey():
    g = pd.read_csv(CAND / "openesm_0058_gainey" / "0058_gainey_ts.tsv", sep="\t")
    st = pd.read_csv(CAND / "openesm_0058_gainey" / "0058_gainey_static.tsv", sep="\t", low_memory=False)
    st = st.groupby("ID").first()
    traits = {}
    for pid, r in st.iterrows():
        t = dict(neuroticism=_num(r["BFI_N"]), brooding_trait=_num(r["RRS"]), dysphoria=_num(r["IDASDys10"]))
        t = {k: v for k, v in t.items() if v is not None}
        if t:
            traits[str(int(pid))] = t
    pos_c = ("active", "interested", "excited", "strong")
    neg_c = ("irritable", "upset", "afraid_anxious", "sad")
    parts = {}
    for pid, d in g.sort_values(["id", "day", "beep"]).groupby("id"):
        seq = []
        for _, r in d.iterrows():
            pos = [_num(r[c]) for c in pos_c]
            neg = [_num(r[c]) for c in neg_c]
            pos = [x for x in pos if x is not None]
            neg = [x for x in neg if x is not None]
            if not pos or not neg:
                continue
            v = float(np.clip((np.mean(pos) - np.mean(neg) + 4.0) / 8.0, 0.0, 1.0))
            seq.append(dict(v=v, e=None, neg=None, w=_num(r["brooding"]), p=0, d=int(r["day"]),
                            seg=int(r["day"])))
        if len(seq) >= MIN_BEEPS:
            parts[str(int(pid))] = seq
    add_time_features(parts, True)     # rank within day
    return parts, traits, False, "brooding item (1-5)"


LOADERS = dict(Geschwind=load_geschwind, Kane=load_kane, Gainey=load_gainey)


# ── fitting ─────────────────────────────────────────────────────────────
def fit_dataset(parts, pids, has_event, iters, seed=0):
    torch.manual_seed(seed)
    d = make_tensors(parts, pids, has_event, H=H)
    idx = np.arange(len(pids))
    model = ModelV2(len(pids), has_event, g=1.0, horizons=H)
    t0 = time.time()
    fit(model, d, idx, iters=iters, seed=seed)
    fit(model, d, idx, iters=max(iters // 3, 50), lr=0.02, lam=1.0, seed=seed + 1, delta_only=True)
    with torch.no_grad():
        var = float((model.delta ** 2).mean())
        model.delta.zero_()
    lam = float(np.clip(1.0 / (2.0 * max(var, 1e-6)), 1.0, 1e4))
    fit(model, d, idx, iters=max(iters // 2, 80), lr=0.02, lam=lam, seed=seed + 2, delta_only=True)
    log(f"    fit {len(pids)} participants in {time.time() - t0:.0f}s, lam={lam:.1f}")
    return model, d, lam


def refit_deltas(model, parts, pids, has_event, lam, iters, seed):
    """Per-person deviations on a different set of sequences, group frozen."""
    d = make_tensors(parts, pids, has_event, H=H)
    with torch.no_grad():
        model.delta.zero_()
    fit(model, d, np.arange(len(pids)), iters=max(iters // 2, 80), lr=0.02, lam=lam, seed=seed, delta_only=True)
    return d


def person_params(model, d):
    idx = torch.arange(d["y"].shape[0])
    with torch.no_grad():
        res = model(d, idx, H=H)
        dl = model.delta.detach().numpy().copy()
    mask = d["mask"].numpy()
    out = []
    for n in range(mask.shape[0]):
        T = int(mask[n].sum())
        lt = float(model.log_theta) + dl[n, HIER.index("log_theta")]
        out.append(dict(
            theta=math.exp(lt) + 2.0,
            log_theta_dev=float(dl[n, HIER.index("log_theta")]),
            km=0.5 / (1 + math.exp(-(float(model.km_logit) + dl[n, HIER.index("km_logit")]))),
            beta_P=float(model.beta_P) + dl[n, HIER.index("beta_P")],
            baseline=float(res["b"][n, T - 1]),
            mood_mean=float(res["m"][n, :T].mean()),
        ))
    return out


def laplace_sd(model, d, lam):
    """Posterior SD of each person's deviations: inverse Hessian of
    sum-over-rows NLL + lam * ||delta_i||^2 (the MAP objective), per person."""
    sds = []
    base = model.delta.detach().clone()
    params = {k: v.detach() for k, v in model.named_parameters()}
    N = d["y"].shape[0]
    for n in range(N):
        dn = subset(d, [n])
        nrows = float(sum(float(dn["mt"][h].sum()) for h in (1, 2, 3)))

        def obj(dv):
            full = torch.cat([base[:n], dv[None, :], base[n + 1:]], 0)
            p = dict(params)
            p["delta"] = full
            res = torch.func.functional_call(model, p, (dn, torch.tensor([n])), {"H": 3})
            return model.nll(dn, torch.tensor([n]), res, (1, 2, 3)) * nrows + lam * (dv ** 2).sum()

        Hm = torch.autograd.functional.hessian(obj, base[n].clone())
        Hm = 0.5 * (Hm + Hm.T) + 1e-6 * torch.eye(Hm.shape[0])
        try:
            cov = torch.linalg.inv(Hm).numpy()
            sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        except Exception:
            sd = np.full(len(HIER), np.nan)
        sds.append(dict(log_theta_sd=float(sd[HIER.index("log_theta")]), beta_P_sd=float(sd[HIER.index("beta_P")])))
    return sds


# ── model-free person statistics ────────────────────────────────────────
def model_free(seq):
    v = np.array([b["v"] for b in seq])
    pairs = [(seq[i]["v"], seq[i + 1]["v"]) for i in range(len(seq) - 1) if seq[i]["seg"] == seq[i + 1]["seg"]]
    ar = np.nan
    if len(pairs) >= 8:
        a = np.array(pairs)
        if a[:, 0].std() > 1e-9:
            ar = float(np.polyfit(a[:, 0] - a[:, 0].mean(), a[:, 1] - a[:, 1].mean(), 1)[0])
    ng = [(b["neg"], b["v"]) for b in seq if b.get("neg") is not None]
    react = np.nan
    if len(ng) >= 8:
        a = np.array(ng)
        if a[:, 0].std() > 1e-9:
            react = float(-np.polyfit(a[:, 0], a[:, 1], 1)[0])     # valence drop per unit of negative event
    wp = [(seq[i]["w"], seq[i + 1]["w"]) for i in range(len(seq) - 1)
          if seq[i]["seg"] == seq[i + 1]["seg"] and seq[i].get("w") is not None and seq[i + 1].get("w") is not None]
    pers = np.nan
    if len(wp) >= 8:
        a = np.array(wp)
        if a[:, 0].std() > 1e-9 and a[:, 1].std() > 1e-9:
            pers = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
    wmean = np.nanmean([b["w"] for b in seq if b.get("w") is not None]) if any(b.get("w") is not None for b in seq) else np.nan
    return dict(mean_v=float(v.mean()), ar1=ar, react=react, persist=pers, w_mean=float(wmean))


# ── statistics ──────────────────────────────────────────────────────────
def z(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / np.nanstd(x)


def std_slope(y, x, w=None, n_boot=2000, seed=0):
    """Standardized slope of y on x (both z-scored), WLS if w given, HC3 SE and
    participant-bootstrap 95% CI."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    if w is not None:
        w = np.asarray(w, float)
        if np.isfinite(w[ok]).sum() < 10:
            w = None
        else:
            ok &= np.isfinite(w)
    if ok.sum() < 10 or np.nanstd(y[ok]) < 1e-12:
        return dict(beta=float("nan"), se=float("nan"), lo=float("nan"), hi=float("nan"), n=int(ok.sum()))
    y, x = z(y[ok]), z(x[ok])
    ww = np.ones_like(y) if w is None else w[ok] / np.mean(w[ok])
    X = np.c_[np.ones_like(x), x]

    def est(Xs, ys, ws):
        W = ws[:, None]
        return np.linalg.solve(Xs.T @ (W * Xs), Xs.T @ (ws * ys))

    b = est(X, y, ww)
    r = y - X @ b
    XtWX_inv = np.linalg.inv(X.T @ (ww[:, None] * X))
    hat = np.einsum("ij,jk,ik->i", X, XtWX_inv, ww[:, None] * X)
    meat = (X * (ww * r / (1 - hat))[:, None]).T @ (X * (ww * r / (1 - hat))[:, None])
    se = float(np.sqrt((XtWX_inv @ meat @ XtWX_inv)[1, 1]))
    rng = np.random.RandomState(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.randint(0, len(y), len(y))
        try:
            bs.append(est(X[i], y[i], ww[i])[1])
        except np.linalg.LinAlgError:
            pass
    return dict(beta=float(b[1]), se=se, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=int(ok.sum()))


def re_weights(est, sd):
    """Weights 1 / (posterior var + between-person var) for a noisy person-level estimate."""
    est, sd = np.asarray(est, float), np.asarray(sd, float).copy()
    # a failed Hessian inversion returns a near-zero SD that would give one person all the
    # weight; floor every SD at the 5th percentile of the plausible ones (> 0.01)
    good = np.isfinite(sd) & (sd > 0.01)
    if good.sum() >= 10:
        sd = np.where(np.isfinite(sd), np.maximum(sd, np.percentile(sd[good], 5)), sd)
    ok = np.isfinite(est) & np.isfinite(sd)
    tau2 = max(np.nanvar(est[ok]) - np.nanmean(sd[ok] ** 2), 1e-9)
    return 1.0 / (sd ** 2 + tau2)


def incremental(trait, model_x, free_x):
    """trait ~ model + model-free (all z): coefficient of the model parameter."""
    t, a, b = map(lambda v: np.asarray(v, float), (trait, model_x, free_x))
    ok = np.isfinite(t) & np.isfinite(a) & np.isfinite(b)
    t, a, b = z(t[ok]), z(a[ok]), z(b[ok])
    X = np.c_[np.ones_like(a), a, b]
    beta = np.linalg.lstsq(X, t, rcond=None)[0]
    r = t - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    hat = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    meat = (X * (r / (1 - hat))[:, None]).T @ (X * (r / (1 - hat))[:, None])
    se = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))
    return dict(b_model=float(beta[1]), se_model=float(se[1]), b_free=float(beta[2]), se_free=float(se[2]),
                r_model_free=float(np.corrcoef(a, b)[0, 1]), n=int(ok.sum()))


def split_half_rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    r = np.corrcoef(a[ok], b[ok])[0, 1]
    return dict(r_half=float(r), spearman_brown=float(2 * r / (1 + r)) if r > -1 else float("nan"), n=int(ok.sum()))


def interaction(parts, traits, trait_name):
    """v_{t+1} ~ v_t + neg_t(person-centred) + trait_z + neg_t x trait_z, same segment,
    participant-clustered sandwich SE; plus the concurrent version v_t ~ neg_t x trait."""
    tv = {p: t[trait_name] for p, t in traits.items() if trait_name in t}
    pids = [p for p in parts if p in tv]
    tz = dict(zip(pids, z([tv[p] for p in pids])))
    out = {}
    for kind in ("lagged", "concurrent"):
        rows, g = [], []
        for p in pids:
            seq = parts[p]
            negs = [b["neg"] for b in seq if b.get("neg") is not None]
            if len(negs) < 5:
                continue
            nm = float(np.mean(negs))
            for i in range(len(seq)):
                b = seq[i]
                if b.get("neg") is None:
                    continue
                nc = b["neg"] - nm
                if kind == "lagged":
                    if i + 1 >= len(seq) or seq[i + 1]["seg"] != b["seg"]:
                        continue
                    rows.append([1.0, b["v"], nc, tz[p], nc * tz[p], seq[i + 1]["v"]])
                else:
                    rows.append([1.0, 0.0, nc, tz[p], nc * tz[p], b["v"]])
                g.append(p)
        if not rows:
            continue
        A = np.array(rows)
        X, y = (A[:, :5] if kind == "lagged" else A[:, [0, 2, 3, 4]]), A[:, 5]
        g = np.array(g)
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        r = y - X @ beta
        meat = np.zeros((X.shape[1], X.shape[1]))
        G = 0
        for gg in np.unique(g):
            m = g == gg
            u = X[m].T @ r[m]
            meat += np.outer(u, u)
            G += 1
        V = XtX_inv @ meat @ XtX_inv * G / (G - 1)
        se = np.sqrt(np.diag(V))
        j = X.shape[1] - 1
        out[kind] = dict(b_neg=float(beta[j - 2]), se_neg=float(se[j - 2]), b_int=float(beta[j]), se_int=float(se[j]),
                         lo=float(beta[j] - 1.96 * se[j]), hi=float(beta[j] + 1.96 * se[j]),
                         z=float(beta[j] / se[j]), n_obs=int(len(y)), n_part=int(G),
                         sd_v=float(np.std(y)))
    return out


def dersimonian_laird(betas, ses):
    b, s = np.asarray(betas, float), np.asarray(ses, float)
    w = 1 / s ** 2
    fe = np.sum(w * b) / np.sum(w)
    Q = np.sum(w * (b - fe) ** 2)
    k = len(b)
    tau2 = max(0.0, (Q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w))) if k > 1 else 0.0
    wr = 1 / (s ** 2 + tau2)
    re = np.sum(wr * b) / np.sum(wr)
    se = math.sqrt(1 / np.sum(wr))
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return dict(beta=float(re), se=float(se), lo=float(re - 1.96 * se), hi=float(re + 1.96 * se), tau2=float(tau2),
                Q=float(Q), I2=float(I2), k=k)


# ── main ────────────────────────────────────────────────────────────────
def clean(o):
    """JSON-safe copy: numpy scalars to Python, NaN to None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def halves(parts):
    h1, h2 = {}, {}
    for p, seq in parts.items():
        m = len(seq) // 2
        a, b = [dict(x) for x in seq[:m]], [dict(x) for x in seq[m:]]
        if len(a) >= MIN_BEEPS // 2 and len(b) >= MIN_BEEPS // 2:
            h1[p], h2[p] = a, b
    return h1, h2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--datasets", default="Geschwind,Kane,Gainey")
    ap.add_argument("--no_laplace", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    summary = {}
    for name in args.datasets.split(","):
        t0 = time.time()
        parts, traits, has_event, w_label = LOADERS[name]()
        pids = sorted(p for p in parts if p in traits)
        if args.quick:
            pids = pids[:40]
        parts = {p: parts[p] for p in pids}
        n_sig = sum(len(s) for s in parts.values())
        log(f"[{name}] {len(pids)} participants with traits, {n_sig} signals, events={has_event}")
        cache = OUT / f"person_{name}{'_quick' if args.quick else ''}.json"
        if cache.exists():
            per = json.loads(cache.read_text())
            log("    loaded cached person-level estimates")
        else:
            model, d, lam = fit_dataset(parts, pids, has_event, args.iters)
            pp = person_params(model, d)
            sds = [dict(log_theta_sd=np.nan, beta_P_sd=np.nan)] * len(pids) if args.no_laplace else laplace_sd(model, d, lam)
            log(f"    laplace done ({time.time() - t0:.0f}s)")
            h1, h2 = halves(parts)
            hp = sorted(h1)
            half_est = {}
            for tag, hp_parts in (("h1", h1), ("h2", h2)):
                dh = refit_deltas(model, hp_parts, hp, has_event, lam, args.iters, seed=11 if tag == "h1" else 12)
                half_est[tag] = dict(zip(hp, person_params(model, dh)))
            log(f"    split-half done ({time.time() - t0:.0f}s)")
            group = {k: (float(v) if v.dim() == 0 else [float(x) for x in v]) for k, v in model.named_parameters() if k != "delta"}
            per = dict(lam=lam, group=group, persons={})
            for i, p in enumerate(pids):
                mf = model_free(parts[p])
                mf1 = model_free(h1[p]) if p in h1 else None
                mf2 = model_free(h2[p]) if p in h2 else None
                per["persons"][p] = dict(trait=traits[p], n=len(parts[p]), **pp[i], **sds[i], **{f"mf_{k}": v for k, v in mf.items()},
                                         half={t: dict(model=half_est[t].get(p), free=(mf1 if t == "h1" else mf2)) for t in ("h1", "h2")})
            cache.write_text(json.dumps(clean(per)))
        P = per["persons"]
        pl = sorted(P)
        g = lambda k: np.array([np.nan if P[p].get(k) is None else P[p][k] for p in pl], float)
        res = dict(n=len(pl), signals=int(sum(P[p]["n"] for p in pl)), lam=per["lam"], w_label=w_label,
                   theta_group=math.exp(per["group"]["log_theta"]) + 2.0, relations={}, reliability={}, incremental={})
        tr_names = sorted({k for p in pl for k in P[p]["trait"]})
        w_theta = re_weights(g("log_theta_dev"), g("log_theta_sd"))
        w_bp = re_weights(g("beta_P"), g("beta_P_sd"))
        outcomes = {
            "a_level_model": (g("baseline"), None), "a_level_free": (g("mf_mean_v"), None),
            "a_mood_mean_model": (g("mood_mean"), None),
            "b_inertia_model": (np.log(g("theta")), w_theta), "b_inertia_free": (g("mf_ar1"), None),
            "d_persist_free": (g("mf_persist"), None), "d_level_free": (g("mf_w_mean"), None),
        }
        if has_event:
            outcomes["c_react_model"] = (g("beta_P"), w_bp)       # larger = stronger event-to-mood coupling
            outcomes["c_react_free"] = (g("mf_react"), None)
        for tn in tr_names:
            t = np.array([P[p]["trait"].get(tn, np.nan) for p in pl], float)
            res["relations"][tn] = {k: std_slope(y, t, w) for k, (y, w) in outcomes.items()}
            pairs = [("a", "baseline", "mf_mean_v"), ("b", "theta", "mf_ar1")] + ([("c", "beta_P", "mf_react")] if has_event else [])
            res["incremental"][tn] = {}
            for lab, mk, fk in pairs:
                mx = np.log(g(mk)) if mk == "theta" else g(mk)
                res["incremental"][tn][lab] = incremental(t, mx, g(fk))
            res["interaction"] = res.get("interaction", {})
            if has_event:
                res["interaction"][tn] = interaction(parts, {p: P[p]["trait"] for p in pl}, tn)
        # split-half reliability
        def hv(tag, src, key):
            out = []
            for p in pl:
                h = P[p]["half"][tag][src]
                v = None if h is None else h.get(key)
                out.append(np.nan if v is None else v)
            return np.array(out, float)
        rel = {"a_model": ("model", "baseline"), "a_free": ("free", "mean_v"), "b_model": ("model", "theta"),
               "b_free": ("free", "ar1"), "d_free": ("free", "persist")}
        if has_event:
            rel.update({"c_model": ("model", "beta_P"), "c_free": ("free", "react")})
        for k, (src, key) in rel.items():
            a, b = hv("h1", src, key), hv("h2", src, key)
            if key == "theta":
                a, b = np.log(a), np.log(b)
            res["reliability"][k] = split_half_rel(a, b)
        res["theta_range"] = [float(np.nanpercentile(g("theta"), q)) for q in (5, 50, 95)]
        res["corr_level_model_free"] = float(np.corrcoef(g("baseline"), g("mf_mean_v"))[0, 1])
        summary[name] = res
        log(f"[{name}] done in {time.time() - t0:.0f}s")

    # pooled neuroticism effects
    pooled = {}
    keys = sorted({k for r in summary.values() for k in r["relations"].get("neuroticism", {})})
    for k in keys:
        rs = [(n, r["relations"]["neuroticism"][k]) for n, r in summary.items() if k in r["relations"].get("neuroticism", {})]
        if len(rs) >= 2:
            pooled[k] = dict(datasets=[n for n, _ in rs], **dersimonian_laird([x["beta"] for _, x in rs], [x["se"] for _, x in rs]))
    ints = [(n, r["interaction"]["neuroticism"]["lagged"]) for n, r in summary.items()
            if "neuroticism" in r.get("interaction", {}) and "lagged" in r["interaction"]["neuroticism"]]
    if len(ints) >= 2:
        pooled["e_interaction_lagged_std"] = dict(
            datasets=[n for n, _ in ints],
            **dersimonian_laird([x["b_int"] / x["sd_v"] for _, x in ints], [x["se_int"] / x["sd_v"] for _, x in ints]))
    summary["_pooled_neuroticism"] = pooled
    tag = "_quick" if args.quick else ""
    (OUT / f"summary{tag}.json").write_text(json.dumps(clean(summary), indent=1))
    log(f"summary written: {OUT / f'summary{tag}.json'}")


if __name__ == "__main__":
    main()
