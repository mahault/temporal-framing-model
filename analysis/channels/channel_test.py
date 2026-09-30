"""
Channel-specific test of the three valence channels on Baumeister et al. (2020)
Study 1 thought-content and emotion items. Predictions are preregistered in
reviews/CHANNEL_TEST.md section 0 (commit 9d9b048) before this script existed.

Usage (repo root):
    python analysis/channels/channel_test.py load     # signals + items
    python analysis/channels/channel_test.py fit      # v2.1 channels, 5 participant folds, two variants
    python analysis/channels/channel_test.py analyse  # regressions, discriminant, incremental, scale
    python analysis/channels/channel_test.py figures
    python analysis/channels/channel_test.py all

model_v2.py and the v1 code are imported unmodified.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "analysis" / "channels" / "out"
OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures"
SAV = ROOT / "data_raw/candidates/baumeister2020_ett/Study1/ETT_ESM_Study1.sav"
N_BOOT = 2000

# item -> (column, kind, branch, own channel)
ITEMS = {
    "regret":       ("typepast_1", "bin", "past", "B"),
    "replaying":    ("typepast_2", "bin", "past", "B"),
    "past_disapp":  ("typepast_5", "bin", "past", "B"),
    "might_have":   ("typepast_11", "bin", "past", "B"),
    "worry":        ("typefuture_10", "bin", "future", "F"),
    "fear":         ("typefuture_9", "bin", "future", "F"),
    "planning":     ("typefuture_1", "bin", "future", "F"),
    "hoping":       ("typefuture_8", "bin", "future", "F"),
    "disappointed": ("disapp", "cont", None, "B"),
    "anxious":      ("anxious", "cont", None, "F"),
    "angry":        ("angry", "cont", None, None),
}
PRED_SIGN = {  # preregistered sign of the own channel
    "regret": -1, "replaying": -1, "past_disapp": -1, "might_have": -1, "disappointed": -1,
    "worry": -1, "fear": -1, "anxious": -1, "planning": +1, "hoping": +1,
}
CH = ["B", "P", "F"]


# ─────────────────────────────── load ───────────────────────────────
def load():
    """Same signal filter and order as analysis/orientation/orientation_test.py
    load_baumeister1 (valence present, sorted by DAY, SIG, >= 4 signals), so the
    cached v1 drives align index by index."""
    import pyreadstat
    df, meta = pyreadstat.read_sav(str(SAV))
    parts = {}
    for pid, g in df.groupby("PID"):
        g = g.sort_values(["DAY", "SIG"])
        seq = []
        prev_day = None
        for _, r in g.iterrows():
            if np.isnan(r["valence"]):
                continue
            t = r["TIME"]
            clock = (t.hour + t.minute / 60.0) if t is not None else float(r["hour"])
            item = {}
            for name, (col, kind, branch, _) in ITEMS.items():
                if kind == "bin":
                    item[name] = float(r[col] == 1.0)
                else:
                    item[name] = None if np.isnan(r[col]) else float(r[col])
            day = int(r["DAY"])
            seq.append(dict(
                v=float((r["valence"] + 3.0) / 6.0), v_raw=float(r["valence"]),
                e=None if np.isnan(r["contentpn"]) else float(r["contentpn"]),
                d=day, sig=int(r["SIG"]), clock=clock,
                tod=(clock - 9.0) / 9.0, first=1.0 if day != prev_day else 0.0, p=0,
                past=float(r["time_1"] == 1.0), present=float(r["time_2"] == 1.0),
                future=float(r["time_3"] == 1.0), notime=float(r["time_5"] == 1.0),
                **item))
            prev_day = day
        if len(seq) >= 4:
            parts[str(int(pid))] = seq
    labels = {c: (meta.column_names_to_labels.get(c) or "") for c, *_ in ITEMS.values()}
    json.dump(dict(parts=parts, labels=labels), open(OUT / "signals.json", "w"))
    n_sig = sum(len(s) for s in parts.values())
    counts = {k: int(sum(b[k] > 0 for s in parts.values() for b in s if b[k] is not None)) for k in ITEMS}
    asked = {k: int(sum(b[k] is not None for s in parts.values() for b in s)) for k in ITEMS}
    print(f"{len(parts)} participants, {n_sig} signals")
    print("positives:", counts)
    print("asked (non-missing):", asked)
    return parts


# ─────────────────────────────── v2.1 channels ───────────────────────────────
def folds_of(pids):
    rng = np.random.RandomState(0)
    order = list(pids)
    rng.shuffle(order)
    return {p: k % 5 for k, p in enumerate(order)}


def fit_channels(parts, variant):
    """variant 'valence': present channel is the valence innovation (no event input);
    'event': thought pleasantness (contentpn, -3..3) as the event input."""
    import torch
    from model_v2 import ModelV2, fit_hierarchical, make_tensors
    torch.set_num_threads(4)
    has_event = variant == "event"
    pids = sorted(parts)
    fold = folds_of(pids)
    idx_of = {p: i for i, p in enumerate(pids)}
    out, params = {}, []
    for k in range(5):
        t0 = time.time()
        tr = [p for p in pids if fold[p] != k]
        te = [p for p in pids if fold[p] == k]
        torch.manual_seed(k)
        model = ModelV2(len(pids), has_event=has_event, g=1.0)
        d_tr = make_tensors(parts, tr, has_event, H=3)
        fit_hierarchical(model, d_tr, [idx_of[p] for p in tr], iters=300, seed=k)
        with torch.no_grad():
            model.delta.zero_()
            d_te = make_tensors(parts, te, has_event, H=3)
            res = model(d_te, torch.tensor([idx_of[p] for p in te]), H=3)
        for n, p in enumerate(te):
            L = len(parts[p])
            out[p] = {c: res[f"v{c}"][n, :L].numpy().tolist() for c in CH}
            out[p]["q"] = res["q"][n, :L].numpy().tolist()
            out[p]["eps"] = res["eps"][n, :L].numpy().tolist()
            out[p]["u"] = res["u"][n, :L].numpy().tolist()
        sp = torch.nn.functional.softplus
        params.append(dict(fold=k, beta=[float(model.beta_B), float(model.beta_P), float(model.beta_F)],
                           tau=[float(x) for x in (sp(model.a_tau) + 1e-3)],
                           nll=float(model.nll(d_te, None, res, (1, 2, 3)))))
        print(f"  [{variant}] fold {k}: {len(tr)} train / {len(te)} test, {time.time() - t0:.0f}s", flush=True)
    json.dump(dict(channels=out, params=params), open(OUT / f"channels_v2_{variant}.json", "w"))
    return out, params


def v1_channels(parts):
    d = json.load(open(ROOT / "analysis/orientation/out/drives.json"))
    out = {}
    for p, seq in parts.items():
        dr = d[f"baumeister1|{p}|1.0|None"]
        assert len(dr) == len(seq), (p, len(dr), len(seq))
        out[p] = {"B": [s["v_model"] for s in dr], "P": [s["v_reward"] for s in dr], "F": [s["v_action"] for s in dr]}
    return out


# ─────────────────────────────── statistics ───────────────────────────────
def design(parts, ch, extra=("valence", "tod", "orient"), centre=False):
    """Signal rows with z-scored channels and covariates."""
    rows = []
    for p, seq in parts.items():
        for i, b in enumerate(seq):
            rows.append(dict(pid=p, B=ch[p]["B"][i], P=ch[p]["P"][i], F=ch[p]["F"][i], val=b["v_raw"],
                             tod=b["tod"], past=b["past"], present=b["present"], future=b["future"],
                             **{k: b[k] for k in ITEMS}))
    pid = np.array([r["pid"] for r in rows])
    Z = {}
    for c in CH + ["val"]:
        x = np.array([r[c] for r in rows], float)
        if centre:
            for u in np.unique(pid):
                m = pid == u
                x[m] = x[m] - x[m].mean()
        Z[c] = (x - x.mean()) / (x.std() + 1e-12)
    cov = {"tod": np.array([r["tod"] for r in rows], float),
           "past": np.array([r["past"] for r in rows]), "present": np.array([r["present"] for r in rows]),
           "future": np.array([r["future"] for r in rows])}
    Y = {k: np.array([np.nan if r[k] is None else r[k] for r in rows], float) for k in ITEMS}
    br = {"past": cov["past"] == 1, "future": cov["future"] == 1}
    return pid, Z, cov, Y, br


def X_of(Z, cov, chans, orient=True, tod=True, val=True):
    cols, names = [np.ones(len(cov["tod"]))], ["const"]
    for c in chans:
        cols.append(Z[c]); names.append(c)
    if val:
        cols.append(Z["val"]); names.append("valence")
    if tod:
        cols.append(cov["tod"]); names.append("tod")
    if orient:
        for o in ("past", "present", "future"):
            cols.append(cov[o]); names.append(o)
    return np.column_stack(cols), names


def logit_fit(X, y, l2=1e-4, iters=100):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (y - p) - l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def cluster_se(X, resid_score, bread, groups):
    meat = np.zeros((X.shape[1], X.shape[1]))
    for u in np.unique(groups):
        m = groups == u
        s = resid_score[m].sum(0)
        meat += np.outer(s, s)
    G = len(np.unique(groups))
    V = bread @ meat @ bread * G / (G - 1)
    return np.sqrt(np.diag(V))


def logit_cluster(X, y, groups):
    w = logit_fit(X, y)
    p = 1 / (1 + np.exp(-(X @ w)))
    bread = np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X + 1e-4 * np.eye(X.shape[1]))
    se = cluster_se(X, X * (y - p)[:, None], bread, groups)
    return w, se


def ols_cluster(X, y, groups):
    bread = np.linalg.inv(X.T @ X)
    w = bread @ X.T @ y
    se = cluster_se(X, X * (y - X @ w)[:, None], bread, groups)
    return w, se


def fit_item(item, pid, Z, cov, Y, br, chans=CH, within_branch=False):
    kind, branch = ITEMS[item][1], ITEMS[item][2]
    y = Y[item]
    m = ~np.isnan(y)
    orient = True
    if within_branch and branch is not None:
        m = m & br[branch]
        orient = False
    X, names = X_of(Z, cov, chans, orient=orient)
    X, y, g = X[m], y[m], pid[m]
    if kind == "bin":
        w, se = logit_cluster(X, y, g)
    else:
        y = (y - y.mean()) / (y.std() + 1e-12)
        w, se = ols_cluster(X, y, g)
    return {n: dict(b=float(w[j]), se=float(se[j]), lo=float(w[j] - 1.96 * se[j]), hi=float(w[j] + 1.96 * se[j]))
            for j, n in enumerate(names)} | {"n": int(m.sum()), "pos": int((y > 0).sum()) if kind == "bin" else None,
                                              "clusters": int(len(np.unique(g)))}


def heldout_ll(item, pid, Z, cov, Y, chans, orient, folds, val=True):
    """Per-participant summed held-out log-likelihood of the item."""
    kind = ITEMS[item][1]
    y = Y[item]
    m = ~np.isnan(y)
    X, _ = X_of(Z, cov, chans, orient=orient, val=val)
    ll = {}
    fold_arr = np.array([folds[p] for p in pid])
    for k in range(5):
        tr = m & (fold_arr != k)
        te = m & (fold_arr == k)
        if kind == "bin":
            w = logit_fit(X[tr], y[tr], l2=1.0)
            p = np.clip(1 / (1 + np.exp(-(X[te] @ w))), 1e-9, 1 - 1e-9)
            l = y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)
        else:
            w = np.linalg.solve(X[tr].T @ X[tr] + 1e-3 * np.eye(X.shape[1]), X[tr].T @ y[tr])
            s2 = np.mean((y[tr] - X[tr] @ w) ** 2)
            l = -0.5 * np.log(2 * np.pi * s2) - (y[te] - X[te] @ w) ** 2 / (2 * s2)
        for u, v in zip(pid[te], l):
            a = ll.setdefault(u, [0.0, 0])
            a[0] += float(v); a[1] += 1
    return ll


def diff_boot(llA, llB, seed=0):
    """Mean per-signal difference A - B (nats/signal) with a participant bootstrap."""
    ps = sorted(set(llA) & set(llB))
    a = np.array([llA[p][0] for p in ps]); b = np.array([llB[p][0] for p in ps]); n = np.array([llA[p][1] for p in ps])
    est = (a.sum() - b.sum()) / n.sum()
    rng = np.random.RandomState(seed)
    bs = []
    for _ in range(N_BOOT):
        j = rng.randint(0, len(ps), len(ps))
        bs.append((a[j].sum() - b[j].sum()) / n[j].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(d=float(est), lo=float(lo), hi=float(hi), n=int(n.sum()), participants=len(ps))


def analyse_set(parts, ch, tag):
    pid, Z, cov, Y, br = design(parts, ch)
    folds = folds_of(sorted(parts))
    res = {"coef": {}, "coef_branch": {}, "coef_centred": {}, "coef_own_only": {}, "discriminant": {}, "incremental": {}}
    pidc, Zc, covc, Yc, brc = design(parts, ch, centre=True)
    for item in ITEMS:
        res["coef"][item] = fit_item(item, pid, Z, cov, Y, br)
        res["coef_centred"][item] = fit_item(item, pidc, Zc, covc, Yc, brc)
        if ITEMS[item][2] is not None:
            res["coef_branch"][item] = fit_item(item, pid, Z, cov, Y, br, within_branch=True)
        own = ITEMS[item][3]
        if own is None:
            continue
        # discriminant: own channel + valence vs each other channel + valence (no orientation, per the brief)
        ll = {c: heldout_ll(item, pid, Z, cov, Y, [c], orient=False, folds=folds) for c in CH}
        ll_none = heldout_ll(item, pid, Z, cov, Y, [], orient=False, folds=folds)
        res["discriminant"][item] = {"own": own,
                                     **{f"own_minus_{c}": diff_boot(ll[own], ll[c]) for c in CH if c != own},
                                     **{f"{c}_minus_valence_only": diff_boot(ll[c], ll_none) for c in CH}}
        # incremental: valence + tod + orientation, then + three channels
        base = heldout_ll(item, pid, Z, cov, Y, [], orient=True, folds=folds)
        full = heldout_ll(item, pid, Z, cov, Y, CH, orient=True, folds=folds)
        res["incremental"][item] = diff_boot(full, base)
    # scale check
    raw = {c: np.concatenate([np.asarray(ch[p][c], float) for p in parts]) for c in CH}
    res["scale"] = {"sd": {c: float(raw[c].std()) for c in CH}, "mean": {c: float(raw[c].mean()) for c in CH},
                    "corr": np.corrcoef(np.vstack([raw[c] for c in CH])).round(3).tolist(),
                    "corr_with_valence": {c: float(np.corrcoef(raw[c], np.array([b["v_raw"] for p in parts for b in parts[p]]))[0, 1]) for c in CH}}
    json.dump(res, open(OUT / f"results_{tag}.json", "w"), indent=1)
    return res


def analyse():
    S = json.load(open(OUT / "signals.json"))
    parts = S["parts"]
    out = {}
    for variant in ("valence", "event"):
        f = OUT / f"channels_v2_{variant}.json"
        if not f.exists():
            continue
        C = json.load(open(f))
        out[f"v2_{variant}"] = analyse_set(parts, C["channels"], f"v2_{variant}")
        out[f"v2_{variant}"]["params"] = C["params"]
        print(f"analysed v2_{variant}", flush=True)
    out["v1"] = analyse_set(parts, v1_channels(parts), "v1")
    print("analysed v1", flush=True)
    json.dump(out, open(OUT / "results_all.json", "w"), indent=1)
    return out


# ─────────────────────────────── figures ───────────────────────────────
def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = json.load(open(OUT / "results_all.json"))
    items = [k for k in ITEMS]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    for ax, tag, title in zip(axes, ("v2_valence", "v2_event", "v1"),
                              ("v2.1, valence only (primary)", "v2.1, thought pleasantness as event", "v1 discrete model")):
        if tag not in R:
            ax.axis("off"); continue
        M = np.array([[R[tag]["coef"][it][c]["b"] for c in CH] for it in items])
        S = np.array([[R[tag]["coef"][it][c]["lo"] > 0 or R[tag]["coef"][it][c]["hi"] < 0 for c in CH] for it in items])
        lim = np.abs(M).max()
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(len(items)):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:+.2f}{'*' if S[i, j] else ''}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(3)); ax.set_xticklabels(["backward", "present", "forward"])
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_yticks(range(len(items))); axes[0].set_yticklabels(items)
    fig.suptitle("Standardized channel coefficients per item, controlling valence, time of day and orientation "
                 "(* 95% cluster-robust CI excludes 0)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "channel_test_coefficients.png", dpi=160)
    # discriminant panel
    tag = "v2_valence"
    D = R[tag]["discriminant"]
    its = list(D)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    xs = np.arange(len(its))
    for off, key_fn, lab in ((-0.15, lambda it: [k for k in D[it] if k.startswith("own_minus_")][0], "own minus other 1"),
                             (0.15, lambda it: [k for k in D[it] if k.startswith("own_minus_")][1], "own minus other 2")):
        ks = [key_fn(it) for it in its]
        d = np.array([D[it][k]["d"] for it, k in zip(its, ks)]) * 1000
        lo = np.array([D[it][k]["lo"] for it, k in zip(its, ks)]) * 1000
        hi = np.array([D[it][k]["hi"] for it, k in zip(its, ks)]) * 1000
        ax.errorbar(xs + off, d, yerr=[d - lo, hi - d], fmt="o", capsize=3, label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(its, rotation=30, ha="right")
    ax.set_ylabel("held-out log-lik. difference (millinats/signal)")
    ax.set_title("Discriminant test, v2.1 channels: own channel vs each other channel (valence in every model)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "channel_test_discriminant.png", dpi=160)
    print("figures written")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("load", "all"):
        load()
    if stage in ("fit", "all"):
        parts = json.load(open(OUT / "signals.json"))["parts"]
        for variant in ("valence", "event"):
            fit_channels(parts, variant)
    if stage in ("analyse", "all"):
        analyse()
    if stage in ("figures", "all"):
        figures()
