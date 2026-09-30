"""
Identifiable latent temporal frame with an orientation likelihood.

Three latent frames (past, present, future). Two observation channels per
signal: reported orientation (categorical, per-frame likelihood A_o, fitted)
and valence (Gaussian, per-frame mean plus lag, time of day, pleasantness).
Dynamics compared with identical likelihoods:

    M0   base rates (no dynamics)
    M1   Markov chain over frames, group matrix, per-participant persistence
    M1b  M1 plus current valence in the transition (affect-aware null)
    M2   active-inference framing agent: RECALL / ENGAGE / FUTURATE chosen by
         softmax over expected free energy (risk under fitted preferences,
         minus expected information gain), fitted transition tendencies,
         fitted policy precision and policy prior

Usage (repo root):
    python analysis/orientation/frame_orientation_model.py fit      # all fits, cached
    python analysis/orientation/frame_orientation_model.py eval     # tables + bootstrap
    python analysis/orientation/frame_orientation_model.py figures
    python analysis/orientation/frame_orientation_model.py all

Data come from analysis/orientation/out/parts.json (loaders in
orientation_test.py). Nothing in the repo model code is touched.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
torch.manual_seed(0)
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "orientation" / "out_frame"
OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures"
PARTS = ROOT / "analysis" / "orientation" / "out" / "parts.json"

FRAMES = ["past", "present", "future"]
N_FOLDS = 5
N_BOOT = 1000
GRID = torch.linspace(0.0, 1.0, 41)
HOUR_BINS = 4  # <11, 11-15, 15-19, >=19
LAM = 5.0      # shrinkage weight on per-participant offsets
AO_PRIOR = 100.0  # Dirichlet pseudo-counts on the diagonal of A_o (anchors frames to reported orientation)
PATIENCE = 150
ITERS = 400
LR = 0.05


# ─────────────────────────────── data ───────────────────────────────
def hour_bin(h):
    if h is None:
        return 1
    if h < 11:
        return 0
    if h < 15:
        return 1
    if h < 19:
        return 2
    return 3


def dominant_label(r):
    """Baumeister S1 multi-focus rule: past if any past, else future if any
    future, else present; 'no time aspect' and missing are unobserved."""
    if r.get("single") in (1, 2, 3):
        return r["single"] - 1
    ori = r.get("ori")
    if ori is None or ori == 8:
        return -1
    if r.get("any_past"):
        return 0
    if r.get("any_future"):
        return 2
    return 1


def build(ds):
    parts = json.load(open(PARTS))[ds]
    pids = list(parts)
    seqs = []  # one per participant-day
    for pi, pid in enumerate(pids):
        s = parts[pid]
        days = sorted(set(r["d"] for r in s))
        for d in days:
            rows = [r for r in s if r["d"] == d]
            rows.sort(key=lambda r: r["sig"])
            if len(rows) < 2:
                continue
            seqs.append(dict(
                pid=pi,
                v=np.array([r["v"] for r in rows], float),
                e=np.array([0.0 if r.get("e") is None else r["e"] / 3.0 for r in rows], float),
                emask=np.array([0.0 if r.get("e") is None else 1.0 for r in rows], float),
                hb=np.array([hour_bin(r["hour"]) for r in rows], int),
                o=np.array([dominant_label(r) for r in rows], int),
            ))
    T = max(len(q["v"]) for q in seqs)
    n = len(seqs)
    V = np.zeros((n, T)); E = np.zeros((n, T)); EM = np.zeros((n, T)); HB = np.ones((n, T), int)
    O = -np.ones((n, T), int); MASK = np.zeros((n, T)); PID = np.zeros(n, int)
    for i, q in enumerate(seqs):
        L = len(q["v"])
        V[i, :L] = q["v"]; E[i, :L] = q["e"]; EM[i, :L] = q["emask"]; HB[i, :L] = q["hb"]
        O[i, :L] = q["o"]; MASK[i, :L] = 1; PID[i] = q["pid"]
    data = dict(V=torch.tensor(V, dtype=torch.float32), E=torch.tensor(E, dtype=torch.float32),
                EM=torch.tensor(EM, dtype=torch.float32), HB=torch.tensor(HB), O=torch.tensor(O),
                MASK=torch.tensor(MASK, dtype=torch.float32), PID=torch.tensor(PID), n_pid=len(pids), pids=pids)
    return data


def folds_of(n_pid):
    idx = np.arange(n_pid)
    return [set(idx[i::N_FOLDS].tolist()) for i in range(N_FOLDS)]


def subset(data, pid_set):
    m = torch.tensor([int(p) in pid_set for p in data["PID"].tolist()])
    return {k: (v[m] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == data["PID"].shape[0] else v)
            for k, v in data.items()}


# ─────────────────────────────── model ───────────────────────────────
class FrameModel(torch.nn.Module):
    """kind in {'M1','M1b','M2'}; frame_valence False collapses the valence
    likelihood across frames; use_e True adds thought pleasantness (S1)."""

    def __init__(self, kind, n_pid, use_e, frame_valence=True, flat_pref=False, no_epist=False, no_precision=False, ao_prior=AO_PRIOR):
        super().__init__()
        self.kind = kind; self.use_e = use_e; self.frame_valence = frame_valence; self.ao_prior = ao_prior
        self.flat_pref = flat_pref; self.no_epist = no_epist; self.no_precision = no_precision
        # emissions
        self.Ao_logit = torch.nn.Parameter(torch.tensor([[1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]]))
        self.mu = torch.nn.Parameter(torch.tensor([0.55, 0.65, 0.60]))
        self.beta = torch.nn.Parameter(torch.zeros(3))
        self.rho = torch.nn.Parameter(torch.tensor(0.4))
        self.tau = torch.nn.Parameter(torch.zeros(HOUR_BINS))
        self.log_sigma = torch.nn.Parameter(torch.tensor(-1.5))
        self.a = torch.nn.Parameter(torch.zeros(n_pid))          # participant valence intercept
        self.pi0_logit = torch.nn.Parameter(torch.tensor([-1.0, 1.0, 0.0]))
        # dynamics
        self.W = torch.nn.Parameter(torch.tensor([[0.0, 1.0, 0.0], [-1.0, 1.5, 0.0], [-1.0, 0.5, 0.5]]))
        self.wv = torch.nn.Parameter(torch.zeros(3))             # M1b valence effect on target
        self.delta = torch.nn.Parameter(torch.zeros(n_pid))      # participant persistence offset
        # M2
        self.eta = torch.nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))  # action pull toward target
        self.kappa = torch.nn.Parameter(torch.tensor(0.5))       # persistence in B_a
        self.c_m = torch.nn.Parameter(torch.tensor(0.8))         # preferred valence
        self.log_c_s = torch.nn.Parameter(torch.tensor(-1.0))    # preference width
        self.log_gamma = torch.nn.Parameter(torch.tensor(1.0))   # policy precision
        self.g = torch.nn.Parameter(torch.zeros(n_pid))          # participant log-precision offset
        self.logE = torch.nn.Parameter(torch.zeros(3))           # policy prior

    # emissions
    def logAo(self):
        return torch.log_softmax(self.Ao_logit, dim=1)

    def mean_v(self, vprev, has_prev, e, em, hb, pid, train):
        a = self.a[pid] if train else torch.zeros_like(vprev)
        mu = self.mu if self.frame_valence else self.mu.mean().expand(3)
        beta = (self.beta if self.frame_valence else self.beta.mean().expand(3)) if self.use_e else torch.zeros(3)
        base = a + self.rho * (vprev - 0.5) * has_prev + self.tau[hb]
        return base[:, None] + mu[None, :] + beta[None, :] * (e * em)[:, None]  # (n, 3)

    def log_norm(self, v, mean):
        s = torch.exp(self.log_sigma)
        return -0.5 * ((v[:, None] - mean) / s) ** 2 - torch.log(s) - 0.5 * np.log(2 * np.pi)

    # transitions
    def trans_M1(self, pid, train, v=None):
        W = self.W[None].expand(len(pid), 3, 3).clone()
        d = self.delta[pid] if train else torch.zeros(len(pid))
        W = W + torch.eye(3)[None] * d[:, None, None]
        if self.kind == "M1b":
            W = W + (self.wv[None, :] * (v - 0.5)[:, None])[:, None, :]
        return torch.softmax(W, dim=2)

    def action_B(self, pid, train):
        d = self.delta[pid] if train else torch.zeros(len(pid))
        Bs = []
        for a_idx in range(3):
            W = torch.eye(3)[None] * (self.kappa + d)[:, None, None]
            pull = torch.zeros(3); pull[a_idx] = 1.0
            W = W + (self.eta[a_idx] * pull)[None, None, :]
            Bs.append(torch.softmax(W, dim=2))
        return torch.stack(Bs, dim=1)  # (n, 3 actions, 3, 3)

    def trans_M2(self, q, v, hb, pid, train):
        """q: filtered posterior over current frame (n,3); returns (n,3,3) transition and p(a)."""
        n = len(pid)
        Ba = self.action_B(pid, train)                      # (n, A, f, f')
        qn = torch.einsum("nf,nafg->nag", q, Ba)             # predicted frame per action (n, A, 3)
        # predicted valence per next frame: lag on current v, current hour bin, no pleasantness
        mean = self.mean_v(v, torch.ones_like(v), torch.zeros_like(v), torch.zeros_like(v), hb, pid, train)  # (n,3)
        s = torch.exp(self.log_sigma)
        lik = torch.exp(-0.5 * ((GRID[None, None, :] - mean[:, :, None]) / s) ** 2)  # (n, 3, G)
        lik = lik / lik.sum(dim=2, keepdim=True)            # discretised on the grid
        pv = torch.einsum("nag,ngx->nax", qn, lik) + 1e-12   # (n, A, G)
        # risk: KL[p(v'|a) || C]
        if self.flat_pref:
            risk = torch.zeros(n, 3)
        else:
            cs = torch.exp(self.log_c_s)
            logC = -0.5 * ((GRID - self.c_m) / cs) ** 2
            logC = logC - torch.logsumexp(logC, dim=0)
            risk = (pv * (torch.log(pv) - logC[None, None, :])).sum(dim=2)
        # epistemic: expected information gain about f' from (v', o')
        if self.no_epist:
            ig = torch.zeros(n, 3)
        else:
            Ao = torch.exp(self.logAo())                     # (3, 3)
            joint = lik[:, :, :, None] * Ao[None, :, None, :]  # (n, f', G, o)
            py = torch.einsum("nag,ngxo->naxo", qn, joint) + 1e-12       # (n, A, G, o)
            post = torch.einsum("nag,ngxo->nagxo", qn, joint) / py[:, :, None, :, :]
            Hpost = -(post * torch.log(post + 1e-12)).sum(dim=2)          # (n, A, G, o)
            EH = (py * Hpost).sum(dim=(2, 3))
            Hprior = -(qn * torch.log(qn + 1e-12)).sum(dim=2)
            ig = Hprior - EH
        G = risk - ig
        gam = torch.exp(self.log_gamma + (self.g[pid] if train else torch.zeros(n)))
        if self.no_precision:
            gam = torch.zeros_like(gam)
        pa = torch.softmax(self.logE[None, :] - gam[:, None] * G, dim=1)   # (n, A)
        B = torch.einsum("na,nafg->nfg", pa, Ba)
        return B, pa, G

    # forward algorithm
    def forward(self, D, train, collect=False):
        V, E, EM, HB, O, MASK, PID = D["V"], D["E"], D["EM"], D["HB"], D["O"], D["MASK"], D["PID"]
        n, T = V.shape
        logAo = self.logAo()
        log_alpha = torch.log_softmax(self.pi0_logit, dim=0)[None, :].expand(n, 3)
        ll_total = torch.zeros(n)
        rec = dict(pred_o=[], ll_o_next=[], acc_o_next=[], mask_next=[], ll_v=[], mask_v=[], vhat=[],
                   ll_v_cond=[], pa=[], q=[])
        vprev = torch.zeros(n); has_prev = torch.zeros(n)
        for t in range(T):
            m = MASK[:, t]
            mean = self.mean_v(vprev, has_prev, E[:, t], EM[:, t], HB[:, t], PID, train)
            lv = self.log_norm(V[:, t], mean)                       # (n,3)
            o = O[:, t]
            lo = torch.where((o >= 0)[:, None], logAo[:, o.clamp(min=0)].T, torch.zeros(n, 3))
            # predictive over f_t is exp(log_alpha) (already includes transition)
            pred = torch.softmax(log_alpha, dim=1)
            if collect:
                # concurrent valence given orientation at t and history
                post_o = torch.softmax(log_alpha + lo, dim=1)
                pv_mix = torch.logsumexp(torch.log(post_o + 1e-12) + lv, dim=1)
                rec["ll_v_cond"].append(pv_mix); rec["vhat"].append((post_o * mean).sum(dim=1))
                rec["mask_v"].append(m)
                # next-orientation prediction is scored below after transition
            joint = log_alpha + lo + lv
            ll_t = torch.logsumexp(joint, dim=1)
            ll_total = ll_total + ll_t * m
            q = torch.softmax(joint, dim=1)                          # filtered posterior after y_t
            if collect:
                rec["q"].append(q)
            # transition to t+1
            if self.kind in ("M1", "M1b"):
                B = self.trans_M1(PID, train, V[:, t])
                pa = None
            else:
                B, pa, _ = self.trans_M2(q, V[:, t], HB[:, t], PID, train)
                if collect:
                    rec["pa"].append(pa)
            log_alpha_next = torch.log(torch.einsum("nf,nfg->ng", q, B) + 1e-12)
            if collect and t + 1 < T:
                pf = torch.softmax(log_alpha_next, dim=1)
                po = pf @ torch.exp(logAo)                            # (n, 3) predicted orientation
                on = O[:, t + 1]
                mn = ((on >= 0) & (MASK[:, t + 1] > 0)).float()
                rec["pred_o"].append(po)
                rec["ll_o_next"].append(torch.log(po[torch.arange(n), on.clamp(min=0)] + 1e-12))
                rec["acc_o_next"].append((po.argmax(dim=1) == on).float())
                rec["mask_next"].append(mn)
            # carry: only where t+1 is observed does the next step matter
            log_alpha = torch.where(m[:, None] > 0, log_alpha_next, log_alpha)
            vprev = torch.where(m > 0, V[:, t], vprev); has_prev = torch.where(m > 0, torch.ones(n), has_prev)
        if collect:
            return ll_total, {k: (torch.stack(v, dim=1) if len(v) else None) for k, v in rec.items()}
        return ll_total

    def penalty(self):
        return LAM * (self.a.pow(2).sum() + self.delta.pow(2).sum() + self.g.pow(2).sum()) - self.ao_prior * torch.diag(self.logAo()).sum()


def fit(model, D, iters=ITERS, lr=LR, log=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_obs = float(D["MASK"].sum())
    best = None
    for it in range(iters):
        opt.zero_grad()
        ll = model(D, train=True).sum()
        loss = (-ll + model.penalty()) / n_obs
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        with torch.no_grad():
            model.log_sigma.clamp_(-4.0, 0.0); model.rho.clamp_(-1.0, 1.0)
            model.log_gamma.clamp_(-3.0, 4.0); model.log_c_s.clamp_(-3.0, 2.0); model.c_m.clamp_(0.0, 1.0)
        if log is not None and it % 50 == 0:
            log(f"    iter {it} loss {loss.item():.5f}")
        if best is None or loss.item() < best - 1e-6:
            best = loss.item(); stall = 0
        else:
            stall += 1
            if stall > PATIENCE:
                break
    return model


# ─────────────────────────────── evaluation ───────────────────────────────
def per_participant(vals, mask, pid, n_pid):
    """mean of vals over masked entries per participant; returns dict pid->mean (only pids with data)."""
    out = {}
    vals = vals.detach().numpy(); mask = mask.detach().numpy(); pid = pid.numpy()
    for p in np.unique(pid):
        sel = pid == p
        mm = mask[sel]
        if mm.sum() > 0:
            out[int(p)] = float((vals[sel] * mm).sum() / mm.sum())
    return out


def base_rate_scores(Dtr, Dte):
    O = Dtr["O"][Dtr["MASK"] > 0]; O = O[O >= 0]
    p = np.bincount(O.numpy(), minlength=3) + 1.0; p = p / p.sum()
    n, T = Dte["O"].shape
    ll = torch.zeros(n, T - 1); acc = torch.zeros(n, T - 1); mask = torch.zeros(n, T - 1)
    for t in range(T - 1):
        on = Dte["O"][:, t + 1]
        mn = ((on >= 0) & (Dte["MASK"][:, t + 1] > 0) & (Dte["MASK"][:, t] > 0)).float()
        ll[:, t] = torch.log(torch.tensor(p)[on.clamp(min=0)])
        acc[:, t] = (on == int(np.argmax(p))).float()
        mask[:, t] = mn
    return ll, acc, mask


SPECS = {
    "M1": dict(kind="M1"),
    "M1b": dict(kind="M1b"),
    "M2": dict(kind="M2"),
    "M1_collapsed": dict(kind="M1", frame_valence=False),
    "M1_free": dict(kind="M1", ao_prior=0.0),
    "M2_flatpref": dict(kind="M2", flat_pref=True),
    "M2_noepist": dict(kind="M2", no_epist=True),
    "M2_noprecision": dict(kind="M2", no_precision=True),
    "M1_anchor5x": dict(kind="M1", ao_prior=5 * AO_PRIOR),
    # shared valence mean across frames (frames defined by orientation only), 2026-09-30
    "M1b_shared": dict(kind="M1b", frame_valence=False),
    "M2_shared": dict(kind="M2", frame_valence=False),
    "M2_shared_flatpref": dict(kind="M2", frame_valence=False, flat_pref=True),
    "M2_shared_noepist": dict(kind="M2", frame_valence=False, no_epist=True),
    "M2_shared_noprecision": dict(kind="M2", frame_valence=False, no_precision=True),
}
SHARED = ["M1b_shared", "M2_shared", "M2_shared_flatpref", "M2_shared_noepist", "M2_shared_noprecision"]


def run_one(ds, name, log):
    """Fit one model spec on all folds of one dataset and save it on its own.
    Skips the fit if its saved part file already exists."""
    part = OUT / f"fitpart_{ds}_{name}.pt"
    if part.exists():
        log(f"{ds} {name}: saved part exists, skipping")
        return torch.load(part, weights_only=False)
    data = build(ds)
    use_e = ds == "baumeister1"
    folds = folds_of(data["n_pid"])
    res = {}
    for name, spec in [(name, SPECS[name])]:
        res[name] = dict(ll_o=[], acc_o=[], mask_o=[], ll_v=[], vhat=[], mask_v=[], params=[], pid=[], pa=[], q=[])
        for k in range(N_FOLDS):
            te = folds[k]; tr = set(range(data["n_pid"])) - te
            Dtr = subset(data, tr); Dte = subset(data, te)
            t0 = time.time()
            torch.manual_seed(k)
            model = FrameModel(n_pid=data["n_pid"], use_e=use_e, **spec)
            fit(model, Dtr, log=None)
            with torch.no_grad():
                _, rec = model(Dte, train=False, collect=True)
            res[name]["ll_o"].append(rec["ll_o_next"]); res[name]["acc_o"].append(rec["acc_o_next"])
            res[name]["mask_o"].append(rec["mask_next"]); res[name]["ll_v"].append(rec["ll_v_cond"])
            res[name]["vhat"].append(rec["vhat"]); res[name]["mask_v"].append(rec["mask_v"]); res[name]["pid"].append(Dte["PID"])
            if rec["pa"] is not None:
                res[name]["pa"].append(rec["pa"]); res[name]["q"].append(rec["q"])
            P = {n_: p.detach().numpy().tolist() for n_, p in model.named_parameters() if n_ not in ("a", "delta", "g")}
            P["Ao"] = torch.exp(model.logAo()).detach().numpy().tolist()
            if spec["kind"] == "M2":
                Ba = model.action_B(torch.zeros(1, dtype=torch.long), False)[0].detach().numpy()
                P["B_actions"] = Ba.tolist()
                P["tendency"] = [float(Ba[a][:, a].mean()) for a in range(3)]
            else:
                B = model.trans_M1(torch.zeros(1, dtype=torch.long), False, torch.full((1,), 0.5))[0].detach().numpy()
                P["B"] = B.tolist()
            res[name]["params"].append(P)
            log(f"{ds} {name} fold {k} done in {time.time()-t0:.0f}s  Ao diag {np.diag(P['Ao']).round(2).tolist()}")
    torch.save(res[name], OUT / f"fitpart_{ds}_{name}.pt")
    return res[name]


def merge(ds, log):
    """Assemble fits_{ds}.pt from per-model parts (plus an existing fits file) and add base rates."""
    data = build(ds)
    folds = folds_of(data["n_pid"])
    res = {}
    old = OUT / f"fits_{ds}.pt"
    if old.exists():
        res.update(torch.load(old, weights_only=False))
    for name in SPECS:
        part = OUT / f"fitpart_{ds}_{name}.pt"
        if part.exists():
            res[name] = torch.load(part, weights_only=False)
    missing = [n for n in SPECS if n not in res]
    if missing:
        log(f"{ds} merge: missing {missing}")
    # base rates
    res["M0"] = dict(ll_o=[], acc_o=[], mask_o=[], pid=[])
    for k in range(N_FOLDS):
        te = folds[k]; tr = set(range(data["n_pid"])) - te
        ll, acc, mask = base_rate_scores(subset(data, tr), subset(data, te))
        res["M0"]["ll_o"].append(ll); res["M0"]["acc_o"].append(acc); res["M0"]["mask_o"].append(mask)
        res["M0"]["pid"].append(subset(data, te)["PID"])
    torch.save(res, OUT / f"fits_{ds}.pt")
    json.dump(dict(pids=data["pids"]), open(OUT / f"pids_{ds}.json", "w"))
    return res


def run_fits(ds, log):
    for name in SPECS:
        run_one(ds, name, log)
    return merge(ds, log)


def paired_boot(d_by_pid, seed=0):
    rng = np.random.default_rng(seed)
    pids = list(d_by_pid); vals = np.array([d_by_pid[p] for p in pids])
    stats = [vals[rng.choice(len(vals), len(vals), replace=True)].mean() for _ in range(N_BOOT)]
    return float(vals.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def evaluate(ds, log):
    res = torch.load(OUT / f"fits_{ds}.pt", weights_only=False)
    data = build(ds)
    out = dict(dataset=ds, n_pid=data["n_pid"], models={})
    scores = {}
    for name in res:
        r = res[name]
        ll = torch.cat(r["ll_o"]); acc = torch.cat(r["acc_o"]); mask = torch.cat(r["mask_o"]); pid = torch.cat(r["pid"])
        s_ll = per_participant(ll, mask, pid, data["n_pid"]); s_acc = per_participant(acc, mask, pid, data["n_pid"])
        scores[name] = dict(ll=s_ll, acc=s_acc)
        entry = dict(n_scored=int(mask.sum()), ll_mean=float(np.mean(list(s_ll.values()))),
                     acc_mean=float(np.mean(list(s_acc.values()))))
        if "ll_v" in r:
            llv = torch.cat(r["ll_v"]); mv = torch.cat(r["mask_v"]); vhat = torch.cat(r["vhat"])
            V = torch.cat([subset(data, folds_of(data["n_pid"])[k])["V"] for k in range(N_FOLDS)])
            s_v = per_participant(llv, mv, pid, data["n_pid"])
            scores[name]["llv"] = s_v
            err = ((V - vhat) ** 2 * mv).sum() / mv.sum()
            tot = ((V - (V * mv).sum() / mv.sum()) ** 2 * mv).sum() / mv.sum()
            entry.update(llv_mean=float(np.mean(list(s_v.values()))), r2_v=float(1 - err / tot))
            entry["params_mean"] = {k: np.mean([p[k] for p in r["params"]], axis=0).tolist()
                                    for k in r["params"][0] if k not in ("pi0_logit",)}
            entry["params_sd"] = {k: np.std([p[k] for p in r["params"]], axis=0).tolist()
                                  for k in ("tendency", "mu", "beta", "rho", "c_m", "log_c_s", "log_gamma", "logE", "eta", "kappa")
                                  if k in r["params"][0]}
        if r.get("pa"):
            pa = torch.cat(r["pa"]); entry["action_rates"] = pa.mean(dim=(0, 1)).tolist()
        out["models"][name] = entry
    # paired contrasts, next-orientation log-likelihood and accuracy
    contrasts = [("M1", "M0"), ("M1b", "M0"), ("M1b", "M1"), ("M2", "M0"), ("M2", "M1"), ("M2", "M1b"),
                 ("M2_flatpref", "M2"), ("M2_noepist", "M2"), ("M2_noprecision", "M2"),
                 ("M1_anchor5x", "M1"), ("M1_anchor5x", "M0"), ("M1_free", "M1"),
                 ("M1_collapsed", "M0"), ("M1_collapsed", "M1"), ("M2", "M1_collapsed"),
                 ("M1b_shared", "M0"), ("M2_shared", "M0"), ("M1b_shared", "M1_collapsed"),
                 ("M2_shared", "M1_collapsed"), ("M2_shared", "M1b_shared"),
                 ("M2_shared_flatpref", "M2_shared"), ("M2_shared_noepist", "M2_shared"),
                 ("M2_shared_noprecision", "M2_shared"), ("M2_shared_noepist", "M1b_shared"),
                 ("M2_shared_noprecision", "M1b_shared")]
    # labelled-signal counts per participant (orientation observed), for the dense-label restriction
    nlab = {}
    for p_ in range(data["n_pid"]):
        sel = data["PID"] == p_
        o = data["O"][sel]; mk = data["MASK"][sel]
        nlab[p_] = int(((o >= 0) & (mk > 0)).sum())
    out["n_labelled_per_pid"] = dict(median=float(np.median(list(nlab.values()))),
                                     n_ge20=int(sum(v >= 20 for v in nlab.values())))
    out["contrasts"] = {}
    out["contrasts_dense"] = {}
    for a, b in contrasts:
        if a not in scores or b not in scores:
            continue
        common = set(scores[a]["ll"]) & set(scores[b]["ll"])
        dense = {p for p in common if nlab.get(p, 0) >= 20}
        if dense:
            d_dense = {p: scores[a]["ll"][p] - scores[b]["ll"][p] for p in dense}
            out["contrasts_dense"][f"{a}-{b}"] = dict(ll=paired_boot(d_dense), n=len(dense))
        # per-participant differences kept for pooling across samples
        out.setdefault("diffs", {})[f"{a}-{b}"] = {str(p): scores[a]["ll"][p] - scores[b]["ll"][p] for p in common}
        d_ll = {p: scores[a]["ll"][p] - scores[b]["ll"][p] for p in common}
        d_acc = {p: scores[a]["acc"][p] - scores[b]["acc"][p] for p in common}
        out["contrasts"][f"{a}-{b}"] = dict(ll=paired_boot(d_ll), acc=paired_boot(d_acc), n=len(common))
    # concurrent valence: frame-specific vs collapsed likelihood (same M1 dynamics), and M2 vs M1
    for a, b in [("M1", "M1_collapsed"), ("M2", "M1"), ("M1b", "M1"), ("M1_anchor5x", "M1_collapsed"),
                 ("M2_shared", "M1_collapsed"), ("M1b_shared", "M1_collapsed")]:
        if a not in scores or b not in scores or "llv" not in scores[a] or "llv" not in scores[b]:
            continue
        common = set(scores[a]["llv"]) & set(scores[b]["llv"])
        d = {p: scores[a]["llv"][p] - scores[b]["llv"][p] for p in common}
        out["contrasts"][f"valence:{a}-{b}"] = dict(llv=paired_boot(d), n=len(common))
    json.dump(out, open(OUT / f"eval_{ds}.json", "w"), indent=1)
    log(json.dumps({k: v for k, v in out["contrasts"].items()}, indent=1))
    return out


def pool(log):
    """Pool per-participant differences across samples (participants are distinct across samples).
    Participant-level bootstrap within each sample, weighted by sample size, plus inverse-variance."""
    rng = np.random.default_rng(0)
    evs = {ds: json.load(open(OUT / f"eval_{ds}.json")) for ds in ("baumeister1", "bayer")}
    keys = ["M2_shared-M1b_shared", "M2_shared-M1_collapsed", "M2-M1b", "M1b_shared-M1_collapsed",
            "M2_shared_noepist-M2_shared", "M2_shared_noprecision-M2_shared", "M2_shared_flatpref-M2_shared"]
    out = {}
    for k in keys:
        ds_vals = {ds: np.array(list(ev["diffs"][k].values())) for ds, ev in evs.items() if k in ev.get("diffs", {})}
        if len(ds_vals) < 2:
            continue
        allv = np.concatenate(list(ds_vals.values()))
        boots = []
        for _ in range(N_BOOT):
            boots.append(np.concatenate([v[rng.choice(len(v), len(v), replace=True)] for v in ds_vals.values()]).mean())
        m = [v.mean() for v in ds_vals.values()]; se = [v.std(ddof=1) / np.sqrt(len(v)) for v in ds_vals.values()]
        w = 1 / np.square(se); iv = float((w * m).sum() / w.sum()); iv_se = float(1 / np.sqrt(w.sum()))
        out[k] = dict(pooled_participant=[float(allv.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                      inverse_variance=[iv, iv - 1.96 * iv_se, iv + 1.96 * iv_se], n=int(len(allv)))
    json.dump(out, open(OUT / "pooled.json", "w"), indent=1)
    log(json.dumps(out, indent=1))
    return out


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    evs = {ds: json.load(open(OUT / f"eval_{ds}.json")) for ds in ("baumeister1", "bayer") if (OUT / f"eval_{ds}.json").exists()}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    # (a) next-orientation LL gain over base rates
    ax = axes[0]
    names = ["M1", "M1b", "M2"]
    for j, (ds, ev) in enumerate(evs.items()):
        vals = [ev["contrasts"][f"{n}-M0"]["ll"] for n in names]
        x = np.arange(3) + j * 0.35
        ax.bar(x, [v[0] for v in vals], 0.3, yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
               label=ds, capsize=3)
    ax.set_xticks(np.arange(3) + 0.17); ax.set_xticklabels(names); ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("next-signal orientation LL gain over base rates (nats)"); ax.legend(fontsize=8)
    # (b) fitted tendencies
    ax = axes[1]
    for j, (ds, ev) in enumerate(evs.items()):
        m = ev["models"]["M2"]["params_mean"]["tendency"]; s = ev["models"]["M2"]["params_sd"]["tendency"]
        ax.bar(np.arange(3) + j * 0.35, m, 0.3, yerr=s, capsize=3, label=ds)
    ax.plot(np.arange(3) + 0.17, [0.70, 0.75, 0.90], "k_", ms=18, label="asserted 70/75/90")
    ax.set_xticks(np.arange(3) + 0.17); ax.set_xticklabels(["RECALL", "ENGAGE", "FUTURATE"]); ax.set_ylim(0, 1)
    ax.set_ylabel("P(target frame | action), fitted"); ax.legend(fontsize=8)
    # (c) frame-specific valence means (fitted) vs empirical within-person contrasts
    ax = axes[2]
    emp = dict(baumeister1=(-0.38 / 6, -0.15 / 6), bayer=(-0.56 / 4, -0.14 / 4))
    for j, (ds, ev) in enumerate(evs.items()):
        mu = np.array(ev["models"]["M1"]["params_mean"]["mu"])
        fitted = (mu[0] - mu[1], mu[2] - mu[1])
        x = np.arange(2) + j * 0.35
        ax.bar(x, fitted, 0.3, label=f"{ds} fitted")
        ax.plot(x, emp[ds], "k_", ms=18)
    ax.set_xticks(np.arange(2) + 0.17); ax.set_xticklabels(["past - present", "future - present"])
    ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("valence difference (unit scale); black = empirical"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "frame_orientation_summary.png", dpi=150)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    logf = open(OUT / "run.log", "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    log(f"=== {stage} {' '.join(sys.argv[2:])} {time.ctime()}")
    if stage == "fit1":          # fit1 <ds> <model>
        run_one(sys.argv[2], sys.argv[3], log)
        return
    if stage == "merge":         # merge <ds>
        merge(sys.argv[2], log)
        return
    if stage == "eval1":         # eval1 <ds>
        evaluate(sys.argv[2], log)
        return
    if stage == "chain":         # chain <ds> <model> <model> ... fit serially, then merge and eval
        ds = sys.argv[2]
        for name in sys.argv[3:]:
            run_one(ds, name, log)
        merge(ds, log); evaluate(ds, log)
        log(f"CHAIN DONE {ds} {time.ctime()}")
        return
    if stage == "pool":
        pool(log)
        return
    for ds in ("baumeister1", "bayer"):
        if stage in ("fit", "all"):
            run_fits(ds, log)
        if stage in ("eval", "all"):
            evaluate(ds, log)
    if stage in ("figures", "all"):
        figures()


if __name__ == "__main__":
    main()
