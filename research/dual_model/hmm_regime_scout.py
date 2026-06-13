"""HMM regime SCOUT (2026-06-06).

Decision question (NOT a deploy): does an HMM regime layer built on ORDER-FLOW
emissions carry signal-quality information that the existing DETERMINISTIC
threshold regime (is_trending_bull/bear: vol_pct>0.6 & ret_24h +/-0.5%) does NOT?

If the HMM merely REDISCOVERS the threshold regime (high argmax agreement) OR does
not separate signal EV better than the threshold regime on a holdout, it is NO-GO:
same data source, no new info -- consistent with the documented V7 AUC-0.54 ceiling
(breakthrough needs a NEW data source, not a new model of the same OHLCV+CG+Deribit).

CHEAP-OPTIMISTIC design (scout, not full WF): single HMM fit on the FIRST half of
the OOS overlap, posteriors inferred on the SECOND half (true holdout -- the HMM
never saw it during fit; scaler fit on train only). The threshold regime is
deterministic, so it is OOS everywhere; BOTH regime systems are compared ON THE
HOLDOUT ONLY for a fair fight. The "favorable bucket" for BOTH systems is derived
on TRAIN (which bucket holds +net_bps signals) and APPLIED to holdout = leakage
free. If even this generous single-fit best-case shows nothing, escalating to full
walk-forward is pointless.

Caveats baked in: holdout signal N is small -> per-bucket n printed, thin buckets
(<30 obs, per mistake.md) flagged. Returns use the fixed-4h-hold OOS proxy
(y_path_ret_4h - round-trip cost), NOT the live trailing-stop exit -- fine for a
RELATIVE regime/sizing comparison, absolute numbers are a proxy.
"""
from __future__ import annotations

import sys
import warnings
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hmmlearn import hmm  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from research.dual_model.position_sizing_oos_backtest import (  # noqa: E402
    COST, OOS_PATH, decode, metrics, normalize,
)

CACHE = "research/dual_model/.cache/features_all.parquet"

# Order-flow emissions (NO price level / return -- the whole point is to give the
# HMM the channels the threshold regime ignores: positioning, vol, taker pressure)
EMISSIONS = [
    "realized_vol_20b",        # vol level
    "cg_funding_close_zscore", # positioning / leverage stress
    "cg_oi_close_pctchg_24h",  # OI build / unwind
    "cg_fcvd_delta_zscore",    # futures taker CVD pressure
    "cg_taker_delta_zscore",   # aggregated taker imbalance
    "dvol_zscore_72h",         # implied-vol regime (Deribit)
]
N_STATES = 3
THIN = 30  # per-bucket signal count below which estimates are flagged unreliable


def fit_best_hmm(X, n_states=N_STATES, n_init=10, n_iter=300):
    """Fit GaussianHMM with sticky transition prior; keep best log-likelihood
    across random inits (Baum-Welch finds local maxima -- mistake.md discipline:
    multiple seeds, keep best)."""
    best, best_ll = None, -np.inf
    stay = 0.95
    leave = (1.0 - stay) / (n_states - 1)
    tmat = np.full((n_states, n_states), leave)
    np.fill_diagonal(tmat, stay)
    for seed in range(n_init):
        m = hmm.GaussianHMM(n_components=n_states, covariance_type="full",
                            n_iter=n_iter, init_params="mcs", params="tmcs",
                            random_state=seed)
        m.transmat_ = tmat.copy()
        try:
            m.fit(X)
            ll = m.score(X)
        except Exception:
            continue
        if ll > best_ll:
            best, best_ll = m, ll
    return best, best_ll


def bucket_stats(reg, dsign, ypath, label):
    """Per-bucket signal n / win-rate / net_bps for a regime label array."""
    r = dsign * ypath - COST
    rows = []
    for b in sorted(pd.unique(reg)):
        mask = reg == b
        n = int(mask.sum())
        if n == 0:
            continue
        rr = r[mask]
        rows.append(dict(bucket=str(b), n=n,
                         win=float((rr > 0).mean() * 100),
                         net_bps=float(rr.mean() * 1e4)))
    print(f"\n  [{label}] signal performance by bucket "
          f"(holdout, cost={COST*1e4:.0f}bps):")
    print(f"    {'bucket':12s} {'n':>5s} {'win%':>6s} {'net_bps':>9s}")
    for x in rows:
        flag = "  <THIN" if x["n"] < THIN else ""
        print(f"    {x['bucket']:12s} {x['n']:5d} {x['win']:6.1f} "
              f"{x['net_bps']:+9.1f}{flag}")
    spread = (max(x["net_bps"] for x in rows) -
              min(x["net_bps"] for x in rows)) if rows else 0.0
    return rows, spread


def favor_set(reg, dsign, ypath):
    """Buckets whose TRAIN signals are net-positive -> 'favor' (size on),
    derived on train, applied to holdout (leakage-free)."""
    r = dsign * ypath - COST
    favor = set()
    for b in pd.unique(reg):
        rr = r[reg == b]
        if len(rr) > 0 and rr.mean() > 0:
            favor.add(b)
    return favor


def main():
    feat = pd.read_parquet(CACHE)
    oos = pd.read_parquet(OOS_PATH).sort_index()

    # Decode pred_ret causally over the FULL oos series (matches production order).
    direction, tier, conf = decode(oos["pred_ret"].values)
    oos["dir"], oos["tier"] = direction, tier

    need = EMISSIONS + ["is_trending_bull", "is_trending_bear"]
    feat_sub = feat[need].copy()
    df = oos.join(feat_sub, how="inner").dropna(subset=EMISSIONS).sort_index()

    # Threshold regime (the incumbent, deterministic).
    df["reg_thr"] = np.where(df["is_trending_bull"] == 1, "TR_BULL",
                     np.where(df["is_trending_bear"] == 1, "TR_BEAR", "CHOPPY"))

    n = len(df)
    cut = n // 2
    train, hold = df.iloc[:cut], df.iloc[cut:]
    print(f"overlap rows={n}  train={len(train)} "
          f"[{train.index[0].date()}..{train.index[-1].date()}]  "
          f"hold={len(hold)} [{hold.index[0].date()}..{hold.index[-1].date()}]")
    print(f"emissions: {EMISSIONS}")

    # Fit HMM on TRAIN only (scaler on train only).
    scaler = StandardScaler().fit(train[EMISSIONS].values)
    Xtr = scaler.transform(train[EMISSIONS].values)
    Xho = scaler.transform(hold[EMISSIONS].values)
    model, ll = fit_best_hmm(Xtr)
    if model is None:
        print("HMM failed to fit on all seeds -- abort.")
        return
    print(f"\nHMM fit: converged={model.monitor_.converged} "
          f"train_loglik={ll:.1f}")

    df.loc[train.index, "hmm"] = model.predict(Xtr)
    df.loc[hold.index, "hmm"] = model.predict(Xho)
    post_ho = model.predict_proba(Xho)  # holdout posteriors (forward algo)
    df["hmm"] = df["hmm"].astype(int)

    # Characterize HMM states on TRAIN (mean emission per state, std-scaled).
    print("\nHMM state character (TRAIN, z-scaled emission means):")
    print(f"    {'state':>5s} " + " ".join(f"{e.split('_')[0][:7]:>8s}"
                                            for e in EMISSIONS) + f" {'n':>5s}")
    for s in range(N_STATES):
        m = (df.loc[train.index, "hmm"] == s).values
        mu = Xtr[m].mean(axis=0) if m.any() else np.zeros(len(EMISSIONS))
        print(f"    {s:5d} " + " ".join(f"{v:8.2f}" for v in mu) +
              f" {int(m.sum()):5d}")

    # ---- Derive favorable buckets on TRAIN (signals only) ----
    tr_sig = train[train["dir"] != "NEUTRAL"].copy()
    tr_sig = df.loc[tr_sig.index]
    tds = np.where(tr_sig["dir"].values == "UP", 1.0, -1.0)
    typ = tr_sig["y_path_ret_4h"].values
    favor_hmm = favor_set(tr_sig["hmm"].values, tds, typ)
    favor_thr = favor_set(tr_sig["reg_thr"].values, tds, typ)
    print(f"\nTRAIN-derived favor sets:  HMM states={sorted(favor_hmm)}  "
          f"threshold={sorted(favor_thr)}")

    # ---- HOLDOUT comparison ----
    hs = df.loc[hold.index]
    sig = hs[hs["dir"] != "NEUTRAL"].copy()
    if len(sig) < 10:
        print(f"\nholdout signals={len(sig)} -- too few to judge. abort.")
        return
    post_sig = post_ho[[hold.index.get_loc(i) for i in sig.index]]
    dsign = np.where(sig["dir"].values == "UP", 1.0, -1.0)
    ypath = sig["y_path_ret_4h"].values
    r = dsign * ypath - COST
    span_days = (sig.index[-1] - sig.index[0]).days or 1
    print(f"\nholdout signals={len(sig)} span={span_days}d")

    # 1) Rediscovery: best-permutation agreement HMM-argmax vs threshold regime.
    thr_labels = ["TR_BULL", "TR_BEAR", "CHOPPY"]
    ct = np.zeros((N_STATES, 3))
    for s in range(N_STATES):
        for j, tl in enumerate(thr_labels):
            ct[s, j] = ((hs["hmm"] == s) & (hs["reg_thr"] == tl)).sum()
    best_agree = max(sum(ct[s, p[s]] for s in range(N_STATES))
                     for p in permutations(range(3))) / ct.sum()
    print(f"\n[1] Rediscovery: HMM-vs-threshold best-permutation agreement "
          f"= {best_agree*100:.1f}% of all holdout bars")
    print("    (>85% => HMM just relabels the threshold regime, no new info)")

    # 2) Signal-EV separation by each regime system.
    _, spread_thr = bucket_stats(sig["reg_thr"].values, dsign, ypath, "threshold")
    _, spread_hmm = bucket_stats(sig["hmm"].astype(str).values, dsign, ypath, "HMM")
    print(f"\n[2] net_bps spread across buckets: threshold={spread_thr:.1f}bps  "
          f"HMM={spread_hmm:.1f}bps  (wider HMM => HMM separates EV better)")

    # 3) Sizing head-to-head (avg leverage normalized to 1.0x -> pure allocation).
    is_strong = (sig["tier"].values == "Strong").astype(float)
    favor_mass = post_sig[:, sorted(favor_hmm)].sum(axis=1) if favor_hmm \
        else np.zeros(len(sig))
    rules = {
        "flat": np.ones(len(sig)),
        "tier (S1.0/M0.5)": np.where(is_strong > 0, 1.0, 0.5),
        "thr-gate (train-favor)": np.array(
            [1.0 if b in favor_thr else 0.0 for b in sig["reg_thr"].values]),
        "hmm-gate (train-favor)": np.array(
            [1.0 if b in favor_hmm else 0.0 for b in sig["hmm"].values]),
        "hmm-P(favor)-scale": favor_mass,
    }
    rows = []
    for name, w in rules.items():
        if w.sum() == 0:
            continue
        L = normalize(w)
        m = metrics(L, r, span_days)
        m["rule"] = name
        m["net_bps"] = float((L * r).mean() * 1e4)
        rows.append(m)

    print(f"\n[3] sizing head-to-head (holdout, avg L=1.0x -> allocation shape):")
    print(f"    {'rule':24s} {'net_bps':>8s} {'Sharpe':>7s} {'MDD%':>7s} "
          f"{'term':>6s} {'n>0':>5s}")
    base = next(x for x in rows if x["rule"] == "flat")
    for m in sorted(rows, key=lambda x: -x["net_bps"]):
        tag = "  <= flat" if m is base else ""
        print(f"    {m['rule']:24s} {m['net_bps']:+8.1f} {m['sharpe']:7.2f} "
              f"{m['mdd']:7.1f} {m['term']:6.2f} {m['n']:5d}{tag}")

    # bootstrap: hmm-gate vs flat per-signal mean-PnL diff real or noise?
    if "hmm-gate (train-favor)" in rules and rules["hmm-gate (train-favor)"].sum() > 0:
        rng = np.random.default_rng(42)
        pg = normalize(rules["hmm-gate (train-favor)"]) * r
        pf = normalize(rules["flat"]) * r
        diffs = [float((pg[idx].mean() - pf[idx].mean()) * 1e4)
                 for idx in (rng.integers(0, len(r), len(r)) for _ in range(3000))]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        frac = float(np.mean(np.array(diffs) > 0)) * 100
        verdict = "SIGNIFICANT" if lo > 0 else "NOT significant (CI spans 0)"
        print(f"\n    hmm-gate vs flat: mean-PnL diff 95% CI "
              f"[{lo:+.1f},{hi:+.1f}]bps  P(better)={frac:.0f}% -> {verdict}")

    print("\n" + "=" * 70)
    print("READ (decision criteria, all must hold to escalate to full WF):")
    print(f"  (a) agreement < 85%        : {best_agree*100:.1f}%  "
          f"{'PASS' if best_agree < 0.85 else 'FAIL (HMM = relabeled threshold)'}")
    print(f"  (b) HMM EV-spread > thr     : HMM {spread_hmm:.0f} vs thr "
          f"{spread_thr:.0f}bps  {'PASS' if spread_hmm > spread_thr else 'FAIL'}")
    hmm_gate = next((x for x in rows if x["rule"].startswith("hmm-gate")), None)
    beats = hmm_gate and hmm_gate["net_bps"] > base["net_bps"] \
        and (not any(x["rule"].startswith("thr-gate") and
                     x["net_bps"] >= hmm_gate["net_bps"] for x in rows))
    print(f"  (c) hmm-gate beats flat&thr : "
          f"{'PASS' if beats else 'FAIL'}")
    print("  NOTE: single-fit best-case + small holdout N. PASS here only means")
    print("  'worth the expensive full walk-forward'; FAIL = stop, save the 2 weeks.")


if __name__ == "__main__":
    main()
