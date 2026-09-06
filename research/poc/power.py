# -*- coding: utf-8 -*-
"""Stage 6 pre-registration support — how big an effect can this design see?

Run BEFORE the criteria are frozen and BEFORE any point estimate is looked at.

This file deliberately prints NO point estimate.  It resamples whole UTC days
with replacement, recomputes the estimator on each replicate, and reports only
the SPREAD of those replicates.  The mean of the replicates is never emitted,
returned or written.  A power calculation needs the variance; looking at the
mean would be peeking.

Why not a hand-written formula, and why not a permutation
    Both were tried on 2026-09-06 and both were wrong, in the same direction:
      · the analytic SE used the per-bar sigma of 1h returns (0.702) instead of
        the event-conditional one (1.20), and ignored clustering -> 0.025
        against a true 0.23, a factor of NINE;
      · the permutation null looked right (all CIs covered zero) but it
        destroys the very within-day dependence that inflates the SE, so it
        reports an optimistic resolution.  It validates the machine, not the
        power.
    So: real labels, the real estimator, the real resampling scheme.

Clustering
    Days are the cluster, pooled ACROSS coins: 2025-10-10 hit all nine at once,
    so clustering per (coin, day) would understate the dependence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STATE = HERE / "data" / "state"
LABELS = HERE / "data" / "labels"
OUT = HERE / "data" / "quality"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
B = 2000
RNG = np.random.default_rng(20260906)


def load():
    frames = []
    for s in CORE9:
        st = pd.read_parquet(STATE / f"{s}.parquet")
        lb = pd.read_parquet(LABELS / f"{s}.parquet")
        m = st.merge(lb.drop(columns=["day"]), on="event_id", how="inner")
        m["sym"] = s
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def delta_pooled(poc, trend, y):
    """Tercile-by-trend, within-bucket q75-vs-q25 difference, n-weighted."""
    if len(y) < 60:
        return np.nan
    t1, t2 = np.quantile(trend, [1 / 3, 2 / 3])
    num, den = 0.0, 0
    for lo, hi in ((-np.inf, t1), (t1, t2), (t2, np.inf)):
        m = (trend >= lo) & (trend < hi)
        if m.sum() < 20:
            continue
        p, yy = poc[m], y[m]
        q25, q75 = np.quantile(p, [0.25, 0.75])
        a, b = p >= q75, p <= q25
        if a.sum() < 5 or b.sum() < 5:
            continue
        num += (yy[a].mean() - yy[b].mean()) * m.sum()
        den += m.sum()
    return num / den if den else np.nan


def se_of(df, poc_col, y_col, b=B):
    """Day-clustered bootstrap SPREAD of delta_pooled.  Returns SE only."""
    d = df.dropna(subset=[poc_col, y_col, "trend_24h"])
    days = d["day"].to_numpy()
    uniq, inv = np.unique(days, return_inverse=True)
    idx_by_day = [np.where(inv == k)[0] for k in range(len(uniq))]
    poc = d[poc_col].to_numpy(float)
    tr = d["trend_24h"].to_numpy(float)
    y = d[y_col].to_numpy(float)
    reps = np.empty(b)
    for i in range(b):
        pick = RNG.integers(0, len(uniq), len(uniq))
        ix = np.concatenate([idx_by_day[k] for k in pick])
        reps[i] = delta_pooled(poc[ix], tr[ix], y[ix])
    reps = reps[np.isfinite(reps)]
    # centred spread only; the location is deliberately not returned
    return dict(n=int(len(d)), n_days=int(len(uniq)), n_reps=int(len(reps)),
                se=float(np.std(reps, ddof=1)),
                ci_halfwidth_95=float((np.percentile(reps, 97.5)
                                       - np.percentile(reps, 2.5)) / 2))


def beta_se(df, poc_col, y_col):
    """Cluster-robust SE of the poc coefficient.  Returns the SE, not beta."""
    import statsmodels.api as sm
    d = df.dropna(subset=[poc_col, y_col, "trend_24h", "er_24h", "rv_1h",
                          "rv_24h", "pierce_atr",
                          poc_col.replace("poc_dist", "vwap_dist")])
    X = pd.DataFrame({
        "poc": d[poc_col].to_numpy(float),
        "vwap": d[poc_col.replace("poc_dist", "vwap_dist")].to_numpy(float),
        "trend_24h": d["trend_24h"].to_numpy(float),
        "er_24h": d["er_24h"].to_numpy(float),
        "rv_1h": d["rv_1h"].to_numpy(float),
        "rv_24h": d["rv_24h"].to_numpy(float),
        "pierce_atr": d["pierce_atr"].to_numpy(float)}, index=d.index)
    X = pd.concat([X, pd.get_dummies(d["session"], prefix="s",
                                     drop_first=True).astype(float),
                   pd.get_dummies(d["sym"], prefix="c",
                                  drop_first=True).astype(float)], axis=1)
    X = sm.add_constant(X)
    fit = sm.OLS(d[y_col].to_numpy(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": d["day"].values})
    return dict(n=int(len(d)), se_beta_poc=float(fit.bse["poc"]),
                se_beta_vwap=float(fit.bse["vwap"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=int, default=3600)
    ap.add_argument("--lookback", default="L2")
    a = ap.parse_args()
    df = load()
    y = f"r_norm_{a.tau}"
    poc = f"poc_dist_{a.lookback}"
    out = {"tau": a.tau, "lookback": a.lookback, "B": B,
           "cluster": "UTC day, pooled across coins",
           "note": "spreads only; no point estimate is computed for display"}
    print(f"pooled events={len(df):,}  UTC days={df.day.nunique():,}  "
          f"coins={df.sym.nunique()}")
    print(f"\ndelta estimator, tau={a.tau}s, {a.lookback}, day-clustered "
          f"bootstrap B={B}\n")
    print(f"{'side':10s} {'n':>7s} {'days':>6s} {'SE':>8s} {'CI half':>9s} "
          f"{'MDE(CI>0)':>10s} {'MDE(80% power)':>15s}")
    for side in ("sellside", "buyside"):
        s = se_of(df[df.side == side], poc, y)
        mde = 1.96 * s["se"]
        mde80 = 2.80 * s["se"]
        out[side] = dict(**s, mde_ci_excludes_zero=mde, mde_80pct_power=mde80)
        print(f"{side:10s} {s['n']:7,d} {s['n_days']:6,d} {s['se']:8.4f} "
              f"{s['ci_halfwidth_95']:9.4f} {mde:10.4f} {mde80:15.4f}")
    print("\nregression coefficient, cluster-robust by day "
          "(coin fixed effects included)\n")
    print(f"{'side':10s} {'n':>7s} {'SE(beta_poc)':>13s} {'|t|=2.5 needs':>14s}")
    for side in ("sellside", "buyside"):
        r = beta_se(df[df.side == side], poc, y)
        out[side + "_reg"] = dict(**r, beta_for_t25=2.5 * r["se_beta_poc"])
        print(f"{side:10s} {r['n']:7,d} {r['se_beta_poc']:13.4f} "
              f"{2.5 * r['se_beta_poc']:14.4f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"stage6_power_{a.lookback}_{a.tau}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nwritten ->", OUT / f"stage6_power_{a.lookback}_{a.tau}.json")


if __name__ == "__main__":
    main()
