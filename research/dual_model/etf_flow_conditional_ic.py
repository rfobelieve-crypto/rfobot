"""ETF-flow CONDITIONAL IC test (2026-06-06).

QUESTION: ETF flow is the one genuinely orthogonal buy-side signal that is
ENGINEERED (cg_etf_flow_daily) but NOT in either production model (v7/v9, 136
feats). Does it carry information v7 does NOT already have -- specifically about
the UP/bull moves v7 fades?

METHOD (mistake.md 2026-06-01): the WRONG test is raw IC(feature, target) -- it
just measures correlation with the target, most of which v7 may already absorb.
The RIGHT test is CONDITIONAL IC = Spearman(feature, v7_residual), residual =
y_path_ret_4h - pred_ret. Only if ETF flow predicts what v7 MISSED is it marginal
alpha worth an ensemble A/B.

Checks:
  1. raw IC vs conditional IC per ETF feature (+ bootstrap CI on conditional IC)
  2. per-month stability of the best feature's conditional IC (frac same-sign)
  3. BULL SLICE: conditional IC on rising-market bars (ret24>+0.5%) -- the exact
     regime where v7 bleeds. This is the bull-relevant test.
  4. daily effective-n caveat: ETF flow is DAILY (carried across 1h bars), so
     bar-level IC OVERSTATES significance; report n_unique_days.

A conditional IC that is (a) significant (CI excl 0), (b) per-month stable, and
(c) present in the bull slice => worth an ensemble A/B. Otherwise NO-GO.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OOS_PATH = "research/results/dual_model/direction_reg_oos_mse.parquet"
CACHE = "research/dual_model/.cache/features_all.parquet"
ETF_FEATS = [
    "etf_net_flow_usd", "etf_flow_3d_sum", "etf_flow_7d_sum",
    "etf_flow_zscore_30d", "etf_flow_sign_persistence", "etf_flow_delta_1d",
    "etf_flow_ibit", "etf_flow_fbtc", "etf_flow_gbtc",
]


def boot_ic(x, y, n=3000, seed=1):
    rng = np.random.default_rng(seed)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    ics = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        ic, _ = spearmanr(x[idx], y[idx])
        ics.append(ic)
    return np.percentile(ics, [2.5, 97.5])


def main():
    oos = pd.read_parquet(OOS_PATH).sort_index()
    feat = pd.read_parquet(CACHE).copy()
    feat["ret24"] = feat["close"].pct_change(24)

    present = [c for c in ETF_FEATS if c in feat.columns]
    missing = [c for c in ETF_FEATS if c not in feat.columns]
    print(f"ETF features present in cache: {present}")
    if missing:
        print(f"MISSING from cache: {missing}")
    if not present:
        print("No ETF features in cache -- abort."); return

    df = oos.join(feat[present + ["ret24"]], how="inner")
    df["resid"] = df["y_path_ret_4h"] - df["pred_ret"]
    y = df["y_path_ret_4h"].values
    resid = df["resid"].values

    # daily effective-n
    n_days = df.index.normalize().nunique()
    print(f"\nrows={len(df)}  unique days={n_days}  "
          f"(ETF flow is DAILY -> effective n ~= days, NOT rows)")
    print(f"v7 residual std={resid.std():.5f}  (the variance ETF must explain)\n")

    print(f"{'feature':26s} {'raw_IC':>8s} {'cond_IC':>8s} "
          f"{'cond_CI(95%)':>18s} {'sig?':>5s}")
    best = (None, 0.0)
    for c in present:
        x = df[c].values.astype(float)
        m = np.isfinite(x) & np.isfinite(resid)
        if m.sum() < 50 or np.nanstd(x[m]) < 1e-12:
            print(f"{c:26s}  (insufficient/constant)"); continue
        raw, _ = spearmanr(x[m], y[m])
        cond, _ = spearmanr(x[m], resid[m])
        lo, hi = boot_ic(x, resid)
        sig = "YES" if (lo > 0 or hi < 0) else "no"
        print(f"{c:26s} {raw:+8.3f} {cond:+8.3f}  [{lo:+.3f},{hi:+.3f}] {sig:>5s}")
        if abs(cond) > abs(best[1]):
            best = (c, cond)

    bc = best[0]
    if bc is None:
        print("\nno usable feature -- abort."); return
    print(f"\nstrongest |cond_IC| feature: {bc} ({best[1]:+.3f})")

    # per-month stability
    df["mon"] = df.index.to_period("M").astype(str)
    print(f"\nper-month conditional IC of {bc} (stability check):")
    signs = []
    for mo, g in df.groupby("mon"):
        x = g[bc].values.astype(float); rr = g["resid"].values
        mm = np.isfinite(x) & np.isfinite(rr)
        if mm.sum() < 20 or np.nanstd(x[mm]) < 1e-12:
            print(f"  {mo}: n={mm.sum()} (thin)"); continue
        ic, _ = spearmanr(x[mm], rr[mm])
        signs.append(np.sign(ic))
        print(f"  {mo}: cond_IC={ic:+.3f}  n={mm.sum()}")
    if signs:
        same = max(np.mean(np.array(signs) > 0), np.mean(np.array(signs) < 0)) * 100
        print(f"  same-sign months: {same:.0f}%  (need >70% for a stable signal)")

    # BULL SLICE: where v7 bleeds (rising market)
    rb = df[df["ret24"] > 0.005]
    print(f"\nBULL SLICE (rising market, ret24>+0.5%, n={len(rb)}): "
          f"does ETF flow explain v7's residual where it FADES the rally?")
    if len(rb) >= 50:
        for c in present:
            x = rb[c].values.astype(float); rr = rb["resid"].values
            mm = np.isfinite(x) & np.isfinite(rr)
            if mm.sum() < 30 or np.nanstd(x[mm]) < 1e-12:
                continue
            ic, p = spearmanr(x[mm], rr[mm])
            print(f"  {c:26s} cond_IC={ic:+.3f}  p={p:.3f}  n={mm.sum()}")
    else:
        print("  too few rising-market bars to test")

    print("\n" + "=" * 60)
    print("READ: cond_IC near 0 / CI spans 0 / not month-stable / absent in bull")
    print("slice => ETF flow adds nothing v7 lacks at 4h -> NO-GO (consistent with")
    print("daily-frequency headwind). A real signal here would justify ensemble A/B.")


if __name__ == "__main__":
    main()
