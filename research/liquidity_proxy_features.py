"""Phase 1.5 — Liquidity-seeking proxy features using 60-day BTC history.

Avoids the "only 20 days of L20 orderbook" limitation by using
indicator_history (59d) + flow_bars_1m (62d) + liquidation_1m (62d)
to compute price-action / volume-profile proxies for the same
"price moves to least resistance" concept.

8 candidate features (all trailing-only):
  A. Swing-high/low distance     (4h, 24h, 48h windows)
  B. Volume-profile HVN distance (past 24h / 168h)
  C. Round-number proximity      ($500 / $1000 bins)
  D. Anchored VWAP from swing
  E. Liq cluster z-score         (60-day baseline)
  F. Sweep+reclaim event flag    (high then close back / low then close back)
  G. Depth-snapshots imbalance trend
  H. Range expansion / vacuum    (TR / 24h-ATR, volume-void)

Output:
  research/results/liq_proxy_walkforward_ic.csv
  research/results/liq_proxy_corr_matrix.csv

Walk-forward: 30-day train / 7-day OOS / slide-by-7d.  Per fold per
feature: Spearman IC on the OOS chunk only.  A feature passes if
its across-fold mean |IC| > 0.05 AND |IC| > 0 in ≥ 60% of folds.

Per CLAUDE.md mistake log 2026-04-13: walk-forward is REQUIRED for any
new feature claim.  In-sample IC is meaningless.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import spearmanr

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.db import get_db_conn  # noqa: E402


HORIZON_H = 4
RESULTS_DIR = Path("research/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loaders ──────────────────────────────────────────────────────


def load_indicator_history() -> pd.DataFrame:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT dt, open, high, low, close
                FROM indicator_history
                ORDER BY dt ASC
            """)
            rows = cur.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"], utc=True)
    return df.set_index("dt").astype(float)


def load_flow_bars() -> pd.DataFrame:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT window_start, buy_notional_usd, sell_notional_usd,
                       volume_usd, delta_usd
                FROM flow_bars_1m
                WHERE canonical_symbol='BTC-USD'
                ORDER BY window_start ASC
            """)
            rows = cur.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["window_start"], unit="ms", utc=True)
    for c in ("buy_notional_usd", "sell_notional_usd",
              "volume_usd", "delta_usd"):
        df[c] = df[c].astype(float)
    return df.set_index("ts").drop(columns=["window_start"])


def load_liq() -> pd.DataFrame:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT window_start, liq_buy_usd, liq_sell_usd,
                       liq_total_usd, liq_count
                FROM liquidation_1m
                WHERE canonical_symbol='BTC-USD'
                ORDER BY window_start ASC
            """)
            rows = cur.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["window_start"], unit="ms", utc=True)
    for c in ("liq_buy_usd", "liq_sell_usd", "liq_total_usd"):
        df[c] = df[c].astype(float)
    return df.set_index("ts").drop(columns=["window_start"])


def load_depth_snaps() -> pd.DataFrame:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT dt, depth_imbalance, near_imbalance, spread_bps
                FROM indicator_depth_snapshots
                ORDER BY dt ASC
            """)
            rows = cur.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"], utc=True)
    for c in ("depth_imbalance", "near_imbalance", "spread_bps"):
        df[c] = df[c].astype(float)
    return df.set_index("dt")


# ── Feature engineering ───────────────────────────────────────────────


def feat_swing_distance(ih: pd.DataFrame) -> pd.DataFrame:
    """A — distance from current close to past swing high / low.

    Positive value = swing-high above (resistance), price has room up.
    Larger A_swing_high_dist_Nh = more resistance distance UP.
    Larger A_swing_low_dist_Nh = more support distance DOWN.
    """
    out = pd.DataFrame(index=ih.index)
    for w in (4, 24, 48, 168):
        rolling_high = ih["high"].rolling(w).max()
        rolling_low = ih["low"].rolling(w).min()
        out[f"A_swing_high_dist_{w}h"] = (rolling_high - ih["close"]) / ih["close"]
        out[f"A_swing_low_dist_{w}h"] = (ih["close"] - rolling_low) / ih["close"]
    return out


def feat_volume_profile_hvn(flow: pd.DataFrame,
                              ih: pd.DataFrame,
                              window_h: int = 24,
                              n_bins: int = 50) -> pd.DataFrame:
    """B — distance from current close to nearest high-volume node (HVN)
    in trailing window.  Approximation: bucket trailing minute prices by
    notional volume, find top bucket (VPOC), compute distance from now.
    """
    # Reindex flow to hourly bars for efficiency
    out = pd.DataFrame(index=ih.index)
    # Per hourly bar, look back `window_h` hours of minute data
    prices = flow.index.values
    vols = flow["volume_usd"].values

    # We need a price column on flow_bars_1m — but flow_bars_1m doesn't
    # have price.  Approximate using mid_price from indicator_history
    # interpolated; OR use buy/sell notional ratio.  For now skip this
    # feature and emit NaN — needs a price-per-minute join we don't have.
    out[f"B_vpoc_dist_{window_h}h"] = np.nan
    return out


def feat_round_proximity(ih: pd.DataFrame) -> pd.DataFrame:
    """C — distance to nearest round $1k / $500 in % terms.

    Stops often cluster at psychological levels.  Smaller distance =
    closer to a stop pool.
    """
    out = pd.DataFrame(index=ih.index)
    c = ih["close"]
    out["C_round_1k_dist_pct"] = abs(c - (c / 1000).round() * 1000) / c
    out["C_round_500_dist_pct"] = abs(c - (c / 500).round() * 500) / c
    return out


def feat_anchored_vwap(ih: pd.DataFrame,
                        flow: pd.DataFrame,
                        window_h: int = 48) -> pd.DataFrame:
    """D — distance from current close to volume-weighted average of
    typical price (HLC/3) since last `window_h`-bar swing high/low.

    Practically: rolling VWAP of typical price over `window_h` bars
    (proxy for AVWAP-from-swing without separate anchor detection).
    """
    out = pd.DataFrame(index=ih.index)
    typ = (ih["high"] + ih["low"] + ih["close"]) / 3
    # No 1h-aggregated volume yet — resample minute flow to hourly
    vol_h = flow["volume_usd"].resample("1H", label="right",
                                          closed="right").sum()
    vol_h = vol_h.reindex(ih.index, method="nearest",
                           tolerance=pd.Timedelta("30min")).fillna(0)
    typ_vol = (typ * vol_h).rolling(window_h, min_periods=4).sum()
    vol_sum = vol_h.rolling(window_h, min_periods=4).sum()
    vwap = typ_vol / vol_sum.replace(0, np.nan)
    out[f"D_avwap_dist_{window_h}h"] = (ih["close"] - vwap) / ih["close"]
    return out


def feat_liq_cluster(liq: pd.DataFrame,
                      ih: pd.DataFrame) -> pd.DataFrame:
    """E — z-score of 4h trailing liquidation intensity vs 7-day baseline."""
    out = pd.DataFrame(index=ih.index)
    liq_h = liq["liq_total_usd"].resample("1H", label="right",
                                            closed="right").sum()
    liq_h = liq_h.reindex(ih.index, method="nearest",
                            tolerance=pd.Timedelta("30min")).fillna(0)
    trail_4h = liq_h.rolling(4).sum()
    base_mean = liq_h.rolling(168).mean()
    base_std = liq_h.rolling(168).std()
    out["E_liq_z_4h_vs_7d"] = (trail_4h / 4 - base_mean) / (base_std + 1)
    # Asymmetry on 4h
    liq_b_h = liq["liq_buy_usd"].resample("1H", label="right",
                                            closed="right").sum()
    liq_s_h = liq["liq_sell_usd"].resample("1H", label="right",
                                            closed="right").sum()
    liq_b_h = liq_b_h.reindex(ih.index, method="nearest",
                                tolerance=pd.Timedelta("30min")).fillna(0)
    liq_s_h = liq_s_h.reindex(ih.index, method="nearest",
                                tolerance=pd.Timedelta("30min")).fillna(0)
    out["E_liq_asym_4h"] = np.log(
        (liq_s_h.rolling(4).sum() + 1)
        / (liq_b_h.rolling(4).sum() + 1)
    )
    return out


def feat_sweep_reclaim(ih: pd.DataFrame) -> pd.DataFrame:
    """F — sweep+reclaim events.

    Bullish sweep: low went BELOW past-24h low, but close > past-24h low
                   (took out lows, reclaimed level).
    Bearish sweep: high went ABOVE past-24h high, but close < past-24h high.
    Magnitude = (extreme - reclaim_level) / close.
    """
    out = pd.DataFrame(index=ih.index)
    past_high = ih["high"].shift(1).rolling(24).max()
    past_low = ih["low"].shift(1).rolling(24).min()

    # Bullish sweep (took out past low, reclaimed) → expected positive return
    bull_sweep = (ih["low"] < past_low) & (ih["close"] > past_low)
    out["F_bull_sweep_mag"] = np.where(
        bull_sweep,
        (past_low - ih["low"]) / ih["close"],
        0.0,
    )
    # Bearish sweep
    bear_sweep = (ih["high"] > past_high) & (ih["close"] < past_high)
    out["F_bear_sweep_mag"] = np.where(
        bear_sweep,
        (ih["high"] - past_high) / ih["close"],
        0.0,
    )
    return out


def feat_depth_imbalance(depth: pd.DataFrame,
                           ih: pd.DataFrame) -> pd.DataFrame:
    """G — depth snapshots imbalance trend (59-day history)."""
    out = pd.DataFrame(index=ih.index)
    di = depth["depth_imbalance"].reindex(ih.index, method="nearest",
                                            tolerance=pd.Timedelta("30min"))
    ni = depth["near_imbalance"].reindex(ih.index, method="nearest",
                                            tolerance=pd.Timedelta("30min"))
    out["G_depth_imb_4h_mean"] = di.rolling(4).mean()
    out["G_near_imb_4h_mean"] = ni.rolling(4).mean()
    out["G_depth_imb_24h_z"] = (di - di.rolling(168).mean()) / (
        di.rolling(168).std() + 1e-6
    )
    return out


def feat_range_vacuum(ih: pd.DataFrame,
                       flow: pd.DataFrame) -> pd.DataFrame:
    """H — true-range / 24h-ATR ratio + volume-void score.

    Vacuum bar = high range, low volume = liquidity void = potential continuation.
    """
    out = pd.DataFrame(index=ih.index)
    pc = ih["close"].shift(1)
    tr = pd.concat([ih["high"] - ih["low"], (ih["high"] - pc).abs(),
                     (ih["low"] - pc).abs()], axis=1).max(axis=1)
    atr24 = tr.rolling(24).mean()
    out["H_tr_atr24_ratio"] = tr / (atr24 + 1)

    vol_h = flow["volume_usd"].resample("1H", label="right",
                                          closed="right").sum()
    vol_h = vol_h.reindex(ih.index, method="nearest",
                           tolerance=pd.Timedelta("30min")).fillna(0)
    vol_med_24h = vol_h.rolling(24).median()
    # Vacuum: high range but BELOW median volume
    out["H_vacuum_bar"] = (
        (tr > atr24 * 1.5) & (vol_h < vol_med_24h)
    ).astype(float) * out["H_tr_atr24_ratio"]
    return out


def compute_target(ih: pd.DataFrame,
                    horizon_h: int = HORIZON_H) -> pd.Series:
    c = ih["close"].astype(float)
    fwd_mean = c.shift(-1).rolling(horizon_h).mean().shift(-(horizon_h - 1))
    return (fwd_mean / c - 1.0).rename("y_path_ret_4h")


# ── Walk-forward IC ──────────────────────────────────────────────────


def walk_forward_ic(feat_df: pd.DataFrame,
                     target: pd.Series,
                     train_days: int = 30,
                     oos_days: int = 7) -> pd.DataFrame:
    """Per-feature: rolling 30-day train (just for index alignment) +
    7-day OOS Spearman IC.  Slide forward by oos_days.

    Returns DataFrame indexed by feature name with columns:
      n_folds, mean_ic, std_ic, frac_positive, frac_significant
    """
    aligned = feat_df.join(target.rename("y"), how="inner").dropna(how="all")
    if aligned.empty:
        return pd.DataFrame()

    start = aligned.index[0] + pd.Timedelta(days=train_days)
    end = aligned.index[-1] - pd.Timedelta(days=oos_days)
    if start >= end:
        return pd.DataFrame()

    fold_rows: dict[str, list[float]] = {c: [] for c in feat_df.columns}
    cur = start
    n_folds = 0
    while cur < end:
        fold_end = cur + pd.Timedelta(days=oos_days)
        oos = aligned[(aligned.index >= cur) & (aligned.index < fold_end)]
        if len(oos) < 20:
            cur = fold_end
            continue
        n_folds += 1
        for c in feat_df.columns:
            pair = oos[[c, "y"]].dropna()
            if len(pair) < 10 or pair[c].std() < 1e-12:
                fold_rows[c].append(np.nan)
                continue
            ic, _ = spearmanr(pair[c], pair["y"])
            fold_rows[c].append(float(ic) if not np.isnan(ic) else np.nan)
        cur = fold_end

    summary = []
    for c, ics in fold_rows.items():
        arr = np.array([x for x in ics if not np.isnan(x)])
        if len(arr) == 0:
            summary.append({"feature": c, "n_folds": 0, "mean_ic": np.nan,
                            "std_ic": np.nan, "frac_positive": np.nan,
                            "frac_signif": np.nan, "passes": False})
            continue
        mean = arr.mean()
        std = arr.std()
        frac_pos = (arr > 0).mean() if mean > 0 else (arr < 0).mean()
        # significance at fold level: |IC| > 0.1 (rough threshold for n~30/fold)
        frac_signif = (np.abs(arr) > 0.1).mean()
        passes = (abs(mean) > 0.05) and (frac_pos > 0.6)
        summary.append({
            "feature": c,
            "n_folds": len(arr),
            "mean_ic": float(mean),
            "std_ic": float(std),
            "frac_positive": float(frac_pos),
            "frac_signif": float(frac_signif),
            "passes": bool(passes),
        })
    return (pd.DataFrame(summary)
            .assign(abs_mean=lambda d: d["mean_ic"].abs())
            .sort_values("abs_mean", ascending=False)
            .drop(columns=["abs_mean"]))


# ── Main ──────────────────────────────────────────────────────────────


def main() -> int:
    print("Loading data…")
    ih = load_indicator_history()
    flow = load_flow_bars()
    liq = load_liq()
    depth = load_depth_snaps()
    print(f"  indicator_history: {len(ih)} bars  "
          f"({ih.index.min()} → {ih.index.max()})")
    print(f"  flow_bars_1m:     {len(flow)} bars")
    print(f"  liquidation_1m:   {len(liq)} events")
    print(f"  depth snapshots:  {len(depth)} bars")

    print("\nComputing features…")
    fA = feat_swing_distance(ih)
    fB = feat_volume_profile_hvn(flow, ih)
    fC = feat_round_proximity(ih)
    fD = feat_anchored_vwap(ih, flow)
    fE = feat_liq_cluster(liq, ih)
    fF = feat_sweep_reclaim(ih)
    fG = feat_depth_imbalance(depth, ih)
    fH = feat_range_vacuum(ih, flow)
    feats = pd.concat([fA, fB, fC, fD, fE, fF, fG, fH], axis=1)
    print(f"  Total features: {feats.shape[1]}")
    print(f"  Non-NaN coverage: {(~feats.isna()).sum().to_dict()}")

    target = compute_target(ih, HORIZON_H)
    print(f"  Target valid bars: {target.notna().sum()}")

    print("\nWalk-forward IC (30d train / 7d OOS, slide 7d)…")
    wf = walk_forward_ic(feats, target)
    wf_path = RESULTS_DIR / "liq_proxy_walkforward_ic.csv"
    wf.to_csv(wf_path, index=False)
    print(f"  Wrote {wf_path}")

    print("\n=== Walk-forward results (top by |mean_ic|) ===")
    for _, r in wf.iterrows():
        mark = " ★ PASS" if r["passes"] else ""
        print(f"  {r['feature']:32s}  mean_IC={r['mean_ic']:+.4f}  "
              f"std={r['std_ic']:.3f}  folds={int(r['n_folds'])}  "
              f"frac_pos={r['frac_positive']:.2f}{mark}")

    passing = wf[wf["passes"]]
    print(f"\n{len(passing)} feature(s) pass the criteria "
          "(|mean_IC| > 0.05 AND consistent sign in > 60% of folds)")

    # Correlation matrix among features
    corr = feats.dropna(how="all").corr(method="spearman")
    corr_path = RESULTS_DIR / "liq_proxy_corr_matrix.csv"
    corr.to_csv(corr_path)
    print(f"\nWrote correlation matrix → {corr_path}")

    if not passing.empty:
        print("\nCross-correlation among passing features:")
        cols = passing["feature"].tolist()
        sub = corr.loc[cols, cols]
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                rc = sub.loc[c1, c2]
                if abs(rc) > 0.5:
                    print(f"  ⚠ {c1} ↔ {c2}  corr={rc:+.2f} "
                          "(redundant, keep only one)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
