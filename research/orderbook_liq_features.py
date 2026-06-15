"""Phase 1 liquidity-seeking features — orderbook + liquidation IC sweep.

Computes ~16 trailing 1h aggregates from:
  - orderbook_snapshots_1m (Binance L20 depth, walls, spread)
  - liquidation_1m         (BTC-USD aggregate forced flow)

Joins each feature to BTC TWAP path-return target (y_path_ret_4h)
computed from BTC 1h OHLC, then evaluates each feature's Spearman IC
with bootstrap 95% CI.

Per CLAUDE.md mistake log 2026-04-13: walk-forward only — every test
bar uses only data observable AT that bar.  Aggregates are trailing
windows of raw 1m data through the bar's close time.

Output: research/results/ob_liq_features_ic.csv (sorted by |IC|)
        printed top-N to stdout.

Usage:
    python research/orderbook_liq_features.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import spearmanr

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.db import get_db_conn  # noqa: E402


HORIZON_HOURS = 4   # match V7 target
SYMBOL = "BTC-USD"
RESULTS_PATH = Path("research/results/ob_liq_features_ic.csv")
N_BOOTSTRAP = 500


# ── Data loaders ──────────────────────────────────────────────────────


def load_orderbook(symbol: str = SYMBOL) -> pd.DataFrame:
    """Load 1m orderbook snapshots for BTC-USD over the full available
    window.  Returns DataFrame indexed by UTC datetime."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ts_ms, mid_price, spread_bps,
                       imbalance_l5, imbalance_l20,
                       bid_depth_usd_l5, ask_depth_usd_l5,
                       bid_depth_usd_l20, ask_depth_usd_l20,
                       bid_max_wall_usd, ask_max_wall_usd,
                       bid_wall_distance_bps, ask_wall_distance_bps
                FROM orderbook_snapshots_1m
                WHERE canonical_symbol = %s
                ORDER BY ts_ms ASC
            """, (symbol,))
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.set_index("ts").drop(columns=["ts_ms"])
    return df


def load_liquidations(symbol: str = SYMBOL) -> pd.DataFrame:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT window_start, liq_buy_usd, liq_sell_usd,
                       liq_total_usd, liq_count
                FROM liquidation_1m
                WHERE canonical_symbol = %s
                ORDER BY window_start ASC
            """, (symbol,))
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["window_start"], unit="ms", utc=True)
    for c in ("liq_buy_usd", "liq_sell_usd", "liq_total_usd"):
        df[c] = df[c].astype(float)
    return df.set_index("ts").drop(columns=["window_start"])


def load_btc_closes() -> pd.DataFrame:
    """1h BTC close prices for target computation.  Source: indicator_history
    (has BTC close per hourly bar)."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT dt, close
                FROM indicator_history
                ORDER BY dt ASC
            """)
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"], utc=True)
    return df.set_index("dt")


# ── Feature engineering ───────────────────────────────────────────────


def compute_orderbook_features(ob: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1m orderbook snapshots into hourly trailing features."""
    if ob.empty:
        return pd.DataFrame()
    # Resample to 1h boundaries; values are computed from ALL 1m rows
    # within the trailing hour.
    h = ob.resample("1H", label="right", closed="right")
    feats = pd.DataFrame(index=h.first().index)

    feats["ob_imbalance_l5_mean_1h"] = h["imbalance_l5"].mean()
    feats["ob_imbalance_l20_mean_1h"] = h["imbalance_l20"].mean()
    feats["ob_imbalance_l5_std_1h"] = h["imbalance_l5"].std()
    feats["ob_imbalance_l20_std_1h"] = h["imbalance_l20"].std()

    # Wall asymmetry — log ratio so symmetric around 0
    feats["ob_wall_asym_log_1h"] = (
        np.log((h["ask_max_wall_usd"].mean() + 1)
                / (h["bid_max_wall_usd"].mean() + 1))
    )
    feats["ob_wall_dist_min_1h"] = (
        h[["bid_wall_distance_bps", "ask_wall_distance_bps"]].min().min(axis=1)
    )

    feats["ob_depth_ratio_l20_1h"] = (
        h["ask_depth_usd_l20"].sum()
        / (h["bid_depth_usd_l20"].sum() + 1)
    )
    feats["ob_spread_mean_1h"] = h["spread_bps"].mean()
    feats["ob_spread_max_1h"] = h["spread_bps"].max()
    return feats


def compute_liq_features(liq: pd.DataFrame) -> pd.DataFrame:
    if liq.empty:
        return pd.DataFrame()
    h = liq.resample("1H", label="right", closed="right")
    feats = pd.DataFrame(index=h.first().index)

    feats["liq_buy_usd_1h"] = h["liq_buy_usd"].sum()
    feats["liq_sell_usd_1h"] = h["liq_sell_usd"].sum()
    feats["liq_total_usd_1h"] = h["liq_total_usd"].sum()
    feats["liq_count_1h"] = h["liq_count"].sum()

    feats["liq_asym_log_1h"] = np.log(
        (feats["liq_sell_usd_1h"] + 1) / (feats["liq_buy_usd_1h"] + 1)
    )
    # 24h trailing z-score for total liquidation intensity
    rolling_mean = feats["liq_total_usd_1h"].rolling(24, min_periods=6).mean()
    rolling_std = feats["liq_total_usd_1h"].rolling(24, min_periods=6).std()
    feats["liq_total_z_24h"] = (
        (feats["liq_total_usd_1h"] - rolling_mean) / (rolling_std + 1)
    )
    return feats


def compute_target(btc: pd.DataFrame,
                    horizon_h: int = HORIZON_HOURS) -> pd.Series:
    """y_path_ret_4h = mean(close[t+1..t+H]) / close[t] - 1"""
    c = btc["close"].astype(float)
    fwd_mean = c.shift(-1).rolling(horizon_h).mean().shift(-(horizon_h - 1))
    return (fwd_mean / c - 1.0).rename("y_path_ret_4h")


# ── IC evaluation ─────────────────────────────────────────────────────


def spearman_ic_bootstrap(x: pd.Series, y: pd.Series,
                           n_boot: int = N_BOOTSTRAP,
                           seed: int = 42) -> dict:
    """Spearman IC + bootstrap CI on aligned non-NaN pairs."""
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 30:
        return {"n": len(df), "ic": np.nan, "ci_lo": np.nan,
                "ci_hi": np.nan, "sig": False}
    ic, _ = spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    rng = np.random.default_rng(seed)
    boots = []
    n = len(df)
    arr = df.values
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = arr[idx]
        b_ic, _ = spearmanr(sample[:, 0], sample[:, 1])
        if not np.isnan(b_ic):
            boots.append(b_ic)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sig = (lo > 0) or (hi < 0)
    return {"n": int(n), "ic": float(ic), "ci_lo": float(lo),
            "ci_hi": float(hi), "sig": sig}


# ── Main ──────────────────────────────────────────────────────────────


def main() -> int:
    print(f"Loading data for {SYMBOL}...")
    ob = load_orderbook()
    liq = load_liquidations()
    btc = load_btc_closes()
    print(f"  orderbook: {len(ob):,} rows  "
          f"({ob.index.min() if not ob.empty else 'empty'} → "
          f"{ob.index.max() if not ob.empty else 'empty'})")
    print(f"  liquidations: {len(liq):,} rows  "
          f"({liq.index.min() if not liq.empty else 'empty'} → "
          f"{liq.index.max() if not liq.empty else 'empty'})")
    print(f"  btc closes: {len(btc):,} rows")

    print("\nComputing features...")
    ob_feats = compute_orderbook_features(ob)
    liq_feats = compute_liq_features(liq)
    print(f"  orderbook features: {ob_feats.shape}")
    print(f"  liquidation features: {liq_feats.shape}")

    # Align on hourly index (BTC close index)
    target = compute_target(btc, HORIZON_HOURS)
    # Make BTC index UTC (indicator_history dt is naive UTC)
    if target.index.tz is None:
        target.index = target.index.tz_localize("UTC")

    all_feats = ob_feats.join(liq_feats, how="outer")
    print(f"  combined features: {all_feats.shape}")

    # Reindex feats to target's hourly grid (drop ones outside coverage)
    aligned = all_feats.reindex(target.index, method="nearest",
                                 tolerance=pd.Timedelta("30min"))
    print(f"  aligned to target: {aligned.dropna(how='all').shape}")

    print("\nEvaluating IC per feature (n_boot=500)...")
    results = []
    for col in aligned.columns:
        stats = spearman_ic_bootstrap(aligned[col], target)
        results.append({"feature": col, **stats})
        sig_mark = " ★" if stats["sig"] else ""
        ic = stats["ic"]
        ci = f"[{stats['ci_lo']:+.3f}, {stats['ci_hi']:+.3f}]"
        n = stats["n"]
        print(f"  {col:32s}  IC={ic:+.4f}  CI={ci}  n={n:5d}{sig_mark}")

    df = pd.DataFrame(results)
    df["abs_ic"] = df["ic"].abs()
    df = df.sort_values("abs_ic", ascending=False).drop(columns=["abs_ic"])

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nWrote {RESULTS_PATH}")

    sig = df[df["sig"]]
    print()
    if not sig.empty:
        print(f"=== {len(sig)} feature(s) significantly non-zero IC: ===")
        for _, r in sig.iterrows():
            print(f"  {r['feature']:32s}  IC={r['ic']:+.4f}  "
                  f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  n={r['n']}")
    else:
        print("No feature passed 95% CI > 0 test.")
        print("Top 5 by |IC| (not statistically significant):")
        for _, r in df.head(5).iterrows():
            print(f"  {r['feature']:32s}  IC={r['ic']:+.4f}  "
                  f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  n={r['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
