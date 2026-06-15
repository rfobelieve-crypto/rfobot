"""
Phase 1a: Microstructure feature engineering + initial IC sanity check.

Uses the 22 days of 1-min BTC-USD flow_bars_1m exported from Railway
(cached at research/cache/flow_bars_1m_btc_raw.json), aggregates to
4h bars aligned with the production feature parquet, and computes
Spearman IC vs path_ret_4h for ~15 candidate features.

Important: with only ~126 overlapping 4h bars the IC estimates are
noisy — this is a SANITY CHECK not a production validator. A proper
77-fold walk-forward validation requires ≥ 90 days of 1-min history.
Features with the right SIGN (aligned with the orderflow thesis) and
|IC| standing out from zero are candidates to keep and re-evaluate
when data accumulates.

Outputs: research/results/microstructure_phase1.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_JSON = Path("research/cache/flow_bars_1m_btc_raw.json")
FEATURES_CACHE = Path("research/dual_model/.cache/features_all.parquet")
OUT = Path("research/results/microstructure_phase1.json")

HORIZON = 4  # 4h bars = 4 × 1h


def load_1min() -> pd.DataFrame:
    d = json.load(open(RAW_JSON))
    df = pd.DataFrame(d["rows"])
    df.index = pd.to_datetime(df["window_start"], unit="ms", utc=True)
    df = df.sort_index()
    # Derived 1-min metrics
    total = df["buy_notional_usd"] + df["sell_notional_usd"]
    df["total_notional"] = total
    df["delta_ratio"] = df["delta_usd"] / df["volume_usd"].replace(0, np.nan)
    df["aggressive_buy_ratio"] = df["buy_notional_usd"] / total.replace(0, np.nan)
    df["delta_sign"] = np.sign(df["delta_usd"])
    return df


def compute_1min_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # CVD slopes — cumulative flow series
    df["cvd_1h_slope"] = (df["cvd_usd"] - df["cvd_usd"].shift(60)) / 60.0
    df["cvd_4h_slope"] = (df["cvd_usd"] - df["cvd_usd"].shift(240)) / 240.0

    # Volume surge: rolling 1-day p90 (1440 min) of volume_usd
    df["vol_p90_1d"] = df["volume_usd"].rolling(1440, min_periods=240).quantile(0.90)
    df["vol_surge"] = (df["volume_usd"] > df["vol_p90_1d"]).astype(float)
    df["vol_surge_with_imb"] = (
        (df["vol_surge"] == 1) & (df["delta_ratio"].abs() > 0.30)
    ).astype(float)

    # Delta persistence (same-sign streak)
    sign = df["delta_sign"].fillna(0)
    group_id = (sign != sign.shift()).cumsum()
    df["delta_streak_len"] = sign.groupby(group_id).cumcount() + 1
    df["delta_streak_signed"] = df["delta_streak_len"] * sign

    # 4h rolling stats (on 1-min bars, window=240)
    df["trade_count_mean_4h"] = df["trade_count"].rolling(240, min_periods=60).mean()
    df["trade_count_std_4h"] = df["trade_count"].rolling(240, min_periods=60).std()
    df["trade_count_zscore_4h"] = (
        (df["trade_count"] - df["trade_count_mean_4h"]) / df["trade_count_std_4h"]
    )
    df["abr_mean_4h"] = df["aggressive_buy_ratio"].rolling(240, min_periods=60).mean()
    df["abr_std_4h"] = df["aggressive_buy_ratio"].rolling(240, min_periods=60).std()
    df["abr_zscore_4h"] = (df["aggressive_buy_ratio"] - df["abr_mean_4h"]) / df["abr_std_4h"]

    return df


def resample_to_4h(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-min to 4h bars aligned to 00/04/08/12/16/20 UTC."""
    agg = {
        "buy_notional_usd": "sum", "sell_notional_usd": "sum",
        "delta_usd": "sum", "volume_usd": "sum", "trade_count": "sum",
        "cvd_usd": "last",
        "cvd_1h_slope": "last", "cvd_4h_slope": "last",
        "vol_surge": "max", "vol_surge_with_imb": "max",
        "delta_streak_signed": "last",
        "trade_count_zscore_4h": "last",
        "abr_zscore_4h": "last",
        "aggressive_buy_ratio": "mean",
        "delta_ratio": "mean",
    }
    df4 = df_1m.resample("4H", label="right", closed="right").agg(agg)

    # 4h-scope derived features
    tot = df4["buy_notional_usd"] + df4["sell_notional_usd"]
    df4["abr_4h"] = df4["buy_notional_usd"] / tot.replace(0, np.nan)
    df4["delta_ratio_4h"] = df4["delta_usd"] / df4["volume_usd"].replace(0, np.nan)
    df4["abr_mom_4h"] = df4["abr_4h"].diff()
    df4["delta_ratio_mom_4h"] = df4["delta_ratio_4h"].diff()
    df4["trade_count_mom_4h"] = df4["trade_count"].pct_change()
    df4["cvd_change_4h"] = df4["cvd_usd"].diff()

    # Price-CVD divergence proxy: sign of 4h cvd_change vs 4h delta_usd.
    # If cumulative CVD change positive but period delta negative,
    # indicates absorption/divergence.
    df4["cvd_vs_delta_divergence"] = np.sign(df4["cvd_change_4h"]) - np.sign(df4["delta_usd"])
    return df4


def align_with_label(df4: pd.DataFrame, features_cache_path: Path) -> pd.DataFrame:
    """Align 4h flow bars with production path_ret_4h label (TWAP)."""
    full = pd.read_parquet(features_cache_path)
    close = full["close"].astype(float)
    fwd_mean = sum(close.shift(-i) for i in range(1, HORIZON + 1)) / HORIZON
    y = pd.Series((fwd_mean / close - 1).values, index=full.index, name="y")

    # Ensure both are UTC
    if df4.index.tz is None:
        df4.index = df4.index.tz_localize("UTC")
    y_idx = y.index
    if y_idx.tz is None:
        y_idx = y_idx.tz_localize("UTC")
    y.index = y_idx

    merged = df4.join(y, how="inner")
    merged = merged.dropna(subset=["y"])
    return merged


def bootstrap_ic(x: np.ndarray, y: np.ndarray, n_boot: int = 500):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 30:
        return None
    ic, _ = spearmanr(x, y)
    rng = np.random.default_rng(42)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = spearmanr(x[idx], y[idx])
        boots[b] = r if r == r else 0.0
    return {
        "n": int(n),
        "ic": float(ic),
        "ci_lo": float(np.quantile(boots, 0.025)),
        "ci_hi": float(np.quantile(boots, 0.975)),
    }


FEATURE_COLS = [
    # CVD family
    "cvd_1h_slope", "cvd_4h_slope", "cvd_change_4h", "cvd_vs_delta_divergence",
    # Meta-order proxy
    "vol_surge", "vol_surge_with_imb", "delta_streak_signed",
    # Arrival rate
    "trade_count", "trade_count_zscore_4h", "trade_count_mom_4h",
    # Aggressive flow
    "aggressive_buy_ratio", "abr_zscore_4h", "abr_4h", "abr_mom_4h",
    "delta_ratio", "delta_ratio_4h", "delta_ratio_mom_4h",
]


def main():
    logger.info("Loading 1-min flow from cache...")
    df1 = load_1min()
    logger.info("Loaded: %d 1-min bars, %s ~ %s", len(df1),
                df1.index[0], df1.index[-1])

    logger.info("Computing 1-min derived features...")
    df1 = compute_1min_derived(df1)

    logger.info("Resampling to 4h...")
    df4 = resample_to_4h(df1)
    logger.info("4h bars: %d", len(df4))

    logger.info("Aligning with path_ret_4h label from features_all.parquet...")
    merged = align_with_label(df4, FEATURES_CACHE)
    logger.info("Aligned bars with label: %d", len(merged))

    if len(merged) < 30:
        logger.error("Too few aligned bars (%d) — abort", len(merged))
        return

    # IC per feature (overall)
    logger.info("Computing IC per feature (n=%d)", len(merged))
    results = []
    for col in FEATURE_COLS:
        if col not in merged.columns:
            logger.warning("  %s: missing", col); continue
        r = bootstrap_ic(merged[col].values, merged["y"].values)
        if r is None:
            logger.warning("  %s: insufficient valid pairs", col); continue
        r["feature"] = col
        results.append(r)

    # Sort by |IC|
    results_sorted = sorted(results, key=lambda r: -abs(r["ic"]))
    logger.info("")
    logger.info("%-28s  %4s  %8s  %-18s", "feature", "n", "IC", "95% CI")
    logger.info("-" * 72)
    for r in results_sorted:
        sig = "✓" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else " "
        logger.info("%-28s  %4d  %+.4f  [%+.3f, %+.3f]  %s",
                    r["feature"], r["n"], r["ic"], r["ci_lo"], r["ci_hi"], sig)

    # Save
    out = {
        "description": "Phase 1a: microstructure feature IC sanity check on 22-day flow_bars_1m",
        "n_1min_bars": int(len(df1)),
        "n_4h_bars_aligned": int(len(merged)),
        "date_range": [str(merged.index[0]), str(merged.index[-1])],
        "features": results_sorted,
        "caveat": ("Only ~126 aligned 4h bars — IC estimates are noisy. "
                   "Features marked ✓ (CI does not cross 0) are candidates "
                   "for re-evaluation with ≥ 90 days of data. This is a "
                   "sanity check, not a production validator."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info("")
    logger.info("Saved → %s", OUT)


if __name__ == "__main__":
    main()
