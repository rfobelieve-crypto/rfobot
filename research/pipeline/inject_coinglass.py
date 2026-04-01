"""
Inject Coinglass data into enhanced features — adds multi-source features.

New features added:
  - cg_oi_close, cg_oi_delta, cg_oi_accel          (Binance OI from Coinglass)
  - cg_oi_agg_close, cg_oi_agg_delta               (all-exchange aggregated OI)
  - cg_oi_binance_share                             (Binance OI / total OI ratio)
  - cg_liq_long, cg_liq_short, cg_liq_ratio        (liquidation volumes)
  - cg_liq_total, cg_liq_imbalance                  (liquidation aggregates)
  - cg_ls_long_pct, cg_ls_short_pct, cg_ls_ratio   (long/short account ratio)
  - cg_funding_close, cg_funding_range              (funding rate OHLC)
  - cg_taker_buy, cg_taker_sell, cg_taker_delta     (taker buy/sell volume)
  - cg_taker_ratio                                  (taker buy/total ratio)
  - Derived z-scores and rolling features

Source: market_data/raw_data/coinglass/*_30m.parquet (forward-filled to 15m)
Target: research/ml_data/BTC_USD_15m_enhanced.parquet (updated in-place)

Usage:
    python -m research.pipeline.inject_coinglass
    python -m research.pipeline.inject_coinglass --dry-run
"""
from __future__ import annotations

import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CG_DIR = ROOT / "market_data" / "raw_data" / "coinglass"
ENHANCED_PATH = ROOT / "research" / "ml_data" / "BTC_USD_15m_enhanced.parquet"

ZSCORE_WIN = 96  # 24h in 15m bars


def _zscore(s: pd.Series, win: int = ZSCORE_WIN) -> pd.Series:
    mu = s.rolling(win, min_periods=4).mean()
    sd = s.rolling(win, min_periods=4).std().replace(0, np.nan)
    return (s - mu) / sd


def _load_cg(name: str, interval: str = "30m") -> pd.DataFrame:
    """Load a Coinglass parquet, set datetime index."""
    path = CG_DIR / f"{name}_BTC_Binance_{interval}.parquet"
    if not path.exists():
        logger.warning("Not found: %s", path)
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["dt"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.set_index("dt").sort_index()
    return df


def build() -> pd.DataFrame:
    # Load existing enhanced features
    enhanced = pd.read_parquet(ENHANCED_PATH)
    if not isinstance(enhanced.index, pd.DatetimeIndex):
        if "dt" in enhanced.columns:
            enhanced = enhanced.set_index("dt")
    enhanced = enhanced.sort_index()
    logger.info("Enhanced: %d rows, %d cols", len(enhanced), len(enhanced.columns))

    # Use 30m data (higher resolution, forward-fill to 15m)
    oi      = _load_cg("oi", "30m")
    oi_agg  = _load_cg("oi_agg", "30m")
    liq     = _load_cg("liquidation", "30m")
    ls      = _load_cg("long_short", "30m")
    funding = _load_cg("funding", "30m")
    taker   = _load_cg("taker", "30m")

    # ═══════════════════════════════════════════════════════════════════════
    # 1. OI (Binance per-exchange)
    # ═══════════════════════════════════════════════════════════════════════
    if not oi.empty:
        oi_feat = pd.DataFrame(index=oi.index)
        oi_feat["cg_oi_close"] = oi["close"]
        oi_feat["cg_oi_delta"] = oi["close"].diff()
        oi_feat["cg_oi_accel"] = oi_feat["cg_oi_delta"].diff()
        enhanced = enhanced.join(oi_feat, how="left")
        logger.info("OI: injected 3 features")

    # ═══════════════════════════════════════════════════════════════════════
    # 2. OI Aggregated (all exchanges)
    # ═══════════════════════════════════════════════════════════════════════
    if not oi_agg.empty:
        agg_feat = pd.DataFrame(index=oi_agg.index)
        agg_feat["cg_oi_agg_close"] = oi_agg["close"]
        agg_feat["cg_oi_agg_delta"] = oi_agg["close"].diff()
        enhanced = enhanced.join(agg_feat, how="left")

        # Binance share of total OI
        if not oi.empty:
            # Align on common index
            common = enhanced.index
            bnc = enhanced["cg_oi_close"]
            agg = enhanced["cg_oi_agg_close"]
            enhanced["cg_oi_binance_share"] = bnc / agg.replace(0, np.nan)
        logger.info("OI Aggregated: injected features")

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Liquidation
    # ═══════════════════════════════════════════════════════════════════════
    if not liq.empty:
        liq_feat = pd.DataFrame(index=liq.index)
        liq_feat["cg_liq_long"]  = liq["long_liquidation_usd"]
        liq_feat["cg_liq_short"] = liq["short_liquidation_usd"]
        total = liq_feat["cg_liq_long"] + liq_feat["cg_liq_short"]
        liq_feat["cg_liq_total"] = total
        liq_feat["cg_liq_ratio"] = liq_feat["cg_liq_long"] / total.replace(0, np.nan)
        liq_feat["cg_liq_imbalance"] = (
            liq_feat["cg_liq_long"] - liq_feat["cg_liq_short"]
        ) / total.replace(0, np.nan)
        enhanced = enhanced.join(liq_feat, how="left")
        logger.info("Liquidation: injected 5 features")

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Long/Short Ratio
    # ═══════════════════════════════════════════════════════════════════════
    if not ls.empty:
        ls_feat = pd.DataFrame(index=ls.index)
        ls_feat["cg_ls_long_pct"]  = ls["global_account_long_percent"]
        ls_feat["cg_ls_short_pct"] = ls["global_account_short_percent"]
        ls_feat["cg_ls_ratio"]     = ls["global_account_long_short_ratio"]
        enhanced = enhanced.join(ls_feat, how="left")
        logger.info("Long/Short: injected 3 features")

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Funding Rate (OHLC)
    # ═══════════════════════════════════════════════════════════════════════
    if not funding.empty:
        fr_feat = pd.DataFrame(index=funding.index)
        fr_feat["cg_funding_close"] = funding["close"]
        fr_feat["cg_funding_range"] = funding["high"] - funding["low"]
        enhanced = enhanced.join(fr_feat, how="left")
        logger.info("Funding: injected 2 features")

    # ═══════════════════════════════════════════════════════════════════════
    # 6. Taker Buy/Sell Volume
    # ═══════════════════════════════════════════════════════════════════════
    if not taker.empty:
        tk_feat = pd.DataFrame(index=taker.index)
        tk_feat["cg_taker_buy"]  = taker["taker_buy_volume_usd"]
        tk_feat["cg_taker_sell"] = taker["taker_sell_volume_usd"]
        tk_feat["cg_taker_delta"] = tk_feat["cg_taker_buy"] - tk_feat["cg_taker_sell"]
        total_tk = tk_feat["cg_taker_buy"] + tk_feat["cg_taker_sell"]
        tk_feat["cg_taker_ratio"] = tk_feat["cg_taker_buy"] / total_tk.replace(0, np.nan)
        enhanced = enhanced.join(tk_feat, how="left")
        logger.info("Taker: injected 4 features")

    # ═══════════════════════════════════════════════════════════════════════
    # Forward-fill (30m data → 15m gaps)
    # ═══════════════════════════════════════════════════════════════════════
    cg_cols = [c for c in enhanced.columns if c.startswith("cg_")]
    enhanced[cg_cols] = enhanced[cg_cols].ffill(limit=2)  # fill at most 2 bars (30m gap)

    # ═══════════════════════════════════════════════════════════════════════
    # Derived features (z-scores, rolling)
    # ═══════════════════════════════════════════════════════════════════════
    if "cg_oi_delta" in enhanced.columns:
        enhanced["cg_oi_delta_zscore"] = _zscore(enhanced["cg_oi_delta"])
        enhanced["cg_oi_close_zscore"] = _zscore(enhanced["cg_oi_close"])

    if "cg_liq_total" in enhanced.columns:
        enhanced["cg_liq_total_zscore"] = _zscore(enhanced["cg_liq_total"])
        enhanced["cg_liq_imbalance_zscore"] = _zscore(enhanced["cg_liq_imbalance"])

    if "cg_ls_ratio" in enhanced.columns:
        enhanced["cg_ls_ratio_zscore"] = _zscore(enhanced["cg_ls_ratio"])

    if "cg_taker_delta" in enhanced.columns:
        enhanced["cg_taker_delta_zscore"] = _zscore(enhanced["cg_taker_delta"])

    if "cg_funding_close" in enhanced.columns:
        enhanced["cg_funding_close_zscore"] = _zscore(enhanced["cg_funding_close"])

    # Cross-variable interactions
    if "cg_liq_imbalance" in enhanced.columns and "cg_oi_delta" in enhanced.columns:
        # Liquidation imbalance × OI change = forced position closing signal
        enhanced["cg_liq_x_oi"] = (
            enhanced["cg_liq_imbalance_zscore"] * enhanced["cg_oi_delta_zscore"]
        )

    if "cg_ls_ratio_zscore" in enhanced.columns and "cg_funding_close_zscore" in enhanced.columns:
        # Crowding signal: extreme L/S ratio + extreme funding
        enhanced["cg_crowding"] = (
            enhanced["cg_ls_ratio_zscore"] * enhanced["cg_funding_close_zscore"]
        )

    if "cg_taker_delta_zscore" in enhanced.columns and "cg_oi_delta_zscore" in enhanced.columns:
        # Taker aggression × OI expansion = conviction signal
        enhanced["cg_conviction"] = (
            enhanced["cg_taker_delta_zscore"] * enhanced["cg_oi_delta_zscore"]
        )

    enhanced = enhanced.replace([np.inf, -np.inf], np.nan)

    # Coverage stats
    cg_cols = [c for c in enhanced.columns if c.startswith("cg_")]
    coverage = enhanced[cg_cols].notna().mean()
    logger.info("\nCoinglass feature coverage:")
    for col, rate in coverage.items():
        logger.info("  %-35s  %.1f%%", col, rate * 100)

    logger.info("\nFinal: %d rows x %d cols (%d Coinglass features)",
                len(enhanced), len(enhanced.columns), len(cg_cols))

    return enhanced


def run(dry_run: bool = False):
    df = build()

    if dry_run:
        logger.info("DRY RUN -- not saving.")
        return

    df.to_parquet(ENHANCED_PATH, index=True)
    logger.info("Saved to %s", ENHANCED_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(dry_run=args.dry_run)
