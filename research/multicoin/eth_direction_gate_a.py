"""
V7 multicoin Step 2 — ETH Direction-model clean-AUC port (2026-07-23).

Per TODO.md §4.6 / CLAUDE.md "V7 多幣化提前啟動" (5th informed override):
backfill ETH history -> build ETH feature table -> run the SAME clean walk-
forward harness (purge+embargo, no early-stop-on-test leak) used to produce
BTC's benchmark clean AUC of 0.5412, so the two numbers are directly
comparable.

Pre-registered Go/No-Go (TODO.md, unchanged by the override):
    ETH clean AUC >= ~0.54  AND  ETH/BTC Strong-signal overlap < 50%
    -> continue (consider SOL, discuss production-ization)
    otherwise -> multicoin has no payoff for V7, resources go back to the
    cancel-flow (異源資料) line.

Design choices (research track, does NOT touch production):
  - Reuses `indicator.data_fetcher._cg_fetch` and `fetch_binance_klines`
    UNMODIFIED (both are already symbol-parametrized; only the CG_ENDPOINTS
    config dict in data_fetcher.py is BTC-hardcoded, so this script defines
    its own ETH_CG_ENDPOINTS instead of editing the production one).
  - Reuses `research.dual_model.shared_data._fetch_klines_paginated`
    UNMODIFIED (already takes `symbol=`).
  - Reuses `indicator.feature_builder_live.build_live_features` UNMODIFIED —
    it takes `klines` + `cg_data` generically. Its internal
    `_inject_alt_historical` step still pulls BTC's ETF-flow/DVOL/F&G
    parquet files (hardcoded paths in that production module) and merges
    them onto the ETH bars as shared macro/vol context. This is a known,
    flagged approximation — NOT a bug to fix here (touching that production
    path is out of scope for a research track) — see CAVEATS in the
    printed report.
  - `coinbase_premium` is excluded from ETH_CG_ENDPOINTS entirely: Step 1's
    audit (research/multicoin/audit_results.md) proved this endpoint
    ignores the `symbol` param and always returns BTC's premium, so
    fetching it "for ETH" would just re-inject BTC data under an ETH label.
  - Reuses `research.feature_search_ab._per_fold_oos` / `_pooled` /
    `BASE_PARAMS` UNMODIFIED — this is the exact function that produced the
    0.5412 BTC number, so there is zero hyperparameter drift between the
    two runs.
  - Own cache dir (`research/multicoin/.cache/`), own results file — zero
    collision risk with BTC's `research/dual_model/.cache/`.

Run:  python research/multicoin/eth_direction_gate_a.py
"""
from __future__ import annotations

import sys
import time
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = PROJECT_ROOT / "research" / "multicoin" / ".cache"
RESULTS_DIR = PROJECT_ROOT / "research" / "results" / "multicoin"

FETCH_LIMIT = 4000  # bars — matches load_and_cache_data() default used for BTC 0.5412

# Endpoints proven distinct-per-symbol in Step 1's audit (audit_results.md).
# coinbase_premium deliberately excluded — proven symbol-ignoring (BTC-only).
ETH_CG_ENDPOINTS = {
    "oi":               {"path": "/futures/open-interest/history", "exchange": "Binance", "symbol": "ETHUSDT"},
    "oi_agg":           {"path": "/futures/open-interest/aggregated-history", "symbol": "ETH"},
    "liquidation":      {"path": "/futures/liquidation/history", "exchange": "Binance", "symbol": "ETHUSDT"},
    "long_short":       {"path": "/futures/top-long-short-account-ratio/history", "exchange": "Binance", "symbol": "ETHUSDT"},
    "global_ls":        {"path": "/futures/global-long-short-account-ratio/history", "exchange": "Binance", "symbol": "ETHUSDT"},
    "funding":          {"path": "/futures/funding-rate/history", "exchange": "Binance", "symbol": "ETHUSDT"},
    "taker":            {"path": "/futures/taker-buy-sell-volume/history", "exchange": "Binance", "symbol": "ETHUSDT"},
    "bitfinex_margin":  {"path": "/bitfinex-margin-long-short", "symbol": "ETH"},
    "top_ls_position":  {"path": "/futures/top-long-short-position-ratio/history", "exchange": "Binance", "symbol": "ETHUSDT"},
    "futures_cvd_agg":  {"path": "/futures/aggregated-cvd/history", "symbol": "ETH", "extra_params": {"exchange_list": "Binance"}},
    "spot_cvd_agg":     {"path": "/spot/aggregated-cvd/history", "symbol": "ETH", "extra_params": {"exchange_list": "Binance"}},
    "liq_agg":          {"path": "/futures/liquidation/aggregated-history", "symbol": "ETH", "extra_params": {"exchange_list": "Binance"}},
    "oi_coin_margin":   {"path": "/futures/open-interest/aggregated-coin-margin-history", "symbol": "ETH", "extra_params": {"exchange_list": "Binance"}},
}


def fetch_coinglass_eth(interval: str = "1h", limit: int = FETCH_LIMIT) -> dict[str, pd.DataFrame]:
    """ETH analogue of indicator.data_fetcher.fetch_coinglass(), reusing the
    unmodified, already symbol-parametrized _cg_fetch() helper."""
    from indicator.data_fetcher import _cg_fetch, CG_API_KEY

    if not CG_API_KEY:
        logger.error("COINGLASS_API_KEY is not set — all CG data will be empty!")

    result = {}
    for name, cfg in ETH_CG_ENDPOINTS.items():
        try:
            df = _cg_fetch(
                path=cfg["path"], exchange=cfg.get("exchange"),
                symbol=cfg.get("symbol"), interval=interval, limit=limit,
                extra_params=cfg.get("extra_params"),
            )
            result[name] = df
            logger.info("CG(ETH) %s: %d rows", name, len(df))
        except Exception as e:
            logger.error("CG(ETH) %s failed: %s", name, e)
            result[name] = pd.DataFrame()
        time.sleep(1)  # rate-limit courtesy, matches production
    return result


def load_and_cache_eth(limit: int = FETCH_LIMIT, force_refresh: bool = False) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "eth_features_all.parquet"
    if cache_path.exists() and not force_refresh:
        df = pd.read_parquet(cache_path)
        logger.info("Loaded cached ETH features: %d bars x %d cols", len(df), len(df.columns))
        return df

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    from research.dual_model.shared_data import _fetch_klines_paginated
    from indicator.feature_builder_live import build_live_features

    klines = _fetch_klines_paginated(limit, symbol="ETHUSDT")
    cg_data = fetch_coinglass_eth(interval="1h", limit=limit)

    features = build_live_features(klines, cg_data)

    if "close" not in features.columns and "close" in klines.columns:
        features["close"] = klines["close"].reindex(features.index)

    features.to_parquet(cache_path)
    logger.info("Cached ETH features: %d bars x %d cols -> %s",
                len(features), len(features.columns), cache_path)
    return features


def main():
    from research.dual_model.build_direction_reg_labels import build_direction_reg_labels
    from research.dual_model.direction_features_v2 import FULL_DIRECTION, filter_available
    from research.feature_search_ab import _per_fold_oos, _pooled

    print("=" * 72)
    print("  V7 MULTICOIN STEP 2 — ETH clean-AUC vs BTC benchmark 0.5412")
    print("=" * 72)

    df = load_and_cache_eth()
    labels = build_direction_reg_labels(df)
    df = df.copy()
    df["y_path_ret_4h"] = labels["y_path_ret_4h"]

    deployed = filter_available(FULL_DIRECTION, list(df.columns))
    missing = sorted(set(FULL_DIRECTION) - set(deployed))
    print(f"  ETH bars={len(df)}  features available={len(deployed)}/{len(FULL_DIRECTION)}")
    if missing:
        print(f"  missing ({len(missing)}): {missing}")

    t0 = time.time()
    folds = _per_fold_oos(df, deployed, leaky=False)
    auc, ic = _pooled(folds)
    n_folds = len(folds)
    print(f"\n  CLEAN walk-forward (purge+embargo, no early-stop leak):")
    print(f"    folds={n_folds}  pooled sign-AUC={auc:.4f}  pooled IC={ic:+.4f}  "
          f"({time.time()-t0:.0f}s)")

    gate_pass = auc >= 0.54
    print(f"\n  Gate (pre-registered, TODO.md §4.6): ETH clean AUC >= ~0.54 "
          f"-> {'PASS' if gate_pass else 'FAIL'} ({auc:.4f})")
    print("  (Overlap gate <50% is Step 3, run separately if this passes.)")

    print("\n  CAVEATS:")
    print("  - _inject_alt_historical (ETF flow / DVOL / Fear&Greed) still reads")
    print("    BTC's own parquet files in market_data/raw_data/ (production code,")
    print("    not modified here) -> those few columns are BTC-sourced macro")
    print("    context merged onto ETH bars, not ETH-native.")
    print("  - coinbase_premium excluded (Step 1 proved it's symbol-ignoring).")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "eth_direction_gate_a.json"
    out.write_text(json.dumps({
        "symbol": "ETHUSDT", "n_bars": len(df), "n_folds": n_folds,
        "n_features_available": len(deployed), "n_features_total": len(FULL_DIRECTION),
        "missing_features": missing,
        "pooled_clean_auc": round(float(auc), 4), "pooled_clean_ic": round(float(ic), 4),
        "btc_benchmark_clean_auc": 0.5412,
        "gate_pass": bool(gate_pass),
        "run_ts": pd.Timestamp.now(tz="UTC").isoformat(),
    }, indent=2))
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
