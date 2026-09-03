# -*- coding: utf-8 -*-
"""Hourly refresh of the three Coinglass parquets variant E depends on.

Why this exists (2026-09-03): E's three panels (OI / CVD / liquidations) are
read from market_data/raw_data/cg_*_1h.parquet, which the DAILY collector
refreshes. That put the panels ~6h behind while /public/raid-signals only
carries a signal for 8h — so a fresh BTC raid was almost always published as
`e_state: "pending"` and E was, in practice, not a followable label.

Three endpoints only, on purpose: this runs every hour, and refreshing all 14
would be 14x the API budget for data nothing hourly reads. The daily
backfill still covers everything else — this does not replace it.

  cg_oi_agg_1h            OI panel   (OI down at the raid hour)
  cg_futures_cvd_agg_1h   CVD panel  (taker flow with the break)
  cg_liq_agg_1h           liq panel  (burst >= causal median)

Run: python research/refresh_cg_hourly.py
Called from research/sweep_failure/shadow_engine.bat BEFORE the engine, so
the annotation in the same run sees the fresh hour.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# backfill_all_parquet resolves RAW relative to the CWD, so the caller's
# working directory decides where the parquets land. Pin it.
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from research.backfill_all_parquet import backfill_cg  # noqa: E402

E_ENDPOINTS = {
    "oi_agg": {"path": "/futures/open-interest/aggregated-history",
               "params": {"symbol": "BTC", "interval": "1h", "limit": 200}},
    "futures_cvd_agg": {"path": "/futures/aggregated-cvd/history",
                        "params": {"symbol": "BTC", "interval": "1h",
                                   "limit": 200, "exchange_list": "Binance"}},
    "liq_agg": {"path": "/futures/liquidation/aggregated-history",
                "params": {"symbol": "BTC", "interval": "1h",
                           "limit": 200, "exchange_list": "Binance"}},
}


def main() -> int:
    backfill_cg(E_ENDPOINTS, "hourly (variant E panels)")
    # Report the actual freshness, not "the command ran" — the whole point of
    # this script is an age number, so it prints one (mistake.md 2026-08-20:
    # the test of "did it run" is the artifact, never the exit code).
    import pandas as pd
    now = pd.Timestamp.now(tz="UTC")
    worst = 0.0
    for name in E_ENDPOINTS:
        p = ROOT / "market_data" / "raw_data" / f"cg_{name}_1h.parquet"
        try:
            idx = pd.read_parquet(p).index
            age = (now - idx.max()).total_seconds() / 3600
            worst = max(worst, age)
            print(f"  {name}: last bar {idx.max()}  ({age:.1f}h old)")
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: unreadable ({type(e).__name__})")
            worst = 99.0
    print(f"cg hourly refresh: worst age {worst:.1f}h "
          f"({'OK' if worst <= 3 else 'STILL STALE — E labels will stay pending'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
