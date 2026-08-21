# -*- coding: utf-8 -*-
"""Daily Fear & Greed parquet refresh — weakness #2 of the data-eng audit.

cg_fear_greed_daily.parquet was written ONCE by backfill_historical_alt.py
(a manual, never-scheduled script) in April and rotted for 130 days — the
same never-scheduled-writer disease as mistake.md 2026-08-01. The freshness
board caught it on its first run (2026-08-20).

Why revive instead of retire (the ETF sibling was retired): F&G measured a
CI-clean incremental +0.080 on forward-8h vol AFTER controlling fast+slow
realized vol (9/9 coins, 2026-08-20 gauge ranking) — it is the registered
phase-2 candidate gauge for the reefing line, so its history must stay
alive. Rides daily_collect.bat.

The upstream call returns FULL history every time and the write is a
single-writer idempotent overwrite — no read-then-merge needed here (the
2026-04-19 rule targets multi-writer files; this one has exactly one).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from backfill_historical_alt import backfill_fear_greed  # noqa: E402

if __name__ == "__main__":
    backfill_fear_greed()
    p = ROOT / "market_data" / "raw_data" / "cg_fear_greed_daily.parquet"
    if not p.exists():
        print("[ERROR] refresh_fng: parquet missing after refresh")
        raise SystemExit(1)
    import time
    age_h = (time.time() - p.stat().st_mtime) / 3600
    if age_h > 1:
        print(f"[ERROR] refresh_fng: parquet not rewritten (age {age_h:.1f}h)")
        raise SystemExit(1)
    print("refresh_fng: OK")
