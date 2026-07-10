"""Daily parquet export of depth_deltas_1m — backup for unbackfillable data.

The cancellation stream (depth_deltas_1m) is forward-collected only: a DB
loss destroys history that can never be re-fetched. Every other data line in
this repo has a parquet fallback; this gives the most irreplaceable one the
same insurance. Full-table dump (tiny: ~1440 rows/day), idempotent, safe to
run any time. Wired into market_data/backfill/daily_collect.bat.

Usage:  python -m market_data.backfill.export_depth_deltas
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared.db import get_db_conn

OUT = ROOT / "market_data" / "raw_data" / "depth_deltas_1m.parquet"


def main() -> int:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT exchange, canonical_symbol, minute_start_ms, "
                "       bid_add_qty, bid_cancel_qty, ask_add_qty, "
                "       ask_cancel_qty, update_count "
                "FROM depth_deltas_1m ORDER BY minute_start_ms")
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        print("depth_deltas_1m empty — nothing to export")
        return 0
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    lo = pd.to_datetime(df["minute_start_ms"].min(), unit="ms")
    hi = pd.to_datetime(df["minute_start_ms"].max(), unit="ms")
    print(f"exported {len(df)} rows ({lo} -> {hi} UTC) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
