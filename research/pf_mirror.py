"""Portfolio ledger phase M1 — create pf_* tables and mirror V7's closed
book into the unified ledger.  Zero live-code change (2026-08-18).

The framework doc's P1 wanted V7 dual-write from the executor; that touches
live code and the AST no-wire test guards exactly that.  This mirror gets
the same outcome — pf_positions carries V7's history in the unified
schema — by READING v7_okx_positions and upserting via the (src_table,
src_id) unique key the DDL was designed with.  Idempotent, hourly, and the
executor never knows it exists.

Approximations (documented, good enough for portfolio views; the live
path will write exact numbers): risk_usd = stop_dist x size x 0.01 (BTC
contract value); net_pnl = net_pct x notional_usd.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicator.portfolio.ledger import ALL_DDL  # noqa: E402


def main() -> None:
    from shared.db import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            for ddl in ALL_DDL:
                cur.execute(ddl)
            cur.execute("""
                INSERT INTO pf_positions
                    (strategy, symbol, side, entry_ts, entry_px, size,
                     risk_usd, stop_px, exit_ts, exit_px, exit_reason,
                     gross_pnl, fees, net_pnl, net_r, equity_after,
                     src_table, src_id)
                SELECT 'v7', 'BTC-USD', direction, entry_time, entry_price,
                       size_contracts,
                       GREATEST(stop_dist * size_contracts * 0.01, 0.01),
                       current_stop, exit_time, exit_price, exit_reason,
                       gross_pct * notional_usd,
                       COALESCE(entry_fees_usd,0) + COALESCE(exit_fees_usd,0),
                       net_pct * notional_usd,
                       CASE WHEN stop_dist > 0 AND size_contracts > 0
                            THEN (net_pct * notional_usd)
                                 / (stop_dist * size_contracts * 0.01)
                            ELSE NULL END,
                       equity_after, 'v7_okx_positions', id
                FROM v7_okx_positions
                ON DUPLICATE KEY UPDATE
                    exit_ts = VALUES(exit_ts),
                    exit_px = VALUES(exit_px),
                    exit_reason = VALUES(exit_reason),
                    gross_pnl = VALUES(gross_pnl),
                    fees = VALUES(fees),
                    net_pnl = VALUES(net_pnl),
                    net_r = VALUES(net_r),
                    equity_after = VALUES(equity_after)
                """)
            n = cur.rowcount
        conn.commit()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) n, SUM(net_pnl) p FROM pf_positions "
                        "WHERE strategy='v7'")
            r = cur.fetchone()
        print(f"pf_mirror: upserted rows affected={n}; "
              f"ledger now v7 n={r['n']} net_pnl_sum={float(r['p'] or 0):+.2f}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
