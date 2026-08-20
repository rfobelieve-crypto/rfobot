"""Publish live raid signals to MySQL for the follow-bot endpoint.

2026-08-20.  The endpoint originally read the shadow CSV shipped inside the
agent's Docker image — but the hourly recorder runs on the operator machine,
so the image only ever holds the snapshot from the last git push.  It was 8
days stale, meaning /public/raid-signals returned 0 rows forever and any
follower wired to it would sit idle while looking perfectly healthy.  Same
class as every other bug found this week: both sides fine, the seam dead.

Fix is the path weather_station already proved: the quant side PERSISTS,
the agent only SELECTs.  This publisher rides the hourly SweepShadow batch
right after the recorder refreshes the log.

Only OPEN, variant-B, side-bearing rows within the HOLD window are
published; the table is replaced wholesale each run so a signal that
closed or aged out simply disappears.  Variant membership (A/B/C/D) is
computed here from the frozen predicates so no consumer re-implements them.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG = ROOT / "research" / "results" / "sweep_shadow_log.csv"
MAX_AGE_H = 8
STOP_ATR = 3.5      # mirrors sweep_core.DIS
HOLD_H = 8          # mirrors sweep_core.HOLD

DDL = """
CREATE TABLE IF NOT EXISTS raid_signals_live (
  id          BIGINT AUTO_INCREMENT PRIMARY KEY,
  symbol      VARCHAR(16)   NOT NULL,
  side        ENUM('LONG','SHORT') NOT NULL,
  level_kind  VARCHAR(16)   NOT NULL,
  fill_ts     BIGINT        NOT NULL,
  fill_utc    VARCHAR(20)   NOT NULL,
  entry_px    DECIMAL(20,8) NOT NULL,
  atr         DECIMAL(20,8) NOT NULL,
  stop_px     DECIMAL(20,8) NOT NULL,
  risk_frac   DECIMAL(12,8) NOT NULL,
  pierce_atr  DECIMAL(12,6) NOT NULL DEFAULT 0,
  universe    VARCHAR(16)   NOT NULL,
  variants    VARCHAR(32)   NOT NULL,
  updated_at  DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY ux_sig (symbol, level_kind, fill_ts),
  KEY ix_fill (fill_ts)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def collect() -> list[tuple]:
    if not LOG.exists():
        return []
    now = int(time.time())
    rows = []
    with LOG.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "OPEN" or r.get("variant_b") != "1":
                continue
            if r.get("side") not in ("LONG", "SHORT"):
                continue
            try:
                fill_ts = int(r["fill_ts"])
            except (KeyError, ValueError):
                continue
            if now - fill_ts > MAX_AGE_H * 3600:
                continue
            entry = float(r["entry_px"])
            atr = float(r["atr"])
            if not (entry > 0 and atr > 0):
                continue
            sgn = 1 if r["side"] == "LONG" else -1
            variants = ["A", "B"]
            if str(r.get("flow_reject", "")) == "1":
                variants.append("C")
                if str(r.get("flow_vhigh", "")) == "1":
                    variants.append("D")
            rows.append((
                r["symbol"], r["side"], r.get("level_kind", "swing"),
                fill_ts, r.get("fill_utc", ""), entry, atr,
                entry - sgn * STOP_ATR * atr, STOP_ATR * atr / entry,
                float(r.get("pierce_atr") or 0), r.get("universe", ""),
                ",".join(variants)))
    return rows


def main() -> None:
    from shared.db import get_db_conn
    rows = collect()
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(DDL)
            # Replace wholesale: closed/aged-out signals must vanish, and a
            # follower must never act on a row this run no longer vouches for.
            cur.execute("DELETE FROM raid_signals_live")
            if rows:
                cur.executemany(
                    "INSERT INTO raid_signals_live (symbol, side, level_kind,"
                    " fill_ts, fill_utc, entry_px, atr, stop_px, risk_frac,"
                    " pierce_atr, universe, variants)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", rows)
        conn.commit()
        core = sum(1 for r in rows if r[10] == "core9")
        print(f"raid_signals_live: published {len(rows)} signals "
              f"({core} core9)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
