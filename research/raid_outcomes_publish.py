# -*- coding: utf-8 -*-
"""Publish CLOSED raid signal outcomes — the missing half of the feed.

Why (TODO §0.62, 2026-08-26): the product side asked whether the slot cap
biases their sample. It does — 25.1% of signals are dropped at K=5 and the
dropped set is not random. The correct fix is not a cleverer allocation
rule (see §0.62: ranking by realised R is look-ahead, and ranking by
predicted R is a new model chosen after seeing the result). The fix is to
RECORD WHAT WAS DROPPED and score it on the same ruler.

But they could not do that, because of a gap on THIS side:
`raid_signals_live` publishes only rows with status OPEN, inside a 4h
window. **A signal's outcome was never published anywhere.** So a blocked
signal could be logged but never scored — and comparing their realised
fills against research-side shadow numbers is exactly the scoring
asymmetry that manufactured the "4.3x" in the first place.

This publisher closes that gap: closed variant-B signals with the same
identity fields the live feed uses, plus the shadow outcome. Same ruler on
both sides of the comparison, which is the whole point.

TWO JOIN PATHS, because the consumer has two entry routes:
  * saw it in /public/raid-signals  -> join on (symbol, level_kind, fill_ts)
  * saw it in /public/raid-pending  -> that row carries trigger_px, and the
    shadow engine runs with SLIP=0, so entry_px here IS the level price;
    join on (symbol, level_kind, entry_px) within a small relative
    tolerance. fill_ts is NOT reliable for this path — the consumer sees a
    touch in real time, the ledger stamps the fill bar's close.

Boundary: quant side writes, agent only SELECTs (agent-boundary.md).
Public-surface: R multiples, prices of levels, timestamps. No sizes, no
dollars, no balances.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOG = ROOT / "research" / "results" / "sweep_shadow_log.csv"
STOP_ATR = 3.5
MAX_AGE_D = 45          # enough for an October review to backfill Sept

DDL = """
CREATE TABLE IF NOT EXISTS raid_outcomes (
  symbol      VARCHAR(16)   NOT NULL,
  side        VARCHAR(8)    NOT NULL,
  level_kind  VARCHAR(24)   NOT NULL,
  fill_ts     BIGINT        NOT NULL,
  fill_utc    VARCHAR(32)   NOT NULL,
  entry_px    DECIMAL(20,8) NOT NULL,
  atr         DECIMAL(20,8) NOT NULL,
  exit_ts     BIGINT        NOT NULL,
  gross_r     DECIMAL(12,6) NOT NULL,
  net_r       DECIMAL(12,6) NOT NULL,
  stopped     TINYINT       NOT NULL DEFAULT 0,
  regime_cell VARCHAR(12)   NOT NULL DEFAULT '',
  universe    VARCHAR(16)   NOT NULL,
  variants    VARCHAR(32)   NOT NULL,
  updated_at  DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP
              ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY ux_out (symbol, level_kind, fill_ts),
  KEY ix_fill (fill_ts)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def collect() -> list[tuple]:
    if not LOG.exists():
        return []
    cutoff = int(time.time()) - MAX_AGE_D * 86400
    out = []
    with LOG.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "CLOSED":
                continue
            if str(r.get("variant_b", "")) != "1":
                continue          # the Gate F track; A-only rows add noise
            if r.get("side") not in ("LONG", "SHORT"):
                continue
            try:
                fill_ts, exit_ts = int(r["fill_ts"]), int(r["exit_ts"])
                entry, atr = float(r["entry_px"]), float(r["atr"])
                gr, nr = float(r["gross_r"]), float(r["net_r"])
            except (KeyError, ValueError, TypeError):
                continue
            if fill_ts < cutoff or not (entry > 0 and atr > 0):
                continue
            # variant attribution mirrors raid_signals_publish exactly —
            # one definition of the ladder, not two.
            is_r = str(r.get("flow_reject", "")) == "1"
            is_v = str(r.get("flow_vhigh", "")) == "1"
            variants = ["A", "B"]
            if is_r:
                variants.append("C")
                if is_v:
                    variants.append("D")
                variants.append("R")
                if is_v:
                    variants.append("RV")
            out.append((
                r["symbol"], r["side"], r.get("level_kind", "swing"),
                fill_ts, r.get("fill_utc", ""), entry, atr, exit_ts,
                gr, nr, 1 if str(r.get("stopped", "")) == "1" else 0,
                r.get("regime_cell", ""), r.get("universe", ""),
                ",".join(variants)))
    return out


def main() -> int:
    rows = collect()
    if not rows:
        print("raid_outcomes: nothing to publish")
        return 0
    from shared.db import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(DDL)
            cur.executemany(
                "INSERT INTO raid_outcomes (symbol, side, level_kind, "
                "fill_ts, fill_utc, entry_px, atr, exit_ts, gross_r, net_r, "
                "stopped, regime_cell, universe, variants) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON DUPLICATE KEY UPDATE exit_ts=VALUES(exit_ts), "
                "gross_r=VALUES(gross_r), net_r=VALUES(net_r), "
                "stopped=VALUES(stopped), regime_cell=VALUES(regime_cell), "
                "variants=VALUES(variants)", rows)
            # keep the table bounded; the consumer only backfills recently
            cur.execute("DELETE FROM raid_outcomes WHERE fill_ts < %s",
                        (int(time.time()) - (MAX_AGE_D + 15) * 86400,))
        conn.commit()
    finally:
        conn.close()
    print(f"raid_outcomes: published {len(rows)} closed "
          f"(last {MAX_AGE_D}d, variant B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
