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

Only OPEN, side-bearing rows within the HOLD window are published; the
table is replaced wholesale each run so a signal that closed or aged out
simply disappears.  Membership is computed here from the frozen predicates
so no consumer re-implements them, and shipped as one comma list:

  A/B/C/D   the cohort ladder (D subset-of C subset-of B subset-of A)
  R / RV    the frozen watchlist combos (registered 2026-08-02) — these do
            NOT require the shallow-pierce condition, so they are not a
            subset of B.  Until 2026-08-20 this publisher dropped every
            variant_b != 1 row, which made R and RV unreachable downstream:
            a consumer asking for R would silently receive only B and-R = C.

Publishing non-B rows changes nothing for existing consumers: a bot asking
for "B" still only matches rows whose list contains B.
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

# 與 shadow_engine 同一條 SWEEP_DATA_DIR 規則（上雲時帳本在 volume 上）
_DD = __import__("os").environ.get("SWEEP_DATA_DIR", "").strip()
LOG = (Path(_DD) / "sweep_shadow_log.csv") if _DD     else ROOT / "research" / "results" / "sweep_shadow_log.csv"
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
  variants    VARCHAR(40)   NOT NULL,
  -- frozen ADX(14) 25/20 label at the FILL hour (§0.59). The route emitted
  -- this field while the table did not have it, so every row carried "" and
  -- any consumer filtering on it blocked everything (2026-09-03).
  regime_cell VARCHAR(12)   NOT NULL DEFAULT '',
  -- variant E membership, BTC-only (§0.474b). Three states, because
  -- "absent" is ambiguous: E / notE / pending (derivative panels not yet
  -- annotated -- Coinglass parquets lag ~6h) / na (not BTC).
  e_state     VARCHAR(8)    NOT NULL DEFAULT 'na',
  updated_at  DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY ux_sig (symbol, level_kind, fill_ts),
  KEY ix_fill (fill_ts)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def _e_membership():
    """(E predicate, ledger rows) from the owning engine, or (None, {}).

    E's causal liq-burst median needs the whole log, so membership cannot be
    decided from a single row -- shadow_engine owns it and this file only
    labels. Failure leaves every row "pending", never a silent "notE".
    """
    try:
        sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))
        import shadow_engine as SE
        log = SE.read_log()
        return SE.variant_e_pred(log), log
    except Exception:
        return None, {}


def _e_state(r, pred, log) -> str:
    if r.get("symbol") != "BTC":
        return "na"
    if pred is None:
        return "pending"
    key = (r["symbol"], r.get("level_kind", "swing"), int(r["fill_ts"]))
    row = log.get(key, r)
    # the derivative panels are annotated hourly but the Coinglass parquets
    # lag ~6h, so a fresh raid is legitimately "not decided yet"
    if row.get("drv_q") in (None, "") or row.get("drv_liqburst") in (None, ""):
        return "pending"
    if row.get("drv_q") == "na":
        return "na"
    return "E" if pred(row) else "notE"


def collect() -> list[tuple]:
    if not LOG.exists():
        return []
    now = int(time.time())
    e_pred, e_log = _e_membership()
    rows = []
    with LOG.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "OPEN":
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
            is_b = str(r.get("variant_b", "")) == "1"
            is_r = str(r.get("flow_reject", "")) == "1"
            is_v = str(r.get("flow_vhigh", "")) == "1"
            variants = ["A"]
            if is_b:
                variants.append("B")
                if is_r:
                    variants.append("C")
                    if is_v:
                        variants.append("D")
            # 組合不套 B：R 是「有縮回就算」，RV 是「放量刺＋縮回來」。
            # 它們跟 C/D 的差別正是少了淺穿越那道門，所以會涵蓋 B 以外的列。
            if is_r:
                variants.append("R")
                if is_v:
                    variants.append("RV")
            est = _e_state(r, e_pred, e_log)
            if est == "E":
                variants.append("E")
            rows.append((
                r["symbol"], r["side"], r.get("level_kind", "swing"),
                fill_ts, r.get("fill_utc", ""), entry, atr,
                entry - sgn * STOP_ATR * atr, STOP_ATR * atr / entry,
                float(r.get("pierce_atr") or 0), r.get("universe", ""),
                ",".join(variants), r.get("regime_cell", "") or "", est))
    return rows


def main() -> None:
    from shared.db import get_db_conn
    rows = collect()
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(DDL)
            # CREATE TABLE IF NOT EXISTS does not add columns to a table that
            # already exists, so evolve explicitly. Idempotent.
            cur.execute("SELECT COLUMN_NAME FROM information_schema.COLUMNS"
                        " WHERE TABLE_SCHEMA=DATABASE()"
                        " AND TABLE_NAME='raid_signals_live'")
            have = {list(c.values())[0] for c in cur.fetchall()}
            for col, ddl in (("regime_cell", "VARCHAR(12) NOT NULL DEFAULT ''"),
                             ("e_state", "VARCHAR(8) NOT NULL DEFAULT 'na'")):
                if col not in have:
                    cur.execute(f"ALTER TABLE raid_signals_live ADD COLUMN {col} {ddl}")
            if "variants" in have:
                cur.execute("ALTER TABLE raid_signals_live"
                            " MODIFY variants VARCHAR(40) NOT NULL")
            # Replace wholesale: closed/aged-out signals must vanish, and a
            # follower must never act on a row this run no longer vouches for.
            cur.execute("DELETE FROM raid_signals_live")
            if rows:
                cur.executemany(
                    "INSERT INTO raid_signals_live (symbol, side, level_kind,"
                    " fill_ts, fill_utc, entry_px, atr, stop_px, risk_frac,"
                    " pierce_atr, universe, variants, regime_cell, e_state)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", rows)
        conn.commit()
        core = sum(1 for r in rows if r[10] == "core9")
        nE = sum(1 for r in rows if r[13] == "E")
        npend = sum(1 for r in rows if r[13] == "pending")
        print(f"raid_signals_live: published {len(rows)} signals "
              f"({core} core9, E={nE}, E待定={npend})")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
