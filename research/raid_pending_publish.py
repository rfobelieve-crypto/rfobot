# -*- coding: utf-8 -*-
"""Publish PENDING raid levels — the fix for TODO §0.57.

The existing publisher ships signals that have ALREADY filled, so a batch
consumer learns about them 65-342 minutes after the fillable moment. The
reachability recompute (research/sweep_realizable.py) priced that gate at
0.1328 R/trade — 158% of variant B's frozen edge, flipping +0.084R into
-0.049R with the day-clustered CI entirely below zero.

The gate is architectural, not market: the follower polls price every 60s
and could act the instant price touches the level. It just never learns
the level is armed.

So publish the WAITING LIST instead: the moment a sweep bar closes, ship
"this level is armed, entry on touch, valid for W bars". The consumer
watches its own price feed and fills AT THE LEVEL — which is exactly what
the frozen backtest assumes, so the 0.1328R gap closes by construction
rather than being estimated away.

Timing is what makes it work: the sweep happens on bar j, the retest fills
somewhere in j+1..j+W (up to 8 hours later). Publishing at j's close needs
only the ~5 minute train latency and still lands before the fill window
opens.

Definitions are IMPORTED, never restated: swing sweeps from
sweep_core.detect_sweeps, time-defined pools from level_types.build_levels
with the same live/hit walk as level_types.trade_levels. Variant tagging
stops at A/B on purpose — C and D need 1-minute flow measured AT the fill
(flow_reject / flow_vhigh), which does not exist yet when the level is
merely armed. B is the Gate F track and the product default, so A/B covers
the registered path.

Writes only its own table. Rides the hourly train.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import level_types as LT  # noqa: E402
import sweep_core as SC  # noqa: E402

from research.crowd_battery2 import adx_state  # noqa: E402

_DD = os.environ.get("SWEEP_DATA_DIR", "").strip()
CACHE = (Path(_DD) / ".cache") if _DD else (
    ROOT / "research" / "sweep_failure" / ".cache")
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
PIERCE_MAX_B = 0.25

DDL = """
CREATE TABLE IF NOT EXISTS raid_pending_levels (
  id          BIGINT AUTO_INCREMENT PRIMARY KEY,
  symbol      VARCHAR(16)   NOT NULL,
  side        ENUM('LONG','SHORT') NOT NULL,
  level_kind  VARCHAR(16)   NOT NULL,
  sweep_ts    BIGINT        NOT NULL,
  sweep_utc   VARCHAR(20)   NOT NULL,
  trigger_px  DECIMAL(20,8) NOT NULL,
  stop_px     DECIMAL(20,8) NOT NULL,
  atr         DECIMAL(20,8) NOT NULL,
  risk_frac   DECIMAL(12,8) NOT NULL,
  pierce_atr  DECIMAL(12,6) NOT NULL,
  variants    VARCHAR(32)   NOT NULL,
  regime_cell VARCHAR(12)   NOT NULL DEFAULT '',
  expires_ts  BIGINT        NOT NULL,
  universe    VARCHAR(16)   NOT NULL,
  updated_at  DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP
                            ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uq_pending (symbol, level_kind, sweep_ts, trigger_px)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def sweep_events(bars):
    """All armed sweeps as (j, kd, lvl, kind) — one definition per kind.

    swing: sweep_core.detect_sweeps. Time pools: the same live/hit walk
    level_types.trade_levels performs, so a pool is 'swept' on the bar that
    trades through it and is consumed exactly once.
    """
    out = []
    for e in SC.detect_sweeps(bars):
        out.append((e["j"], 1 if e["kind"] == "buy" else -1, e["level"],
                    "swing"))
    h = [b[SC.H] for b in bars]
    lo = [b[SC.L] for b in bars]
    lv = LT.build_levels(bars)
    for kind in ("session", "pdh_pdl", "pwh_pwl"):
        pending = sorted(lv.get(kind, []))
        live: list[tuple[float, int]] = []
        idx = 0
        for j in range(len(bars)):
            while idx < len(pending) and pending[idx][0] <= j:
                live.append((pending[idx][1], pending[idx][2]))
                idx += 1
            hit = [(p, s) for p, s in live
                   if (h[j] > p if s == 1 else lo[j] < p)]
            if not hit:
                continue
            live = [x for x in live if x not in hit]
            for lvl, s in hit:
                out.append((j, s, lvl, kind))
    # build_levels can register the same price twice (a session high that is
    # also a prior-day high inside its own family); the frozen trade_levels
    # walk absorbs that by consuming the pool, but a pending row would be a
    # duplicate invitation. Dedup on the identity that matters.
    return list(dict.fromkeys(out))


def armed_levels(bars):
    """Sweeps still inside their retest window with no touch yet."""
    n = len(bars)
    h = [b[SC.H] for b in bars]
    lo = [b[SC.L] for b in bars]
    a = SC.atr14(bars)
    c = [b[SC.C] for b in bars]
    last_ts = bars[-1][0]
    # §0.59: the frozen ADX(14) 25/20 cell with the §0.54b direction split,
    # evaluated AT THE SWEEP BAR — that is the moment a consumer can act on.
    # The shadow log scores the cell at the FILL bar instead (1-8h later, so
    # the label can move); that difference is intentional and harmless: the
    # product side validates the pipeline, the ledger decides the edge.
    adx = adx_state(bars)
    LB = 24
    out = []
    for j, kd, lvl, kind in sweep_events(bars):
        if a[j] is None or a[j] == 0:
            continue
        # window is j+1 .. j+W; the last CLOSED bar is n-1
        if n - 1 >= j + SC.W:
            continue                      # window already elapsed
        touched = any((kd == 1 and lo[f] <= lvl) or (kd == -1 and h[f] >= lvl)
                      for f in range(j + 1, n))
        if touched:
            continue                      # already filled — the other
            #                               publisher owns it from here
        A = a[j]
        d = -kd                            # trade direction
        risk = SC.DIS * A
        lab = adx.get(bars[j][0] // 3600 * 3600)
        if lab is None or j < LB:
            cell = ""
        elif lab == "RANGING":
            cell = "RANGING"
        elif lab != "TRENDING":
            cell = "NEUTRAL"
        else:
            cell = "TREND_UP" if c[j] / c[j - LB] - 1 > 0 else "TREND_DOWN"
        out.append({
            "regime_cell": cell,
            "sweep_ts": bars[j][0], "kd": kd, "lvl": lvl, "kind": kind,
            "atr": A, "side": "LONG" if d == 1 else "SHORT",
            "stop_px": lvl - d * risk,
            "risk_frac": risk / lvl if lvl else 0.0,
            "pierce": ((h[j] - lvl) if kd == 1 else (lvl - lo[j])) / A,
            "expires_ts": bars[j][0] + SC.W * 3600,
            "last_bar_ts": last_ts,
        })
    return out


def main() -> int:
    rows = []
    for sym in CORE9:
        fp = CACHE / f"{sym}USDT_1h.csv"
        if not fp.exists():
            continue
        try:
            bars = SC.load_csv(str(fp))
        except Exception as e:  # noqa: BLE001
            print(f"  {sym}: load failed ({e})")
            continue
        if len(bars) < 100:
            continue
        for x in armed_levels(bars):
            variants = "A,B" if x["pierce"] <= PIERCE_MAX_B else "A"
            rows.append((
                sym, x["side"], x["kind"], int(x["sweep_ts"]),
                f"{datetime.fromtimestamp(x['sweep_ts'], timezone.utc):%Y-%m-%d %H:%M}",
                round(x["lvl"], 8), round(x["stop_px"], 8),
                round(x["atr"], 8), round(x["risk_frac"], 8),
                round(x["pierce"], 6), variants, x.get("regime_cell", ""),
                int(x["expires_ts"]), "core9"))

    from shared.db import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(DDL)
            # wholesale replace: a level that filled or expired must simply
            # vanish, never linger as a stale invitation to trade.
            cur.execute("DELETE FROM raid_pending_levels")
            if rows:
                cur.executemany(
                    "INSERT IGNORE INTO raid_pending_levels (symbol, side, "
                    "level_kind, sweep_ts, sweep_utc, trigger_px, stop_px, "
                    "atr, risk_frac, pierce_atr, variants, regime_cell, "
                    "expires_ts, universe) VALUES "
                    "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    rows)
        conn.commit()
    finally:
        conn.close()
    nb = sum(1 for r in rows if "B" in r[10].split(","))
    now = int(time.time())
    soon = sum(1 for r in rows if r[11] > now)
    print(f"raid_pending_levels: published {len(rows)} armed "
          f"({nb} variant-B, {soon} not yet expired)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
