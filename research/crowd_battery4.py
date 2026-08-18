"""Crowd battery v4 — carry farmers and liquidation-bounce hunters.
PRE-REGISTERED 2026-08-18 (TODO §0.49g); frozen before any number.

Source audit (user-supplied awesome-systematic-trading): its crypto section
is thin (2 strategies) and carry lives in the FX section — the crypto-crowd
translation is ours and is stated as such.  Two tested cells:

  CY-P1  funding-carry farmers (basis farming, a genuinely large crypto
         crowd).  RICH = trailing 7d mean funding > +0.01%/8h (Binance
         baseline).  Prediction: V7 Strong DOWN win-rate RICH > THIN —
         rich carry = crowded longs = squeeze fuel for short signals (the
         mechanism candidate behind the live SHORT-side edge).  UP row
         reported, not bet.
  LB-P1  liquidation-bounce hunters (crypto-native, the priority gap;
         absent from the source repo — noted honestly).  Burst = hourly
         long-liq USD >= 3.0x trailing-24h mean (3.0 per the frozen
         cancel_shock intensity precedent); the crowd's play is LONG for
         the 4 bars after a burst.  Prediction: BTC sweep-core LONG-side
         trades (the M2 side column's first research use) earn more when
         the bounce crowd is PAID — a liquidation cascade sweeping the
         lows and bouncing IS the low-side sweep-failure trade.

Criteria: §0.49d two tiers, single-asset (no breadth term): tier-1 = sign
plus >=2pp (WR) / >=0.01R; tier-2 = CI95 clear of zero.
Read-only research code.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from datetime import timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research.crowd_battery2 import report_cell  # noqa: E402
from research.crowd_battery3 import (  # noqa: E402
    fetch_funding, paid_states_from_pos)
from research.survival_cards import CACHE, SC, day_of  # noqa: E402

# ── frozen 2026-08-18 ───────────────────────────────────────────────────
CARRY_BASELINE = 0.0001      # +0.01%/8h, the Binance baseline rate
CARRY_WIN_H = 168            # 7d of hourly forward-filled marks
LIQ_BURST_MULT = 3.0         # cancel_shock intensity precedent
LIQ_TRAIL_H = 24
LIQ_HOLD_H = 4
# ────────────────────────────────────────────────────────────────────────

LIQ_PARQUET = ROOT / "market_data" / "raw_data" / "cg_liq_agg_1h.parquet"


def carry_states() -> dict[int, str]:
    """hour_ts -> RICH / THIN from trailing 7d mean funding (ffilled 1h)."""
    fund = fetch_funding("BTC")
    if not fund:
        return {}
    hours = sorted(fund)
    out: dict[int, str] = {}
    window: list[float] = []
    for ts in hours:
        window.append(fund[ts])
        if len(window) > CARRY_WIN_H:
            window.pop(0)
        if len(window) >= CARRY_WIN_H:
            m = sum(window) / len(window)
            out[ts] = "RICH" if m > CARRY_BASELINE else "THIN"
    return out


def liq_burst_hours() -> set[int]:
    import pandas as pd
    df = pd.read_parquet(LIQ_PARQUET)
    long_liq = df["aggregated_long_liquidation_usd"]
    trail = long_liq.rolling(LIQ_TRAIL_H, min_periods=LIQ_TRAIL_H).mean()
    burst = long_liq >= LIQ_BURST_MULT * trail
    burst &= trail > 0
    return {int(ts.timestamp()) // 3600 * 3600
            for ts, b in burst.items() if b}


def pos_liq_bounce(bars, bursts: set[int]):
    """LONG for LIQ_HOLD_H bars after a long-liquidation burst."""
    pos = [0] * len(bars)
    for i, b in enumerate(bars):
        h = b[0] // 3600 * 3600
        if any((h - k * 3600) in bursts for k in range(1, LIQ_HOLD_H + 1)):
            pos[i] = 1
    return pos


def main() -> None:
    from shared.db import get_db_conn
    btc = SC.load_csv(str(CACHE / "BTCUSDT_1h.csv"))

    # ── CY-P1: V7 Strong DOWN x carry richness ──────────────────────────
    cs = carry_states()
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT signal_time, direction, correct FROM tracked_signals "
                "WHERE strength='Strong' AND actual_return_4h IS NOT NULL "
                "AND direction IN ('UP','DOWN')")
            rows = cur.fetchall()
    finally:
        conn.close()

    print("════ CY-P1  V7 Strong DOWN × carry 農夫（RICH−THIN）════")
    for direction, bet in (("DOWN", True), ("UP", False)):
        hi, lo = [], []
        for r in rows:
            if r["direction"] != direction:
                continue
            ts = int(r["signal_time"].replace(tzinfo=timezone.utc)
                     .timestamp()) // 3600 * 3600
            st = cs.get(ts)
            if st is None:
                continue
            item = (day_of(ts), float(r["correct"] or 0))
            (hi if st == "RICH" else lo).append(item)
        tag = "CY-P1" if bet else "(UP 列，照報不下注)"
        report_cell(f"{tag} {direction} RICH−THIN", hi, lo, unit="pp")
    share = (sum(1 for v in cs.values() if v == "RICH") / len(cs)
             if cs else float("nan"))
    print(f"  carry RICH 時間佔比 {100*share:.0f}%  (n={len(cs)} hours)")

    # ── LB-P1: BTC SF LONG-side trades x bounce-crowd state ─────────────
    bursts = liq_burst_hours()
    lb_states = paid_states_from_pos(btc, pos_liq_bounce(btc, bursts))
    hi, lo = [], []
    n_long = 0
    for fill_ts, _x, R, _lvl, _atr, _st, _pc, side in SC.backtest_symbol(btc):
        if side != "LONG":
            continue
        n_long += 1
        st = lb_states.get(int(fill_ts) // 3600 * 3600)
        if st is None:
            continue
        item = (day_of(int(fill_ts)), R)
        (hi if st > 0 else lo).append(item)
    print("\n════ LB-P1  BTC SF LONG側 × 清算搶反彈群眾（PAID−STARVED）════")
    report_cell("LB-P1 LONG trades", hi, lo, unit="R")
    print(f"  (BTC LONG-side trades total {n_long}; with liq-state "
          f"{len(hi)+len(lo)}; bursts {len(bursts)} hours; "
          f"liq data from 2025-10-22)")

    # sensor snapshot for the weather station
    cur_carry = list(cs.values())[-1] if cs else "?"
    cur_lb = ("PAID" if list(lb_states.values())[-1] > 0 else "STARVED") \
        if lb_states else "?"
    print(f"\nsensors now: carry {cur_carry} / liq-bounce {cur_lb}")


if __name__ == "__main__":
    main()
