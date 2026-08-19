"""Weather-station hold-out check — do the wired gauges still hold on data
they were NOT fitted on?

Every gauge on the board was judged on history up to its registration day.
That is the honest way to register, but it leaves one question open: does
the relationship survive out of sample?  This script re-scores the three
wired gauges on trades filled STRICTLY AFTER their registration dates, i.e.
data that did not exist when the prediction was frozen.

  ADX x SF        registered 2026-08-17 (§0.49d, the only tier-2 gauge)
  Donchian x SF   registered 2026-08-17 (§0.49c)
  PSAR x V7       registered 2026-08-17 (§0.49f)

Same arithmetic as the original cells (day-clustered bootstrap, identical
frozen definitions) — only the sample window moves.  Small-n by
construction: two days of hold-out cannot confirm anything, it can only
show whether the sign flipped.  Reported as a watch item, never as a
verdict.  Read-only.
"""
from __future__ import annotations

import sys
from datetime import timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research.crowd_battery import clustered_diff_ci, paid_states  # noqa: E402
from research.crowd_battery2 import adx_state  # noqa: E402
from research.crowd_battery3 import (  # noqa: E402
    paid_states_from_pos, pos_psar)
from research.survival_cards import CACHE, CORE9, SC, day_of  # noqa: E402

REGISTERED = "2026-08-17"


def sf_split(state_fn, good_label):
    """(better, worse, per-coin diffs) over SF trades filled after REGISTERED."""
    hi, lo, diffs = [], [], []
    for sym in CORE9:
        fp = CACHE / f"{sym}USDT_1h.csv"
        if not fp.exists():
            continue
        bars = SC.load_csv(str(fp))
        st = state_fn(bars)
        a, b = [], []
        for fill_ts, _x, R, *_ in SC.backtest_symbol(bars):
            if day_of(int(fill_ts)) <= REGISTERED:
                continue
            s = st.get(int(fill_ts) // 3600 * 3600)
            if s is None:
                continue
            (a if good_label(s) else b).append((day_of(int(fill_ts)), R))
        hi += a
        lo += b
        if a and b:
            diffs.append(sum(v for _, v in a) / len(a)
                         - sum(v for _, v in b) / len(b))
    return hi, lo, diffs


def report(name, hi, lo, diffs, unit, orig):
    if not hi or not lo:
        print(f"  {name:<26} 樣本不足（better={len(hi)} worse={len(lo)}）"
              f"  註冊時 {orig}")
        return
    pt, clo, chi = clustered_diff_ci(hi, lo)
    scale = 100 if unit == "pp" else 1
    npos = sum(1 for d in diffs if d > 0)
    same = "同號 ✓" if (pt > 0) else "反號 ✗"
    print(f"  {name:<26} n={len(hi)+len(lo):>4}  diff {scale*pt:+7.3f}{unit}"
          f"  CI[{scale*clo:+.3f},{scale*chi:+.3f}]  逐幣 {npos}/{len(diffs)}"
          f"  {same}   註冊時 {orig}")


def main():
    print(f"════ 天氣站 hold-out（僅用 {REGISTERED} 之後成交的樣本）════")

    hi, lo, diffs = sf_split(adx_state, lambda s: s == "RANGING")
    report("ADX×SF (RANGING−TRENDING)", hi, lo, diffs, "R", "+0.059R (tier-2)")

    hi, lo, diffs = sf_split(
        lambda b: paid_states(b)["breakout"], lambda s: s < 0)
    report("突破派×SF (STARVED−PAID)", hi, lo, diffs, "R", "+0.028R")

    # PSAR x V7: BTC only, signal-layer win rate
    from shared.db import get_db_conn
    btc = SC.load_csv(str(CACHE / "BTCUSDT_1h.csv"))
    st = paid_states_from_pos(btc, pos_psar(btc))
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT signal_time, correct FROM tracked_signals "
                "WHERE strength='Strong' AND actual_return_4h IS NOT NULL "
                "AND direction IN ('UP','DOWN') AND signal_time > %s",
                (REGISTERED,))
            rows = cur.fetchall()
    finally:
        conn.close()
    a, b = [], []
    for r in rows:
        ts = int(r["signal_time"].replace(tzinfo=timezone.utc)
                 .timestamp()) // 3600 * 3600
        s = st.get(ts)
        if s is None:
            continue
        (a if s < 0 else b).append((day_of(ts), float(r["correct"] or 0)))
    report("PSAR×V7 (STARVED−PAID)", a, b, [], "pp", "+5.2pp")

    print("\n  註：hold-out 期只有兩天，樣本量不足以確認任何事——這裡只看"
          "符號有沒有翻。翻了才是新聞，沒翻不算證據。")


if __name__ == "__main__":
    main()
