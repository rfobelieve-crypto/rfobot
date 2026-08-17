"""Crowd battery v3 — pivot S/R crowd, PSAR challenger, funding + grid
sensors.  PRE-REGISTERED 2026-08-17 (TODO §0.49f); frozen before any number.

Tested cells (3): PV-P1 pivot-fade x SF (S/R holding is the daily-scale
isomorph of sweep-failure), PS-P1 psar x V7, PS-P2 psar x SF with a
head-to-head against Donchian for the counterparty seat; PSAR also
challenges SMA50/200 for the V7 headwind seat.  Funding-contrarian and
grid-bot are SENSORS: funding's mechanism link is ambiguous (a squeeze is
violent continuation THEN reversion), grid is the mean-reversion family's
inventory profile and that family is 0-for-3.  Their splits are printed as
EXPLORATORY, not bet on.  Criteria per §0.49d two tiers.
Read-only research code.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research.crowd_battery import WINDOW, clustered_diff_ci, share  # noqa: E402
from research.crowd_battery2 import report_cell  # noqa: E402
from research.survival_cards import CACHE, CORE9, SC, day_of  # noqa: E402

FUND_CACHE = ROOT / "research" / "results" / "funding_cache"
FUND_CACHE.mkdir(parents=True, exist_ok=True)
FUND_TH = 0.0005          # 0.05%/8h — the common "elevated" retail heuristic


# ── position series ─────────────────────────────────────────────────────

def pos_pivot(bars):
    """Floor-pivot fade: long below S1 until back to P, short above R1."""
    n = len(bars)
    pos = [0] * n
    by_day: dict[int, list] = defaultdict(list)
    for i, b in enumerate(bars):
        by_day[b[0] // 86400].append(i)
    days = sorted(by_day)
    state = 0
    for d_prev, d_cur in zip(days, days[1:]):
        idx = by_day[d_prev]
        H = max(bars[i][SC.H] for i in idx)
        L = min(bars[i][SC.L] for i in idx)
        C_ = bars[idx[-1]][SC.C]
        P = (H + L + C_) / 3
        R1, S1 = 2 * P - L, 2 * P - H
        for i in by_day[d_cur]:
            c = bars[i][SC.C]
            if state == 0:
                if c < S1:
                    state = 1
                elif c > R1:
                    state = -1
            elif state == 1 and c >= P:
                state = 0
            elif state == -1 and c <= P:
                state = 0
            pos[i] = state
    return pos


def pos_psar(bars, step=0.02, cap=0.2):
    h = [b[SC.H] for b in bars]
    l = [b[SC.L] for b in bars]
    n = len(bars)
    pos = [0] * n
    if n < 3:
        return pos
    up = True
    sar, ep, af = l[0], h[0], step
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < sar:
                up, sar, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, cap)
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > sar:
                up, sar, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, cap)
        pos[i] = 1 if up else -1
    return pos


def fetch_funding(sym: str) -> dict[int, float]:
    """hour_ts -> funding rate, forward-filled from 8h marks (12mo)."""
    fp = FUND_CACHE / f"{sym}.csv.gz"
    if fp.exists():
        with gzip.open(fp, "rt", newline="") as f:
            marks = [(int(r[0]), float(r[1])) for r in csv.reader(f)]
    else:
        end = int(time.time() * 1000)
        start = end - 365 * 86_400_000
        marks = []
        cur = start
        while cur < end:
            req = urllib.request.Request(
                "https://fapi.binance.com/fapi/v1/fundingRate"
                f"?symbol={sym}USDT&startTime={cur}&endTime={end}&limit=1000",
                headers={"User-Agent": "battery-v3/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.loads(r.read().decode())
            if not d:
                break
            marks += [(int(x["fundingTime"]), float(x["fundingRate"]))
                      for x in d]
            cur = int(d[-1]["fundingTime"]) + 1
            if len(d) < 1000:
                break
            time.sleep(0.1)
        with gzip.open(fp, "wt", newline="") as f:
            w = csv.writer(f)
            for m in marks:
                w.writerow(m)
    out: dict[int, float] = {}
    marks.sort()
    j = 0
    if not marks:
        return out
    lo = marks[0][0] // 3600_000 * 3600
    hi = int(time.time()) // 3600 * 3600
    rate = marks[0][1]
    for ts in range(lo, hi + 1, 3600):
        while j + 1 < len(marks) and marks[j + 1][0] <= ts * 1000:
            j += 1
            rate = marks[j][1]
        out[ts] = rate
    return out


def pos_funding(bars, fund: dict[int, float]):
    pos = [0] * len(bars)
    for i, b in enumerate(bars):
        r = fund.get(b[0] // 3600 * 3600)
        if r is None:
            continue
        if r >= FUND_TH:
            pos[i] = -1
        elif r <= -FUND_TH:
            pos[i] = 1
    return pos


def pos_grid(bars):
    c = [b[SC.C] for b in bars]
    n = len(c)
    pos = [0.0] * n
    s = 0.0
    atr = _atr14(bars)
    for i in range(n):
        s += c[i] - (c[i - WINDOW] if i >= WINDOW else 0)
        if i < WINDOW or atr[i] <= 0:
            continue
        anchor = s / WINDOW
        inv = -(c[i] - anchor) / (0.5 * atr[i])
        pos[i] = max(-5.0, min(5.0, inv)) / 5.0
    return pos


def _atr14(bars):
    n = len(bars)
    atr = [0.0] * n
    trs = []
    for i in range(n):
        if i == 0:
            tr = bars[i][SC.H] - bars[i][SC.L]
        else:
            pc = bars[i - 1][SC.C]
            tr = max(bars[i][SC.H] - bars[i][SC.L],
                     abs(bars[i][SC.H] - pc), abs(bars[i][SC.L] - pc))
        trs.append(tr)
        if i >= 13:
            atr[i] = sum(trs[i - 13:i + 1]) / 14
    return atr


def paid_states_from_pos(bars, pos):
    c = [b[SC.C] for b in bars]
    n = len(c)
    rets = [0.0] + [math.log(c[i] / c[i - 1]) for i in range(1, n)]
    pnl = [0.0] * n
    for i in range(1, n):
        pnl[i] = pos[i - 1] * rets[i]
    roll: dict[int, int] = {}
    acc = sum(pnl[:WINDOW])
    for i in range(WINDOW, n):
        acc += pnl[i] - pnl[i - WINDOW]
        roll[bars[i][0] // 3600 * 3600] = 1 if acc > 0 else -1
    return roll


def sf_split_by(pos_fn, better_is_paid, with_funding=False):
    hi, lo, diffs = [], [], []
    for sym in CORE9:
        fp = CACHE / f"{sym}USDT_1h.csv"
        if not fp.exists():
            continue
        bars = SC.load_csv(str(fp))
        if with_funding:
            st = paid_states_from_pos(bars, pos_fn(bars, fetch_funding(sym)))
        else:
            st = paid_states_from_pos(bars, pos_fn(bars))
        a, b = [], []
        for fill_ts, _e, R, *_ in SC.backtest_symbol(bars):
            s = st.get(int(fill_ts) // 3600 * 3600)
            if s is None:
                continue
            good = (s > 0) if better_is_paid else (s < 0)
            (a if good else b).append((day_of(int(fill_ts)), R))
        hi += a
        lo += b
        if a and b:
            diffs.append(sum(v for _, v in a) / len(a)
                         - sum(v for _, v in b) / len(b))
    return hi, lo, sum(1 for d in diffs if d > 0), len(diffs)


def main():
    from shared.db import get_db_conn
    btc = SC.load_csv(str(CACHE / "BTCUSDT_1h.csv"))
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT signal_time, correct FROM tracked_signals "
                "WHERE strength='Strong' AND actual_return_4h IS NOT NULL "
                "AND direction IN ('UP','DOWN')")
            sigs = [(int(r["signal_time"].replace(tzinfo=timezone.utc)
                         .timestamp()), int(r["correct"] or 0))
                    for r in cur.fetchall()]
    finally:
        conn.close()

    print("════ 受測格 ════")
    hi, lo, br, brn = sf_split_by(pos_pivot, better_is_paid=True)
    report_cell("PV-P1 Pivot-fade×SF (PAID−STARVED)", hi, lo, br, brn, unit="R")

    st = paid_states_from_pos(btc, pos_psar(btc))
    g_hi = [(day_of(t), float(c)) for t, c in sigs
            if st.get(t // 3600 * 3600) == -1]
    g_lo = [(day_of(t), float(c)) for t, c in sigs
            if st.get(t // 3600 * 3600) == 1]
    report_cell("PS-P1 PSAR×V7 (STARVED−PAID)", g_hi, g_lo, unit="pp")

    hi, lo, br, brn = sf_split_by(pos_psar, better_is_paid=False)
    report_cell("PS-P2 PSAR×SF (STARVED−PAID)", hi, lo, br, brn, unit="R")

    print("\n════ 感測器（探索性讀數，不下注）════")
    hi, lo, br, brn = sf_split_by(pos_funding, better_is_paid=True,
                                  with_funding=True)
    report_cell("funding-contra×SF (PAID−STARVED) [EXPLORATORY]",
                hi, lo, br, brn, unit="R")
    for name, fn in (("grid", pos_grid),):
        stx = paid_states_from_pos(btc, fn(btc))
        cur_st = "PAID" if list(stx.values())[-1] > 0 else "STARVED"
        print(f"  {name:<8} now {cur_st}  (PAID share {100*share(stx):.0f}%)")
    fst = paid_states_from_pos(btc, pos_funding(btc, fetch_funding("BTC")))
    if fst:
        print(f"  funding  now "
              f"{'PAID' if list(fst.values())[-1] > 0 else 'STARVED'}"
              f"  (PAID share {100*share(fst):.0f}%)")


if __name__ == "__main__":
    main()
