"""Crowd-strategy battery — are popular strategies' P&L states the regime?

PRE-REGISTERED 2026-08-17 (TODO §0.49c); definitions and predictions frozen
and committed before this script produced any number.

The user's insight: the strategies worth monitoring need not be ours.
Popular textbook strategies are run by real crowds, so their paper P&L is a
free sensor of WHO IS BEING FED right now — and our strategies' counter-
parties are often exactly those crowds.  Sweep-failure harvests trapped
breakout entrants, so "is the breakout crowd being paid" is nearly the
direct complement of SF's mechanism — one mechanical layer closer than the
trend_z of §0.49 (whose first run was too wide to hold).

Battery (textbook defaults, deliberately untuned — tuning would stop them
being "popular"): SMA50/200 cross (trend archetype), Wilder RSI(14) 30/70
(mean-reversion archetype), Donchian-20 stop-and-reverse (breakout
archetype).  State = sign of trailing 720-bar (30d) paper return: PAID /
STARVED, binary.

Boundary note: the 2026-04-01 rule (no technical indicators in model
feature sets) is untouched — the indicators here are the OBJECT of study
(crowd-behavior proxies), feed no model, predict no price.

Frozen predictions: B-P1 SF netR STARVED(breakout) > PAID(breakout);
B-P2 V7 Strong WR PAID(mr) > STARVED(mr); B-P3 V7 Strong WR
PAID(trend) < STARVED(trend).  Marginals only; the 2x2x2 grid is reported
but not bet on.  Any state with <5% time-share = instrument suspect.
Read-only research code.
"""
from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research.survival_cards import CACHE, CORE9, SC, day_of  # noqa: E402

WINDOW = 720          # 30d of 1h bars — same lookback as trend_z
BOOT_N = 2000
SEED = 7


# ── archetype position series (each: bars -> pos[i] in {-1,0,+1}) ───────

def pos_trend(bars) -> list[int]:
    c = [b[SC.C] for b in bars]
    n = len(c)
    pos = [0] * n
    s50 = s200 = 0.0
    for i in range(n):
        s50 += c[i] - (c[i - 50] if i >= 50 else 0)
        s200 += c[i] - (c[i - 200] if i >= 200 else 0)
        if i >= 200:
            pos[i] = 1 if s50 / 50 > s200 / 200 else -1
    return pos


def pos_mr(bars) -> list[int]:
    c = [b[SC.C] for b in bars]
    n = len(c)
    pos = [0] * n
    up = dn = 0.0
    state = 0
    for i in range(1, n):
        ch = c[i] - c[i - 1]
        gain, loss = max(ch, 0.0), max(-ch, 0.0)
        if i <= 14:
            up += gain / 14
            dn += loss / 14
            continue
        up = (up * 13 + gain) / 14        # Wilder smoothing
        dn = (dn * 13 + loss) / 14
        rsi = 100.0 if dn == 0 else 100 - 100 / (1 + up / dn)
        if state == 0:
            if rsi < 30:
                state = 1
            elif rsi > 70:
                state = -1
        elif state == 1 and rsi > 50:
            state = 0
        elif state == -1 and rsi < 50:
            state = 0
        pos[i] = state
    return pos


def pos_breakout(bars) -> list[int]:
    h = [b[SC.H] for b in bars]
    l = [b[SC.L] for b in bars]
    c = [b[SC.C] for b in bars]
    n = len(c)
    pos = [0] * n
    state = 0
    for i in range(20, n):
        hi20 = max(h[i - 20:i])
        lo20 = min(l[i - 20:i])
        if c[i] > hi20:
            state = 1
        elif c[i] < lo20:
            state = -1
        pos[i] = state
    return pos


ARCHETYPES = {"trend": pos_trend, "mr": pos_mr, "breakout": pos_breakout}


def paid_states(bars) -> dict[str, dict[int, int]]:
    """archetype -> {hour_ts: +1 PAID / -1 STARVED} from trailing 720-bar
    paper return.  Position taken at bar close earns the NEXT bar's return
    (no same-bar lookahead)."""
    c = [b[SC.C] for b in bars]
    n = len(c)
    rets = [0.0] + [math.log(c[i] / c[i - 1]) for i in range(1, n)]
    out: dict[str, dict[int, int]] = {}
    for name, fn in ARCHETYPES.items():
        pos = fn(bars)
        pnl = [0.0] * n
        for i in range(1, n):
            pnl[i] = pos[i - 1] * rets[i]
        roll: dict[int, int] = {}
        acc = sum(pnl[:WINDOW])
        for i in range(WINDOW, n):
            acc += pnl[i] - pnl[i - WINDOW]
            roll[bars[i][0] // 3600 * 3600] = 1 if acc > 0 else -1
        out[name] = roll
    return out


def share(states: dict[int, int]) -> float:
    v = list(states.values())
    return sum(1 for x in v if x > 0) / len(v) if v else float("nan")


def clustered_diff_ci(a, b):
    """Day-clustered bootstrap CI on mean(a)-mean(b); items (day, value)."""
    if not a or not b:
        return 0.0, 0.0, 0.0
    da, db = defaultdict(list), defaultdict(list)
    for d, v in a:
        da[d].append(v)
    for d, v in b:
        db[d].append(v)
    ka, kb = list(da.values()), list(db.values())
    rng = random.Random(SEED)
    diffs = []
    for _ in range(BOOT_N):
        fa = [x for _ in range(len(ka)) for x in ka[rng.randrange(len(ka))]]
        fb = [x for _ in range(len(kb)) for x in kb[rng.randrange(len(kb))]]
        diffs.append(sum(fa) / len(fa) - sum(fb) / len(fb))
    diffs.sort()
    pt = sum(v for _, v in a) / len(a) - sum(v for _, v in b) / len(b)
    return pt, diffs[int(0.025 * BOOT_N)], diffs[int(0.975 * BOOT_N)]


def main():
    # ── B-P1: SF trades vs the breakout crowd's state (per-coin battery) ─
    print("════ B-P1  掃單失敗 × 突破原型（對手盤直連）════")
    poolP, poolS = [], []
    coin_sign = []
    for sym in CORE9:
        fp = CACHE / f"{sym}USDT_1h.csv"
        if not fp.exists():
            continue
        bars = SC.load_csv(str(fp))
        st = paid_states(bars)["breakout"]
        pP, pS = [], []
        for fill_ts, _e, R, *_ in SC.backtest_symbol(bars):
            s = st.get(int(fill_ts) // 3600 * 3600)
            if s is None:
                continue
            (pP if s > 0 else pS).append((day_of(int(fill_ts)), R))
        mP = sum(v for _, v in pP) / len(pP) if pP else float("nan")
        mS = sum(v for _, v in pS) / len(pS) if pS else float("nan")
        poolP += pP
        poolS += pS
        if pP and pS:
            coin_sign.append((sym, mS - mP))
        print(f"  {sym:<6} PAID n={len(pP):>4} meanR={mP:+.4f}   "
              f"STARVED n={len(pS):>4} meanR={mS:+.4f}   S−P={mS-mP:+.4f}")
    pt, lo, hi = clustered_diff_ci(poolS, poolP)
    npos = sum(1 for _, d in coin_sign if d > 0)
    print(f"  pooled STARVED−PAID: {pt:+.4f} CI95[{lo:+.4f},{hi:+.4f}]"
          f"  逐幣同號 {npos}/{len(coin_sign)}"
          f"  {'✓' if pt > 0 else '✗'}")

    # ── B-P2/P3: V7 Strong vs mr / trend crowd states (BTC battery) ─────
    from shared.db import get_db_conn
    bars = SC.load_csv(str(CACHE / "BTCUSDT_1h.csv"))
    states = paid_states(bars)
    print("\n狀態佔比 sanity（PAID 時間佔比；<5% 或 >95% = 儀器嫌疑）")
    for name, st in states.items():
        print(f"  {name:<9} PAID {100*share(st):.0f}%  (n={len(st)})")

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

    for pred, arch, want in (("B-P2", "mr", "PAID>STARVED"),
                             ("B-P3", "trend", "PAID<STARVED")):
        st = states[arch]
        gP, gS = [], []
        for r in rows:
            ts = int(r["signal_time"].replace(tzinfo=timezone.utc).timestamp())
            s = st.get(ts // 3600 * 3600)
            if s is None:
                continue
            item = (day_of(ts), float(r["correct"] or 0))
            (gP if s > 0 else gS).append(item)
        wrP = 100 * sum(v for _, v in gP) / len(gP) if gP else float("nan")
        wrS = 100 * sum(v for _, v in gS) / len(gS) if gS else float("nan")
        pt, lo, hi = clustered_diff_ci(gP, gS)
        ok = (pt > 0) if want == "PAID>STARVED" else (pt < 0)
        print(f"\n════ {pred}  V7 Strong × {arch} 原型（凍結預測 {want}）════")
        print(f"  PAID    n={len(gP):>4}  WR={wrP:.1f}%")
        print(f"  STARVED n={len(gS):>4}  WR={wrS:.1f}%")
        print(f"  PAID−STARVED: {100*pt:+.1f}pp CI95[{100*lo:+.1f},{100*hi:+.1f}]pp"
              f"  {'✓' if ok else '✗'}")

    # ── full 2x2x2 grid, reported not bet on ────────────────────────────
    print("\n2×2×2 全格（照報不下注）: V7 Strong WR by (trend, mr, breakout)")
    grid = defaultdict(list)
    for r in rows:
        ts = int(r["signal_time"].replace(tzinfo=timezone.utc).timestamp())
        key = tuple("P" if states[a].get(ts // 3600 * 3600, 0) > 0 else "S"
                    for a in ("trend", "mr", "breakout"))
        if all(states[a].get(ts // 3600 * 3600) is not None
               for a in ("trend", "mr", "breakout")):
            grid[key].append(int(r["correct"] or 0))
    for key in sorted(grid):
        v = grid[key]
        print(f"  T{key[0]} M{key[1]} B{key[2]}  n={len(v):>4}  "
              f"WR={100*sum(v)/len(v):.0f}%")


if __name__ == "__main__":
    main()
