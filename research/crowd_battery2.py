"""Crowd battery v2 — user-supplied crowd toolbox, six tested cells, two
head-to-heads.  PRE-REGISTERED 2026-08-17 (TODO §0.49d); frozen before any
number existed.

The user (an active discretionary trader) supplied what crypto crowds
actually run, replacing my textbook guesses: SuperTrend is the real
breakout/trend tool, ADX is the crowd's own regime gauge.  Six tested
predictions (B-P4..P9), three display-only sensors (Stoch/VWAP/Ichimoku),
and two head-to-heads whose winners take the clocks seat:

    SuperTrend vs Donchian      -> who is SF's counterparty gauge
    ADX vs trend_z              -> whose regime split is tighter

Pass criteria FROZEN in two tiers this time (the omission that bit twice):
tier-1 (display wiring): correct sign AND magnitude >= 2pp (V7 WR) /
0.01R (SF) AND, on the SF side, per-coin agreement >= 6/9.
tier-2 (alert eligibility): day-clustered bootstrap CI95 clear of zero.
Head-to-head: same-direction points, then narrower CI AND no-worse breadth
wins.  All archetypes 1h, crowd-default params, untuned.
Read-only research code.
"""
from __future__ import annotations

import math
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

from research.crowd_battery import (  # noqa: E402
    WINDOW, clustered_diff_ci, pos_breakout, share)
from research.survival_cards import CACHE, CORE9, SC, day_of  # noqa: E402


# ── helpers ─────────────────────────────────────────────────────────────

def ema_series(vals, n):
    out = [0.0] * len(vals)
    k = 2 / (n + 1)
    out[0] = vals[0]
    for i in range(1, len(vals)):
        out[i] = vals[i] * k + out[i - 1] * (1 - k)
    return out


# ── new archetype position series (crowd-default params, frozen) ────────

def pos_macd(bars):
    c = [b[SC.C] for b in bars]
    e12, e26 = ema_series(c, 12), ema_series(c, 26)
    macd = [a - b for a, b in zip(e12, e26)]
    sig = ema_series(macd, 9)
    return [0 if i < 35 else (1 if macd[i] > sig[i] else -1)
            for i in range(len(c))]


def pos_ema_fast(bars):
    c = [b[SC.C] for b in bars]
    e9, e21 = ema_series(c, 9), ema_series(c, 21)
    return [0 if i < 21 else (1 if e9[i] > e21[i] else -1)
            for i in range(len(c))]


def pos_bb_mr(bars):
    c = [b[SC.C] for b in bars]
    n = len(c)
    pos = [0] * n
    state = 0
    s = sq = 0.0
    for i in range(n):
        s += c[i]
        sq += c[i] * c[i]
        if i >= 20:
            s -= c[i - 20]
            sq -= c[i - 20] * c[i - 20]
        if i < 19:
            continue
        mid = s / 20
        var = max(sq / 20 - mid * mid, 0.0)
        sd = math.sqrt(var)
        if state == 0:
            if c[i] < mid - 2 * sd:
                state = 1
            elif c[i] > mid + 2 * sd:
                state = -1
        elif state == 1 and c[i] >= mid:
            state = 0
        elif state == -1 and c[i] <= mid:
            state = 0
        pos[i] = state
    return pos


def pos_supertrend(bars, atr_n=10, mult=3.0):
    h = [b[SC.H] for b in bars]
    l = [b[SC.L] for b in bars]
    c = [b[SC.C] for b in bars]
    n = len(c)
    # Wilder ATR
    atr = [0.0] * n
    for i in range(1, n):
        tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        atr[i] = tr if i <= atr_n else (atr[i - 1] * (atr_n - 1) + tr) / atr_n
    pos = [0] * n
    up = dn = 0.0
    trend = 0
    for i in range(atr_n + 1, n):
        mid = (h[i] + l[i]) / 2
        bu = mid + mult * atr[i]
        bl = mid - mult * atr[i]
        up = min(bu, up) if c[i - 1] < up else bu      # ratchet
        dn = max(bl, dn) if c[i - 1] > dn else bl
        if trend <= 0 and c[i] > up:
            trend = 1
        elif trend >= 0 and c[i] < dn:
            trend = -1
        pos[i] = trend
    return pos


def adx_state(bars, n_=14):
    """hour_ts -> 'TRENDING' (ADX>25) / 'RANGING' (<20) / 'NEUTRAL'."""
    h = [b[SC.H] for b in bars]
    l = [b[SC.L] for b in bars]
    c = [b[SC.C] for b in bars]
    n = len(c)
    out: dict[int, str] = {}
    tr_s = pdm_s = ndm_s = 0.0
    adx = None
    for i in range(1, n):
        tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        um, dm = h[i] - h[i - 1], l[i - 1] - l[i]
        pdm = um if (um > dm and um > 0) else 0.0
        ndm = dm if (dm > um and dm > 0) else 0.0
        if i <= n_:
            tr_s += tr
            pdm_s += pdm
            ndm_s += ndm
            continue
        tr_s = tr_s - tr_s / n_ + tr
        pdm_s = pdm_s - pdm_s / n_ + pdm
        ndm_s = ndm_s - ndm_s / n_ + ndm
        if tr_s <= 0:
            continue
        pdi, ndi = 100 * pdm_s / tr_s, 100 * ndm_s / tr_s
        dx = 100 * abs(pdi - ndi) / (pdi + ndi) if pdi + ndi > 0 else 0.0
        adx = dx if adx is None else (adx * (n_ - 1) + dx) / n_
        if i > 2 * n_:
            st = ("TRENDING" if adx > 25 else
                  "RANGING" if adx < 20 else "NEUTRAL")
            out[bars[i][0] // 3600 * 3600] = st
    return out


# sensors (display-only, zero predictions)

def pos_stoch(bars):
    h = [b[SC.H] for b in bars]
    l = [b[SC.L] for b in bars]
    c = [b[SC.C] for b in bars]
    n = len(c)
    pos = [0] * n
    state = 0
    for i in range(14, n):
        hh, ll = max(h[i - 13:i + 1]), min(l[i - 13:i + 1])
        k = 100 * (c[i] - ll) / (hh - ll) if hh > ll else 50.0
        if state == 0:
            if k < 20:
                state = 1
            elif k > 80:
                state = -1
        elif state == 1 and k > 50:
            state = 0
        elif state == -1 and k < 50:
            state = 0
        pos[i] = state
    return pos


def pos_vwap(bars):
    pos = [0] * len(bars)
    day = None
    cum_pv = cum_v = 0.0
    for i, b in enumerate(bars):
        d = b[0] // 86400
        if d != day:
            day, cum_pv, cum_v = d, 0.0, 0.0
        tp = (b[SC.H] + b[SC.L] + b[SC.C]) / 3
        cum_pv += tp * b[SC.V]
        cum_v += b[SC.V]
        if cum_v > 0:
            pos[i] = 1 if b[SC.C] > cum_pv / cum_v else -1
    return pos


def pos_ichimoku(bars):
    h = [b[SC.H] for b in bars]
    l = [b[SC.L] for b in bars]
    c = [b[SC.C] for b in bars]
    n = len(c)
    pos = [0] * n
    for i in range(78, n):        # 52 + 26 displacement
        j = i - 26
        ten = (max(h[j - 8:j + 1]) + min(l[j - 8:j + 1])) / 2
        kij = (max(h[j - 25:j + 1]) + min(l[j - 25:j + 1])) / 2
        sa = (ten + kij) / 2
        sb = (max(h[j - 51:j + 1]) + min(l[j - 51:j + 1])) / 2
        top, bot = max(sa, sb), min(sa, sb)
        pos[i] = 1 if c[i] > top else (-1 if c[i] < bot else 0)
    return pos


ARCH_V2 = {"macd": pos_macd, "ema_fast": pos_ema_fast, "bb_mr": pos_bb_mr,
           "supertrend": pos_supertrend, "donchian": pos_breakout,
           "stoch": pos_stoch, "vwap": pos_vwap, "ichimoku": pos_ichimoku}


def paid_states_v2(bars, names):
    c = [b[SC.C] for b in bars]
    n = len(c)
    rets = [0.0] + [math.log(c[i] / c[i - 1]) for i in range(1, n)]
    out = {}
    for name in names:
        pos = ARCH_V2[name](bars)
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


def sf_split(arch: str):
    """(pooled_hi, pooled_lo, breadth) for SF trades: hi = the cohort the
    prediction says is BETTER (STARVED for trend/breakout tools, PAID for
    bb_mr), lo = the other."""
    better_is_paid = arch == "bb_mr"
    hi, lo, diffs = [], [], []
    for sym in CORE9:
        fp = CACHE / f"{sym}USDT_1h.csv"
        if not fp.exists():
            continue
        bars = SC.load_csv(str(fp))
        st = paid_states_v2(bars, [arch])[arch]
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
    breadth = sum(1 for d in diffs if d > 0)
    return hi, lo, breadth, len(diffs)


def sf_adx_split():
    hi, lo, diffs = [], [], []
    for sym in CORE9:
        fp = CACHE / f"{sym}USDT_1h.csv"
        if not fp.exists():
            continue
        bars = SC.load_csv(str(fp))
        st = adx_state(bars)
        a, b = [], []
        for fill_ts, _e, R, *_ in SC.backtest_symbol(bars):
            s = st.get(int(fill_ts) // 3600 * 3600)
            if s == "RANGING":
                a.append((day_of(int(fill_ts)), R))
            elif s == "TRENDING":
                b.append((day_of(int(fill_ts)), R))
        hi += a
        lo += b
        if a and b:
            diffs.append(sum(v for _, v in a) / len(a)
                         - sum(v for _, v in b) / len(b))
    return hi, lo, sum(1 for d in diffs if d > 0), len(diffs)


def v7_signals():
    from shared.db import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT signal_time, correct FROM tracked_signals "
                "WHERE strength='Strong' AND actual_return_4h IS NOT NULL "
                "AND direction IN ('UP','DOWN')")
            return [(int(r["signal_time"].replace(tzinfo=timezone.utc)
                         .timestamp()), int(r["correct"] or 0))
                    for r in cur.fetchall()]
    finally:
        conn.close()


def report_cell(name, hi, lo, breadth=None, breadth_n=None,
                unit="R", thresh=0.01):
    if not hi or not lo:
        print(f"  {name}: empty cohort — instrument suspect")
        return
    pt, clo, chi = clustered_diff_ci(hi, lo)
    mh = sum(v for _, v in hi) / len(hi)
    ml = sum(v for _, v in lo) / len(lo)
    scale = 100 if unit == "pp" else 1
    tier1 = pt > 0 and abs(pt) * scale >= (2 if unit == "pp" else thresh) * (
        1 if unit == "pp" else 1)
    if unit == "pp":
        tier1 = pt > 0 and pt * 100 >= 2
    else:
        tier1 = pt > 0 and pt >= 0.01
    if breadth is not None:
        tier1 = tier1 and breadth >= 6
    tier2 = clo > 0
    b = f"  breadth {breadth}/{breadth_n}" if breadth is not None else ""
    print(f"  {name}: better n={len(hi)} ({scale*mh:+.3f}) vs "
          f"worse n={len(lo)} ({scale*ml:+.3f})  "
          f"diff {scale*pt:+.3f}{unit} CI[{scale*clo:+.3f},{scale*chi:+.3f}]{b}"
          f"  tier1 {'✓' if tier1 else '✗'}  tier2 {'✓' if tier2 else '✗'}")
    return pt, clo, chi


def main():
    sigs = v7_signals()
    btc = SC.load_csv(str(CACHE / "BTCUSDT_1h.csv"))

    print("════ V7 側（單位 pp of WR）════")
    states = paid_states_v2(btc, ["macd", "ema_fast"])
    for pred, arch in (("B-P4 MACD", "macd"), ("B-P6 EMA9/21", "ema_fast")):
        st = states[arch]
        # prediction: STARVED better for V7
        hi = [(day_of(t), float(c)) for t, c in sigs
              if st.get(t // 3600 * 3600) == -1]
        lo = [(day_of(t), float(c)) for t, c in sigs
              if st.get(t // 3600 * 3600) == 1]
        report_cell(pred + " (STARVED−PAID)", hi, lo, unit="pp")

    ast = adx_state(btc)
    hi = [(day_of(t), float(c)) for t, c in sigs
          if ast.get(t // 3600 * 3600) == "RANGING"]
    lo = [(day_of(t), float(c)) for t, c in sigs
          if ast.get(t // 3600 * 3600) == "TRENDING"]
    report_cell("B-P8 ADX (RANGING−TRENDING)", hi, lo, unit="pp")
    sh = defaultdict(int)
    for v in ast.values():
        sh[v] += 1
    tot = sum(sh.values())
    print(f"  ADX 佔比: " + " ".join(f"{k} {100*v/tot:.0f}%"
                                     for k, v in sorted(sh.items())))

    print("\n════ SF 側（單位 R）════")
    for pred, arch in (("B-P5 BB回歸 (PAID−STARVED)", "bb_mr"),
                       ("B-P7 SuperTrend (STARVED−PAID)", "supertrend"),
                       ("head2head Donchian (STARVED−PAID)", "donchian")):
        hi, lo, br, brn = sf_split(arch)
        report_cell(pred, hi, lo, br, brn, unit="R")

    hi, lo, br, brn = sf_adx_split()
    report_cell("B-P9 ADX (RANGING−TRENDING)", hi, lo, br, brn, unit="R")

    print("\n════ 感測器現況（純顯示，零預測）════")
    sens = paid_states_v2(btc, ["stoch", "vwap", "ichimoku"])
    for name, st in sens.items():
        cur = "PAID" if list(st.values())[-1] > 0 else "STARVED"
        print(f"  {name:<9} now {cur:<8} (PAID share {100*share(st):.0f}%)")


if __name__ == "__main__":
    main()
