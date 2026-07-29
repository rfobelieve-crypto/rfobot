"""Sweep-failure reversal — core detection & measurement (strategy #3 candidate).

Hypothesis (pre-registered, see README.md):
    A liquidity sweep of a swing high/low that RETURNS to the swept level
    within W bars has *failed*; price then drifts in the pierce-through
    direction for ~8-12 (1h) bars.

Origin & discipline notes:
    · Distilled from the user's discretionary "假突破反轉" thesis, mechanised
      2026-07-11 in the trading-view-MCP research sandbox.
    · The hypothesis was reached after observing the *negative* of a
      limit-retest continuation test (sign-flip origin) → apply a snooping
      discount; forward validation is the real gate.
    · Known artifact class this went through: outcome MUST be measured from a
      genuinely fillable price (the level at retest-touch), never from
      pre-signal prices. See README §history.

Validation summary (2024-01→2026-07, 9 × 22k 1h bars, cost 0.05 ATR/side):
    · 9/9 symbols positive (PF 1.16-1.54, WR 54-59%), pooled t=8.27
    · Param-robust: W∈{4,8,12}, HOLD∈{8,12}, PIVOT∈{5,10} all positive
    · Effect dies by HOLD=20 (short-lived flow effect)
    · Cost-sensitive: 0.10 ATR total → not significant. Maker-tier fees are
      a hard requirement for live.

Read-only research code — no production imports, no DB writes.
"""
from __future__ import annotations

import csv
import math
import os
from collections import deque

PIVOT = int(os.environ.get("PIVOT", "10"))
W = int(os.environ.get("W", "8"))            # bars to wait for retest
HOLD = int(os.environ.get("HOLD", "8"))      # holding bars after fill
DIS = float(os.environ.get("DIS", "3.5"))    # disaster stop, ATR mult
SLIP = float(os.environ.get("SLIP", "0.05")) # per-side cost, ATR units

O, H, L, C, V = 1, 2, 3, 4, 5


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        next(r)
        for x in r:
            if len(x) < 6:
                continue
            try:
                rows.append((int(float(x[0])), float(x[1]), float(x[2]),
                             float(x[3]), float(x[4]), float(x[5])))
            except ValueError:
                continue
    return rows


def atr14(bars):
    n = len(bars)
    out = [None] * n
    trs = []
    prev = None
    a = None
    for i in range(n):
        h, l = bars[i][H], bars[i][L]
        tr = h - l if i == 0 else max(h - l, abs(h - bars[i - 1][C]), abs(l - bars[i - 1][C]))
        trs.append(tr)
        if len(trs) == 14 and a is None:
            a = sum(trs) / 14
        elif a is not None:
            a = (a * 13 + tr) / 14
        out[i] = a
    return out


def detect_sweeps(bars):
    """Pivot(PIVOT) swings; first later bar piercing the level = sweep event.

    Pivot is only *confirmed* PIVOT bars after its extreme (no look-ahead);
    the sweep scan starts strictly after confirmation.
    Returns [{j, kind, level}] sorted by sweep bar j.
    """
    n = len(bars)
    h = [b[H] for b in bars]
    l = [b[L] for b in bars]
    ev = []
    for i in range(PIVOT, n - PIVOT):
        seg = range(i - PIVOT, i + PIVOT + 1)
        if all(h[i] >= h[k] for k in seg) and any(h[i] > h[k] for k in seg if k != i):
            for j in range(i + PIVOT + 1, n):
                if h[j] > h[i]:
                    ev.append(dict(j=j, kind="buy", level=h[i]))
                    break
        if all(l[i] <= l[k] for k in seg) and any(l[i] < l[k] for k in seg if k != i):
            for j in range(i + PIVOT + 1, n):
                if l[j] < l[i]:
                    ev.append(dict(j=j, kind="sell", level=l[i]))
                    break
    ev.sort(key=lambda e: e["j"])
    return ev


def backtest_symbol(bars):
    """Tradeable rules: retest-touch stop-entry at level, disaster stop,
    time exit. One position per symbol (non-overlapping). Returns trade list
    [(fill_ts, exit_ts, R)] with R in disaster-stop units, costs included."""
    n = len(bars)
    h = [b[H] for b in bars]
    l = [b[L] for b in bars]
    c = [b[C] for b in bars]
    a = atr14(bars)
    trades = []
    last_exit = -1
    for e in detect_sweeps(bars):
        j, lvl = e["j"], e["level"]
        if a[j] is None or a[j] == 0:
            continue
        kd = 1 if e["kind"] == "buy" else -1
        d = -kd                      # pierce-through direction
        fill = None
        for f in range(j + 1, min(j + 1 + W, n)):
            if kd == 1 and l[f] <= lvl:
                fill = f
                break
            if kd == -1 and h[f] >= lvl:
                fill = f
                break
        if fill is None or fill <= last_exit or fill + 1 >= n:
            continue
        A = a[j]
        # BUGFIX 2026-07-28: was `lvl - d*SLIP*A` — a FAVORABLE fill that
        # cancelled against the adverse exit slip, making the whole backtest
        # effectively zero-cost (verified: SLIP=0.05 run beat SLIP=0 run).
        # The comment always said "against us"; now the code does too.
        entry = lvl + d * SLIP * A   # stop-entry slippage against us
        risk = DIS * A
        stop = entry - d * risk
        R = None
        exitbar = min(fill + HOLD, n - 1)
        for k in range(fill + 1, min(fill + HOLD + 1, n)):
            if d == 1 and l[k] <= stop:
                # stop-out pays exit slippage too (fill beyond the stop)
                R, exitbar = -1.0 - SLIP / DIS, k
                break
            if d == -1 and h[k] >= stop:
                R, exitbar = -1.0 - SLIP / DIS, k
                break
        stopped = R is not None
        if R is None:
            ex = c[exitbar] - d * SLIP * A
            R = d * (ex - entry) / risk
        # 2026-07-28 additive fields (lvl, A, stopped) so downstream can
        # re-score with per-symbol bps costs; rules unchanged.
        # 2026-07-29 additive: pierce_atr = how far the sweep bar went past the
        # level, in ATR. Known at the sweep bar, i.e. strictly before the fill —
        # it enables the pre-registered shallow-pierce variant without touching
        # the frozen entry/exit rules.
        pierce = (h[j] - lvl if kd == 1 else lvl - l[j]) / A
        trades.append((bars[fill][0], bars[exitbar][0], R, lvl, A, stopped,
                       pierce))
        last_exit = exitbar
    return trades


def metrics(trade_rs, risk_pct=1.0):
    if not trade_rs:
        return None
    eq = peak = 1.0
    mdd = 0.0
    for r in trade_rs:
        eq *= 1 + risk_pct / 100.0 * r
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak * 100)
    wins = [x for x in trade_rs if x > 0]
    losses = [x for x in trade_rs if x <= 0]
    pf = (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else float("inf")
    n = len(trade_rs)
    m = sum(trade_rs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in trade_rs) / (n - 1)) if n > 1 else 0
    t = m / (sd / math.sqrt(n)) if sd > 0 else 0
    return dict(n=n, net=(eq - 1) * 100, pf=pf, wr=len(wins) / n * 100,
                exp=m, mdd=mdd, t=t)
