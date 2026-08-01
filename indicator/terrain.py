# -*- coding: utf-8 -*-
"""Terrain stamp for signal alerts (2026-08-02).

One display-only line per Strong/Moderate alert describing the liquidity
terrain the signal fires into, using the three VERIFIED structural
conditions (research/v7_structural_final.py — all passed permutation +
bootstrap + quarterly signs):

  追突破  the signal chases a pool-sweep break from the last 4h  (52% WR)
  前方牆  nearest un-swept pool ahead within 1.4 ATR             (57% vs 65%)
  背靠支撐 nearest un-swept pool behind within 1.8 ATR            (68% WR)

The thresholds are the rounded in-sample near-tercile boundaries; they are
DISPLAY heuristics — the trading rules stay untouched until the registered
forward trigger fires (TODO 0.483). This module must never break an alert:
`terrain_line` returns "" on any problem.

Pool definitions mirror research/sweep_failure (swing pivots ±10 confirmed,
session extremes at session close, prev-day and prev-week extremes), and
the sweep lifecycle consumes a pool the first time price trades through it.
Everything is computed from the trailing window of the live features frame
(needs high/low/close + a UTC DatetimeIndex) — causal by construction.
"""
from __future__ import annotations

PIVOT = 10
LOOK_RAID_H = 4
WALL_ATR = 1.4
SUPPORT_ATR = 1.8
SESSIONS = {"asia": (0, 8), "london": (7, 16), "ny": (12, 21)}


def terrain_line(features, direction: str) -> str:
    try:
        return _terrain(features, direction) or ""
    except Exception:
        return ""


def _atr14(h, lo, cl):
    trs = []
    prev = cl[0]
    for i in range(len(h)):
        tr = max(h[i] - lo[i], abs(h[i] - prev), abs(lo[i] - prev))
        trs.append(tr)
        prev = cl[i]
    if len(trs) < 15:
        return None
    atr = sum(trs[1:15]) / 14.0
    for t in trs[15:]:
        atr = (atr * 13 + t) / 14.0
    return atr


def _pools(h, lo, dts):
    out = []
    n = len(h)
    for i in range(PIVOT, n - PIVOT):
        seg = range(i - PIVOT, i + PIVOT + 1)
        if all(h[i] >= h[k] for k in seg) and any(h[i] > h[k] for k in seg if k != i):
            out.append((i + PIVOT + 1, h[i], 1))
        if all(lo[i] <= lo[k] for k in seg) and any(lo[i] < lo[k] for k in seg if k != i):
            out.append((i + PIVOT + 1, lo[i], -1))
    for _name, (h0, h1) in SESSIONS.items():
        hi = lo_ = None
        prev_in = False
        for i in range(n):
            inside = h0 <= dts[i].hour < h1
            if inside:
                hi = h[i] if not prev_in else max(hi, h[i])
                lo_ = lo[i] if not prev_in else min(lo_, lo[i])
            elif prev_in and hi is not None:
                out.append((i, hi, 1))
                out.append((i, lo_, -1))
                hi = lo_ = None
            prev_in = inside
    for keyfn in (lambda d: d.date(),
                  lambda d: d.isocalendar()[:2]):
        cur = None
        hi = lo_ = None
        for i in range(n):
            k = keyfn(dts[i])
            if cur is None:
                cur, hi, lo_ = k, h[i], lo[i]
            elif k != cur:
                out.append((i, hi, 1))
                out.append((i, lo_, -1))
                cur, hi, lo_ = k, h[i], lo[i]
            else:
                hi = max(hi, h[i])
                lo_ = min(lo_, lo[i])
    return sorted(out)


def _terrain(df, direction: str) -> str:
    cols = {"high", "low", "close"}
    if direction not in ("UP", "DOWN") or not cols.issubset(df.columns):
        return ""
    d = df.tail(500)
    if len(d) < 60:
        return ""
    h = d["high"].astype(float).tolist()
    lo = d["low"].astype(float).tolist()
    cl = d["close"].astype(float).tolist()
    idx = d.index
    try:
        idx = idx.tz_convert("UTC")
    except (TypeError, AttributeError):
        pass
    dts = list(idx.to_pydatetime()) if hasattr(idx, "to_pydatetime") else list(idx)
    atr = _atr14(h, lo, cl)
    if not atr:
        return ""
    n = len(h)
    pools = [list(p) + [None] for p in _pools(h, lo, dts)]  # est,lvl,side,swept
    live: list = []
    pi = 0
    recent_sweeps: list = []       # (bar_i, side)
    for j in range(n):
        while pi < len(pools) and pools[pi][0] <= j:
            live.append(pools[pi])
            pi += 1
        for p in list(live):
            if (h[j] > p[1] if p[2] == 1 else lo[j] < p[1]):
                p[3] = j
                live.remove(p)
                recent_sweeps.append((j, p[2]))
    c = cl[-1]
    up = direction == "UP"
    above = [p[1] for p in live if p[1] > c]
    below = [p[1] for p in live if p[1] < c]
    ahead = ((min(above) - c) / atr if up and above else
             (c - max(below)) / atr if (not up) and below else None)
    behind = ((c - max(below)) / atr if up and below else
              (min(above) - c) / atr if (not up) and above else None)
    ctx = ""
    for (j, side) in reversed(recent_sweeps):
        if n - 1 - j <= LOOK_RAID_H:
            fade = (side == 1 and not up) or (side == -1 and up)
            ctx = "順獵取" if fade else "追突破⚠️"
            break
    parts = []
    if ctx:
        parts.append(ctx)
    if ahead is not None:
        parts.append(f"前方{'牆' if ahead <= WALL_ATR else '淨'} {ahead:.1f}ATR"
                     + ("⚠️" if ahead <= WALL_ATR else "✅"))
    if behind is not None:
        parts.append(f"背後{'支撐' if behind <= SUPPORT_ATR else '空'} "
                     f"{behind:.1f}ATR"
                     + ("✅" if behind <= SUPPORT_ATR else ""))
    if not parts:
        return ""
    return "\n\U0001f5fa 地形: " + " · ".join(parts)
