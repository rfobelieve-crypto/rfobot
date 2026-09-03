# -*- coding: utf-8 -*-
"""Nested-martingale grid: the two rules the spec never had, tested.

TODO 0.93 settled the venue (exchange grid, not AMM) and showed the shape is
sound, but the backtest ran on a deliberately naive pair of rules because the
spec has neither: the ladder was anchored once and never moved, and price
falling out of the bottom meant "keep the inventory, no stop". Those two are
the whole execution design, and the cross-asset table said what the second one
costs (alt worst windows -29%..-52%).

This file runs the ladder CONTINUOUSLY over the full history (not rolling
windows) so an equity curve exists, and compares policies on the same path:

  re-anchor   none     place once, never move (the TODO 0.93 baseline)
              above    when price leaves the TOP (fully in cash, grid idle),
                       re-place the ladder around the current price
              time     re-place every N days, but ONLY when flat -- never
                       while holding, because re-anchoring with inventory is
                       just averaging down with extra steps
  stop        none     hold everything below the range (the martingale tail)
              hard     when price closes below bottom*(1-buf), sell the whole
                       inventory at market and re-place around the new price

NEVER TESTED, ON PURPOSE: re-anchoring DOWNWARD while holding inventory. That
is the same move that turned a $1218 account into $16 on 2026-07-27 -- it adds
capital to a losing position and calls it a new grid.

Costs: maker 2 bps per side on grid fills; a hard stop pays taker 5 bps plus
5 bps slippage (it fires in a fast tape, that is the point).

Run: python research/lp_ladder/grid_exec.py
Out: research/results/lp_grid_exec.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nested_martingale import grid, nested_alloc, uniform_alloc  # noqa: E402

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
OUT = ROOT / "research" / "results" / "lp_grid_exec.json"
MAKER = 2.0 / 1e4
STOP_COST = 10.0 / 1e4        # taker + fast-tape slippage
SYMS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load(sym):
    lo, hi, cl = [], [], []
    with open(CACHE / f"{sym}USDT_1h.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            lo.append(float(row["low"])); hi.append(float(row["high"]))
            cl.append(float(row["close"]))
    return np.array(lo), np.array(hi), np.array(cl)


def simulate(low, high, close, *, drop=0.25, N=30, profile="nested", r=1.5,
             reanchor="none", reanchor_days=90, stop=None, stop_buf=0.03,
             gate=None, gate_scale=0.0):
    """One continuous run. Returns metrics + the hourly equity curve.

    gate: optional per-bar bool array. False = do not OPEN new rungs on that
    bar (sells, stops and re-anchors are untouched -- a filter that also
    blocked exits would be a different strategy, not a filter). gate_scale
    lets a blocked bar still buy a fraction (0.0 = fully off, 0.5 = half
    size). The caller is responsible for the array being causal; this
    function does not shift it.
    """
    m = 5 if N % 5 == 0 else 1
    alloc = (nested_alloc(1.0, r, m, N // m) if profile == "nested"
             else uniform_alloc(1.0, r, m, N // m))
    cash = 1.0                      # everything in units of starting capital
    qty = np.zeros(N)               # base units held per bin
    cost = np.zeros(N)              # quote spent per bin (for MTM)
    edges = grid(close[0], close[0] * (1 - drop), N)
    lo_e, hi_e = edges[1:], edges[:-1]
    width = float(hi_e[0] / lo_e[0] - 1)
    anchor_i = 0
    eq = np.empty(len(close))
    n_anchor = n_stop = n_fill = n_gated_off = 0

    def replace(px, i):
        nonlocal edges, lo_e, hi_e, anchor_i, n_anchor
        edges = grid(px, px * (1 - drop), N)
        lo_e, hi_e = edges[1:], edges[:-1]
        anchor_i = i
        n_anchor += 1

    for i in range(len(close)):
        held = qty > 0
        # sells first: a bar that reaches the top edge of a held bin
        s = held & (high[i] >= hi_e)
        if s.any():
            proceeds = (qty[s] * hi_e[s]).sum()
            cash += proceeds * (1 - MAKER)
            qty[s] = 0.0
            cost[s] = 0.0
        # buys: a bar that reaches the low edge of an empty bin
        b = (qty <= 0) & (low[i] <= lo_e)
        scale = 1.0 if (gate is None or gate[i]) else gate_scale
        if b.any() and scale > 0:
            spend = alloc[b].sum() * scale
            if spend <= cash + 1e-12:
                qty[b] = alloc[b] * scale / lo_e[b] * (1 - MAKER)
                cost[b] = alloc[b] * scale
                cash -= spend
                n_fill += int(b.sum())
                n_gated_off += 0
        px = close[i]
        eq[i] = cash + (qty * px).sum()
        # ── policies ────────────────────────────────────────────────
        if stop == "hard" and (qty > 0).any() and px < lo_e[-1] * (1 - stop_buf):
            cash += (qty * px).sum() * (1 - STOP_COST)
            qty[:] = 0.0
            cost[:] = 0.0
            n_stop += 1
            replace(px, i)
            eq[i] = cash
            continue
        flat = not (qty > 0).any()
        if reanchor == "above" and flat and px > hi_e[0]:
            replace(px, i)
        elif reanchor == "time" and flat and (i - anchor_i) >= reanchor_days * 24:
            replace(px, i)

    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    yrs = len(close) / (365 * 24)
    w90 = min((eq[j + 90 * 24] / eq[j] - 1)
              for j in range(0, len(eq) - 90 * 24, 24)) if len(eq) > 90 * 24 else 0.0
    return {"final": float(eq[-1]), "cagr": float(eq[-1] ** (1 / yrs) - 1),
            "mdd": float(dd.min()), "worst90d": float(w90),
            "gated_frac": float(0.0 if gate is None else 1 - gate.mean()),
            "anchors": n_anchor, "stops": n_stop, "fills": n_fill,
            "end_deployed": float((qty * close[-1]).sum() / eq[-1]),
            "width_pct": width}, eq


POLICIES = [
    ("靜態·照抱（0.93 基準）", dict(reanchor="none", stop=None)),
    ("上緣重錨", dict(reanchor="above", stop=None)),
    ("上緣重錨＋硬停損", dict(reanchor="above", stop="hard")),
    ("只有硬停損", dict(reanchor="none", stop="hard")),
    ("每 90 天重錨（僅空倉時）", dict(reanchor="time", stop=None)),
    ("每90天重錨＋硬停損", dict(reanchor="time", stop="hard")),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", type=float, default=0.25)
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--profile", default="nested")
    a = ap.parse_args()

    res = {"params": vars(a), "btc": {}, "cross": {}}
    low, high, close = load("BTC")
    yrs = len(close) / (365 * 24)
    print("=" * 92)
    print(f"  GRID EXECUTION — BTC 1h, {len(close)} bars ({yrs:.2f} 年), "
          f"區間 −{a.drop:.0%}, {a.bins} 格, {a.profile}, maker {MAKER*1e4:.0f}bps")
    print(f"  買進並持有同期： {close[-1]/close[0]-1:+.1%}")
    print("=" * 92)
    print(f"  {'政策':<26}{'總報酬':>9}{'年化':>9}{'MDD':>9}"
          f"{'最差90天':>10}{'重錨':>6}{'停損':>6}{'期末持倉':>9}")
    for name, kw in POLICIES:
        m, _ = simulate(low, high, close, drop=a.drop, N=a.bins,
                        profile=a.profile, **kw)
        res["btc"][name] = m
        print(f"  {name:<26}{m['final']-1:>+9.1%}{m['cagr']:>+9.2%}"
              f"{m['mdd']:>+9.1%}{m['worst90d']:>+10.1%}"
              f"{m['anchors']:>6}{m['stops']:>6}{m['end_deployed']:>9.0%}")

    best = ("每90天重錨＋硬停損", dict(reanchor="time", stop="hard"))
    print(f"\n  同一組政策（{best[0]}）換 9 個幣，不調參：")
    print(f"  {'幣':<6}{'總報酬':>9}{'年化':>9}{'MDD':>9}{'最差90天':>10}"
          f"{'重錨':>6}{'停損':>6}   {'買進持有':>9}")
    for sym in SYMS:
        try:
            l2, h2, c2 = load(sym)
        except FileNotFoundError:
            continue
        m, _ = simulate(l2, h2, c2, drop=a.drop, N=a.bins,
                        profile=a.profile, **best[1])
        res["cross"][sym] = dict(m, buyhold=float(c2[-1] / c2[0] - 1))
        print(f"  {sym:<6}{m['final']-1:>+9.1%}{m['cagr']:>+9.2%}"
              f"{m['mdd']:>+9.1%}{m['worst90d']:>+10.1%}{m['anchors']:>6}"
              f"{m['stops']:>6}   {c2[-1]/c2[0]-1:>+9.1%}")

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
