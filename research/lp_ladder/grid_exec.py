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
             stop_delay_h=0, gate=None, gate_scale=0.0,
             maker=None, stop_cost=None, fill_pen=0.0, fund_hourly=0.0,
             n_series=None, size_series=None, inv_skew=0.0):
    """One continuous run. Returns metrics + the hourly equity curve.

    stop_delay_h: after a hard stop, stay FLAT (no ladder at all) for this
    many hours before re-anchoring at the then-current price. 0 = the
    2026-09-03 七/八 behaviour (stop then immediately stand back in), which
    TODO 0.93 flagged as an open question. Time re-anchoring is suspended
    while the pause runs -- the pause is the re-entry rule, not a second one.

    gate: optional per-bar bool array. False = do not OPEN new rungs on that
    bar (sells, stops and re-anchors are untouched -- a filter that also
    blocked exits would be a different strategy, not a filter). gate_scale
    lets a blocked bar still buy a fraction (0.0 = fully off, 0.5 = half
    size). The caller is responsible for the array being causal; this
    function does not shift it.
    """
    # ── venue realism (2026-09-05, §0.93 十二) — all default to the frozen constants
    #    maker       fee per grid fill (spot ~10 bps, perp ~2 bps)
    #    stop_cost   taker + fast-tape slippage + half-spread + depth impact
    #    fill_pen    a resting order fills only when the bar trades THROUGH the
    #                level by this fraction (≈ one spread) — queue-position proxy
    #    fund_hourly funding charged per hour on held inventory value (perp only)
    maker = MAKER if maker is None else maker
    stop_cost = STOP_COST if stop_cost is None else stop_cost
    m = 5 if N % 5 == 0 else 1
    # ── 自適應格距（2026-09-05，§1.18h）────────────────────────────────
    # n_series[i] = 在 bar i 重錨時要用幾格（格數多 = 格距窄）。None = 沿用
    # 固定 N，行為與先前完全相同（回歸測試釘住）。size_series[i] 同理縮放
    # 每次重錨後投入的資金比例（庫存上限與 σ̂ 成反比的實作）。
    # **兩者只在重錨時生效**——格距不能在持倉中間改，那等於把既有掛單全撤
    # 重掛，成本模型接不住那個動作。
    def _alloc_for(n_):
        mm = 5 if n_ % 5 == 0 else 1
        return (nested_alloc(1.0, r, mm, n_ // mm) if profile == "nested"
                else uniform_alloc(1.0, r, mm, n_ // mm))

    alloc = _alloc_for(N)
    size_mult = 1.0
    # ── 反應式層三（2026-09-05）：庫存就是路徑形態的實現 ──────────────
    # 事前分不出震盪或單向（§1.18f），但**單向行情的定義就是庫存單邊累積**，
    # 那個不用預測、會直接觀測到。A-S 的偏移項 r = S − qγσ²(T−t) 在網格上的
    # 對應動作是：庫存往一側堆積時，接貨速度按 q 遞減。
    # 代價是必然落後——你一定先吃到一段虧損才觸發——但它不需要一個不存在的
    # 預測能力。inv_skew = γ，0 = 關閉（預設，回歸行為不變）。
    cash = 1.0                      # everything in units of starting capital
    qty = np.zeros(N)               # base units held per bin
    cost = np.zeros(N)              # quote spent per bin (for MTM)
    edges = grid(close[0], close[0] * (1 - drop), N)
    lo_e, hi_e = edges[1:], edges[:-1]
    width = float(hi_e[0] / lo_e[0] - 1)
    anchor_i = 0
    paused_until = -1               # bar index at which a post-stop pause ends
    eq = np.empty(len(close))
    n_anchor = n_stop = n_fill = n_gated_off = 0

    def replace(px, i):
        nonlocal edges, lo_e, hi_e, anchor_i, n_anchor, N, alloc, qty, cost, size_mult
        if n_series is not None:
            n_new = int(n_series[min(i, len(n_series) - 1)])
            if n_new != N:
                N = n_new
                alloc = _alloc_for(N)
                qty = np.zeros(N)      # 重錨時本來就是空手（呼叫端保證），格數改變安全
                cost = np.zeros(N)
        if size_series is not None:
            size_mult = float(size_series[min(i, len(size_series) - 1)])
        edges = grid(px, px * (1 - drop), N)
        lo_e, hi_e = edges[1:], edges[:-1]
        anchor_i = i
        n_anchor += 1

    for i in range(len(close)):
        if i < paused_until:
            # flat by construction (the stop emptied every bin); no ladder
            eq[i] = cash
            continue
        if paused_until >= 0 and i == paused_until:
            replace(close[i], i)
            paused_until = -1
        held = qty > 0
        # sells first: a bar that reaches the top edge of a held bin
        s = held & (high[i] >= hi_e * (1 + fill_pen))
        if s.any():
            proceeds = (qty[s] * hi_e[s]).sum()
            cash += proceeds * (1 - maker)
            qty[s] = 0.0
            cost[s] = 0.0
        # buys: a bar that reaches the low edge of an empty bin
        b = (qty <= 0) & (low[i] <= lo_e * (1 - fill_pen))
        scale = 1.0 if (gate is None or gate[i]) else gate_scale
        if inv_skew > 0.0:
            eq_now = cash + (qty * close[i]).sum()
            q_frac = (qty * close[i]).sum() / eq_now if eq_now > 0 else 0.0
            scale *= max(0.0, 1.0 - inv_skew * q_frac)
        if b.any() and scale > 0:
            spend = alloc[b].sum() * scale * size_mult
            if spend <= cash + 1e-12:
                qty[b] = alloc[b] * scale * size_mult / lo_e[b] * (1 - maker)
                cost[b] = alloc[b] * scale * size_mult
                cash -= spend
                n_fill += int(b.sum())
                n_gated_off += 0
        px = close[i]
        if fund_hourly and (qty > 0).any():
            cash -= (qty * px).sum() * fund_hourly
        eq[i] = cash + (qty * px).sum()
        # ── policies ────────────────────────────────────────────────
        if stop == "hard" and (qty > 0).any() and px < lo_e[-1] * (1 - stop_buf):
            cash += (qty * px).sum() * (1 - stop_cost)
            qty[:] = 0.0
            cost[:] = 0.0
            n_stop += 1
            if stop_delay_h > 0:
                paused_until = i + int(stop_delay_h)
            else:
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
