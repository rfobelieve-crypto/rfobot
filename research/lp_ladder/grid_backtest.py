# -*- coding: utf-8 -*-
"""Nested-martingale ladder run as a CEX GRID: OOS split, walk-forward, Monte Carlo.

Venue was settled in TODO 0.93 (2026-09-03): as an AMM LP the ladder is
negative in every configuration, as an exchange grid it clears, because a grid
keeps the bin width (58-140 bps) that an AMM hands to the arbitrageur. So this
file only backtests the grid reading.

RULES FROZEN BEFORE RUNNING (the spec has no exit rule; without one PnL is
undefined, so these are stated, not tuned):
  anchor      ladder is placed once at the window's first close, range
              [P0, P0*(1-drop)], m*n bins, equal in LOG price
  orders      limit BUY at each bin's low edge, limit SELL at its high edge
  fills       a bar fills a buy if low <= edge, a sell if high >= edge; when
              one bar does both, the BUY wins (no same-bar round trip) --
              conservative, it understates turnover
  fees        maker 2 bps per side (Bitget/OKX maker tier)
  below range no re-anchor, no stop: keep the inventory, stop buying (this is
              the martingale's own tail, and it must be visible)
  above range fully in cash, ladder stays where it is
  accounting  equity(t) = capital + realised(t) + unrealised(t), marked every
              hour; the reported MDD is on that equity path, not on price

WHAT EACH SECTION ANSWERS
  [OOS]   pick parameters on the first 60% of history, look once at the last
          40%. The gap between in-sample-best and its out-of-sample result is
          the selection cost -- the number people leave out.
  [WF]    walk-forward: re-pick parameters on a trailing window, trade the
          next quarter, roll. Compares against a fixed default and against
          hindsight-best, so "does re-picking help" gets an answer.
  [MC]    stationary block bootstrap of BTC hourly returns (block 48h) --
          3 drift settings: as-is, drift removed, and +-30%/yr. A grid lives
          on oscillation and dies in trend; the drift sweep is where it dies.
  [XA]    the same fixed parameters on the other 8 core symbols. Not a tuning
          knob, a different-asset check.

Run: python research/lp_ladder/grid_backtest.py
Out: research/results/lp_grid_backtest.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nested_martingale import (grid, nested_alloc, single_mf_alloc,  # noqa: E402
                               uniform_alloc)

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
OUT = ROOT / "research" / "results" / "lp_grid_backtest.json"
MAKER = 2.0 / 1e4        # per side
SYMS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load(sym):
    lo, hi, cl = [], [], []
    with open(CACHE / f"{sym}USDT_1h.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            lo.append(float(row["low"]))
            hi.append(float(row["high"]))
            cl.append(float(row["close"]))
    return np.array(lo), np.array(hi), np.array(cl)


def alloc_for(profile, T, N, r):
    """m*n = N; keep the spec's 5 big intervals when N allows."""
    m = 5 if N % 5 == 0 else 1
    n = N // m
    if profile == "nested":
        return nested_alloc(T, r, m, n)
    if profile == "uniform":
        return uniform_alloc(T, r, m, n)
    if profile == "single":
        return single_mf_alloc(T, r, m, n)
    raise ValueError(profile)


def run_grid(low, high, close, edges, alloc):
    """Vectorised per-bin state machine. Returns dict of window metrics."""
    T = float(alloc.sum())
    nb = len(alloc)
    lo_e, hi_e = edges[1:], edges[:-1]
    steps = len(close)
    # per-bin event streams, accumulated into two weight paths + realised path
    w_qty = np.zeros(steps + 1)      # sum(alloc_i / lo_i) while holding
    w_cost = np.zeros(steps + 1)     # sum(alloc_i) while holding
    realised = np.zeros(steps + 1)
    traversals = 0.0
    turn = 0.0
    for i in range(nb):
        b = low <= lo_e[i]
        s = high >= hi_e[i]
        s = s & ~b                                  # buy wins ties
        ev = np.zeros(steps, np.int8)
        ev[s] = 1
        ev[b] = -1
        idx = np.nonzero(ev)[0]
        if idx.size == 0:
            continue
        v = ev[idx]
        keep = np.ones(v.size, bool)
        keep[1:] = v[1:] != v[:-1]
        v, idx = v[keep], idx[keep]
        if v[0] == 1:                               # cannot sell before buying
            v, idx = v[1:], idx[1:]
        if v.size == 0:
            continue
        buys = idx[v == -1]
        sells = idx[v == 1]
        k = len(sells)
        traversals += k
        turn += k * alloc[i]
        # realised profit per completed traversal: bin width minus two fees
        gain = alloc[i] * (hi_e[i] / lo_e[i] - 1 - 2 * MAKER)
        np.add.at(realised, sells + 1, gain)
        # holding intervals: buys[j] .. sells[j] (last buy may stay open)
        np.add.at(w_qty, buys + 1, alloc[i] / lo_e[i])
        np.add.at(w_cost, buys + 1, alloc[i])
        if k:
            np.add.at(w_qty, sells[:k] + 1, -alloc[i] / lo_e[i])
            np.add.at(w_cost, sells[:k] + 1, -alloc[i])
    w_qty = np.cumsum(w_qty)[1:]
    w_cost = np.cumsum(w_cost)[1:]
    realised = np.cumsum(realised)[1:]
    unreal = w_qty * close - w_cost - w_cost * MAKER
    equity = T + realised + unreal
    peak = np.maximum.accumulate(equity)
    mdd = float((equity / peak - 1).min())
    return {"ret": float(equity[-1] / T - 1), "mdd": mdd,
            "traversals": float(traversals), "turnover": float(turn / T),
            "deployed_max": float(w_cost.max() / T),
            "held_end": float(w_cost[-1] / T)}


def windows(n_bars, bars, step):
    return range(0, max(n_bars - bars, 0), step)


def sweep(low, high, close, params, bars, step, T=100_000.0):
    """Mean/median metrics for every parameter combo over rolling windows."""
    out = {}
    for (prof, drop, N, r) in params:
        rets, mdds = [], []
        for s in windows(len(close), bars, step):
            p0 = close[s]
            edges = grid(p0, p0 * (1 - drop), N)
            w = slice(s + 1, s + 1 + bars)
            m = run_grid(low[w], high[w], close[w], edges, alloc_for(prof, T, N, r))
            rets.append(m["ret"]); mdds.append(m["mdd"])
        if not rets:
            continue
        rets = np.array(rets); mdds = np.array(mdds)
        yr = 365 * 24 / bars
        out[(prof, drop, N, r)] = {
            "n_win": len(rets), "mean_ann": float(rets.mean() * yr),
            "med_ann": float(np.median(rets) * yr),
            "p5_ann": float(np.percentile(rets, 5) * yr),
            "worst": float(rets.min()), "mdd_med": float(np.median(mdds)),
            "mdd_worst": float(mdds.min()),
            "loss_rate": float((rets < 0).mean()),
        }
    return out


def fmt(k, v):
    prof, drop, N, r = k
    return (f"  {prof:<8}{drop:>6.0%}{N:>5}{r:>6.2f} | "
            f"年化 mean {v['mean_ann']:+7.2%}  med {v['med_ann']:+7.2%}  "
            f"p5 {v['p5_ann']:+8.2%}  最差窗 {v['worst']:+7.2%}  "
            f"MDD中位 {v['mdd_med']:+6.2%} 最深 {v['mdd_worst']:+7.2%}  "
            f"虧損窗 {v['loss_rate']:.0%}")


def block_bootstrap(rets, length, block, rng):
    out = np.empty(length)
    i = 0
    while i < length:
        s = rng.integers(0, len(rets) - block)
        take = min(block, length - i)
        out[i:i + take] = rets[s:s + take]
        i += take
    return out


def synth_path(rets, length, block, rng, drift_ann=None, demean=False):
    r = block_bootstrap(rets, length, block, rng)
    if demean:
        r = r - r.mean()
    if drift_ann is not None:
        r = r - r.mean() + math.log(1 + drift_ann) / (365 * 24)
    px = 100_000 * np.exp(np.cumsum(r))
    return px


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=90 * 24)
    ap.add_argument("--step", type=int, default=14 * 24)
    ap.add_argument("--mc", type=int, default=400)
    a = ap.parse_args()

    low, high, close = load("BTC")
    print("=" * 96)
    print(f"  GRID BACKTEST — BTC 1h, {len(close)} bars, "
          f"{a.bars//24}d windows, maker {MAKER*1e4:.0f}bps/side")
    print("=" * 96)

    PARAMS = [(p, d, N, 1.5)
              for p in ("nested", "uniform", "single")
              for d in (0.15, 0.25, 0.35, 0.50)
              for N in (30, 50)]
    res = {}

    # ── OOS split ────────────────────────────────────────────────────
    cut = int(len(close) * 0.6)
    isl, ish, isc = low[:cut], high[:cut], close[:cut]
    osl, osh, osc = low[cut:], high[cut:], close[cut:]
    print(f"\n[OOS] 前 60%（{cut} 根）選參數，後 40% 只看一次")
    IS = sweep(isl, ish, isc, PARAMS, a.bars, a.step)
    OS = sweep(osl, osh, osc, PARAMS, a.bars, a.step)
    best_is = max(IS, key=lambda k: IS[k]["mean_ann"])
    best_os = max(OS, key=lambda k: OS[k]["mean_ann"])
    print("  IS 最佳:"); print(fmt(best_is, IS[best_is]))
    print("  同一組參數在 OOS:"); print(fmt(best_is, OS[best_is]))
    print("  OOS 事後最佳（不可用，只為量選擇成本）:"); print(fmt(best_os, OS[best_os]))
    print(f"  選擇成本 = {OS[best_is]['mean_ann'] - OS[best_os]['mean_ann']:+.2%} "
          f"年化；IS→OOS 衰減 = "
          f"{OS[best_is]['mean_ann'] - IS[best_is]['mean_ann']:+.2%}")
    print("\n  OOS 全表（每一組都報，不挑）：")
    for k in sorted(OS, key=lambda k: -OS[k]["mean_ann"]):
        print(fmt(k, OS[k]))
    res["oos"] = {"best_is": list(best_is), "is": IS[best_is], "oos": OS[best_is],
                  "oos_hindsight_best": [list(best_os), OS[best_os]],
                  "table": {str(k): v for k, v in OS.items()}}

    # ── walk-forward ─────────────────────────────────────────────────
    print("\n[WF] 走勢前推：用前 180 天選參數 → 交易後 90 天 → 滾動")
    train, trade = 180 * 24, 90 * 24
    picks, wf, fixed, hind = [], [], [], []
    FIXED = ("nested", 0.25, 50, 1.5)
    s = 0
    while s + train + trade <= len(close):
        tr = slice(s, s + train)
        sub = sweep(low[tr], high[tr], close[tr], PARAMS, a.bars, a.step)
        if not sub:
            break
        pick = max(sub, key=lambda k: sub[k]["mean_ann"])
        te = slice(s + train, s + train + trade)
        p0 = close[s + train]
        results = {}
        for k in set([pick, FIXED] + PARAMS):
            prof, drop, N, r = k
            e = grid(p0, p0 * (1 - drop), N)
            results[k] = run_grid(low[te], high[te], close[te], e,
                                  alloc_for(prof, 100_000.0, N, r))["ret"]
        picks.append(pick)
        wf.append(results[pick])
        fixed.append(results[FIXED])
        hind.append(max(results.values()))
        s += trade
    yr = 365 / 90
    if wf:
        print(f"  {len(wf)} 段；每段 90 天")
        print(f"  WF 選參數    年化 {np.mean(wf)*yr:+.2%}  段勝率 "
              f"{np.mean(np.array(wf) > 0):.0%}  最差段 {min(wf):+.2%}")
        print(f"  固定參數     年化 {np.mean(fixed)*yr:+.2%}  段勝率 "
              f"{np.mean(np.array(fixed) > 0):.0%}  最差段 {min(fixed):+.2%}   "
              f"({FIXED[0]} {FIXED[1]:.0%} N={FIXED[2]})")
        print(f"  事後最佳     年化 {np.mean(hind)*yr:+.2%}（不可得，天花板）")
        from collections import Counter
        print("  每段選到的參數:", Counter(str(p) for p in picks).most_common())
        res["wf"] = {"n_seg": len(wf), "wf_ann": float(np.mean(wf) * yr),
                     "fixed_ann": float(np.mean(fixed) * yr),
                     "hindsight_ann": float(np.mean(hind) * yr),
                     "wf_worst": float(min(wf)), "fixed_worst": float(min(fixed))}

    # ── Monte Carlo ──────────────────────────────────────────────────
    print("\n[MC] 區塊自助法合成路徑（block 48h），固定參數 "
          f"{FIXED[0]} {FIXED[1]:.0%} N={FIXED[2]}，每組 {a.mc} 條 90 天路徑")
    logret = np.diff(np.log(close))
    rng = np.random.default_rng(20260903)
    prof, drop, N, r = FIXED
    al = alloc_for(prof, 100_000.0, N, r)
    scen = [("原樣（含 BTC 歷史漂移）", dict()),
            ("去漂移", dict(demean=True)),
            ("多頭 +30%/年", dict(drift_ann=0.30)),
            ("空頭 −30%/年", dict(drift_ann=-0.30)),
            ("空頭 −60%/年", dict(drift_ann=-0.60))]
    res["mc"] = {}
    print(f"  {'情境':<18}{'年化中位':>10}{'年化平均':>10}{'p5':>10}"
          f"{'p95':>10}{'虧損機率':>10}{'MDD中位':>10}{'MDD p95':>10}")
    for name, kw in scen:
        rets, mdds = [], []
        for _ in range(a.mc):
            px = synth_path(logret, a.bars, 48, rng, **kw)
            # synthetic bars: use close as low/high (no intrabar range) --
            # conservative for a grid, it can only reduce fills
            m = run_grid(px, px, px, grid(px[0], px[0] * (1 - drop), N), al)
            rets.append(m["ret"]); mdds.append(m["mdd"])
        rets = np.array(rets) * (365 * 24 / a.bars)
        mdds = np.array(mdds)
        res["mc"][name] = {"med": float(np.median(rets)), "mean": float(rets.mean()),
                           "p5": float(np.percentile(rets, 5)),
                           "p95": float(np.percentile(rets, 95)),
                           "loss": float((rets < 0).mean()),
                           "mdd_med": float(np.median(mdds)),
                           "mdd_p95": float(np.percentile(mdds, 5))}
        v = res["mc"][name]
        print(f"  {name:<18}{v['med']:>+10.2%}{v['mean']:>+10.2%}{v['p5']:>+10.2%}"
              f"{v['p95']:>+10.2%}{v['loss']:>10.0%}{v['mdd_med']:>+10.2%}"
              f"{v['mdd_p95']:>+10.2%}")

    # ── cross-asset ──────────────────────────────────────────────────
    print(f"\n[XA] 同一組固定參數換 8 個幣（不調參）")
    res["xa"] = {}
    for sym in SYMS:
        try:
            l2, h2, c2 = load(sym)
        except FileNotFoundError:
            continue
        sw = sweep(l2, h2, c2, [FIXED], a.bars, a.step)
        v = list(sw.values())[0]
        res["xa"][sym] = v
        print(f"  {sym:<6}" + fmt(FIXED, v).split("|", 1)[1])

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
