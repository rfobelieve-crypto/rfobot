# -*- coding: utf-8 -*-
"""Sweep-failure — realistic bps-cost re-score + FORWARD progress tracker (Gate F).

Why this exists (2026-07-28 audit):
  The migrated sweep_core charged slippage with a FAVORABLE entry sign, which
  cancelled the adverse exit leg — the README's headline (pool exp +0.062R,
  t=+8.27, "cost 0.05 ATR/side included") was effectively a ZERO-COST result
  (proven: the SLIP=0.05 run beat SLIP=0). Sign fixed in sweep_core the same
  day. This script is the corrected accounting plus the pre-registered
  forward gate.

Cost model (categorical, fixed BEFORE forward accrual — not tuned):
  Charged in PRICE bps per leg, converted per trade into R units through that
  trade's own risk (DIS x ATR14 at the sweep bar):
      cost_R = sum(leg_bps)/1e4 * lvl / (DIS * A)
  so low-ATR%% symbols (BTC, BNB) honestly bear ~2x the relative cost — no
  uniform-ATR flattery, no basket cherry-picking.

  Scenario A "target execution":
      entry stop-market  taker 5 + slip 2 = 7 bps
      time-exit          worked limit, maker 2 + miss 1 = 3 bps
      disaster stop-out  taker 5 + fast-tape slip 5 = 10 bps
  Scenario B "all-taker conservative":
      entry 7 / time-exit 6 / stop-out 10 bps

Forward gate — Gate F (pre-registered 2026-07-28, do not move):
  FREEZE = 2026-07-28 (rules committed to flow_system main today; git hash is
  the proof). Trades with fill_ts >= FREEZE are forward. PASS requires, on
  POOLED forward trades under scenario A:
      n >= 1400
      AND bootstrap 95%% CI-low of mean R_net > 0
      AND >= 6/9 symbols with positive forward sum
  (n=1400 from the corrected history: mu~=0.033R, sd~=0.63R ->
  (1.96*sd/mu)^2 ~= 1400; ~6 months at the historical ~230 pooled trades/mo.)
  Checkpoints before n=1400 are directional only — no early pass.
  The 2026-07-11 sandbox freeze (per README, git-unprovable) is reported as
  "quasi-forward", labeled separately, never merged into the gate.

Usage:
    python research/sweep_failure/fetch_klines.py   # refresh data first
    python research/sweep_failure/sweep_forward.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["SLIP"] = "0"          # gross engine; costs are applied here
import sweep_core as SC            # noqa: E402  (env must precede import)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import math                        # noqa: E402
import random                      # noqa: E402

SYMS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
CACHE = Path(__file__).parent / ".cache"

FREEZE_TS = int(datetime(2026, 7, 28, tzinfo=timezone.utc).timestamp())
QUASI_TS = int(datetime(2026, 7, 11, tzinfo=timezone.utc).timestamp())
GATE_N = 1400

SCEN = {
    "A": {"entry": 7.0, "texit": 3.0, "sexit": 10.0},
    "B": {"entry": 7.0, "texit": 6.0, "sexit": 10.0},
}


def rescore(trades, scen):
    """gross (fill_ts, exit_ts, R, lvl, A, stopped) -> net R list (same order)."""
    s = SCEN[scen]
    out = []
    for fill_ts, exit_ts, r, lvl, atr, stopped in trades:
        legs = s["entry"] + (s["sexit"] if stopped else s["texit"])
        cost_r = legs / 1e4 * lvl / (SC.DIS * atr)
        out.append((fill_ts, r - cost_r))
    return out


def stats(rs):
    n = len(rs)
    if n == 0:
        return None
    m = sum(rs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in rs) / (n - 1)) if n > 1 else 0.0
    t = m / (sd / math.sqrt(n)) if sd > 0 else 0.0
    wins = sum(1 for x in rs if x > 0)
    losses = [x for x in rs if x <= 0]
    pf = (sum(x for x in rs if x > 0) / -sum(losses)) if losses and sum(losses) < 0 else float("inf")
    return dict(n=n, mean=m, sd=sd, t=t, wr=100.0 * wins / n, pf=pf)


def boot_ci(rs, nb=4000, seed=7):
    rng = random.Random(seed)
    n = len(rs)
    if n == 0:
        return (float("nan"),) * 2
    means = []
    for _ in range(nb):
        means.append(sum(rs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * nb)], means[int(0.975 * nb)]


def main() -> int:
    per_sym = {}
    for s in SYMS:
        p = CACHE / f"{s}USDT_1h.csv"
        if not p.exists():
            print(f"{s}: missing {p} — run fetch_klines.py first")
            return 1
        per_sym[s] = SC.backtest_symbol(SC.load_csv(str(p)))

    print("=" * 78)
    print("  SWEEP-FAILURE — corrected bps-cost accounting  (gross engine, SLIP=0)")
    print("=" * 78)
    for scen in ("A", "B"):
        pool = []
        pos = 0
        print(f"\n  Scenario {scen}  (entry {SCEN[scen]['entry']:.0f} / "
              f"t-exit {SCEN[scen]['texit']:.0f} / stop {SCEN[scen]['sexit']:.0f} bps)")
        print(f"  {'sym':<6}{'n':>6}{'meanR':>9}{'WR%':>6}{'PF':>6}{'t':>7}   halves(meanR)")
        for s in SYMS:
            rs = [r for _, r in rescore(per_sym[s], scen)]
            st = stats(rs)
            half = st["n"] // 2
            e1 = sum(rs[:half]) / max(half, 1)
            e2 = sum(rs[half:]) / max(st["n"] - half, 1)
            if sum(rs) > 0:
                pos += 1
            pool += rs
            print(f"  {s:<6}{st['n']:>6}{st['mean']:>+9.4f}{st['wr']:>5.0f}%"
                  f"{st['pf']:>6.2f}{st['t']:>+7.2f}   {e1:+.4f}/{e2:+.4f}")
        st = stats(pool)
        lo, hi = boot_ci(pool)
        print(f"  {'pool':<6}{st['n']:>6}{st['mean']:>+9.4f}{st['wr']:>5.0f}%"
              f"{st['pf']:>6.2f}{st['t']:>+7.2f}   CI95[{lo:+.4f},{hi:+.4f}]"
              f"   positive {pos}/9")

    # ── forward sections ──────────────────────────────────────────────
    for label, ts0 in (("FORWARD (freeze 2026-07-28, THE gate)", FREEZE_TS),
                       ("quasi-forward (sandbox freeze 2026-07-11, informational)", QUASI_TS)):
        print("\n" + "=" * 78)
        print(f"  {label}")
        print("=" * 78)
        pool = []
        pos = 0
        for s in SYMS:
            rs = [r for ts, r in rescore(per_sym[s], "A") if ts >= ts0]
            if rs and sum(rs) > 0:
                pos += 1
            pool += rs
            if rs:
                print(f"  {s:<6} n={len(rs):>4}  sumR={sum(rs):+8.3f}  meanR={sum(rs)/len(rs):+8.4f}")
        st = stats(pool)
        if st is None:
            print("  (no trades yet)")
            continue
        lo, hi = boot_ci(pool)
        print(f"  pool   n={st['n']:>4}  meanR={st['mean']:+8.4f}  WR={st['wr']:.0f}%  "
              f"CI95[{lo:+.4f},{hi:+.4f}]  positive {pos}/9")
        if ts0 == FREEZE_TS:
            print(f"\n  Gate F progress: n={st['n']}/{GATE_N}"
                  f"  |  CI-low>0: {'YES' if lo > 0 else 'no'}"
                  f"  |  >=6/9 positive: {'YES' if pos >= 6 else 'no'}"
                  f"  ->  {'PASS' if (st['n'] >= GATE_N and lo > 0 and pos >= 6) else 'accumulating'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
