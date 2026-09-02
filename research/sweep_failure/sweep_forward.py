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
      AND DAY-CLUSTERED bootstrap 95% CI-low of mean R_net > 0
      AND >= 6/9 symbols with positive forward sum
  AMENDED 2026-07-28 (same day, BEFORE any forward trade accrued — a
  tightening, never to be loosened): cluster_stats.py measured VIF=2.95 on
  the pooled mean (nine correlated symbols take the same shock), so the iid
  CI overstates precision ~sqrt(3)x; the gate now uses the day-clustered CI.
  Honest runway at the historical mean (+0.0255R, clustered se): ~5-7k
  forward trades (~2 years) unless forward runs hotter. The counterweight is
  the pre-registered early-stop: if after 3 months the combined
  quasi-forward+forward mean is significantly NEGATIVE, kill the line
  without waiting. Checkpoints before the gate are directional only.
  The 2026-07-11 sandbox freeze (per README, git-unprovable) is reported as
  "quasi-forward", labeled separately, never merged into the gate.

VARIANT B — shallow-pierce filter (pre-registered 2026-07-29, threshold
  fixed BEFORE any forward trade of its own accrued):
      take the signal ONLY when the sweep bar pierced the level by
      <= 0.25 ATR (pierce_atr, computed at the sweep bar, strictly before
      the fill).
  Discovered in-sample by winner_anatomy.py, which asked why the top 1% of
  trades carry ~69% of total R. It survived every robustness check that was
  run: monotonic across pre-declared terciles (+0.079 / +0.021 / -0.006),
  consistent in core9 AND added20 separately, first half AND second half
  separately, 11/11 quarters positive, no cliff at the threshold (the whole
  0.10-1.00 ATR sweep is positive and decays smoothly to the unfiltered
  mean), and NOT a mechanical stop-distance artifact (terciles cut at
  0.23/0.53 ATR against a 3.5 ATR stop; only 0.1% of trades pierce past the
  stop; ATR% is identical across buckets; the stop-rate gap explains <half
  the effect). In-sample it holds 84% of total profit in 33% of the trades,
  raising SR/trade 0.050 -> 0.128 while cross-coin VIF FALLS 6.85 -> 2.69
  (shallow pokes are idiosyncratic; deep pierces are market-wide shocks),
  which takes the deflated-Sharpe MinTRL from ~7900 effective observations
  (unreachable) to ~300 (already exceeded).
  AMENDED the same day, hours after registration, with 8 forward signals
  accrued (i.e. before any meaningful forward evidence): the liquidity SOURCE
  widens from swing pivots alone to all four pool types — swing, session
  extremes (Asia/London/NY), PDH/PDL, PWH/PWL. Prompted by the user's
  critique that defining liquidity by swing pivot alone is one arbitrary
  choice among several, and tested in level_types.py:
      swing    +0.0320 all / +0.0786 shallow (n 22222 / 8053)
      pdh_pdl  +0.0368       / +0.0829       (n 26224 / 10041)
      pwh_pwl  +0.0413       / +0.0953       (n  4580 /  1334)
      session  +0.0212       / +0.0533       (n 46253 / 20373)
  All four were tested and all four are reported; none was dropped (session
  is the weakest and is kept). Time-defined levels cannot leak the outcome
  the way a pivot's NEIGHBOURHOOD can — which is exactly how the equal-levels
  idea died the same afternoon (see below). The pierce filter roughly doubling
  the mean on every level type INDEPENDENTLY is the reason to trust the
  widening: it generalises rather than fitting swing pivots.
  Combined and filtered: n=39801, mean +0.0673, VIF 7.70, 1327 trades/month;
  core9 +0.0719 vs added20 +0.0653, halves +0.0667 / +0.0680, 11/11 quarters
  positive (worst +0.0351). The clustered CI clears zero at n~2488, i.e.
  ~1.9 months rather than ~8.
  This does NOT loosen the statistical bar — the gate arithmetic is untouched;
  it broadens the test surface.

  RETRACTED the same day: an "equal levels" density filter (count of same-side
  pivots within a tolerance) appeared to add a large independent effect, but
  the count was taken over ALL pivots including ones confirmed AFTER the
  sweep. Recomputed causally it adds nothing (+0.0292 alone, BELOW the
  +0.0320 baseline). Levels that price kept respecting accumulate pivots, and
  "price came back" is the label — so the non-causal count was reading the
  outcome. Not registered, not used.

  IT IS STILL IN-SAMPLE. Variant A (the frozen unfiltered rules on swing
  pivots) remains THE Gate F track, unchanged and unretrofitted. Variant B
  accrues its own forward record from 2026-07-29 under the same gate
  arithmetic, and only a forward pass promotes it.

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

import json                        # noqa: E402
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
    for fill_ts, exit_ts, r, lvl, atr, stopped, _pierce, *_ in trades:
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


def boot_ci_clustered(pairs, nb=4000, seed=7):
    """Day-clustered bootstrap CI of mean R. pairs = [(fill_ts, r)].
    Gate F's CI (2026-07-28 amendment): resample calendar days, not trades —
    nine correlated symbols share the same market shock within a day."""
    from collections import defaultdict
    from datetime import datetime, timezone
    byd = defaultdict(list)
    for ts, r in pairs:
        byd[datetime.fromtimestamp(ts, tz=timezone.utc).date()].append(r)
    days = list(byd.values())
    if not days:
        return (float("nan"),) * 2
    rng = random.Random(seed)
    means = []
    for _ in range(nb):
        acc, cnt = 0.0, 0
        for _ in range(len(days)):
            g = days[rng.randrange(len(days))]
            acc += sum(g)
            cnt += len(g)
        means.append(acc / cnt)
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
        pairs = []
        pos = 0
        for s in SYMS:
            sp = [(ts, r) for ts, r in rescore(per_sym[s], "A") if ts >= ts0]
            rs = [r for _, r in sp]
            if rs and sum(rs) > 0:
                pos += 1
            pool += rs
            pairs += sp
            if rs:
                print(f"  {s:<6} n={len(rs):>4}  sumR={sum(rs):+8.3f}  meanR={sum(rs)/len(rs):+8.4f}")
        st = stats(pool)
        if st is None:
            print("  (no trades yet)")
            continue
        lo, hi = boot_ci_clustered(pairs)
        print(f"  pool   n={st['n']:>4}  meanR={st['mean']:+8.4f}  WR={st['wr']:.0f}%  "
              f"clustered-CI95[{lo:+.4f},{hi:+.4f}]  positive {pos}/9")
        if ts0 == FREEZE_TS:
            at_floor = st['n'] >= GATE_N
            status = ('PASS' if (at_floor and lo > 0 and pos >= 6)
                      else 'FAIL' if at_floor else 'accumulating')
            print(f"\n  Gate F progress: n={st['n']}/{GATE_N}"
                  f"  |  clustered CI-low>0: {'YES' if lo > 0 else 'no'}"
                  f"  |  >=6/9 positive: {'YES' if pos >= 6 else 'no'}"
                  f"  ->  {status}")
            # 2026-09-02: this scorer OWNS the formal-track (variant A) number.
            # The prereg board reads this artifact instead of re-counting the
            # CSV itself (mistake.md 2026-08-26: a second implementation
            # silently disagrees). Refreshes only when this script runs
            # (monthly on the 5th, or by hand) -- the board says so.
            out = Path(__file__).resolve().parents[1] / "results" / "sweep_forward_gate.json"
            art = {
                "variant": "A", "universe": "core9", "freeze": "2026-07-28",
                "n": int(st["n"]), "gate_n": GATE_N,
                "mean_r": round(float(st["mean"]), 4), "wr": round(float(st["wr"]), 1),
                "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
                "pos": int(pos), "pos_of": len(SYMS), "status": status,
                "asof_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            with open(out, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(art, fh, ensure_ascii=False, indent=2)
            print(f"  artifact -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
