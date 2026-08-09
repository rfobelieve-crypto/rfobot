"""Position-sizing Monte Carlo for the sweep-failure line — how much per trade?

Question (user, 2026-08-09): "change the frozen rule to 5-10% risk per trade,
that's what small-account compounding needs". Answered with the strategy's OWN
forward record instead of an argument.

── 2026-08-09 REWRITE — the v1 model printed $5e20 after two years ──────────
That number was the bug reporting itself (mistake.md: a result that
contradicts priors means check the instrument, not interpret it). Three
missing constraints, all now enforced:

  1. CONCURRENCY. The log fires up to 15 trades in one hour across 29 coins.
     The frozen rule caps simultaneous positions at 5-10; v1 took all 15, so
     the fattest blocks were unreachable in reality — in both directions.
  2. CAPACITY. v1 let risk scale with equity forever, so $2k compounded past
     $1e20 while still assuming backtest fills. Real edge has a size ceiling:
     past some notional you ARE the book. Modelled as an absolute cap on
     risk-dollars per trade; beyond it growth turns additive, not geometric.
     The true ceiling is UNKNOWN — hence a sensitivity sweep, not one number.
  3. EDGE DECAY. v1 assumed +0.079R/trade forever. Modelled as an exponential
     half-life on the MEAN only; dispersion is preserved, so tail risk does
     not conveniently shrink along with the edge.

WHY BLOCK-RESAMPLE BY HOUR: trades firing in the same hour are one crypto-beta
event hitting many symbols, not independent draws. The resampling unit is the
whole hour-block, so concurrent losses stay correlated.

Reads: research/results/sweep_shadow_log.csv — variant B, CLOSED, and
GENUINELY PROSPECTIVE only (first_seen < exit; the rest is freeze-day backfill).

Usage:
    python research/sweep_risk_sizing_mc.py                      # 90d, $2000
    python research/sweep_risk_sizing_mc.py --start 2000 --days 365
    python research/sweep_risk_sizing_mc.py --cap-usd 200 --halflife 180
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOG = ROOT / "research" / "results" / "sweep_shadow_log.csv"
N_PATHS = 20_000
RISKS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.10)
RUIN_AT = 0.10          # equity <= 10% of start
TOTAL_KILL = -0.30      # project CAP-4 (DEMOTE — stops, needs manual restart)
RNG = np.random.default_rng(20260809)


def load_blocks(concurrency: int):
    rows = [r for r in csv.DictReader(LOG.open(encoding="utf-8"))
            if r.get("variant_b") == "1" and r.get("status") == "CLOSED"
            and r.get("net_r") not in (None, "")
            and r.get("first_seen_utc") and r.get("exit_utc")
            and r["first_seen_utc"] < r["exit_utc"]]
    by: dict[str, list[float]] = collections.defaultdict(list)
    for r in rows:
        by[r["fill_utc"][:13]].append(float(r["net_r"]))
    blocks = [np.asarray(v[:concurrency]) for v in by.values()]
    hours = sorted(by)
    P = "%Y-%m-%d %H"
    d0 = dt.datetime.strptime(hours[0], P)
    d1 = dt.datetime.strptime(hours[-1], P)
    return blocks, len(rows), max((d1 - d0).days, 1)


def simulate(sums, risk, n_blocks, start, cap_usd, halflife_blocks):
    idx = RNG.integers(0, len(sums), size=(N_PATHS, n_blocks))
    draw = sums[idx]
    mu = float(draw.mean())
    eq = np.full(N_PATHS, float(start))
    alive = np.ones(N_PATHS, bool)
    ruined = np.zeros(N_PATHS, bool)
    killed = np.zeros(N_PATHS, bool)
    capped = 0.0
    for t in range(n_blocks):
        r = draw[:, t]
        if halflife_blocks:
            r = (r - mu) + mu * 0.5 ** (t / halflife_blocks)
        want = eq * risk
        risk_usd = np.minimum(want, cap_usd)
        capped += float(np.mean(want > cap_usd))
        eq = eq + np.where(alive, risk_usd * r, 0.0)
        killed |= alive & (eq <= start * (1 + TOTAL_KILL))
        new = alive & (eq <= start * RUIN_AT)
        ruined |= new
        alive &= ~new
        eq = np.maximum(eq, 0.0)
    return eq, ruined, killed, capped / n_blocks


def money(x: float) -> str:
    return "$" + format(x, ",.0f")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=2000.0)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--cap-usd", type=float, default=None)
    ap.add_argument("--halflife", type=int, default=365,
                    help="edge half-life in days (0 = none)")
    a = ap.parse_args()

    blocks, n_raw, span = load_blocks(a.concurrency)
    sums = np.array([b.sum() for b in blocks])
    rate = len(blocks) / span
    n_blocks = int(round(rate * a.days))
    hl_blocks = int(a.halflife * rate) if a.halflife else 0

    print(f"真前瞻 {n_raw} 筆 / {span} 天 · 火時區塊 {len(blocks)} 個 = "
          f"{rate:.1f} 區塊/天（並發上限 {a.concurrency}）")
    print(f"  每區塊 ΣnetR  平均 {sums.mean():+.3f}  最差 {sums.min():+.3f}  "
          f"最好 {sums.max():+.3f}")
    print(f"\n{a.days} 天 = {n_blocks:,} 區塊 × {N_PATHS:,} 路徑 · 起始 {money(a.start)}"
          f" · edge 半衰期 {a.halflife or '無'} 天")
    print(f"  ⚠ 樣本僅 {span} 天，{a.days} 天是外推；假設 edge 持續、成本不變、"
          "並發填得滿\n")

    caps = [a.cap_usd] if a.cap_usd else [50.0, 200.0, 1000.0, 1e12]
    for cap in caps:
        label = ("無容量上限（不現實·對照組）" if cap >= 1e11
                 else f"容量上限：每筆風險 ≤ {money(cap)}")
        print(f"── {label} ──")
        print(f"{'每筆風險':>8} {'破產率':>8} {'撞-30%':>9} {'中位末值':>14} "
              f"{'p5':>14} {'p95':>14} {'受限%':>7}")
        for risk in RISKS:
            eq, ruin, kill, capf = simulate(sums, risk, n_blocks, a.start,
                                            cap, hl_blocks)
            print(f"{risk*100:>7.1f}% {ruin.mean()*100:>7.1f}% "
                  f"{kill.mean()*100:>8.1f}% {money(np.median(eq)):>14} "
                  f"{money(np.percentile(eq, 5)):>14} "
                  f"{money(np.percentile(eq, 95)):>14} {capf*100:>6.0f}%")
        print()
    print("讀法：容量上限決定複利能走多遠 —— 它是本模型最不確定、卻最影響結果的假設。")
    print("      受限% 接近 100 代表成長已從幾何轉為線性（不再複利）。")
    print("      破產 = 權益剩 10%；撞-30% = CAP-4 DEMOTE，停機需人工重驗。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
