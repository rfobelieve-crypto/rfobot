"""Inspect v7_okx_positions live cohort: three WR definitions side by side.

Read-only. Prints one row per closed trade plus headline stats.

Three "win" definitions reconciled here:
  gross_win  = gross_pct > 0                   (price moved the right way)
  net_win    = net_pct   > 0                   (gross - 8 bps assumed cost)
  equity_win = equity_after > equity_before    (actual wallet truth: includes
                                                real OKX fees, funding, slip)

Usage:
    python scripts/show_live_wr.py
    python scripts/show_live_wr.py --tier Strong
    python scripts/show_live_wr.py --since 2026-06-10
"""
from __future__ import annotations

import argparse
import statistics
from datetime import datetime

from shared.db import get_db_conn


def fetch_closed(tier: str | None, since: datetime | None) -> list[dict]:
    where = ["status = 'CLOSED'"]
    params: list = []
    if tier:
        where.append("entry_tier = %s")
        params.append(tier)
    if since:
        where.append("entry_time >= %s")
        params.append(since)
    sql = f"""
        SELECT id, entry_time, exit_time, direction, entry_tier,
               entry_price, exit_price,
               gross_pct, net_pct, equity_ret_pct,
               equity_before, equity_after,
               exit_reason
        FROM v7_okx_positions
        WHERE {' AND '.join(where)}
        ORDER BY entry_time
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())
    finally:
        conn.close()


def fmt_dt(x) -> str:
    if not x:
        return "-"
    return x.strftime("%m-%d %H:%M")


def hours_between(a, b) -> float:
    if not (a and b):
        return 0.0
    return (b - a).total_seconds() / 3600.0


def safe(x, default=0.0) -> float:
    return float(x) if x is not None else default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["Strong", "Moderate"], default=None)
    ap.add_argument("--since", type=str, default=None,
                    help="YYYY-MM-DD (entry_time >=)")
    args = ap.parse_args()

    since = datetime.strptime(args.since, "%Y-%m-%d") if args.since else None
    rows = fetch_closed(args.tier, since)

    label = f"tier={args.tier or 'ALL'}"
    if since:
        label += f"  since={args.since}"
    print(f"\n=== v7_okx_positions closed trades [{label}] ===")

    if not rows:
        print("(no rows)")
        return 0

    # Per-trade table
    print()
    print(f"{'id':>4} {'entry':>11} {'exit':>11} {'hrs':>5} "
          f"{'dir':>5} {'tier':>8} {'gross_bps':>10} {'net_bps':>9} "
          f"{'eq_Δ%':>7} {'eq_after':>9} {'reason':<14}")
    print("-" * 110)

    gross_bps_all: list[float] = []
    net_bps_all: list[float] = []
    eq_delta_all: list[float] = []  # equity_after - equity_before in USD
    gross_wins = net_wins = eq_wins = 0
    last_eq_after = None

    for r in rows:
        gross_bps = safe(r["gross_pct"]) * 10000
        net_bps = safe(r["net_pct"]) * 10000
        eq_before = safe(r["equity_before"])
        eq_after = safe(r["equity_after"])
        eq_delta_usd = eq_after - eq_before
        eq_delta_pct = (eq_delta_usd / eq_before * 100) if eq_before else 0.0

        gross_bps_all.append(gross_bps)
        net_bps_all.append(net_bps)
        eq_delta_all.append(eq_delta_usd)

        if gross_bps > 0:
            gross_wins += 1
        if net_bps > 0:
            net_wins += 1
        if eq_delta_usd > 0:
            eq_wins += 1

        last_eq_after = eq_after

        print(
            f"{r['id']:>4} "
            f"{fmt_dt(r['entry_time']):>11} "
            f"{fmt_dt(r['exit_time']):>11} "
            f"{hours_between(r['entry_time'], r['exit_time']):>5.1f} "
            f"{(r['direction'] or '-'):>5} "
            f"{(r['entry_tier'] or '-'):>8} "
            f"{gross_bps:>+10.1f} "
            f"{net_bps:>+9.1f} "
            f"{eq_delta_pct:>+7.2f} "
            f"{eq_after:>9.2f} "
            f"{(r['exit_reason'] or '-'):<14}"
        )

    n = len(rows)
    print("-" * 110)

    # Three WR definitions
    print()
    print(f"=== WR comparison (n={n}) ===")
    print(f"  gross WR  (gross_pct > 0)               : "
          f"{gross_wins}/{n} = {gross_wins / n * 100:5.1f}%   "
          f"<- what /okx-perf currently shows")
    print(f"  net   WR  (net_pct > 0, assumes 8 bps)  : "
          f"{net_wins}/{n} = {net_wins / n * 100:5.1f}%   "
          f"<- closer to 'made money'")
    print(f"  equity WR (equity_after > before, real) : "
          f"{eq_wins}/{n} = {eq_wins / n * 100:5.1f}%   "
          f"<- wallet truth, includes real fees + funding")
    print(f"  delta (gross - equity)                  : "
          f"{(gross_wins - eq_wins) / n * 100:+5.1f}pp   "
          f"<- inflation in the headline WR")

    # Headline economics
    print()
    print("=== Economics ===")
    avg_gross = statistics.mean(gross_bps_all)
    avg_net = statistics.mean(net_bps_all)
    cum_net_bps = sum(net_bps_all)
    sum_eq_delta = sum(eq_delta_all)

    print(f"  avg gross  : {avg_gross:+8.2f} bps / trade")
    print(f"  avg net    : {avg_net:+8.2f} bps / trade   "
          f"(Gate B threshold: ≥ 0)")
    print(f"  cum net    : {cum_net_bps:+8.2f} bps total")
    print(f"  Σ equity Δ : ${sum_eq_delta:+8.2f}   "
          f"(actual wallet change across all closed trades)")
    if last_eq_after is not None:
        print(f"  last eq    : ${last_eq_after:8.2f}")

    # Profit factor on net
    wins_sum = sum(x for x in net_bps_all if x > 0)
    losses_sum = abs(sum(x for x in net_bps_all if x < 0))
    pf = wins_sum / losses_sum if losses_sum > 0 else float("inf")
    print(f"  PF (net)   : {pf:8.2f}   (wins {wins_sum:+.1f} / losses {losses_sum:.1f})")

    # Gate B context
    print()
    print("=== Gate B status ===")
    print(f"  samples this cohort : {n}")
    print(f"  needed for Gate B   : 30-50 clean trades post trail-fix")
    print(f"  Gate B pass req     : avg_net ≥ 0  AND  0 kill triggers  "
          f"AND  trail amends working  AND  no negative slip surprise")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
