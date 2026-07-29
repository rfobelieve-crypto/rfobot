# -*- coding: utf-8 -*-
"""Sweep-failure SHADOW engine — prospective signal log, no money, no orders.

Purpose (2026-07-29): stop serialising engineering behind validation. Gate F
needs ~1.5 years of forward data either way; the execution path can be built
and debugged during that time instead of after it, so that the day the gate
passes the plumbing is already proven.

What it does, hourly:
  1. incrementally refresh the 1H kline cache (append-only, dedup by ts)
  2. run the FROZEN backtest function on full history — literally the same
     code Gate F scores, so the shadow log cannot drift from the gate
  3. write every post-freeze trade to a prospective CSV log, keyed by
     (symbol, fill_ts), stamping first_seen_utc the first time it appears
  4. re-derive outcomes each run; a trade stays OPEN until its full HOLD
     window has elapsed in the data

What it deliberately does NOT do: place orders, touch the OKX executor,
write to any production table. It is a recorder.

Why first_seen matters: sweep_forward recomputes forward trades from history
on every run, which is fine but leaves no proof the signal was known before
its outcome. first_seen_utc is that proof — a row whose first_seen precedes
its exit_ts is a genuinely prospective observation.

Universes: `core9` = the frozen Gate F basket. `added20` = the 2026-07-29
expansion (informational until the portfolio framework settles the
concurrency cap). Rows are tagged so Gate F can keep scoring core9 alone.

Run:      python research/sweep_failure/shadow_engine.py
Summary:  python research/sweep_failure/shadow_engine.py --summary
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
os.environ["SLIP"] = "0"          # gross engine; bps costs applied here
import sweep_core as SC            # noqa: E402
import level_types as LT           # noqa: E402  (same trade fn = no drift)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CACHE = HERE / ".cache"
LOG = Path(__file__).resolve().parents[2] / "research/results/sweep_shadow_log.csv"
BASE = "https://api.binance.com/api/v3/klines"
FREEZE_TS = int(datetime(2026, 7, 28, tzinfo=timezone.utc).timestamp())
SCEN_A = {"entry": 7.0, "texit": 3.0, "sexit": 10.0}

CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
ADDED20 = ["TRX", "DOT", "LTC", "UNI", "ATOM", "ETC", "NEAR", "APT", "FIL",
           "ARB", "OP", "INJ", "SUI", "AAVE", "ICP", "ALGO", "VET", "HBAR",
           "SAND", "AXS"]

# Variant B (pre-registered 2026-07-29, threshold fixed BEFORE any forward
# trade): take the signal ONLY when the sweep bar pierced the level by
# <= 0.25 ATR. Chosen as the round number nearest the 0.23 tercile boundary
# from the pre-declared tercile bucketing in winner_anatomy.py — not the
# t-peak, and not load-bearing: the whole 0.10-1.00 ATR sweep is positive
# and decays smoothly to the unfiltered mean, so there is no cliff to sit on.
PIERCE_MAX_B = 0.25
# AMENDED 2026-07-29, hours after registration and with 8 forward signals
# accrued (i.e. before any meaningful forward evidence): the liquidity SOURCE
# widens from swing pivots alone to all four pool types the LMSR map draws —
# swing, session extremes, PDH/PDL, PWH/PWL. Every type was tested and every
# type is reported; none were dropped (session is the weakest and is kept).
# This does not loosen the statistical bar — the gate arithmetic is untouched.
# It broadens the test surface, which takes the forward answer from ~8 months
# to ~2 (1327 filtered trades/month vs 245). The pierce filter generalising
# across all four independently is the reason to trust the widening at all.
LEVEL_KINDS = ("swing", "session", "pdh_pdl", "pwh_pwl")

FIELDS = ["symbol", "universe", "level_kind", "first_seen_utc", "fill_ts",
          "fill_utc", "entry_px", "atr", "pierce_atr", "variant_b", "status",
          "exit_ts", "exit_utc", "stopped", "gross_r", "net_r"]


def refresh(sym: str) -> Path | None:
    """Append new closed 1H bars to the cache. Never rewrites old bars —
    a revision by the exchange would show up as a shadow/backtest mismatch
    rather than being silently absorbed."""
    p = CACHE / f"{sym}USDT_1h.csv"
    if not p.exists():
        return None
    rows = SC.load_csv(str(p))
    last = rows[-1][0]
    got = []
    try:
        req = urllib.request.Request(
            f"{BASE}?symbol={sym}USDT&interval=1h&startTime={(last + 3600) * 1000}"
            f"&limit=1000", headers={"User-Agent": "sweep-shadow/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read().decode())
        now_ms = int(time.time() * 1000)
        for k in d:
            if int(k[6]) > now_ms:      # close_time in the future = live bar
                continue
            got.append([int(k[0]) // 1000, float(k[1]), float(k[2]),
                        float(k[3]), float(k[4]), float(k[5])])
    except Exception as e:  # noqa: BLE001
        print(f"  {sym}: refresh failed ({e})")
        return p
    if got:
        with p.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(got)
    return p


def net_r(r: float, lvl: float, atr: float, stopped: bool) -> float:
    legs = SCEN_A["entry"] + (SCEN_A["sexit"] if stopped else SCEN_A["texit"])
    return r - legs / 1e4 * lvl / (SC.DIS * atr)


def read_log() -> dict[tuple[str, str, int], dict]:
    out = {}
    if LOG.exists():
        with LOG.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[(row["symbol"], row.get("level_kind", "swing"),
                     int(row["fill_ts"]))] = row
    return out


def write_log(log: dict[tuple[str, int], dict]) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(log, key=lambda x: (x[2], x[0], x[1])):
            w.writerow({c: log[k].get(c, "") for c in FIELDS})


def summary(log: dict) -> None:
    rows = list(log.values())
    if not rows:
        print("  (empty log)")
        return
    print(f"  logged rows: {len(rows)}")
    # freshly-built rows hold int 1; rows read back from CSV hold "1"
    def _isb(r):
        return str(r.get("variant_b", "")) == "1"
    groups = [("ALL", lambda r: True), ("B (pierce)", _isb)]
    groups += [(f"B:{k}", (lambda k_: lambda r: _isb(r)
                           and r.get("level_kind") == k_)(k))
               for k in LEVEL_KINDS]
    for uni, pred in groups:
        sub = [r for r in rows if pred(r)]
        closed = [r for r in sub if r["status"] == "CLOSED"]
        if not sub:
            continue
        line = f"  {uni:<8} n={len(sub):>4}  open={len(sub)-len(closed):>3}"
        if closed:
            rs = [float(r["net_r"]) for r in closed]
            m = sum(rs) / len(rs)
            sd = (math.sqrt(sum((x - m) ** 2 for x in rs) / (len(rs) - 1))
                  if len(rs) > 1 else 0.0)
            t = m / (sd / math.sqrt(len(rs))) if sd > 0 else 0.0
            wr = 100 * sum(1 for x in rs if x > 0) / len(rs)
            line += (f"  closed={len(closed):>4}  meanR={m:+.4f}"
                     f"  WR={wr:.0f}%  t={t:+.2f}")
        print(line)
    # prospectiveness: rows whose outcome was still unknown when first seen
    pro = sum(1 for r in rows if r["status"] == "CLOSED" and r["exit_ts"]
              and r["first_seen_utc"] and
              datetime.strptime(r["first_seen_utc"], "%Y-%m-%d %H:%M").replace(
                  tzinfo=timezone.utc).timestamp() < int(r["exit_ts"]))
    print(f"  genuinely prospective (first_seen < exit): {pro}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true",
                    help="print the log summary without refreshing data")
    args = ap.parse_args()
    log = read_log()
    if args.summary:
        summary(log)
        return 0

    now = datetime.now(timezone.utc)
    stamp = f"{now:%Y-%m-%d %H:%M}"
    new = 0
    for uni, syms in (("core9", CORE9), ("added20", ADDED20)):
        for sym in syms:
            p = refresh(sym)
            if p is None:
                continue
            bars = SC.load_csv(str(p))
            last_ts = bars[-1][0]
            # one source of truth per kind: the frozen engine for swing, the
            # shared level engine for the time-defined pools
            # one source of truth per kind: the frozen engine for swing,
            # the shared level engine for the time-defined pools. Tuples carry
            # (kind, fill_ts, exit_ts, gross, net, pierce, lvl, atr, stopped);
            # swing yields gross R (bps model applied here), LT.trade_levels
            # already returns net, so both are carried explicitly rather than
            # re-derived.
            evts: list[tuple] = [
                ("swing", t[0], t[1], t[2],
                 net_r(t[2], t[3], t[4], t[5]), t[6], t[3], t[4], t[5])
                for t in SC.backtest_symbol(bars)]
            lv = LT.build_levels(bars)
            for kind in ("session", "pdh_pdl", "pwh_pwl"):
                for (f_ts, x_ts, netr, pc, lvl, atr, st_) in LT.trade_levels(
                        bars, lv.get(kind, [])):
                    evts.append((kind, f_ts, x_ts, None, netr, pc, lvl, atr, st_))

            for (kind, fill_ts, exit_ts, gross, netv, pierce, lvl, atr,
                 stopped) in evts:
                if fill_ts < FREEZE_TS:
                    continue
                key = (sym, kind, fill_ts)
                done = fill_ts + SC.HOLD * 3600 <= last_ts
                row = log.get(key)
                if row is None:
                    row = {"symbol": sym, "universe": uni, "level_kind": kind,
                           "first_seen_utc": stamp, "fill_ts": fill_ts,
                           "fill_utc": f"{datetime.fromtimestamp(fill_ts, timezone.utc):%Y-%m-%d %H:%M}",
                           "entry_px": f"{lvl:.6f}", "atr": f"{atr:.6f}",
                           "pierce_atr": f"{pierce:.4f}",
                           "variant_b": int(pierce <= PIERCE_MAX_B)}
                    log[key] = row
                    new += 1
                row.update({
                    "status": "CLOSED" if done else "OPEN",
                    "exit_ts": exit_ts if done else "",
                    "exit_utc": (f"{datetime.fromtimestamp(exit_ts, timezone.utc):%Y-%m-%d %H:%M}"
                                 if done else ""),
                    "stopped": int(bool(stopped)) if done else "",
                    "gross_r": f"{gross:.6f}" if (done and gross is not None) else "",
                    "net_r": f"{netv:.6f}" if done else "",
                })
    write_log(log)
    print(f"shadow run {stamp}  new signals: {new}")
    summary(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
