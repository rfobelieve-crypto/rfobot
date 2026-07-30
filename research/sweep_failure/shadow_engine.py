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
          "exit_ts", "exit_utc", "stopped", "gross_r", "net_r",
          # Order-flow annotation (2026-07-31, operator: "shadow 不整合我就
          # 沒辦法看到訊號的樣子"). PROSPECTIVE observation columns computed
          # from public 1m klines at record time — they change NOTHING in
          # the gate arithmetic (gate_stats reads net_r/variant_b/status
          # only; the registered Variant B track is untouched). Stored raw,
          # no thresholds baked in, so the October pre-registration can
          # apply its cuts to genuinely prospective values instead of
          # retro-computed ones. Definitions mirror the raid-anatomy round-4
          # attack-window features, sourced from klines so all 29 symbols
          # get them (taker delta = 2*taker_buy_base - volume):
          #   flow_reject   1m close crossed back inside before hour end
          #   flow_att_min  minutes spent beyond the level in the sweep hour
          #   flow_vshock   attack-minute vol / trailing-24h median 1m vol
          #   flow_taker    signed taker share INTO the break, attack mins
          #   flow_absorb   flow_taker / max(pierce, 0.1)
          "flow_reject", "flow_att_min", "flow_vshock", "flow_taker",
          "flow_absorb"]

FLOW_BACKFILL_PER_RUN = 40      # cap 1m-kline fetch work per hourly run


def fetch_1m_window(sym: str, start_s: int, end_s: int) -> dict[int, tuple]:
    """minute -> (high, low, close, volume, taker_buy_base) from Binance
    spot 1m klines, [start_s, end_s). No disk cache: callers keep windows
    small (~25h) and the hourly run touches only new/blank rows."""
    out: dict[int, tuple] = {}
    cur = start_s * 1000
    while cur < end_s * 1000:
        req = urllib.request.Request(
            f"{BASE}?symbol={sym}USDT&interval=1m&startTime={int(cur)}"
            f"&endTime={end_s * 1000}&limit=1000",
            headers={"User-Agent": "sweep-shadow/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read().decode())
        if not d:
            break
        for k in d:
            out[int(k[0]) // 60_000] = (float(k[2]), float(k[3]), float(k[4]),
                                        float(k[5]), float(k[9]))
        cur = int(d[-1][0]) + 60_000
        if len(d) < 1000:
            break
        time.sleep(0.05)
    return out


def find_sweep(bars, fill_ts: int, lvl: float, atr: float, pierce: float):
    """Recover the sweep bar (and side) the log row came from: the nearest
    bar before the fill whose pierce of `lvl` matches the recorded
    pierce_atr. The log predates these columns, so this stays derivable for
    every historical row instead of only new ones."""
    idx = {b[0]: i for i, b in enumerate(bars)}
    j0 = idx.get(fill_ts)
    if j0 is None or atr <= 0:
        return None
    for j in range(j0 - 1, max(-1, j0 - 1 - SC.W), -1):
        hi, lo = bars[j][SC.H], bars[j][SC.L]
        if abs((hi - lvl) / atr - pierce) < 5e-4 and hi > lvl:
            return bars[j][0], 1
        if abs((lvl - lo) / atr - pierce) < 5e-4 and lo < lvl:
            return bars[j][0], -1
    return None


def flow_flags(sym: str, sweep_ts: int, side: int, lvl: float) -> dict | None:
    """The round-4 attack-window features for ONE sweep, minutes-only."""
    m0 = sweep_ts // 60
    m1 = fetch_1m_window(sym, sweep_ts - 24 * 3600, sweep_ts + 3600)
    mins = [(m, *m1[m]) for m in range(m0, m0 + 60) if m in m1]
    base = [m1[m][3] for m in range(m0 - 1440, m0) if m in m1]
    if len(mins) < 50 or len(base) < 500:
        return None
    beyond = [(m, hi, lo, c, v, tb) for (m, hi, lo, c, v, tb) in mins
              if (hi > lvl if side == 1 else lo < lvl)]
    if not beyond:
        return None
    last_beyond = beyond[-1][0]
    rej = 0
    for (m, _hi, _lo, c, _v, _tb) in mins:
        if m > last_beyond and ((c <= lvl) if side == 1 else (c >= lvl)):
            rej = 1
            break
    vol = sum(b[4] for b in beyond)
    med = sorted(base)[len(base) // 2]
    taker = (side * sum(2 * b[5] - b[4] for b in beyond) / vol
             if vol > 0 else None)
    return {
        "flow_reject": rej,
        "flow_att_min": len(beyond),
        "flow_vshock": (round((vol / len(beyond)) / med, 2)
                        if med > 0 else None),
        "flow_taker": round(taker, 4) if taker is not None else None,
    }


def annotate_flow(log: dict, sym: str, bars) -> int:
    """Fill blank flow_* columns for this symbol's rows, oldest first,
    respecting the per-run fetch cap. Values are immutable once written
    (the sweep hour is always closed by the time a fill exists)."""
    todo = sorted((k for k, r in log.items()
                   if k[0] == sym and r.get("flow_reject") in (None, "")
                   and r.get("pierce_atr")),
                  key=lambda k: k[2])[:FLOW_BACKFILL_PER_RUN]
    done = 0
    for k in todo:
        r = log[k]
        try:
            sw = find_sweep(bars, int(r["fill_ts"]), float(r["entry_px"]),
                            float(r["atr"]), float(r["pierce_atr"]))
            if sw is None:
                r["flow_reject"] = "na"
                continue
            sweep_ts, side = sw
            f = flow_flags(sym, sweep_ts, side, float(r["entry_px"]))
            if f is None:
                r["flow_reject"] = "na"
                continue
            pierce = float(r["pierce_atr"])
            f["flow_absorb"] = (round(f["flow_taker"] / max(pierce, 0.1), 3)
                                if f.get("flow_taker") is not None else "")
            for c in ("flow_reject", "flow_att_min", "flow_vshock",
                      "flow_taker", "flow_absorb"):
                r[c] = f.get(c, "")
            done += 1
        except Exception as e:  # noqa: BLE001 — leave blank, retry next run
            print(f"  {sym}: flow annotate failed ({type(e).__name__}: {e})")
            break
    return done


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


def lt_to_scen_a(netr: float, lvl: float, atr: float, stopped: bool) -> tuple:
    """(gross, scenario-A net) recovered from an LT.trade_levels row.

    LT.net charges a flat 2 x 5 bps taker regardless of exit type; the Gate F
    cost spec (scenario A) charges 7 entry + 3 time-exit / 10 stop-exit. The
    two agree on time exits (10 = 10) but under-cost stop-outs by 7 bps.
    Found 2026-07-30 — until then the shadow log mixed the two models across
    level kinds. Un-net LT's flat cost exactly, then apply scenario A.
    """
    gross = netr + 2 * LT.TAKER / 1e4 * lvl / (SC.DIS * atr)
    return gross, net_r(gross, lvl, atr, stopped)


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


GATE_N = 1400          # same floor as Variant A's Gate F


def gate_stats(log: dict) -> dict:
    """Structured Variant B gate progress — same arithmetic as Gate F:
    day-clustered bootstrap CI on net R plus the n>=1400 floor. Consumed by
    gate_progress (weekly report string) and the agent's /public/sweep-status
    JSON route (product-site strategy board). Without this the shadow log
    accumulates unwatched — a two-month clock nobody reads is a clock that
    does not exist."""
    import random
    from collections import defaultdict
    rows = [r for r in log.values()
            if str(r.get("variant_b", "")) == "1" and r["status"] == "CLOSED"
            and r["net_r"] != ""]
    n_open = sum(1 for r in log.values()
                 if str(r.get("variant_b", "")) == "1" and r["status"] == "OPEN")
    if not rows:
        return {"n_closed": 0, "n_open": n_open, "floor": GATE_N,
                "mean_r": None, "ci_low": None, "wr_pct": None, "status": "empty"}
    byd = defaultdict(list)
    for r in rows:
        d = datetime.fromtimestamp(int(r["fill_ts"]), timezone.utc).date()
        byd[d].append(float(r["net_r"]))
    days = list(byd.values())
    rs = [float(r["net_r"]) for r in rows]
    n = len(rs)
    mean = sum(rs) / n
    rng = random.Random(7)
    means = []
    for _ in range(2000):
        acc = cnt = 0.0
        for _ in range(len(days)):
            g = days[rng.randrange(len(days))]
            acc += sum(g)
            cnt += len(g)
        means.append(acc / cnt)
    means.sort()
    lo = means[50]
    ok = n >= GATE_N and lo > 0
    return {"n_closed": n, "n_open": n_open, "floor": GATE_N,
            "mean_r": round(mean, 4), "ci_low": round(lo, 4),
            "wr_pct": round(100 * sum(1 for x in rs if x > 0) / n, 1),
            "status": "PASS" if ok else "accumulating"}


def gate_progress(log: dict) -> str:
    """One line of Variant B gate progress, for the weekly report."""
    s = gate_stats(log)
    if s["n_closed"] == 0:
        return "Variant B: 0/%d (no closed signals yet)" % GATE_N
    return (f"Variant B: n={s['n_closed']}/{s['floor']} meanR={s['mean_r']:+.4f} "
            f"clustered-CI-low={s['ci_low']:+.4f} -> "
            f"{'PASS' if s['status'] == 'PASS' else 'accumulating'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true",
                    help="print the log summary without refreshing data")
    ap.add_argument("--gate", action="store_true",
                    help="one-line Variant B gate progress (for the weekly report)")
    args = ap.parse_args()
    log = read_log()
    if args.gate:
        print(gate_progress(log))
        return 0
    if args.summary:
        summary(log)
        return 0

    now = datetime.now(timezone.utc)
    stamp = f"{now:%Y-%m-%d %H:%M}"
    new = 0
    flow_done = 0
    for uni, syms in (("core9", CORE9), ("added20", ADDED20)):
        for sym in syms:
            p = refresh(sym)
            if p is None:
                continue
            bars = SC.load_csv(str(p))
            last_ts = bars[-1][0]
            # one source of truth per kind: the frozen engine for swing,
            # the shared level engine for the time-defined pools. Tuples carry
            # (kind, fill_ts, exit_ts, gross, net, pierce, lvl, atr, stopped);
            # both paths are costed under scenario A (lt_to_scen_a un-nets
            # LT's flat taker model first).
            evts: list[tuple] = [
                ("swing", t[0], t[1], t[2],
                 net_r(t[2], t[3], t[4], t[5]), t[6], t[3], t[4], t[5])
                for t in SC.backtest_symbol(bars)]
            lv = LT.build_levels(bars)
            for kind in ("session", "pdh_pdl", "pwh_pwl"):
                for (f_ts, x_ts, netr, pc, lvl, atr, st_) in LT.trade_levels(
                        bars, lv.get(kind, [])):
                    gross, neta = lt_to_scen_a(netr, lvl, atr, st_)
                    evts.append((kind, f_ts, x_ts, gross, neta, pc, lvl, atr, st_))

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
            flow_done += annotate_flow(log, sym, bars)
    write_log(log)
    n_flow = sum(1 for r in log.values()
                 if r.get("flow_reject") not in (None, "", "na"))
    print(f"shadow run {stamp}  new signals: {new}  "
          f"flow annotated: +{flow_done} ({n_flow}/{len(log)} covered)")
    summary(log)
    print("  " + gate_progress(log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
