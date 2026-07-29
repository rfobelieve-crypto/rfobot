# -*- coding: utf-8 -*-
"""Universe expansion — can more symbols shrink Gate F's runway, and do the
frozen rules survive on coins that were never part of development?

The binding Gate F condition is the day-clustered CI, which needs raw
n ~ 7000 at the historical mean; at the 9-coin rate (~230/mo) that is 2.5
years. Trades/month scales with the symbol count, so a wider universe is
the single biggest legitimate accelerator — it changes NO rule, only the
test surface.

Two things must both hold for it to count:
  1. the NEW coins (never seen during development) must be positive on
     their own — this is a genuine out-of-sample test along the symbol
     dimension, not a rerun of the same evidence;
  2. the day-clustered VIF must not rise faster than n does, or effective
     n stalls (more correlated coins = more of the same shock).

Discipline: the added universe is declared BEFORE running (top liquid
USDT perps by 24h volume, mechanical rule) and EVERY added symbol is
reported — no dropping losers after the fact.

Run: python research/sweep_failure/universe_expand.py
Out: research/results/sweep_universe_expand.json
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
os.environ["SLIP"] = "0"
import sweep_core as SC  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = Path(__file__).resolve().parents[2] / "research/results/sweep_universe_expand.json"
CACHE = HERE / ".cache"
BASE = "https://api.binance.com/api/v3/klines"
SCEN_A = {"entry": 7.0, "texit": 3.0, "sexit": 10.0}

ORIGINAL = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
# Declared before running: next liquid USDT pairs, mechanical selection.
ADDED = ["TRX", "DOT", "LTC", "UNI", "ATOM", "ETC", "NEAR", "APT", "FIL",
         "ARB", "OP", "INJ", "SUI", "AAVE", "ICP", "ALGO", "VET", "HBAR",
         "SAND", "AXS"]


def fetch(sym: str, days: int = 900) -> Path | None:
    p = CACHE / f"{sym}USDT_1h.csv"
    if p.exists():
        return p
    end = int(time.time() * 1000)
    cur = end - days * 86400 * 1000
    rows = {}
    try:
        while cur < end:
            req = urllib.request.Request(
                f"{BASE}?symbol={sym}USDT&interval=1h&startTime={cur}&limit=1000",
                headers={"User-Agent": "sweep-research/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.loads(r.read().decode())
            if not d:
                break
            for k in d:
                rows[int(k[0]) // 1000] = (float(k[1]), float(k[2]),
                                           float(k[3]), float(k[4]), float(k[5]))
            cur = int(d[-1][0]) + 3600_000
            if len(d) < 1000:
                break
    except Exception as e:  # noqa: BLE001
        print(f"  {sym}: fetch failed ({e})")
        return None
    if len(rows) < 3000:
        print(f"  {sym}: only {len(rows)} bars — skipped (too short)")
        return None
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        for ts in sorted(rows):
            w.writerow([ts, *rows[ts]])
    return p


def net_trades(sym: str) -> list[tuple[int, float]]:
    p = CACHE / f"{sym}USDT_1h.csv"
    out = []
    for fill_ts, _, r, lvl, atr, stopped in SC.backtest_symbol(SC.load_csv(str(p))):
        legs = SCEN_A["entry"] + (SCEN_A["sexit"] if stopped else SCEN_A["texit"])
        out.append((fill_ts, r - legs / 1e4 * lvl / (SC.DIS * atr)))
    return out


def clustered_vif(pairs: list[tuple[int, float]], nb: int = 2000) -> tuple[float, float]:
    """(VIF, clustered CI-low) for a pooled set of (ts, r)."""
    byd = defaultdict(list)
    for ts, r in pairs:
        byd[datetime.fromtimestamp(ts, tz=timezone.utc).date()].append(r)
    days = list(byd.values())
    rs = [r for _, r in pairs]
    mu = sum(rs) / len(rs)
    rng = random.Random(11)
    cl, iid = [], []
    for _ in range(nb):
        acc, cnt = 0.0, 0
        for _ in range(len(days)):
            g = days[rng.randrange(len(days))]
            acc += sum(g)
            cnt += len(g)
        cl.append(acc / cnt)
        iid.append(sum(rs[rng.randrange(len(rs))] for _ in range(len(rs))) / len(rs))
    v_cl = sum((x - mu) ** 2 for x in cl) / nb
    v_ii = sum((x - mu) ** 2 for x in iid) / nb
    cl.sort()
    return (v_cl / v_ii if v_ii > 0 else float("nan")), cl[int(0.025 * nb)]


def stats(rs):
    n = len(rs)
    m = sum(rs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in rs) / (n - 1))
    return n, m, (m / (sd / math.sqrt(n)) if sd > 0 else 0.0)


def main() -> int:
    print("=" * 74)
    print("  UNIVERSE EXPANSION — frozen rules, declared-before-running list")
    print("=" * 74)
    print(f"  fetching {len(ADDED)} added symbols...")
    ok_added = []
    for s in ADDED:
        if fetch(s):
            ok_added.append(s)
    print(f"  usable: {len(ok_added)}/{len(ADDED)}\n")

    orig_pairs, add_pairs = [], []
    print(f"  {'sym':<7}{'n':>6}{'meanR':>10}{'t':>8}   group")
    for grp, syms, bucket in (("ORIG", ORIGINAL, orig_pairs),
                              ("NEW ", ok_added, add_pairs)):
        for s in syms:
            tr = net_trades(s)
            bucket += tr
            n, m, t = stats([r for _, r in tr])
            print(f"  {s:<7}{n:>6}{m:>+10.4f}{t:>+8.2f}   {grp}")

    res = {}
    for label, pairs in (("original 9", orig_pairs),
                         ("added only", add_pairs),
                         ("combined", orig_pairs + add_pairs)):
        n, m, t = stats([r for _, r in pairs])
        vif, ci_lo = clustered_vif(pairs)
        span_mo = ((max(p[0] for p in pairs) - min(p[0] for p in pairs))
                   / 86400 / 30.44)
        rate = n / span_mo
        # raw n needed for clustered CI-low > 0 at this mean
        sd = math.sqrt(sum((r - m) ** 2 for _, r in pairs) / (n - 1))
        need = (1.96 * sd / m) ** 2 * vif if m > 0 else float("inf")
        yrs = need / rate / 12 if rate > 0 else float("inf")
        res[label] = {"n": n, "mean": m, "t_iid": t, "vif": vif,
                      "ci_lo": ci_lo, "rate_per_month": rate,
                      "need_raw_n": need, "years_to_gate": yrs}
        print(f"\n  [{label}]  n={n}  meanR={m:+.4f}  iid t={t:+.2f}  "
              f"VIF={vif:.2f}  clustered CI-low={ci_lo:+.4f}")
        print(f"      rate={rate:.0f} trades/mo   need raw n≈{need:.0f}   "
              f"-> {yrs:.1f} yr to Gate F")

    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    print("  READ: 'added only' is the symbol-dimension out-of-sample test — "
          "these coins were never part of developing the rules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
