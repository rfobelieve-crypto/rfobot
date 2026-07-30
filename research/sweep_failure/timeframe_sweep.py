# -*- coding: utf-8 -*-
"""Which timeframe does sweep-failure live on? 15m / 30m / 1h / 2h / 4h.

User question (2026-07-30): 時間級別用什麼最好. The frozen rules run on 1H
because that is where they were born, not because anyone scanned. Like the
pivot-strength test, the healthy outcome is a PLATEAU — the phenomenon
existing across scales — and a sharp peak at 1h would be a fitting mark.

Held fixed across timeframes: the RULES in bars (PIVOT=10, W=8, HOLD=8,
DIS=3.5) and the swing-pool definition. That means wall-clock horizons scale
with the timeframe (15m holds ~2h, 4h holds ~32h) — deliberate: the question
is which timescale the stop-run/failure dynamic lives on, not how to retune.

Costs are where lower timeframes must pay their dues: the bps cost is fixed
in PRICE terms while ATR shrinks with the timeframe, so cost in R units
grows as the bars get smaller — the exact mechanism that killed the
subhourly system (G2). Gross and net are both reported so that squeeze is
visible rather than hidden.

Data windows (honest, not hidden): 1h/2h/4h come from the 900d cache
(2h/4h resampled exactly, UTC-aligned). 15m/30m are fetched fresh with a
400d window (Binance request budget); the 1h row is ALSO recomputed on the
same 400d window so the low-TF comparison is window-matched.

Multiple-comparison note: this scan adds to the repo's trial count (the
Deflated-Sharpe N). It is run as CHARACTERIZATION of a frozen rule — the 1h
track stays the registered one regardless of the winner here; any TF change
would be a new variant paying its own N.

Run: python research/sweep_failure/timeframe_sweep.py
Out: research/results/sweep_timeframe.json
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
os.environ["SLIP"] = "0"
import sweep_core as SC  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = Path(__file__).resolve().parents[2] / "research/results/sweep_timeframe.json"
CACHE = HERE / ".cache"
TF_CACHE = CACHE / "tf"
COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX",
         "TRX", "DOT", "LTC", "UNI", "ATOM", "ETC", "NEAR", "APT", "FIL",
         "ARB", "OP", "INJ", "SUI", "AAVE", "ICP", "ALGO", "VET", "HBAR",
         "SAND", "AXS"]
TAKER = 5.0
PIERCE_MAX = 0.25
LTF_DAYS = 400
BASE = "https://api.binance.com/api/v3/klines"


def fetch_tf(sym: str, interval: str, days: int) -> list | None:
    TF_CACHE.mkdir(parents=True, exist_ok=True)
    p = TF_CACHE / f"{sym}_{interval}.csv"
    if p.exists():
        return SC.load_csv(str(p))
    end = int(time.time() * 1000)
    cur = end - days * 86400 * 1000
    rows = {}
    try:
        while cur < end:
            req = urllib.request.Request(
                f"{BASE}?symbol={sym}USDT&interval={interval}"
                f"&startTime={cur}&limit=1000",
                headers={"User-Agent": "sweep-tf/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.loads(r.read().decode())
            if not d:
                break
            for k in d:
                if int(k[6]) > end:
                    continue
                rows[int(k[0]) // 1000] = (float(k[1]), float(k[2]),
                                           float(k[3]), float(k[4]), float(k[5]))
            cur = int(d[-1][0]) + 1
            if len(d) < 1000:
                break
            time.sleep(0.05)
    except Exception as e:  # noqa: BLE001
        print(f"  {sym} {interval}: fetch failed ({e})")
        return None
    if len(rows) < 2000:
        return None
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        for ts in sorted(rows):
            w.writerow([ts, *rows[ts]])
    return SC.load_csv(str(p))


def resample(bars: list, hours: int) -> list:
    """Exact OHLC aggregation of the 1h cache, UTC-aligned buckets."""
    out = {}
    step = hours * 3600
    for b in bars:
        k = b[0] // step * step
        if k not in out:
            out[k] = [k, b[1], b[2], b[3], b[4], b[5]]
        else:
            o = out[k]
            o[2] = max(o[2], b[2])
            o[3] = min(o[3], b[3])
            o[4] = b[4]
            o[5] += b[5]
    return [tuple(v) for _, v in sorted(out.items())]


def score(bars_by_sym: dict) -> dict:
    g_all, g_sh = [], []
    for sym, bars in bars_by_sym.items():
        if not bars:
            continue
        for (_t, _x, R, lvl, atr, stopped, pc) in SC.backtest_symbol(bars):
            net = R - 2 * TAKER / 1e4 * lvl / (SC.DIS * atr)
            g_all.append((R, net))
            if pc <= PIERCE_MAX:
                g_sh.append((R, net))
    def st(pairs):
        if len(pairs) < 100:
            return None
        gr = [a for a, _ in pairs]
        nt = [b for _, b in pairs]
        n = len(nt)
        mg = sum(gr) / n
        mn = sum(nt) / n
        sd = math.sqrt(sum((x - mn) ** 2 for x in nt) / (n - 1))
        return {"n": n, "gross": mg, "net": mn,
                "t_net": mn / (sd / math.sqrt(n)),
                "cost_R": mg - mn}
    return {"all": st(g_all), "shallow": st(g_sh)}


def fmt(s):
    if not s:
        return "thin"
    return (f"n={s['n']:>6}  gross {s['gross']:+.4f}  cost {s['cost_R']:.4f}"
            f"  net {s['net']:+.4f} (t{s['t_net']:+.1f})")


def main() -> int:
    print("=" * 78)
    print("  TIMEFRAME SWEEP — frozen bar-rules, swing pools, all TFs reported")
    print("=" * 78)
    res = {}

    h1 = {s: (SC.load_csv(str(CACHE / f"{s}USDT_1h.csv"))
              if (CACHE / f"{s}USDT_1h.csv").exists() else None)
          for s in COINS}

    # 900d family: 1h baseline + exact resamples
    for label, bars in (("1h (900d)", h1),
                        ("2h (900d)", {s: resample(b, 2) if b else None
                                       for s, b in h1.items()}),
                        ("4h (900d)", {s: resample(b, 4) if b else None
                                       for s, b in h1.items()})):
        r = score(bars)
        res[label] = r
        print(f"\n  {label:<12} all: {fmt(r['all'])}")
        print(f"  {'':<12} sh≤0.25: {fmt(r['shallow'])}")

    # 400d family: fetched LTF + window-matched 1h
    cutoff = int(time.time()) - LTF_DAYS * 86400
    h1_400 = {s: ([b for b in bars if b[0] >= cutoff] if bars else None)
              for s, bars in h1.items()}
    fam = [("1h (400d)", h1_400)]
    for iv in ("30m", "15m"):
        print(f"\n  fetching {iv} x {len(COINS)} coins ({LTF_DAYS}d)...")
        fam.append((f"{iv} (400d)",
                    {s: fetch_tf(s, iv, LTF_DAYS) for s in COINS}))
    for label, bars in fam:
        r = score(bars)
        res[label] = r
        print(f"\n  {label:<12} all: {fmt(r['all'])}")
        print(f"  {'':<12} sh≤0.25: {fmt(r['shallow'])}")

    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    print("  READ: wall-clock horizons scale with the TF (15m holds 2h, 4h "
          "holds 32h);\n  cost_R shows the fee squeeze growing as ATR shrinks. "
          "A plateau = structural;\n  a sharp 1h peak = fitting mark. The 1h "
          "track stays the registered one either way.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
