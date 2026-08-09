"""How much money can the sweep-failure line actually absorb?

Context (2026-08-09): the sizing Monte Carlo showed capacity is THE parameter
— same 2% risk, same year, median outcome moved 180k -> 812k -> 32bn purely on
the capacity assumption. So the number is worth measuring, not assuming.

WHAT LIMITS SIZE HERE
  Entries are stop-hunt reversals: they fire on a wick through a level, in a
  1h bar, on 29 mostly mid-cap perps. The binding constraint is not exchange
  liquidity in the abstract — it is how much you can push through in the few
  minutes around the fill without moving price a meaningful fraction of the
  edge. The edge is thin: +0.079R per trade with a stop at ~1 ATR, so ATR% is
  the natural yardstick.

METHOD (deliberately crude, and labelled as such)
  For every genuinely-prospective variant-B trade we know: symbol, fill hour,
  entry price, ATR. We add hourly traded volume (Binance USDT-perp klines) and
  compute, per trade:

      notional = risk_usd / atr_pct          (stop ~= 1 ATR, so risk = notional*atr_pct)
      participation = notional / hourly_volume_usd

  Then invert: what risk_usd keeps participation under a chosen ceiling?

  Ceilings reported: 0.5% / 1% / 2% of the hour's volume. These are rules of
  thumb, NOT measured impact — the honest reading is the ORDER OF MAGNITUDE
  and the RANKING between coins, not the exact dollar.

  The portfolio number then accounts for concurrency: the binding case is the
  worst hour (up to 8 simultaneous positions under the frozen cap), so the
  account-level capacity is driven by the *median* per-trade capacity across
  the coins actually traded, times the concurrency the executor allows.

WHAT THIS DOES NOT MODEL
  Real impact curves, order-book depth at the exact level, the fact that a
  stop-hunt wick is exactly when liquidity is thinnest, maker/taker mix, and
  the possibility that our own order changes whether the sweep "fails". All of
  these push the true capacity DOWN, so treat the output as an upper bound.

Usage:  python research/sweep_capacity_estimate.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOG = ROOT / "research" / "results" / "sweep_shadow_log.csv"
CACHE = ROOT / "research" / "results" / "sweep_capacity_volumes.json"
PARTICIPATION = (0.005, 0.01, 0.02)
CONCURRENCY = 8


def load_trades():
    rows = [r for r in csv.DictReader(LOG.open(encoding="utf-8"))
            if r.get("variant_b") == "1" and r.get("status") == "CLOSED"
            and r.get("first_seen_utc") and r.get("exit_utc")
            and r["first_seen_utc"] < r["exit_utc"]]
    out = []
    for r in rows:
        try:
            px, atr = float(r["entry_px"]), float(r["atr"])
            if px > 0 and atr > 0:
                out.append({"sym": r["symbol"], "hour": r["fill_utc"][:13],
                            "atr_pct": atr / px})
        except (ValueError, KeyError):
            continue
    return out


def fetch_hourly_volume(symbols: list[str]) -> dict:
    """Median hourly quote volume per symbol (30d), cached."""
    if CACHE.exists():
        cached = json.loads(CACHE.read_text())
        if all(s in cached for s in symbols):
            return cached
    vols = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    for s in symbols:
        if s in vols:
            continue
        url = ("https://fapi.binance.com/fapi/v1/klines?symbol="
               f"{s}USDT&interval=1h&limit=720")
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                data = json.loads(resp.read().decode())
            qv = [float(k[7]) for k in data]      # quote asset volume
            vols[s] = {"median": float(np.median(qv)),
                       "p10": float(np.percentile(qv, 10))}
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {s}: {str(e)[:60]}")
            vols[s] = None
        time.sleep(0.12)
    CACHE.write_text(json.dumps(vols, indent=1))
    return vols


def main() -> int:
    trades = load_trades()
    per_sym = defaultdict(list)
    for t in trades:
        per_sym[t["sym"]].append(t["atr_pct"])
    syms = sorted(per_sym, key=lambda s: -len(per_sym[s]))
    print(f"變體 B 真前瞻 {len(trades)} 筆 · {len(syms)} 幣")
    print("抓各幣近 30 天每小時成交額（Binance USDT 永續）…")
    vols = fetch_hourly_volume(syms)

    print(f"\n每筆容量 = 參與率上限 × 該小時成交額 × ATR%")
    print(f"（停損 ~1 ATR，所以 風險$ = 名目 × ATR%）\n")
    hdr = (f"{'幣':<7}{'筆數':>5}{'ATR%':>7}{'1h成交額(中位)':>16}"
           f"{'風險$@0.5%':>12}{'@1%':>10}{'@2%':>10}")
    print(hdr)
    print("-" * len(hdr))
    caps = {p: [] for p in PARTICIPATION}
    for s in syms:
        v = vols.get(s)
        if not v:
            continue
        atr = float(np.median(per_sym[s]))
        row = f"{s:<7}{len(per_sym[s]):>5}{atr*100:>6.2f}%${v['median']:>14,.0f}"
        for p in PARTICIPATION:
            risk = v["median"] * p * atr
            caps[p].append(risk)
            row += f"${risk:>9,.0f}"
        print(row)

    print("\n── 帳戶層容量（並發 %d 筆，取各幣中位）──" % CONCURRENCY)
    for p in PARTICIPATION:
        c = np.array(caps[p])
        med, low = float(np.median(c)), float(np.percentile(c, 25))
        print(f"  參與率 ≤{p*100:>4.1f}%：每筆風險上限 中位 ${med:,.0f} "
              f"（保守取 p25 ${low:,.0f}）")
        for rf in (0.01, 0.02):
            print(f"      → 若每筆風險 {rf*100:.0f}%，可承載本金約 "
                  f"${med/rf:,.0f}（保守 ${low/rf:,.0f}）")
    print("\n⚠ 這是上界：未計入 wick 當下深度最薄、自身單會改變『掃單是否失敗』、")
    print("   真實衝擊曲線非線性。參與率上限本身是經驗法則，不是量測值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
