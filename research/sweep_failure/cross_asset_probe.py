# -*- coding: utf-8 -*-
"""Cross-asset probe — run the FROZEN sweep-failure rules on non-crypto 1H.

Why: the portfolio direction's ceiling is "all-crypto shares one beta"
(TODO 0.4). The README already carries a seed (MNQ 15m same-direction,
+0.205 ATR, t=1.31). This runs the frozen engine (PIVOT=10/W=8/HOLD=8/
DIS=3.5 — untouched) on index futures / gold / FX at 1H via Yahoo.

Discipline:
  * rules frozen — zero tuning, zero symbol cherry-picking (report all)
  * gross R first; net under two flat cost lines per side (futures 1.5 bps,
    FX 1.0 bps) plus the crypto scenario-A bps as an over-conservative
    stress — all converted per trade through that trade's own ATR
  * honest caveats printed: Yahoo continuous futures carry roll gaps that
    can fabricate sweeps (ES/NQ quarterly, GC ~monthly); session breaks
    mean W/HOLD count TRADING bars not wall-clock hours; 730d is Yahoo's
    1h history cap.

Run: python research/sweep_failure/cross_asset_probe.py
Out: research/results/sweep_cross_asset.json
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
os.environ["SLIP"] = "0"                     # gross engine; costs applied here
import sweep_core as SC                       # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd                           # noqa: E402
import yfinance as yf                         # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "research/results/sweep_cross_asset.json"
CACHE = HERE / ".cache" / "xasset"

SYMS = {                                      # name -> (yahoo ticker, per-side bps)
    "NQ":     ("NQ=F", 1.5),
    "ES":     ("ES=F", 1.5),
    "GOLD":   ("GC=F", 1.5),
    "EURUSD": ("EURUSD=X", 1.0),
}
SCEN_A = {"entry": 7.0, "texit": 3.0, "sexit": 10.0}   # crypto stress line


def fetch(name: str, ticker: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{name}_1h.csv"
    df = yf.download(ticker, interval="1h", period="730d",
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"yahoo empty for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    tcol = df.columns[0]
    out = pd.DataFrame({
        "time": (pd.to_datetime(df[tcol], utc=True).astype("int64") // 10**9),
        "open": df["Open"], "high": df["High"],
        "low": df["Low"], "close": df["Close"],
        "volume": df.get("Volume", 0),
    }).dropna()
    out.to_csv(p, index=False)
    return p


def stats(rs: list[float]) -> dict:
    n = len(rs)
    if n < 30:
        return {"n": n}
    m = sum(rs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in rs) / (n - 1))
    t = m / (sd / math.sqrt(n)) if sd > 0 else 0.0
    wr = 100.0 * sum(1 for x in rs if x > 0) / n
    half = n // 2
    return {"n": n, "mean": m, "t": t, "wr": wr,
            "h1": sum(rs[:half]) / max(half, 1),
            "h2": sum(rs[half:]) / max(n - half, 1)}


def main() -> int:
    print("=" * 76)
    print("  SWEEP-FAILURE x CROSS-ASSET — frozen rules, Yahoo 1H (730d cap)")
    print("=" * 76)
    rows = []
    pool_net = []
    for name, (ticker, bps_side) in SYMS.items():
        try:
            p = fetch(name, ticker)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:<7} fetch failed: {e}")
            continue
        bars = SC.load_csv(str(p))
        trades = SC.backtest_symbol(bars)
        gross = [t[2] for t in trades]
        net_flat, net_cryA = [], []
        for _, _, r, lvl, atr, stopped in trades:
            net_flat.append(r - (2 * bps_side) / 1e4 * lvl / (SC.DIS * atr))
            legs = SCEN_A["entry"] + (SCEN_A["sexit"] if stopped else SCEN_A["texit"])
            net_cryA.append(r - legs / 1e4 * lvl / (SC.DIS * atr))
        g, f, c = stats(gross), stats(net_flat), stats(net_cryA)
        rows.append({"sym": name, "bars": len(bars), "gross": g,
                     "net_flat": f, "net_cryptoA": c})
        pool_net += net_flat
        if "mean" in g:
            print(f"  {name:<7} bars={len(bars):>6}  n={g['n']:>4}  "
                  f"gross {g['mean']:+.4f} (t{g['t']:+.2f} WR{g['wr']:.0f}%)  "
                  f"net@{bps_side}bps/side {f['mean']:+.4f} (t{f['t']:+.2f})  "
                  f"cryA {c['mean']:+.4f}  halves {g['h1']:+.3f}/{g['h2']:+.3f}")
        else:
            print(f"  {name:<7} bars={len(bars):>6}  n={g['n']} (<30, skip stats)")
    pm = stats(pool_net)
    if "mean" in pm:
        print(f"\n  POOL(non-crypto, net@flat)  n={pm['n']}  mean {pm['mean']:+.4f}"
              f"  t={pm['t']:+.2f}  WR {pm['wr']:.0f}%  "
              f"halves(symbol-ordered, do not cite) {pm['h1']:+.3f}/{pm['h2']:+.3f}")

    # ── forward cohort (informational, NOT part of Gate F) ────────────
    # Same freeze date as the crypto rules commit (2026-07-28). Purpose:
    # let the cross-asset sleeve accumulate a forward track record at zero
    # cost, so that IF crypto Gate F passes and capital reaches futures
    # scale, the sleeve arrives with a year of out-of-sample evidence
    # instead of starting cold. Monthly re-run alongside sweep_forward.
    from datetime import datetime, timezone
    FREEZE = int(datetime(2026, 7, 28, tzinfo=timezone.utc).timestamp())
    fwd_pool = []
    print("\n  FORWARD cohort (freeze 2026-07-28, informational only):")
    for row in rows:
        name = row["sym"]
        bps_side = SYMS[name][1]
        p = CACHE / f"{name}_1h.csv"
        trades = SC.backtest_symbol(SC.load_csv(str(p)))
        rs = [t[2] - (2 * bps_side) / 1e4 * t[3] / (SC.DIS * t[4])
              for t in trades if t[0] >= FREEZE]
        fwd_pool += rs
        if rs:
            print(f"    {name:<7} n={len(rs):>3}  sumR={sum(rs):+7.3f}")
    if fwd_pool:
        m = sum(fwd_pool) / len(fwd_pool)
        print(f"    pool    n={len(fwd_pool):>3}  meanR={m:+.4f}")
    else:
        print("    (no forward trades yet)")
    print("\n  caveats: Yahoo continuous-contract roll gaps can fabricate "
          "sweeps (ES/NQ quarterly, GC ~monthly; EURUSD none); session "
          "breaks make W/HOLD trading-bars not wall-clock; correlated "
          "indices (NQ/ES) inflate any pooled t — read per-symbol first.")
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
