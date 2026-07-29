# -*- coding: utf-8 -*-
"""Sweep-failure — adversarial fill/robustness diagnostics (crypto + cross-asset).

Three ways this backtest could be manufacturing an edge, none checked when
the results were produced:

  1 UNFILLABLE GAP ENTRY. The entry is modelled at the swept level. If price
    crossed that level between bars without trading at it, the modelled fill
    never existed and the real fill is worse.

    CORRECT TEST (v2, 2026-07-29): a gap requires the PREVIOUS bar to sit
    entirely on the far side and this bar to open past the level:
        kd=+1 (swept high, entering short on the way back down)
            gap  <=>  close[f-1] > lvl AND open[f] <= lvl
    The v1 test used only `open[f] <= lvl`, which also flags the common and
    perfectly fillable case where the SWEEP bar itself reversed and closed
    back under the level — that flagged ~60% of crypto entries as gaps in a
    market that trades continuously and has 0% real gaps, which is what
    exposed the bug. Any diagnostic that fires on a 24/7 market the same way
    it fires on session-traded futures is measuring itself, not the data.

  2 ILLIQUID-HOUR ENTRIES (futures only; crypto is 24/7). An edge living in
    3am prints is not tradeable at the modelled 1.5-2 bps.

  3 OUTLIER CONCENTRATION. mean vs median vs 5% trimmed mean, plus the share
    of total R contributed by the best 1% of trades.

Run: python research/sweep_failure/xasset_diagnostics.py [--crypto]
Out: research/results/xasset_diagnostics.json / crypto_fill_diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
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

RESULTS = Path(__file__).resolve().parents[2] / "research/results"
XA_CACHE = HERE / ".cache" / "xasset"
CR_CACHE = HERE / ".cache"
XA_SYMS = {"NQ": 1.5, "ES": 1.5, "YM": 1.5, "RTY": 1.5, "NIKKEI": 2.0,
           "GOLD": 1.5, "SILVER": 2.0, "COPPER": 2.0, "CRUDE": 1.5,
           "NATGAS": 2.0, "UST10Y": 1.0, "UST30Y": 1.0,
           "EURUSD": 1.0, "USDJPY": 1.0, "GBPUSD": 1.0, "AUDUSD": 1.0}
CR_SYMS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX",
           "TRX", "DOT", "LTC", "UNI", "ATOM", "ETC", "NEAR", "APT", "FIL",
           "ARB", "OP", "INJ", "SUI", "AAVE", "ICP", "ALGO", "VET", "HBAR",
           "SAND", "AXS"]
LIQUID_UTC = set(range(7, 22))


def annotate(path: Path, is_crypto: bool):
    if not path.exists():
        return []
    bars = SC.load_csv(str(path))
    n = len(bars)
    H, L, C, O = SC.H, SC.L, SC.C, SC.O
    h = [b[H] for b in bars]
    lo = [b[L] for b in bars]
    op = [b[O] for b in bars]
    cl = [b[C] for b in bars]
    a = SC.atr14(bars)
    out, last_exit = [], -1
    for e in SC.detect_sweeps(bars):
        j, lvl = e["j"], e["level"]
        if a[j] is None or a[j] == 0:
            continue
        kd = 1 if e["kind"] == "buy" else -1
        d = -kd
        fill = None
        for f in range(j + 1, min(j + 1 + SC.W, n)):
            if (kd == 1 and lo[f] <= lvl) or (kd == -1 and h[f] >= lvl):
                fill = f
                break
        if fill is None or fill <= last_exit or fill + 1 >= n:
            continue
        A = a[j]
        risk = SC.DIS * A
        stop = lvl - d * risk
        R, exitbar = None, min(fill + SC.HOLD, n - 1)
        for k in range(fill + 1, min(fill + SC.HOLD + 1, n)):
            if (d == 1 and lo[k] <= stop) or (d == -1 and h[k] >= stop):
                R, exitbar = -1.0, k
                break
        if R is None:
            R = d * (cl[exitbar] - lvl) / risk
        last_exit = exitbar
        # v2 gap test: previous bar entirely on the far side AND this one opens past
        prev_far = (cl[fill - 1] > lvl) if kd == 1 else (cl[fill - 1] < lvl)
        opens_past = (op[fill] <= lvl) if kd == 1 else (op[fill] >= lvl)
        gap_entry = bool(prev_far and opens_past)
        hr = datetime.fromtimestamp(bars[fill][0], tz=timezone.utc).hour
        out.append({"R": R, "lvl": lvl, "atr": A, "gap_entry": gap_entry,
                    "liquid": True if is_crypto else (hr in LIQUID_UTC)})
    return out


def net(t, bps):
    return t["R"] - 2 * bps / 1e4 * t["lvl"] / (SC.DIS * t["atr"])


def stat(rs):
    n = len(rs)
    if n < 20:
        return None
    m = sum(rs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in rs) / (n - 1))
    s = sorted(rs)
    k = max(1, n // 20)
    return {"n": n, "mean": m, "t": m / (sd / math.sqrt(n)),
            "median": s[n // 2], "trimmed": sum(s[k:n - k]) / (n - 2 * k)}


def run(label: str, items, cache: Path, is_crypto: bool, out_name: str):
    print("=" * 78)
    print(f"  {label} — fill/robustness diagnostics (v2 gap test)")
    print("=" * 78)
    print(f"  {'sym':<8}{'n':>6}{'gapEntry%':>11}{'illiq%':>8}"
          f"{'net t all':>11}{'net t clean':>13}")
    res, all_pool, clean_pool = {}, [], []
    for sym, bps in items:
        p = cache / (f"{sym}_1h.csv" if not is_crypto else f"{sym}USDT_1h.csv")
        tr = annotate(p, is_crypto)
        if not tr:
            continue
        rs_all = [net(t, bps) for t in tr]
        clean = [t for t in tr if not t["gap_entry"] and t["liquid"]]
        rs_cl = [net(t, bps) for t in clean]
        sa, sc = stat(rs_all), stat(rs_cl)
        all_pool += rs_all
        clean_pool += rs_cl
        gp = 100 * sum(t["gap_entry"] for t in tr) / len(tr)
        il = 100 * sum(not t["liquid"] for t in tr) / len(tr)
        res[sym] = {"all": sa, "clean": sc, "gap_pct": gp, "illiquid_pct": il}
        print(f"  {sym:<8}{len(tr):>6}{gp:>11.1f}{il:>8.0f}"
              f"{(sa['t'] if sa else float('nan')):>11.2f}"
              f"{(sc['t'] if sc else float('nan')):>13.2f}")
    pa, pc = stat(all_pool), stat(clean_pool)
    for tag, s in (("all", pa), ("clean", pc)):
        print(f"\n  POOLED {tag:<6} n={s['n']:<6} mean {s['mean']:+.4f}  "
              f"median {s['median']:+.4f}  5%-trimmed {s['trimmed']:+.4f}  "
              f"t={s['t']:+.2f}")
    srt = sorted(all_pool, reverse=True)
    tot = sum(all_pool)
    print(f"\n  outlier: top 1% of trades = "
          f"{100*sum(srt[:max(1,len(srt)//100)])/tot:.0f}% of total R")
    (RESULTS / out_name).write_text(json.dumps(
        {"per_symbol": res, "pooled_all": pa, "pooled_clean": pc}, indent=2),
        encoding="utf-8")
    print(f"  wrote {RESULTS / out_name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crypto", action="store_true")
    a = ap.parse_args()
    if a.crypto:
        run("CRYPTO 29", [(s, 5.0) for s in CR_SYMS], CR_CACHE, True,
            "crypto_fill_diagnostics.json")
    else:
        run("CROSS-ASSET 16", list(XA_SYMS.items()), XA_CACHE, False,
            "xasset_diagnostics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
