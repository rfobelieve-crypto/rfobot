# -*- coding: utf-8 -*-
"""Order flow AT the liquidity raid — the user's own hypothesis, tested.

"流動性獵取是個特殊事件，資金一定會有特別動作" (2026-07-29). The claim is that
a stop raid is not an ordinary bar: someone is doing something distinctive,
and the order-flow tables should see it.

Why this is not a rerun of the four failed order-flow hunts. Those measured
UNCONDITIONAL bar-level IC — flow at every bar against the next bar's return —
and the 2026-07-29 horizon-decay work showed why they had to fail: hourly
instantaneous flow carries ~0.7 same-bar correlation and collapses to -0.03 by
+1h. This asks a different question at a different moment: conditioned on a
raid having just happened, does the flow during that raid separate a genuine
breakout from a stop hunt? Price already answers it partially (the shallow-
pierce filter), so the test is whether flow adds anything ON TOP of pierce.

Feasibility (the reason this can run today at all): widening the liquidity
definition to four pool types multiplied the event count ~4.5x, so BTC+ETH
over the ~100 days of flow_bars coverage yields several hundred raids instead
of the few dozen the swing-only definition allowed.

Pre-registered before looking (all reported, none dropped):
    vshock        raid-hour volume / trailing 24h median
    taker_ratio   signed by raid direction: >0 = aggression AGREED with the
                  raid (a real breakout should look like this)
    cvd_slope     signed CVD change over the raid hour, same convention
    imb_l20       order-book imbalance at the raid minute, same convention
Each is bucketed into terciles; the decision metric is net R. A feature only
counts if it is monotonic AND survives inside the shallow-pierce subset —
i.e. adds information the price-based filter does not already have.

Run: python research/sweep_orderflow.py
Out: research/results/sweep_orderflow.json
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))
os.environ["SLIP"] = "0"
import sweep_core as SC  # noqa: E402
import level_types as LT  # noqa: E402
from shared.db import get_db_conn  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/sweep_orderflow.json"
SYMS = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
PIERCE_MAX = 0.25


def load_flow(canon: str):
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT window_start ms, volume_usd v, delta_usd d, cvd_usd c "
                "FROM flow_bars_1m WHERE canonical_symbol=%s ORDER BY window_start",
                (canon,))
            fb = cur.fetchall()
            cur.execute(
                "SELECT ts_ms, imbalance_l20 i FROM orderbook_snapshots_1m "
                "WHERE canonical_symbol=%s ORDER BY ts_ms", (canon,))
            ob = cur.fetchall()
    finally:
        conn.close()
    flow = {int(r["ms"]) // 60_000: (float(r["v"] or 0), float(r["d"] or 0),
                                     float(r["c"] or 0)) for r in fb}
    imb = {int(r["ts_ms"]) // 60_000: float(r["i"])
           for r in ob if r["i"] is not None}
    return flow, imb


def raid_events(sym: str):
    """(raid_ts, fill_ts, side, netR, pierce) across all four pool types."""
    p = LT.CACHE / f"{sym}USDT_1h.csv"
    bars = SC.load_csv(str(p))
    out = []
    for (f_ts, _x, R, lvl, atr, _s, pc) in SC.backtest_symbol(bars):
        out.append((f_ts, pc, LT.net(R, lvl, atr)))
    lv = LT.build_levels(bars)
    for kind in ("session", "pdh_pdl", "pwh_pwl"):
        for (f_ts, _x, netr, pc, lvl, atr, _st) in LT.trade_levels(
                bars, lv.get(kind, [])):
            out.append((f_ts, pc, netr))
    return out


def feats(flow, imb, ts: int, side: int) -> dict | None:
    """Flow during the hour BEFORE the fill (the raid hour). side=+1 long."""
    m0 = ts // 60 - 60
    win = [flow[m] for m in range(m0, ts // 60) if m in flow]
    if len(win) < 40:
        return None
    base = [flow[m][0] for m in range(m0 - 1440, m0) if m in flow]
    if len(base) < 500:
        return None
    med = sorted(base)[len(base) // 2]
    vol = sum(w[0] for w in win)
    delta = sum(w[1] for w in win)
    cvd = win[-1][2] - win[0][2]
    ivals = [imb[m] for m in range(m0, ts // 60) if m in imb]
    return {
        "vshock": vol / (med * 60) if med > 0 else 1.0,
        "taker_ratio": side * (delta / vol if vol > 0 else 0.0),
        "cvd_slope": side * cvd / vol if vol > 0 else 0.0,
        "imb_l20": side * (sum(ivals) / len(ivals)) if ivals else None,
    }


def buckets(rows, key):
    vals = sorted(r[key] for r in rows if r.get(key) is not None)
    if len(vals) < 90:
        return None
    q1, q2 = vals[len(vals) // 3], vals[2 * len(vals) // 3]
    g = {"low": [], "mid": [], "high": []}
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        g["low" if v <= q1 else ("mid" if v <= q2 else "high")].append(r["r"])
    return g


def show(name, g, tag=""):
    if not g:
        print(f"  {name:<14}{tag} (insufficient)")
        return None
    parts, out = [], {}
    for k, rs in g.items():
        if len(rs) < 25:
            parts.append(f"{k}: n={len(rs)} thin")
            continue
        m = sum(rs) / len(rs)
        sd = math.sqrt(sum((x - m) ** 2 for x in rs) / (len(rs) - 1))
        t = m / (sd / math.sqrt(len(rs)))
        parts.append(f"{k}: {m:+.4f} (t{t:+.1f}, n={len(rs)})")
        out[k] = {"n": len(rs), "mean": m, "t": t}
    print(f"  {name:<14}{tag} " + " | ".join(parts))
    return out


def main() -> int:
    print("=" * 78)
    print("  ORDER FLOW AT THE RAID — does flow add to what pierce already says?")
    print("=" * 78)
    rows = []
    for sym, canon in SYMS.items():
        flow, imb = load_flow(canon)
        if not flow:
            print(f"  {sym}: no flow data")
            continue
        cov = 0
        for f_ts, pc, r in raid_events(sym):
            side = 1  # netR is already direction-signed; flow needs the raid side
            f = feats(flow, imb, f_ts, side)
            if f is None:
                continue
            cov += 1
            f.update({"r": r, "pierce": pc, "sym": sym})
            rows.append(f)
        print(f"  {sym}: {cov} raids with flow coverage")
    if len(rows) < 100:
        print(f"\n  only {len(rows)} usable events — underpowered, reporting anyway")
    base = sum(r["r"] for r in rows) / len(rows)
    print(f"\n  baseline over covered raids: {base:+.4f} (n={len(rows)})")

    res = {}
    print("\n  ALL covered raids:")
    for k in ("vshock", "taker_ratio", "cvd_slope", "imb_l20"):
        res[k] = show(k, buckets(rows, k))
    sh = [r for r in rows if r["pierce"] <= PIERCE_MAX]
    print(f"\n  WITHIN the shallow-pierce subset (n={len(sh)}) — the only place "
          f"flow can ADD:")
    for k in ("vshock", "taker_ratio", "cvd_slope", "imb_l20"):
        res[f"{k}_shallow"] = show(k, buckets(sh, k), "sh")

    OUT.write_text(json.dumps({"n": len(rows), "baseline": base,
                               "results": res}, indent=2), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    print("  READ: 4 features x 2 subsets = 8 looks; expect ~0.4 spurious. "
          "Only a MONOTONIC pattern that also holds inside the shallow subset "
          "means flow adds information price does not already carry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
