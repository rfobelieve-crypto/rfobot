# -*- coding: utf-8 -*-
"""Raid microstructure round 4 — flow measured ONLY in the attack minutes.

User's critique (2026-07-30, mid-round-3): 一小時級別標記出流動性沒錯，但
獵取只需要抓那幾分鐘，不然訂單流的優勢會衰退. Round 3 aggregated the whole
raid HOUR; if the attack itself lasts ~5 minutes, that dilutes the flow
signature ~10x — the same aggregation sin the horizon-decay work exposed.

So: fetch 1m klines (same venue as the hourly cache), locate the EXACT
minutes where price traded beyond the level inside the raid hour (the
attack window), and measure flow there and only there.

Features (attack window only; all knowable by the raid hour's close, i.e.
strictly prior to any retest entry):
  att_min          attack duration in minutes beyond the level
  att_taker        s * taker delta / volume DURING the attack minutes
  att_vshock       attack per-minute volume / trailing-24h median minute vol
  reject_in_hour   did 1m price re-cross back inside before the hour closed
  post_rej_taker   s * delta/vol in the <=15 minutes AFTER the re-cross
                   (within the raid hour; None if no in-hour rejection)

Same targets (BREAKOUT vs retest; netR when retested), same bar: monotone +
both symbols + magnitude. BTC+ETH, ~100d of 1m flow coverage.

Run: python research/sweep_raid_attack.py
Out: research/results/sweep_raid_attack.json
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import sweep_raid_anatomy as A  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/sweep_raid_attack.json"
M1_CACHE = ROOT / "research" / "sweep_failure" / ".cache" / "m1"
DAYS = 110


def fetch_1m(sym: str) -> dict[int, tuple]:
    """minute -> (high, low, close). Cached CSV; Binance public REST."""
    M1_CACHE.mkdir(parents=True, exist_ok=True)
    p = M1_CACHE / f"{sym}_1m.csv"
    if not p.exists():
        end = int(time.time() * 1000)
        cur = end - DAYS * 86400 * 1000
        rows = {}
        while cur < end:
            req = urllib.request.Request(
                "https://api.binance.com/api/v3/klines"
                f"?symbol={sym}USDT&interval=1m&startTime={cur}&limit=1000",
                headers={"User-Agent": "raid-attack/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.loads(r.read().decode())
            if not d:
                break
            for k in d:
                if int(k[6]) > end:
                    continue
                rows[int(k[0]) // 60_000] = (float(k[2]), float(k[3]),
                                             float(k[4]))
            cur = int(d[-1][0]) + 60_000
            if len(d) < 1000:
                break
            time.sleep(0.04)
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["minute", "high", "low", "close"])
            for m in sorted(rows):
                w.writerow([m, *rows[m]])
    out = {}
    with p.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[int(r["minute"])] = (float(r["high"]), float(r["low"]),
                                     float(r["close"]))
    return out


def attach_attack(rows, flow, m1):
    out = []
    for r in rows:
        m0 = r["ts"] // 60
        s = r["side"]
        lvl = r.get("lvl")
        if lvl is None:
            continue
        mins = [(m, *m1[m]) for m in range(m0, m0 + 60) if m in m1]
        if len(mins) < 50:
            continue
        beyond = [m for (m, hi, lo_, _c) in mins
                  if (hi > lvl if s == 1 else lo_ < lvl)]
        if not beyond:
            continue
        first = beyond[0]
        # rejection: first minute AFTER the last beyond-minute whose close is
        # back inside; within the hour only
        last_beyond = beyond[-1]
        rej_min = None
        for (m, _hi, _lo, c) in mins:
            if m > last_beyond and ((c <= lvl) if s == 1 else (c >= lvl)):
                rej_min = m
                break
        fwin = [flow[m] for m in beyond if m in flow]
        base = [flow[m][0] for m in range(m0 - 1440, m0) if m in flow]
        if len(fwin) < max(1, len(beyond) // 2) or len(base) < 500:
            continue
        med = sorted(base)[len(base) // 2]
        vol = sum(w[0] for w in fwin)
        f = dict(r)
        f["att_min"] = len(beyond)
        f["att_taker"] = s * sum(w[1] for w in fwin) / vol if vol > 0 else None
        f["att_vshock"] = (vol / len(fwin)) / med if med > 0 else None
        f["reject_in_hour"] = int(rej_min is not None)
        if rej_min is not None:
            pw = [flow[m] for m in range(rej_min, min(rej_min + 15, m0 + 60))
                  if m in flow]
            pv = sum(w[0] for w in pw)
            f["post_rej_taker"] = (s * sum(w[1] for w in pw) / pv
                                   if pv > 0 else None)
        else:
            f["post_rej_taker"] = None
        out.append(f)
    return out


def raids_with_level(sym: str):
    """A.raids() carries no level; re-derive with it (same logic)."""
    import sweep_core as SC
    import level_types as LT
    bars = SC.load_csv(str(LT.CACHE / f"{sym}USDT_1h.csv"))
    n = len(bars)
    H, L, C = SC.H, SC.L, SC.C
    h = [b[H] for b in bars]
    lo = [b[L] for b in bars]
    cl = [b[C] for b in bars]
    a = SC.atr14(bars)
    pools = []
    P = SC.PIVOT
    for i in range(P, n - P):
        seg = range(i - P, i + P + 1)
        if all(h[i] >= h[k] for k in seg) and any(h[i] > h[k] for k in seg if k != i):
            pools.append((i + P + 1, h[i], 1))
        if all(lo[i] <= lo[k] for k in seg) and any(lo[i] < lo[k] for k in seg if k != i):
            pools.append((i + P + 1, lo[i], -1))
    lv = LT.build_levels(bars)
    for kind in ("session", "pdh_pdl", "pwh_pwl"):
        pools += list(lv.get(kind, []))
    out = []
    pending = sorted(pools)
    idx = 0
    live = []
    for j in range(n - SC.W - SC.HOLD - 1):
        while idx < len(pending) and pending[idx][0] <= j:
            live.append(pending[idx][1:])
            idx += 1
        if a[j] is None or a[j] == 0:
            continue
        hit = [t for t in live if (h[j] > t[0] if t[1] == 1 else lo[j] < t[0])]
        if not hit:
            continue
        live = [t for t in live if t not in hit]
        for lvl, s in hit:
            kd, d = s, -s
            pierce = (h[j] - lvl if kd == 1 else lvl - lo[j]) / a[j]
            fill = None
            for f in range(j + 1, j + 1 + SC.W):
                if (kd == 1 and lo[f] <= lvl) or (kd == -1 and h[f] >= lvl):
                    fill = f
                    break
            rec = {"sym": sym, "ts": bars[j][0], "pierce": pierce,
                   "side": kd, "lvl": lvl}
            if fill is None:
                rec.update({"cls": "BREAKOUT", "netR": None})
            else:
                A_ = a[j]
                stop = lvl - d * SC.DIS * A_
                R, xb = None, min(fill + SC.HOLD, n - 1)
                for k in range(fill + 1, min(fill + SC.HOLD + 1, n)):
                    if (d == 1 and lo[k] <= stop) or (d == -1 and h[k] >= stop):
                        R, xb = -1.0, k
                        break
                if R is None:
                    R = d * (cl[xb] - lvl) / (SC.DIS * A_)
                import level_types as LT2
                net = LT2.net(R, lvl, A_)
                rec.update({"cls": "REVERSAL" if net > 0 else "FAKE_RETEST",
                            "netR": net})
            out.append(rec)
    return out


def main() -> int:
    print("=" * 78)
    print("  RAID ATTACK-WINDOW — 流只量攻擊那幾分鐘（使用者批評的直接實作）")
    print("=" * 78)
    allr = []
    for sym, canon in A.SYMS.items():
        m1 = fetch_1m(sym)
        flow, _i, _c = A.load_flow(canon)
        rr = attach_attack(raids_with_level(sym), flow, m1)
        print(f"  {sym}: {len(rr)} raids with attack-window coverage")
        allr += rr
    durs = sorted(r["att_min"] for r in allr)
    print(f"\n  攻擊時長分佈: 中位 {durs[len(durs)//2]} 分鐘, "
          f"p90 {durs[int(0.9*len(durs))]} 分鐘 — 確認「獵取只有幾分鐘」")
    res = {}
    print("\n  [terciles] 突破率 / 反轉率 / netR|回踩")
    for k in ("att_min", "att_taker", "att_vshock", "post_rej_taker"):
        rec, line = A.profile(allr, k)
        res[k] = rec
        print(line)
    # reject_in_hour is binary
    from collections import defaultdict
    g = defaultdict(list)
    for r in allr:
        g[r["reject_in_hour"]].append(r)
    line = []
    for k in sorted(g):
        rs = g[k]
        n = len(rs)
        br = 100 * sum(1 for x in rs if x["cls"] == "BREAKOUT") / n
        nets = [x["netR"] for x in rs if x["netR"] is not None]
        m = sum(nets) / len(nets) if nets else float("nan")
        res[f"reject_{k}"] = {"n": n, "breakout_pct": br, "netR": m}
        line.append(f"{'小時內被拒' if k else '收在外面'}: 突破{br:.0f}% "
                    f"netR{m:+.3f} (n={n})")
    print("  reject_in_hour " + " | ".join(line))
    print("\n  [split] BTC vs ETH")
    for k in ("att_min", "att_taker", "att_vshock"):
        for sym in A.SYMS:
            _, line = A.profile([r for r in allr if r["sym"] == sym], k)
            print(f"  {sym:<4}" + line)
    OUT.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
