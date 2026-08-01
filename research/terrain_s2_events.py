# -*- coding: utf-8 -*-
"""S2 — distance/alignment to the last STRUCTURE EVENT (catalog #3, the
structure layer's last stand after S1 died and S3 fell at the wire).

Causal event definitions on confirmed swing pools (PIVOT=10, 1H bars):
  sweep of a swing level; the sweeping bar's CLOSE decides the type:
    假BOS  closes back inside  -> implied direction = REVERSAL of break
    BOS    closes beyond      -> implied direction = break direction
    CHoCH  a BOS whose direction flips the previous BOS's direction
Signals join the most recent event within 24 bars; alignment = signal
direction vs the event's implied direction.

Frozen predictions (TODO 0.484):
  P1 CHoCH 後順新結構 = best bucket
  P2 假BOS 後順原結構 (fade the sweep) = good
Gates: G1 buckets+halves on each named contrast; survivors -> G2 residual
inside D1-D3; -> G3 permutation/bootstrap/quarters. Thin cells reported,
never argued.

Run: python research/terrain_s2_events.py
Out: research/results/terrain_s2_events.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import sweep_core as SC  # noqa: E402
import level_types as LT  # noqa: E402
from v7_price_location_verify import build_rows  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/terrain_s2_events.json"
LOOK = 24


def structure_events(bars):
    """[(bar_j, kind, implied_dir, is_choch)] on swing pools only."""
    H, L, C = SC.H, SC.L, SC.C
    n = len(bars)
    h = [b[H] for b in bars]
    lo = [b[L] for b in bars]
    cl = [b[C] for b in bars]
    P = SC.PIVOT
    pools = []
    for i in range(P, n - P):
        seg = range(i - P, i + P + 1)
        if all(h[i] >= h[k] for k in seg) and any(h[i] > h[k] for k in seg if k != i):
            pools.append((i + P + 1, h[i], 1))
        if all(lo[i] <= lo[k] for k in seg) and any(lo[i] < lo[k] for k in seg if k != i):
            pools.append((i + P + 1, lo[i], -1))
    pools.sort()
    live = []
    pi = 0
    events = []
    last_bos_dir = 0
    for j in range(n):
        while pi < len(pools) and pools[pi][0] <= j:
            live.append(pools[pi])
            pi += 1
        for p in list(live):
            _e, lvl, s = p
            if (h[j] > lvl if s == 1 else lo[j] < lvl):
                live.remove(p)
                closed_back = (cl[j] <= lvl) if s == 1 else (cl[j] >= lvl)
                if closed_back:
                    events.append((j, "假BOS", -s, False))
                else:
                    choch = last_bos_dir != 0 and s != last_bos_dir
                    events.append((j, "BOS", s, choch))
                    last_bos_dir = s
    return events


def wr(g):
    return 100 * sum(r["c"] for r in g) / len(g) if g else None


def main() -> int:
    print("=" * 78)
    print("  S2 結構事件 — 假BOS/BOS/CHoCH 後的訊號對齊（結構層最後一張牌）")
    print("=" * 78)
    bars = SC.load_csv(str(LT.CACHE / "BTCUSDT_1h.csv"))
    ts2i = {b[0]: i for i, b in enumerate(bars)}
    evts = structure_events(bars)
    by_j = {}
    for (j, kind, imp, choch) in evts:
        by_j.setdefault(j, []).append((kind, imp, choch))
    rows = []
    for r in build_rows():
        j = ts2i.get(r["ts"])
        if j is None:
            continue
        ev = None
        for k in range(0, LOOK + 1):
            if j - k in by_j:
                ev = by_j[j - k][-1]
                break
        r2 = dict(r)
        if ev is None:
            r2["bk"] = "無事件"
        else:
            kind, imp, choch = ev
            al = "順" if ((r["dir"] == "UP") == (imp == 1)) else "逆"
            base = "CHoCH" if choch else kind
            r2["bk"] = f"{base}{al}"
        rows.append(r2)
    rows.sort(key=lambda r: r["ts"])
    n = len(rows)
    res = {}
    half = n // 2
    buckets = ["假BOS順", "假BOS逆", "BOS順", "BOS逆", "CHoCH順", "CHoCH逆",
               "無事件"]
    print(f"\n  [G1] 分桶（n={n}, 整體 {wr(rows):.0f}%）")
    for tag, seg in (("全期", rows), ("H1", rows[:half]), ("H2", rows[half:])):
        parts = []
        for b in buckets:
            g = [r for r in seg if r["bk"] == b]
            parts.append(f"{b} {wr(g):.0f}%({len(g)})" if len(g) >= 15
                         else f"{b} thin({len(g)})")
        print(f"  {tag:<4}" + " | ".join(parts))
        res[f"g1_{tag}"] = {b: [wr([r for r in seg if r["bk"] == b]),
                                len([r for r in seg if r["bk"] == b])]
                            for b in buckets}

    def contrast(name, good_b, bad_b):
        segs = {}
        for tag, seg in (("全期", rows), ("H1", rows[:half]), ("H2", rows[half:])):
            gg = [r for r in seg if r["bk"] == good_b]
            bb = [r for r in seg if r["bk"] == bad_b]
            segs[tag] = (wr(gg) - wr(bb)
                         if len(gg) >= 15 and len(bb) >= 15 else None)
        ds = segs["全期"], segs["H1"], segs["H2"]
        ok = (ds[0] is not None and ds[1] is not None and ds[2] is not None
              and ds[1] * ds[2] > 0 and abs(ds[0]) >= 4)
        show = " ".join("—" if d is None else f"{d:+.0f}" for d in ds)
        print(f"  [{name}] {good_b}−{bad_b}: {show} → {'PASS' if ok else 'FAIL'}")
        res[name] = {"deltas": ds, "pass": ok}
        return ok

    print()
    p1 = contrast("P1 CHoCH順 vs 其餘BOS", "CHoCH順", "BOS順")
    p1b = contrast("P1b CHoCH順 vs CHoCH逆", "CHoCH順", "CHoCH逆")
    p2 = contrast("P2 假BOS順 vs 假BOS逆", "假BOS順", "假BOS逆")
    res["any_pass"] = bool(p1 or p1b or p2)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                              default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    print("  （任一具名對比過 G1 才繼續 G2/G3；全滅=結構層蓋棺）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
