# -*- coding: utf-8 -*-
"""V7 raid-chase veto — robustness deep-dive before any registration.

Context (2026-08-02): v7_raid_context.py found live Strong signals that
FOLLOW a fresh raid break win 52% vs 64% with no recent raid, gradient
consistent in both halves. Before this becomes a registered veto rule it
has to survive the slicing that killed weaker ideas:

  1. lookback sensitivity  — 2/4/6/8h windows (4h is the named default;
     the others show whether the effect is a cliff or a curve)
  2. direction split       — UP vs DOWN signals (the known 6pp asymmetry
     must not be the whole story)
  3. regime split          — the veto must not be a regime proxy
  4. thirds stability      — three time slices, gradient direction each
  5. counterfactual        — portfolio effect of applying the veto:
     kept-WR vs vetoed-WR, and the signal-count cost
  6. actual_return_4h view — average realized 4h return per bucket (the
     money-adjacent number, not just hit rate)

`--clock` prints one line for the weekly PortfolioClocks report: trailing
90d kept vs vetoed WR — the forward-confirmation clock that must stay
separated before production adoption is discussed.

Run: python research/v7_raid_veto.py [--clock]
Out: research/results/v7_raid_veto.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

from shared.db import get_db_conn  # noqa: E402
from sweep_raid_postflow import raids_with_fill  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/v7_raid_veto.json"


def load():
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT signal_time, direction, regime, correct, "
                "actual_return_4h FROM tracked_signals "
                "WHERE strength='Strong' AND correct IS NOT NULL "
                "ORDER BY signal_time")
            sigs = cur.fetchall()
    finally:
        conn.close()
    by_hh: dict[int, list] = defaultdict(list)
    for r in raids_with_fill("BTC"):
        by_hh[r["ts"] // 3600].append(r["side"])
    return sigs, by_hh


def bucket(by_hh, ts_h, direction, look):
    for k in range(0, look + 1):
        sides = by_hh.get(ts_h - k)
        if sides:
            s = sides[0]
            fade = ((s == 1 and direction == "DOWN")
                    or (s == -1 and direction == "UP"))
            return "fade" if fade else "follow"
    return "none"


def wr(rows):
    return 100 * sum(r["c"] for r in rows) / len(rows) if rows else None


def ret(rows):
    xs = [r["ret"] for r in rows if r["ret"] is not None]
    # realized 4h return IN THE SIGNAL'S DIRECTION (positive = signal paid)
    return 100 * sum(xs) / len(xs) if xs else None


def fmt(rows, tag):
    parts = []
    for b in ("none", "fade", "follow"):
        g = [r for r in rows if r["b"] == b]
        if len(g) >= 15:
            parts.append(f"{b} {wr(g):.0f}%/{ret(g):+.2f}% (n={len(g)})")
        else:
            parts.append(f"{b} thin({len(g)})")
    return f"  {tag:<14}" + " | ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clock", action="store_true")
    args = ap.parse_args()
    sigs, by_hh = load()
    rows = []
    for s in sigs:
        ts = int(s["signal_time"].replace(tzinfo=timezone.utc).timestamp())
        d = s["direction"]
        sgn = 1 if d == "UP" else -1
        rr = (float(s["actual_return_4h"]) * sgn
              if s["actual_return_4h"] is not None else None)
        rows.append({"ts": ts, "dir": d, "regime": s["regime"] or "?",
                     "c": int(s["correct"]), "ret": rr,
                     "b": bucket(by_hh, ts // 3600, d, 4)})

    if args.clock:
        cut = max(r["ts"] for r in rows) - 90 * 86400
        rec = [r for r in rows if r["ts"] >= cut]
        kept = [r for r in rec if r["b"] != "follow"]
        veto = [r for r in rec if r["b"] == "follow"]
        k = wr(kept)
        v = wr(veto)
        print(f"V7 raid-veto (90d): kept {k:.0f}% (n={len(kept)}) vs "
              f"vetoed {v:.0f}% (n={len(veto)}) — gap "
              f"{k - v:+.0f}pp" if k and v else "V7 raid-veto: thin")
        return 0

    print("=" * 78)
    print("  V7 RAID-CHASE VETO — 註冊前的穩健性切片（勝率% / 方向報酬%）")
    print("=" * 78)
    res = {}
    n = len(rows)
    print(f"  Strong n={n} · 整體 {wr(rows):.0f}% / {ret(rows):+.2f}%\n")

    print("  [1] 回看窗口敏感度")
    for look in (2, 4, 6, 8):
        rr = [dict(r, b=bucket(by_hh, r["ts"] // 3600, r["dir"], look))
              for r in rows]
        print(fmt(rr, f"look={look}h"))
        res[f"look_{look}"] = {b: wr([x for x in rr if x["b"] == b])
                               for b in ("none", "fade", "follow")}

    print("\n  [2] 方向分拆（4h 窗）")
    for d in ("UP", "DOWN"):
        print(fmt([r for r in rows if r["dir"] == d], f"{d}"))

    print("\n  [3] regime 分拆")
    for g in sorted({r["regime"] for r in rows}):
        sub = [r for r in rows if r["regime"] == g]
        if len(sub) >= 60:
            print(fmt(sub, g[:13]))

    print("\n  [4] 三等分時間切片")
    third = n // 3
    for i, tag in ((0, "T1"), (1, "T2"), (2, "T3")):
        seg = rows[i * third: (i + 1) * third if i < 2 else n]
        print(fmt(seg, tag))
        res[f"third_{tag}"] = {b: wr([x for x in seg if x["b"] == b])
                               for b in ("none", "fade", "follow")}

    print("\n  [5] 反事實：套用否決的組合效果")
    kept = [r for r in rows if r["b"] != "follow"]
    veto = [r for r in rows if r["b"] == "follow"]
    print(f"    保留 {len(kept)} 筆: WR {wr(kept):.0f}% / {ret(kept):+.2f}%"
          f"  |  否決 {len(veto)} 筆 ({100*len(veto)/n:.0f}%): "
          f"WR {wr(veto):.0f}% / {ret(veto):+.2f}%")
    res["counterfactual"] = {"kept_n": len(kept), "kept_wr": wr(kept),
                             "kept_ret": ret(kept), "veto_n": len(veto),
                             "veto_wr": wr(veto), "veto_ret": ret(veto)}
    OUT.write_text(json.dumps(res, indent=1, default=float),
                   encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
