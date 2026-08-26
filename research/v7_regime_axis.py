# -*- coding: utf-8 -*-
"""V7 x regime — TODO §0.60 questions 1, 2, 4. Frozen before the run.

Why now: on the raid line the directional axis overturned a simplification
(§0.54b — TREND_DOWN scores like the home regime; only TREND_UP hurts).
V7's own regime verdict B-P8 (RANGING 63.6% vs TRENDING 58.3%) was scored
on the SAME undirected ADX, so the same simplification may sit there too.

Q1  directional axis      four cells instead of TRENDING/RANGING
Q2  weak side x regime    live losses concentrate in LONG (-3.13% vs SHORT
                          +2.63%, §0.51b). If LONG is bad specifically in
                          TREND_UP, "weak side" and "wrong regime" are one
                          thing wearing two names and filtering either one
                          suffices.
Q4  within-cell decay     the raid line's forward gap was 75% within-cell
                          (§0.58). V7's signal layer reads 53.7% over 90d
                          vs 59.5% all-time — same decomposition needed:
                          composition shift or genuine per-cell decay?

Q3 (terrain filter vs regime filter — same effect or two?) waits for the
terrain trigger to settle (~first week of September); a historical cross
table without forward data would only show shape, so it is deliberately
NOT in this file.

Discipline, per §0.60 and the terrain-campaign ritual:
  * every cell reported, no cherry-picking
  * bucket counts checked against physical plausibility BEFORE reading any
    win rate — an empty or >90% bucket means broken instrument first
    (mistake.md 2026-08-02)
  * cross-model-version comparisons flagged; sample_floor() reported
    alongside so a reader can see what a clean-slice version would hold

Read-only. Signal layer only (tracked_signals.correct = 4h TWAP direction),
which is a different question from trade P&L — never conflate the two.
"""
from __future__ import annotations

import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from datetime import timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import sweep_core as SC                                    # noqa: E402
from research.crowd_battery2 import adx_state              # noqa: E402
from shared.db import get_db_conn                          # noqa: E402

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
OUT = ROOT / "research" / "results" / "v7_regime_axis.json"
LB = 24
CELLS = ("RANGING", "TREND_UP", "TREND_DOWN", "NEUTRAL")
random.seed(7)


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def load():
    """Strong signals with outcome, tagged with the BTC regime cell."""
    bars = SC.load_csv(str(CACHE / "BTCUSDT_1h.csv"))
    c = [b[SC.C] for b in bars]
    adx = adx_state(bars)
    ret = {bars[i][0]: c[i] / c[i - LB] - 1 for i in range(LB, len(bars))}

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT signal_time, direction, correct, actual_return_4h, "
                "model_version FROM tracked_signals "
                "WHERE strength='Strong' AND correct IS NOT NULL "
                "ORDER BY signal_time")
            rows = cur.fetchall()
    finally:
        conn.close()

    out = []
    for r in rows:
        ts = int(r["signal_time"].replace(tzinfo=timezone.utc).timestamp())
        h = ts // 3600 * 3600
        lab = adx.get(h)
        if lab is None or h not in ret:
            continue
        if lab == "RANGING":
            cell = "RANGING"
        elif lab != "TRENDING":
            cell = "NEUTRAL"
        else:
            cell = "TREND_UP" if ret[h] > 0 else "TREND_DOWN"
        sgn = 1 if r["direction"] == "UP" else -1
        bps = (float(r["actual_return_4h"]) * sgn * 10000
               if r["actual_return_4h"] is not None else None)
        out.append({"ts": ts, "dir": r["direction"], "cell": cell,
                    "ok": int(r["correct"]), "bps": bps,
                    "mv": r["model_version"] or ""})
    return out


def block(rows, label):
    n = len(rows)
    k = sum(r["ok"] for r in rows)
    b = [r["bps"] for r in rows if r["bps"] is not None]
    ci = wilson(k, n)
    return {"label": label, "n": n, "wr": 100 * k / n if n else None,
            "ci": [100 * ci[0], 100 * ci[1]] if ci else None,
            "bps": st.mean(b) if b else None}


def show(d, indent="  "):
    ci = f"[{d['ci'][0]:.1f},{d['ci'][1]:.1f}]" if d["ci"] else "—"
    bps = f"{d['bps']:+7.1f}" if d["bps"] is not None else "      —"
    print(f"{indent}{d['label']:<22} n={d['n']:<5} WR {d['wr']:5.1f}%  "
          f"CI {ci:<14} {bps} bps")


def main() -> int:
    rows = load()
    res = {}
    print(f"V7 x regime — Strong signals, signal layer (4h TWAP direction)")
    print(f"total tagged n={len(rows)}\n")

    # ── physical sanity BEFORE any win rate is read ────────────────────
    print("── 分桶 n 的物理檢查（先於任何勝率解讀）──")
    counts = {c: sum(1 for r in rows if r["cell"] == c) for c in CELLS}
    tot = sum(counts.values())
    bad = [c for c, n in counts.items() if n == 0 or n / tot > 0.9]
    for c in CELLS:
        print(f"  {c:<12} n={counts[c]:<5} {100*counts[c]/tot:5.1f}%")
    if bad:
        print(f"\n  儀器疑慮：{bad} 佔比不合物理，停止解讀")
        return 1
    print("  → 四桶皆有樣本、無單桶 >90%，通過\n")

    # ── Q1 directional axis ────────────────────────────────────────────
    print("── Q1  方向軸：B-P8 的 TRENDING 拆成上下 ──")
    overall = block(rows, "全體")
    show(overall)
    q1 = {}
    for c in CELLS:
        d = block([r for r in rows if r["cell"] == c], c)
        q1[c] = d
        show(d)
    tr = [r for r in rows if r["cell"] in ("TREND_UP", "TREND_DOWN")]
    show(block(tr, "TRENDING（合併，B-P8 口徑）"))
    up, dn = q1["TREND_UP"], q1["TREND_DOWN"]
    print(f"\n  上下之差 {up['wr'] - dn['wr']:+.1f}pp  "
          f"({'方向軸成立，B-P8 合併掩蓋了差異' if abs(up['wr'] - dn['wr']) >= 5 else '方向差異不足 5pp，合併無害'})")
    res["q1"] = {**{k: v for k, v in q1.items()}, "overall": overall}

    # ── Q2 weak side x regime ──────────────────────────────────────────
    print("\n── Q2  弱側 × regime：LONG 是否特別死在 TREND_UP ──")
    print(f"  {'':<22} {'UP 訊號':>22}   {'DOWN 訊號':>22}")
    q2 = {}
    for c in CELLS:
        u = block([r for r in rows if r["cell"] == c and r["dir"] == "UP"], "")
        d = block([r for r in rows if r["cell"] == c and r["dir"] == "DOWN"], "")
        q2[c] = {"UP": u, "DOWN": d}
        us = f"n={u['n']:<4} {u['wr']:5.1f}%" if u["n"] else "n=0"
        ds = f"n={d['n']:<4} {d['wr']:5.1f}%" if d["n"] else "n=0"
        print(f"  {c:<22} {us:>22}   {ds:>22}")
    res["q2"] = q2
    uu = q2["TREND_UP"]["UP"]
    ur = q2["RANGING"]["UP"]
    if uu["n"] and ur["n"]:
        print(f"\n  UP 訊號在 TREND_UP vs RANGING: "
              f"{uu['wr']:.1f}% vs {ur['wr']:.1f}%  ({uu['wr']-ur['wr']:+.1f}pp)")

    # ── Q4 within-cell decay ───────────────────────────────────────────
    print("\n── Q4  格內衰退：近 90 天 vs 更早（同格比較）──")
    cut = max(r["ts"] for r in rows) - 90 * 86400
    early = [r for r in rows if r["ts"] < cut]
    late = [r for r in rows if r["ts"] >= cut]
    print(f"  切點：最後一筆往前 90 天；早期 n={len(early)}、近期 n={len(late)}")
    print(f"  {'cell':<12} {'早期':>16} {'近期':>16} {'格內差':>9}")
    q4 = {}
    for c in CELLS:
        e = block([r for r in early if r["cell"] == c], "")
        l = block([r for r in late if r["cell"] == c], "")
        q4[c] = {"early": e, "late": l}
        es = f"n={e['n']:<4} {e['wr']:5.1f}%" if e["n"] else "n=0        "
        ls = f"n={l['n']:<4} {l['wr']:5.1f}%" if l["n"] else "n=0        "
        gap = (f"{l['wr']-e['wr']:+8.1f}pp" if e["n"] and l["n"] else "       —")
        print(f"  {c:<12} {es:>16} {ls:>16} {gap:>9}")
    res["q4"] = q4
    # composition vs within-cell split
    if all(q4[c]["early"]["n"] and q4[c]["late"]["n"] for c in CELLS):
        e_mix = {c: q4[c]["early"]["n"] / len(early) for c in CELLS}
        l_mix = {c: q4[c]["late"]["n"] / len(late) for c in CELLS}
        e_wr = st.mean([r["ok"] for r in early]) * 100
        l_wr = st.mean([r["ok"] for r in late]) * 100
        # early cells under late mix = pure composition effect
        comp = sum(q4[c]["early"]["wr"] * l_mix[c] for c in CELLS)
        print(f"\n  早期整體 {e_wr:.1f}%  近期整體 {l_wr:.1f}%  "
              f"落差 {l_wr - e_wr:+.1f}pp")
        print(f"  早期各格 @ 近期組成 = {comp:.1f}%  → "
              f"組成效應 {comp - e_wr:+.1f}pp、格內效應 {l_wr - comp:+.1f}pp")
        res["q4_split"] = {"early_wr": round(e_wr, 2), "late_wr": round(l_wr, 2),
                           "composition_pp": round(comp - e_wr, 2),
                           "within_pp": round(l_wr - comp, 2)}

    OUT.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
