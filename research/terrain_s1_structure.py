# -*- coding: utf-8 -*-
"""S1 — market-structure direction (the catalog's first untested dim).

Definition (causal, PIVOT=10 confirmed swings, est = i+P+1):
  at a signal bar, take the last two CONFIRMED swing highs and lows
  (ordered by pivot bar). HH+HL = 上升結構, LH+LL = 下降結構, else 盤整.
  Signal alignment: UP in 上升 = 順結構; UP in 下降 = 逆結構; 盤整 = its
  own bucket.

Pre-stated predictions (frozen in TODO 0.484): textbook says 順結構 wins;
V7's mean-reversion character (and the dead 7d-trend alignment) says it
may invert. No lean recorded — the 766 live signals decide.

Three gates in one run (protocol fixed):
  G1 buckets + halves same-direction
  G2 residual inside the confirmed margins (D1 ctx / D2 wall / D3 support)
  G3 permutation p + bootstrap CI + quarterly signs on the headline
     contrast — only reached if G1 passes; a too-pretty number triggers
     the extra checks regardless (operator: 數據太好看記得反復驗證).

Run: python research/terrain_s1_structure.py
Out: research/results/terrain_s1_structure.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import numpy as np  # noqa: E402
import sweep_core as SC  # noqa: E402
import level_types as LT  # noqa: E402
from v7_price_location_verify import build_rows  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/terrain_s1_structure.json"
WALL, SUP = 1.4, 1.8


def structure_series(bars):
    """bar index -> 'up'/'down'/'range' from the last two confirmed swing
    highs/lows (by pivot order) at that bar."""
    H, L = SC.H, SC.L
    n = len(bars)
    h = [b[H] for b in bars]
    lo = [b[L] for b in bars]
    P = SC.PIVOT
    piv = []          # (pivot_i, est_i, price, side)
    for i in range(P, n - P):
        seg = range(i - P, i + P + 1)
        if all(h[i] >= h[k] for k in seg) and any(h[i] > h[k] for k in seg if k != i):
            piv.append((i, i + P + 1, h[i], 1))
        if all(lo[i] <= lo[k] for k in seg) and any(lo[i] < lo[k] for k in seg if k != i):
            piv.append((i, i + P + 1, lo[i], -1))
    piv.sort(key=lambda x: x[1])          # by confirmation time
    out = ["range"] * n
    highs, lows = [], []
    pi = 0
    for j in range(n):
        while pi < len(piv) and piv[pi][1] <= j:
            _i, _e, price, side = piv[pi]
            (highs if side == 1 else lows).append((_i, price))
            pi += 1
        if len(highs) >= 2 and len(lows) >= 2:
            hh = sorted(highs)[-2:]
            ll = sorted(lows)[-2:]
            if hh[1][1] > hh[0][1] and ll[1][1] > ll[0][1]:
                out[j] = "up"
            elif hh[1][1] < hh[0][1] and ll[1][1] < ll[0][1]:
                out[j] = "down"
    return out


def wr(g):
    return 100 * sum(r["c"] for r in g) / len(g) if g else None


def main() -> int:
    print("=" * 78)
    print("  S1 結構方向 — 三關協議（兩半 → 殘餘 → 統計關）")
    print("=" * 78)
    bars = SC.load_csv(str(LT.CACHE / "BTCUSDT_1h.csv"))
    ts2i = {b[0]: i for i, b in enumerate(bars)}
    struct = structure_series(bars)
    rows = []
    for r in build_rows():
        j = ts2i.get(r["ts"])
        if j is None:
            continue
        st = struct[j]
        if st == "range":
            al = "盤整"
        else:
            with_ = (st == "up") == (r["dir"] == "UP")
            al = "順結構" if with_ else "逆結構"
        r2 = dict(r)
        r2["st"] = st
        r2["al"] = al
        rows.append(r2)
    rows.sort(key=lambda r: r["ts"])
    n = len(rows)
    res = {}

    print(f"\n  [G1] 分桶（n={n}, 整體 {wr(rows):.0f}%）")
    half = n // 2
    for tag, seg in (("全期", rows), ("H1", rows[:half]), ("H2", rows[half:])):
        parts = []
        for b in ("順結構", "逆結構", "盤整"):
            g = [r for r in seg if r["al"] == b]
            parts.append(f"{b} {wr(g):.0f}% (n={len(g)})" if len(g) >= 15
                         else f"{b} thin({len(g)})")
        print(f"  {tag:<4}" + " | ".join(parts))
        res[f"g1_{tag}"] = {b: wr([r for r in seg if r["al"] == b])
                            for b in ("順結構", "逆結構", "盤整")}

    d_full = (res["g1_全期"]["順結構"] or 0) - (res["g1_全期"]["逆結構"] or 0)
    d1 = (res["g1_H1"]["順結構"] or 0) - (res["g1_H1"]["逆結構"] or 0)
    d2 = (res["g1_H2"]["順結構"] or 0) - (res["g1_H2"]["逆結構"] or 0)
    g1_pass = d1 * d2 > 0 and abs(d_full) >= 4
    print(f"  順−逆 gap: 全期 {d_full:+.0f}pp · H1 {d1:+.0f} · H2 {d2:+.0f}"
          f" → G1 {'PASS' if g1_pass else 'FAIL'}")
    res["g1_pass"] = g1_pass
    if not g1_pass:
        OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                  default=float), encoding="utf-8")
        print("\n  S1 止步於 G1 — 記錄後收檔")
        return 0

    sign = 1 if d_full > 0 else -1     # which side is "good"
    good = "順結構" if sign == 1 else "逆結構"
    bad = "逆結構" if sign == 1 else "順結構"
    print(f"\n  [G2] 殘餘檢定（{good}−{bad}，已定案邊際桶內）")
    ok_cells = tot_cells = 0
    for name, pred in (
            ("ctx=none", lambda r: r["ctx"] == "none"),
            ("ctx=fade", lambda r: r["ctx"] == "fade"),
            ("ctx=follow", lambda r: r["ctx"] == "follow"),
            ("前方牆", lambda r: r["ahead"] is not None and r["ahead"] <= WALL),
            ("前方淨", lambda r: r["ahead"] is not None and r["ahead"] > WALL),
            ("背後支撐", lambda r: r["behind"] is not None and r["behind"] <= SUP),
            ("背後空", lambda r: r["behind"] is not None and r["behind"] > SUP)):
        seg = [r for r in rows if pred(r)]
        gg = [r for r in seg if r["al"] == good]
        bb = [r for r in seg if r["al"] == bad]
        if len(gg) >= 20 and len(bb) >= 20:
            d_ = wr(gg) - wr(bb)
            tot_cells += 1
            ok_cells += d_ > 0
            print(f"    {name:<10} {d_:+.0f}pp (n={len(gg)}/{len(bb)})")
    g2_pass = tot_cells >= 4 and ok_cells / tot_cells >= 0.7
    print(f"  桶內同向 {ok_cells}/{tot_cells} → G2 {'PASS' if g2_pass else 'FAIL'}")
    res["g2"] = {"ok": ok_cells, "tot": tot_cells, "pass": g2_pass}
    if not g2_pass:
        OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                  default=float), encoding="utf-8")
        print("\n  S1 止步於 G2（可能是既有邊際的換皮）— 記錄後收檔")
        return 0

    print(f"\n  [G3] 統計關（{good} vs {bad}）")
    ga = np.array([r["c"] for r in rows if r["al"] == good])
    ba = np.array([r["c"] for r in rows if r["al"] == bad])
    obs = 100 * (ga.mean() - ba.mean())
    rng = np.random.default_rng(7)
    pool = np.concatenate([ga, ba])
    null = []
    for _ in range(2000):
        p_ = rng.permutation(pool)
        null.append(100 * (p_[:len(ga)].mean() - p_[len(ga):].mean()))
    pval = float((np.array(null) >= obs).mean())
    boots = []
    for _ in range(2000):
        boots.append(100 * (rng.choice(ga, len(ga), True).mean()
                            - rng.choice(ba, len(ba), True).mean()))
    lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
    byq = {}
    for r in rows:
        dt = datetime.fromtimestamp(r["ts"], timezone.utc)
        byq.setdefault(f"{dt.year}-Q{(dt.month-1)//3+1}", []).append(r)
    qsigns = []
    for q in sorted(byq):
        gg = [r["c"] for r in byq[q] if r["al"] == good]
        bb = [r["c"] for r in byq[q] if r["al"] == bad]
        if len(gg) >= 12 and len(bb) >= 12:
            qsigns.append((q, round(100 * (np.mean(gg) - np.mean(bb)), 1)))
    adverse = sum(1 for _q, d_ in qsigns if d_ < 0)
    g3_pass = pval < 0.05 and lo_ci > 0 and adverse <= 1
    print(f"    gap {obs:+.1f}pp · 置換 p={pval:.4f} · CI [{lo_ci:+.1f},{hi_ci:+.1f}]")
    print(f"    逐季 " + "  ".join(f"{q} {d_:+.0f}" for q, d_ in qsigns)
          + f" → 逆風 {adverse}/{len(qsigns)}")
    print(f"    G3 {'PASS ✅ S1 取得席位' if g3_pass else 'FAIL — S1 記錄收檔'}")
    res["g3"] = {"gap": round(float(obs), 1), "p": pval,
                 "ci": [round(float(lo_ci), 1), round(float(hi_ci), 1)],
                 "quarters": qsigns, "pass": g3_pass, "good_side": good}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                              default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
