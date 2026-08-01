# -*- coding: utf-8 -*-
"""L1 — liquidation levels as a FIFTH pool kind (2026-08-02).

What the data actually is: Coinglass gives HOURLY liquidation notional
(long/short), not a forward heatmap of where open leverage would die.
So the only causal construct available is the liquidation SITE — the
price extreme of a bar where leverage was actually wiped out:
  long liquidations  = forced selling  -> the bar's LOW,  side -1
  short liquidations = forced buying   -> the bar's HIGH, side +1
Burst threshold = causal expanding 90th percentile of that side's own
history (>=200 prior hours required), so no tuned constant and no
future information. Pool goes live the NEXT bar and dies when swept —
identical lifecycle to the four standard kinds.

Frozen contrasts (declared before the run; both are built so that a
pass means the fifth kind adds what the four kinds MISS, never a
repackaging of D2/D3):
  L-A 墊背側: population = signals with NO standard un-swept support
      behind (D6's vacuum, ~504). Does a liq level within 1.8 ATR
      behind substitute? Predicted YES (toward D3's 68%).
  L-B 前牆側: population = signals with a clean runway by standard
      pools (ahead > 1.4 ATR or none). Does a liq level within 1.4 ATR
      ahead hurt? Predicted YES (toward D2's 57%).
Note D6 ran exactly the L-A shape on flipped levels and FAILED — that
is the bar this has to clear.

Gates: G1 (|gap|>=4pp + halves same sign) on each named contrast;
survivors go to G2 residual + G3 permutation/bootstrap/quarters.

Run: python research/terrain_l1_liqlevels.py
Out: research/results/terrain_l1_liqlevels.json
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
import pandas as pd  # noqa: E402
import sweep_core as SC  # noqa: E402
import level_types as LT  # noqa: E402
from v7_price_location_verify import build_rows  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/terrain_l1_liqlevels.json"
LIQ = ROOT / "market_data/raw_data/cg_liquidation_1h.parquet"
WALL, SUP = 1.4, 1.8
WARM, PCT = 200, 90


def liq_pools(bars):
    """[est, swept_or_None, lvl, side] from causal liquidation bursts."""
    df = pd.read_parquet(LIQ)
    df = df[["long_liquidation_usd", "short_liquidation_usd"]].fillna(0.0)
    ts_of = {int(t.timestamp()): i for i, t in enumerate(df.index)}
    longs = df["long_liquidation_usd"].to_numpy(float)
    shorts = df["short_liquidation_usd"].to_numpy(float)
    h = [b[SC.H] for b in bars]
    lo = [b[SC.L] for b in bars]
    pools = []
    for j, b in enumerate(bars):
        i = ts_of.get(b[0])
        if i is None or i < WARM:
            continue
        # causal: percentile of everything STRICTLY before this hour
        if longs[i] > 0 and longs[i] >= np.percentile(longs[:i], PCT):
            pools.append([j + 1, None, lo[j], -1])
        if shorts[i] > 0 and shorts[i] >= np.percentile(shorts[:i], PCT):
            pools.append([j + 1, None, h[j], 1])
    pools.sort(key=lambda x: x[0])
    live = []
    idx = 0
    for j in range(len(bars)):
        while idx < len(pools) and pools[idx][0] <= j:
            live.append(pools[idx])
            idx += 1
        for p in list(live):
            if (h[j] > p[2] if p[3] == 1 else lo[j] < p[2]):
                p[1] = j
                live.remove(p)
    return pools


def wr(g):
    return 100 * sum(r["c"] for r in g) / len(g) if g else None


def sh(g):
    return f"{wr(g):.0f}%({len(g)})" if len(g) >= 15 else f"thin({len(g)})"


def gates(name, rows, pred, lab_a, lab_b, res):
    """G1 -> G2 -> G3 on a boolean predicate over `rows`."""
    rows = sorted(rows, key=lambda r: r["ts"])
    half = len(rows) // 2
    ds = []
    for tag, seg in (("全期", rows), ("H1", rows[:half]), ("H2", rows[half:])):
        a_ = [r for r in seg if pred(r)]
        b_ = [r for r in seg if not pred(r)]
        d_ = wr(a_) - wr(b_) if len(a_) >= 15 and len(b_) >= 15 else None
        ds.append(d_)
        print(f"  {tag:<4} {lab_a} {sh(a_)} | {lab_b} {sh(b_)}"
              + (f" | gap {d_:+.0f}pp" if d_ is not None else " | thin"))
    ok = (None not in ds and ds[1] * ds[2] > 0 and abs(ds[0]) >= 4)
    show = " · ".join("thin" if d is None else f"{d:+.0f}" for d in ds)
    print(f"  {lab_a}−{lab_b} gap: {show} → G1 {'PASS' if ok else 'FAIL'}")
    res[name] = {"deltas": ds, "g1_pass": ok, "n": len(rows)}
    if not ok:
        return False

    sign = 1 if ds[0] > 0 else -1
    print(f"  [G2] 殘餘（{'A優' if sign == 1 else 'B優'}）")
    good = (lambda r: pred(r)) if sign == 1 else (lambda r: not pred(r))
    okc = tot = 0
    for cn, cp in (("ctx=none", lambda r: r["ctx"] == "none"),
                   ("ctx=fade", lambda r: r["ctx"] == "fade"),
                   ("ctx=follow", lambda r: r["ctx"] == "follow"),
                   ("UP", lambda r: r["dir"] == "UP"),
                   ("DOWN", lambda r: r["dir"] == "DOWN")):
        seg = [r for r in rows if cp(r)]
        a_ = [r for r in seg if good(r)]
        b_ = [r for r in seg if not good(r)]
        if len(a_) >= 20 and len(b_) >= 20:
            d_ = wr(a_) - wr(b_)
            tot += 1
            okc += d_ > 0
            print(f"    {cn:<10} {d_:+.0f}pp (n={len(a_)}/{len(b_)})")
    g2 = tot >= 3 and okc / tot >= 0.67
    print(f"  桶內同向 {okc}/{tot} → G2 {'PASS' if g2 else 'FAIL'}")
    res[name]["g2"] = {"ok": okc, "tot": tot, "pass": g2}
    if not g2:
        return False

    ga = np.array([r["c"] for r in rows if good(r)])
    ba = np.array([r["c"] for r in rows if not good(r)])
    obs = 100 * (ga.mean() - ba.mean())
    rg = np.random.default_rng(7)
    pool_ = np.concatenate([ga, ba])
    null = [100 * (lambda p: p[:len(ga)].mean() - p[len(ga):].mean())(
        rg.permutation(pool_)) for _ in range(2000)]
    pval = float((np.array(null) >= obs).mean())
    boots = [100 * (rg.choice(ga, len(ga), True).mean()
                    - rg.choice(ba, len(ba), True).mean()) for _ in range(2000)]
    lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
    byq = {}
    for r in rows:
        dt = datetime.fromtimestamp(r["ts"], timezone.utc)
        byq.setdefault(f"{dt.year}-Q{(dt.month-1)//3+1}", []).append(r)
    qs = []
    for q in sorted(byq):
        a_ = [r["c"] for r in byq[q] if good(r)]
        b_ = [r["c"] for r in byq[q] if not good(r)]
        if len(a_) >= 10 and len(b_) >= 10:
            qs.append((q, round(100 * (np.mean(a_) - np.mean(b_)), 1)))
    adv = sum(1 for _q, d in qs if d < 0)
    g3 = pval < 0.05 and lo_ci > 0 and adv <= 1
    print(f"  [G3] gap {obs:+.1f}pp · p={pval:.4f} · CI [{lo_ci:+.1f},{hi_ci:+.1f}]"
          f" · 逐季 " + " ".join(f"{q}{d:+.0f}" for q, d in qs) + f" · 逆風 {adv}")
    print(f"       G3 {'PASS ✅ 取得席位' if g3 else 'FAIL — 記錄收檔'}")
    res[name]["g3"] = {"gap": round(float(obs), 1), "p": pval,
                       "ci": [round(float(lo_ci), 1), round(float(hi_ci), 1)],
                       "quarters": qs, "pass": g3}
    return g3


def main() -> int:
    print("=" * 78)
    print("  L1 清算價位 — 第五種池子（爆量清算現場當支撐/阻力）")
    print("=" * 78)
    bars = SC.load_csv(str(LT.CACHE / "BTCUSDT_1h.csv"))
    ts2i = {b[0]: i for i, b in enumerate(bars)}
    atr = SC.atr14(bars)
    cl = [b[SC.C] for b in bars]
    lp = liq_pools(bars)
    print(f"\n  清算池 {len(lp)} 個（多爆 {sum(1 for p in lp if p[3]==-1)} / "
          f"空爆 {sum(1 for p in lp if p[3]==1)}）· "
          f"已被掃 {sum(1 for p in lp if p[1] is not None)}")

    rows = []
    for r in build_rows():
        j = ts2i[r["ts"]]
        c = cl[j]
        up = r["dir"] == "UP"
        la = lb = None
        for p in lp:
            if p[0] <= j and (p[1] is None or p[1] > j):
                d_ = (p[2] - c) / atr[j]
                ad, bd = (d_, -d_) if up else (-d_, d_)
                if ad > 0 and (la is None or ad < la):
                    la = ad
                if bd > 0 and (lb is None or bd < lb):
                    lb = bd
        r2 = dict(r)
        r2["liq_ahead"] = la
        r2["liq_behind"] = lb
        r2["has_sup"] = r["behind"] is not None and r["behind"] <= SUP
        r2["has_wall"] = r["ahead"] is not None and r["ahead"] <= WALL
        rows.append(r2)
    rows = [r for r in rows if r["liq_ahead"] is not None
            or r["liq_behind"] is not None]
    rows.sort(key=lambda r: r["ts"])
    res = {"n_pools": len(lp), "n_rows": len(rows)}
    print(f"  可用訊號 n={len(rows)}（清算資料 2025-10 起 + 200h 暖身）"
          f" · 整體 {wr(rows):.0f}%")

    print(f"\n  [L-A] 墊背側：標準池真空樣本內，清算位能否頂替支撐")
    vac = [r for r in rows if not r["has_sup"]]
    a_ok = gates("L_A", vac,
                 lambda r: r["liq_behind"] is not None and r["liq_behind"] <= SUP,
                 "清算墊背", "真空", res)

    print(f"\n  [L-B] 前牆側：標準跑道乾淨樣本內，清算位是否構成牆")
    clean = [r for r in rows if not r["has_wall"]]
    b_ok = gates("L_B", clean,
                 lambda r: r["liq_ahead"] is not None and r["liq_ahead"] <= WALL,
                 "清算牆", "無牆", res)

    res["any_seat"] = bool(a_ok or b_ok)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                              default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
