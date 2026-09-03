# -*- coding: utf-8 -*-
"""出場第二輪：結構型止盈（成交量分佈高量節點）＋真的會開機的移動停損。

**預註冊，先 commit 再跑。**

為什麼要有第二輪（§0.96 的兩個量測結果）
------------------------------------------
1. 這條規則的 edge 全在命中率上（盈虧比 0.93 < 1），而**平均 0.863R 的
   有利波動只兌現成 0.029R**——32.0% 的交易摸到過 +1R，只有 5.3% 收在
   +1R 以上（守住率 0.17）。錢在波動裡，被時間出場還回去了。
2. **第一輪的三個「保護贏家」變體全部以 +1R 啟動**（trail_2atr /
   trail_1atr / giveback），而只有 32% 的交易摸得到 +1R——**它們在三分之
   二的樣本上根本沒開機**。所以它們的 FAIL 不是「保護贏家沒用」，是這個
   問題還沒被問過。這一輪把啟動水位放到**量測出來的**位置。
3. 拉長持有已經被否證（持有曲線單調往下到 12–16 根，hold_12 FAIL），
   所以這一輪**不動時間上限**，只換「在期限內怎麼離場」。

兩個變體（凍結，各自帶機制敘述，不是參數掃描）
----------------------------------------------
  X1 hvn_target   在交易方向上的第一個**高量節點**止盈（使用者手動用的
                  固定區間成交量分佈）。機制：價格在成交量堆積處會停，
                  在真空處會滑過去——所以止盈該掛在結構上，不是掛在第 8
                  根收盤。與第一輪 FAIL 的 V8 pool_target **不是同一件事**：
                  那個用的是流動性池（沒人成交過的價位），這個用的是
                  成交量堆積（大家都成交過的價位），兩者在 ICT 裡是相反
                  的東西。
  X2 trail_half   2.0 ATR 移動停損，**+0.5R 啟動**（V1 是 +1R）。0.5R
                  不是掃出來的：§0.96 量到 MFE 中位是 0.70R，啟動水位必須
                  低於中位才會在半數以上的交易開機，0.5 是中位下方的整數。
                  它要回答的是「V1 的 FAIL 是因為移動停損沒用，還是因為
                  它幾乎沒開過機」。

固定區間怎麼定（不留選擇權）
----------------------------
「固定區間」在手動操作是人框的。機械版必須寫死，而且不能新增一個可調
參數，所以直接用**這筆交易自己的結構**：從被掃的價位，回推到**掃穿之前
最近一個已確認的反向 pivot**（沿用凍結引擎的 PIVOT=10，不新增參數），
區間就是這兩點之間的價格帶，bar 取這兩根之間。成交量分佈用 50 格（固定），
每根 bar 的量在它自己的高低之間均勻攤開（只有 OHLCV 時的標準近似）。
高量節點 = 直方圖的區域極大值且量 ≥ 全區間平均。取交易方向上**第一個**。
沒有節點就退回凍結出場（變體永不變成「沒有出場」）。

成本與保守假設（先寫死）
------------------------
  X1 以限價單成交 -> 用 scenario A 的 texit 3 bps
  X2 以市價停損離場 -> 用 sexit 10 bps（偏保守，移動停損不是災難停損）
  同一根 bar 同時碰到停損與目標 -> **一律算停損先到**（對我們不利那邊）

判準（沿用第一輪 exit_variants 的三關，不放寬）
-----------------------------------------------
  G1 成對均差 dR > 0 且前後兩半同號
  G2 ≥ 6/9 幣同號
  G3 成對 bootstrap CI 排除 0 且成對置換（翻符號）檢定 p < 0.05
三關全過才是候選；候選仍需自己的前瞻註冊才可能取代凍結出場——Gate F
正在現行出場上跑，不得污染。

本輪檢定數 = 2。連同 §0.94 的 2 個、第一輪的 9 個，這條線的出場/地形族
累計 13 次檢定，判準一律維持單次 5%，**不事後放寬**；任何一個通過都要
在前瞻上重新賺一次。

Run: python research/sweep_failure/exit_volume_profile.py
Out: research/results/sweep_exit_volume_profile.json
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["SLIP"] = "0"
import sweep_core as SC            # noqa: E402
from sweep_forward import SCEN     # noqa: E402
import room_ahead as RA            # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/sweep_exit_volume_profile.json"
BINS = 50
TRAIL_ATR = 2.0
ARM_R = 0.5
V = 5                              # volume column in sweep_core rows


def pivot_bars(bars):
    """Confirmed pivot bars -> (bar_index_of_extreme, price, side).

    Same PIVOT as the frozen engine; used only to bound the volume-profile
    range, never to create a signal.
    """
    n = len(bars)
    h = [b[SC.H] for b in bars]
    l = [b[SC.L] for b in bars]
    out = []
    for i in range(SC.PIVOT, n - SC.PIVOT):
        seg = range(i - SC.PIVOT, i + SC.PIVOT + 1)
        if all(h[i] >= h[k] for k in seg) and any(h[i] > h[k] for k in seg if k != i):
            out.append((i, h[i], 1))
        if all(l[i] <= l[k] for k in seg) and any(l[i] < l[k] for k in seg if k != i):
            out.append((i, l[i], -1))
    out.sort()
    return out


def hvn_targets(bars, i0, i1, lo, hi):
    """High-volume nodes inside [lo,hi] over bars[i0..i1], low->high price."""
    if i1 <= i0 or hi <= lo:
        return []
    w = (hi - lo) / BINS
    hist = [0.0] * BINS
    for k in range(i0, i1 + 1):
        bh, bl, bv = bars[k][SC.H], bars[k][SC.L], bars[k][V]
        top, bot = min(bh, hi), max(bl, lo)
        if top <= bot or bv <= 0:
            continue
        share = bv / (top - bot)
        a = int((bot - lo) / w)
        b = min(BINS - 1, int((top - lo) / w))
        for q in range(max(0, a), b + 1):
            s, e = lo + q * w, lo + (q + 1) * w
            ov = min(top, e) - max(bot, s)
            if ov > 0:
                hist[q] += share * ov
    avg = sum(hist) / BINS
    nodes = []
    for q in range(1, BINS - 1):
        if hist[q] >= avg and hist[q] >= hist[q - 1] and hist[q] >= hist[q + 1]:
            nodes.append(lo + (q + 0.5) * w)
    return nodes


def run(bars, sym):
    n = len(bars)
    h = [b[SC.H] for b in bars]
    l = [b[SC.L] for b in bars]
    c = [b[SC.C] for b in bars]
    idx = {b[0]: i for i, b in enumerate(bars)}
    piv = pivot_bars(bars)
    s = SCEN["A"]
    rows = []
    for fill_ts, _e, _r, lvl, atr, _st, _pi, side in SC.backtest_symbol(bars):
        f = idx.get(fill_ts)
        if f is None or f + 1 >= n:
            continue
        d = 1 if side == "LONG" else -1
        entry = lvl + d * SC.SLIP * atr
        risk = SC.DIS * atr
        stop = entry - d * risk
        last = min(f + SC.HOLD, n - 1)

        # --- volume-profile range: swept level back to the last confirmed
        #     opposite pivot before the sweep. d=+1 (long) means a low was
        #     swept, so the opposite pivot is a high.
        opp = [p for p in piv if p[0] < f and p[2] == d]
        i0 = opp[-1][0] if opp else max(0, f - 200)
        top_p = opp[-1][1] if opp else max(h[max(0, f - 200):f + 1])
        lo_p, hi_p = (min(lvl, top_p), max(lvl, top_p))
        nodes = hvn_targets(bars, i0, f, lo_p, hi_p)
        ahead = [p for p in nodes if (p - entry) * d > 0]
        tgt = (min(ahead) if d == 1 else max(ahead)) if ahead else None

        base = x1 = x2 = None
        peak = 0.0
        armed = False
        trail = None
        for k in range(f + 1, last + 1):
            hit_stop = (d == 1 and l[k] <= stop) or (d == -1 and h[k] >= stop)
            fav = (h[k] - entry) * d / risk if d == 1 else (entry - l[k]) / risk
            # --- baseline (frozen)
            if base is None:
                if hit_stop:
                    base = (-1.0 - SC.SLIP / SC.DIS, "stop")
                elif k == last:
                    base = (d * (c[k] - d * SC.SLIP * atr - entry) / risk, "time")
            # --- X1 structural take-profit
            if x1 is None:
                if hit_stop:
                    x1 = (-1.0 - SC.SLIP / SC.DIS, "stop")
                elif tgt is not None and ((d == 1 and h[k] >= tgt) or
                                          (d == -1 and l[k] <= tgt)):
                    x1 = (d * (tgt - entry) / risk, "hvn")
                elif k == last:
                    x1 = (d * (c[k] - d * SC.SLIP * atr - entry) / risk, "time")
            # --- X2 trailing armed at +0.5R
            if x2 is None:
                if hit_stop:
                    x2 = (-1.0 - SC.SLIP / SC.DIS, "stop")
                else:
                    peak = max(peak, fav)
                    if not armed and peak >= ARM_R:
                        armed = True
                    if armed:
                        pk = entry + d * peak * risk
                        trail = pk - d * TRAIL_ATR * atr
                        touched = ((d == 1 and l[k] <= trail) or
                                   (d == -1 and h[k] >= trail))
                        if touched:
                            x2 = (d * (trail - entry) / risk, "trail")
                    if x2 is None and k == last:
                        x2 = (d * (c[k] - d * SC.SLIP * atr - entry) / risk, "time")
            if base and x1 and x2:
                break
        if not (base and x1 and x2):
            continue

        def net(v):
            r, how = v
            leg = s["sexit"] if how in ("stop", "trail") else s["texit"]
            return r - (s["entry"] + leg) / 1e4 * lvl / (SC.DIS * atr)

        rows.append(dict(sym=sym, ts=fill_ts, base=net(base), x1=net(x1),
                         x2=net(x2), how1=x1[1], how2=x2[1],
                         has_target=tgt is not None))
    return rows


def gates(rows, key, name):
    d = [r[key] - r["base"] for r in rows]
    n = len(d)
    m = sum(d) / n
    half = n // 2
    h1 = sum(d[:half]) / half
    h2 = sum(d[half:]) / (n - half)
    per = {}
    for sym in RA.SYMS:
        sub = [r[key] - r["base"] for r in rows if r["sym"] == sym]
        per[sym] = sum(sub) / len(sub) if sub else None
    pos = sum(1 for v in per.values() if v is not None and v > 0)
    rng = random.Random(7)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(4000))
    ci = (boots[100], boots[3900])
    cnt = 0
    for _ in range(4000):
        z = sum(x if rng.random() < 0.5 else -x for x in d) / n
        cnt += abs(z) >= abs(m)
    p = cnt / 4000
    g1 = m > 0 and h1 > 0 and h2 > 0
    g2 = pos >= 6
    g3 = ci[0] > 0 and p < 0.05
    print(f"  {name:<14}dR {m:>+8.4f}  兩半 {h1:>+7.4f}/{h2:>+7.4f}  "
          f"幣正 {pos}/9  CI [{ci[0]:+.4f},{ci[1]:+.4f}]  p={p:.3f}  "
          f"{'PASS' if (g1 and g2 and g3) else 'FAIL'}")
    print(f"  {'':<14}G1 {'✅' if g1 else '❌'}  G2 {'✅' if g2 else '❌'}"
          f"  G3 {'✅' if g3 else '❌'}")
    return dict(mean_dR=m, h1=h1, h2=h2, sym_positive=pos, ci=list(ci), p=p,
                verdict="PASS" if (g1 and g2 and g3) else "FAIL", per_symbol=per)


def main() -> int:
    print("=" * 92)
    print("  出場第二輪 —— 結構型止盈(高量節點) vs 真的會開機的移動停損(+0.5R)")
    print("=" * 92)
    rows = []
    for sym in RA.SYMS:
        p = RA.CACHE / f"{sym}USDT_1h.csv"
        if p.exists():
            rows += run(SC.load_csv(str(p)), sym)
    rows.sort(key=lambda r: r["ts"])
    n = len(rows)
    print(f"\n  成對樣本 n={n}  凍結出場 meanR "
          f"{sum(r['base'] for r in rows)/n:+.4f}")
    ht = sum(1 for r in rows if r["has_target"])
    print(f"  找得到方向上高量節點的：{ht} 筆（{100*ht/n:.1f}%）")
    for k, lab in (("how1", "X1 出場方式"), ("how2", "X2 出場方式")):
        c = {}
        for r in rows:
            c[r[k]] = c.get(r[k], 0) + 1
        print(f"  {lab}: " + "  ".join(f"{a} {b} ({100*b/n:.0f}%)"
                                       for a, b in sorted(c.items())))
    print()
    res = {"n": n, "base_mean": sum(r["base"] for r in rows) / n,
           "has_target_frac": ht / n}
    res["hvn_target"] = gates(rows, "x1", "X1 hvn_target")
    res["trail_half"] = gates(rows, "x2", "X2 trail_half")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
