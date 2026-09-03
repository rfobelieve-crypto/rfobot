# -*- coding: utf-8 -*-
"""ICT 的規則能不能改善這條掃單規則的進場？（預註冊，先 commit 再跑）

使用者說「其實就是 ICT 的概念」，並要求「用 ICT 的規則結合我的訂單流特徵」。
把這句話拆成可否證的東西之前，先講清楚兩件事：

**這條規則已經有的 ICT 元素**：流動性池（buyside/sellside liquidity）、
掃單（liquidity sweep）、回歸區間。**§0.94 的後方磁鐵**其實是意外量到的
displacement 的反面（剛突破進處女地＝續破）。

**還沒被測過的 ICT 元素**（本檔案要測的）：收復強度、結構轉換（MSS）、
公允價值缺口（FVG）進場。**killzone 時段依使用者指示不列入**（2026-09-03，
「Killzone 時段不限制」）；OTE 與 FVG 高度重疊，只留 FVG 一個代表。

**訂單流那半的誠實邊界（先寫在這裡，免得結論被過度延伸）**
真正的訂單流（逐價位撤單、OI、CVD、清算）在這個 repo 只覆蓋 **BTC/ETH、
42–56 天**（§0.95 的盤點），而本檔案的統計力來自 **core9、805 天、
n≈7,000**。**兩者不能混**。所以本輪唯一算得上訂單流的是**成交量衝擊**
（1 小時量能，九幣全歷史都有）。真正的撤單流要另立一個事件條件化的
前瞻註冊——bar 級的方向性主張已經 FAIL（CLAUDE.md §撤單流），沒有
被否證過的只剩「掃單事件當下的撤單行為」，那要另外寫。

**2026-09-03 修正（第一版跑完後撤回 I1/I4 的定義，結果作廢）**
第一版把 I1 定義成「**成交 bar 的收盤** 相對被掃價位」、I4 定義成
「**成交 bar 的總量**」，並宣稱兩者無前視。**那是錯的**：`sweep_core`
的成交發生在該 bar 的**盤中觸價**（`backtest_symbol` 在 j+1..j+W 找第一根
碰到 lvl 的 bar），所以那根 bar 的收盤與總量在成交當下都還不知道。第一版
的 I1 因此跑出 +0.52R 的分格差距（是既有任何維度的四倍）——那是「這筆
已經賺了」在冒充預測力。凍結引擎自己的註解早就寫過這條線
（pierce 特地取**掃單 bar j**，「Known at the sweep bar, i.e. strictly
before the fill」），是我沒照著做。**第一版的 I1 存活與 I4 陣亡一律作廢**，
改成下面兩個可執行的定義重跑。I2/I3 不受影響（它們本來就是延後進場，
用的是已經收完的 bar）。

四個檢定（凍結，各自帶方向預測，寫在跑之前）
--------------------------------------------
  I1 收復進場（改為替代進場規則，不再是條件化）：成交 bar **收盤**若
     收回被掃價位的正確一側，就在**那根收盤**進場；沒收回就不做這筆。
     ICT：掃了流動性又收回區間內＝被拒絕。收盤價在收盤當下是知道的，
     所以這是可執行的；代價是讓掉成交那根剩下的行情。
     **預測：成對 dR > 0。**
  I4 量能衝擊（改用**掃單 bar j** 的量，與 pierce 同一個時點，
     嚴格早於成交）vshock = v[j] / 前 20 根均量。
     ICT/訂單流：掃單要有人真的成交，沒有量的假突破是雜訊。
     **預測 netR 隨它上升**（三等分格）。

替代進場規則（會延後進場，所以整筆重算，與凍結規則成對比較）：

  I2 MSS 進場：成交後 3 根內，收盤突破「掃穿之前最後一個已確認的短期
     反向 pivot」（PIVOT_S=3 的分形，ICT 的 short-term high/low）才進場，
     進場價＝確認那根的收盤。沒確認就不做這筆。
     **預測：成對 dR > 0**（ICT 說沒有 MSS 的掃單只是流動性被拿走）。
  I3 FVG 進場：成交後 3 根內若出現同方向的公允價值缺口
     （多方：low[j+1] > high[j−1]），在缺口的近端邊界掛限價，價格回撤
     進缺口才成交。**預測：成對 dR > 0**（ICT 說回撤進缺口是折價進場）。
     這一條同時回答 §0.57：**限價掛在一個事前已知的價位**，正是發布
     落差 −0.133 R 需要的那種進場方式。

凍結的實作決定（不留選擇權）
----------------------------
* 停損價與 R 的分母**一律沿用凍結規則**（被掃價位往外 3.5 ATR），
  變體只換「什麼時候進、進在哪」，不換風險定義——否則 R 不可比。
* 時間上限一律是原來的 f+HOLD（同一個牆鐘時刻收工），成對比較才乾淨。
* **進場成本一律 7 bps**（scenario A 的 entry），連 I3 的限價進場也用
  市價的成本——寧可讓 I3 吃虧，也不要讓一個成本假設替它加分。
* 沒觸發的交易記 **R = 0**（＝不參與，錢留在手上），成對均差因此同時
  包含「挑選」與「每筆」兩種效果；另外單獨報告「有觸發那批」的均值。
* 同一根 bar 同時碰到停損與觸發條件 → **一律算停損先到**。

判準（沿用 §0.94 / 出場輪的四關與三關，不放寬）
------------------------------------------------
  I4（分格）：方向如宣稱 ∧ 兩半同號 ∧ ≥6/9 幣同號 ∧ 日聚類 CI 離零
  I1/I2/I3（成對）：dR>0 ∧ 兩半同號 ∧ ≥6/9 幣為正 ∧ 日聚類 CI 下緣>0
                    ∧ 成對置換 p<0.05

本輪檢定數 = 4。這條線（出場／地形／進場族）累計 17 次，判準一律維持
單次 5%，**不事後放寬**；通過的仍需自己的前瞻註冊才可能動凍結規則。

Run: python research/sweep_failure/ict_entry.py
Out: research/results/sweep_ict_entry.json
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
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

OUT = ROOT / "research/results/sweep_ict_entry.json"
PIVOT_S = 3        # ICT short-term fractal
CONF_W = 3         # bars allowed for MSS / FVG to appear
VOL_LB = 20        # volume shock lookback


def short_pivots(bars):
    """(confirm_index, price, side) for PIVOT_S fractals. side=1 -> a high."""
    n = len(bars)
    h = [b[SC.H] for b in bars]
    l = [b[SC.L] for b in bars]
    out = []
    for i in range(PIVOT_S, n - PIVOT_S):
        seg = range(i - PIVOT_S, i + PIVOT_S + 1)
        if all(h[i] >= h[k] for k in seg) and any(h[i] > h[k] for k in seg if k != i):
            out.append((i + PIVOT_S, h[i], 1))
        if all(l[i] <= l[k] for k in seg) and any(l[i] < l[k] for k in seg if k != i):
            out.append((i + PIVOT_S, l[i], -1))
    out.sort()
    return out


def sweep_bar_map(bars):
    """fill_bar_index -> sweep_bar_index j, replaying backtest_symbol's own
    fill search. Needed because the frozen trade tuple carries pierce (taken
    at j) but not j itself, and I4 must live at j to stay ahead of the fill.
    """
    n = len(bars)
    h = [b[SC.H] for b in bars]
    l = [b[SC.L] for b in bars]
    out = {}
    for e in SC.detect_sweeps(bars):
        j, lvl, kd = e["j"], e["level"], (1 if e["kind"] == "buy" else -1)
        for f in range(j + 1, min(j + 1 + SC.W, n)):
            if (kd == 1 and l[f] <= lvl) or (kd == -1 and h[f] >= lvl):
                out.setdefault(f, j)
                break
    return out


def run(bars, sym):
    n = len(bars)
    o = [b[SC.O] for b in bars]
    h = [b[SC.H] for b in bars]
    l = [b[SC.L] for b in bars]
    c = [b[SC.C] for b in bars]
    v = [b[SC.V] for b in bars]
    idx = {b[0]: i for i, b in enumerate(bars)}
    piv = short_pivots(bars)
    jmap = sweep_bar_map(bars)
    s = SCEN["A"]
    missing = 0
    ecost = s["entry"] / 1e4
    rows = []
    for fill_ts, _e, _r, lvl, atr, _st, pierce, side in SC.backtest_symbol(bars):
        f = idx.get(fill_ts)
        if f is None or f < VOL_LB or f + 1 >= n:
            continue
        d = 1 if side == "LONG" else -1
        risk = SC.DIS * atr
        entry0 = lvl
        stop = entry0 - d * risk
        last = min(f + SC.HOLD, n - 1)
        cu = lvl / risk / 1e4          # bps -> R conversion for this trade

        def close_out(k_from, entry_px):
            """Walk k_from..last with the frozen stop; return net R."""
            for k in range(k_from, last + 1):
                if (d == 1 and l[k] <= stop) or (d == -1 and h[k] >= stop):
                    gross = d * (stop - entry_px) / risk
                    return gross - (s["entry"] + s["sexit"]) * cu
                if k == last:
                    gross = d * (c[k] - entry_px) / risk
                    return gross - (s["entry"] + s["texit"]) * cu
            return None

        base = close_out(f + 1, entry0)
        if base is None:
            continue

        # ---- I4: volume shock AT THE SWEEP BAR (strictly before the fill,
        #      same timing rule the frozen engine uses for `pierce`).
        j = jmap.get(f)
        if j is None or j < VOL_LB:
            missing += 1
            continue
        vbase = sum(v[j - VOL_LB:j]) / VOL_LB
        vshock = v[j] / vbase if vbase > 0 else 0.0

        # ---- I1 close-back entry: act at the fill bar's CLOSE (known then),
        #      only if it closed back on the correct side of the swept level.
        r1, hit1 = 0.0, False
        if d * (c[f] - lvl) > 0 and d * (c[f] - stop) > 0 and f < last:
            got = close_out(f + 1, c[f])
            if got is not None:
                r1, hit1 = got, True

        # ---- I2 MSS entry
        opp = [p for p in piv if p[0] <= f and p[2] == d]
        mss_lvl = opp[-1][1] if opp else None
        r2, hit2 = 0.0, False
        if mss_lvl is not None:
            for k in range(f + 1, min(f + CONF_W, last) + 1):
                if (d == 1 and l[k] <= stop) or (d == -1 and h[k] >= stop):
                    break                       # stopped before confirming
                if d * (c[k] - mss_lvl) > 0:
                    if d * (c[k] - stop) > 0 and k < last:
                        got = close_out(k + 1, c[k])
                        if got is not None:
                            r2, hit2 = got, True
                    break

        # ---- I3 FVG entry (limit at the proximal edge of the gap)
        r3, hit3 = 0.0, False
        for j in range(f, min(f + CONF_W, last - 1) + 1):
            if j - 1 < 0 or j + 1 > last:
                break
            if (d == 1 and l[j + 1] > h[j - 1]) or (d == -1 and h[j + 1] < l[j - 1]):
                edge = l[j + 1] if d == 1 else h[j + 1]
                if d * (edge - stop) <= 0:
                    break                       # gap sits beyond the stop
                for m in range(j + 2, last + 1):
                    if (d == 1 and l[m] <= stop) or (d == -1 and h[m] >= stop):
                        break                   # stopped before the retrace
                    if (d == 1 and l[m] <= edge) or (d == -1 and h[m] >= edge):
                        if m < last:
                            got = close_out(m + 1, edge)
                            if got is not None:
                                r3, hit3 = got, True
                        break
                break

        rows.append(dict(sym=sym, ts=fill_ts, base=base, reclaim=r1,
                         vshock=vshock, pierce=pierce, mss=r2, fvg=r3,
                         hit_reclaim=hit1, hit_mss=hit2, hit_fvg=hit3))
    if missing:
        print(f"  [WARN] {sym}: {missing} 筆對不到掃單 bar，已剔除")
    return rows


def paired_ci(rows, key, nb=3000, seed=7):
    byd = defaultdict(list)
    for r in rows:
        day = datetime.fromtimestamp(r["ts"], tz=timezone.utc).date()
        byd[day].append(r[key] - r["base"])
    days = list(byd.values())
    rng = random.Random(seed)
    out = []
    for _ in range(nb):
        tot = cnt = 0.0
        for _ in range(len(days)):
            for x in days[rng.randrange(len(days))]:
                tot += x
                cnt += 1
        if cnt:
            out.append(tot / cnt)
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def cell_gate(rows, key, name):
    """Tertile cells; predicted: high tertile > low tertile."""
    vals = sorted(r[key] for r in rows)
    lo_q = vals[len(vals) // 3]
    hi_q = vals[2 * len(vals) // 3]
    lo = [r for r in rows if r[key] <= lo_q]
    hi = [r for r in rows if r[key] > hi_q]
    mid = [r for r in rows if lo_q < r[key] <= hi_q]
    m = lambda g: sum(x["base"] for x in g) / len(g)   # noqa: E731
    gap = m(hi) - m(lo)
    half = len(rows) // 2
    g_half = []
    for part in (rows[:half], rows[half:]):
        a = [r for r in part if r[key] <= lo_q]
        b = [r for r in part if r[key] > hi_q]
        g_half.append(m(b) - m(a) if a and b else float("nan"))
    per = {}
    for sym in RA.SYMS:
        a = [r for r in rows if r["sym"] == sym and r[key] <= lo_q]
        b = [r for r in rows if r["sym"] == sym and r[key] > hi_q]
        per[sym] = (m(b) - m(a)) if a and b else None
    pos = sum(1 for x in per.values() if x is not None and x > 0)
    trip = [(r["ts"], r["base"], 1 if r[key] > hi_q else 0)
            for r in rows if r[key] > hi_q or r[key] <= lo_q]
    ci = RA.gap_ci(trip)
    ok = (gap > 0 and g_half[0] > 0 and g_half[1] > 0 and pos >= 6 and ci[0] > 0)
    print(f"  {name:<16}切點 {lo_q:>7.3f}/{hi_q:>7.3f}   低 {m(lo):+.4f}"
          f"  中 {m(mid):+.4f}  高 {m(hi):+.4f}   差 {gap:+.4f}")
    print(f"  {'':<16}兩半 {g_half[0]:+.4f}/{g_half[1]:+.4f}  幣正 {pos}/9  "
          f"CI [{ci[0]:+.4f},{ci[1]:+.4f}]   {'存活' if ok else '陣亡'}")
    return dict(cut=[lo_q, hi_q], lo=m(lo), mid=m(mid), hi=m(hi), gap=gap,
                halves=g_half, sym_positive=pos, ci=list(ci),
                verdict="PASS" if ok else "FAIL", per_symbol=per)


def pair_gate(rows, key, hitkey, name):
    d = [r[key] - r["base"] for r in rows]
    n = len(d)
    mean = sum(d) / n
    half = n // 2
    h1 = sum(d[:half]) / half
    h2 = sum(d[half:]) / (n - half)
    per = {}
    for sym in RA.SYMS:
        sub = [r[key] - r["base"] for r in rows if r["sym"] == sym]
        per[sym] = sum(sub) / len(sub) if sub else None
    pos = sum(1 for x in per.values() if x is not None and x > 0)
    ci = paired_ci(rows, key)
    rng = random.Random(11)
    cnt = 0
    for _ in range(4000):
        z = sum(x if rng.random() < 0.5 else -x for x in d) / n
        cnt += abs(z) >= abs(mean)
    p = cnt / 4000
    hit = [r for r in rows if r[hitkey]]
    ok = mean > 0 and h1 > 0 and h2 > 0 and pos >= 6 and ci[0] > 0 and p < 0.05
    print(f"  {name:<16}觸發 {len(hit)} 筆 ({100*len(hit)/n:.0f}%)   "
          f"dR {mean:+.4f}   兩半 {h1:+.4f}/{h2:+.4f}")
    print(f"  {'':<16}幣正 {pos}/9  CI [{ci[0]:+.4f},{ci[1]:+.4f}]  p={p:.3f}"
          f"   {'PASS' if ok else 'FAIL'}")
    if hit:
        print(f"  {'':<16}只看觸發那批：凍結 "
              f"{sum(r['base'] for r in hit)/len(hit):+.4f} → 變體 "
              f"{sum(r[key] for r in hit)/len(hit):+.4f}")
    return dict(fired=len(hit), fire_rate=len(hit) / n, mean_dR=mean,
                halves=[h1, h2], sym_positive=pos, ci=list(ci), p=p,
                verdict="PASS" if ok else "FAIL", per_symbol=per,
                fired_base=(sum(r["base"] for r in hit) / len(hit)) if hit else None,
                fired_var=(sum(r[key] for r in hit) / len(hit)) if hit else None)


def main() -> int:
    print("=" * 94)
    print("  ICT 規則 × 掃單失敗 —— 收復強度 / 量能衝擊 / MSS 進場 / FVG 進場")
    print("=" * 94)
    rows = []
    for sym in RA.SYMS:
        p = RA.CACHE / f"{sym}USDT_1h.csv"
        if p.exists():
            rows += run(SC.load_csv(str(p)), sym)
    rows.sort(key=lambda r: r["ts"])
    n = len(rows)
    print(f"\n  n={n}   凍結進場 meanR {sum(r['base'] for r in rows)/n:+.4f}\n")
    res = {"n": n, "base_mean": sum(r["base"] for r in rows) / n,
           "pivot_s": PIVOT_S, "conf_w": CONF_W}
    print("  A 無前視條件化（掃單 bar 的量，三等分格，預測：高格 > 低格）")
    res["vshock"] = cell_gate(rows, "vshock", "I4 量能衝擊")
    print("\n  B 替代進場規則（成對，未觸發記 0）")
    res["reclaim"] = pair_gate(rows, "reclaim", "hit_reclaim", "I1 收復進場")
    res["mss"] = pair_gate(rows, "mss", "hit_mss", "I2 MSS 進場")
    res["fvg"] = pair_gate(rows, "fvg", "hit_fvg", "I3 FVG 進場")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
