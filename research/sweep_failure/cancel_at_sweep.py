# -*- coding: utf-8 -*-
"""掃單當下的簿口行為，能不能分辨「反轉」與「續破」？（**探索，不是判決**）

為什麼只有這一個形狀還沒被否證
------------------------------
撤單流這條線的 **bar 級方向性主張已經 FAIL 並定案**（CLAUDE.md §撤單流：
`cancel_lead_ic` 的 h5/h15/h30/h60 四格全滅、`cancel_shock_ic` TEST A 四格
全不過）。唯一活著的是**波動**（TEST B 四個 horizon 全過），而使用者
2026-07-29 已否決那條（系統沒有旋鈕接得住波動預測）。

那次問的是「**任何一根 bar** 之後價格往哪走」。**沒有被問過的是**：
在一個**特定事件**發生的當下——流動性被掃、價格穿過一個大家都看得到的
價位——簿口的反應是不是不一樣。這正是當時預註冊留下的那條路（「擠壓
事件條件化」），也是 §0.98 之後唯一還沒關門的方向。

**這份腳本產出的不是判決，是「值不值得開一個前瞻時鐘」。**
理由是樣本：撤單資料只覆蓋 42 天（BTC 56 天），重疊窗內只有約 270 筆
成交。這個 n 在探索上夠看形狀，在判決上不夠——而且它是**回測窗內的
探索**，正是 §0.92 規則 3 與 C/D 陷阱要擋的形狀。任何看起來好的東西，
**只能拿去註冊，不能拿去用**。

資料的兩個硬限制（先寫在這裡，結論不得超過它們）
------------------------------------------------
1. **「撤單」其實是「流動性消失」**。`depth_delta_collector` 自己的註解：
   `Δqty = adds − cancels − fills`，而 fills 在只有 L2 的情況下**分不出來**，
   所以 `cancel_qty = Σ max(0, −Δqty)`。價格穿過掛單造成的**成交**會被
   記成撤單。這對本研究特別致命，因為掃單的定義就是價格穿過去——所以
   「掃單時撤單量大」有一部分是同義反覆。控制組 F0 就是為此而設。
2. **只有全側總量，沒有逐價位**（§0.95）。所以問不了「延續方向前方的
   價位有沒有被抽掉」，只能問「這一側整體在補還是在撤」。

宣告的假設（寫在跑之前；d=+1 表示低點被掃、我們做多，我方＝bid）
-----------------------------------------------------------------
  H1 我方補單 refill  = our_add / (our_add + our_cancel)
     機制：掃到低點之後買方還在補掛單＝這個價位真的有人要守。
     **預測 netR 隨它上升。**
  H2 對手撤退 opp_pull = opp_cancel / (opp_cancel + opp_add)
     機制：反轉方向前面的掛單在消失＝阻力讓開。**預測 netR 隨它上升。**
  H3 淨傾斜 net = ((our_add−our_cancel) − (opp_add−opp_cancel)) / 總活動量
     H1 與 H2 的合體。**預測 netR 隨它上升。**
  F0 撤單強度（控制組）shock = 窗內每分鐘總撤單 / 前 6 小時基線
     **預測：對方向沒有效應。** 這是已知只跟波動有關的那個量
     （2026-07-29 TEST B）。**如果 F0 看起來跟 H1–H3 一樣強，就代表這
     整組東西量的是波動，不是方向**——那時三個假設一律作廢。

時點（嚴格早於成交，沿用 `pierce` 的規矩）
------------------------------------------
特徵一律取**掃單 bar j 的那 60 分鐘**。成交發生在 j+1..j+W 的某根 bar 的
盤中，所以 bar j 的整個小時嚴格更早。基線取 j 之前 6 小時。窗內至少要有
50 分鐘、基線至少 300 分鐘的紀錄，否則整筆剔除。

宇宙
----
**主樣本＝core9 裡有撤單資料的 7 個幣**（SOL / AVAX 沒錄）。
**另外單獨報一組宇宙外複製**（SUI / UNI / AAVE，有撤單資料但不在 core9）
——事前宣告、分開報告、**不與主樣本合池**。

「值得註冊前瞻」的門檻（現在寫死）
----------------------------------
  方向與宣稱一致 ∧ 主樣本 ≥5/7 幣同號 ∧ 宇宙外三幣同號 ∧ F0 明顯較弱。
四項全中才寫成一個前瞻變體；缺一就是「這條路也走完了」。
**不論結果如何，都不得就地改變任何凍結規則。**

Run: python research/sweep_failure/cancel_at_sweep.py
Out: research/results/sweep_cancel_at_sweep.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["SLIP"] = "0"
import sweep_core as SC              # noqa: E402
from sweep_forward import SCEN       # noqa: E402
import room_ahead as RA              # noqa: E402
from ict_entry import sweep_bar_map  # noqa: E402
from shared.db import get_db_conn    # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/sweep_cancel_at_sweep.json"
PRIMARY = ["BTC", "ETH", "BNB", "XRP", "DOGE", "ADA", "LINK"]   # core9 ∩ 有資料
OUTSIDE = ["SUI", "UNI", "AAVE"]                                # 宇宙外複製
WIN = 3600
BASE_H = 6
MIN_WIN, MIN_BASE = 50, 300


def vals(r):
    return list(r.values()) if isinstance(r, dict) else list(r)


def load_depth(sym):
    cn = get_db_conn()
    cu = cn.cursor()
    cu.execute("""SELECT minute_start_ms, bid_add_qty, bid_cancel_qty,
                         ask_add_qty, ask_cancel_qty
                  FROM depth_deltas_1m WHERE canonical_symbol=%s""",
               (f"{sym}-USD",))
    out = {}
    for r in cu.fetchall():
        v = vals(r)
        out[int(v[0]) // 1000] = (float(v[1]), float(v[2]), float(v[3]), float(v[4]))
    cu.close()
    cn.close()
    return out


def build(sym):
    p = RA.CACHE / f"{sym}USDT_1h.csv"
    if not p.exists():
        return []
    bars = SC.load_csv(str(p))
    dep = load_depth(sym)
    if not dep:
        return []
    idx = {b[0]: i for i, b in enumerate(bars)}
    jmap = sweep_bar_map(bars)
    s = SCEN["A"]
    rows = []
    for fill_ts, _e, r, lvl, atr, stopped, _pi, side in SC.backtest_symbol(bars):
        f = idx.get(fill_ts)
        j = jmap.get(f) if f is not None else None
        if j is None:
            continue
        tj = bars[j][0]
        win = [dep[t] for t in range(tj, tj + WIN, 60) if t in dep]
        bas = [dep[t] for t in range(tj - BASE_H * 3600, tj, 60) if t in dep]
        if len(win) < MIN_WIN or len(bas) < MIN_BASE:
            continue
        d = 1 if side == "LONG" else -1
        # our side supports the trade: LONG -> bids (cols 0,1); SHORT -> asks
        oa = sum(w[0] if d == 1 else w[2] for w in win)
        oc = sum(w[1] if d == 1 else w[3] for w in win)
        pa = sum(w[2] if d == 1 else w[0] for w in win)
        pc = sum(w[3] if d == 1 else w[1] for w in win)
        act = oa + oc + pa + pc
        if act <= 0:
            continue
        bcan = sum(w[1] + w[3] for w in bas) / len(bas)
        wcan = (oc + pc) / len(win)
        cost = (s["entry"] + (s["sexit"] if stopped else s["texit"])) \
            / 1e4 * lvl / (SC.DIS * atr)
        rows.append(dict(
            sym=sym, ts=fill_ts, net=r - cost,
            refill=oa / (oa + oc) if oa + oc > 0 else 0.5,
            opp_pull=pc / (pc + pa) if pc + pa > 0 else 0.5,
            netskew=((oa - oc) - (pa - pc)) / act,
            shock=wcan / bcan if bcan > 0 else 1.0))
    return rows


def cells(rows, key, name, syms):
    v = sorted(r[key] for r in rows)
    lo_q, hi_q = v[len(v) // 3], v[2 * len(v) // 3]
    g = lambda sel: [r for r in rows if sel(r)]                     # noqa: E731
    m = lambda a: sum(x["net"] for x in a) / len(a) if a else float("nan")  # noqa: E731
    lo, mid, hi = (g(lambda r: r[key] <= lo_q),
                   g(lambda r: lo_q < r[key] <= hi_q),
                   g(lambda r: r[key] > hi_q))
    gap = m(hi) - m(lo)
    per = {}
    for sym in syms:
        a = [r for r in lo if r["sym"] == sym]
        b = [r for r in hi if r["sym"] == sym]
        per[sym] = (m(b) - m(a)) if a and b else None
    pos = sum(1 for x in per.values() if x is not None and x > 0)
    got = sum(1 for x in per.values() if x is not None)
    ci = RA.gap_ci([(r["ts"], r["net"], 1 if r[key] > hi_q else 0)
                    for r in rows if r[key] > hi_q or r[key] <= lo_q])
    print(f"  {name:<16}n={len(rows):<5} 低 {m(lo):+.4f} 中 {m(mid):+.4f} "
          f"高 {m(hi):+.4f}   差 {gap:+.4f}")
    print(f"  {'':<16}幣正 {pos}/{got}   CI [{ci[0]:+.4f},{ci[1]:+.4f}]")
    return dict(n=len(rows), cut=[lo_q, hi_q], lo=m(lo), mid=m(mid), hi=m(hi),
                gap=gap, sym_positive=pos, sym_measured=got, ci=list(ci),
                per_symbol=per)


def main() -> int:
    print("=" * 92)
    print("  掃單當下的簿口行為 —— 探索（不是判決），撤單資料 42 天")
    print("=" * 92)
    res = {}
    for label, syms in (("主樣本 core9∩有資料", PRIMARY),
                        ("宇宙外複製", OUTSIDE)):
        rows = []
        for sym in syms:
            sub = build(sym)
            print(f"  {sym:<6} {len(sub):>4} 筆")
            rows += sub
        if not rows:
            continue
        rows.sort(key=lambda r: r["ts"])
        print(f"\n  == {label} ==  n={len(rows)}  "
              f"meanR {sum(r['net'] for r in rows)/len(rows):+.4f}")
        blk = {}
        for k, nm in (("refill", "H1 我方補單"), ("opp_pull", "H2 對手撤退"),
                      ("netskew", "H3 淨傾斜"), ("shock", "F0 撤單強度*對照")):
            blk[k] = cells(rows, k, nm, syms)
        res[label] = blk
        print()
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
