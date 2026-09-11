# -*- coding: utf-8 -*-
"""Gate 0：§0.75 換成「不對沖 + 掛單進場」的成本（2026-09-11）

===========================================================================
為什麼重算
===========================================================================
外部閱讀〈Ultimate Crypto Arbitrage Guide〉指出三件事，而 §0.75 三件都踩反：

  1. **現貨/永續套利基本上不該對沖** —— 對沖是兩倍成本
  2. **進場要用掛單不用吃單** —— 「套利機會往往是被 maker fill 創造出來的」
  3. **兩邊不會在中間相遇，小所做掉幾乎全部的移動**

§0.75 的成本模型 `round_trip_bps = 2 x (兩腿)` = **四次穿越**，
而判決是「帶寬不夠付費用」。**那個判決是在一個付了多餘成本的構造上量的。**

**這支只算成本那半（Gate 0），不算損益。** 毛利直接沿用既有模型的
`band/2`，不重新推導 —— 重推毛利就變成在找好看的數字了。

===========================================================================
先驗他的核心假設：到底誰在移動
===========================================================================
「不對沖」整個站在「小所做掉幾乎全部移動」這個前提上。
**那個前提在我們的配對上成不成立是可以量的**，而且必須先量 ——
如果是大所在動，不對沖就是把套利換成了裸部位。

量法：取價差擴大之後的收斂窗，看這段期間
  |Δ小所中價| vs |Δ大所中價|
小所佔比接近 1 -> 假設成立；接近 0.5 -> 兩邊各動一半，對沖是必要的。

**2026-09-11 更正這一關的判讀（量測本身沒有改）**：讀完他的
〈Small Trader Alpha #6: Perpetual Arbitrage〉付費全文之後，他的主張
**是條件於成交量的**，不是無條件的：

> 「大單打進來時，交易所會在中間相遇（**成交量加權**）；
>   小單打進來推動價格時，**只有一家會動**，另一家完全無感。」

所以「大單 50/50 ＋ 小單 100/0」的混合，**無條件中位數就會落在 0.5 附近**
—— 本支量到的 0.485–0.543 與他的主張**相容**，不構成反證。
原本的結論句「他的核心假設失敗」**過強，已撤回**（docs/external_reading.md
第 13 則）。

**而我們現在量不到那個條件**：`minutes.csv` 只有報價（`e_bid_sz` 是掛單量），
**沒有任何成交量欄位**。所以這一關現在能說的只有：
「平均而言兩所各動一半，**他的條件式主張我們沒有能力檢定**。」
—— 這是一個**錄製缺口**（TODO §1.22），而錄製缺口不可回填。

**對判決沒有影響**：不對沖仍然不做，但理由換成他自己的階梯
（taker/taker -> **maker/taker，仍然全程對沖** -> maker/maker），
而本支的「誠實版」就是第一級。

===========================================================================
新構造的成本，逐桶說明改了什麼
===========================================================================
    1 費用    兩腿 x 進出 -> **一腿 x (掛單進 + 吃單出)**
    2 滑價    掛單進場沒有滑價 -> **只剩出場那次**（這裡取原值的一半）
    3 資金    只用一個場館的保證金（略減，這裡保守維持原值）
    4 carry   維持
    5 轉帳    **歸零** —— 不跨場館就不需要搬錢
    6 ops     維持
    7 尾部    **歸零** —— 該桶的定義就是「第二腿失敗的損失」，沒有第二腿就沒有

===========================================================================
這支**不會**算到的東西（必須寫出來）
===========================================================================
**不對沖 = 裸的方向曝險。** 舊模型沒有這一桶，因為舊構造是對沖的。
它的**期望值是零**（公平賭注）但**變異數很大**，所以它不改變淨值的期望，
**但會把夏普從套利等級打到策略等級**。本支會把它的量級算出來並列，
但不把它併進 net —— 因為它不是成本，是風險。

    python research/gate0_arb_unhedged.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import arb_home as AH                          # noqa: E402
AH.add_to_path()          # 讓 arblib 可以 import（唯一知道它在哪的地方）

OUT = ROOT / "research" / "results" / "gate0_arb_unhedged.json"


def fee(venue, maker, rebate=True):
    from arblib.fees import fee_bps
    return fee_bps(venue, maker, rebate)


def who_moves(pid, min_band_bps=5.0):
    """收斂窗裡，小所（entropy）貢獻了多少比例的移動。

    取「價差絕對值從高於門檻回落到門檻一半以內」的每一段，
    比較兩邊中價在該段的位移絕對值。
    """
    f = AH.LOGS / pid / "minutes.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    need = {"entropy_bid", "entropy_ask", "hedge_bid", "hedge_ask",
            "premium_close_bps"}
    if not need.issubset(d.columns):
        return None
    d = d.dropna(subset=list(need))
    if len(d) < 50:
        return None
    e = (d.entropy_bid + d.entropy_ask) / 2.0
    h = (d.hedge_bid + d.hedge_ask) / 2.0
    p = d.premium_close_bps.values
    shares, n_ev = [], 0
    i, N = 0, len(d)
    while i < N:
        if abs(p[i]) >= min_band_bps:
            j = i + 1
            while j < N and abs(p[j]) > min_band_bps / 2.0:
                j += 1
            if j < N and j > i:
                de = abs(e.iloc[j] - e.iloc[i]) / max(e.iloc[i], 1e-9)
                dh = abs(h.iloc[j] - h.iloc[i]) / max(h.iloc[i], 1e-9)
                if de + dh > 1e-9:
                    shares.append(de / (de + dh))
                    n_ev += 1
            i = j + 1
        else:
            i += 1
    if not shares:
        return None
    s = np.array(shares)
    return dict(n_events=n_ev, share_median=float(np.median(s)),
                share_mean=float(s.mean()),
                frac_above_0_7=float((s > 0.7).mean()))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    cm = json.loads((AH.RESULTS / "arb_cost_model.json")
                    .read_text(encoding="utf-8"))
    res = dict(asof=time.strftime("%Y-%m-%d %H:%M:%S"),
               source_mode=cm.get("mode"), size_usd=cm.get("size_usd"),
               pairs={})

    print("=== 第一關：誰在移動（他的核心假設）===")
    print("%-10s %8s %12s %12s %14s" % ("配對", "事件數", "小所佔比中位",
                                        "均值", ">0.7 的比例"))
    moves = {}
    for pid in cm["pairs"]:
        w = who_moves(pid)
        moves[pid] = w
        if w:
            print("%-10s %8d %12.3f %12.3f %14.1f%%"
                  % (pid, w["n_events"], w["share_median"], w["share_mean"],
                     100 * w["frac_above_0_7"]))
        else:
            print("%-10s %8s" % (pid, "資料不足"))
    res["who_moves"] = moves

    print("\n=== 第二關：新構造的成本（只算成本，毛利沿用舊模型）===")
    print("%-10s %8s %9s %9s %9s %9s %9s"
          % ("配對", "毛利", "舊成本", "舊淨值", "新成本", "新淨值", "翻轉?"))
    for pid, v in cm["pairs"].items():
        sp, bk = v["spec"], v["buckets_bps"]
        a = sp["leg_a"]
        # 在哪一腿交易：小所 = entropy = leg_a（recorder 的命名）
        f_mk, f_tk = fee(a, True, sp.get("rebate", True)), fee(a, False, sp.get("rebate", True))
        new = {
            "1_fees": f_mk + f_tk,                    # 一腿，掛單進 + 吃單出
            "2_slippage": 0.5 * (bk.get("2_slippage") or 0.0),
            "3_capital": bk.get("3_capital") or 0.0,
            "4_carry": bk.get("4_carry") or 0.0,
            "5_transfer": 0.0,                        # 不跨場館
            "6_ops": bk.get("6_ops") or 0.0,
            "7_tail": 0.0,                            # 沒有第二腿可失敗
        }
        new_cost = sum(x for x in new.values() if x is not None)
        gross = v["gross_bps"]
        old_net, new_net = v["net_bps"], gross - new_cost
        res["pairs"][pid] = dict(leg=a, gross_bps=gross,
                                 old_cost=v["total_cost_bps"], old_net=old_net,
                                 new_buckets=new, new_cost=round(new_cost, 3),
                                 new_net=round(new_net, 3),
                                 flipped=bool(old_net <= 0 < new_net))
        print("%-10s %8.2f %9.2f %+9.2f %9.2f %+9.2f %9s"
              % (pid, gross, v["total_cost_bps"], old_net, new_cost, new_net,
                 "**是**" if old_net <= 0 < new_net else ""))

    print("\n=== 沒有被算進去的那一桶：裸方向曝險 ===")
    print("不對沖 = 持有期間吃下標的自己的移動。**期望值是零，變異數不是。**")
    print("%-10s %10s %12s %12s %12s"
          % ("配對", "持有分鐘", "每筆雜訊bps", "新淨值bps", "每筆夏普"))
    for pid, v in cm["pairs"].items():
        sp = cm["pairs"][pid]["spec"]
        hold = sp.get("hold_minutes") or 0
        # 用該配對自己的分鐘中價算實現波動，換算到持有期
        f = AH.LOGS / pid / "minutes.csv"
        sig = None
        if f.exists():
            d = pd.read_csv(f)
            if {"entropy_bid", "entropy_ask"}.issubset(d.columns):
                m = ((d.entropy_bid + d.entropy_ask) / 2.0).dropna()
                r = np.diff(np.log(m[m > 0]))
                if len(r) > 100:
                    sig = float(np.std(r) * np.sqrt(max(hold, 1)) * 1e4)
        nn = res["pairs"][pid]["new_net"]
        res["pairs"][pid]["noise_bps_per_trade"] = sig
        res["pairs"][pid]["sharpe_per_trade"] = (nn / sig) if sig else None
        print("%-10s %10s %12s %+12.2f %12s"
              % (pid, hold, ("%.1f" % sig) if sig else "—", nn,
                 ("%.3f" % (nn / sig)) if sig else "—"))

    print("\n**本支只算成本。毛利沿用舊模型的 band/2，沒有重新推導。**")
    print("**裸方向曝險沒有併進 net** —— 它不是成本是風險，期望值為零。")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print("written -> " + str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
