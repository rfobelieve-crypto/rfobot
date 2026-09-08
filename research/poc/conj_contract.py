# -*- coding: utf-8 -*-
"""交會線交給產品端的契約 —— **產品端不得推導任何東西**

===========================================================================
為什麼契約要長這樣
===========================================================================
使用者（2026-09-08）：「放上去之後記得檢查策略有完整複製過來喔」。

**正確的答案不是「仔細複製」，是「根本不複製」。** 這個專案已經被
「規格用文字描述、產品端照著實作、然後兩邊漂開」咬過：

    2026-08-26  §0.59 的規格寫「只在 RANGING 開火」,產品端照做,
                然後問「為什麼把 TREND_DOWN 也擋掉」——**規格真的漏了一格**
    2026-08-26  `regime_cell` 四層都對,只有回傳的 dict 沒有它;產品端
                拿到 undefined,擋掉每一筆訊號,skip 原因還記成 regime
    本 session  偵測層的第二份實作,五個 bug,對照測試自己改了三版

所以交會線的交接**不傳規格,傳訂單**。`conj_intents` 的每一列就是一張
可以直接送出去的單:方向、張數、停損價、出場時刻全部在裡面,而且全部是
研究端算的。產品端要做的只有「照抄、送出、回報成交」。

**沒有東西被複製,所以沒有東西會漂開。**

===========================================================================
契約
===========================================================================
產品端**可以**讀（而且只能讀）這些欄位：

    canonical_symbol  哪個標的（產品端只需要一張靜態的市場代號對照表）
    side              LONG / SHORT —— **不得自己判方向**
    size_base         下多少（幣為單位）—— **不得自己算 sizing**
    stop_price        停損觸發價 —— **不得自己算停損距離**
    exit_ts           什麼時候平倉 —— **不得自己算持有時間**
    expires_ts        超過這個時刻就不要送了（陳舊的單比沒有單更糟）
    ref_price         研究端假設的進場價 —— **只用來對帳,不是限價**
    anchor_ts         事件錨點,回報成交時要帶回來

產品端**不得**知道、也不需要知道的（全部留在研究端）：

    ATR 的定義與計算          事件偵測的門檻（p99、滾動 30 日）
    價位（樞紐）的定義        掃單/穿越的判定
    impulse 的方向定義        停損距離 STOP_ATR
    持有時間 HOLD_MIN         同時持倉/單幣/單日的上限

    ^ 這些**一個都不該出現在 jarvis 的程式碼裡**。出現了就是第二份實作,
      而它會在研究端改參數的那天安靜地不同意。

產品端**必須**回報的（這是本次實盤唯一要買的東西）：

    實際成交價、成交時刻、實際成交數量、送單時刻
    -> 判準:實際成交價 vs ref_price 差幾 bps
       <= 2 bps  成本模型站得住,2 分鐘死線的算術有效
       >= 5 bps  死線與淨值全部要重算

===========================================================================
本檔驗什麼
===========================================================================
    C1  意圖列**自足**:上面那張「可以讀」的清單,每一欄都存在且非空
    C2  意圖列**足以下單**:不需要任何研究端常數就能組出一張完整的單
    C3  **反向證明**:拿掉任一必要欄位,C2 必須失敗
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "sweep_failure"))
sys.path.insert(0, str(HERE.parents[1]))
import conj_watch as cw  # noqa: E402

# 產品端可以讀的欄位（白名單，加欄位要連同契約一起改）
ALLOWED = ["canonical_symbol", "side", "size_base", "stop_price",
           "exit_ts", "expires_ts", "ref_price", "anchor_ts"]
# 下一張單最少需要的
REQUIRED_FOR_ORDER = ["canonical_symbol", "side", "size_base",
                      "stop_price", "exit_ts", "expires_ts"]
# 絕不可出現在產品端的研究端常數
FORBIDDEN_CONSTANTS = ["STOP_ATR", "HOLD_MIN", "PIVOT", "MERGE_GAP",
                       "COOLDOWN", "thr_delta", "thr_vol", "vol_base_tod",
                       "impulse", "atr_h14"]


def build_order(row):
    """只用白名單欄位組出一張單。少任何一欄就 raise ——這就是 C2/C3。"""
    miss = [k for k in REQUIRED_FOR_ORDER if row.get(k) in (None, "")]
    if miss:
        raise KeyError(f"意圖列缺欄位,無法下單: {miss}")
    if row["side"] not in ("LONG", "SHORT"):
        raise ValueError(f"方向不合法: {row['side']}")
    if not (float(row["size_base"]) > 0):
        raise ValueError("張數必須 > 0")
    return {
        "market": row["canonical_symbol"],
        "side": "buy" if row["side"] == "LONG" else "sell",
        "sizeBase": float(row["size_base"]),
        "stopTrigger": float(row["stop_price"]),
        "closeAtMs": int(row["exit_ts"]),
        "dropAfterMs": int(row["expires_ts"]),
    }


def main():
    now = int(time.time() * 1000)
    e = ("BTC-USD", now - 120_000, now, 60_000, "S+D+V", 1,
         79000.0, 79200.0, 280.0)
    row = cw.make_intent(e, now)
    ok = True

    print("=== C1 意圖列自足（白名單每一欄都在且非空）===")
    for k in ALLOWED:
        v = row.get(k)
        good = v is not None and v != ""
        ok &= good
        print(f"  {k:18s} = {str(v)[:34]:34s} {'PASS' if good else '**FAIL**'}")

    print()
    print("=== C2 只用白名單就組得出一張完整的單 ===")
    try:
        o = build_order(row)
        for k, v in o.items():
            print(f"  {k:14s} {v}")
        print("  -> PASS")
    except Exception as ex:
        ok = False
        print(f"  -> **FAIL** {type(ex).__name__}: {ex}")

    print()
    print("=== C3 反向證明：拿掉任一必要欄位必須失敗 ===")
    for k in REQUIRED_FOR_ORDER:
        bad = dict(row)
        bad[k] = None
        try:
            build_order(bad)
            print(f"  拿掉 {k:18s} -> **FAIL（竟然還組得出來）**")
            ok = False
        except Exception as ex:
            print(f"  拿掉 {k:18s} -> PASS（{type(ex).__name__}）")

    print()
    print("=== 交給 jarvis 的規格（照抄，不要重寫）===")
    print("  可讀欄位 :", ", ".join(ALLOWED))
    print("  下單必要 :", ", ".join(REQUIRED_FOR_ORDER))
    print("  必須回報 : 實際成交價 / 成交時刻 / 實際數量 / 送單時刻 + anchor_ts")
    print("  **不得出現在 jarvis 程式碼裡的研究端常數**:")
    print("   ", ", ".join(FORBIDDEN_CONSTANTS))
    print("    出現任何一個 = 第二份實作 = 研究端改參數那天會安靜地不同意")

    print()
    print("=== 判決 ===")
    print("  -> " + ("**PASS —— 契約成立：產品端不需要推導任何東西**"
                     if ok else "**FAIL —— 契約不成立，不得交接**"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
