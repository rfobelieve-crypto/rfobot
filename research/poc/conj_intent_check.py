# -*- coding: utf-8 -*-
"""訂單意圖的已知答案對照——**這些數字會變成真的訂單**

為什麼一定要有這支
    交會事件約每 6-8 小時一個，所以 `conj_watch` 正常跑一輪
    `events=0` -> `intents=0`，**意圖那段程式碼平常一次都不會被執行**。
    那個 0 同時代表「這分鐘沒事件」與「意圖算錯了／根本沒算」，
    畫面上一模一樣（mistake.md 2026-08-26；本 session 已經踩過一次，
    當時是 `conj_watch_inject` 抓到發射路徑在真實事件上 2.8% 命中）。

    而這次的下游是**真錢**：張數、方向、停損價這三個算錯就是直接賠。

判準（跑之前寫死，任一不過就不得接真錢）
    I1  方向：LONG 的停損必須在進場價**下方**、SHORT 在**上方**
    I2  停損距離 = STOP_ATR x ATR，誤差 < 1e-9
    I3  張數 x 進場價 = NOTIONAL_USD，誤差 < 0.01 美元
    I4  出場時刻 = 錨點 + HOLD_MIN 分鐘；過期時刻 = 錨點 + INTENT_TTL_S 秒
    I5  三道煞車（同時持倉／單幣／單日）各自**擋得住**——反向證明，
        不是只看它平常放行
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


def main():
    now = int(time.time() * 1000)
    ok_all = True

    # e = (sym, event_ts, det_ts, latency, sig, direction, level, px, atr)
    cases = [
        ("BTC-USD", now - 120_000, now, 60_000, "S+D+V", 1, 79000.0,
         79200.0, 280.0),                       # LONG
        ("SOL-USD", now - 120_000, now, 60_000, "S+D", -1, 104.5,
         104.0, 0.83),                          # SHORT
    ]
    print("=== I1-I4 逐筆核對（這些數字會變成真的訂單）===")
    for e in cases:
        it = cw.make_intent(e, now)
        d = 1 if e[5] > 0 else -1
        px, atr = float(e[7]), float(e[8])
        i1 = (it["stop_price"] < px) if d > 0 else (it["stop_price"] > px)
        i2 = abs(abs(px - it["stop_price"]) - cw.STOP_ATR * atr) < 1e-9
        i3 = abs(it["size_base"] * px - cw.NOTIONAL_USD) < 0.01
        i4 = (it["exit_ts"] == e[1] + cw.HOLD_MIN * 60_000
              and it["expires_ts"] == e[1] + cw.INTENT_TTL_S * 1000)
        ok_all &= i1 and i2 and i3 and i4
        print(f"  {it['canonical_symbol']:9s} {it['side']:5s} "
              f"進場 {px:,.4f}  停損 {it['stop_price']:,.4f}  "
              f"距離 {abs(px-it['stop_price']):,.4f} = "
              f"{abs(px-it['stop_price'])/atr:.3f} ATR")
        print(f"    張數 {it['size_base']:.8f} x {px:,.4f} = "
              f"${it['size_base']*px:,.4f}（目標 ${cw.NOTIONAL_USD:,.0f}）")
        print(f"    出場 +{(it['exit_ts']-e[1])/60000:.0f} 分   "
              f"過期 +{(it['expires_ts']-e[1])/1000:.0f} 秒")
        print(f"    I1 方向 {'PASS' if i1 else '**FAIL**'}   "
              f"I2 距離 {'PASS' if i2 else '**FAIL**'}   "
              f"I3 名目 {'PASS' if i3 else '**FAIL**'}   "
              f"I4 時刻 {'PASS' if i4 else '**FAIL**'}")

    # ---- I6 相對成交的三個欄位（2026-09-08 實盤體檢加）----
    print()
    print("=== I6 stop_dist / hold_ms / intent_id（產品端套在成交上的三個數）===")
    for e in cases:
        it = cw.make_intent(e, now)
        atr = float(e[8])
        a1 = abs(it["stop_dist"] - cw.STOP_ATR * atr) < 1e-9
        a2 = it["hold_ms"] == cw.HOLD_MIN * 60_000
        a3 = it["intent_id"] == f"{e[0]}:{int(e[1])}"
        ok_all &= a1 and a2 and a3
        print(f"  {it['canonical_symbol']:9s} stop_dist {it['stop_dist']:.6g} "
              f"= {it['stop_dist']/atr:.3f} ATR {'PASS' if a1 else '**FAIL**'}   "
              f"hold_ms {it['hold_ms']} {'PASS' if a2 else '**FAIL**'}   "
              f"intent_id {it['intent_id']} {'PASS' if a3 else '**FAIL**'}")

    # ---- I7 live ATR 與研究 ATR 是同一顆（已知答案對照）----
    print()
    print("=== I7 atr_h14_now 截到 parquet 同一小時 vs parquet 的 atr_h14（同一顆 -> 逐位相同）===")
    import pandas as pd
    for sym in ("BTC", "ETH", "SOL"):
        b = pd.read_parquet(HERE / "data" / "bars" / f"{sym}.parquet",
                            columns=["ts", "atr_h14"])
        last_ts = int(b["ts"].iloc[-1])
        ref = float(b["atr_h14"].dropna().iloc[-1])
        # 第一版拿「此刻」的 live 值去比 parquet 最後一根：差 6-13% -> FAIL。
        # 查證後三項全 0.0000%（配方、資料源、同一小時截斷）——差的是 cache
        # 多了一個剛收盤、剛好很大的小時。**比錯時點不是配方錯**。
        # 所以對照必須截到 parquet 同一小時；那時兩者是同一顆，容差 0.01%。
        live = cw.atr_h14_now(sym, hi_ts=last_ts)
        rel = abs(live - ref) / ref if (live and ref) else float("inf")
        good = rel < 1e-4
        ok_all &= good
        print(f"  {sym:4s} live {live:.6g}  parquet {ref:.6g}  差 {rel*100:.3f}% "
              f"-> {'PASS' if good else '**FAIL —— 不是同一個配方**'}")

    # ---- I5 三道煞車的反向證明 ----
    print()
    print("=== I5 煞車反向證明（要看到它們擋，不是只看它們放行）===")

    class FakeCur:
        def __init__(self, o, n):
            self.o, self.n, self.q = o, n, ""

        def execute(self, q, a=None):
            self.q = q

        def fetchall(self):
            return list(self.o.items())

        def fetchone(self):
            return [self.n]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class FakeConn:
        def __init__(self, o, n):
            self.o, self.n = o, n

        def cursor(self):
            return FakeCur(self.o, self.n)

    base = [cw.make_intent(cases[0], now), cw.make_intent(cases[1], now)]
    trials = [
        ("無限制（應全放行）", {}, 0, 2),
        (f"同時持倉已滿 {cw.MAX_CONCURRENT}", {"ETH-USD": cw.MAX_CONCURRENT},
         0, 0),
        ("BTC 已有部位（只擋 BTC）", {"BTC-USD": cw.MAX_PER_SYMBOL}, 0, 1),
        (f"今日已達 {cw.MAX_DAILY}", {}, cw.MAX_DAILY, 0),
    ]
    for lab, openby, ntoday, expect in trials:
        got = len(cw.intent_gate(FakeConn(openby, ntoday), base))
        good = got == expect
        ok_all &= good
        print(f"  {lab:24s} 放行 {got}（預期 {expect}）"
              f" -> {'PASS' if good else '**FAIL**'}")

    print()
    print("=== 判決 ===")
    print("  -> " + ("**PASS —— 意圖層可以接真錢**" if ok_all else
                     "**FAIL —— 不得接真錢**"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
