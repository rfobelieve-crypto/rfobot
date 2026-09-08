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
