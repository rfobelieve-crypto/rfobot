# -*- coding: utf-8 -*-
"""清算資料的橋：`liq_events`（逐筆）-> 每幣每分鐘，並對舊表做已知答案對照

===========================================================================
為什麼需要它
===========================================================================
交會線（TODO §1.03）的核心機制主張是「**強制流**推動價格」。2026-09-08 用
獨立的清算資料驗過名字站得住（旗標開火分鐘落在清算額 p95 以上的比例
47-57%，基準 5%，約 10x lift）。**但那只有 BTC/ETH**，而且沒排除共同因子
——「大行情」可能同時造成極端流與清算，不必然是清算造成後續延續。

要分辨這兩者需要一個對照組：**同樣量能爆發、但沒有清算**的分鐘。
那需要 core9 全部的清算資料。

**這裡沒有東西壞掉，只是有兩張表而且用錯了那張：**

    liquidation_1m   OKX + Binance   **硬編碼只有 BTC/ETH 兩個符號映射**
                     （`market_data/adapters/liquidation_collector.py` 的
                      OKX_SYMBOL_MAP / BINANCE_SYMBOL_MAP 各只有兩筆
                      —— 不是限流，是寫死的）160 天
    liq_events       OKX + Bybit     456 個標的、core9 全覆蓋，2026-09-05 起
                     （Binance fstream 從這台機器連不上，`liq_recorder.py`
                      檔頭有用 aggTrade 對照組證明過是連線被擋不是清算稀疏）

所以修法不是「修錄製器」，是**把新表接起來**，然後等它長出樣本。

===========================================================================
已知答案對照（判準跑之前寫死）
===========================================================================
兩張表**只有 OKX 這個場館重疊**（舊 = OKX+Binance，新 = OKX+Bybit），
所以對照必須**限定 OKX 那一側**，否則場館組合不同會系統性差一截，
而那個差會被誤讀成橋寫錯了。

母體：BTC/ETH、兩表都有資料的分鐘、且至少一邊有清算。

    L1  逐分鐘總額的 Spearman 相關 >= 0.90
    L2  非零分鐘的「新/舊」比值中位落在 [0.80, 1.25]（同一個尺度）
    L3  「這分鐘有沒有清算」的 Jaccard 相似度 >= 0.80
    任一不過 -> **橋是錯的**，之後任何用它的結論都不解讀。

這一關存在的理由：本 session 已經連續三次被「對照測試自己有病」咬到
（`conj_watch_parity` 改了三版才有分辨力）。新寫的儀器在推翻或支持任何
結論之前，先在答案已知的資料上跑一次。

===========================================================================
混淆對照的註冊（criteria 現在寫死，資料還沒到）
===========================================================================
**問題**：「量能爆發但**沒有**清算」的分鐘，事後反應跟「量能爆發**且有**
清算」一樣嗎？

    一樣   -> 「強制流」是多餘的標籤，真正的變數是「大行情」
    不一樣 -> 清算是機制的一部分

**設計（事前寫死，不得在看到資料後調整）**
    母體    core9 的交會事件（現行註冊定義，NO-OI）
    分組    事件錨點 ±5 分鐘內的清算名目金額：
            高 = 該幣非零清算分鐘的 p75 以上；低 = 該分鐘清算額為 0
            （**中間帶不判**——避免用連續變數的切點做搜尋）
    結果    與現行一致：impulse x (close(t+60) - 進場價) / ATR，
            進場 +2 分、停損 1.0 ATR
    判準    D 高低兩組的差，日聚類 bootstrap CI 不含零 -> 清算帶資訊
            且逐幣 >= 6/9 同號
    對照    另跑一個安慰劑：用**事件前 60-65 分鐘**的清算額分組
            （同樣的量、錯的時窗）。安慰劑顯著 = 抓到的是幣別/時段的
            共同因子，不是事件當下的清算。

**功效（用判決那台機器實測，不是手算——mistake.md 2026-09-06）**
    逐事件 SD = **1.382 ATR**（我第一版手算假設 0.900，低估 54%）
    日聚類 bootstrap 的兩組差 SE（n=836 的 BTC/ETH 母體）= 0.096 ATR
    **DEFF = 1.01** —— 日聚類在這裡幾乎不膨脹（836 事件散在 508 天，
    平均 1.6/天），與掃單線「一根 1h bar 塞 27 個事件」的情況完全不同。

    外推：
        n= 400（3.5 個月）  MDE 0.272 ATR
        n= 800（7.0 個月）  MDE 0.192 ATR
        n=1200（10.4 個月） MDE 0.157 ATR

**這個對照能回答什麼、不能回答什麼（寫在跑之前）**
    交會效應本身是 +0.23 ATR。
    * 若清算是**全部**的機制 → 無清算組應該 ~0、有清算組 ~0.35+，
      差約 0.35 ATR → **n=400 就判得動**。
    * 若清算只是**一部分**（差 ~0.12 ATR）→ 需要 MDE <= 0.12 → n≈2000
      → **約 17 個月**。
    所以本對照**只對「清算是不是全部」有測量能力**，對「清算是不是一部分」
    沒有。判決文字必須照這個寫，不得把「沒測到」講成「沒有效應」
    （mistake.md 2026-09-04 的「無效判決 vs FAIL」）。

**不得跑早期試水**：BTC/ETH 在舊表的 160 天窗內約有 263 個事件、
MDE ≈ 0.335 ATR。那個解析度只有在效應極大時才有結論，而先看一眼再決定
要不要繼續，就是 §0.92 判掉 C/D 的那件事。**等 n=400 再跑，一次。**

Run:
    python research/poc/liq_bridge.py            # 建橋 + 已知答案對照 + 功效
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

OUT = HERE / "data" / "results"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
MIN_MS = 60_000


def minute_liq(venues=None, symbols=None):
    """`liq_events` 逐筆 -> 每幣每分鐘，欄位對齊 `liquidation_1m`。

    side 的約定與兩張表一致（`liq_recorder.py` 檔頭寫死）：
        SELL = 多單被強平（被迫賣出）-> liq_sell_usd
        BUY  = 空單被強平（被迫買回）-> liq_buy_usd
    """
    from shared.db import get_db_conn
    where, args = [], []
    if venues:
        where.append("venue IN (" + ",".join(["%s"] * len(venues)) + ")")
        args += list(venues)
    if symbols:
        where.append("symbol IN (" + ",".join(["%s"] * len(symbols)) + ")")
        args += list(symbols)
    w = ("WHERE " + " AND ".join(where)) if where else ""
    q = f"""SELECT symbol,
              FLOOR(ts_event/{MIN_MS})*{MIN_MS} AS window_start,
              SUM(CASE WHEN side='BUY'  THEN notional_usd ELSE 0 END) AS liq_buy_usd,
              SUM(CASE WHEN side='SELL' THEN notional_usd ELSE 0 END) AS liq_sell_usd,
              SUM(notional_usd) AS liq_total_usd,
              COUNT(*) AS liq_count
            FROM liq_events {w}
            GROUP BY symbol, window_start"""
    conn = get_db_conn()
    d = pd.read_sql(q, conn, params=args or None)
    conn.close()
    for c in ("liq_buy_usd", "liq_sell_usd", "liq_total_usd"):
        d[c] = d[c].astype(float)
    return d


def main():
    from shared.db import get_db_conn

    # ── 橋：只取 OKX（與舊表唯一重疊的場館）──
    new = minute_liq(venues=("okx",), symbols=("BTC", "ETH"))
    conn = get_db_conn()
    old = pd.read_sql(
        "SELECT canonical_symbol, window_start, liq_total_usd, liq_count "
        "FROM liquidation_1m WHERE canonical_symbol IN ('BTC-USD','ETH-USD')",
        conn)
    conn.close()
    old["symbol"] = old["canonical_symbol"].str.replace("-USD", "", regex=False)
    old["liq_total_usd"] = old["liq_total_usd"].astype(float)

    lo = max(new.window_start.min(), old.window_start.min())
    hi = min(new.window_start.max(), old.window_start.max())
    print("=== 清算橋：liq_events -> 每分鐘（已知答案對照）===")
    print(f"重疊窗 {pd.Timestamp(int(lo),unit='ms')} ~ "
          f"{pd.Timestamp(int(hi),unit='ms')}  "
          f"（{(hi-lo)/86400000:.2f} 天）")
    print("**只比 OKX**——舊表是 OKX+Binance、新表是 OKX+Bybit，"
          "場館組合不同會被誤讀成橋壞了")
    print()

    res = {"window": [int(lo), int(hi)], "checks": {}}
    ok_all = True
    for sym in ("BTC", "ETH"):
        a = new[(new.symbol == sym) & new.window_start.between(lo, hi)]
        b = old[(old.symbol == sym) & old.window_start.between(lo, hi)]
        m = pd.merge(a[["window_start", "liq_total_usd", "liq_count"]],
                     b[["window_start", "liq_total_usd", "liq_count"]],
                     on="window_start", how="outer",
                     suffixes=("_new", "_old")).fillna(0.0)
        both = (m.liq_total_usd_new > 0) & (m.liq_total_usd_old > 0)
        rho = m.liq_total_usd_new.corr(m.liq_total_usd_old, method="spearman")
        ratio = (m.liq_total_usd_new[both] / m.liq_total_usd_old[both]).median()
        A = set(m.window_start[m.liq_total_usd_new > 0])
        B = set(m.window_start[m.liq_total_usd_old > 0])
        jac = len(A & B) / max(len(A | B), 1)
        l1, l2, l3 = rho >= 0.90, 0.80 <= ratio <= 1.25, jac >= 0.80
        ok_all &= (l1 and l2 and l3)
        print(f"  {sym}: 分鐘 {len(m):,}（兩邊都有清算 {both.sum():,}）")
        print(f"    L1 Spearman {rho:.3f} >= 0.90 ? {'PASS' if l1 else '**FAIL**'}")
        print(f"    L2 新/舊 比值中位 {ratio:.3f} in [0.80,1.25] ? "
              f"{'PASS' if l2 else '**FAIL**'}")
        print(f"    L3 有無清算的 Jaccard {jac:.3f} >= 0.80 ? "
              f"{'PASS' if l3 else '**FAIL**'}")
        res["checks"][sym] = dict(n=int(len(m)), rho=float(rho),
                                  ratio=float(ratio), jaccard=float(jac),
                                  L1=bool(l1), L2=bool(l2), L3=bool(l3))

    print()
    print("=== 橋的判決 ===")
    print("  -> " + ("**PASS —— 橋可用**" if ok_all else
                     "**FAIL —— 橋是錯的，之後任何用它的結論都不解讀**"))
    res["bridge_ok"] = bool(ok_all)

    # ── core9 覆蓋與功效 ──
    print()
    print("=== core9 覆蓋（新表，雙所）===")
    cov = minute_liq(symbols=tuple(CORE9))
    g = cov.groupby("symbol").agg(分鐘=("window_start", "size"),
                                  總額=("liq_total_usd", "sum"))
    g["天"] = ((cov.groupby("symbol").window_start.max()
                - cov.groupby("symbol").window_start.min()) / 86400000).round(2)
    print(g.to_string(float_format=lambda x: f"{x:,.0f}"))

    days = (hi - lo) / 86400000
    ev_per_day = 3003 / 785          # 現行註冊母體的事件率
    print()
    print("=== 混淆對照要等多久（功效）===")
    print(f"  交會事件率 {ev_per_day:.2f} 筆/天（9 幣合計，現行註冊定義）")
    print(f"  新表目前 {days:.2f} 天 -> 約 {days*ev_per_day:.0f} 個事件")
    # 用現行母體的逐事件 SD 推：高/低兩組各 n/2
    for need in (200, 400, 800):
        se = np.sqrt(2) * 0.9 / np.sqrt(need / 2)   # SD~0.9 ATR（停損後）
        print(f"  n={need:4d}（{need/ev_per_day:5.0f} 天 ≈ {need/ev_per_day/30:.1f} 個月）"
              f"  兩組差的 SE ≈ {se:.3f} ATR   MDE ≈ {1.96*se:.3f} ATR")
    print("  參考量級：交會事件本身的效應是 +0.23 ATR（2 分鐘延遲）。")
    print("  -> 要偵測「有清算 vs 沒清算」之間 0.2 ATR 級別的差，"
          "**約需 n≈400、也就是 3.5 個月**。")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "liq_bridge.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print()
    print("written ->", OUT / "liq_bridge.json")


if __name__ == "__main__":
    main()
