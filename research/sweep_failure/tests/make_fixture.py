# -*- coding: utf-8 -*-
"""產生凍結的測試用 bar 切片（一次性，產物進 git）。

**為什麼需要這個**：`.cache/*.csv` 是 `fetch_klines.py` 抓的，而它的起點是
`now - days*86400` —— 那是一個**滾動 930 天窗**，每次刷新都從頭部丟掉舊
bar。所以任何釘在「絕對筆數」或「整份輸出的 sha」上的檢查，在下一次刷新
就必紅，而且它紅掉的原因跟被保護的那個東西（引擎的算術）完全無關。

2026-09-10 實測：`test_backtest_detail_parity` 的 `n == 7083` 已經紅成
7,064，原因純粹是頭部被丟掉 —— 引擎一行都沒改。一條永遠紅的守衛跟壞掉
的守衛一樣沒用（mistake.md 2026-09-03）。

修法：把一段**固定的**時間切片存成檔案進 git，sha 釘在它上面。這樣守衛
量的是「引擎的算術有沒有變」，不是「資料窗滾到哪裡」。

跑法（只有要換基準時才跑，平常不跑）：
    python research/sweep_failure/tests/make_fixture.py
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / ".cache"
OUT = HERE / "fixtures"
# 固定的絕對區間（UTC）。選在 .cache 目前一定覆蓋得到的範圍內，
# 存成檔案之後就與滾動窗脫鉤。
T0 = 1719792000      # 2024-07-01 00:00 UTC
T1 = 1751328000      # 2025-07-01 00:00 UTC
SYMS = ["BTC", "DOGE"]      # 兩種價格尺度（五位數 / 小數點後四位）


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for s in SYMS:
        src = CACHE / (s + "USDT_1h.csv")
        rows = []
        with open(src, newline="", encoding="utf-8-sig") as f:
            r = csv.reader(f)
            head = next(r)
            for x in r:
                if len(x) < 6:
                    continue
                t = int(float(x[0]))
                if T0 <= t < T1:
                    rows.append(x[:6])
        dst = OUT / (s + "USDT_1h_frozen.csv")
        with open(dst, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(head[:6])
            w.writerows(rows)
        print("%s  %d bars  %.0f KB  -> %s"
              % (s, len(rows), dst.stat().st_size / 1024, dst.name))


if __name__ == "__main__":
    main()
