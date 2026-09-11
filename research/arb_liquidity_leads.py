# -*- coding: utf-8 -*-
"""§1.22b 深度能不能預測「誰在移動」—— 先導測試（2026-09-11）

===========================================================================
為什麼這支現在才做得起來
===========================================================================
`gate0_arb_unhedged.who_moves()` 量到小場館貢獻 **0.485–0.543** 的移動，
我當時寫成「他的核心假設失敗」。今天兩次更正它：

  1. Quant Arb〈Small Trader Alpha #6〉：那個主張**條件於成交量**，
     而我們沒有成交量 -> §1.22「量不到」
  2. 該文附的碩論〈High Frequency Lead-Lag Relationships In The Bitcoin
     Market〉（127 頁）：**「本論文的分析指向交易所的流動性是最可能的
     解釋」**、「較小、較不流動的交易所對其他**全部**交易所呈現落後」

第 2 點把死結解開了：**流動性我們有**（掃描器的逐檔深度）。
所以「誰領先」不必等成交量。

===========================================================================
假說與判準（先寫，後跑）
===========================================================================
    H: 深度較淺的那一腿，做掉較多的移動。
       -> `who_moves` 的小所佔比 應該**隨「小所相對深度」下降而上升**

    量法：x = log10(深度_A / 深度_B)（A = who_moves 量的那一腿）
          y = who_moves 的小所移動佔比中位
          看 Spearman(x, y) **應該為負**

**這是先導不是判決。** n = 8 個配對 —— 在 n=8 上，|ρ| 要到 **0.71** 才
達到 p<0.05，所以這支**幾乎只能偵測到很強的效應**。
它的用途是：**告訴我們 149 個配對的資料長出來之後值不值得認真測**，
以及**符號對不對**。不得拿它下任何判決
（mistake.md 2026-09-04：SE >= 門檻 ⇒ 這個設計不能做決定）。

===========================================================================
自曝檢查
===========================================================================
C1  `who_moves` 的值必須重現 `gate0_arb_unhedged.json` 裡凍結的那一份
    （同一顆函式、同一份資料，不得有第二種算法）。
C2  深度必須來自**掃描器**（獨立的儀器），不是錄製器自己的頂檔——
    用同一支腳本的兩個欄位去相關，會量到儀器不是市場。
C3  **對照**：把深度比換成一個與流動性無關的量（配對的 band 寬度），
    它不該有同樣的關係；有的話代表我量到的是「配對本身」不是「深度」。

    python research/arb_liquidity_leads.py
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import arb_home as AH                          # noqa: E402
AH.add_to_path()

OUT = ROOT / "research" / "results" / "arb_liquidity_leads.json"
FROZEN = ROOT / "research" / "results" / "gate0_arb_unhedged.json"
REF_TOL = 0.02

# pid -> 掃描器的配對名（與 gate0_arb_capacity.SCAN_PAIR 同一張表；
# 那支是唯一寫過這個對應的地方，所以直接 import 它而不是抄一份）
def scan_pairs():
    from research.gate0_arb_capacity import SCAN_PAIR
    return dict(SCAN_PAIR)


def depth_by_pair():
    """逐配對、逐腿的 3 bps 深度中位（掃描器，獨立儀器）。"""
    files = sorted(glob.glob(str(AH.HOME / "engine" / "logs" / "scan"
                                 / "scan_v5_*.csv")))
    if not files:
        return {}
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out = {}
    for pid, pair in scan_pairs().items():
        m = d[d.pair == pair]
        if len(m) < 50:
            continue
        # 兩側各取「該腿能成交的那一邊」的 3bps 深度，再取中位
        a = np.nanmedian(np.concatenate([m.a_bid_usd_3bps.to_numpy(float),
                                         m.a_ask_usd_3bps.to_numpy(float)]))
        b = np.nanmedian(np.concatenate([m.b_bid_usd_3bps.to_numpy(float),
                                         m.b_ask_usd_3bps.to_numpy(float)]))
        out[pid] = dict(n_rows=int(len(m)), depth_a=float(a), depth_b=float(b),
                        log_ratio=float(np.log10(max(a, 1e-9) / max(b, 1e-9))))
    return out


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return float("nan"), 0
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1]), len(x)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if not FROZEN.exists():
        print("找不到 gate0_arb_unhedged.json —— 先跑那一支。")
        return 2
    fz = json.loads(FROZEN.read_text(encoding="utf-8"))
    moves = fz.get("who_moves") or {}
    dep = depth_by_pair()
    cm = json.loads((AH.RESULTS / "arb_cost_model.json").read_text(encoding="utf-8"))

    res = dict(asof=time.strftime("%Y-%m-%d %H:%M:%S"), pairs={})
    rows = []
    print("=== 假說：深度較淺的那一腿做掉較多的移動 ===")
    print("（A 腿 = who_moves 量的那一腿 = entropy/小所）\n")
    print("%-9s %10s %12s %12s %9s %9s %8s"
          % ("配對", "小所佔比", "A腿3bps$", "B腿3bps$", "log比", "帶bps", "事件"))
    for pid, w in sorted(moves.items()):
        if not w or pid not in dep:
            continue
        d = dep[pid]
        band = ((cm.get("pairs") or {}).get(pid) or {}).get("spec", {}).get("band_bps")
        rows.append((pid, w["share_median"], d["log_ratio"], band or np.nan,
                     w["n_events"]))
        print("%-9s %10.3f %12.0f %12.0f %+9.2f %9s %8d"
              % (pid, w["share_median"], d["depth_a"], d["depth_b"],
                 d["log_ratio"], ("%.1f" % band) if band else "—", w["n_events"]))
        res["pairs"][pid] = dict(share=w["share_median"], **d, band_bps=band)

    if len(rows) < 4:
        print("\n可用配對不足 4 個，停。")
        return 2
    share = [r[1] for r in rows]
    lr = [r[2] for r in rows]
    band = [r[3] for r in rows]

    rho, n = spearman(lr, share)
    print("\n=== 主檢定 ===")
    print("  Spearman(log 深度比, 小所移動佔比) = **%+.3f**（n=%d）" % (rho, n))
    print("  假說要求**為負**（A 腿越淺 -> log比越小 -> 佔比越高）-> %s"
          % ("符號對" if rho < 0 else "**符號相反**"))
    # n=8 的臨界值
    crit = {5: 0.90, 6: 0.83, 7: 0.75, 8: 0.71, 9: 0.68, 10: 0.65}.get(n, 0.7)
    print("  n=%d 在 p<0.05 需要 |ρ| >= %.2f -> **%s**"
          % (n, crit, "達到" if abs(rho) >= crit else
             "沒達到（**先導測試，不下判決**）"))

    rho_b, _ = spearman(band, share)
    print("\n=== C3 對照：把深度比換成帶寬（與流動性無關的量）===")
    print("  Spearman(帶寬, 小所移動佔比) = %+.3f" % rho_b)
    print("  它**不該**比主檢定強；比較強代表我量到的是「配對本身」不是深度。")
    print("  -> %s" % ("對照較弱，主檢定站得住" if abs(rho_b) < abs(rho)
                       else "**對照一樣強或更強 —— 主檢定不可解讀**"))

    res.update(rho_depth=rho, n=n, crit=crit, rho_band_control=rho_b,
               conclusive=bool(abs(rho) >= crit and abs(rho_b) < abs(rho)))

    print("")
    print("=== 2026-09-11 跑完當場發現的設計問題（比結果重要）===")
    print("  `who_moves` 的窗是「|偏離| >= band 到回到 band/2」，")
    print("  而那段窗**同時包含兩個相反的階段**：")
    print("    擴大期  某一腿繼續走開 -> 移動的那個是**領先**的")
    print("    收斂期  另一腿追上來   -> 移動的那個是**落後**的")
    print("  所以「窗內誰移動比較多」**分不出領先與落後**，")
    print("  它的符號因此不能拿去對照論文的主張——不管正負。")
    print("  我在假說裡寫「應該為負」是**沒有想清楚這件事就寫下的方向**。")
    print("  -> 修法（給 n=149 那一輪）：**把窗拆成擴大期與收斂期**，")
    print("     逐腿分別算位移，再看深度預測的是哪一個。")
    print("")
    print("=== 讀法 ===")
    print("  **這是先導不是判決，而且上面那個設計問題讓它連符號都不能讀。**")
    print("  n=%d 太小，它只回答兩件事：" % n)
    print("  (1) 符號對不對；(2) 值不值得在 149 個配對上認真測。")
    print("  §1.25 的宇宙錄製器從 2026-09-11 起在錄 149 個配對的逐分鐘簿口，")
    print("  幾天之後這個檢定的 n 會從 8 變成 149 —— 那時候才寫得出門檻。")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print("written -> " + str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
