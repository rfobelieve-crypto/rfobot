# -*- coding: utf-8 -*-
"""庫存會長到多大、倒得掉嗎 —— Stage 1 的部位上限從這裡來（2026-09-13）

===========================================================================
為什麼是這個量，而不是「對沖速度」
===========================================================================
`market-making-for-dummies` 對小市值標的的建議是**持有風險、不要急著平倉**
（急著平倉要在有毒的大所付價差，而那邊不會因為你的庫存來自無毒場館就給
折扣）。而我們量到對沖腿成本 1.4–5 bps > 流動端的全部 markout，
所以「成交就立刻對沖」在那裡是負的。

**那麼部位上限就不是一個偏好，是這門生意的主要風控旋鈕**，
而它取決於一件事：**做市方的庫存會長到多大、多久回得來。**

===========================================================================
怎麼量
===========================================================================
做市方的庫存變化由吃單方向決定：
    `is_maker_ask=True`  -> 吃單方買、**做市方賣** -> 庫存往空
    `is_maker_ask=False` -> 吃單方賣、**做市方買** -> 庫存往多

所以把逐筆成交加上符號再累加，就是「如果我們吃下 100% 的流量」的庫存軌跡。
我們實際只拿到份額 s，所以**我們的庫存 = s x 這條軌跡**，
而**時間尺度與 s 無關** —— 這是這個量測能外推的理由。

兩個數字：
  I1  **滾動 T 分鐘內的 |淨流量|**（p50 / p95，美元）。乘上份額就是部位上限。
  I2  **|淨流量| / 總流量**（同窗）。**這是單邊性的讀數**：
      接近 1 = 流量完全單邊 = 庫存倒不掉，我們會被輾過；
      接近 0 = 買賣交替 = 庫存自己會平掉。
      **I2 才是風險，I1 只是規模。**

===========================================================================
自曝檢查
===========================================================================
C1  窗越長，|淨流量|/總流量 應該**越小**（隨機漫步 ~1/√n）。
    反而變大代表流量有持續的方向性，那是更壞的消息但也要看得見。
C2  全格報告，不挑窗也不挑幣。
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "research", "results", "inventory_bound.json")
TAPE = "D:/flowbot_data/lighter/trades/*/*.parquet"
WINDOWS = [60, 300, 900, 3600]          # 秒
MIN_USD = 100_000.0


def main():
    d = pd.concat([pd.read_parquet(f, columns=[
        "ts", "coin", "usd", "is_liq", "is_maker_ask",
        "ask_acct", "bid_acct", "block_height"])
        for f in sorted(glob.glob(TAPE))], ignore_index=True)
    d = d[(~d.is_liq) & (d.usd > 0)]
    # **只用密集期，而且理由是量出來的。** 第一版寫 `ts >= 最早的一般成交`，
    # 跑出「實錄 7.92 天」—— 而逐日攤開來 09-11 之前每天只有 2~522 筆，
    # 橫跨 39 個日曆天。那些是**重訂閱的重播**（`subscribed/trade` 每次重訂閱
    # 回放 50 筆近期成交，它們帶原本的 ts 但被寫進當前的小時檔），不是當時
    # 真的在錄。
    #
    # **而它污染了結論**：一個整段只有 2 筆成交的市場，
    # `|淨流量|/總流量` 必然是 1.000 —— 於是 ANTHROPIC / WLD / SAMSUNGUSD
    # 被排進「單邊性最高、會被輾過」那一欄，而真相是它們根本沒有資料。
    # 這是今天第三個同族（分母/跨度）的錯。
    day = pd.to_datetime(d.ts, unit="ms").dt.floor("D")
    cnt = d.groupby(day).size()
    dense = cnt[cnt >= 0.05 * cnt.max()].index          # 至少是最忙那天的 5%
    d = d[day.isin(dense)]
    hrs = (d.ts // 3600000).nunique()
    print("一般成交 %s 筆｜%d 個幣｜**密集期 %d 天、%d 小時 = %.2f 天**"
          % (format(len(d), ","), d.coin.nunique(), len(dense), hrs, hrs / 24))
    print("  （丟掉 %d 個稀疏日 —— 那些是重訂閱重播的舊成交，不是當時在錄）"
          % (len(cnt) - len(dense)))
    MIN_N = 20          # 每個滾動窗至少這麼多**張吃單**才算單邊性
    # 做市方視角的符號
    d["inv"] = np.where(d.is_maker_ask, -d.usd, d.usd)

    # **必須先聚合成吃單，不能用逐筆成交算單邊性。**
    # 第二版用逐筆算，6 個市場跑出單邊性恰好 **1.000**，而查下去它們的方向
    # 其實是平衡的（XPL maker_ask 56.3%、INTC 52.0%、RAY 52.2%）——
    # 真正的原因是它們整段只有 400-750 筆成交，所以通得過「窗內 >=20 筆」
    # 的窗幾乎都是**單一次掃單**，而**一張吃單吃穿 N 檔必然 N 筆同一邊**。
    # 於是 1.000 的意思是「這個窗裡只有一張吃單」，不是「流量單邊」。
    #
    # 而吃單層也是經濟上對的單位：庫存能不能平掉取決於
    # **獨立的吃單方有沒有兩邊都來**，不是一張掃單吃掉我們幾檔。
    d["taker"] = np.where(d.is_maker_ask, d.bid_acct, d.ask_acct)
    o = (d.groupby(["coin", "block_height", "taker"], as_index=False)
          .agg(ts=("ts", "min"), usd=("usd", "sum"), inv=("inv", "sum")))
    print("  還原成吃單：%s 張（逐筆 %s）—— 單邊性在這一層算"
          % (format(len(o), ","), format(len(d), ",")))
    d = o

    # ── 洗量偵測（backtest-audit 第 20 項，2026-09-13 加）─────────────
    # **這一關殺掉了我原本每一項指標上的最佳候選。** MSFT 看起來完美：
    # 成交額 $10.8M、單邊性 0.029（庫存自己平）、報價存活 1061 ms
    # （50ms 很舒服）、曝露 0.3%。然後查了成交的形狀：
    #     單筆成交額 中位 $9,800 / p95 $9,900  <- 固定單量,不是自然流量
    #     相鄰成交**換邊** 84%（隨機 50%）      <- 刻意交替
    #     分散在 57 個吃單 / 46 個做市帳戶,最大配對只佔 5%
    # 固定單量 + 交替換邊 + 刻意分散對手 = **刷量**（掙積分/空投）。
    # 而它的 0.029 單邊性（低於隨機漫步的 ~0.22）正是交替的機械後果 ——
    # **那個「最好的數字」本身就是洗量的指紋。**
    #
    # 對照：BTC 中位 $16、換邊 25%、4,542 個吃單帳戶 = 自然流量。
    #       GOOGL 最大吃單帳戶佔 36%、換邊 22% = 單一方向性大戶,另一種病。
    #
    # 兩個判準都便宜：**單量集中度**（p95/中位接近 1）與**換邊率偏離 50%**。
    wash = {}
    for coin, x in d.groupby("coin"):
        if len(x) < 100:
            continue
        xs = x.sort_values("ts")
        med, p95 = float(xs.usd.median()), float(xs.usd.quantile(.95))
        wash[coin] = dict(
            clipr=med / p95 if p95 > 0 else np.nan,      # -> 1 = 固定單量
            alt=float((xs.inv.gt(0).diff() != 0).mean()))  # 換邊率
    rows = []
    for coin, x in d.groupby("coin"):
        if x.usd.sum() < MIN_USD:
            continue
        x = x.sort_values("ts")
        t = pd.to_datetime(x.ts.values, unit="ms")
        s = pd.Series(x.inv.values, index=t)
        g = pd.Series(x.usd.values, index=t)
        r = dict(coin=coin, usd=float(x.usd.sum()), n=len(x),
                 clipr=wash.get(coin, {}).get("clipr", np.nan),
                 alt=wash.get(coin, {}).get("alt", np.nan))
        for w in WINDOWS:
            net = s.rolling("%ds" % w).sum().abs()
            gro = g.rolling("%ds" % w).sum()
            nn = g.rolling("%ds" % w).count()
            # **窗內筆數不足就不算單邊性** —— 一個只有 1-2 筆成交的窗,
            # |淨|/總 必然接近 1,那是樣本不足不是單邊流量。
            k = (gro > 0) & (nn >= MIN_N)
            cov = float(k.mean())
            r["cov_%d" % w] = cov
            # **規模（I1）與單邊性（I2）的可測條件不同，所以分開擋。**
            # I1 只需要窗裡有成交；I2 需要窗裡有**足夠多張獨立吃單**，
            # 否則 |淨|/總 會被單一張吃單頂到 1.000。
            # 前兩版就是這樣把 XPL / INTC / RAY 誤判成「會被輾過」——
            # 它們的方向其實平衡（maker_ask 52-56%），只是整段只有
            # 400-750 筆成交、約 150 張吃單，300 秒窗湊不到 20 張。
            # **所以覆蓋率不足時 I2 是「說不出來」，不是 1.000。**
            # 同一條：今天早上「沒錄到不是 0」。
            kk = gro > 0
            if kk.sum() >= 30:
                r["net_p50_%d" % w] = float(net[kk].quantile(.50))
                r["net_p95_%d" % w] = float(net[kk].quantile(.95))
            if k.sum() >= 30 and cov >= 0.20:
                r["ratio_p50_%d" % w] = float((net[k] / gro[k]).quantile(.50))
                r["ratio_p95_%d" % w] = float((net[k] / gro[k]).quantile(.95))
        rows.append(r)
    R = pd.DataFrame(rows).sort_values("usd", ascending=False)
    print("  納入 %d 個幣（成交額 > $%s）" % (len(R), format(int(MIN_USD), ",")))

    print("\nI2 單邊性：|淨流量| / 總流量（中位｜p95）—— **這個才是風險**")
    print("  %-10s" % "窗" + "".join("%16s" % ("%ds" % w) for w in WINDOWS))
    w0 = R.usd / R.usd.sum()
    cells50, cells95 = [], []
    for w in WINDOWS:
        a = pd.to_numeric(R.get("ratio_p50_%d" % w), errors="coerce")
        b = pd.to_numeric(R.get("ratio_p95_%d" % w), errors="coerce")
        k = a.notna()
        cells50.append("%16.3f" % np.average(a[k], weights=w0[k]))
        cells95.append("%16.3f" % np.average(b[k], weights=w0[k]))
    print("  %-10s%s" % ("中位", "".join(cells50)))
    print("  %-10s%s" % ("p95", "".join(cells95)))
    def _usd(v):
        return "—" if pd.isna(v) else format(int(v), ",")

    def _wavg(col):
        v = pd.to_numeric(R.get(col), errors="coerce")
        k = v.notna()
        return float(np.average(v[k], weights=w0[k])) if k.any() else float("nan")
    r1 = _wavg("ratio_p50_%d" % WINDOWS[0])
    r2 = _wavg("ratio_p50_%d" % WINDOWS[-1])
    print("  C1 窗從 %ds 到 %ds，比值 %.3f -> %.3f：**%s**"
          % (WINDOWS[0], WINDOWS[-1], r1, r2,
             "越長越小（庫存會自己平）" if r2 < r1 else
             "**越長越大 —— 流量有持續方向，庫存倒不掉**"))

    print("\nI1 滾動窗內的 |淨流量|（美元）—— 乘上份額就是部位上限")
    print("  %-10s %12s %12s %12s" % ("coin", "60s p95", "300s p95", "3600s p95"))
    for _, x in R.head(14).iterrows():
        print("  %-10s %12s %12s %12s"
              % (x.coin, _usd(x.get("net_p95_60")),
                 _usd(x.get("net_p95_300")), _usd(x.get("net_p95_3600"))))

    # 洗量閘門：固定單量（clip -> 1）或換邊率遠離 50% 就不可解讀
    # **欄名不可叫 clip** —— 它撞到 DataFrame.clip 方法，`R.clip` 回的是
    # 方法不是欄位，而錯誤只在比較時才現形。
    #
    # **兩個方向是兩種病，不可以貼同一個標籤**（第一版都叫「洗量」）：
    #   換邊率 **遠高於** 50%（MSFT 89%）= 人為交替 -> **洗量**
    #   換邊率 **遠低於** 50%（SNDK 21%、US100 6%）= 連續同向 -> **方向性流量**
    # 兩個都對做市有害但原因相反：前者是假的量，後者是真的量但會輾過你。
    # 而 **SNDK 換邊率 21% 正好是 markout −27 bps 那個市場**（`quote_life.py`
    # 量到它一個幣佔窗內虧損的 74%）—— 兩個獨立指標交叉驗證同一件事。
    R["wash"] = (R["clipr"] > 0.8) | (R["alt"] > 0.75)
    R["trend"] = R["alt"] < 0.25
    print("\n流量品質閘門（backtest-audit #20）")
    bad = R[R.wash.fillna(False)]
    tr = R[R.trend.fillna(False)]
    print("  **洗量（固定單量 or 換邊率 > 75%%）：%d / %d**" % (len(bad), len(R)))
    for _, x in bad.sort_values("usd", ascending=False).head(5).iterrows():
        print("    %-10s 單量集中 %.2f｜換邊 %.0f%%｜$%s"
              % (x.coin, x["clipr"], 100 * x["alt"], _usd(x.usd)))
    print("  **方向性流量（換邊率 < 25%%，會輾過做市方）：%d / %d**"
          % (len(tr), len(R)))
    for _, x in tr.sort_values("usd", ascending=False).head(8).iterrows():
        print("    %-10s 換邊 %.0f%%｜$%s"
              % (x.coin, 100 * x["alt"], _usd(x.usd)))

    print("\n單邊性最低的 10 個（庫存最容易平掉 = 最適合做市，**已過洗量閘門**）")
    R["r300"] = pd.to_numeric(R.get("ratio_p50_300"), errors="coerce")
    ok = R[R.r300.notna() & ~R.wash.fillna(False)
           & ~R.trend.fillna(False)].sort_values("r300")
    print("  %-10s %10s %10s %8s %8s %14s"
          % ("coin", "單邊性", "300s p95 淨$", "單量集中", "換邊率", "成交額$"))
    for _, x in ok.head(10).iterrows():
        print("  %-10s %10.3f %10s %8.2f %7.0f%% %14s"
              % (x.coin, x.r300, _usd(x.net_p95_300), x["clipr"],
                 100 * x["alt"], _usd(x.usd)))
    print("\n單邊性最高的 6 個（會被輾過的那些）")
    for _, x in ok.tail(6).iloc[::-1].iterrows():
        print("  %-10s %10.3f %10s %14s"
              % (x.coin, x.r300, _usd(x.net_p95_300), _usd(x.usd)))

    print("\n部位上限怎麼定（把份額代進去）")
    for sh in (0.05, 0.20, 0.50):
        v = _wavg("net_p95_300") * sh
        print("  份額 %.0f%% -> 300 秒 p95 的庫存約 **$%.0f**（成交額加權平均）"
              % (100 * sh, v))
    R.to_json(OUT, orient="records", force_ascii=False)
    print("\n寫出 %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
