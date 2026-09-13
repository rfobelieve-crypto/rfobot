# -*- coding: utf-8 -*-
"""Stage 1：為了賺錢挑標的，以及 size 要多大才跑得動（2026-09-13，TODO §1.40）

===========================================================================
兩次更正我自己的閘門結構，而兩次都是同一個病
===========================================================================
**第一次**：`candidate_join.py` 用五個閘門（A 曝露 / B 半價差 / C 換邊率 /
D 單邊性 / E markout），0/69 全過。而 **A 與 B 被 E 包含**：
markout 是**已實現**的結果，那 16% 落在不利移動 50 ms 內的成交，虧損
**已經在 markout 的平均裡**。再用曝露擋一次是同一件事扣兩次。

**第二次**：把 D（單邊性）也當成硬閘門，於是 AI（markout **+57.3 bps**）、
UNI（+3.14）、NEAR（+2.98）全被排除。**但庫存是用上限管的，不是用選標的管的**
—— 單邊的市場只會撞到上限然後停那一側，那是**設計回應不是排除條件**。

共同的病：**把設計參數做成了排除閘門。** 每排除一次，候選名單就少一截，
而少掉的那些不是不能做，是需要不同的 size。

所以現在只有**兩個**硬閘門，第三個量改成決定 size：
  **E  markout@1s > maker 費**   賺不賺（已含逆選擇）—— 硬
  **C  換邊率 25–75%**           流量是不是真的（擋洗量與方向性）—— 硬
  **D  單邊性**                  -> **決定庫存上限與 size**，不排除

===========================================================================
size 的四個約束，取最緊的
===========================================================================
  S1 **場館最小單** `min_quote_amount`（BTC 實測 $10）
  S2 **不要變成整本簿口** —— 掛超過該市場單筆成交額的量級，我們就是簿口，
     而 markout 的量測前提（我們是邊際參與者）就破了
  S3 **樣本速度** —— 份額 = 1/(現有做市帳戶 + 1)，**與 size 無關**
     （排隊按時間不按量），所以 size 不影響多久拿到 30 筆
  S4 **庫存上限** —— 不對沖時庫存是唯一風控。
     上限 = 300 秒 p95 淨流量 x 份額 x 安全倍數

本支自己從成交帶算做市帳戶數與流量（前一版 join 到只含 9 個 ticker 的
舊檔，於是整張表的 size 與損益都是 0 —— 那是「空輸出不可當合法狀態」
的同一族，mistake.md 2026-09-11）。
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "research", "results")
TAPE = "D:/flowbot_data/lighter/trades/*/*.parquet"
API = "https://mainnet.zklighter.elliot.ai"
MAKER_FEE = 0.40
TARGET_FILLS = 30
SAFETY = 2.0            # 庫存硬上限 = 量級 x 這個
CAP_SIZE = 500.0        # 單筆上限（Stage 1 刻意壓住）


def tape_stats():
    d = pd.concat([pd.read_parquet(f, columns=[
        "ts", "coin", "usd", "is_liq", "is_maker_ask",
        "ask_acct", "bid_acct", "block_height"])
        for f in sorted(glob.glob(TAPE))], ignore_index=True)
    d = d[(~d.is_liq) & (d.usd > 0)]
    day = pd.to_datetime(d.ts, unit="ms").dt.floor("D")
    cnt = d.groupby(day).size()
    d = d[day.isin(cnt[cnt >= 0.05 * cnt.max()].index)]
    days = (d.ts // 3600000).nunique() / 24.0
    d["taker"] = np.where(d.is_maker_ask, d.bid_acct, d.ask_acct)
    d["maker"] = np.where(d.is_maker_ask, d.ask_acct, d.bid_acct)
    rows = []
    for c, x in d.groupby("coin"):
        o = x.groupby(["block_height", "taker"]).usd.sum()
        rows.append(dict(coin=c, makers=int(x.maker.nunique()),
                         orders_day=len(o) / days,
                         usd_day=float(x.usd.sum()) / days,
                         ord_med=float(o.median())))
    print("成交帶：%.2f 天、%d 個幣" % (days, d.coin.nunique()))
    return pd.DataFrame(rows)


def main():
    ib = pd.read_json(os.path.join(RES, "inventory_bound.json"))
    mo = pd.read_json(os.path.join(RES, "sweep_markout.json"))
    st = tape_stats()
    MOC = [c for c in ["hs", "mo_1.0", "mo_15.0", "mo_30.0", "mo_60.0"]
           if c in mo.columns]
    j = (ib.merge(mo[["coin"] + MOC], on="coin", how="inner")
           .merge(st, on="coin", how="inner"))
    try:
        r = requests.get(API + "/api/v1/orderBookDetails", timeout=20).json()
        mins = {o["symbol"]: float(o.get("min_quote_amount") or 0)
                for o in (r.get("order_book_details") or [])}
    except Exception as e:                                   # noqa: BLE001
        print("  orderBookDetails 拿不到（%r）-> 最小單用 $10" % e)
        mins = {}
    j["min_usd"] = j.coin.map(mins).replace(0, np.nan).fillna(10.0)

    # ── E 必須跨視窗穩定，不是單一個 1 秒（2026-09-13 更正）──────────
    # **AI 的 markout@1s 是 +57.3 bps，而它是假的。** 查它的時候發現
    # 半價差 64.88 bps、**簿口年齡中位 917 ms、p90 28 秒** —— 所以
    # `mid(t+1s)` 常常就是 `mid(t)`，markout@1s 量到的是**半價差本身**，
    # 不是扣掉逆選擇之後的東西。1 秒的視窗比簿口的更新間隔還短，裡面沒有資訊。
    # （mistake.md 2026-09-07：「兩個時點拿到一模一樣的東西永遠是切點失效
    # 的徵兆」—— 這次是視窗短於更新率。）
    #
    # 拉長視窗就現形了：
    #   AI    +47.0(1s) +41.5(5s) +27.4(15s) **−32.6(30s)** −13.3(60s) −36.0(300s)
    #   GRAM  +4.7 +5.0 +5.5 +7.1 **+8.1** +4.6    <- 跨視窗穩定
    #   NEAR  +2.8 +3.4 +5.2 +5.4 **+5.6** +3.4    <- 跨視窗穩定
    #   ZEC   +0.8 +1.1 +0.4 −0.3 **−1.3** −1.8    <- 衰退轉負
    #
    # **所以 E 改成「15s / 30s / 60s 三個視窗都 > maker 費」** ——
    # 單一視窗的 markout 對「簿口更新率」這個混淆因子沒有免疫力。
    # 通則：**markout 的視窗必須長於該市場簿口的更新間隔**，而一個固定視窗
    # 套在更新率差 100 倍的宇宙上必然錯。
    HZ = ["mo_15.0", "mo_30.0", "mo_60.0"]
    have = [h for h in HZ if h in j.columns]
    if len(have) < len(HZ):
        print("  **sweep_markout.json 沒有 15/30/60 秒的欄位 —— "
              "先把 HORIZONS 加上去重跑那一支，否則 E 退化成單視窗**")
        j["E"] = (j["mo_1.0"] > MAKER_FEE).fillna(False)
    else:
        j["E"] = np.all([(j[h] > MAKER_FEE).fillna(False) for h in have],
                        axis=0)
    j["C"] = j["alt"].between(0.25, 0.75).fillna(False)
    print("\n兩個硬閘門")
    print("  E markout > maker %.2f **在 %s 全部成立**   **%d / %d**"
          % (MAKER_FEE, "/".join("%gs" % float(h.split("_")[1])
                                 for h in have) or "1s",
             int(j.E.sum()), len(j)))
    print("  C 換邊率 25-75%%（非洗量非方向性） **%d / %d**"
          % (int(j.C.sum()), len(j)))
    k = j[j.E & j.C].copy()
    print("  **兩關全過：%d 個**" % len(k))

    k["share"] = 1.0 / (k.makers + 1.0)
    k["myfills_day"] = k.orders_day * k.share
    k["days_to_30"] = TARGET_FILLS / k.myfills_day.replace(0, np.nan)
    # S2：壓在該市場單筆吃單額中位的一半
    k["size"] = np.maximum(k.min_usd, np.minimum(k.ord_med * 0.5, CAP_SIZE))
    # S4：庫存 —— **上界是我們自己的成交額，不是市場的淨流量。**
    # 第一版寫 `net_p95_300 x share`，那假設我們的 size 跟市場單量一樣大；
    # 而 S2 刻意把 size 壓在市場單量的一半，所以那個數字高估約 20 倍
    # （AI：$752 vs 真實 $528/天）。**錯的方向是對使用者不利**：
    # 我差一點交出一個大 20 倍的風險數字。
    #
    # 真實上界：我們一天的成交額 x 單邊性（= 不會自己抵銷的那部分）。
    # 單邊性取 3600 秒那一格；沒有值（太稀疏）時保守取 1.0（全部同一邊）。
    k["onesided"] = pd.to_numeric(k.get("ratio_p50_3600"),
                                  errors="coerce").fillna(1.0)
    k["myusd_day"] = k.myfills_day * k["size"]
    k["inv"] = k.myusd_day * k.onesided            # 一天累積的庫存
    k["cap"] = k.inv * SAFETY
    # **損益的邊際要跟閘門信任的視窗一致。** E 要求 15/30/60 秒三格全過,
    # 而損益如果還用 1 秒,就是用一個已被判定不可信的量去算錢
    # (AI 的 1 秒是 +47 bps,30 秒是 −32.6)。取三格的**最小值** = 下界。
    k["mo_edge"] = (k[have].min(axis=1) if len(have) == len(HZ)
                    else k["mo_1.0"])
    k["pnl_day"] = k.myfills_day * k["size"] * (k.mo_edge - MAKER_FEE) / 1e4
    # **Stage 1 受限於風險不是機會數，所以排序要用報酬÷風險。**
    # 按絕對損益排會把最大的庫存上限配給最弱的候選：GOOGL 佔庫存上限 72%
    # 卻只貢獻 4% 的損益（markout +0.76、單邊性 0.984、而且它有一個
    # 佔 36% 的單一吃單方）。
    k["roi_day"] = 100 * k.pnl_day / k.cap.replace(0, np.nan)
    k = k.sort_values("roi_day", ascending=False)

    print("\n" + "=" * 112)
    print("候選（兩關全過）：size 與庫存上限由**單邊性**決定，不是排除它")
    print("=" * 112)
    print("  %-9s %8s %6s %7s %6s %8s %7s %8s %9s %9s %8s"
          % ("coin", "邊際bps", "換邊%", "單邊性", "做市者", "我成交/天",
             "size$", "幾天到30", "庫存上限$", "日損益$", "日報酬%"))
    for _, x in k.head(20).iterrows():
        print("  %-9s %8.3f %5.0f%% %7s %6.0f %8.1f %7.0f %8s %9s %9.2f %8s"
              % (x.coin, x.mo_edge, 100 * x["alt"],
                 "—" if pd.isna(x.ratio_p50_300) else "%.3f" % x.ratio_p50_300,
                 x.makers, x.myfills_day, x["size"],
                 "—" if pd.isna(x.days_to_30) else "%.2f" % x.days_to_30,
                 "—" if pd.isna(x.cap) else format(int(x.cap), ","),
                 x.pnl_day,
                 "—" if pd.isna(x.roi_day) else "%.3f" % x.roi_day))
    print("  ---")
    print("  **合計日損益量級 $%.2f**（%d 個市場）｜我方日成交額合計 $%.0f"
          % (k.pnl_day.sum(), len(k), k.myusd_day.sum()))
    # 數字大到不合理先查儀器：markout 超過 maker 費 20 倍的要標出來
    od = k[k.mo_edge > 20 * MAKER_FEE]
    if len(od):
        print("  **邊際超過 maker 費 20 倍的 %d 個 —— 先查儀器不要直接用**："
              "%s" % (len(od), ", ".join("%s %+.1f bps" % (x.coin, x.mo_edge)
                                        for _, x in od.iterrows())))
    print("  拿到 30 筆成交最快的：%s（%.2f 天）"
          % (k.sort_values("days_to_30").iloc[0].coin,
             k.days_to_30.min()))

    print("\nStage 1 的建議配置")
    top = k.head(5)
    print("  先上 5 個：%s" % ", ".join(top.coin))
    print("  單筆 size $%.0f ~ $%.0f｜**庫存硬上限合計 $%s**（= 量級 x %.0f）"
          % (top["size"].min(), top["size"].max(),
             format(int(top.cap.sum()), ","), SAFETY))
    print("  預期 %.1f 天內拿到 30 筆成交｜日損益量級 $%.2f"
          % (top.days_to_30.max(), top.pnl_day.sum()))
    print("\n  **庫存上限那個數字是 Stage 1 的真實風險暴露**（不對沖），")
    print("  所以它要你點頭，不是我決定。上表用安全倍數 %.0f；" % SAFETY)
    print("  要更保守就把 SAFETY 調小，或把 CAP_SIZE 從 $%.0f 壓下去。" % CAP_SIZE)
    print("\n  日損益是**量級不是預測**：markout 的樣本只有幾小時、"
          "G3 集中度 74% 在單一市場。")
    k.to_json(os.path.join(RES, "stage1_sizing.json"), orient="records",
              force_ascii=False)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
