# -*- coding: utf-8 -*-
"""Stage 0 的第一關：被同時掃的頻率與代價（2026-09-13，TODO §1.40）

===========================================================================
這一關決定整條線的可行性
===========================================================================
`execution-without-the-fluff` 裡有一段直接在講我們這個構造 —— 小場館的做市商
拿大所的價格重新報價、成交後去大所對沖 —— 然後他寫下：

    「A lot of the arbitrages are merely risk premiums that reflect the fact
     that you'll pay the arbitrage back **once you get swept and have to hedge
     at a loss**.」

也就是說：**我們賺的那個價差，就是「被掃的時候要賠回去」的保費。**
所以可行性不取決於價差多寬，取決於**保費夠不夠賠**。

===========================================================================
量法：markout，而它同時就是 G2
===========================================================================
被掃的代價不需要第二個場館的 tape 才量得出來。一個做市方被成交之後，
價格往不利方向走多少 —— 那就是 **markout**，而它同時是逆選擇的定義。

    做市方賣在 ask      markout(δ) = (成交價 − mid(t+δ)) / mid x 1e4
    做市方買在 bid      markout(δ) = (mid(t+δ) − 成交價) / mid x 1e4

**markout(δ) 就是「扣掉逆選擇之後真正賺到的毛邊際」**，所以它可以直接跟
我們的費用比（maker 0.40 ＋ 對沖腿成本），不需要再另外扣一次。
δ 取我們對沖得了的那幾個尺度：0 / 0.1 / 0.5 / 1 / 5 / 30 秒
（實測撤單往返 49.9 ms，所以 0.1–1 秒是真實的對沖窗）。

**而「被同時掃」就是 markout 的一個子集**：把成交分成「單獨成交」與
「被掃的一部分」，兩邊的 markout 分開看。那個差就是掃單的代價。

===========================================================================
還原吃單：`(coin, block_height, 吃單帳戶)`
===========================================================================
`tx_us` **是逐筆唯一的不是逐交易**（實測 278,206 筆成交對 278,206 個 group），
所以不能用它。`tx_hash` 沒有存進 COLS。用的是同一個幣、同一個區塊、
同一個吃單方 —— 那幾乎必然是一張吃單。
**掃單 = 一張吃單吃掉 >= 2 筆**（穿過一個以上的掛單）。

===========================================================================
自曝檢查
===========================================================================
C1  **markout(δ=0) 必須約等於 +半價差**。做市方賣在 ask，而 mid 在 ask 之下，
    所以剛成交那一刻做市方是賺半價差的。不是的話就是 mid 的 join 錯了。
C2  **吃單方的 markout 必須是鏡像**（約 −做市方）。同號就是符號寫反了。
C3  用 **mid 不用成交價**（mistake.md 2026-09-11：成交價在薄的標的上自帶
    負自相關，會偽裝成均值回歸 —— 在 markout 上它會偽裝成「沒有逆選擇」）。
C4  兩邊都用**交易所時戳**對齊（tape 的 `ts` vs tob 的 `book_time`），
    不用各自的 rx_ms。實測 tape 的 rx−ts 中位 150 ms、tob 的 67 ms，
    混用會注入 83 ms 的假偏移。
C5  tob 有 **2.4% 的列 `book_time` 是微秒**（2026-09-13 修正前），丟掉。
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "research", "results", "sweep_markout.json")

TAPE = "D:/flowbot_data/lighter/trades/*/*.parquet"
TOB = "D:/flowbot_data/lighter/tob/*/*.parquet"
# **視窗必須長於該市場簿口的更新間隔**,否則 markout 量到的是半價差本身
# (AI 的簿口年齡中位 917ms / p90 28 秒,所以它的 markout@1s = +57 bps
#  其實是半價差;拉到 30 秒變 −32.6)。所以這裡要有長視窗。
HORIZONS = [0.0, 0.1, 0.5, 1.0, 5.0, 15.0, 30.0, 60.0, 300.0]   # 秒


def load():
    tp = pd.concat([pd.read_parquet(f, columns=[
        "ts", "coin", "px", "usd", "is_liq", "is_maker_ask",
        "ask_acct", "bid_acct", "block_height"]) for f in sorted(glob.glob(TAPE))],
        ignore_index=True)
    tp = tp[(~tp.is_liq) & (tp.usd > 0) & (tp.px > 0)]
    tb = pd.concat([pd.read_parquet(f, columns=[
        "book_time", "coin", "bid", "ask"]) for f in sorted(glob.glob(TOB))],
        ignore_index=True)
    # C5 丟掉微秒列
    n0 = len(tb)
    tb = tb[tb.book_time.between(1e12, 1e14)]
    tb = tb[(tb.bid > 0) & (tb.ask >= tb.bid)]
    tb["mid"] = (tb.bid + tb.ask) / 2.0
    tb["hs"] = (tb.ask - tb.bid) / 2.0 / tb.mid * 1e4
    print("tape %s 筆｜tob %s 列（丟掉 %s 列微秒 book_time）"
          % (format(len(tp), ","), format(len(tb), ","), format(n0 - len(tb), ",")))
    lo = max(tp.ts.min(), tb.book_time.min())
    hi = min(tp.ts.max(), tb.book_time.max())
    tp = tp[tp.ts.between(lo, hi)].copy()
    print("重疊窗 %s ~ %s = **%.2f 小時**｜窗內成交 %s 筆"
          % (pd.to_datetime(lo, unit="ms"), pd.to_datetime(hi, unit="ms"),
             (hi - lo) / 3.6e6, format(len(tp), ",")))
    return tp, tb


def reconstruct(tp):
    """還原吃單：(coin, block_height, 吃單帳戶)。掃單 = 吃掉 >= 2 筆。"""
    tp["taker"] = np.where(tp.is_maker_ask, tp.bid_acct, tp.ask_acct)
    g = tp.groupby(["coin", "block_height", "taker"])
    tp["n_fills"] = g.usd.transform("size")
    tp["order_usd"] = g.usd.transform("sum")
    tp["is_sweep"] = tp.n_fills >= 2
    n_ord = g.ngroups
    print("\n還原吃單：%s 張｜其中掃單（>=2 筆）**%s 張 = %.1f%%**"
          % (format(n_ord, ","),
             format(int(tp.loc[tp.is_sweep].groupby(
                 ["coin", "block_height", "taker"]).ngroups), ","),
             100 * tp.loc[tp.is_sweep].groupby(
                 ["coin", "block_height", "taker"]).ngroups / max(n_ord, 1)))
    print("  **掃單佔成交額 %.1f%%**（這是風險曝露的正確分母，不是張數）"
          % (100 * tp.loc[tp.is_sweep, "usd"].sum() / tp.usd.sum()))
    q = tp.groupby("n_fills").usd.agg(["size", "sum"])
    q["usd_share"] = 100 * q["sum"] / tp.usd.sum()
    print("  吃掉幾筆的分布（前 8）：")
    for k, r in q.head(8).iterrows():
        print("    %2d 筆：%8s 個成交、佔成交額 %5.1f%%"
              % (k, format(int(r["size"]), ","), r.usd_share))
    return tp


def markout(tp, tb):
    """逐 δ 算做市方的 markout（usd 加權）。"""
    out = {}
    for coin, f in tp.groupby("coin"):
        b = tb[tb.coin == coin].sort_values("book_time")
        if len(b) < 100 or len(f) < 50:
            continue
        f = f.sort_values("ts")
        # C1 的基準：成交當下（或之前）最後一筆 mid
        base = pd.merge_asof(f[["ts"]].rename(columns={"ts": "t"}),
                             b[["book_time", "mid", "hs"]].rename(
                                 columns={"book_time": "t"}),
                             on="t", direction="backward")
        # **C1 的對照要用「成交當下的半價差」，而且要 usd 加權** ——
        # 不是該幣半價差的中位。宇宙放寬到 200 個幣之後 C1 從 PASS 變成
        # 「查 join」（0.577 vs 0.358），而那不是 join 壞了：薄的標的價差
        # 變動大，**成交集中在價差寬的時刻**，所以 markout(0) 依建構就大於
        # 半價差的中位。拿中位去比是把兩個不同的加權混在一起
        # （mistake.md 2026-09-12：觀測÷預期型的判準，分子分母要同一種統計量）。
        # 母體也要一致:C1 比的是 mo_no(單獨成交),所以對照的半價差
        # 只能取**單獨成交那一刻**的,不能把掃單那一刻的混進來。
        hsf = pd.to_numeric(base.hs, errors="coerce").values
        uw = f.usd.values
        okh = np.isfinite(hsf) & (~f.is_sweep.values)
        rec = dict(coin=coin, n=len(f), usd=float(f.usd.sum()),
                   hs=float(np.nanmedian(base.hs)),
                   hs_fill=float(np.average(hsf[okh], weights=uw[okh]))
                   if okh.sum() > 10 else np.nan)
        sign = np.where(f.is_maker_ask.values, 1.0, -1.0)   # 賣在 ask -> +1
        px = f.px.values
        for d in HORIZONS:
            tgt = f[["ts"]].copy()
            tgt["t"] = tgt.ts + int(d * 1000)
            m = pd.merge_asof(tgt[["t"]].sort_values("t"),
                              b[["book_time", "mid"]].rename(
                                  columns={"book_time": "t"}),
                              on="t", direction="backward")["mid"].values
            mo = sign * (px - m) / m * 1e4
            ok = np.isfinite(mo)
            if ok.sum() < 20:
                rec["mo_%s" % d] = np.nan
                continue
            rec["mo_%s" % d] = float(np.average(mo[ok],
                                                weights=f.usd.values[ok]))
            # 分掃單／非掃單
            for lab, msk in (("sw", f.is_sweep.values), ("no", ~f.is_sweep.values)):
                k = ok & msk
                rec["mo_%s_%s" % (lab, d)] = (
                    float(np.average(mo[k], weights=f.usd.values[k]))
                    if k.sum() >= 20 else np.nan)
        out[coin] = rec
    return pd.DataFrame(out).T.reset_index(drop=True)


def main():
    tp, tb = load()
    tp = reconstruct(tp)
    r = markout(tp, tb)
    r["usd"] = r.usd.astype(float)
    w = r.usd / r.usd.sum()

    print("\n" + "=" * 100)
    print("做市方的 markout（usd 加權，bps）—— **這就是扣掉逆選擇之後的毛邊際**")
    print("=" * 100)
    print("  %-8s" % "δ(秒)" + "".join("%11s" % ("%g" % d) for d in HORIZONS))
    for lab, pre in (("全體", "mo"), ("掃單成交", "mo_sw"), ("單獨成交", "mo_no")):
        cells = []
        for d in HORIZONS:
            v = pd.to_numeric(r["%s_%s" % (pre, d)], errors="coerce")
            k = v.notna()
            cells.append("%11.3f" % np.average(v[k], weights=w[k])
                         if k.any() else "%11s" % "—")
        print("  %-8s%s" % (lab, "".join(cells)))

    print("\n自曝檢查")
    # **C1 要拿「單獨成交」比半價差，不是拿全體比。**
    # 掃單吃穿好幾檔，深處那幾筆的成交價離 mid 比頂檔半價差遠得多，
    # 所以掃單成交的 markout(0) 依建構就大於半價差（實測 1.450 vs 0.394）。
    # 第一版拿全體比，看起來像 join 壞了 —— 而那是我比錯對象。
    # **而對照的半價差也要 usd 加權、也要只取單獨成交那一刻**（`hs_fill`）。
    # 宇宙從 80 放寬到 200 個幣之後這一關紅了（0.577 vs 中位 0.358），
    # 而它不是 join 壞了：薄的標的價差本身在變動，**成交集中在價差寬的
    # 時刻**，所以「成交加權的半價差」必然大於「時間中位的半價差」。
    # 拿中位去比 = 觀測與預期用了兩種不同的統計量
    # （mistake.md 2026-09-12 第二條）。中位那個仍然印出來當對照。
    m0 = pd.to_numeric(r["mo_no_0.0"], errors="coerce")
    hsf = pd.to_numeric(r["hs_fill"], errors="coerce")
    hsm = pd.to_numeric(r["hs"], errors="coerce")
    k = m0.notna() & hsf.notna() & hsm.notna()
    a = np.average(m0[k], weights=w[k])
    b = np.average(hsf[k], weights=w[k])
    bm = np.average(hsm[k], weights=w[k])
    print("  C1 **單獨成交**的 markout(δ=0) = **%.3f bps** vs "
          "**成交加權**半價差 **%.3f**  -> %s"
          % (a, b, "**PASS**（差 %.0f%%）" % (100 * abs(a / b - 1))
             if b > 0 and abs(a / b - 1) < 0.5 else "**查 join**"))
    print("     （時間中位的半價差 %.3f —— 比成交加權低 %.0f%%，"
          "那是「價差寬的時候才有人交易」，不是 join 錯）"
          % (bm, 100 * (1 - bm / b) if b > 0 else float("nan")))
    ma = pd.to_numeric(r["mo_sw_0.0"], errors="coerce")
    ka = ma.notna()
    print("     （掃單成交 %.3f —— 比半價差大是對的：那些成交在簿口深處）"
          % np.average(ma[ka], weights=w[ka]))
    print("  C3 用 mid 不用成交價 ✓｜C4 兩邊都用交易所時戳 ✓｜C5 微秒列已丟 ✓")

    print("\n逐幣（成交額前 12）")
    r2 = r.sort_values("usd", ascending=False).head(12)
    print("  %-10s %10s %8s %9s %9s %9s %9s"
          % ("coin", "成交額$", "半價差", "mo@0.1s", "mo@1s", "掃單@1s", "單獨@1s"))
    for _, x in r2.iterrows():
        f = lambda v: ("%9.3f" % v) if pd.notna(v) else "%9s" % "—"   # noqa: E731
        print("  %-10s %10s %8.2f %s %s %s %s"
              % (x.coin, format(int(x.usd), ","), x.hs,
                 f(x["mo_0.1"]), f(x["mo_1.0"]),
                 f(x["mo_sw_1.0"]), f(x["mo_no_1.0"])))

    print("\n判讀（對沖窗 0.1–1 秒）")
    for d in (0.1, 1.0):
        v = pd.to_numeric(r["mo_%s" % d], errors="coerce")
        k = v.notna()
        net = np.average(v[k], weights=w[k]) - 0.40      # 扣 Lighter maker
        print("  δ=%.1fs：markout %.3f − maker 0.40 = **淨 %.3f bps**（未扣對沖腿）"
              % (d, np.average(v[k], weights=w[k]), net))
    r.to_json(OUT, orient="records", force_ascii=False)
    print("\n寫出 %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
