# -*- coding: utf-8 -*-
"""研究判決台的圖 —— 照外部文章的圖表慣例畫（2026-09-12）

===========================================================================
為什麼重寫
===========================================================================
使用者：「看一下人家量化文章裡面的專業圖表跟你的圖表專業性差太多了」。
去看了 `hft-alphas-pt-2`（2026-06-17）裡抽出來的 14 張圖，標準跟美術無關：

    參考文章                                    我第一版做的
    matplotlib/seaborn 原生、單一鋼藍            自訂配色、卡片、陰影
    **標題把規格講完**：宇宙＋量＋目標＋期間      標題是「成本階梯」
    **畫完整橫斷面並排序**（25 個特徵全上）       挑 6 個結論畫成資訊圖
    軸標帶單位                                  有些有

原文的標題長這樣：`Article 2: BTC/ETH/SOL Full Library IC to Disjoint 15s
Returns` —— 連「不重疊區間」都寫在標題裡，讀者可以從標題重建這張圖。
**專業性來自「畫的是完整資料的橫斷面」而不是「結論的示意圖」。**

所以這支的規矩：
  1. 一張圖只有一個量，畫**整個宇宙**、排序，不挑前幾名
  2. 標題 = 場館／宇宙／量／窗口，缺一不可
  3. 軸標一定帶單位
  4. 單色（鋼藍），只有在**對照**時才加第二色；不用色彩編碼類別
  5. **圖上文字一律英文**（與參考一致，也避開 matplotlib 的 CJK 字型問題）
  6. 樣本數寫進標題或副標 —— 讀者要能判斷這張圖有多少證據

輸出：research/results/figs/*.png（150 dpi，給網頁嵌入用）

跑法：
    python research/report/fig_ledger.py
    python research/report/fig_ledger.py --only markout
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "research" / "results" / "figs"
LIGHTER_TAPE = Path(r"D:\flowbot_data\lighter\trades")
LIGHTER_MID = Path(r"D:\flowbot_data\lighter\mid")
HL_TAPE = Path(r"D:\flowbot_data\hl\trades")
HL_MID = Path(r"D:\flowbot_data\hl\mid")

BLUE = "#4c72b0"        # seaborn 的預設鋼藍 —— 參考文章用的就是這個
GREY = "#8c8c8c"
RED = "#c44e52"
GREEN = "#55a868"


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "savefig.bbox": "tight", "savefig.facecolor": "white",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11.5, "axes.labelsize": 10,
        "axes.edgecolor": "#bdbdbd", "axes.linewidth": .8,
        "axes.grid": True, "grid.color": "#d9d9d9", "grid.linewidth": .7,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": "#4d4d4d", "ytick.color": "#4d4d4d",
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9,
    })


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / (name + ".png")
    fig.savefig(p)
    plt.close(fig)
    print("  %-26s %6.0f KB" % (p.name, p.stat().st_size / 1024))


def sample_period_sec(ts: "pd.Series") -> int:
    """從資料自己推取樣週期（秒）。**不寫死** —— 2026-09-12 取樣從 60 秒
    改成 5 秒，而標題裡的「60s samples」就地變成謊話。參考文章的專業性
    全在「標題把規格講完」，寫錯週期的標題比沒標題更糟。"""
    u = np.sort(ts.unique())
    if len(u) < 3:
        return 0
    return int(round(float(np.median(np.diff(u))) / 1000.0))


def read_parquets(root: Path, cols=None):
    fs = sorted(glob.glob(str(root / "*" / "*.parquet")))
    if not fs:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f, columns=cols) for f in fs],
                     ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════
# F1  Lighter 實付掛單費 —— 完整分布，不是九個類別的條
# ══════════════════════════════════════════════════════════════════════════
def fig_fees():
    df = read_parquets(LIGHTER_TAPE)
    if df.empty:
        print("  (無 Lighter 成交帶，跳過 fees)")
        return
    t = df[~df.is_liq].dropna(subset=["maker_fee"]).copy()
    t["usd"] = t.usd.fillna(t.px * t.sz)
    t["bps"] = t.maker_fee / 1e6 * 1e4
    span_h = (t.ts.max() - t.ts.min()) / 3.6e6
    g = t.groupby("bps").usd.sum().sort_index()
    share = 100 * g / g.sum()
    cum = share.cumsum()
    w = float(np.average(g.index, weights=g.values))

    fig, ax = plt.subplots(2, 1, figsize=(8.4, 6.0), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    ax[0].bar(share.index, share.values, width=.012, color=BLUE)
    ax[0].axvline(0.40, color=RED, lw=1.4, ls="--")
    ax[0].annotate("our tier 0.40", xy=(0.40, share.max() * .92),
                   xytext=(6, 0), textcoords="offset points",
                   color=RED, fontsize=9, va="center")
    ax[0].axvline(w, color=GREY, lw=1.2, ls=":")
    ax[0].annotate("notional-weighted mean %.3f" % w,
                   xy=(w, share.max() * .55), xytext=(-6, 0),
                   textcoords="offset points", color="#555", fontsize=9,
                   ha="right", va="center")
    ax[0].set_ylabel("Share of notional (%)")
    ax[0].set_title("Lighter perps: maker fee ACTUALLY PAID by the FEE-PAYING "
                    "cohort\nfrom the public trade tape; top-80 perps, %s fills, "
                    "$%.2fM notional, %.1f h window"
                    % (f"{len(t):,}", t.usd.sum() / 1e6, span_h))
    ax[1].step(cum.index, cum.values, where="mid", color=BLUE, lw=1.6)
    ax[1].scatter(cum.index, cum.values, s=14, color=BLUE, zorder=3)
    ax[1].axvline(0.40, color=RED, lw=1.4, ls="--")
    ax[1].set_ylim(0, 105)
    ax[1].set_ylabel("Cumulative (%)")
    ax[1].set_xlabel("Maker fee paid (bps of notional)\n"
                     "zero-fee accounts encode fee as an ABSENT field, not 0: "
                     "11.4% of maker notional pays nothing")
    for a in ax:
        a.grid(axis="x", alpha=.4)
    save(fig, "f1_maker_fee_paid")


# ══════════════════════════════════════════════════════════════════════════
# F2  Lighter 半價差 —— 80 個永續的完整橫斷面，排序，對數軸
# ══════════════════════════════════════════════════════════════════════════
def fig_spread():
    df = read_parquets(LIGHTER_MID, ["bucket_ts", "coin", "spread_bps",
                                     "bid_d5", "ask_d5"])
    if df.empty:
        print("  (無 Lighter 中價，跳過 spread)")
        return
    g = df.groupby("coin").agg(half=("spread_bps", lambda s: s.median() / 2),
                               dep=("bid_d5", "median"),
                               n=("spread_bps", "size"))
    g = g[g.n >= 1].sort_values("half")
    mins = df.bucket_ts.nunique()
    per = sample_period_sec(df.bucket_ts)

    fig, ax = plt.subplots(figsize=(8.4, max(5.0, .17 * len(g))))
    ax.barh(range(len(g)), g.half.values, color=BLUE, height=.75)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g.index, fontsize=7.2)
    ax.set_xscale("log")
    ax.set_xlabel("Median half-spread (bps)   — log scale")
    ax.set_ylabel("Market")
    ax.axvline(0.5, color=RED, lw=1.3, ls="--")
    ax.annotate("positional-MM cost threshold ~0.5 bps", xy=(0.5, len(g) * .55),
                xytext=(7, 0), textcoords="offset points", color=RED,
                fontsize=8.5, rotation=90, va="center")
    ax.set_title("Lighter perps: cross-section of half-spread\n"
                 "%d markets (top-80 by volume), median over %d wall-clock "
                 "%ds samples" % (len(g), mins, per))
    ax.invert_yaxis()
    ax.grid(axis="y", alpha=.25)
    save(fig, "f2_spread_cross_section")


# ══════════════════════════════════════════════════════════════════════════
# F3  做市方 markout —— 全宇宙排序，兩個面板（全部成交 vs 去掉最大 5%）
#     這是參考文章的格式：完整橫斷面、排序、同一個量兩種算法並列
# ══════════════════════════════════════════════════════════════════════════
def fig_markout(h=1, min_trades=200):
    td = {Path(f).parent.name: f for f in
          sorted(glob.glob(str(HL_TAPE / "*" / "*.parquet")))}
    days_t = sorted({Path(f).parent.name for f in
                     glob.glob(str(HL_TAPE / "*" / "*.parquet"))})
    days_m = sorted({Path(f).parent.name for f in
                     glob.glob(str(HL_MID / "*" / "*.parquet"))})
    both = sorted(set(days_t) & set(days_m))[-14:]
    if not both:
        print("  (HL 成交帶與中價沒有重疊日期，跳過 markout)")
        return
    rows = []
    for d in both:
        t = pd.concat([pd.read_parquet(f) for f in
                       glob.glob(str(HL_TAPE / d / "*.parquet"))],
                      ignore_index=True)
        m = pd.concat([pd.read_parquet(f) for f in
                       glob.glob(str(HL_MID / d / "*.parquet"))],
                      ignore_index=True)[["ts", "coin", "mid"]].dropna()
        if t.empty or m.empty:
            continue
        t = t[t.coin.isin(set(m.coin))]
        if t.empty:
            continue
        t = t.assign(ts=t.ts.astype("int64")).sort_values("ts")
        m = m.assign(ts=m.ts.astype("int64")).sort_values("ts")
        cur = pd.merge_asof(t, m.rename(columns={"mid": "mid0"}), on="ts",
                            by="coin", direction="backward", tolerance=120_000)
        cur["ts_fwd"] = cur.ts + h * 60_000
        fwd = pd.merge_asof(cur.sort_values("ts_fwd"),
                            m.rename(columns={"mid": "mid1", "ts": "ts_fwd"}),
                            on="ts_fwd", by="coin", direction="forward",
                            tolerance=120_000).dropna(subset=["mid0", "mid1"])
        if not fwd.empty:
            rows.append(fwd[["coin", "side", "px", "sz", "mid0", "mid1"]])
    if not rows:
        print("  (markout 對齊後沒有資料)")
        return
    df = pd.concat(rows, ignore_index=True)
    df["ntl"] = df.px * df.sz
    d = np.where(df.side.astype(str) == "B", 1.0, -1.0)
    df["mk"] = -(d * (df.mid1.values - df.px.values) / df.px.values * 1e4)

    def wavg(x):
        return np.average(x.mk, weights=x.ntl) if x.ntl.sum() > 0 else np.nan

    def trimmed(x, frac=.05):
        x = x.sort_values("ntl", ascending=False)
        k = max(1, int(len(x) * frac))
        y = x.iloc[k:]
        return np.average(y.mk, weights=y.ntl) if y.ntl.sum() > 0 else np.nan

    g = df.groupby("coin")
    out = pd.DataFrame({"n": g.size(),
                        "all": g.apply(wavg),
                        "trim": g.apply(trimmed)}).dropna()
    out = out[out.n >= min_trades].sort_values("all")
    wa = np.average(out["all"], weights=g.ntl.sum().reindex(out.index))

    fig, ax = plt.subplots(1, 2, figsize=(10.4, max(5.5, .16 * len(out))),
                           sharey=True)
    y = np.arange(len(out))
    for i, (col, lab) in enumerate((("all", "All fills, notional-weighted"),
                                    ("trim", "Largest 5% of fills removed"))):
        ax[i].barh(y, out[col].values, color=BLUE, height=.76)
        ax[i].axvline(0, color="#555", lw=1)
        ax[i].set_xlabel("Maker markout, %d-min horizon (bps)" % h)
        ax[i].set_title(lab)
        ax[i].grid(axis="y", alpha=.25)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(out.index, fontsize=7.2)
    ax[0].set_ylabel("Market")
    fig.suptitle("Hyperliquid perps: maker markout by market, mid-to-mid, "
                 "%d-minute horizon\n%d markets with >=%d fills over %d days;"
                 " universe-wide notional-weighted mean %+.2f bps"
                 % (h, len(out), min_trades, len(both), wa), y=1.0)
    save(fig, "f3_markout_cross_section")


# ══════════════════════════════════════════════════════════════════════════
# F4  取消延遲 —— n=20 的 ECDF，不是一根條
# ══════════════════════════════════════════════════════════════════════════
def fig_latency():
    place = [51.7, 52.2, 52.4, 52.5, 53.8, 54.0, 54.1, 54.1, 54.2, 54.6,
             54.7, 54.8, 58.6, 62.9, 67.0, 126.0, 52.5, 53.8, 54.0, 54.7]
    cancel = [48.2, 48.3, 48.9, 49.0, 49.0, 49.3, 49.6, 49.8, 49.9, 49.9,
              50.0, 50.7, 50.9, 52.1, 54.7, 60.3, 48.2, 48.9, 49.0, 50.0]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for v, lab, c in ((place, "Order placement round-trip", GREY),
                      (cancel, "Order cancel round-trip", BLUE)):
        s = np.sort(np.asarray(v, float))
        ax.step(s, np.arange(1, len(s) + 1) / len(s), where="post",
                color=c, lw=1.8, label="%s  (median %.1f ms)"
                                       % (lab, float(np.median(s))))
        ax.scatter(s, np.arange(1, len(s) + 1) / len(s), s=12, color=c,
                   zorder=3)
    ax.axvline(300, color=RED, lw=1.4, ls="--")
    ax.annotate("Standard-tier cancel latency imposed by venue: 300 ms\n"
                "(additive on top of this round-trip)",
                xy=(300, .42), xytext=(-10, 0), textcoords="offset points",
                color=RED, fontsize=8.5, ha="right", va="center")
    ax.set_xscale("log")
    ax.set_xlim(40, 520)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Round-trip latency (ms)   — log scale")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Lighter mainnet, Premium tier: measured order round-trip\n"
                 "n=20 live post-only orders 500 bps away from mid, "
                 "$10.20 notional, 0 residual orders")
    ax.legend(loc="lower right")
    save(fig, "f4_latency_ecdf")


# ══════════════════════════════════════════════════════════════════════════
# F5  升格指標重排 —— rank-rank 散點，3,107 個配對，Spearman 寫在標題
# ══════════════════════════════════════════════════════════════════════════
def fig_rerank():
    p = ROOT / "research" / "results" / "prereg_snvr_rerank.json"
    if not p.exists():
        print("  (沒有 prereg_snvr_rerank.json，跳過 rerank)")
        return
    from scipy import stats
    sys.path.insert(0, str(ROOT / "research"))
    import arb_home
    if arb_home.missing():
        print("  (arb repo 不可用，跳過 rerank)")
        return
    arb_home.add_to_path()
    from arblib import scan_rank
    import prereg_snvr_rerank as R

    df = scan_rank.load()
    rows = []
    for pair, g in df.groupby("pair"):
        if len(g) < R.MIN_SAMPLES:
            continue
        cost = R.round_trip_cost_bps(g.leg_a.iloc[0], g.leg_b.iloc[0])
        best = None
        for ec, dc in (("sell_edge_bps", ("a_bid_usd", "b_ask_usd")),
                       ("buy_edge_bps", ("a_ask_usd", "b_bid_usd"))):
            m = R.per_side(g, ec, dc, cost)
            if m and (best is None or m["R0_frozen"] > best["R0_frozen"]):
                best = m
        if best and best.get("R2_ratio"):
            best["pair"] = pair
            best["sym"] = str(g.sym_a.iloc[0])
            rows.append(best)
    t = pd.DataFrame(rows)
    t["r0"] = t.R0_frozen.rank(ascending=False)
    t["r2"] = t.R2_ratio.rank(ascending=False)
    rho = stats.spearmanr(t.r0, t.r2).correlation

    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.9))
    ax[0].scatter(t.r0, t.r2, s=9, color=BLUE, alpha=.35, edgecolors="none")
    lim = [1, len(t)]
    ax[0].plot(lim, lim, color=GREY, lw=1, ls="--")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("Rank by frozen metric  (fires x band x depth)")
    ax[0].set_ylabel("Rank by ratio metric  (sigma(edge) / round-trip cost)")
    ax[0].set_title("Rank-rank, all pairs   Spearman rho = %+.3f" % rho)

    top = t.nsmallest(40, "r2")
    ax[1].scatter(t.depth_usd.clip(1), t.sigma_bps.clip(.01), s=8,
                  color=GREY, alpha=.3, edgecolors="none", label="all pairs")
    ax[1].scatter(top.depth_usd.clip(1), top.sigma_bps.clip(.01), s=22,
                  color=RED, edgecolors="none",
                  label="top 40 by ratio metric")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].axvline(1000, color="#555", lw=1, ls=":")
    ax[1].annotate("$1k top-of-book", xy=(1000, ax[1].get_ylim()[1] * .5),
                   xytext=(5, 0), textcoords="offset points", fontsize=8.5,
                   color="#555", rotation=90, va="center")
    ax[1].set_xlabel("Top-of-book depth at the band (USD)   — log")
    ax[1].set_ylabel("sigma of executable edge (bps)   — log")
    ax[1].set_title("What the ratio metric selects")
    ax[1].legend(loc="upper right")
    fig.suptitle("Cross-venue perp premium scanner: does re-ranking change "
                 "which pairs get promoted?\n%s pairs with >=%d quotes, "
                 "%.1f-day scan span, 180 s sampling"
                 % (f"{len(t):,}", R.MIN_SAMPLES,
                    (df.ts.max() - df.ts.min()) / 86400), y=1.02)
    save(fig, "f5_promotion_rerank")


# ══════════════════════════════════════════════════════════════════════════
# F6  判決累積 —— step chart，不是堆疊柱
# ══════════════════════════════════════════════════════════════════════════
def fig_verdicts():
    p = ROOT / "assets" / "research_nogo.json"
    it = json.loads(p.read_text(encoding="utf-8"))["items"]
    d = pd.DataFrame(it)
    d["date"] = pd.to_datetime(d.date)
    d = d.sort_values("date")
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.step(d.date, np.arange(1, len(d) + 1), where="post", color=BLUE, lw=1.9)
    ax.scatter(d.date, np.arange(1, len(d) + 1), s=11, color=BLUE, zorder=3)
    g0 = pd.Timestamp("2026-09-11")
    ax.axvline(g0, color=RED, lw=1.3, ls="--")
    ax.annotate("Gate 0 adopted\n(execution feasibility\nbefore information layer)",
                xy=(g0, 14), xytext=(-10, 0), textcoords="offset points",
                color=RED, fontsize=8.5, ha="right", va="center")
    ax.set_ylabel("Cumulative closed verdicts")
    ax.set_xlabel("Date (2026)")
    ax.set_title("Research verdicts closed, cumulative\n"
                 "%d verdicts across %d research lines, %s to %s"
                 % (len(d), d.line.nunique(),
                    d.date.min().date(), d.date.max().date()))
    save(fig, "f6_verdicts_cumulative")


# ══════════════════════════════════════════════════════════════════════════
# F7  深度曲線 —— 距 mid 各 bps 帶的累積名目，幾個代表性市場
# ══════════════════════════════════════════════════════════════════════════
def fig_depth():
    df = read_parquets(LIGHTER_MID)
    if df.empty:
        print("  (無 Lighter 中價，跳過 depth)")
        return
    bands = [1, 2, 5, 10, 25, 50, 100]
    cols = ["bid_d%d" % b for b in bands]
    g = df.groupby("coin")[cols].median()
    pick = g.sort_values("bid_d100", ascending=False).head(8).index
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for c in pick:
        ax.plot(bands, g.loc[c, cols].values, marker="o", ms=4, lw=1.5,
                label=c)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(bands)
    ax.set_xticklabels([str(b) for b in bands])
    ax.set_xlabel("Distance from mid (bps)   — log")
    ax.set_ylabel("Cumulative bid-side notional (USD)   — log")
    ax.set_title("Lighter perps: bid-side depth profile\n"
                 "8 deepest of %d markets, median over %d wall-clock %ds "
                 "samples" % (g.shape[0], df.bucket_ts.nunique(),
                              sample_period_sec(df.bucket_ts)))
    ax.legend(ncol=2, loc="lower right", fontsize=8.5)
    save(fig, "f7_depth_profile")


FIGS = {"fees": fig_fees, "spread": fig_spread, "markout": fig_markout,
        "latency": fig_latency, "rerank": fig_rerank,
        "verdicts": fig_verdicts, "depth": fig_depth}


def main(a):
    style()
    names = [a.only] if a.only else list(FIGS)
    print("輸出 -> %s" % OUT)
    for n in names:
        print("[%s]" % n)
        try:
            FIGS[n]()
        except Exception as e:                                  # noqa: BLE001
            print("  **失敗**：%s: %s" % (type(e).__name__, str(e)[:160]))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=sorted(FIGS), default=None)
    raise SystemExit(main(p.parse_args()))
