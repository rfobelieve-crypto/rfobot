# -*- coding: utf-8 -*-
"""淨偏斜基線/假象檢查 (TODO 2026-07-17 登記, 優先於狀態機判斷器).

背景: 手機覆盤圖觀察到現貨淨偏斜綠柱長期偏高 (賣側淨撤離結構性 > 買側),
且資料期間全程陰跌. 分辨兩個候選解釋:
  (A) 量測假象 — 偏斜跟著價格方向機械性移動 (原假設「窗口遷移」經查
      不成立: depth_delta_collector 用 diff 全簿流, 無可見窗; 真實候選
      機制是 fills-as-cancels 近似 — 但陰跌吃 bid, 機械上應推負不推正)
  (B) 真結構偏差 — 現貨簿兩側翻動率天生不對稱, 去均值即可

檢查法 (TODO 凍結):
  ① 淨偏斜按小時漲/跌/橫盤分組 — 綠度跟著下跌時段走→A、橫盤也綠→B
  ② 現貨 vs perp 獨立簿對照 — 同綠→B、只現貨綠→查收集邏輯

定義沿用 watcher v1 凍結式 (raw, 未去均值 — 覆盤圖畫的就是這個):
  net_raw  = ((askC-askA)-(bidC-bidA)) / (bidC+askC)   >0 = 賣側淨撤離(綠)
  skew_raw = (askC-bidC) / (bidC+askC)

Usage: python research/netskew_baseline_check.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

from shared.db import get_db_conn


def load() -> tuple[pd.DataFrame, pd.Series]:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT exchange, minute_start_ms ms,
                       bid_add_qty ba, bid_cancel_qty bc,
                       ask_add_qty aa, ask_cancel_qty ac
                FROM depth_deltas_1m ORDER BY minute_start_ms
            """)
            dd = pd.DataFrame(cur.fetchall())
            cur.execute("""
                SELECT ts_ms, mid_price mid FROM orderbook_snapshots_1m
                WHERE canonical_symbol='BTC-USD' ORDER BY ts_ms
            """)
            ob = pd.DataFrame(cur.fetchall())
    finally:
        conn.close()
    for f in (dd, ob):
        for c in f.columns:
            if c != "exchange":
                f[c] = pd.to_numeric(f[c])
    dd["minute"] = dd["ms"] // 60_000
    ob["minute"] = ob["ts_ms"] // 60_000
    mid = ob.groupby("minute")["mid"].last()
    return dd, mid


def per_minute(dd: pd.DataFrame, exchange: str) -> pd.DataFrame:
    g = (dd[dd["exchange"] == exchange]
         .groupby("minute")[["ba", "bc", "aa", "ac"]].last())
    tot = (g["bc"] + g["ac"]).replace(0, np.nan)
    g["net_raw"] = ((g["ac"] - g["aa"]) - (g["bc"] - g["ba"])) / tot
    g["skew_raw"] = (g["ac"] - g["bc"]) / tot
    return g


def regime_split(g: pd.DataFrame, mid: pd.Series, label: str) -> None:
    """檢查①: 小時報酬 tercile 分組 (漲/橫盤/跌), 比較淨偏斜綠度."""
    df = g.join(mid, how="inner")
    df["hour"] = df.index // 60
    hourly_mid = df.groupby("hour")["mid"].last()
    hret = hourly_mid.pct_change()

    q_lo, q_hi = hret.quantile([1 / 3, 2 / 3])
    regime = pd.Series(np.where(hret <= q_lo, "跌",
                       np.where(hret >= q_hi, "漲", "橫盤")),
                       index=hret.index)
    df["regime"] = df["hour"].map(regime)
    df["hret"] = df["hour"].map(hret)
    df = df.dropna(subset=["regime", "net_raw"])

    print(f"\n── 檢查① {label}: 小時報酬 tercile 分組 "
          f"(界: {q_lo * 100:+.3f}% / {q_hi * 100:+.3f}%) ──")
    print(f"{'組':4s} {'n分鐘':>7s} {'net均值':>9s} {'net中位':>9s} "
          f"{'綠佔比':>7s} {'skew均值':>9s} {'小時報酬均':>9s}")
    for r in ("漲", "橫盤", "跌"):
        s = df[df["regime"] == r]
        if not len(s):
            continue
        print(f"{r:4s} {len(s):7d} {s['net_raw'].mean():+9.4f} "
              f"{s['net_raw'].median():+9.4f} "
              f"{(s['net_raw'] > 0).mean():6.1%} "
              f"{s['skew_raw'].mean():+9.4f} {s['hret'].mean():+8.3%}")
    # 分鐘級相關 (淨偏斜 vs 同分鐘報酬) — A 機制的直接指紋
    ret_1m = df["mid"].pct_change()
    c = df["net_raw"].corr(ret_1m, method="spearman")
    print(f"   分鐘級 spearman(net_raw, ret_1m) = {c:+.4f}  (n={len(df)})")


def spot_vs_perp(spot: pd.DataFrame, perp: pd.DataFrame) -> None:
    """檢查②: 同分鐘現貨 vs perp 獨立簿對照."""
    j = spot[["net_raw", "skew_raw"]].join(
        perp[["net_raw", "skew_raw"]], how="inner",
        lsuffix="_spot", rsuffix="_perp").dropna()
    if len(j) < 100:
        print(f"\n── 檢查② n={len(j)} 重疊分鐘不足, 跳過 ──")
        return
    print(f"\n── 檢查② 現貨 vs perp 同分鐘對照 (n={len(j)}) ──")
    for c in ("net_raw", "skew_raw"):
        s, p = j[f"{c}_spot"], j[f"{c}_perp"]
        print(f"{c:9s} 現貨均值={s.mean():+.4f} 綠佔比={(s > 0).mean():.1%} | "
              f"perp均值={p.mean():+.4f} 綠佔比={(p > 0).mean():.1%} | "
              f"spearman={s.corr(p, method='spearman'):+.3f}")


def main() -> None:
    dd, mid = load()
    for ex, sub in dd.groupby("exchange"):
        lo = pd.Timestamp(sub["ms"].min(), unit="ms")
        hi = pd.Timestamp(sub["ms"].max(), unit="ms")
        print(f"{ex}: {len(sub)} rows, {lo} → {hi} UTC")

    spot = per_minute(dd, "binance")
    perp = per_minute(dd, "binance_perp")
    print(f"\n全期基線: 現貨 net_raw 均值 {spot['net_raw'].mean():+.4f} "
          f"綠佔比 {(spot['net_raw'] > 0).mean():.1%} | "
          f"skew_raw 均值 {spot['skew_raw'].mean():+.4f}")

    regime_split(spot, mid, "現貨")
    if len(perp) > 500:
        regime_split(perp, mid, "perp")
    spot_vs_perp(spot, perp)


if __name__ == "__main__":
    main()
