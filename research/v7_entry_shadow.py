# -*- coding: utf-8 -*-
"""V7 進場執行層 shadow —— 分批 maker 進場 vs 市價一次打滿。

**這是純研究/記帳，不碰交易路徑**：只讀 tracked_signals、只寫
research/results/ 底下的檔案，不 import executor、不下任何單。

## 為什麼做這個

2026-08-04 在 jarvis 網格系統上做完七項「網格當策略」的改動，全部失敗
（庫存上限、時間止血、側向不對稱、delta 對沖、十大幣種、funding 偏斜），
失敗模式一致：任何在單邊行情中保護你的機制本質上都是方向性押注，換半段
或換窗口就翻號。

但把網格重新定位成**執行層**（不產生 edge，只讓已驗證的 edge 進場更便宜）
第一次測就通過：用 V7 的 WF OOS 訊號重放，Strong tier 分批 maker 進場相對
市價一次打滿 +2.7~+3.0 bps，前後兩半同向。Moderate 基準是 −1.6 bps（負 EV，
獨立佐證「只開 Strong」），網格改善它但仍為負。

那次用的是 research 的 WF OOS 預測 + 簡化 tier 解碼。這支腳本改用**真實的
live 訊號**（tracked_signals），是同一個假設的正式前瞻軌道。

## 預註冊（2026-08-04 凍結，不因結果調整）

規則：
  band     0.3%（訊號反方向的鋪單深度；jarvis 那輪 0.3/0.6/1.0 三選一的勝出者）
  K        4 格，等距等量
  窗口     開火後 1 小時
  未成交   窗口結束時市價補齊 —— 兩個變體曝險相同，只差進場均價與費率，可比
  評估     4h（V7 的 target），用 tracked_signals.actual_return_4h
  費率     maker 0.02% / taker 0.05% / 滑點 0.05%（OKX 級別，比 jarvis 那輪保守）

判準（要進 executor 需全部成立）：
  1. Strong 的淨差額為正，且效果量 ≥ +2 bps
  2. 前後兩半同向
  3. n ≥ 200 筆**凍結日之後**的新訊號（回填的歷史只當基準，不算前瞻證據）
  4. 成交假設加嚴後仍成立（見下）

已知的樂觀假設（下一輪要修）：
  用 5m K 線的 low/high 觸及即判定成交。真實限價單在觸價時可能排不到隊，
  這會高估填單率。填單率打折時 maker 費率優勢按比例縮小但不會消失。

時間語意（mistake.md 2026-07-28）：
  signal_time 是 **bar 標籤**，訊號在 label+1h 那根收盤後才誕生。
  進場錨點一律用 created_at（牆鐘），並帶守門：不在 label+[55,75]min 內
  就 fallback 到 label+65min（防 backfill 列的 created_at 失真）。

用法：python research/v7_entry_shadow.py [--tier Strong|Moderate|both]
輸出：research/results/v7_entry_shadow.csv（逐筆）
      research/results/v7_entry_shadow.json（摘要）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from shared.db import get_db_conn  # noqa: E402

RESULTS = PROJECT_ROOT / "research" / "results"
CACHE = PROJECT_ROOT / "research" / ".cache"
KLINE_CACHE = CACHE / "btc_5m_klines.parquet"

# ── 凍結參數 ──
BAND = 0.003
K_LEVELS = 4
WINDOW_H = 1
MAKER = 0.0002
TAKER = 0.0005
SLIP = 0.0005
FROZEN_AT = "2026-08-04"


def fetch_5m_klines(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Binance USDT 永續 5m K 線，本地快取。只讀公開行情，不需金鑰。"""
    if KLINE_CACHE.exists():
        df = pd.read_parquet(KLINE_CACHE)
        if df["time"].min() <= start_ms and df["time"].max() >= end_ms - 300_000:
            return df
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        url = ("https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=5m"
               f"&limit=1000&startTime={cursor}")
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        for k in data:
            rows.append({"time": k[0], "open": float(k[1]), "high": float(k[2]),
                         "low": float(k[3]), "close": float(k[4])})
        cursor = data[-1][0] + 300_000
        if len(data) < 1000:
            break
        time.sleep(0.12)
    df = pd.DataFrame(rows).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(KLINE_CACHE, index=False)
    return df


def load_signals(tier: str) -> pd.DataFrame:
    conn = get_db_conn()
    try:
        q = ("SELECT id, signal_time, created_at, direction, strength, entry_price, "
             "actual_return_4h, correct FROM tracked_signals "
             "WHERE actual_return_4h IS NOT NULL AND entry_price > 0")
        if tier != "both":
            q += f" AND strength = '{tier}'"
        df = pd.read_sql(q, conn)
    finally:
        conn.close()
    return df


def anchor_ms(row) -> int:
    """開火時刻。created_at 落在 label+[55,75]min 才採信，否則 fallback label+65min。"""
    label = pd.Timestamp(row["signal_time"]).tz_localize(timezone.utc)
    created = row["created_at"]
    if pd.notna(created):
        c = pd.Timestamp(created).tz_localize(timezone.utc)
        gap_min = (c - label).total_seconds() / 60
        if 55 <= gap_min <= 75:
            return int(c.timestamp() * 1000)
    return int((label + timedelta(minutes=65)).timestamp() * 1000)


def replay(row, kl: pd.DataFrame) -> dict | None:
    side = 1 if str(row["direction"]).upper() in ("UP", "LONG") else -1
    t0 = anchor_ms(row)
    seg = kl[(kl["time"] >= t0) & (kl["time"] <= t0 + WINDOW_H * 3_600_000)]
    if seg.empty:
        return None
    p0 = float(row["entry_price"])          # 開火價（已對 live 成交驗證過）
    ret4 = float(row["actual_return_4h"])   # DB 已回填的 4h 實際報酬（相對 p0）
    p_eval = p0 * (1 + ret4)

    # A. 市價一次打滿
    entry_mkt = p0 * (1 + side * SLIP)
    pnl_mkt = side * (p_eval - entry_mkt) / entry_mkt - TAKER * 2

    # B. 分批 maker，未成交補市價
    levels = [p0 * (1 - side * BAND * k / K_LEVELS) for k in range(1, K_LEVELS + 1)]
    filled = [False] * K_LEVELS
    for _, c in seg.iterrows():
        for i, px in enumerate(levels):
            if filled[i]:
                continue
            if (side > 0 and c["low"] <= px) or (side < 0 and c["high"] >= px):
                filled[i] = True
    n_fill = sum(filled)
    px_close = float(seg.iloc[-1]["close"])
    sum_fill = sum(px for px, f in zip(levels, filled) if f)
    rest = (K_LEVELS - n_fill) * px_close * (1 + side * SLIP)
    entry_grid = (sum_fill + rest) / K_LEVELS
    fee_grid = (n_fill * MAKER + (K_LEVELS - n_fill) * TAKER) / K_LEVELS
    pnl_grid = side * (p_eval - entry_grid) / entry_grid - fee_grid - TAKER

    return {
        "id": int(row["id"]), "signal_time": row["signal_time"], "anchor_ms": t0,
        "tier": row["strength"], "direction": row["direction"], "side": side,
        "entry_price": p0, "actual_return_4h": ret4,
        "fill_rate": n_fill / K_LEVELS, "entry_grid": entry_grid, "entry_mkt": entry_mkt,
        "entry_imp_bps": side * (entry_mkt - entry_grid) / entry_mkt * 10000,
        "pnl_mkt_bps": pnl_mkt * 10000, "pnl_grid_bps": pnl_grid * 10000,
        "diff_bps": (pnl_grid - pnl_mkt) * 10000,
    }


def summarise(df: pd.DataFrame, tier: str) -> dict:
    if df.empty:
        return {"tier": tier, "n": 0}
    df = df.sort_values("anchor_ms")
    mid = len(df) // 2
    h1, h2 = df.iloc[:mid], df.iloc[mid:]
    frozen_ms = int(pd.Timestamp(FROZEN_AT, tz=timezone.utc).timestamp() * 1000)
    fwd = df[df["anchor_ms"] >= frozen_ms]
    return {
        "tier": tier, "n": len(df),
        "mkt_bps": round(df["pnl_mkt_bps"].mean(), 2),
        "grid_bps": round(df["pnl_grid_bps"].mean(), 2),
        "diff_bps": round(df["diff_bps"].mean(), 2),
        "entry_imp_bps": round(df["entry_imp_bps"].mean(), 2),
        "fill_rate": round(df["fill_rate"].mean(), 3),
        "h1_diff_bps": round(h1["diff_bps"].mean(), 2),
        "h2_diff_bps": round(h2["diff_bps"].mean(), 2),
        "halves_agree": bool(h1["diff_bps"].mean() * h2["diff_bps"].mean() > 0),
        "n_forward": len(fwd),
        "forward_diff_bps": round(fwd["diff_bps"].mean(), 2) if len(fwd) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="both", choices=["Strong", "Moderate", "both"])
    args = ap.parse_args()

    sigs = load_signals(args.tier)
    print(f"訊號 n={len(sigs)}（{args.tier}）")
    if sigs.empty:
        return

    t_min = int(pd.Timestamp(sigs["signal_time"].min(), tz=timezone.utc).timestamp() * 1000)
    t_max = int(pd.Timestamp(sigs["signal_time"].max(), tz=timezone.utc).timestamp() * 1000) + 8 * 3_600_000
    print("抓 5m K 線 …")
    kl = fetch_5m_klines(t_min, t_max)
    print(f"K 線 n={len(kl)}  {pd.to_datetime(kl['time'].min(), unit='ms')} → "
          f"{pd.to_datetime(kl['time'].max(), unit='ms')}")

    rows = [r for r in (replay(s, kl) for _, s in sigs.iterrows()) if r]
    out = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "v7_entry_shadow.csv", index=False)

    summary = {"frozen_at": FROZEN_AT,
               "params": {"band": BAND, "k": K_LEVELS, "window_h": WINDOW_H,
                          "maker": MAKER, "taker": TAKER, "slip": SLIP},
               "by_tier": []}
    for tier in sorted(out["tier"].unique()):
        s = summarise(out[out["tier"] == tier], tier)
        summary["by_tier"].append(s)
        print(f"\n── {tier}（n={s['n']}）")
        print(f"   市價 {s['mkt_bps']:+.2f} bps → 網格 {s['grid_bps']:+.2f} bps"
              f"   差額 {s['diff_bps']:+.2f} bps")
        print(f"   進場改善 {s['entry_imp_bps']:+.2f} bps · 填單率 {s['fill_rate']*100:.0f}%")
        print(f"   兩半 {s['h1_diff_bps']:+.2f} / {s['h2_diff_bps']:+.2f}"
              f"   → {'同向 ✓' if s['halves_agree'] else '反向 ✗'}")
        print(f"   凍結日後前瞻 n={s['n_forward']}"
              + (f"（{s['forward_diff_bps']:+.2f} bps）" if s["n_forward"] else "（尚未累積）"))
    (RESULTS / "v7_entry_shadow.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n已存 {RESULTS/'v7_entry_shadow.csv'} 與 .json")


if __name__ == "__main__":
    main()
