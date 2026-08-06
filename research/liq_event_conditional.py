"""大型清算事件：條件機率 vs 基準率 —— 這個問題到底可不可答？

問題（使用者 2026-08-06）：「接下來 BTC 會不會有大型清算事件」。

系統裡**沒有**清算事件預測器，而且這正是它反覆失敗過的那類宣稱：
  - 地形戰役 L1-A「清算現場」：被消耗掉的流動性不留下任何效應，全滅
  - 地形戰役 L1-B「清算牆」：n=49 攤不出殘餘格，只拿到門口候選一次復審權
所以不能直接給答案。能做的是：用歷史資料量「在今天這種狀態下，未來 N 小時
出現大型清算的機率，比無條件基準率高多少」，讓資料自己說有沒有 edge。

定義（先寫，後看數據）：
  事件   1h 合計清算額（多+空）落在**全歷史前 1%**（另報前 5% 當穩健性）
  預測窗 未來 4h / 24h 內至少發生一次
  條件   全部 trailing-only、當下可觀測：
           oi_z      OI 相對 trailing 30d 的 z 分數（部位堆積）
           fund_z    資金費率 z（多空成本失衡）
           ls_skew   全市場多空帳戶比 z（散戶站邊）
           top_skew  大戶持倉多空比 z（大戶站邊）
           rv_z      realized vol z（波動狀態；今天在雙錨那邊剛證實它是
                     這份資料裡持續性最強的量）
           liq_z     近 24h 清算額 z（叢集性——清算會不會招來清算）
  判準   條件機率要**明顯高於基準率**，且前後兩半同向，否則就是沒有 edge。
         單一分位桶的漂亮數字不算——要看整條單調性。

用法：python research/liq_event_conditional.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RAW = ROOT / "market_data" / "raw_data"
RNG = np.random.default_rng(3)
NL = chr(10)


def load() -> pd.DataFrame:
    def rd(name, cols=None):
        d = pd.read_parquet(RAW / f"{name}.parquet")
        if not isinstance(d.index, pd.DatetimeIndex):
            tcol = "time" if "time" in d.columns else d.columns[0]
            ts = pd.to_numeric(d[tcol], errors="coerce")
            unit = "s" if ts.dropna().max() < 1e12 else "ms"
            d = d.set_index(pd.to_datetime(ts, unit=unit))
            d = d.drop(columns=[tcol], errors="ignore")
        d = d[~d.index.duplicated(keep="last")].sort_index()
        return d[cols] if cols else d

    liq = rd("cg_liquidation_1h")
    oi = rd("cg_oi_agg_1h")[["close"]].rename(columns={"close": "oi"})
    fund = rd("cg_funding_1h")[["close"]].rename(columns={"close": "fund"})
    gls = rd("cg_global_ls_1h")[["global_account_long_short_ratio"]].rename(
        columns={"global_account_long_short_ratio": "gls"})
    tls = rd("cg_top_ls_position_1h")[["top_position_long_short_ratio"]].rename(
        columns={"top_position_long_short_ratio": "tls"})
    px = rd("binance_klines_1h")[["close"]].rename(columns={"close": "px"})

    df = liq.join([oi, fund, gls, tls, px], how="inner")
    df["liq"] = (pd.to_numeric(df["long_liquidation_usd"], errors="coerce")
                 + pd.to_numeric(df["short_liquidation_usd"], errors="coerce"))
    for c in ("oi", "fund", "gls", "tls", "px"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["liq", "px"]).sort_index()


def z(s: pd.Series, win: int = 720) -> pd.Series:
    """trailing z（30 天窗）。嚴格只用過去。"""
    m = s.rolling(win, min_periods=win // 3).mean()
    sd = s.rolling(win, min_periods=win // 3).std()
    return (s - m) / sd.replace(0, np.nan)


def main() -> int:
    df = load()
    print(f"樣本 {len(df)} 小時  {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d %H:%M}")

    ret = df["px"].pct_change()
    df["rv"] = ret.rolling(24).std()
    df["liq24"] = df["liq"].rolling(24).sum()

    feats = {
        "oi_z": z(df["oi"]),
        "fund_z": z(df["fund"]),
        "ls_skew": z(df["gls"]),
        "top_skew": z(df["tls"]),
        "rv_z": z(df["rv"]),
        "liq_z": z(df["liq24"]),
    }
    for k, v in feats.items():
        df[k] = v

    for pct, tag in ((99, "前 1%"), (95, "前 5%")):
        thr = np.nanpercentile(df["liq"], pct)
        ev = (df["liq"] >= thr).astype(float)
        print(f"\n{'='*70}\n事件定義：1h 合計清算 ≥ {tag}分位 = ${thr/1e6:,.1f}M")
        for H in (4, 24):
            # 未來 H 小時內至少發生一次（不含當下那根）
            fwd = ev.shift(-1).rolling(H, min_periods=1).max().shift(-(H - 1))
            base = np.nanmean(fwd)
            print(f"\n  ── 未來 {H}h 內至少一次 ── 無條件基準率 {base*100:.1f}%")
            print(f"  {'條件':<10}{'最低五分位':>12}{'Q2':>8}{'Q3':>8}{'Q4':>8}"
                  f"{'最高五分位':>12}{'  單調?':>8}{'  兩半同向?':>10}")
            half = len(df) // 2
            for k in feats:
                x = df[k]
                ok = x.notna() & fwd.notna()
                if ok.sum() < 500:
                    print(f"  {k:<10}樣本不足")
                    continue
                q = pd.qcut(x[ok], 5, labels=False, duplicates="drop")
                rates = [np.nanmean(fwd[ok][q == i]) * 100 for i in range(5)]
                mono = (rates[-1] - rates[0])
                # 兩半：各自算最高分位 vs 最低分位的差
                h1 = ok & (np.arange(len(df)) < half)
                h2 = ok & (np.arange(len(df)) >= half)
                d = []
                for m in (h1, h2):
                    if m.sum() < 200:
                        d.append(np.nan); continue
                    qq = pd.qcut(x[m], 5, labels=False, duplicates="drop")
                    d.append(np.nanmean(fwd[m][qq == 4]) * 100
                             - np.nanmean(fwd[m][qq == 0]) * 100)
                agree = "✓" if (len(d) == 2 and np.isfinite(d).all()
                                and d[0] * d[1] > 0 and abs(mono) > 3) else "✗"
                print(f"  {k:<10}" + "".join(f"{r:>11.1f}%" for r in rates)
                      + f"{mono:>+7.1f}pp{agree:>9}")
    print("\n判準：最高分位要明顯高於基準率、整條單調、且兩半同向。"
          "\n單一桶漂亮不算 —— 那正是地形戰役全格報告要擋的事。")

    # ── 當下狀態：每個條件現在落在第幾分位 ──────────────────────────────
    print(NL + "=" * 70)
    print(f"當下狀態（資料截至 {df.index[-1]:%Y-%m-%d %H:%M} UTC）")
    print(f"  BTC ${df['px'].iloc[-1]:,.0f}   近 24h 清算 ${df['liq24'].iloc[-1]/1e6:,.1f}M")
    thr99 = np.nanpercentile(df["liq"], 99)
    ev99 = (df["liq"] >= thr99).astype(float)
    f4 = ev99.shift(-1).rolling(4, min_periods=1).max().shift(-3)
    f24 = ev99.shift(-1).rolling(24, min_periods=1).max().shift(-23)
    b4, b24 = np.nanmean(f4) * 100, np.nanmean(f24) * 100
    print(NL + f"  {'條件':<10}{'現值 z':>9}{'歷史分位':>10}{'落在':>7}"
          f"{'該桶 4h':>10}{'該桶 24h':>11}")
    cur = {}
    for k in feats:
        x = df[k]
        v = x.iloc[-1]
        if not np.isfinite(v):
            print(f"  {k:<10}     —  現值缺")
            continue
        pctl = (x < v).mean() * 100
        ok = x.notna()
        cuts = [np.nanpercentile(x[ok], p) for p in (20, 40, 60, 80)]
        b = int(np.clip(np.searchsorted(cuts, v), 0, 4))
        q = pd.qcut(x[ok], 5, labels=False, duplicates="drop")
        cur[k] = b
        m4 = np.nanmean(f4[ok][q == b]) * 100
        m24 = np.nanmean(f24[ok][q == b]) * 100
        print(f"  {k:<10}{v:>+9.2f}{pctl:>9.0f}%{'Q'+str(b+1):>7}"
              f"{m4:>9.1f}%{m24:>10.1f}%")
    print(NL + f"  無條件基準率（前 1% 事件）： 4h {b4:.1f}%   24h {b24:.1f}%")
    print(f"  現在落在最高分位(Q5)的條件：{[k for k, b in cur.items() if b == 4] or '無'}")
    print(f"  現在落在最低分位(Q1)的條件：{[k for k, b in cur.items() if b == 0] or '無'}")
    print(NL + "  ⚠ 這是**同期條件敘述**，不是預測器：沒有樣本外、沒有 null model、")
    print("    分位曲線多半不單調（只有最極端那格跳起來），而 liq_z 本質是")
    print("    目標自身的自相關（清算叢集），不是獨立資訊源。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
