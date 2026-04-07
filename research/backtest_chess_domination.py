"""
ChessDomination Pressure Framework — 信號回測模組
===================================================
提供 CDP 信號的歷史回測與 IC 分析，整合到現有研究管線。

使用方式：
    python research/backtest_chess_domination.py

功能：
    1. IC 分析（CDP 特徵 vs 未來 4h 收益的 Spearman 相關）
    2. 信號勝率統計（按 zone 拆解）
    3. 分位 Monotonicity 檢驗（CDP 越強 → 收益越大？）
    4. 整合範例（如何嵌入現有 feature_builder_live 或 vectorbt）
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.features.chess_domination import (
    CDPConfig,
    CDPResult,
    CDPSignalGenerator,
    ChessDominationPressure,
)

logger = logging.getLogger(__name__)

HORIZON = 4  # 4h 預測視窗（與系統一致）


# ═══════════════════════════════════════════════════════════════════════════
# IC 分析
# ═══════════════════════════════════════════════════════════════════════════

def compute_ic_analysis(
    df: pd.DataFrame,
    result: Optional[CDPResult] = None,
    config: Optional[CDPConfig] = None,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """
    計算 CDP 所有子特徵與未來 {horizon}h 收益的 Spearman IC。

    Returns:
        DataFrame with columns: feature, ic, ic_pvalue, abs_ic, icir_7d, icir_30d
    """
    if result is None:
        engine = ChessDominationPressure(config)
        result = engine.compute_all(df)

    # 目標：未來 4h 收益
    fwd_ret = df["close"].pct_change(horizon).shift(-horizon)
    fwd_ret.name = "y_return_4h"

    features = {
        "cdp_x": result.x,
        "cdp_y": result.y,
        "cdp_z": result.z,
        "cdp_cufd": result.cufd,
        "cdp_score": result.cdp,
        "cdp_abs": result.cdp.abs(),
        "cdp_zscore": result.cdp_zscore,
    }

    rows = []
    for name, feat in features.items():
        # 對齊有效行
        valid = pd.concat([feat, fwd_ret], axis=1).dropna()
        if len(valid) < 50:
            rows.append({"feature": name, "ic": np.nan, "ic_pvalue": np.nan})
            continue

        ic, pval = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])

        # Rolling IC (7d = 168h, 30d = 720h)
        rolling_ic_7d = feat.rolling(168, min_periods=48).corr(fwd_ret)
        rolling_ic_30d = feat.rolling(720, min_periods=168).corr(fwd_ret)

        icir_7d = rolling_ic_7d.mean() / rolling_ic_7d.std() if rolling_ic_7d.std() > 0 else 0
        icir_30d = rolling_ic_30d.mean() / rolling_ic_30d.std() if rolling_ic_30d.std() > 0 else 0

        rows.append({
            "feature": name,
            "ic": round(ic, 4),
            "ic_pvalue": round(pval, 6),
            "abs_ic": round(abs(ic), 4),
            "icir_7d": round(icir_7d, 3),
            "icir_30d": round(icir_30d, 3),
            "n_samples": len(valid),
        })

    ic_df = pd.DataFrame(rows).sort_values("abs_ic", ascending=False)
    return ic_df


# ═══════════════════════════════════════════════════════════════════════════
# 信號勝率統計
# ═══════════════════════════════════════════════════════════════════════════

def compute_signal_stats(
    df: pd.DataFrame,
    result: Optional[CDPResult] = None,
    config: Optional[CDPConfig] = None,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """
    統計各類 CDP 信號的勝率、平均收益、勝負比。

    Returns:
        DataFrame with signal_type breakdown
    """
    if result is None:
        engine = ChessDominationPressure(config)
        result = engine.compute_all(df)

    gen = CDPSignalGenerator(config)
    signals_df = gen.generate_signals(result)

    # 目標收益
    fwd_ret = df["close"].pct_change(horizon).shift(-horizon)

    # 合併
    merged = pd.concat([signals_df, fwd_ret.rename("fwd_return")], axis=1).dropna(subset=["fwd_return"])

    stats = []
    for sig_type in merged["cdp_signal_type"].unique():
        if sig_type == "none":
            continue
        mask = merged["cdp_signal_type"] == sig_type
        subset = merged[mask]
        if len(subset) < 5:
            continue

        signal = subset["cdp_signal"]
        ret = subset["fwd_return"]

        # 信號方向乘以收益 = 調整後收益
        adj_ret = signal * ret

        win_rate = (adj_ret > 0).mean()
        avg_ret = adj_ret.mean()
        avg_win = adj_ret[adj_ret > 0].mean() if (adj_ret > 0).any() else 0
        avg_loss = adj_ret[adj_ret <= 0].mean() if (adj_ret <= 0).any() else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

        stats.append({
            "signal_type": sig_type,
            "count": len(subset),
            "win_rate": round(win_rate, 4),
            "avg_adj_return": round(avg_ret * 100, 4),  # %
            "avg_win_pct": round(avg_win * 100, 4),
            "avg_loss_pct": round(avg_loss * 100, 4),
            "profit_factor": round(profit_factor, 2),
            "avg_strength": round(subset["cdp_signal_strength"].mean(), 3),
        })

    return pd.DataFrame(stats)


# ═══════════════════════════════════════════════════════════════════════════
# Zone 拆解分析
# ═══════════════════════════════════════════════════════════════════════════

def compute_zone_analysis(
    df: pd.DataFrame,
    result: Optional[CDPResult] = None,
    config: Optional[CDPConfig] = None,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """
    按棋局區域拆解未來收益統計。

    核心問題：「紅方過度延伸危險區」後真的會跌嗎？
    """
    if result is None:
        engine = ChessDominationPressure(config)
        result = engine.compute_all(df)

    fwd_ret = df["close"].pct_change(horizon).shift(-horizon)
    merged = pd.concat([result.zone.rename("zone"), fwd_ret.rename("fwd_return")], axis=1).dropna()

    stats = []
    for zone in merged["zone"].unique():
        subset = merged[merged["zone"] == zone]
        ret = subset["fwd_return"]

        stats.append({
            "zone": zone,
            "count": len(subset),
            "pct_of_total": round(len(subset) / len(merged) * 100, 1),
            "avg_return_pct": round(ret.mean() * 100, 4),
            "median_return_pct": round(ret.median() * 100, 4),
            "std_pct": round(ret.std() * 100, 4),
            "up_rate": round((ret > 0).mean() * 100, 1),
            "down_rate": round((ret < 0).mean() * 100, 1),
            "avg_abs_return_pct": round(ret.abs().mean() * 100, 4),
        })

    return pd.DataFrame(stats).sort_values("count", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════
# 分位 Monotonicity 檢驗
# ═══════════════════════════════════════════════════════════════════════════

def compute_quantile_monotonicity(
    df: pd.DataFrame,
    result: Optional[CDPResult] = None,
    config: Optional[CDPConfig] = None,
    n_bins: int = 5,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """
    CDP 分位校準檢驗：CDP 越極端 → 反向收益越大？

    將 |CDP| 分成 n_bins 等分，看每個分位的平均 adjusted return。
    理想狀態：嚴格單調遞增（CDP 越強，反向收益越高）。
    """
    if result is None:
        engine = ChessDominationPressure(config)
        result = engine.compute_all(df)

    cdp_abs = result.cdp.abs()
    fwd_ret = df["close"].pct_change(horizon).shift(-horizon)
    cdp_sign = np.sign(result.cdp)

    # Adjusted return: CDP 正（多方過載）× 負收益 = 正值 → 信號正確
    adj_ret = -cdp_sign * fwd_ret

    merged = pd.DataFrame({
        "cdp_abs": cdp_abs,
        "adj_return": adj_ret,
    }).dropna()

    if len(merged) < n_bins * 10:
        return pd.DataFrame()

    # 分位
    merged["quantile"] = pd.qcut(merged["cdp_abs"], n_bins, labels=False, duplicates="drop")

    stats = []
    for q in sorted(merged["quantile"].unique()):
        subset = merged[merged["quantile"] == q]
        stats.append({
            "quantile": int(q),
            "cdp_abs_range": f"[{subset['cdp_abs'].min():.4f}, {subset['cdp_abs'].max():.4f}]",
            "count": len(subset),
            "avg_adj_return_pct": round(subset["adj_return"].mean() * 100, 4),
            "hit_rate": round((subset["adj_return"] > 0).mean() * 100, 1),
        })

    result_df = pd.DataFrame(stats)

    # 單調性檢驗
    returns = result_df["avg_adj_return_pct"].values
    is_monotonic = all(returns[i] <= returns[i + 1] for i in range(len(returns) - 1))
    logger.info("Monotonicity check: %s (returns: %s)",
                "PASS" if is_monotonic else "FAIL", returns)

    return result_df


# ═══════════════════════════════════════════════════════════════════════════
# 整合範例
# ═══════════════════════════════════════════════════════════════════════════

def integration_example_feature_builder():
    """
    範例：如何將 CDP 整合到 feature_builder_live.py

    在 build_live_features() 末尾加入：

        # ── CDP 四維因果棋局特徵 ──────────────────────────────────
        try:
            from research.features.chess_domination import compute_cdp_features
            cdp_feats = compute_cdp_features(df)
            new_cols = [c for c in cdp_feats.columns if c not in df.columns]
            if new_cols:
                df = pd.concat([df, cdp_feats[new_cols]], axis=1)
                logger.info("CDP features added: %d columns", len(new_cols))
        except ImportError:
            logger.debug("chess_domination not available — skipping")
        except Exception as e:
            logger.warning("CDP features failed (non-critical): %s", e)

    並在 feature_config.py 的 ALL_FEATURES 中加入：

        CDP_FEATURES = [
            "cdp_x",              # 收益區間 [-1, 1]
            "cdp_y",              # 主力資金流向 [-1, 1]
            "cdp_z",              # 巨鲸大單強度 [0, 1]
            "cdp_cufd",           # 因果累積壓力
            "cdp_cufd_side",      # 壓力方向 +1/-1
            "cdp_score",          # CDP 濃縮特徵（帶方向）
            "cdp_zscore",         # CDP z-score
            "cdp_zone_code",      # 區域編碼 -5~+5
            "cdp_pressure_level", # 壓力等級 0~5
        ]
    """
    pass


def integration_example_vectorbt():
    """
    範例：如何用 vectorbt 回測 CDP 信號

        import vectorbt as vbt

        # 1. 計算 CDP
        engine = ChessDominationPressure()
        result = engine.compute_all(df)

        # 2. 生成信號
        gen = CDPSignalGenerator()
        signals = gen.generate_signals(result)

        # 3. 建立 entry/exit
        entries = signals["cdp_signal"] != 0
        direction = signals["cdp_signal"]  # 1 = long, -1 = short

        long_entries = direction == 1
        short_entries = direction == -1

        # 4. 固定持倉 4h 後平倉
        long_exits = long_entries.shift(4).fillna(False)
        short_exits = short_entries.shift(4).fillna(False)

        # 5. 回測
        pf = vbt.Portfolio.from_signals(
            close=df["close"],
            entries=long_entries,
            exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            freq="1h",
            init_cash=100000,
            fees=0.0004,  # taker fee
        )

        print(pf.stats())
    """
    pass


# ═══════════════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 70)
    print("  CDP Framework Backtest")
    print("  ChessDomination Pressure — 四維因果棋局回測分析")
    print("=" * 70)

    # 嘗試載入真實資料
    try:
        from indicator.data_fetcher import fetch_binance_klines, fetch_coinglass
        from indicator.feature_builder_live import build_live_features

        print("\n[1/4] 取得 Binance + Coinglass 資料...")
        klines = fetch_binance_klines(limit=500)
        cg_data = {}
        for ep in ["oi", "oi_agg", "funding", "taker", "long_short",
                    "global_ls", "liquidation", "futures_cvd_agg"]:
            try:
                cg_data[ep] = fetch_coinglass(ep, "BTC", interval="1h", limit=500)
            except Exception as e:
                print(f"  跳過 {ep}: {e}")

        df = build_live_features(klines, cg_data)
        print(f"  資料範圍: {df.index[0]} ~ {df.index[-1]} ({len(df)} bars)")

    except ImportError:
        print("\n[1/4] 使用模擬資料...")
        np.random.seed(42)
        n = 500
        idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
        close = 85000 + np.cumsum(np.random.randn(n) * 100)
        df = pd.DataFrame({
            "open": close - np.random.rand(n) * 50,
            "high": close + np.random.rand(n) * 100,
            "low": close - np.random.rand(n) * 100,
            "close": close,
            "volume": np.random.lognormal(8, 0.5, n),
            "taker_buy_vol": np.random.lognormal(7.5, 0.5, n),
            "trade_count": np.random.randint(5000, 50000, n),
            "cg_funding_close": np.random.randn(n) * 0.0003,
            "cg_oi_close": 15e9 + np.cumsum(np.random.randn(n) * 1e8),
            "cg_taker_buy": np.random.lognormal(20, 0.3, n),
            "cg_taker_sell": np.random.lognormal(20, 0.3, n),
        }, index=idx)
        df["cg_taker_delta"] = df["cg_taker_buy"] - df["cg_taker_sell"]

    # 計算 CDP
    print("\n[2/4] 計算 CDP 框架...")
    engine = ChessDominationPressure()
    result = engine.compute_all(df)

    # IC 分析
    print("\n[3/4] IC 分析")
    print("-" * 60)
    ic_df = compute_ic_analysis(df, result)
    print(ic_df.to_string(index=False))

    # 信號統計
    print("\n[3/4] 信號勝率統計")
    print("-" * 60)
    sig_stats = compute_signal_stats(df, result)
    if not sig_stats.empty:
        print(sig_stats.to_string(index=False))
    else:
        print("  資料量不足，無法產生信號統計")

    # Zone 分析
    print("\n[3/4] 棋局區域拆解")
    print("-" * 60)
    zone_stats = compute_zone_analysis(df, result)
    print(zone_stats.to_string(index=False))

    # Monotonicity
    print("\n[4/4] 分位 Monotonicity 檢驗")
    print("-" * 60)
    mono = compute_quantile_monotonicity(df, result)
    if not mono.empty:
        print(mono.to_string(index=False))
    else:
        print("  資料量不足")

    # 最新狀態
    print("\n" + "=" * 60)
    print("  當前棋局狀態")
    print("=" * 60)
    summary = result.summary(-1)
    for k, v in summary.items():
        print(f"  {k:>18s}: {v}")

    # 整合說明
    print("\n" + "=" * 60)
    print("  整合方式")
    print("=" * 60)
    print("  1. 特徵整合到 feature_builder_live.py:")
    print("     → 見 integration_example_feature_builder() docstring")
    print("  2. vectorbt 回測:")
    print("     → 見 integration_example_vectorbt() docstring")
    print("  3. 3D 可視化:")
    print("     → python research/visualize_chess_domination.py")
    print("=" * 60)
