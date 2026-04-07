"""
ChessDomination Pressure Framework (CDP Framework)
=====================================================
四維因果棋局框架 — 將「三維棋局心法」升級為含因果累積壓力的完整量化特徵。

維度定義：
  X 軸（收益區間）：價格在近期波動區間的相對位置 → 誰掌握價格位置
  Y 軸（主力資金流向）：taker 資金流向的標準化值 → 誰掌握布局主動
  Z 軸（巨鲸大單強度）：大單佔比 × 方向不平衡 → 攻擊是否持續
  CUFD（因果累積壓力）：正 funding 累積未釋放天數 → 弱勢持倉被迫平倉壓力

最終濃縮特徵：
  CDP = (X² + Z²) × f(Y) × Z × g(CUFD)

使用資料源：
  - Binance 1h klines (OHLCV, taker_buy_vol)
  - Coinglass: funding_close, oi_close, taker_buy/sell
  - aggTrades (optional): large_ratio, large_buy_ratio, large_delta_usd
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 參數配置（可根據回測結果調整）
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CDPConfig:
    """CDP Framework 所有可調參數集中管理。"""

    # ── X 軸：收益區間 ──
    x_lookback: int = 48          # 回看 48 根 H1（2 天）計算 high-low 區間
    x_smooth: int = 4             # 4h EMA 平滑，減少噪音

    # ── Y 軸：主力資金流向 ──
    y_zscore_window: int = 24     # 24h rolling z-score 標準化窗口
    y_clip: float = 1.0           # Y 值裁剪到 [-1, 1]

    # ── Z 軸：巨鲸大單強度 ──
    z_lookback: int = 24          # 大單佔比的標準化回看窗口
    z_floor: float = 0.0          # Z 下限
    z_cap: float = 1.0            # Z 上限

    # ── CUFD：因果累積壓力 ──
    cufd_decay: float = 0.97      # 每小時指數衰減因子（λ=0.97 → 半衰期 ≈ 23h）
    cufd_oi_release_k: float = 0.5  # OI 增量釋放係數（新開倉越多，釋放越多）
    cufd_theta_quantile: float = 0.87  # 歷史痛點閾值 θ 的分位數（87%）
    cufd_theta_window: int = 720  # 計算 θ 的回看窗口（720h ≈ 30 天）

    # ── CDP 合成 ──
    y_pressure_offset: float = 0.65   # f(Y) 激活門檻
    y_pressure_scale: float = 5.0     # sigmoid 平滑版的 scale
    cufd_amplify_exp: float = 1.2     # g(CUFD) 的冪次放大
    cufd_softplus_beta: float = 3.0   # softplus 的 beta 參數

    # ── 狀態區域判定 ──
    y_dominance: float = 0.70     # Y ≥ 此值 = 紅方壓境
    z_heavy_fire: float = 0.50    # Z ≥ 此值 = 重炮未熄火
    y_overextend: float = 0.85    # Y ≥ 此值 + CUFD > θ = 過度延伸危險區
    z_exhaustion: float = 0.15    # Z < 此值 = 火力枯竭


# ═══════════════════════════════════════════════════════════════════════════
# 核心類別
# ═══════════════════════════════════════════════════════════════════════════

class ChessDominationPressure:
    """
    四維因果棋局框架核心計算引擎。

    輸入：含有 OHLCV + Coinglass 欄位的 1h DataFrame
    輸出：X, Y, Z, CUFD, CDP series + 當前狀態判定

    使用範例：
        cdp_engine = ChessDominationPressure()
        result = cdp_engine.compute_all(df)
        # result.cdp → 最終 CDP 特徵 series
        # result.state → 當前棋局狀態
    """

    def __init__(self, config: Optional[CDPConfig] = None):
        self.cfg = config or CDPConfig()
        self._theta: Optional[float] = None  # 快取 CUFD 閾值

    # ─────────────────────────────────────────────────────────────────────
    # X 軸：收益區間歸一化（價格在近期波動區間的相對位置）
    # ─────────────────────────────────────────────────────────────────────
    def compute_x(self, df: pd.DataFrame) -> pd.Series:
        """
        X = (close - rolling_low) / (rolling_high - rolling_low) → [0, 1] → 映射到 [-1, 1]

        意義：
          +1 = 價格在近期區間最高處（多方完全掌控）
          -1 = 價格在近期區間最低處（空方完全掌控）
           0 = 價格在區間中央（勢均力敵）

        使用 EMA 平滑避免單根極端 bar 導致跳動。
        """
        c = self.cfg
        close = df["close"]
        rolling_high = df["high"].rolling(c.x_lookback, min_periods=10).max()
        rolling_low = df["low"].rolling(c.x_lookback, min_periods=10).min()

        # 區間歸一化到 [0, 1]
        range_width = (rolling_high - rolling_low).replace(0, np.nan)
        x_raw = (close - rolling_low) / range_width  # [0, 1]

        # EMA 平滑
        x_smooth = x_raw.ewm(span=c.x_smooth, min_periods=2).mean()

        # 映射到 [-1, 1]
        x = 2.0 * x_smooth - 1.0

        return x.rename("cdp_x")

    # ─────────────────────────────────────────────────────────────────────
    # Y 軸：主力資金流向（taker delta 的標準化值）
    # ─────────────────────────────────────────────────────────────────────
    def compute_y(self, df: pd.DataFrame) -> pd.Series:
        """
        Y = zscore(ΔCVD / total_volume) → clipped to [-1, 1]

        意義：
          +1 = 主力強力買入，資金流入多方（紅方話語權）
          -1 = 主力強力賣出，資金流入空方（綠方話語權）
           0 = 楚河漢界（兩軍對峙）

        優先使用 Coinglass taker_delta（跨交易所聚合），
        回退到 Binance kline 的 taker_buy_vol proxy。
        """
        c = self.cfg

        # 方案 1：Coinglass taker delta（更精準，跨交易所）
        if "cg_taker_delta" in df.columns and df["cg_taker_delta"].notna().sum() > 10:
            taker_delta = df["cg_taker_delta"]
            total_vol = (df["cg_taker_buy"] + df["cg_taker_sell"]).replace(0, np.nan)
        else:
            # 方案 2：Binance kline proxy
            taker_sell = df["volume"] - df["taker_buy_vol"]
            taker_delta = df["taker_buy_vol"] - taker_sell
            total_vol = df["volume"].replace(0, np.nan)

        # ΔCVD / total_volume → 標準化的資金流向
        flow_ratio = taker_delta / total_vol

        # Rolling z-score
        mu = flow_ratio.rolling(c.y_zscore_window, min_periods=6).mean()
        sd = flow_ratio.rolling(c.y_zscore_window, min_periods=6).std().replace(0, np.nan)
        y_zscore = (flow_ratio - mu) / sd

        # Clip 到 [-1, 1]（超過 1σ 就視為極端）
        # 用 tanh 做平滑壓縮，比硬 clip 更好
        y = np.tanh(y_zscore / 2.0)  # tanh(z/2) 在 z=±2 時 ≈ ±0.76

        return y.clip(-c.y_clip, c.y_clip).rename("cdp_y")

    # ─────────────────────────────────────────────────────────────────────
    # Z 軸：巨鲸大單強度
    # ─────────────────────────────────────────────────────────────────────
    def compute_z(self, df: pd.DataFrame) -> pd.Series:
        """
        Z = large_order_intensity × directional_imbalance → [0, 1]

        意義：
          1.0 = 巨鲸全力單方向攻擊（重炮齊射）
          0.5 = 中等強度（重炮門檻）
          0.0 = 無大單活動或雙向均衡（火力枯竭）

        三種計算路徑（按資料可用性回退）：
        1. aggTrades 精確路徑：large_ratio × |large_buy_ratio - 0.5| × 2
        2. Coinglass Futures CVD 路徑：|cg_fcvd_delta| 的 z-score
        3. Kline proxy 路徑：avg_trade_size × |taker_delta_ratio|
        """
        c = self.cfg

        # ── 路徑 1：aggTrades（最精確）──
        if "agg_large_ratio" in df.columns and df["agg_large_ratio"].notna().sum() > 5:
            large_ratio = df["agg_large_ratio"].fillna(0)
            # large_buy_ratio = 大單中買入佔比，0.5 = 均衡
            if "agg_large_buy_ratio" in df.columns:
                buy_ratio = df["agg_large_buy_ratio"].fillna(0.5)
            else:
                buy_ratio = pd.Series(0.5, index=df.index)
            # 方向不平衡度：|buy_ratio - 0.5| × 2 → [0, 1]
            dir_imbalance = (buy_ratio - 0.5).abs() * 2.0
            z_raw = large_ratio * dir_imbalance
            logger.debug("Z 軸使用 aggTrades 精確路徑")

        # ── 路徑 2：Coinglass Futures CVD ──
        elif "cg_fcvd_delta" in df.columns and df["cg_fcvd_delta"].notna().sum() > 10:
            fcvd_abs = df["cg_fcvd_delta"].abs()
            fcvd_mu = fcvd_abs.rolling(c.z_lookback, min_periods=6).mean()
            fcvd_sd = fcvd_abs.rolling(c.z_lookback, min_periods=6).std().replace(0, np.nan)
            z_raw = ((fcvd_abs - fcvd_mu) / fcvd_sd).clip(0, 3) / 3.0
            logger.debug("Z 軸使用 Coinglass Futures CVD 路徑")

        # ── 路徑 3：Kline proxy ──
        else:
            tc = df.get("trade_count", pd.Series(1, index=df.index)).replace(0, np.nan)
            avg_size = df["volume"] / tc
            avg_size_z = self._rolling_zscore(avg_size, c.z_lookback)

            taker_sell = df["volume"] - df["taker_buy_vol"]
            taker_delta_ratio = (df["taker_buy_vol"] - taker_sell) / df["volume"].replace(0, np.nan)
            dir_strength = taker_delta_ratio.abs()

            z_raw = (avg_size_z.clip(0, 3) / 3.0) * dir_strength
            logger.debug("Z 軸使用 Kline proxy 路徑")

        # 壓縮到 [0, 1]
        z = z_raw.clip(c.z_floor, c.z_cap)

        return z.rename("cdp_z")

    # ─────────────────────────────────────────────────────────────────────
    # CUFD：因果累積壓力（Cumulative Unreleased Funding Days）
    # ─────────────────────────────────────────────────────────────────────
    def compute_cufd(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, float]:
        """
        CUFD 計算邏輯：

        1. 每根 bar 的 funding 成本 = |funding_rate| × OI（持倉量）
        2. 只累積「逆勢方的成本」：
           - funding > 0 → 多方付費，累積多方壓力（CUFD_long）
           - funding < 0 → 空方付費，累積空方壓力（CUFD_short）
        3. OI 增量釋放效果：新開倉越多 → 成本被新參與者分攤 → 釋放壓力
           release_factor = max(0, ΔOI / OI) × release_k
        4. 指數衰減：每小時 × decay_factor，讓近期 funding 權重更高
        5. CUFD = max(CUFD_long, CUFD_short)（取壓力較大的一方）

        Returns:
            cufd: CUFD series（正值，壓力越大越高）
            cufd_side: +1 = 多方承壓, -1 = 空方承壓
            theta: 歷史痛點閾值
        """
        c = self.cfg

        # 取得 funding rate 和 OI
        funding = df.get("cg_funding_close", pd.Series(0, index=df.index)).fillna(0)
        oi = df.get("cg_oi_close", pd.Series(1, index=df.index)).fillna(method="ffill").fillna(1)

        # OI 變化率（正 = 新開倉，負 = 平倉）
        oi_change_pct = oi.pct_change().fillna(0)

        # 逐 bar 累積（向量化展開）
        n = len(df)
        cufd_long = np.zeros(n)   # 多方累積壓力
        cufd_short = np.zeros(n)  # 空方累積壓力

        decay = c.cufd_decay
        release_k = c.cufd_oi_release_k

        funding_vals = funding.values
        oi_vals = oi.values
        oi_chg_vals = oi_change_pct.values

        for i in range(1, n):
            # 衰減前一期的累積
            cufd_long[i] = cufd_long[i - 1] * decay
            cufd_short[i] = cufd_short[i - 1] * decay

            f = funding_vals[i]
            oi_val = oi_vals[i]

            # OI 釋放效果：新開倉越多，壓力釋放越多
            oi_release = max(0, oi_chg_vals[i]) * release_k
            release_factor = max(0, 1.0 - oi_release)

            if f > 0:
                # 多方付費 → 累積多方壓力
                # 成本 = funding_rate × OI（歸一化到合理尺度）
                cost = f * (oi_val / 1e9)  # 以十億 USD OI 為基準歸一化
                cufd_long[i] = cufd_long[i] * release_factor + cost
            elif f < 0:
                # 空方付費 → 累積空方壓力
                cost = abs(f) * (oi_val / 1e9)
                cufd_short[i] = cufd_short[i] * release_factor + cost

        cufd_long_s = pd.Series(cufd_long, index=df.index)
        cufd_short_s = pd.Series(cufd_short, index=df.index)

        # 取壓力較大方作為 CUFD 主值
        cufd = np.maximum(cufd_long, cufd_short)
        cufd_side = np.where(cufd_long >= cufd_short, 1.0, -1.0)
        # 當壓力極低時標記為中性
        cufd_side = np.where(cufd < 1e-10, 0.0, cufd_side)

        cufd_s = pd.Series(cufd, index=df.index, name="cdp_cufd")
        cufd_side_s = pd.Series(cufd_side, index=df.index, name="cdp_cufd_side")

        # 計算歷史痛點閾值 θ
        lookback = min(c.cufd_theta_window, len(cufd_s))
        theta = float(cufd_s.iloc[-lookback:].quantile(c.cufd_theta_quantile))
        self._theta = theta

        return cufd_s, cufd_side_s, theta

    # ─────────────────────────────────────────────────────────────────────
    # CDP：最終濃縮特徵
    # ─────────────────────────────────────────────────────────────────────
    def compute_cdp(self, x: pd.Series, y: pd.Series,
                    z: pd.Series, cufd: pd.Series,
                    theta: float) -> pd.Series:
        """
        CDP = (X² + Z²) × f(Y) × Z × g(CUFD)

        f(Y) — Y 軸壓力函數（sigmoid 平滑版）：
          當 |Y| > 0.65 時開始激活，用 sigmoid 平滑過渡
          f(Y) = sigmoid((|Y| - offset) × scale)
          → |Y| = 0.5 時 ≈ 0.12（低壓力）
          → |Y| = 0.7 時 ≈ 0.57（中壓力）
          → |Y| = 0.9 時 ≈ 0.92（高壓力）

        g(CUFD) — 痛點放大函數（softplus + 冪次）：
          g(CUFD) = 1 + softplus(CUFD - θ, β)^α
          → CUFD < θ 時 ≈ 1（無放大）
          → CUFD > θ 時指數放大（因果反噬啟動）

        CDP 帶方向：正 = 多方壓力（利空），負 = 空方壓力（利多）
        """
        c = self.cfg

        # (X² + Z²) — 空間距離（越遠離中心 = 勢能越大）
        spatial_energy = x ** 2 + z ** 2

        # f(Y) — 資金流向壓力函數（sigmoid 平滑）
        y_abs = y.abs()
        f_y = 1.0 / (1.0 + np.exp(-c.y_pressure_scale * (y_abs - c.y_pressure_offset)))

        # g(CUFD) — 因果累積放大函數
        # softplus(x) = (1/β) × ln(1 + exp(β × x))
        excess = cufd - theta
        softplus_val = (1.0 / c.cufd_softplus_beta) * np.log(
            1.0 + np.exp(c.cufd_softplus_beta * excess)
        )
        g_cufd = 1.0 + softplus_val ** c.cufd_amplify_exp

        # CDP = 空間勢能 × 資金壓力 × 巨鲸強度 × 因果放大
        cdp_magnitude = spatial_energy * f_y * z * g_cufd

        # 帶方向（Y 的符號決定壓力方向）
        # Y > 0 = 多方主導 → CDP 正（可能反轉向下）
        # Y < 0 = 空方主導 → CDP 負（可能反轉向上）
        cdp = cdp_magnitude * np.sign(y)

        return cdp.rename("cdp_score")

    # ─────────────────────────────────────────────────────────────────────
    # 狀態區域判定
    # ─────────────────────────────────────────────────────────────────────
    def get_3d_state(self, x: pd.Series, y: pd.Series,
                     z: pd.Series, cufd: pd.Series,
                     theta: float) -> Tuple[pd.Series, pd.Series]:
        """
        根據 XYZ + CUFD 判定當前所屬區域與壓力等級。

        區域定義：
        ┌────────────────────────────────────────────────────────────────┐
        │ 紅方過度延伸危險區  │ Y ≥ 0.85, Z ≥ 0.5, CUFD > θ          │
        │                     │ → 淝水之戰局面，正回饋反噬即將啟動     │
        │ 紅方壓境優勢區      │ Y ≥ 0.70, Z ≥ 0.5                     │
        │                     │ → 多方掌控全局，重炮持續火力           │
        │ 紅方輕度優勢區      │ Y ≥ 0.35, Z < 0.5                     │
        │                     │ → 多方偏強但火力不足                   │
        │ 楚河漢界（中性區）  │ |Y| < 0.35                            │
        │                     │ → 兩軍對峙，方向不明                   │
        │ 空方壓境優勢區      │ Y ≤ -0.70, Z ≥ 0.5                    │
        │ 空方過度延伸危險區  │ Y ≤ -0.85, Z ≥ 0.5, CUFD > θ         │
        │ 火力枯竭區          │ Z < 0.15 （任何 Y 值）                │
        │                     │ → 大單消失，難以持續方向               │
        └────────────────────────────────────────────────────────────────┘

        Returns:
            zone_names: 區域名稱 series
            pressure_level: 壓力等級 0~5
        """
        c = self.cfg
        n = len(x)
        zones = pd.Series("中性區", index=x.index, dtype="object")
        levels = pd.Series(0, index=x.index, dtype="int")

        for i in range(n):
            yi, zi, ci = y.iloc[i], z.iloc[i], cufd.iloc[i]

            # 火力枯竭（優先判定，Z 太低什麼都做不了）
            if zi < c.z_exhaustion:
                zones.iloc[i] = "火力枯竭區"
                levels.iloc[i] = 0
                continue

            # 多方場景
            if yi >= c.y_overextend and zi >= c.z_heavy_fire and ci > theta:
                zones.iloc[i] = "紅方過度延伸危險區"
                levels.iloc[i] = 5  # 最高壓力
            elif yi >= c.y_dominance and zi >= c.z_heavy_fire:
                zones.iloc[i] = "紅方壓境優勢區"
                levels.iloc[i] = 4
            elif yi >= 0.35:
                zones.iloc[i] = "紅方輕度優勢區"
                levels.iloc[i] = 2

            # 空方場景（鏡像）
            elif yi <= -c.y_overextend and zi >= c.z_heavy_fire and ci > theta:
                zones.iloc[i] = "空方過度延伸危險區"
                levels.iloc[i] = 5
            elif yi <= -c.y_dominance and zi >= c.z_heavy_fire:
                zones.iloc[i] = "空方壓境優勢區"
                levels.iloc[i] = 4
            elif yi <= -0.35:
                zones.iloc[i] = "空方輕度優勢區"
                levels.iloc[i] = 2

            # 中性
            else:
                zones.iloc[i] = "楚河漢界"
                levels.iloc[i] = 1

        return zones.rename("cdp_zone"), levels.rename("cdp_pressure_level")

    # ─────────────────────────────────────────────────────────────────────
    # 一鍵計算全部（主入口）
    # ─────────────────────────────────────────────────────────────────────
    def compute_all(self, df: pd.DataFrame) -> CDPResult:
        """
        一次計算所有 CDP 相關特徵。

        Parameters:
            df: 含有 OHLCV + Coinglass 欄位的 1h DataFrame
                必要欄位：open, high, low, close, volume, taker_buy_vol
                可選欄位：cg_funding_close, cg_oi_close, cg_taker_buy, cg_taker_sell,
                          agg_large_ratio, agg_large_buy_ratio, cg_fcvd_delta

        Returns:
            CDPResult dataclass with all computed series
        """
        # 驗證必要欄位
        required = ["open", "high", "low", "close", "volume", "taker_buy_vol"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"缺少必要欄位: {missing}")

        x = self.compute_x(df)
        y = self.compute_y(df)
        z = self.compute_z(df)
        cufd, cufd_side, theta = self.compute_cufd(df)
        cdp = self.compute_cdp(x, y, z, cufd, theta)
        zones, levels = self.get_3d_state(x, y, z, cufd, theta)

        # CDP z-score（用於信號生成）
        cdp_zscore = self._rolling_zscore(cdp.abs(), min(120, len(df) // 2))

        logger.info(
            "CDP computed: X=[%.2f,%.2f] Y=[%.2f,%.2f] Z=[%.2f,%.2f] "
            "CUFD=%.4f (θ=%.4f) CDP=%.4f | Zone=%s",
            x.min(), x.max(), y.min(), y.max(), z.min(), z.max(),
            cufd.iloc[-1], theta, cdp.iloc[-1], zones.iloc[-1],
        )

        return CDPResult(
            x=x, y=y, z=z,
            cufd=cufd, cufd_side=cufd_side, theta=theta,
            cdp=cdp, cdp_zscore=cdp_zscore,
            zone=zones, pressure_level=levels,
        )

    # ─────────────────────────────────────────────────────────────────────
    # 提取特徵欄位（用於整合到 feature_builder_live）
    # ─────────────────────────────────────────────────────────────────────
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算並回傳可直接 concat 到主 feature DataFrame 的欄位。

        回傳欄位：
          cdp_x, cdp_y, cdp_z, cdp_cufd, cdp_cufd_side,
          cdp_score, cdp_zscore, cdp_zone_code, cdp_pressure_level
        """
        result = self.compute_all(df)

        out = pd.DataFrame(index=df.index)
        out["cdp_x"] = result.x
        out["cdp_y"] = result.y
        out["cdp_z"] = result.z
        out["cdp_cufd"] = result.cufd
        out["cdp_cufd_side"] = result.cufd_side
        out["cdp_score"] = result.cdp
        out["cdp_zscore"] = result.cdp_zscore
        out["cdp_pressure_level"] = result.pressure_level

        # Zone 編碼為數值（模型友善）
        zone_map = {
            "火力枯竭區": -2,
            "空方過度延伸危險區": -5,
            "空方壓境優勢區": -4,
            "空方輕度優勢區": -1,
            "楚河漢界": 0,
            "中性區": 0,
            "紅方輕度優勢區": 1,
            "紅方壓境優勢區": 4,
            "紅方過度延伸危險區": 5,
        }
        out["cdp_zone_code"] = result.zone.map(zone_map).fillna(0).astype(int)

        return out

    # ─────────────────────────────────────────────────────────────────────
    # 工具函式
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _rolling_zscore(s: pd.Series, win: int) -> pd.Series:
        mu = s.rolling(win, min_periods=max(4, win // 6)).mean()
        sd = s.rolling(win, min_periods=max(4, win // 6)).std().replace(0, np.nan)
        return (s - mu) / sd


# ═══════════════════════════════════════════════════════════════════════════
# 結果容器
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CDPResult:
    """CDP 計算結果的結構化容器。"""
    x: pd.Series                  # X 軸：收益區間 [-1, 1]
    y: pd.Series                  # Y 軸：主力資金流向 [-1, 1]
    z: pd.Series                  # Z 軸：巨鲸大單強度 [0, 1]
    cufd: pd.Series               # CUFD：因果累積壓力（正值）
    cufd_side: pd.Series          # CUFD 壓力方向：+1 多方承壓, -1 空方承壓
    theta: float                  # CUFD 歷史痛點閾值
    cdp: pd.Series                # CDP 最終濃縮特徵（帶方向）
    cdp_zscore: pd.Series         # CDP 的 rolling z-score
    zone: pd.Series               # 當前區域名稱
    pressure_level: pd.Series     # 壓力等級 0~5

    def summary(self, idx: int = -1) -> dict:
        """回傳指定 bar 的摘要 dict。"""
        return {
            "x": round(float(self.x.iloc[idx]), 3),
            "y": round(float(self.y.iloc[idx]), 3),
            "z": round(float(self.z.iloc[idx]), 3),
            "cufd": round(float(self.cufd.iloc[idx]), 6),
            "cufd_side": int(self.cufd_side.iloc[idx]),
            "theta": round(self.theta, 6),
            "cdp": round(float(self.cdp.iloc[idx]), 4),
            "cdp_zscore": round(float(self.cdp_zscore.iloc[idx]), 2),
            "zone": str(self.zone.iloc[idx]),
            "pressure_level": int(self.pressure_level.iloc[idx]),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """合併所有 series 為 DataFrame。"""
        return pd.concat([
            self.x, self.y, self.z,
            self.cufd, self.cufd_side,
            self.cdp, self.cdp_zscore,
            self.zone, self.pressure_level,
        ], axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# 信號生成器
# ═══════════════════════════════════════════════════════════════════════════

class CDPSignalGenerator:
    """
    根據 CDP 四維條件產生方向信號。

    信號規則：
    1. 強烈反向警報（overextend_reversal）：
       - CDP > 95th 分位 + 過度延伸危險區 → 預期反轉
       - 這是「淝水之戰」信號

    2. 壓境順勢信號（dominance_continuation）：
       - 壓境優勢區 + Z > 0.5 + CUFD < θ → 強勢延續
       - 重炮持續但未過載

    3. 火力枯竭反轉（exhaustion_fade）：
       - Z < 0.15 + 前期 Y 偏離 > 0.5 → 可能反轉
    """

    def __init__(self, config: Optional[CDPConfig] = None,
                 reversal_quantile: float = 0.95,
                 continuation_min_z: float = 0.50):
        self.cfg = config or CDPConfig()
        self.reversal_q = reversal_quantile
        self.cont_min_z = continuation_min_z

    def generate_signals(self, result: CDPResult,
                         lookback: int = 720) -> pd.DataFrame:
        """
        根據 CDPResult 生成信號 DataFrame。

        Parameters:
            result: CDPResult from ChessDominationPressure.compute_all()
            lookback: 計算分位數的回看窗口

        Returns:
            DataFrame with columns:
              signal: 1(看多) / -1(看空) / 0(中性)
              signal_type: 信號類型名稱
              signal_strength: 信號強度 0~1
        """
        n = len(result.cdp)
        signals = pd.Series(0, index=result.cdp.index, name="cdp_signal")
        sig_types = pd.Series("none", index=result.cdp.index, name="cdp_signal_type")
        sig_strength = pd.Series(0.0, index=result.cdp.index, name="cdp_signal_strength")

        cdp_abs = result.cdp.abs()
        zone = result.zone
        z = result.z
        y = result.y
        cufd = result.cufd

        for i in range(max(lookback, 48), n):
            # CDP 歷史分位
            lb = max(0, i - lookback)
            cdp_q = cdp_abs.iloc[lb:i].quantile(self.reversal_q)

            zi = z.iloc[i]
            yi = y.iloc[i]
            ci = cufd.iloc[i]
            zone_i = zone.iloc[i]
            cdp_i = result.cdp.iloc[i]

            # ── 信號 1：過度延伸反轉 ──
            if "過度延伸危險區" in zone_i and cdp_abs.iloc[i] > cdp_q:
                # CDP 方向 > 0 = 多方過載 → 預期下跌（signal = -1）
                # CDP 方向 < 0 = 空方過載 → 預期上漲（signal = +1）
                signals.iloc[i] = -int(np.sign(cdp_i))
                sig_types.iloc[i] = "overextend_reversal"
                sig_strength.iloc[i] = min(1.0, cdp_abs.iloc[i] / max(cdp_q, 1e-10))

            # ── 信號 2：壓境順勢 ──
            elif "壓境優勢區" in zone_i and zi >= self.cont_min_z and ci < result.theta:
                signals.iloc[i] = int(np.sign(yi))
                sig_types.iloc[i] = "dominance_continuation"
                sig_strength.iloc[i] = min(1.0, abs(yi) * zi)

            # ── 信號 3：火力枯竭 ──
            elif zi < self.cfg.z_exhaustion:
                # 看前 4 根 bar 的平均 Y，如果之前有偏離現在枯竭了
                y_prev = y.iloc[max(0, i - 4):i].mean()
                if abs(y_prev) > 0.5:
                    signals.iloc[i] = -int(np.sign(y_prev))
                    sig_types.iloc[i] = "exhaustion_fade"
                    sig_strength.iloc[i] = min(1.0, abs(y_prev) * 0.5)

        return pd.concat([signals, sig_types, sig_strength], axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# 便捷函式（整合到 feature_builder_live 的接口）
# ═══════════════════════════════════════════════════════════════════════════

def compute_cdp_features(df: pd.DataFrame,
                         config: Optional[CDPConfig] = None) -> pd.DataFrame:
    """
    一行呼叫：計算全部 CDP 特徵，回傳可直接 concat 的 DataFrame。

    用法（在 feature_builder_live.py 中）：
        from research.features.chess_domination import compute_cdp_features
        cdp_feats = compute_cdp_features(df)
        df = pd.concat([df, cdp_feats], axis=1)
    """
    engine = ChessDominationPressure(config)
    return engine.extract_features(df)
