"""
Order Flow Toxicity 特徵模組（完整加強版）
============================================

核心概念：
  「毒性訂單流」(Toxic Order Flow) 源自做市商理論。
  當知情交易者（informed traders）大量進場時，做市商持續被逆向選擇
  （adverse selection）——他們掛的限價單不斷被知情方吃掉。

  VPIN (Volume-Synchronized Probability of Informed Trading) 量化的就是
  這種「知情方主導程度」。VPIN 高 = 市場上知情交易者在大量單邊操作，
  做市商虧損風險飆升 → 流動性即將撤出 → 大波動即將到來。

  在 crypto 永續合約市場，毒性訂單流通常預示：
  1. 清算級別大波動（VPIN 高 + OI 高 = 定時炸彈）
  2. 流動性真空（做市商撤單 + depth 失衡 = 閃崩/閃漲）
  3. Funding 正回饋循環（毒性累積 + 極端 funding = 強制平倉連鎖）

輸出特徵：
  tox_bv_vpin           — Bulk Volume VPIN（核心毒性強度）
  tox_bv_vpin_zscore    — VPIN 的 z-score 標準化
  tox_accum             — 毒性累積壓力（帶指數衰減）
  tox_accum_zscore      — 累積壓力 z-score
  tox_div_taker         — VPIN 與 taker delta 背離度
  tox_liq_exhaust       — 毒性 × 流動性枯竭組合
  tox_funding_pressure  — 毒性 × funding × OI 正回饋壓力
  tox_pressure          — 最終綜合毒性壓力（多因子融合）
  tox_pressure_zscore   — 綜合壓力 z-score

必要輸入欄位：
  close, volume, taker_buy_vol (Binance klines)

可選增強欄位（有則更精確，無則自動 fallback）：
  cg_taker_buy, cg_taker_sell     — Coinglass 跨交易所 taker
  cg_funding_close                — Funding rate
  cg_oi_close                     — Open Interest
  depth_imbalance                 — Order book depth 失衡
  vol_regime                      — 波動率 regime

使用方式：
  from research.features.order_flow_toxicity import OrderFlowToxicity
  tox = OrderFlowToxicity()
  df = tox.transform(df)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class OrderFlowToxicity:
    """Order Flow Toxicity 完整特徵生成器（H1 頻率）"""

    def __init__(
        self,
        # VPIN 參數
        vpin_window: int = 40,        # VPIN 滾動窗口（40h ≈ 不到 2 天）
        sigma_window: int = 100,      # z-score 標準化回看窗口
        # 毒性累積參數
        accum_decay: float = 0.06,    # 每小時衰減率（≈ 半衰期 11h）
        # 閾值（None = 自適應 P75，推薦）
        toxicity_threshold: float | None = None,
        # 通用
        min_periods: int = 10,
    ):
        self.vpin_window = vpin_window
        self.sigma_window = sigma_window
        self.accum_decay = accum_decay
        self.toxicity_threshold = toxicity_threshold  # None = 自適應 P75
        self.min_periods = min_periods

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算所有毒性特徵並新增到 df。不修改原始欄位。

        Parameters:
            df: 含有 OHLCV + 可選 Coinglass 欄位的 1h DataFrame

        Returns:
            新增 tox_* 欄位後的 DataFrame
        """
        out = df.copy()

        # ── 1. BV-VPIN：Bulk Volume VPIN ──────────────────────────────────
        # 核心毒性指標。使用 Bulk Volume Classification (BVC) 分類每根 bar
        # 的買賣量，再計算 volume-normalized 的知情交易佔比。
        #
        # BVC 原理：用 close 相對 bar 中點的位置估算買/賣佔比
        #   buy_frac = Φ((close - mid) / σ)  （正態 CDF）
        #   其中 mid = (high + low) / 2, σ = rolling realized vol
        #
        # VPIN = mean(|V_buy - V_sell|) / mean(V_total)
        #   高 VPIN = 單方向知情交易者主導，做市商被逆向選擇
        out["tox_bv_vpin"] = self._compute_bv_vpin(out)
        out["tox_bv_vpin_zscore"] = self._zscore(
            out["tox_bv_vpin"], self.sigma_window
        )

        # ── 自適應閾值：用 VPIN 歷史 P75 作為毒性門檻 ─────────────────
        if self.toxicity_threshold is None:
            self._adaptive_threshold = float(
                out["tox_bv_vpin"].expanding(min_periods=100).quantile(0.75).iloc[-1]
            )
        else:
            self._adaptive_threshold = self.toxicity_threshold

        # ── 2. Toxic Accum：毒性累積壓力 ─────────────────────────────────
        # 類似 CUFD，但累積的是「VPIN 超過閾值的部分」。
        # 每小時指數衰減，讓近期高毒性權重更大。
        #
        # 意義：持續高毒性 = 知情交易者連續多小時單邊操作
        #       → 做市商庫存極度不平衡 → 大規模倉位調整即將發生
        out["tox_accum"] = self._compute_toxic_accum(out["tox_bv_vpin"])
        out["tox_accum_zscore"] = self._zscore(
            out["tox_accum"], self.sigma_window
        )

        # ── 3. Tox_Div_Taker：毒性-Taker 背離 ───────────────────────────
        # VPIN 高但 taker_delta 低 = 知情交易者用限價單（而非市價單）操作
        # 這種「靜默毒性」更危險：表面平靜但暗流湧動
        #
        # VPIN 高且 taker delta 方向一致 = 正常趨勢
        # VPIN 高但 taker delta 很小 = 隱蔽型知情交易（背離 → 更危險）
        out["tox_div_taker"] = self._compute_tox_div_taker(out)

        # ── 4. Tox_Liq_Exhaust：毒性 × 流動性枯竭 ────────────────────────
        # VPIN 高 + depth_imbalance 極端 = 做市商已經開始撤單
        # 這是流動性真空的前兆：毒性導致做市商退出 → 剩餘流動性薄
        # → 下一波市價單會造成極端滑點
        out["tox_liq_exhaust"] = self._compute_tox_liq_exhaust(out)

        # ── 5. Tox_Funding_Pressure：毒性 × Funding × OI ─────────────────
        # 永續合約特有的正回饋循環：
        #   高毒性 + 極端 funding + OI 增長
        #   = 知情交易者在建倉 + 逆勢方被迫付出高 funding
        #   + 沒人在平倉（OI 還在漲）
        #   = 壓力鍋效應，一旦觸發清算就是連環爆破
        out["tox_funding_pressure"] = self._compute_tox_funding_pressure(out)

        # ── 6. Toxicity_Pressure：最終綜合毒性壓力 ────────────────────────
        # 多因子非線性融合：
        #   基礎 = VPIN zscore（毒性強度）
        #   × (1 + accum_zscore)（持續性放大）
        #   × (1 + |funding_pressure|)（正回饋放大）
        #   × (1 + liq_exhaust)（流動性放大）
        #
        # 帶方向：用 taker_delta 的符號決定壓力方向
        #   正 = 多方毒性壓力（可能反轉向下）
        #   負 = 空方毒性壓力（可能反轉向上）
        out["tox_pressure"] = self._compute_tox_pressure(out)
        out["tox_pressure_zscore"] = self._zscore(
            out["tox_pressure"].abs(), self.sigma_window
        )

        # ── 安全清理 ─────────────────────────────────────────────────────
        tox_cols = [c for c in out.columns if c.startswith("tox_")]
        for c in tox_cols:
            out[c] = (
                out[c]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .astype(float)
            )

        return out

    # ═══════════════════════════════════════════════════════════════════════
    # 內部計算方法
    # ═══════════════════════════════════════════════════════════════════════

    def _compute_bv_vpin(self, df: pd.DataFrame) -> pd.Series:
        """
        Bulk Volume VPIN。

        使用 Bulk Volume Classification 將每根 bar 的成交量分為
        知情買方和知情賣方，再計算知情交易佔比。

        BVC 公式：
          z = (close - midpoint) / σ_price
          buy_frac = Φ(z)  （標準正態 CDF）
          V_buy = volume × buy_frac
          V_sell = volume × (1 - buy_frac)
          order_imbalance = |V_buy - V_sell|

        VPIN = rolling_mean(order_imbalance) / rolling_mean(volume)
        """
        close = df["close"]
        volume = df["volume"].replace(0, np.nan)

        # Bar 中點 & 價格波動率
        midpoint = (df["high"] + df["low"]) / 2
        price_range = (df["high"] - df["low"]).replace(0, np.nan)

        # 使用 realized vol 作為 σ（更穩定）
        log_ret = np.log(close / close.shift(1))
        sigma = log_ret.rolling(20, min_periods=4).std().replace(0, np.nan)

        # BVC: z-score of close vs midpoint
        z = (close - midpoint) / (sigma * close).replace(0, np.nan)

        # Φ(z) — 用 tanh 近似（計算快，對極端值更穩健）
        # tanh(z * 0.8) * 0.5 + 0.5 ≈ Φ(z) for moderate z
        buy_frac = np.tanh(z * 0.8) * 0.5 + 0.5
        buy_frac = buy_frac.clip(0.01, 0.99)

        # 如果有 taker 數據，用 BVC 和 taker 加權平均（更精確）
        if "taker_buy_vol" in df.columns:
            taker_buy = df["taker_buy_vol"]
            taker_sell = df["volume"] - taker_buy
            taker_frac = taker_buy / volume
            taker_frac = taker_frac.clip(0.01, 0.99)
            # 加權：70% taker 真實數據 + 30% BVC 估算
            buy_frac = 0.7 * taker_frac + 0.3 * buy_frac

        # Order imbalance
        v_buy = volume * buy_frac
        v_sell = volume * (1 - buy_frac)
        order_imbalance = (v_buy - v_sell).abs()

        # VPIN = mean(|imbalance|) / mean(volume)
        w = self.vpin_window
        mp = self.min_periods
        vpin = (
            order_imbalance.rolling(w, min_periods=mp).mean()
            / volume.rolling(w, min_periods=mp).mean().replace(0, np.nan)
        )

        return vpin.clip(0, 1)

    def _compute_toxic_accum(self, vpin: pd.Series) -> pd.Series:
        """
        毒性累積壓力（帶指數衰減）。

        只累積 VPIN > threshold 的超額部分。
        每小時衰減 decay_rate，讓近期毒性權重更高。

        高 toxic_accum = 過去多小時持續高毒性，壓力未釋放。
        """
        threshold = self._adaptive_threshold
        decay = 1.0 - self.accum_decay

        vals = vpin.fillna(0).values
        accum = np.zeros(len(vals))

        for i in range(1, len(vals)):
            # 衰減前一期
            prev = accum[i - 1] * decay
            # 只累積超過閾值的部分
            excess = max(vals[i] - threshold, 0)
            accum[i] = prev + excess

        return pd.Series(accum, index=vpin.index)

    def _compute_tox_div_taker(self, df: pd.DataFrame) -> pd.Series:
        """
        毒性-Taker 背離度。

        VPIN 高但 |taker_delta| 低 = 隱蔽型知情交易（限價單為主）
        VPIN 低但 |taker_delta| 高 = 普通趨勢（市價單為主，非知情）

        divergence = vpin_zscore - |taker_delta|_zscore
        正值 = 靜默毒性（更危險）
        負值 = 正常趨勢動量
        """
        vpin_z = df["tox_bv_vpin_zscore"]

        # Taker delta magnitude (z-score)
        if "taker_delta_ratio" in df.columns:
            td_abs = df["taker_delta_ratio"].abs()
        elif "taker_buy_vol" in df.columns:
            taker_sell = df["volume"] - df["taker_buy_vol"]
            td = df["taker_buy_vol"] - taker_sell
            td_abs = (td / df["volume"].replace(0, np.nan)).abs()
        else:
            return pd.Series(0.0, index=df.index)

        td_z = self._zscore(td_abs, self.sigma_window)

        return (vpin_z - td_z).fillna(0)

    def _compute_tox_liq_exhaust(self, df: pd.DataFrame) -> pd.Series:
        """
        毒性 × 流動性枯竭。

        VPIN 高 + depth 極度失衡 = 做市商正在撤離
        用 |depth_imbalance| 作為流動性枯竭的代理。
        無 depth 數據時用 price_impact 的 z-score 替代。
        """
        vpin = df["tox_bv_vpin"]

        # 流動性枯竭指標
        if "depth_imbalance" in df.columns and df["depth_imbalance"].notna().sum() > 10:
            liq_stress = df["depth_imbalance"].abs()
        elif "price_impact_zscore" in df.columns:
            # price_impact 高 = 流動性薄（每單位 volume 的價格影響大）
            liq_stress = df["price_impact_zscore"].abs().clip(0, 3) / 3
        elif "fragility_zscore" in df.columns:
            liq_stress = df["fragility_zscore"].abs().clip(0, 3) / 3
        else:
            return pd.Series(0.0, index=df.index)

        # 乘積：VPIN × 流動性枯竭
        return (vpin * liq_stress).fillna(0)

    def _compute_tox_funding_pressure(self, df: pd.DataFrame) -> pd.Series:
        """
        毒性 × Funding × OI 正回饋壓力。

        永續合約的定時炸彈公式：
          高 VPIN（知情方在操作）
          × 極端 funding（逆勢方在付出高成本）
          × OI 增長（沒人在平倉，壓力在累積）

        帶方向：funding 正 = 多方承壓，funding 負 = 空方承壓
        """
        vpin = df["tox_bv_vpin"]

        # Funding 壓力（z-score，保留方向）
        if "cg_funding_close_zscore" in df.columns:
            funding_z = df["cg_funding_close_zscore"]
        elif "cg_funding_close" in df.columns:
            funding_z = self._zscore(df["cg_funding_close"], 24)
        else:
            return pd.Series(0.0, index=df.index)

        # OI 變化（正 = 新開倉增加 = 壓力累積）
        if "cg_oi_close" in df.columns:
            oi_growth = df["cg_oi_close"].pct_change(4).clip(-0.1, 0.1)
            oi_factor = 1 + oi_growth.fillna(0) * 10  # 放大 OI 變化的影響
        else:
            oi_factor = 1.0

        # 正回饋壓力 = VPIN × funding_z × OI_factor
        # 帶方向（funding 的符號）
        return (vpin * funding_z * oi_factor).fillna(0)

    def _compute_tox_pressure(self, df: pd.DataFrame) -> pd.Series:
        """
        最終綜合毒性壓力（多因子非線性融合）。

        公式：
          base = vpin_zscore（毒性強度）
          amplified = base × (1 + 0.3 × accum_z) × (1 + 0.2 × |funding_p|) × (1 + 0.2 × liq_ex)

        帶方向：
          用 taker_delta 方向決定壓力方向
          正 = 買方毒性主導（價格可能被高估 → 看空壓力）
          負 = 賣方毒性主導（價格可能被低估 → 看多壓力）
        """
        vpin_z = df["tox_bv_vpin_zscore"]
        accum_z = df["tox_accum_zscore"]
        funding_p = df["tox_funding_pressure"]
        liq_ex = df["tox_liq_exhaust"]

        # 非線性融合
        base = vpin_z.abs()
        amplified = (
            base
            * (1 + 0.3 * accum_z.clip(0, 5))
            * (1 + 0.2 * funding_p.abs().clip(0, 5))
            * (1 + 0.2 * liq_ex.clip(0, 5))
        )

        # 方向：taker delta 的符號
        if "taker_delta_ratio" in df.columns:
            direction = np.sign(df["taker_delta_ratio"])
        elif "taker_buy_vol" in df.columns:
            taker_sell = df["volume"] - df["taker_buy_vol"]
            direction = np.sign(df["taker_buy_vol"] - taker_sell)
        else:
            direction = np.sign(vpin_z)

        return (amplified * direction).fillna(0)

    # ── 工具 ──────────────────────────────────────────────────────────────

    def _zscore(self, s: pd.Series, window: int) -> pd.Series:
        mp = max(4, window // 6)
        mu = s.rolling(window, min_periods=mp).mean()
        sd = s.rolling(window, min_periods=mp).std().replace(0, np.nan)
        return (s - mu) / sd

    def get_feature_names(self) -> list[str]:
        return [
            "tox_bv_vpin", "tox_bv_vpin_zscore",
            "tox_accum", "tox_accum_zscore",
            "tox_div_taker",
            "tox_liq_exhaust",
            "tox_funding_pressure",
            "tox_pressure", "tox_pressure_zscore",
        ]
