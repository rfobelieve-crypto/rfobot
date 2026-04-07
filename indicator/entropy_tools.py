"""
熵診斷與風險管理工具
====================

提供兩個獨立但可協作的工具：

1. EntropyAnalyzer — 診斷工具
   計算市場熵、特徵互資訊、預測熵，產生 insight 文字摘要。
   用途：每日/每週檢視模型健康度、特徵品質、市場狀態。

2. EntropyRiskManager — 風險管理工具
   根據熵值動態計算風險分數和建議倉位。
   用途：每小時 update_cycle 後呼叫，決定信號可信度權重。

使用方式：
    from indicator.entropy_tools import EntropyAnalyzer, EntropyRiskManager

    # 診斷（在研究/回顧時使用）
    analyzer = EntropyAnalyzer()
    report = analyzer.analyze(features_df, predictions_df)
    print(report["insight"])

    # 風險管理（在每次預測後使用）
    risk_mgr = EntropyRiskManager()
    risk = risk_mgr.assess(features_df, dir_prob_up=0.72, market_regime="CHOPPY")
    print(risk["risk_score"], risk["position_scale"])
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TZ_TPE = timezone(timedelta(hours=8))


# ═══════════════════════════════════════════════════════════════════════════
# EntropyAnalyzer — 診斷工具
# ═══════════════════════════════════════════════════════════════════════════

class EntropyAnalyzer:
    """
    熵診斷分析器

    三個核心指標：
      1. Market Entropy — 市場報酬分佈的不確定性（高=混沌，低=趨勢明確）
      2. Mutual Information — 每個特徵對未來收益的資訊量（高=有用，低=噪音）
      3. Prediction Entropy — 模型預測機率的熵（高=模型不確定，低=模型有信心）

    使用時機：
      - 每日收盤後跑一次，監控模型健康度
      - 模型重訓前，確認哪些特徵資訊量在衰退
      - 市場 regime 切換時，理解不確定性來源
    """

    def __init__(
        self,
        return_bins: int = 20,       # 報酬率分箱數（用於市場熵計算）
        mi_bins: int = 10,           # 互資訊分箱數
        lookback: int = 168,         # 預設回看窗口（168h = 7 天）
        min_periods: int = 50,       # 最少有效數據點
    ):
        self.return_bins = return_bins
        self.mi_bins = mi_bins
        self.lookback = lookback
        self.min_periods = min_periods

    def analyze(
        self,
        features_df: pd.DataFrame,
        predictions_df: pd.DataFrame | None = None,
        top_k_features: int = 20,
    ) -> dict:
        """
        完整診斷分析。

        Parameters:
            features_df: 含有 close 和所有特徵的 DataFrame
            predictions_df: 含有 dir_prob_up 的預測結果（可選）
            top_k_features: 互資訊排序前 K 名

        Returns:
            dict with keys: market_entropy, prediction_entropy,
                           mutual_info (sorted list), insight (text)
        """
        result = {}

        # ── 1. Market Entropy（市場報酬分佈的不確定性）────────────────
        result["market_entropy"] = self._market_entropy(features_df)

        # ── 2. Mutual Information（特徵 vs 未來收益）──────────────────
        result["mutual_info"] = self._mutual_information(
            features_df, top_k=top_k_features
        )

        # ── 3. Prediction Entropy（模型預測信心）─────────────────────
        if predictions_df is not None and "dir_prob_up" in predictions_df.columns:
            result["prediction_entropy"] = self._prediction_entropy(predictions_df)
        elif "dir_prob_up" in features_df.columns:
            result["prediction_entropy"] = self._prediction_entropy(features_df)
        else:
            result["prediction_entropy"] = {"mean": None, "current": None}

        # ── 4. Insight 文字總結 ──────────────────────────────────────
        result["insight"] = self._generate_insight(result)
        result["timestamp"] = datetime.now(TZ_TPE).strftime("%Y-%m-%d %H:%M UTC+8")

        return result

    def _market_entropy(self, df: pd.DataFrame) -> dict:
        """
        市場熵：衡量報酬率分佈的不確定性。

        使用過去 lookback 根 bar 的 log_return，分箱後計算 Shannon entropy。
        高熵 = 報酬率分散（混沌市場，方向不明）
        低熵 = 報酬率集中（趨勢市場，方向明確）

        也計算 rolling 熵來偵測熵的突變（regime shift 訊號）。
        """
        if "close" not in df.columns:
            return {"current": None, "rolling_mean": None, "zscore": None}

        returns = np.log(df["close"] / df["close"].shift(1)).dropna()
        recent = returns.tail(self.lookback)

        if len(recent) < self.min_periods:
            return {"current": None, "rolling_mean": None, "zscore": None}

        # Shannon entropy of return distribution
        current_h = self._shannon_entropy(recent.values, self.return_bins)

        # Rolling entropy (24h window, step 1)
        rolling_h = returns.rolling(24, min_periods=12).apply(
            lambda x: self._shannon_entropy(x.values, self.return_bins), raw=False
        )
        rolling_mean = rolling_h.tail(self.lookback).mean()
        rolling_std = rolling_h.tail(self.lookback).std()

        # Z-score: current vs historical
        zscore = (current_h - rolling_mean) / rolling_std if rolling_std > 0 else 0

        # 理論最大熵（均勻分佈）
        max_entropy = np.log2(self.return_bins)
        normalized = current_h / max_entropy if max_entropy > 0 else 0

        return {
            "current": round(float(current_h), 4),
            "normalized": round(float(normalized), 4),  # 0~1, 1=完全混沌
            "rolling_mean": round(float(rolling_mean), 4),
            "zscore": round(float(zscore), 2),
            "max_theoretical": round(float(max_entropy), 4),
        }

    def _mutual_information(self, df: pd.DataFrame, top_k: int = 20) -> list[dict]:
        """
        互資訊：衡量每個特徵攜帶多少關於未來收益的資訊。

        MI(X, Y) = H(Y) - H(Y|X)
        高 MI = 特徵對預測有價值
        低 MI = 特徵是噪音

        使用離散化（分箱）估算法。
        """
        if "close" not in df.columns:
            return []

        # Target: 4h forward return
        target = df["close"].shift(-4) / df["close"] - 1
        valid_mask = target.notna()

        # 排除非特徵欄位
        exclude = {"open", "high", "low", "close", "volume", "taker_buy_vol",
                    "taker_buy_quote", "trade_count", "ts_open", "close_time",
                    "quote_vol", "ignore"}
        feature_cols = [c for c in df.columns
                        if c not in exclude
                        and not c.startswith("y_")
                        and df[c].dtype in ("float64", "float32", "int64")
                        and df[c].notna().sum() > self.min_periods]

        results = []
        y = target[valid_mask].values
        y_binned = self._safe_digitize(y, self.mi_bins)

        for col in feature_cols:
            x = df.loc[valid_mask, col].values
            if np.isnan(x).sum() > len(x) * 0.5:
                continue
            x_clean = np.nan_to_num(x, nan=0)
            x_binned = self._safe_digitize(x_clean, self.mi_bins)

            mi = self._discrete_mi(x_binned, y_binned)
            results.append({"feature": col, "mi": round(float(mi), 6)})

        results.sort(key=lambda r: r["mi"], reverse=True)
        return results[:top_k]

    def _prediction_entropy(self, df: pd.DataFrame) -> dict:
        """
        預測熵：衡量模型對自己預測的信心。

        H(pred) = -p*log(p) - (1-p)*log(1-p)
        其中 p = dir_prob_up

        高熵 = 模型不確定（prob 接近 0.5）
        低熵 = 模型有信心（prob 接近 0 或 1）
        """
        prob = df["dir_prob_up"].dropna()
        if len(prob) < 5:
            return {"mean": None, "current": None}

        recent = prob.tail(self.lookback)
        h = self._binary_entropy(recent.values)

        return {
            "current": round(float(h.iloc[-1]) if len(h) > 0 else 0, 4),
            "mean": round(float(h.mean()), 4),
            "std": round(float(h.std()), 4),
            "high_uncertainty_pct": round(float((h > 0.9).mean() * 100), 1),
            # 最大二元熵 = 1.0 (at p=0.5)
        }

    def _generate_insight(self, result: dict) -> str:
        """產生中文 insight 摘要。"""
        lines = []
        now = datetime.now(TZ_TPE).strftime("%m/%d %H:%M")
        lines.append(f"🔍 熵診斷報告 | {now} UTC+8\n")

        # Market entropy
        me = result.get("market_entropy", {})
        if me.get("normalized") is not None:
            norm = me["normalized"]
            if norm > 0.85:
                level = "🔴 極高（混沌市場）"
            elif norm > 0.7:
                level = "🟡 偏高（方向不明）"
            elif norm > 0.5:
                level = "🟢 中等（正常波動）"
            else:
                level = "🔵 偏低（趨勢明確）"
            lines.append(f"市場熵: {me['current']:.3f} ({norm:.0%}) {level}")
            if me.get("zscore") and abs(me["zscore"]) > 1.5:
                lines.append(f"  ⚠️ 熵 z-score={me['zscore']:+.1f}，偏離歷史均值")

        # Prediction entropy
        pe = result.get("prediction_entropy", {})
        if pe.get("mean") is not None:
            if pe["mean"] > 0.9:
                conf = "🔴 模型很不確定（多數預測接近 50/50）"
            elif pe["mean"] > 0.7:
                conf = "🟡 模型信心偏低"
            elif pe["mean"] > 0.4:
                conf = "🟢 模型有中等信心"
            else:
                conf = "🔵 模型信心很高"
            lines.append(f"預測熵: {pe['mean']:.3f} {conf}")
            lines.append(f"  高不確定比例: {pe.get('high_uncertainty_pct', 0):.0f}%")

        # Top MI features
        mi = result.get("mutual_info", [])
        if mi:
            lines.append(f"\n資訊量前 5 特徵:")
            for i, m in enumerate(mi[:5]):
                lines.append(f"  {i+1}. {m['feature']:<35} MI={m['mi']:.5f}")

        return "\n".join(lines)

    # ── 工具方法 ──────────────────────────────────────────────────────

    @staticmethod
    def _shannon_entropy(arr: np.ndarray, bins: int) -> float:
        """Shannon entropy (bits) of a continuous array via histogram binning."""
        counts, _ = np.histogram(arr, bins=bins)
        counts = counts + 1e-10  # avoid log(0)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs))

    @staticmethod
    def _binary_entropy(probs: np.ndarray) -> pd.Series:
        """Binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p)."""
        p = np.clip(probs, 1e-10, 1 - 1e-10)
        h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        return pd.Series(h)

    @staticmethod
    def _safe_digitize(arr: np.ndarray, bins: int) -> np.ndarray:
        """Safely digitize array into bins, handling edge cases."""
        if arr.std() < 1e-10:
            return np.zeros(len(arr), dtype=int)
        edges = np.percentile(arr[~np.isnan(arr)],
                              np.linspace(0, 100, bins + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            return np.zeros(len(arr), dtype=int)
        return np.clip(np.digitize(arr, edges[1:-1]), 0, len(edges) - 2)

    @staticmethod
    def _discrete_mi(x: np.ndarray, y: np.ndarray) -> float:
        """Mutual information between two discrete arrays."""
        n = len(x)
        if n == 0:
            return 0.0

        # Joint and marginal counts
        xy_pairs = list(zip(x, y))
        from collections import Counter
        joint = Counter(xy_pairs)
        x_counts = Counter(x)
        y_counts = Counter(y)

        mi = 0.0
        for (xi, yi), count in joint.items():
            p_xy = count / n
            p_x = x_counts[xi] / n
            p_y = y_counts[yi] / n
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * np.log2(p_xy / (p_x * p_y))

        return max(mi, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# EntropyRiskManager — 風險管理工具
# ═══════════════════════════════════════════════════════════════════════════

class EntropyRiskManager:
    """
    熵風險管理器

    根據市場熵和預測熵動態調整風險水位。核心邏輯：

      高市場熵 + 高預測熵 = 高風險（市場混沌且模型沒信心）→ 降低倉位
      低市場熵 + 低預測熵 = 低風險（趨勢明確且模型有信心）→ 正常/加大倉位
      高市場熵 + 低預測熵 = 模型逆勢有信心 → 中等風險，值得觀察
      低市場熵 + 高預測熵 = 趨勢中但模型猶豫 → 中等風險

    使用時機：
      - 每小時 update_cycle 後呼叫
      - 在 signal generation 之後、推送之前，作為信號過濾器
      - 與 EntropyAnalyzer 協作取得更完整的風險圖像

    輸出：
      risk_score: 0~100（100=最高風險）
      position_scale: 0.0~1.5（建議倉位倍數）
      alert: 風險警報文字（如有）
    """

    def __init__(
        self,
        # 市場熵參數
        market_entropy_lookback: int = 168,  # 7 天
        market_entropy_bins: int = 20,
        # 風險閾值
        high_risk_threshold: float = 75,     # risk_score > 此值 = 高風險
        low_risk_threshold: float = 30,      # risk_score < 此值 = 低風險
        # 倉位調整
        max_position_scale: float = 1.5,     # 最大倉位倍數
        min_position_scale: float = 0.2,     # 最小倉位倍數
        # 熵突變偵測
        entropy_spike_zscore: float = 2.0,   # 熵 z-score > 此值 = 突變警報
    ):
        self.market_entropy_lookback = market_entropy_lookback
        self.market_entropy_bins = market_entropy_bins
        self.high_risk_threshold = high_risk_threshold
        self.low_risk_threshold = low_risk_threshold
        self.max_position_scale = max_position_scale
        self.min_position_scale = min_position_scale
        self.entropy_spike_zscore = entropy_spike_zscore

        self._analyzer = EntropyAnalyzer(
            return_bins=market_entropy_bins,
            lookback=market_entropy_lookback,
        )

    def assess(
        self,
        features_df: pd.DataFrame,
        dir_prob_up: float | None = None,
        confidence: float | None = None,
        market_regime: str = "",
    ) -> dict:
        """
        評估當前風險水位。

        Parameters:
            features_df: 含有 close 和特徵的 DataFrame
            dir_prob_up: 當前 Direction Model 的 P(UP)
            confidence: 當前信心分數 (0~100)
            market_regime: 當前 regime (CHOPPY/TRENDING_BULL/TRENDING_BEAR)

        Returns:
            dict with: risk_score, position_scale, alert, components
        """
        components = {}

        # ── 1. 市場熵 → 市場風險分量 ────────────────────────────────
        me = self._analyzer._market_entropy(features_df)
        market_risk = 0.0
        if me.get("normalized") is not None:
            # 正規化熵越高，風險越高
            market_risk = me["normalized"] * 100  # 0~100
            components["market_entropy"] = me["normalized"]
            components["market_entropy_zscore"] = me.get("zscore", 0)

        # ── 2. 預測熵 → 模型不確定性分量 ────────────────────────────
        pred_risk = 50.0  # 預設中等
        if dir_prob_up is not None:
            # 二元熵：p 接近 0.5 = 高熵 = 高不確定性
            p = np.clip(dir_prob_up, 1e-10, 1 - 1e-10)
            pred_entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
            pred_risk = pred_entropy * 100  # 0~100 (max at p=0.5)
            components["pred_entropy"] = round(float(pred_entropy), 4)
            components["dir_prob_up"] = dir_prob_up

        # ── 3. Regime 調整 ───────────────────────────────────────────
        regime_factor = 1.0
        if market_regime == "CHOPPY":
            regime_factor = 1.2  # 盤整時風險上調 20%
        elif market_regime in ("TRENDING_BULL", "TRENDING_BEAR"):
            regime_factor = 0.8  # 趨勢時風險下調 20%
        elif market_regime == "WARMUP":
            regime_factor = 1.5  # 暖機期風險大幅上調
        components["regime_factor"] = regime_factor

        # ── 4. 信心調整 ─────────────────────────────────────────────
        conf_factor = 1.0
        if confidence is not None:
            if confidence >= 80:
                conf_factor = 0.8  # 高信心降低風險
            elif confidence < 50:
                conf_factor = 1.3  # 低信心提高風險
        components["conf_factor"] = conf_factor

        # ── 5. 綜合風險分數 ─────────────────────────────────────────
        # 加權：市場熵 40% + 預測熵 40% + regime/conf 修正 20%
        raw_risk = (market_risk * 0.4 + pred_risk * 0.4) * regime_factor * conf_factor
        risk_score = np.clip(raw_risk, 0, 100)

        # ── 6. 倉位建議 ─────────────────────────────────────────────
        # 線性映射：risk 0→max_scale, risk 100→min_scale
        position_scale = (
            self.max_position_scale
            - (self.max_position_scale - self.min_position_scale) * risk_score / 100
        )
        position_scale = round(np.clip(position_scale, self.min_position_scale,
                                        self.max_position_scale), 2)

        # ── 7. 風險警報 ─────────────────────────────────────────────
        alerts = []
        if risk_score > self.high_risk_threshold:
            alerts.append(f"🔴 高風險 ({risk_score:.0f}/100) — 建議降低倉位至 {position_scale:.0%}")
        if me.get("zscore") and abs(me["zscore"]) > self.entropy_spike_zscore:
            alerts.append(f"⚡ 市場熵突變 (z={me['zscore']:+.1f}) — regime shift 可能發生")
        if market_regime == "WARMUP":
            alerts.append("⏳ WARMUP 期間 — 預測不可靠，最低倉位")

        return {
            "risk_score": round(float(risk_score), 1),
            "position_scale": position_scale,
            "risk_level": (
                "HIGH" if risk_score > self.high_risk_threshold
                else "LOW" if risk_score < self.low_risk_threshold
                else "MEDIUM"
            ),
            "alerts": alerts,
            "components": components,
            "timestamp": datetime.now(TZ_TPE).strftime("%Y-%m-%d %H:%M UTC+8"),
        }

    def format_for_telegram(self, assessment: dict) -> str:
        """格式化風險評估為 Telegram 顯示文字。"""
        r = assessment
        level_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[r["risk_level"]]

        lines = [
            f"{level_icon} <b>風險評估</b> | {r['timestamp']}",
            f"  風險分數: {r['risk_score']:.0f}/100 ({r['risk_level']})",
            f"  建議倉位: {r['position_scale']:.0%}",
        ]

        c = r.get("components", {})
        if "market_entropy" in c:
            lines.append(f"  市場熵: {c['market_entropy']:.2f}")
        if "pred_entropy" in c:
            lines.append(f"  預測熵: {c['pred_entropy']:.3f}")

        for alert in r.get("alerts", []):
            lines.append(f"\n{alert}")

        return "\n".join(lines)
