"""
Strong signal SHAP explainer — TreeSHAP for XGBoost direction model.

Only runs when a Strong signal fires (rare, <10ms per call).
Records top contributing features to understand WHY the signal triggered.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(__file__).parent / "model_artifacts"
DUAL_DIR = ARTIFACT_DIR / "dual_model"

# Feature → group mapping for order flow context
FEATURE_GROUPS = {
    # Large trade flow
    "large_delta": "大單流", "small_delta": "大單流", "large_delta_zscore": "大單流",
    "large_buy_ratio": "大單流", "large_buy_bias": "大單流",
    "large_delta_ma_4": "大單流", "large_delta_slope_4": "大單流",
    "large_delta_persistence": "大單流", "large_taker_signal": "大單流",
    "agg_large_delta": "大單流", "agg_small_delta": "大單流",
    "agg_large_ratio": "大單流", "agg_large_buy_ratio": "大單流",
    # Imbalance
    "imb_1b": "失衡持續性", "imb_3b": "失衡持續性", "imb_5b": "失衡持續性",
    "imb_8b": "失衡持續性", "imb_std_5b": "失衡持續性",
    "imb_sign_persistence": "失衡持續性", "imb_slope_4b": "失衡持續性",
    "imb_3b_zscore": "失衡持續性",
    # Absorption
    "absorption_score": "吸收代理", "absorption_zscore": "吸收代理",
    "absorption_ma_4h": "吸收代理", "absorption_cumsum_4h": "吸收代理",
    "absorption_buy": "吸收代理", "absorption_sell": "吸收代理",
    "absorption_net": "吸收代理", "impact_asymmetry": "吸收代理",
    # Momentum
    "ret_1b": "短期動量", "ret_2b": "短期動量", "ret_3b": "短期動量",
    "ret_5b": "短期動量", "wick_asymmetry": "短期動量",
    "body_ratio": "短期動量", "signed_body_ratio": "短期動量",
    "reversal_1v3": "短期動量", "log_return": "短期動量",
    "return_lag_1": "短期動量", "return_lag_2": "短期動量",
    # Sentiment / contrarian
    "cg_funding_close": "情緒/反向", "cg_funding_range": "情緒/反向",
    "cg_funding_close_zscore": "情緒/反向", "funding_zscore": "情緒/反向",
    "cg_ls_ratio": "情緒/反向", "cg_ls_ratio_zscore": "情緒/反向",
    "cg_gls_ratio": "情緒/反向", "cg_crowding": "情緒/反向",
    "cg_ls_divergence": "情緒/反向", "cg_ls_divergence_zscore": "情緒/反向",
    "crowding_zscore": "情緒/反向",
    # OI
    "cg_oi_close": "OI", "cg_oi_delta": "OI", "cg_oi_accel": "OI",
    "cg_oi_delta_zscore": "OI", "cg_oi_close_zscore": "OI",
    "cg_oi_binance_share": "OI", "oi_price_divergence": "OI",
    # Liquidation
    "cg_liq_total": "清算", "cg_liq_long": "清算", "cg_liq_short": "清算",
    "cg_liq_imbalance": "清算", "cg_liq_surge": "清算",
    "cg_liq_x_oi": "清算", "cg_liq_cascade": "清算",
    # Taker
    "cg_taker_delta": "Taker", "cg_taker_ratio": "Taker",
    "cg_taker_buy": "Taker", "cg_taker_sell": "Taker",
    "taker_delta_ratio": "Taker", "taker_delta_ma_24h": "Taker",
    # Volatility
    "realized_vol_20b": "波動率", "vol_regime": "波動率",
    "vol_acceleration": "波動率", "vol_kurtosis": "波動率",
    "vol_entropy": "波動率", "squeeze_proxy": "波動率",
    "bvol_close": "波動率", "bvol_zscore": "波動率",
    # Premium / margin
    "cg_cb_premium": "溢價/保證金", "cg_cb_premium_rate": "溢價/保證金",
    "cg_bfx_margin_delta": "溢價/保證金", "cg_bfx_margin_ratio": "溢價/保證金",
    # Cross features
    "cg_conviction": "交叉特徵", "funding_taker_align": "交叉特徵",
    "cg_spot_futures_cvd_divergence": "交叉特徵",
    "cg_pos_account_divergence": "交叉特徵",
    # Time
    "hour_sin": "時間", "hour_cos": "時間",
    "weekday_sin": "時間", "weekday_cos": "時間",
}


def _get_group(feature_name: str) -> str:
    """Map feature name to group. Fallback to prefix-based matching."""
    if feature_name in FEATURE_GROUPS:
        return FEATURE_GROUPS[feature_name]
    # Prefix matching
    if feature_name.startswith("cg_oi"):
        return "OI"
    if feature_name.startswith("cg_liq"):
        return "清算"
    if feature_name.startswith("cg_ls") or feature_name.startswith("cg_gls"):
        return "情緒/反向"
    if feature_name.startswith("cg_taker"):
        return "Taker"
    if feature_name.startswith("cg_funding"):
        return "情緒/反向"
    if feature_name.startswith("cg_bfx") or feature_name.startswith("cg_cb"):
        return "溢價/保證金"
    if feature_name.startswith("bvol"):
        return "波動率"
    if feature_name.startswith("return_lag"):
        return "短期動量"
    if feature_name.startswith("agg_"):
        return "大單流"
    return "其他"


class SignalExplainer:
    """SHAP explainer for Strong direction signals. Lazy-initialized."""

    def __init__(self):
        self._explainer = None
        self._feature_cols = None

    def _init_explainer(self):
        """Lazy init: load model + create TreeExplainer on first use."""
        if self._explainer is not None:
            return True
        try:
            import shap
            import xgboost as xgb

            model_path = DUAL_DIR / "direction_xgb.json"
            fc_path = DUAL_DIR / "direction_feature_cols.json"

            if not model_path.exists():
                logger.warning("Direction model not found, SHAP disabled")
                return False

            model = xgb.XGBClassifier()
            model.load_model(str(model_path))

            with open(fc_path) as f:
                self._feature_cols = json.load(f)

            # TreeExplainer: no background data needed, exact computation
            self._explainer = shap.TreeExplainer(model)
            logger.info("SHAP TreeExplainer initialized (%d features)", len(self._feature_cols))
            return True
        except ImportError:
            logger.warning("shap not installed — SHAP analysis disabled")
            return False
        except Exception as e:
            logger.warning("SHAP init failed: %s", e)
            return False

    def explain_signal(self, features_row: dict, direction: str) -> dict | None:
        """
        Compute SHAP explanation for a single Strong signal.

        Parameters:
            features_row: dict of {feature_name: value} for the signal bar
            direction: "UP" or "DOWN"

        Returns:
            dict with top_positive, top_negative, summary, or None on failure
        """
        if not self._init_explainer():
            return None

        try:
            import pandas as pd

            # Build single-row DataFrame with correct feature order
            X = pd.DataFrame([{
                col: float(features_row.get(col, 0) or 0)
                for col in self._feature_cols
            }])

            # TreeSHAP: returns array of shape (1, n_features) for binary classifier
            shap_values = self._explainer.shap_values(X)

            # For binary classifier, shap_values may be a list [class0, class1]
            if isinstance(shap_values, list):
                # Use class 1 (UP) values
                sv = shap_values[1][0] if direction == "UP" else shap_values[0][0]
            elif shap_values.ndim == 3:
                sv = shap_values[0, :, 1] if direction == "UP" else shap_values[0, :, 0]
            else:
                sv = shap_values[0]

            # Pair feature names with SHAP values
            pairs = list(zip(self._feature_cols, sv))

            # Sort by SHAP value
            pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)

            # Top 5 positive contributors
            top_pos = []
            total_pos_shap = sum(v for _, v in pairs if v > 0) or 1e-10
            for name, val in pairs_sorted[:5]:
                if val <= 0:
                    break
                top_pos.append({
                    "feature": name,
                    "shap": round(float(val), 4),
                    "pct": round(float(val / total_pos_shap * 100), 1),
                    "group": _get_group(name),
                    "value": round(float(features_row.get(name, 0) or 0), 6),
                })

            # Top 3 negative contributors
            top_neg = []
            for name, val in reversed(pairs_sorted):
                if val >= 0 or len(top_neg) >= 3:
                    break
                top_neg.append({
                    "feature": name,
                    "shap": round(float(val), 4),
                    "group": _get_group(name),
                    "value": round(float(features_row.get(name, 0) or 0), 6),
                })

            # Group-level summary
            group_contrib = {}
            for name, val in pairs:
                g = _get_group(name)
                group_contrib[g] = group_contrib.get(g, 0) + val
            group_sorted = sorted(group_contrib.items(), key=lambda x: abs(x[1]), reverse=True)
            top_groups = [{"group": g, "shap": round(float(v), 4)} for g, v in group_sorted[:5]]

            result = {
                "top_positive": top_pos,
                "top_negative": top_neg,
                "top_groups": top_groups,
                "base_value": round(float(self._explainer.expected_value
                                         if not isinstance(self._explainer.expected_value, list)
                                         else self._explainer.expected_value[1]), 4),
            }
            logger.info("SHAP explanation: top driver = %s (%s)",
                        top_pos[0]["feature"] if top_pos else "?",
                        top_pos[0]["group"] if top_pos else "?")
            return result

        except Exception as e:
            logger.warning("SHAP computation failed: %s", e)
            return None


def format_shap_for_telegram(shap_result: dict, direction: str) -> str:
    """Format SHAP explanation as Telegram-friendly text."""
    if not shap_result:
        return ""

    lines = [f"\n<b>驅動因子分析</b>"]

    # Top positive
    top_pos = shap_result.get("top_positive", [])
    if top_pos:
        arrow = "🟢" if direction == "UP" else "🔴"
        for i, item in enumerate(top_pos[:3]):
            lines.append(
                f"  {arrow} {item['feature']} ({item['group']}) "
                f"+{item['shap']:.3f} ({item['pct']:.0f}%)"
            )

    # Top negative (opposing)
    top_neg = shap_result.get("top_negative", [])
    if top_neg:
        lines.append("  反向拉力:")
        for item in top_neg[:2]:
            lines.append(
                f"  ⚪ {item['feature']} ({item['group']}) {item['shap']:.3f}"
            )

    # Group summary
    top_groups = shap_result.get("top_groups", [])
    if top_groups:
        lines.append("  群組貢獻:")
        for g in top_groups[:3]:
            sign = "+" if g["shap"] > 0 else ""
            lines.append(f"    {g['group']}: {sign}{g['shap']:.3f}")

    return "\n".join(lines)


def get_shap_stats_report() -> str:
    """Generate cumulative SHAP stats from stored strong_signals data."""
    try:
        from indicator.signal_tracker import _get_db_conn, _ensure_table
        _ensure_table()
        conn = _get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT shap_top FROM strong_signals
                WHERE shap_top IS NOT NULL AND shap_top != ''
            """)
            rows = cur.fetchall()
        conn.close()

        if not rows:
            return ""

        # Aggregate top features across all Strong signals
        feature_counts = {}  # feature → count of times in top 5
        feature_total_shap = {}  # feature → cumulative SHAP
        group_total_shap = {}

        for row in rows:
            try:
                data = json.loads(row["shap_top"])
            except (json.JSONDecodeError, TypeError):
                continue
            for item in data.get("top_positive", []):
                f = item["feature"]
                feature_counts[f] = feature_counts.get(f, 0) + 1
                feature_total_shap[f] = feature_total_shap.get(f, 0) + item["shap"]
            for g in data.get("top_groups", []):
                gn = g["group"]
                group_total_shap[gn] = group_total_shap.get(gn, 0) + g["shap"]

        if not feature_counts:
            return ""

        # Top 5 most frequent drivers
        top_freq = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        lines = [f"\n<b>累積驅動因子 ({len(rows)} 筆)</b>"]
        for feat, cnt in top_freq:
            avg_shap = feature_total_shap.get(feat, 0) / cnt
            group = _get_group(feat)
            lines.append(f"  {feat} ({group}): {cnt}次 avg+{avg_shap:.3f}")

        return "\n".join(lines)
    except Exception as e:
        logger.warning("SHAP stats failed: %s", e)
        return ""


# Module-level singleton (lazy init)
_explainer = SignalExplainer()


def explain_strong_signal(features_row: dict, direction: str) -> dict | None:
    """Module-level convenience function."""
    return _explainer.explain_signal(features_row, direction)
