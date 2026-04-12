"""
FeatureGuard Agent — AI-powered feature quality monitor.

Monitors the 130+ engineered features for:
  - NaN epidemics (which features, which data source caused it)
  - Distribution drift (feature values shifting outside training range)
  - Correlation breakdown (features that should co-move diverging)
  - Upstream data source impact (if CG fails, which features go NaN)
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from indicator.agents.base import BaseAgent

logger = logging.getLogger(__name__)

TZ_TPE = timezone(timedelta(hours=8))

# Feature groups by data source — if a source fails, these go NaN
FEATURE_GROUPS = {
    "binance_klines": [
        "log_return", "realized_vol_20b", "return_kurtosis", "close_to_ema20_ratio",
        "vol_regime", "hour_sin", "hour_cos", "day_sin", "day_cos",
        "volume", "taker_ratio", "volume_zscore_20",
    ],
    "coinglass_oi": ["oi_change_1h", "oi_change_4h", "oi_change_24h", "oi_zscore_20"],
    "coinglass_funding": ["funding_rate", "funding_deviation", "funding_zscore_20"],
    "coinglass_liquidation": ["liq_long_usd_1h", "liq_short_usd_1h", "liq_ratio_1h"],
    "coinglass_sentiment": [
        "long_short_ratio", "global_ls_ratio", "top_ls_position_ratio",
        "coinbase_premium",
    ],
    "coinglass_flow": ["futures_cvd_agg", "spot_cvd_agg", "taker_buy_vol", "taker_sell_vol"],
    "deribit": ["dvol", "dvol_change", "pc_vol_ratio", "pc_oi_ratio", "iv_skew"],
    "depth": ["depth_imbalance", "near_bid_usd", "near_ask_usd", "spread_bps"],
    "aggtrades": [
        "large_buy_usd", "large_sell_usd", "large_delta_usd",
        "large_trade_ratio", "avg_trade_usd",
    ],
    "engineered_alpha": [
        "impact_asymmetry", "post_absorb_breakout", "liq_fragility",
        "toxicity_flow_ratio", "smart_money_divergence",
    ],
}


class FeatureGuardAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "FeatureGuard"

    @property
    def system_prompt(self) -> str:
        return """\
You are a senior quantitative researcher monitoring feature quality for a \
BTC prediction model. You understand that garbage features → garbage predictions, \
and your job is to catch feature degradation BEFORE it corrupts model output.

## System Context
- 130+ engineered features from 12 groups, feeding dual XGBoost models
- Features built hourly from: Binance (klines/depth/aggTrades), Coinglass (14+ endpoints), Deribit
- All features are trailing-only (no look-ahead bias)
- Feature builder: `build_live_features()` in `feature_builder_live.py`

## What You Monitor

### 1. NaN Epidemics
- If an upstream source fails, entire feature groups go NaN
- NaN features are zero-filled before model input — this silently degrades predictions
- Key question: are NaNs random (one-off) or systematic (upstream failure)?
- A single NaN column is noise. An entire group going NaN = upstream problem.

### 2. Distribution Drift
- Features should stay within their training range
- Extreme outliers (>5σ from training mean) might be: data error, market regime change, or real signal
- You need to distinguish "BTC just moved 10% so vol is extreme" from "funding_rate returned 999"
- Key ratios to watch: realized_vol_20b, funding_rate, oi_change, depth_imbalance

### 3. Value Sanity
- `realized_vol_20b` should be 0.001-0.05 range (hourly std of log returns)
- `funding_rate` should be -0.001 to +0.003 range (extreme: ±0.01)
- `depth_imbalance` should be -1 to +1
- `pred_return_4h` should be -0.05 to +0.05 (if larger → magnitude scale bug)
- `mag_pred` should be 0 to 0.05 (if >1 → σ-scale not converted)

### 4. Upstream Attribution
- Feature groups map to data sources. If you see NaN pattern, trace it to the source.
- Coinglass groups: oi, funding, liquidation, sentiment, flow
- Binance groups: klines (most critical), depth, aggtrades
- Deribit: dvol, options

## Your Judgment
- Don't just flag NaN%. Explain WHY and WHAT IMPACT on predictions.
- "funding features 100% NaN" → "model loses funding signal, direction accuracy may drop 3-5%"
- "realized_vol at 0.08 (4x normal)" → "market crash? check if BTC moved >8% today"
- Correlate findings: if OI features drift AND funding extreme → leverage cascade happening
- Respond in Traditional Chinese for summary/report fields."""

    def collect_context(self) -> dict:
        """Build features and analyze their quality."""
        context = {}

        # Build current features
        try:
            features, build_info = self._build_current_features()
            context["build_success"] = True
            context["build_info"] = build_info

            # NaN analysis by group
            context["nan_analysis"] = self._analyze_nans(features)

            # Distribution stats for key features
            context["distribution"] = self._analyze_distribution(features)

            # Sanity checks on critical values
            context["sanity_checks"] = self._check_sanity(features)

            # Last row (what the model will actually see)
            context["latest_features"] = self._extract_latest(features)

        except Exception as e:
            context["build_success"] = False
            context["build_error"] = str(e)

        return context

    def get_available_actions(self) -> list[dict]:
        return [
            {"name": "ALERT", "description": "Send Telegram alert about feature degradation"},
            {"name": "LOG", "description": "Log finding for historical tracking"},
            {"name": "NONE", "description": "Informational, no action needed"},
        ]

    def _build_current_features(self) -> tuple:
        """Build features from live data, capture timing and errors."""
        from indicator.data_fetcher import (
            fetch_binance_klines, fetch_coinglass,
            fetch_binance_depth, fetch_binance_aggtrades,
            fetch_deribit_dvol, fetch_deribit_options_summary,
            fetch_cg_options, fetch_cg_etf_flow,
        )
        from indicator.feature_builder_live import build_live_features

        t0 = time.time()

        klines = fetch_binance_klines(limit=200)
        cg_data = fetch_coinglass(interval="1h", limit=200)
        depth = fetch_binance_depth()
        aggtrades = fetch_binance_aggtrades()

        options_data = {}
        for fn in [fetch_cg_options, fetch_cg_etf_flow,
                    fetch_deribit_dvol, fetch_deribit_options_summary]:
            try:
                options_data.update(fn())
            except Exception:
                pass

        features = build_live_features(klines, cg_data, depth=depth,
                                        aggtrades=aggtrades, options_data=options_data)
        build_time = time.time() - t0

        build_info = {
            "rows": len(features),
            "columns": len(features.columns),
            "build_time_s": round(build_time, 2),
            "time_range": f"{features.index[0]} ~ {features.index[-1]}" if len(features) > 0 else "empty",
        }

        return features, build_info

    def _analyze_nans(self, features: pd.DataFrame) -> dict:
        """Analyze NaN patterns by feature group."""
        result = {}
        total_cols = len(features.columns)
        last_row = features.iloc[-1] if len(features) > 0 else pd.Series()

        # Overall
        nan_counts = features.isna().sum()
        nan_pct = (nan_counts / len(features) * 100).round(1)
        overall_nan_rate = features.isna().mean().mean()

        result["overall"] = {
            "total_features": total_cols,
            "overall_nan_rate_pct": round(overall_nan_rate * 100, 2),
            "last_row_nan_count": int(last_row.isna().sum()) if len(last_row) > 0 else None,
            "last_row_nan_pct": round(last_row.isna().mean() * 100, 1) if len(last_row) > 0 else None,
        }

        # By group
        group_analysis = {}
        for group_name, expected_features in FEATURE_GROUPS.items():
            present = [f for f in expected_features if f in features.columns]
            missing = [f for f in expected_features if f not in features.columns]

            if present:
                group_nan_pct = features[present].iloc[-5:].isna().mean().mean() * 100
                last_nan = last_row[present].isna().sum() if len(last_row) > 0 else 0
            else:
                group_nan_pct = 100.0
                last_nan = len(expected_features)

            group_analysis[group_name] = {
                "expected": len(expected_features),
                "present": len(present),
                "missing_from_df": missing,
                "nan_pct_last5": round(group_nan_pct, 1),
                "last_row_nan": int(last_nan),
            }

        result["by_group"] = group_analysis

        # Top NaN features (last 5 rows)
        last5_nan = features.iloc[-5:].isna().mean().sort_values(ascending=False)
        top_nan = {col: round(pct * 100, 1) for col, pct in last5_nan.head(10).items() if pct > 0}
        result["top_nan_features"] = top_nan

        return result

    def _analyze_distribution(self, features: pd.DataFrame) -> dict:
        """Check distribution stats for key features."""
        key_features = [
            "realized_vol_20b", "funding_rate", "depth_imbalance",
            "oi_change_1h", "long_short_ratio", "volume_zscore_20",
            "coinbase_premium", "dvol",
        ]

        result = {}
        for feat in key_features:
            if feat not in features.columns:
                result[feat] = {"present": False}
                continue

            vals = features[feat].dropna()
            if len(vals) < 10:
                result[feat] = {"present": True, "insufficient_data": True}
                continue

            current = float(vals.iloc[-1]) if len(vals) > 0 else None
            result[feat] = {
                "present": True,
                "current": current,
                "mean": round(float(vals.mean()), 6),
                "std": round(float(vals.std()), 6),
                "min": round(float(vals.min()), 6),
                "max": round(float(vals.max()), 6),
                "zscore_current": round((current - float(vals.mean())) / max(float(vals.std()), 1e-10), 2) if current else None,
            }

        return result

    def _check_sanity(self, features: pd.DataFrame) -> dict:
        """Check critical values against known valid ranges."""
        checks = {}
        last = features.iloc[-1] if len(features) > 0 else pd.Series()

        sanity_rules = {
            "realized_vol_20b": (0.0001, 0.1, "hourly vol should be 0.01%-10%"),
            "funding_rate": (-0.01, 0.01, "funding rate ±1% is extreme"),
            "depth_imbalance": (-1.0, 1.0, "imbalance is bounded [-1, 1]"),
            "volume_zscore_20": (-5, 5, "z-score beyond ±5 is very unusual"),
        }

        for feat, (lo, hi, note) in sanity_rules.items():
            if feat in last.index and pd.notna(last[feat]):
                val = float(last[feat])
                checks[feat] = {
                    "value": val,
                    "in_range": lo <= val <= hi,
                    "range": f"[{lo}, {hi}]",
                    "note": note,
                }

        return checks

    def _extract_latest(self, features: pd.DataFrame) -> dict:
        """Extract the latest feature row (what model actually sees)."""
        if features.empty:
            return {}

        last = features.iloc[-1]
        # Only include non-NaN values, limit to 30 most important
        important = [
            "close", "log_return", "realized_vol_20b", "volume",
            "funding_rate", "oi_change_1h", "long_short_ratio",
            "depth_imbalance", "large_delta_usd", "dvol",
            "coinbase_premium", "taker_ratio",
            "impact_asymmetry", "post_absorb_breakout",
        ]
        result = {}
        for feat in important:
            if feat in last.index and pd.notna(last[feat]):
                result[feat] = round(float(last[feat]), 6)
        return result


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json as _json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    agent = FeatureGuardAgent()
    if "--context-only" in sys.argv:
        ctx = agent.collect_context()
        print(_json.dumps(ctx, indent=2, default=str, ensure_ascii=False))
    else:
        result = agent.run()
        print(f"\nAgent: {result.agent_name} | Status: {result.status}")
        print(f"Summary: {result.summary}")
        print(f"\nDiagnosis:\n{_json.dumps(result.diagnosis, indent=2, ensure_ascii=False)}")
