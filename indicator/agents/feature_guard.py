"""
FeatureGuard Agent — Tool-Use Architecture.

Claude gets tools to build features and inspect quality. It can check
NaN rates, distributions, specific feature groups, and trace NaN patterns
back to their upstream data source.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from indicator.agents.base import BaseAgent

logger = logging.getLogger(__name__)

TZ_TPE = timezone(timedelta(hours=8))

# Cached features from build — shared across tool calls within one run
_cached_features: pd.DataFrame | None = None


class FeatureGuardAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "FeatureGuard"

    @property
    def system_prompt(self) -> str:
        return """\
You are a senior quant researcher monitoring feature quality for a BTC prediction model.

## Your Investigation Protocol
1. First, build features from live data (build_features tool) — this gives you overall stats
2. Check NaN rates by feature group — trace NaN epidemics to upstream source
3. Check distribution of key features — catch drift or outliers
4. Check sanity of critical values (vol, funding rate, magnitudes)
5. Conclude with diagnosis

## What You Know
- 130+ features from 12 groups, built from Binance + Coinglass + Deribit data
- If upstream source fails → entire feature group goes NaN → zero-filled → silent degradation
- Feature naming: Coinglass features use `cg_` prefix, Deribit uses `bvol_`, aggtrades uses `agg_`
- Feature groups: binance_klines, coinglass_oi (cg_oi_*), coinglass_funding (cg_funding_*),
  coinglass_sentiment (cg_ls_*, cg_gls_*, cg_cb_*), coinglass_flow (cg_fcvd_*, cg_scvd_*, cg_taker_*),
  deribit (bvol_*), depth, aggtrades (agg_*), engineered_alpha
- Key sanity bounds:
  - realized_vol_20b: 0.0001~0.1 (hourly std of log returns)
  - cg_funding_close: -0.01~+0.01
  - depth_imbalance: -1 to +1
  - pred_return_4h: ±0.001~0.05 (if >0.5 → σ-scale bug)
  - mag_pred: 0~0.05 (if >1 → vol conversion missing)

## Self-Healing Actions (use repair tools when criteria are met)
- If NaN rate >30% AND cache files are stale → `repair_clear_stale_cache` then `repair_trigger_update`
- If a specific feature group is 100% NaN → likely upstream failure, clear cache for fresh fetch
- Always verify: after repair, re-check NaN rates before concluding.
- Do NOT repair if NaN rate is low (<10%) — small NaN rates are normal edge effects.

Respond in Traditional Chinese."""

    def get_tools(self) -> list[dict]:
        return [
            {
                "name": "build_features",
                "description": "Build features from live data. Returns: row count, column count, build time, overall NaN rate, and list of columns.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "check_nan_by_group",
                "description": "Check NaN rates grouped by data source. Shows which feature groups are affected and how severely.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "check_distribution",
                "description": "Get distribution stats (mean, std, min, max, current z-score) for key features.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "check_feature_values",
                "description": "Get the actual values of the latest feature row (what the model sees right now).",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "check_sanity",
                "description": "Check critical feature values against known valid ranges.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
        ]

    def execute_tool(self, tool_name: str, tool_input: dict) -> str:
        global _cached_features
        handlers = {
            "build_features": self._tool_build,
            "check_nan_by_group": self._tool_nan_groups,
            "check_distribution": self._tool_distribution,
            "check_feature_values": self._tool_latest_values,
            "check_sanity": self._tool_sanity,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            return json.dumps(handler(), default=str, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _get_features(self) -> pd.DataFrame:
        global _cached_features
        if _cached_features is not None:
            return _cached_features
        from indicator.data_fetcher import (
            fetch_binance_klines, fetch_coinglass, fetch_binance_depth,
            fetch_binance_aggtrades, fetch_deribit_dvol,
            fetch_deribit_options_summary, fetch_cg_options, fetch_cg_etf_flow,
        )
        from indicator.feature_builder_live import build_live_features

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
        _cached_features = build_live_features(
            klines, cg_data, depth=depth, aggtrades=aggtrades, options_data=options_data
        )
        return _cached_features

    def _tool_build(self) -> dict:
        t0 = time.time()
        features = self._get_features()
        build_time = time.time() - t0
        nan_rate = features.isna().mean().mean()
        last_nan = features.iloc[-1].isna().mean() if len(features) > 0 else 1.0
        return {
            "rows": len(features),
            "columns": len(features.columns),
            "build_time_s": round(build_time, 2),
            "overall_nan_rate_pct": round(nan_rate * 100, 2),
            "last_row_nan_pct": round(last_nan * 100, 1),
            "time_range": f"{features.index[0]} ~ {features.index[-1]}" if len(features) > 0 else "empty",
        }

    def _tool_nan_groups(self) -> dict:
        features = self._get_features()
        groups = {
            "binance_klines": ["log_return", "realized_vol_20b", "return_kurtosis", "volume", "taker_ratio"],
            "coinglass_oi": ["cg_oi_delta", "cg_oi_close_pctchg_4h", "cg_oi_close_pctchg_24h"],
            "coinglass_funding": ["cg_funding_close", "cg_funding_range", "cg_funding_close_zscore"],
            "coinglass_sentiment": ["cg_ls_ratio", "cg_gls_ratio", "cg_cb_premium"],
            "coinglass_flow": ["cg_fcvd_delta", "cg_scvd_delta", "cg_taker_delta"],
            "deribit": ["bvol_close", "bvol_change_1h", "bvol_intra_range"],
            "depth": ["depth_imbalance", "spread_bps", "near_imbalance"],
            "aggtrades": ["agg_large_delta", "agg_large_buy_ratio", "agg_large_delta_frac"],
            "engineered_alpha": ["impact_asymmetry", "post_absorb_breakout", "oi_price_divergence"],
        }
        result = {}
        last = features.iloc[-1] if len(features) > 0 else pd.Series()
        for group, expected in groups.items():
            present = [f for f in expected if f in features.columns]
            missing = [f for f in expected if f not in features.columns]
            if present:
                group_nan = features[present].iloc[-5:].isna().mean().mean() * 100
                last_nan = last[present].isna().sum() if len(last) > 0 else len(present)
            else:
                group_nan = 100.0
                last_nan = len(expected)
            result[group] = {
                "expected": len(expected), "present": len(present),
                "missing": missing,
                "nan_pct_last5": round(group_nan, 1),
                "last_row_nan": int(last_nan),
            }
        return result

    def _tool_distribution(self) -> dict:
        features = self._get_features()
        key_feats = ["realized_vol_20b", "cg_funding_close", "depth_imbalance",
                     "cg_oi_delta", "cg_ls_ratio", "volume_zscore_20",
                     "cg_cb_premium", "bvol_close"]
        result = {}
        for feat in key_feats:
            if feat not in features.columns:
                result[feat] = {"present": False}
                continue
            vals = features[feat].dropna()
            if len(vals) < 10:
                result[feat] = {"present": True, "insufficient_data": True}
                continue
            current = float(vals.iloc[-1])
            std = float(vals.std())
            result[feat] = {
                "current": round(current, 6),
                "mean": round(float(vals.mean()), 6),
                "std": round(std, 6),
                "min": round(float(vals.min()), 6),
                "max": round(float(vals.max()), 6),
                "zscore": round((current - float(vals.mean())) / max(std, 1e-10), 2),
            }
        return result

    def _tool_latest_values(self) -> dict:
        features = self._get_features()
        if features.empty:
            return {"error": "no features"}
        last = features.iloc[-1]
        important = ["close", "log_return", "realized_vol_20b", "volume",
                     "cg_funding_close", "cg_oi_delta", "cg_ls_ratio",
                     "depth_imbalance", "agg_large_delta", "bvol_close",
                     "cg_cb_premium", "taker_ratio",
                     "impact_asymmetry", "post_absorb_breakout"]
        return {f: round(float(last[f]), 6) for f in important if f in last.index and pd.notna(last[f])}

    def _tool_sanity(self) -> dict:
        features = self._get_features()
        if features.empty:
            return {"error": "no features"}
        last = features.iloc[-1]
        rules = {
            "realized_vol_20b": (0.0001, 0.1, "hourly vol 0.01%-10%"),
            "cg_funding_close": (-0.01, 0.01, "funding ±1% is extreme"),
            "depth_imbalance": (-1.0, 1.0, "bounded [-1,1]"),
        }
        result = {}
        for feat, (lo, hi, note) in rules.items():
            if feat in last.index and pd.notna(last[feat]):
                val = float(last[feat])
                result[feat] = {"value": val, "in_range": lo <= val <= hi, "range": f"[{lo},{hi}]", "note": note}
        return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    agent = FeatureGuardAgent()
    if "--context-only" in sys.argv:
        for tool in agent.get_tools():
            print(f"\n--- {tool['name']} ---")
            print(agent.execute_tool(tool["name"], {}))
    else:
        result = agent.run()
        print(f"\nAgent: {result.agent_name} | Status: {result.status}")
        print(f"Summary: {result.summary}")
        print(f"Tools: {result.tool_calls_made} | Turns: {result.turns_used}")
