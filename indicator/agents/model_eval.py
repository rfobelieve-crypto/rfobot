"""
ModelEval Agent — Tool-Use Architecture.

Claude investigates model performance by querying prediction history,
computing IC, checking accuracy by regime/tier, and inspecting recent
predictions. It decides how deep to investigate based on initial findings.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from indicator.agents.base import BaseAgent

logger = logging.getLogger(__name__)

TZ_TPE = timezone(timedelta(hours=8))
DUAL_MODEL_START = "2026-03-20"


class ModelEvalAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "ModelEval"

    @property
    def system_prompt(self) -> str:
        return """\
You are a senior quant evaluating a live BTC prediction model (dual XGBoost).

## Investigation Protocol
1. Start with IC overview (most important single metric)
2. Check direction accuracy overall and by tier (Strong/Moderate/Weak)
3. If IC or accuracy looks bad, drill into regime breakdown and recent trend
4. Check magnitude scale sanity (σ-scale bug detection)
5. Check signal outcomes from tracked_signals table
6. Conclude

## Key Thresholds (but use judgment, not blind thresholds)
- IC > 0.05 = useful signal, > 0.10 = good, < -0.05 sustained = broken
- IC between -0.03 and +0.03 over <100 bars = noise, don't panic
- Direction accuracy: Strong >55%, Moderate >50%, overall >48%
- pred_return_4h range: ±0.001~0.05 normal, >0.5 = σ-scale bug
- mag_pred: 0~0.05 normal, >1 = vol conversion missing
- Sample size matters: 30 bars = noise, 200+ bars = meaningful trend

## When to Recommend Retraining
Only if ALL of: IC < -0.05 for 200+ bars, accuracy < 45%, calibration inverted.
Otherwise: adjust thresholds, check features, or wait for more data.

## Actions — SUGGEST ONLY (you cannot auto-modify any parameter)
All hyperparameters are ★★★★★ risk — you can only send recommendations.
- IC near zero for 7d+ → `suggest_widen_deadzone` with IC data as evidence
- IC < -0.05 sustained → `suggest_threshold_change` with full stats
- Accuracy dropping in specific regime → `suggest_regime_change`
- Always include: sample size, time range, confidence interval in evidence.
- Be CONSERVATIVE: small IC fluctuations are noise. Only suggest on strong evidence.

Respond in Traditional Chinese."""

    def get_tools(self) -> list[dict]:
        return [
            {
                "name": "get_ic_overview",
                "description": "Compute Spearman IC (pred_return_4h vs actual) for: all-time, 7-day, 30-day. Also returns sample size and date range.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "get_direction_accuracy",
                "description": "Get direction accuracy breakdown: overall, by tier (Strong/Moderate/Weak), and recent 48h trend.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "get_regime_performance",
                "description": "Get accuracy breakdown by regime (CHOPPY/BULL/BEAR). Shows if model works in some regimes but fails in others.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "get_signal_distribution",
                "description": "Get signal distribution: NEUTRAL%, UP%, DOWN%, Strong%, flip rate, regime breakdown.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "get_magnitude_sanity",
                "description": "Check pred_return_4h and mag_pred value ranges to detect σ-scale bugs.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "get_recent_predictions",
                "description": "Get last 15 predictions with actual outcomes, direction, strength, regime.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer", "description": "Number of recent predictions to return (default 15)"},
                    },
                    "required": [],
                },
            },
            {
                "name": "get_signal_outcomes",
                "description": "Get tracked signal (Strong+Moderate) win rates, avg returns, streaks, and recent signal list.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
        ]

    def execute_tool(self, tool_name: str, tool_input: dict) -> str:
        handlers = {
            "get_ic_overview": self._tool_ic,
            "get_direction_accuracy": self._tool_accuracy,
            "get_regime_performance": self._tool_regime,
            "get_signal_distribution": self._tool_distribution,
            "get_magnitude_sanity": self._tool_magnitude,
            "get_recent_predictions": lambda: self._tool_recent(tool_input.get("count", 15)),
            "get_signal_outcomes": self._tool_signals,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            return json.dumps(handler(), default=str, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _load_eval_data(self):
        from shared.db import get_db_conn
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT dt, close, pred_return_4h, pred_direction_code,
                       confidence_score, strength_code, regime_code, dir_prob_up, mag_pred
                FROM indicator_history WHERE dt >= %s ORDER BY dt ASC
            """, (DUAL_MODEL_START,))
            rows = cur.fetchall()
        conn.close()
        if not rows:
            return None
        df = pd.DataFrame(rows).sort_values("dt").reset_index(drop=True)
        df["actual_4h"] = df["close"].shift(-4) / df["close"] - 1
        dir_map = {1: "UP", -1: "DOWN", 0: "NEUTRAL"}
        str_map = {3: "Strong", 2: "Moderate", 1: "Weak"}
        regime_map = {2: "BULL", -2: "BEAR", 0: "CHOPPY", -99: "WARMUP"}
        df["direction"] = df["pred_direction_code"].map(dir_map).fillna("NEUTRAL")
        df["strength"] = df["strength_code"].map(str_map).fillna("Weak")
        df["regime"] = df["regime_code"].map(regime_map).fillna("CHOPPY")
        return df

    def _tool_ic(self) -> dict:
        df = self._load_eval_data()
        if df is None:
            return {"error": "no data"}
        from scipy.stats import spearmanr
        eval_df = df.dropna(subset=["actual_4h", "pred_return_4h"])

        def _ic(sub):
            if len(sub) < 30:
                return None
            ic, p = spearmanr(sub["pred_return_4h"], sub["actual_4h"])
            return {"ic": round(ic, 4), "p_value": round(p, 4), "n": len(sub)}

        return {
            "all_time": _ic(eval_df),
            "last_7d": _ic(eval_df.tail(168)),
            "last_30d": _ic(eval_df.tail(720)),
            "last_3d": _ic(eval_df.tail(72)),
            "total_bars": len(df),
            "eval_bars": len(eval_df),
            "date_range": f"{df['dt'].iloc[0]} ~ {df['dt'].iloc[-1]}",
        }

    def _tool_accuracy(self) -> dict:
        df = self._load_eval_data()
        if df is None:
            return {"error": "no data"}
        eval_df = df.dropna(subset=["actual_4h"])
        d = eval_df[eval_df["direction"].isin(["UP", "DOWN"])].copy()
        d["correct"] = ((d["direction"] == "UP") & (d["actual_4h"] > 0)) | \
                       ((d["direction"] == "DOWN") & (d["actual_4h"] < 0))
        result = {}
        if len(d) > 0:
            result["overall"] = {"total": len(d), "correct": int(d["correct"].sum()),
                                 "pct": round(d["correct"].mean() * 100, 1)}
        for tier in ["Strong", "Moderate", "Weak"]:
            sub = d[d["strength"] == tier]
            if len(sub) >= 5:
                result[tier] = {"total": len(sub), "correct": int(sub["correct"].sum()),
                                "pct": round(sub["correct"].mean() * 100, 1)}
        recent = d.tail(48)
        if len(recent) >= 10:
            result["recent_48h"] = {"total": len(recent), "correct": int(recent["correct"].sum()),
                                    "pct": round(recent["correct"].mean() * 100, 1)}
        return result

    def _tool_regime(self) -> dict:
        df = self._load_eval_data()
        if df is None:
            return {"error": "no data"}
        eval_df = df.dropna(subset=["actual_4h"])
        d = eval_df[eval_df["direction"].isin(["UP", "DOWN"])].copy()
        d["correct"] = ((d["direction"] == "UP") & (d["actual_4h"] > 0)) | \
                       ((d["direction"] == "DOWN") & (d["actual_4h"] < 0))
        result = {}
        for regime in ["CHOPPY", "BULL", "BEAR", "WARMUP"]:
            sub = d[d["regime"] == regime]
            if len(sub) >= 5:
                result[regime] = {"total": len(sub), "correct": int(sub["correct"].sum()),
                                  "pct": round(sub["correct"].mean() * 100, 1)}
            # By direction within regime
            for dir_ in ["UP", "DOWN"]:
                sub2 = sub[sub["direction"] == dir_]
                if len(sub2) >= 3:
                    result[f"{regime}_{dir_}"] = {
                        "total": len(sub2), "correct": int(sub2["correct"].sum()),
                        "pct": round(sub2["correct"].mean() * 100, 1)}
        return result

    def _tool_distribution(self) -> dict:
        df = self._load_eval_data()
        if df is None:
            return {"error": "no data"}
        eval_df = df.dropna(subset=["actual_4h"])
        n = len(eval_df)
        dirs = eval_df["direction"].values
        flips = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i-1])
        return {
            "total_bars": n,
            "neutral_pct": round((eval_df["direction"] == "NEUTRAL").mean() * 100, 1),
            "up_pct": round((eval_df["direction"] == "UP").mean() * 100, 1),
            "down_pct": round((eval_df["direction"] == "DOWN").mean() * 100, 1),
            "strong_pct": round((eval_df["strength"] == "Strong").mean() * 100, 1),
            "moderate_pct": round((eval_df["strength"] == "Moderate").mean() * 100, 1),
            "flip_rate_pct": round(flips / n * 100, 1) if n > 1 else 0,
            "regime_pct": {r: round((eval_df["regime"] == r).mean() * 100, 1)
                          for r in ["CHOPPY", "BULL", "BEAR", "WARMUP"]},
        }

    def _tool_magnitude(self) -> dict:
        df = self._load_eval_data()
        if df is None:
            return {"error": "no data"}
        result = {}
        for col in ["pred_return_4h", "mag_pred"]:
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals) > 0:
                    result[col] = {
                        "mean": round(float(vals.mean()), 6),
                        "std": round(float(vals.std()), 6),
                        "min": round(float(vals.min()), 6),
                        "max": round(float(vals.max()), 6),
                        "abs_mean": round(float(vals.abs().mean()), 6),
                        "pct_above_0.1": round((vals.abs() > 0.1).mean() * 100, 1),
                        "pct_above_1.0": round((vals.abs() > 1.0).mean() * 100, 1),
                    }
        return result

    def _tool_recent(self, count: int = 15) -> list:
        df = self._load_eval_data()
        if df is None:
            return {"error": "no data"}
        eval_df = df.dropna(subset=["actual_4h"]).tail(count)
        return [
            {
                "dt": str(r["dt"]), "pred_ret": round(float(r["pred_return_4h"]), 6),
                "actual_ret": round(float(r["actual_4h"]), 6),
                "direction": r["direction"], "strength": r["strength"],
                "confidence": round(float(r["confidence_score"]), 1) if pd.notna(r["confidence_score"]) else None,
                "regime": r["regime"],
                "correct": bool((r["direction"] == "UP" and r["actual_4h"] > 0) or
                               (r["direction"] == "DOWN" and r["actual_4h"] < 0))
                           if r["direction"] in ("UP", "DOWN") else None,
            }
            for _, r in eval_df.iterrows()
        ]

    def _tool_signals(self) -> dict:
        from shared.db import get_db_conn
        conn = get_db_conn()
        result = {}
        with conn.cursor() as cur:
            for tier in ["Strong", "Moderate"]:
                cur.execute("""
                    SELECT COUNT(*) as total, SUM(filled) as filled, SUM(correct) as wins,
                           AVG(CASE WHEN filled=1 THEN actual_return_4h END) as avg_ret
                    FROM tracked_signals WHERE strength=%s AND signal_time>=%s
                """, (tier, DUAL_MODEL_START))
                r = cur.fetchone()
                t, f, w = int(r["total"] or 0), int(r["filled"] or 0), int(r["wins"] or 0)
                result[tier] = {
                    "total": t, "filled": f, "wins": w, "losses": f - w,
                    "win_rate_pct": round(w / f * 100, 1) if f > 0 else None,
                    "avg_return_pct": round(float(r["avg_ret"] or 0) * 100, 3),
                    "pending": t - f,
                }
            # Recent signals
            cur.execute("""
                SELECT signal_time, direction, strength, confidence, actual_return_4h, correct, filled
                FROM tracked_signals WHERE signal_time >= DATE_SUB(NOW(), INTERVAL 72 HOUR)
                ORDER BY signal_time DESC LIMIT 15
            """)
            result["recent"] = [
                {"time": str(r["signal_time"]), "dir": r["direction"], "tier": r["strength"],
                 "conf": float(r["confidence"]) if r["confidence"] else None,
                 "ret_pct": round(float(r["actual_return_4h"]) * 100, 3) if r["actual_return_4h"] else None,
                 "correct": bool(r["correct"]) if r["correct"] is not None else None,
                 "filled": bool(r["filled"])}
                for r in cur.fetchall()
            ]
        conn.close()
        return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    agent = ModelEvalAgent()
    if "--context-only" in sys.argv:
        for tool in agent.get_tools():
            print(f"\n--- {tool['name']} ---")
            print(agent.execute_tool(tool["name"], {}))
    else:
        result = agent.run()
        print(f"\nAgent: {result.agent_name} | Status: {result.status}")
        print(f"Summary: {result.summary}")
        print(f"Tools: {result.tool_calls_made} | Turns: {result.turns_used}")
