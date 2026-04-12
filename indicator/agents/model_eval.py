"""
ModelEval Agent — AI-powered model performance evaluator.

This is the most critical agent. It reads actual prediction outcomes
from MySQL and gives Claude the raw IC, accuracy, calibration data
to judge whether the model is healthy, degrading, or broken.

Claude can detect patterns humans miss: regime-dependent degradation,
calibration drift, systematic biases, and the difference between
"normal variance" and "model is actually broken".
"""
from __future__ import annotations

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
You are a senior quant researcher evaluating a live BTC prediction model. \
You understand that live model performance is noisy, and your job is to \
separate real degradation from normal statistical variance.

## Model Architecture
- Dual XGBoost: Direction classifier (P(UP)) + Magnitude regressor (|return_4h|/vol)
- Predictions every 1h, outcome = close[t+4] / close[t] - 1
- Signals: Strong (conf≥80), Moderate (conf≥65), Weak (<65)
- Regime detection: CHOPPY / TRENDING_BULL / TRENDING_BEAR

## Key Metrics You Evaluate

### Spearman IC (Information Coefficient)
- pred_return_4h vs actual_return_4h rank correlation
- Healthy: IC > 0.05 (weak but positive), good: > 0.10
- IC ≈ 0: model adds no value over random
- IC < -0.05: model is INVERSELY correlated — something is seriously wrong
- IC between -0.03 and +0.03: statistical noise, not meaningful
- **Critical**: distinguish "IC=-0.01 for 3 days" (noise) from "IC=-0.08 sustained" (broken)

### Direction Accuracy
- Overall: % of directional (non-NEUTRAL) predictions that were correct
- By tier: Strong should be >55%, Moderate >50%
- Recent vs overall: if recent accuracy drops >10% below overall → regression
- By regime: model may work in CHOPPY but fail in TRENDING or vice versa

### Calibration
- Strong signals should have higher accuracy than Moderate
- High confidence should correlate with larger actual moves
- If calibration inverts (weak signals outperform strong) → model confidence is broken

### Magnitude Scale
- pred_return_4h should be in range ±0.001 to ±0.05
- mag_pred should be 0 to 0.05 (return scale, after vol conversion)
- If pred_return_4h > 0.5 or mag_pred > 1 → σ-scale bug (not converted from vol-adjusted)

### Signal Distribution
- NEUTRAL rate: 50-70% is normal, >80% too conservative, <30% too aggressive
- Flip rate: direction change frequency. >40% = too noisy
- Strong signal frequency: ~2-5% of all bars is healthy

## Your Judgment Framework
1. **Sample size matters**: 30 bars = noise. 200 bars = trend. 500+ bars = reliable.
2. **Regime awareness**: IC may be negative in CHOPPY but positive in TRENDING — that's OK if TRENDING compensates.
3. **Recent vs historical**: weight recent 48h performance more if you suspect regime change.
4. **Don't cry wolf**: IC=-0.01 over 50 bars is NOT an emergency. IC=-0.10 over 200 bars IS.
5. **Root cause thinking**: bad IC could be: model decay, feature degradation, σ-scale bug, data gap, or just a hard market.

## Retraining Recommendation
Only recommend retraining if:
- IC < -0.05 sustained over 200+ bars, AND
- Direction accuracy < 45% sustained, AND
- Calibration is inverted (weak > strong accuracy)
Otherwise suggest: adjust thresholds, check features, wait for more data.

Respond in Traditional Chinese for summary/report."""

    def collect_context(self) -> dict:
        """Pull all prediction outcomes from MySQL for analysis."""
        context = {}

        # Load prediction history
        context["predictions"] = self._load_predictions()

        # Signal tracking outcomes
        context["signal_outcomes"] = self._load_signal_outcomes()

        # Model config
        context["model_config"] = self._load_model_config()

        return context

    def get_available_actions(self) -> list[dict]:
        return [
            {"name": "ALERT", "description": "Send Telegram alert about model degradation"},
            {"name": "LOG", "description": "Log finding for tracking"},
            {"name": "NONE", "description": "Informational, no action needed"},
        ]

    def _load_predictions(self) -> dict:
        """Load prediction history and compute actual outcomes."""
        try:
            from shared.db import get_db_conn
            conn = get_db_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT dt, close, pred_return_4h, pred_direction_code,
                           confidence_score, strength_code, regime_code,
                           dir_prob_up, mag_pred
                    FROM indicator_history
                    WHERE dt >= %s
                    ORDER BY dt ASC
                """, (DUAL_MODEL_START,))
                rows = cur.fetchall()
            conn.close()
        except Exception as e:
            return {"error": str(e)}

        if not rows:
            return {"error": "no data", "row_count": 0}

        df = pd.DataFrame(rows)
        df = df.sort_values("dt").reset_index(drop=True)

        # Compute actual 4h return
        df["actual_4h"] = df["close"].shift(-4) / df["close"] - 1

        # Map codes
        dir_map = {1: "UP", -1: "DOWN", 0: "NEUTRAL"}
        str_map = {3: "Strong", 2: "Moderate", 1: "Weak"}
        regime_map = {2: "BULL", -2: "BEAR", 0: "CHOPPY", -99: "WARMUP"}
        df["direction"] = df["pred_direction_code"].map(dir_map).fillna("NEUTRAL")
        df["strength"] = df["strength_code"].map(str_map).fillna("Weak")
        df["regime"] = df["regime_code"].map(regime_map).fillna("CHOPPY")

        # Only rows with known outcomes
        eval_df = df.dropna(subset=["actual_4h"])
        directional = eval_df[eval_df["direction"].isin(["UP", "DOWN"])]

        # Correctness
        if len(directional) > 0:
            directional = directional.copy()
            directional["correct"] = (
                ((directional["direction"] == "UP") & (directional["actual_4h"] > 0)) |
                ((directional["direction"] == "DOWN") & (directional["actual_4h"] < 0))
            ).astype(int)

        # ── Compute metrics ──

        # Overall IC
        from scipy.stats import spearmanr
        ic_all = None
        if len(eval_df) >= 30:
            valid = eval_df.dropna(subset=["pred_return_4h", "actual_4h"])
            if len(valid) >= 30:
                ic_all, _ = spearmanr(valid["pred_return_4h"], valid["actual_4h"])

        # Rolling IC (7d, 30d)
        ic_7d, ic_30d = None, None
        if len(eval_df) >= 168:
            last_7d = eval_df.tail(168).dropna(subset=["pred_return_4h", "actual_4h"])
            if len(last_7d) >= 30:
                ic_7d, _ = spearmanr(last_7d["pred_return_4h"], last_7d["actual_4h"])
        if len(eval_df) >= 720:
            last_30d = eval_df.tail(720).dropna(subset=["pred_return_4h", "actual_4h"])
            if len(last_30d) >= 30:
                ic_30d, _ = spearmanr(last_30d["pred_return_4h"], last_30d["actual_4h"])

        # Direction accuracy
        accuracy = {}
        if len(directional) > 0:
            accuracy["overall"] = {
                "total": len(directional),
                "correct": int(directional["correct"].sum()),
                "pct": round(directional["correct"].mean() * 100, 1),
            }
            # By tier
            for tier in ["Strong", "Moderate", "Weak"]:
                sub = directional[directional["strength"] == tier]
                if len(sub) >= 5:
                    accuracy[tier] = {
                        "total": len(sub),
                        "correct": int(sub["correct"].sum()),
                        "pct": round(sub["correct"].mean() * 100, 1),
                    }
            # By regime
            for regime in ["CHOPPY", "BULL", "BEAR"]:
                sub = directional[directional["regime"] == regime]
                if len(sub) >= 10:
                    accuracy[f"regime_{regime}"] = {
                        "total": len(sub),
                        "correct": int(sub["correct"].sum()),
                        "pct": round(sub["correct"].mean() * 100, 1),
                    }

        # Recent accuracy (48h)
        recent = directional.tail(48)
        if len(recent) >= 10:
            accuracy["recent_48h"] = {
                "total": len(recent),
                "correct": int(recent["correct"].sum()),
                "pct": round(recent["correct"].mean() * 100, 1),
            }

        # Signal distribution
        total_bars = len(eval_df)
        dist = {
            "total_bars": total_bars,
            "neutral_pct": round((eval_df["direction"] == "NEUTRAL").mean() * 100, 1),
            "up_pct": round((eval_df["direction"] == "UP").mean() * 100, 1),
            "down_pct": round((eval_df["direction"] == "DOWN").mean() * 100, 1),
        }
        # Strong signal rate
        dist["strong_pct"] = round((eval_df["strength"] == "Strong").mean() * 100, 1)

        # Flip rate
        dirs = eval_df["direction"].values
        flips = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i-1])
        dist["flip_rate_pct"] = round(flips / len(dirs) * 100, 1) if len(dirs) > 1 else 0

        # Regime breakdown
        dist["regime_pct"] = {
            regime: round((eval_df["regime"] == regime).mean() * 100, 1)
            for regime in ["CHOPPY", "BULL", "BEAR", "WARMUP"]
        }

        # Magnitude sanity
        mag_stats = {}
        if "pred_return_4h" in eval_df.columns:
            pred_r = eval_df["pred_return_4h"].dropna()
            if len(pred_r) > 0:
                mag_stats["pred_return_4h"] = {
                    "mean": round(float(pred_r.mean()), 6),
                    "std": round(float(pred_r.std()), 6),
                    "min": round(float(pred_r.min()), 6),
                    "max": round(float(pred_r.max()), 6),
                    "abs_mean": round(float(pred_r.abs().mean()), 6),
                }
        if "mag_pred" in eval_df.columns:
            mp = eval_df["mag_pred"].dropna()
            if len(mp) > 0:
                mag_stats["mag_pred"] = {
                    "mean": round(float(mp.mean()), 6),
                    "std": round(float(mp.std()), 6),
                    "min": round(float(mp.min()), 6),
                    "max": round(float(mp.max()), 6),
                }

        # Last 10 predictions
        last_10 = []
        for _, row in eval_df.tail(10).iterrows():
            last_10.append({
                "dt": str(row["dt"]),
                "pred_return": round(float(row.get("pred_return_4h", 0)), 6),
                "actual_return": round(float(row["actual_4h"]), 6) if pd.notna(row["actual_4h"]) else None,
                "direction": row["direction"],
                "strength": row["strength"],
                "confidence": round(float(row["confidence_score"]), 1) if pd.notna(row["confidence_score"]) else None,
                "regime": row["regime"],
            })

        return {
            "row_count": len(df),
            "eval_rows": len(eval_df),
            "directional_rows": len(directional),
            "date_range": f"{df['dt'].iloc[0]} ~ {df['dt'].iloc[-1]}",
            "ic": {
                "all": round(ic_all, 4) if ic_all is not None else None,
                "7d": round(ic_7d, 4) if ic_7d is not None else None,
                "30d": round(ic_30d, 4) if ic_30d is not None else None,
            },
            "accuracy": accuracy,
            "signal_distribution": dist,
            "magnitude_stats": mag_stats,
            "last_10": last_10,
        }

    def _load_signal_outcomes(self) -> dict:
        """Load tracked signal win rates."""
        try:
            from shared.db import get_db_conn
            conn = get_db_conn()
            result = {}
            with conn.cursor() as cur:
                for tier in ["Strong", "Moderate"]:
                    cur.execute("""
                        SELECT COUNT(*) as total,
                               SUM(filled) as filled,
                               SUM(correct) as wins,
                               AVG(CASE WHEN filled=1 THEN actual_return_4h END) as avg_ret,
                               AVG(CASE WHEN filled=1 AND correct=1 THEN actual_return_4h END) as avg_win,
                               AVG(CASE WHEN filled=1 AND correct=0 THEN actual_return_4h END) as avg_loss
                        FROM tracked_signals
                        WHERE strength = %s AND signal_time >= %s
                    """, (tier, DUAL_MODEL_START))
                    row = cur.fetchone()
                    total = int(row["total"] or 0)
                    filled = int(row["filled"] or 0)
                    wins = int(row["wins"] or 0)
                    result[tier] = {
                        "total": total,
                        "filled": filled,
                        "wins": wins,
                        "losses": filled - wins,
                        "win_rate_pct": round(wins / filled * 100, 1) if filled > 0 else None,
                        "avg_return_pct": round(float(row["avg_ret"] or 0) * 100, 3),
                        "avg_win_pct": round(float(row["avg_win"] or 0) * 100, 3) if row["avg_win"] else None,
                        "avg_loss_pct": round(float(row["avg_loss"] or 0) * 100, 3) if row["avg_loss"] else None,
                        "pending": total - filled,
                    }

                # Recent signals (last 72h)
                cur.execute("""
                    SELECT signal_time, direction, strength, confidence,
                           actual_return_4h, correct, filled
                    FROM tracked_signals
                    WHERE signal_time >= DATE_SUB(NOW(), INTERVAL 72 HOUR)
                    ORDER BY signal_time DESC
                    LIMIT 20
                """)
                recent = []
                for r in cur.fetchall():
                    recent.append({
                        "time": str(r["signal_time"]),
                        "direction": r["direction"],
                        "strength": r["strength"],
                        "confidence": float(r["confidence"]) if r["confidence"] else None,
                        "return_pct": round(float(r["actual_return_4h"]) * 100, 3) if r["actual_return_4h"] else None,
                        "correct": bool(r["correct"]) if r["correct"] is not None else None,
                        "filled": bool(r["filled"]),
                    })
                result["recent_signals"] = recent

            conn.close()
            return result
        except Exception as e:
            return {"error": str(e)}

    def _load_model_config(self) -> dict:
        """Load model configuration for context."""
        import json
        from pathlib import Path

        config = {}
        dual_dir = Path("indicator/model_artifacts/dual_model")

        for name in ["magnitude_config.json"]:
            p = dual_dir / name
            if p.exists():
                with open(p) as f:
                    config[name] = json.load(f)

        # Pred history stats
        stats_path = dual_dir / "training_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            ph = stats.get("pred_history", [])
            config["pred_history"] = {
                "count": len(ph),
                "mean": round(np.mean(ph), 3) if ph else None,
                "std": round(np.std(ph), 3) if ph else None,
                "min": round(min(ph), 3) if ph else None,
                "max": round(max(ph), 3) if ph else None,
                "note": "Values in σ-scale (vol-adjusted). Should be ~0.5-5.0 range.",
            }

        return config


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json as _json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    agent = ModelEvalAgent()
    if "--context-only" in sys.argv:
        ctx = agent.collect_context()
        print(_json.dumps(ctx, indent=2, default=str, ensure_ascii=False))
    else:
        result = agent.run()
        print(f"\nAgent: {result.agent_name} | Status: {result.status}")
        print(f"Summary: {result.summary}")
        print(f"\nDiagnosis:\n{_json.dumps(result.diagnosis, indent=2, ensure_ascii=False)}")
