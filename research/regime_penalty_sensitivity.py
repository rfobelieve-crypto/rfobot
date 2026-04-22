"""
Research: find the optimal BULL_CONTRA_PENALTY by sweeping values and
measuring their effect on OOS signal quality.

Why this exists
---------------
The production `BULL_CONTRA_PENALTY = 2.5` constant was introduced in
commit abcea69 (2026-04-09) as a heuristic round number — no sensitivity
sweep justified the choice. This script closes that gap.

Method
------
1. Use the production walk-forward OOS predictions
   (research/results/dual_model/direction_reg_oos_mse.parquet, 3696 bars
   from the deployed model's training run, Spearman IC ≈ 0.18).
2. Rebuild regime labels via the same trailing-only rule as inference.py
   (_assign_regime: vol_pct>0.6 + |ret_24h|>0.5%).
3. For each penalty p ∈ {1.0, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0}, replay
   the production rolling-percentile decoder with the contra-trend
   threshold multiplied by p on the regime-opposed side. All other
   behavior (warmup fallback, buffer window, top-fracs) matches
   inference.py exactly.
4. Report per-penalty: total Strong+Moderate signal count, overall
   precision, breakdown by contra-trend vs aligned vs choppy, with
   bootstrap 95% CI.

Outputs: research/results/regime_penalty_sensitivity.json
"""
from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIR_OOS = Path("research/results/dual_model/direction_reg_oos_mse.parquet")
FEATURE_CACHE = Path("research/dual_model/.cache/features_all.parquet")
CONFIG_FILE = Path("indicator/model_artifacts/dual_model/direction_reg_config.json")
RESULTS_FILE = Path("research/results/regime_penalty_sensitivity.json")

PENALTIES = [1.0, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0]


def assign_regime(close: pd.Series) -> np.ndarray:
    """Identical trailing-only rule to inference.py _assign_regime."""
    WARMUP_BARS = 72
    log_ret = np.log(close / close.shift(1))
    ret_24h = close.pct_change(24)
    vol_24h = log_ret.rolling(24).std()
    vol_pct = vol_24h.expanding(min_periods=WARMUP_BARS).rank(pct=True)

    regime = np.full(len(close), "CHOPPY", dtype=object)
    regime[(vol_pct > 0.6).values & (ret_24h > 0.005).values] = "TRENDING_BULL"
    regime[(vol_pct > 0.6).values & (ret_24h < -0.005).values] = "TRENDING_BEAR"
    regime[:WARMUP_BARS] = "WARMUP"
    return regime


def decode(pred_ret: np.ndarray, regime: np.ndarray, penalty: float,
           cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Replay production rolling-percentile decoder with custom penalty.
    Matches inference.py line 298-338 exactly.
    """
    warmup = cfg["warmup_bars"]
    strong_frac = cfg["strong_top_frac"]
    mod_frac = cfg["moderate_top_frac"]
    win = cfg["percentile_window"]
    fb = cfg["fallback"]

    n = len(pred_ret)
    buf = deque(maxlen=win)
    direction = np.full(n, "NEUTRAL", dtype=object)
    strength = np.full(n, "Weak", dtype=object)

    for i in range(n):
        p = float(pred_ret[i])
        buf.append(p)
        buf_len = len(buf)

        if buf_len < warmup:
            up_strong = fb["strong_up"]
            dn_strong = fb["strong_dn"]
            up_mod = fb["moderate_up"]
            dn_mod = fb["moderate_dn"]
        else:
            arr = np.fromiter(buf, dtype=float)
            up_strong = float(np.quantile(arr, 1.0 - strong_frac / 2.0))
            dn_strong = float(np.quantile(arr, strong_frac / 2.0))
            up_mod = float(np.quantile(arr, 1.0 - mod_frac / 2.0))
            dn_mod = float(np.quantile(arr, mod_frac / 2.0))

        reg = regime[i]
        up_mul = penalty if reg == "TRENDING_BEAR" else 1.0
        dn_mul = penalty if reg == "TRENDING_BULL" else 1.0

        up_strong_eff = up_strong * up_mul
        up_mod_eff = up_mod * up_mul
        dn_strong_eff = dn_strong * dn_mul
        dn_mod_eff = dn_mod * dn_mul

        if p >= up_strong_eff:
            direction[i] = "UP"; strength[i] = "Strong"
        elif p <= dn_strong_eff:
            direction[i] = "DOWN"; strength[i] = "Strong"
        elif p >= up_mod_eff:
            direction[i] = "UP"; strength[i] = "Moderate"
        elif p <= dn_mod_eff:
            direction[i] = "DOWN"; strength[i] = "Moderate"
    return direction, strength


def bootstrap_precision(correct: np.ndarray, n_boot: int = 500) -> tuple[float, float]:
    n = len(correct)
    if n < 20:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(42)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = correct[idx].mean()
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def evaluate(pred_ret: np.ndarray, regime: np.ndarray, actual_ret: np.ndarray,
             penalty: float, cfg: dict) -> dict:
    direction, strength = decode(pred_ret, regime, penalty, cfg)

    signal_mask = strength != "Weak"
    strong_mask = strength == "Strong"

    if signal_mask.sum() == 0:
        return {
            "penalty": penalty,
            "signals_total": 0,
            "overall_precision": None,
            "overall_ci": [None, None],
            "strong_count": 0,
            "strong_precision": None,
            "by_regime_contra": {},
            "by_regime_aligned": {},
            "choppy": {},
        }

    # Hit map: for Strong+Moderate signals, was direction right?
    correct = np.zeros(len(pred_ret), dtype=bool)
    correct |= (direction == "UP") & (actual_ret > 0)
    correct |= (direction == "DOWN") & (actual_ret < 0)

    ovr_correct = correct[signal_mask]
    ovr_prec = float(ovr_correct.mean())
    ovr_lo, ovr_hi = bootstrap_precision(ovr_correct)

    strong_correct = correct[strong_mask]
    strong_prec = float(strong_correct.mean()) if len(strong_correct) else None

    def _seg(mask: np.ndarray) -> dict:
        m = mask & signal_mask
        n = int(m.sum())
        if n == 0:
            return {"n": 0, "precision": None, "ci": [None, None]}
        c = correct[m]
        p = float(c.mean())
        lo, hi = bootstrap_precision(c)
        return {"n": n, "precision": p, "ci": [lo, hi]}

    # Contra-trend signals: BULL+DOWN or BEAR+UP (the side the penalty targets)
    contra_mask = (
        ((regime == "TRENDING_BULL") & (direction == "DOWN")) |
        ((regime == "TRENDING_BEAR") & (direction == "UP"))
    )
    # Aligned (trend-following): BULL+UP or BEAR+DOWN
    aligned_mask = (
        ((regime == "TRENDING_BULL") & (direction == "UP")) |
        ((regime == "TRENDING_BEAR") & (direction == "DOWN"))
    )
    choppy_mask = (regime == "CHOPPY")

    return {
        "penalty": penalty,
        "signals_total": int(signal_mask.sum()),
        "overall_precision": ovr_prec,
        "overall_ci": [ovr_lo, ovr_hi],
        "strong_count": int(strong_mask.sum()),
        "strong_precision": strong_prec,
        "contra": _seg(contra_mask),
        "aligned": _seg(aligned_mask),
        "choppy": _seg(choppy_mask),
    }


def main():
    logger.info("Loading production WF OOS predictions...")
    oos = pd.read_parquet(DIR_OOS).sort_index()
    logger.info("OOS: %d bars, %s ~ %s", len(oos), oos.index[0], oos.index[-1])

    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    logger.info("Config: warmup=%d, strong_frac=%.2f, mod_frac=%.2f, window=%d",
                cfg["warmup_bars"], cfg["strong_top_frac"],
                cfg["moderate_top_frac"], cfg["percentile_window"])

    logger.info("Loading feature cache for regime assignment...")
    feats = pd.read_parquet(FEATURE_CACHE)
    feats.index = feats.index.astype("datetime64[ns, UTC]")
    oos.index = oos.index.astype("datetime64[ns, UTC]")

    # Align regime to OOS index
    feats_aligned = feats.reindex(oos.index, method=None)
    close = feats_aligned["close"]
    regime_full = assign_regime(feats["close"])  # compute on full history
    regime_ser = pd.Series(regime_full, index=feats.index).reindex(oos.index)
    regime_arr = regime_ser.values

    n_by_reg = pd.Series(regime_arr).value_counts().to_dict()
    logger.info("Regime distribution in OOS: %s", n_by_reg)

    pred_ret = oos["pred_ret"].values.astype(float)
    actual_ret = oos["y_path_ret_4h"].values.astype(float)

    ic, _ = spearmanr(pred_ret, actual_ret)
    logger.info("OOS sanity IC: %.4f (expect ~0.18)", ic)

    # Sweep
    results_list = []
    for p in PENALTIES:
        r = evaluate(pred_ret, regime_arr, actual_ret, p, cfg)
        results_list.append(r)
        logger.info(
            "penalty=%-4.1f  sig=%-4d  prec=%.3f CI[%.3f,%.3f]  "
            "contra_n=%-3d contra_prec=%s  aligned_n=%-3d aligned_prec=%s",
            p, r["signals_total"],
            r["overall_precision"] or 0, *(r["overall_ci"] or [0, 0]),
            r["contra"]["n"],
            f'{r["contra"]["precision"]:.3f}' if r["contra"]["precision"] is not None else "  —  ",
            r["aligned"]["n"],
            f'{r["aligned"]["precision"]:.3f}' if r["aligned"]["precision"] is not None else "  —  ",
        )

    # Find penalties on the Pareto frontier (precision vs volume)
    pts = [(r["penalty"], r["overall_precision"] or 0, r["signals_total"])
           for r in results_list if r["overall_precision"] is not None]
    # Sort by signal count descending; note dominating points
    pts_sorted = sorted(pts, key=lambda x: -x[2])
    frontier = []
    best_prec = -1.0
    for pen, prec, n in pts_sorted:
        if prec > best_prec:
            frontier.append({"penalty": pen, "precision": prec, "signals": n})
            best_prec = prec

    # Baseline: penalty=1.0 (no penalty)
    base = next(r for r in results_list if r["penalty"] == 1.0)
    current = next((r for r in results_list if r["penalty"] == 2.5), None)
    best_ovr = max(results_list, key=lambda r: (r["overall_precision"] or 0))

    out = {
        "oos_source": str(DIR_OOS.name),
        "n_oos": len(oos),
        "date_range": [str(oos.index[0]), str(oos.index[-1])],
        "config_used": {k: cfg[k] for k in (
            "warmup_bars", "strong_top_frac", "moderate_top_frac",
            "percentile_window", "fallback",
        )},
        "regime_distribution": {k: int(v) for k, v in n_by_reg.items()},
        "sanity_ic": float(ic),
        "sweep": results_list,
        "pareto_frontier": frontier,
        "summary": {
            "baseline_no_penalty": {
                "penalty": base["penalty"],
                "precision": base["overall_precision"],
                "signals": base["signals_total"],
            },
            "current_production": {
                "penalty": current["penalty"],
                "precision": current["overall_precision"],
                "signals": current["signals_total"],
            } if current else None,
            "best_precision": {
                "penalty": best_ovr["penalty"],
                "precision": best_ovr["overall_precision"],
                "signals": best_ovr["signals_total"],
            },
        },
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info("Results → %s", RESULTS_FILE)


if __name__ == "__main__":
    main()
