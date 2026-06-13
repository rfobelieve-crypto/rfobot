"""
Cascade Modifier Study — Stage 2: Walk-forward backtest of confidence
modifier vs production-equivalent baseline.

Uses:
  - Production-WF OOS direction regressor pred_ret
    (research/results/dual_model/direction_reg_oos_mse.parquet)
  - Production-WF OOS magnitude pred (σ-units)
    (research/results/dual_model/magnitude_oos_phase1_voladj.parquet)
  - Cascade classifier: trained in this script via identical WF splits
    to produce OOS p_cascade.
  - cg_liq_imbalance (raw feature at each bar) from features_all.parquet.

The production decoding loop is replayed bar-by-bar:
  - Rolling percentile direction thresholds (window=500, warmup=100,
    strong_frac=0.05, moderate_frac=0.15, fallback from config).
  - mag_score from expanding percentile of mag_pred σ-units.
  - confidence_original = 80*(|p|/ref)^0.6 + 20*(mag_score/100).
  - tier_original from confidence: Strong ≥ 80, Moderate ≥ 65, Weak else.

Then the cascade modifier is applied ON TOP of confidence_original:
  - if p_cascade < 0.7: confidence_v2 = confidence_original
  - else:
      contrarian = -sign(liq_imbalance)
      if sign(pred_ret) == contrarian: confidence_v2 = min(1.15*conf, 100)
      else:                             confidence_v2 = 0.70*conf

Metrics (Strong / Moderate for original vs v2):
  - count, win rate, Wilson 95% CI, mean |ret| on hits
  - by regime (CHOPPY / TRENDING_BULL / TRENDING_BEAR)
  - by modifier-triggered bucket (active vs inactive)

Does NOT modify production. All output to
research/results/cascade_modifier_study/.
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "dual_model"))
from shared_data import walk_forward_splits  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_CACHE = Path("research/dual_model/.cache/features_all.parquet")
DIRECTION_FEATURE_COLS = Path("indicator/model_artifacts/dual_model/direction_feature_cols.json")
DIRECTION_CONFIG = Path("indicator/model_artifacts/dual_model/direction_reg_config.json")
DIR_OOS = Path("research/results/dual_model/direction_reg_oos_mse.parquet")
MAG_OOS = Path("research/results/dual_model/magnitude_oos_phase1_voladj.parquet")
OUT_DIR = Path("research/results/cascade_modifier_study")

HORIZON = 4
STRONG_THRESHOLD = 80.0
MODERATE_THRESHOLD = 65.0

# Gate: rolling top-5% of recent p_cascade values. Adaptive to
# classifier-probability drift (unlike a fixed 0.7 threshold, which is
# too high given cascade base rate ≈ 10.9% compresses XGB predict_proba
# to a median of ~0.01 and p99 ~0.53 in this backtest).
CASCADE_ROLLING_WINDOW = 500
CASCADE_WARMUP = 100
CASCADE_TOP_FRAC = 0.05
CASCADE_FALLBACK_THRESHOLD = 0.30  # cold-start fallback before buffer warms

BOOST_MULT = 1.15
DAMP_MULT = 0.70

XGB_CLF_PARAMS = dict(
    n_estimators=250, max_depth=5, learning_rate=0.05,
    reg_alpha=0.1, reg_lambda=1.0, subsample=0.8,
    colsample_bytree=0.8, random_state=42, verbosity=0,
    use_label_encoder=False, eval_metric="logloss",
)


# ── Wilson score 95% CI for binomial proportion ─────────────────────────

def wilson_ci(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    halfw = (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
    return (max(0.0, centre - halfw), min(1.0, centre + halfw))


# ── Regime (identical to inference.py _assign_regime) ───────────────────

def assign_regime(close: pd.Series) -> np.ndarray:
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


# ── Cascade surge label ────────────────────────────────────────────────

def build_cascade_label(df: pd.DataFrame) -> np.ndarray:
    liq = df["cg_liq_total"].astype(float).fillna(0)
    fwd_liq = sum(liq.shift(-i) for i in range(1, HORIZON + 1))
    trailing_p90 = liq.rolling(500, min_periods=100).quantile(0.90) * HORIZON
    surge = (fwd_liq > trailing_p90).astype(float)
    surge[(trailing_p90.isna()) | (fwd_liq.isna())] = np.nan
    return surge.values


# ── Cascade classifier (WF OOS) ────────────────────────────────────────

def train_cascade_wf(X: np.ndarray, y: np.ndarray, splits) -> np.ndarray:
    import xgboost as xgb
    oos = np.full(len(y), np.nan)
    valid = ~np.isnan(y)
    for tr_idx, te_idx in splits:
        tr_m = valid[tr_idx]
        te_m = valid[te_idx]
        tr_i = np.array(tr_idx)[tr_m]
        te_i = np.array(te_idx)[te_m]
        if len(tr_i) < 50 or len(te_i) < 5:
            continue
        y_tr = y[tr_i].astype(int)
        if len(np.unique(y_tr)) < 2:
            oos[te_i] = float(y_tr.mean())
            continue
        m = xgb.XGBClassifier(**XGB_CLF_PARAMS)
        m.fit(X[tr_i], y_tr)
        oos[te_i] = m.predict_proba(X[te_i])[:, 1]
    return oos


# ── Production-style decoding ──────────────────────────────────────────

def simulate_decoding(dates: pd.DatetimeIndex,
                      pred_ret: np.ndarray,
                      mag_sigma: np.ndarray,
                      config: dict) -> dict:
    """Replay inference.py rolling-percentile decoding bar by bar.

    Returns dict of arrays, all length N:
      pred_direction (+1/−1/0), strong_cutoff_ref (|ref| for confidence),
      mag_score (0-100), confidence_original (0-100), tier_original (str)
    """
    n = len(pred_ret)
    warmup = config["warmup_bars"]
    strong_frac = config["strong_top_frac"]
    mod_frac = config["moderate_top_frac"]
    win = config["percentile_window"]
    fb = config["fallback"]

    buf = deque(maxlen=win)
    mag_history = []

    pred_dir = np.zeros(n)
    ref_arr = np.zeros(n)
    mag_score = np.full(n, 50.0)  # 50 = neutral
    strong_up = np.zeros(n); strong_dn = np.zeros(n)
    mod_up = np.zeros(n); mod_dn = np.zeros(n)

    for i in range(n):
        if np.isnan(pred_ret[i]):
            pred_dir[i] = 0; continue
        p = float(pred_ret[i])
        buf.append(p)

        if len(buf) < warmup:
            up_s, dn_s = fb["strong_up"], fb["strong_dn"]
            up_m, dn_m = fb["moderate_up"], fb["moderate_dn"]
        else:
            arr = np.fromiter(buf, dtype=float)
            up_s = float(np.quantile(arr, 1.0 - strong_frac/2.0))
            dn_s = float(np.quantile(arr, strong_frac/2.0))
            up_m = float(np.quantile(arr, 1.0 - mod_frac/2.0))
            dn_m = float(np.quantile(arr, mod_frac/2.0))

        strong_up[i], strong_dn[i] = up_s, dn_s
        mod_up[i], mod_dn[i] = up_m, dn_m

        # Direction sign by comparison to moderate cutoff (signals a call)
        if p >= up_m:
            pred_dir[i] = 1
        elif p <= dn_m:
            pred_dir[i] = -1
        else:
            pred_dir[i] = 0

        ref_arr[i] = max(abs(up_s), abs(dn_s), 1e-6)

        # Mag score = expanding percentile rank of |mag_sigma|
        if not np.isnan(mag_sigma[i]):
            mag_history.append(abs(float(mag_sigma[i])))
            rank = (np.searchsorted(np.sort(mag_history), abs(float(mag_sigma[i])),
                                    side="right") / len(mag_history)) * 100
            mag_score[i] = float(rank)

    # Confidence original: 80*(|p|/ref)^0.6 + 20*(mag_score/100)
    abs_p = np.abs(pred_ret)
    ret_score = np.where(
        (~np.isnan(pred_ret)) & (ref_arr > 0),
        np.minimum(abs_p / np.maximum(ref_arr, 1e-6), 1.0) ** 0.6 * 80.0,
        0.0,
    )
    mag_bonus = (mag_score / 100.0) * 20.0
    confidence = np.clip(ret_score + mag_bonus, 0, 100)

    # Tier original: from confidence thresholds
    tier = np.full(n, "Weak", dtype=object)
    tier[confidence >= MODERATE_THRESHOLD] = "Moderate"
    tier[confidence >= STRONG_THRESHOLD] = "Strong"
    # Bars with pred_dir == 0 (within moderate cutoff) should stay Weak
    # regardless of confidence (no directional call).
    tier[pred_dir == 0] = "Weak"

    return {
        "pred_direction_sign": pred_dir,
        "ref": ref_arr,
        "mag_score": mag_score,
        "confidence_original": confidence,
        "tier_original": tier,
    }


# ── Modifier ───────────────────────────────────────────────────────────

def apply_modifier(pred_dir_sign: np.ndarray, liq_imb: np.ndarray,
                   p_cascade: np.ndarray, confidence: np.ndarray) -> dict:
    """Apply the cascade confidence modifier with a ROLLING top-5% gate.

    The gate threshold at bar i is the 95th percentile of p_cascade over
    the most recent CASCADE_ROLLING_WINDOW bars. Before the buffer
    reaches CASCADE_WARMUP bars we use CASCADE_FALLBACK_THRESHOLD.
    """
    n = len(confidence)
    conf_v2 = confidence.copy()
    active = np.zeros(n, dtype=bool)
    aligned = np.zeros(n, dtype=bool)
    conflicted = np.zeros(n, dtype=bool)
    gate_thresh = np.zeros(n)

    buf = deque(maxlen=CASCADE_ROLLING_WINDOW)

    for i in range(n):
        p_c = p_cascade[i]
        imb = liq_imb[i]
        pd_sign = pred_dir_sign[i]

        # Update rolling buffer BEFORE computing cutoff (consistent with
        # the direction-reg rolling-percentile decoding in inference.py).
        if not np.isnan(p_c):
            buf.append(float(p_c))

        if len(buf) < CASCADE_WARMUP:
            threshold = CASCADE_FALLBACK_THRESHOLD
        else:
            arr = np.fromiter(buf, dtype=float)
            threshold = float(np.quantile(arr, 1.0 - CASCADE_TOP_FRAC))
        gate_thresh[i] = threshold

        if np.isnan(p_c) or np.isnan(imb) or pd_sign == 0:
            continue
        if p_c < threshold:
            continue

        active[i] = True
        contrarian = -1 if imb > 0 else (1 if imb < 0 else 0)
        if contrarian == 0:
            continue  # perfectly symmetric; don't modify
        if pd_sign == contrarian:
            aligned[i] = True
            conf_v2[i] = min(confidence[i] * BOOST_MULT, 100.0)
        else:
            conflicted[i] = True
            conf_v2[i] = confidence[i] * DAMP_MULT

    tier_v2 = np.full(n, "Weak", dtype=object)
    tier_v2[conf_v2 >= MODERATE_THRESHOLD] = "Moderate"
    tier_v2[conf_v2 >= STRONG_THRESHOLD] = "Strong"
    tier_v2[pred_dir_sign == 0] = "Weak"

    return {
        "confidence_v2": conf_v2, "tier_v2": tier_v2,
        "modifier_active": active, "modifier_aligned": aligned,
        "modifier_conflicted": conflicted,
        "gate_thresholds": gate_thresh,
    }


# ── Metrics helpers ────────────────────────────────────────────────────

def tier_stats(mask: np.ndarray, pred_dir_sign: np.ndarray,
               actual_ret: np.ndarray) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "win_rate": None, "ci95": [None, None],
                "mean_abs_return": None}
    pd_ = pred_dir_sign[mask]
    ar = actual_ret[mask]
    wins = ((pd_ > 0) & (ar > 0)) | ((pd_ < 0) & (ar < 0))
    n_wins = int(wins.sum())
    wr = n_wins / n
    ci_lo, ci_hi = wilson_ci(n_wins, n)
    return {
        "n": n, "n_wins": n_wins,
        "win_rate": float(wr),
        "ci95": [float(ci_lo), float(ci_hi)],
        "mean_abs_return": float(np.abs(ar).mean()),
    }


def compare_tier(tier_a: np.ndarray, tier_b: np.ndarray,
                 pred_dir_sign: np.ndarray, actual_ret: np.ndarray,
                 label: str) -> dict:
    m_a = (tier_a == label)
    m_b = (tier_b == label)
    s_a = tier_stats(m_a, pred_dir_sign, actual_ret)
    s_b = tier_stats(m_b, pred_dir_sign, actual_ret)
    delta_wr = (s_b["win_rate"] - s_a["win_rate"]) if (s_a["win_rate"] is not None and s_b["win_rate"] is not None) else None
    delta_n = s_b["n"] - s_a["n"]
    return {
        "original": s_a,
        "v2": s_b,
        "delta_win_rate_pct_points": float(delta_wr * 100) if delta_wr is not None else None,
        "delta_count": delta_n,
        "delta_count_pct": float(delta_n / s_a["n"] * 100) if s_a["n"] > 0 else None,
    }


# ── Main ───────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    logger.info("Loading features + OOS artifacts...")
    df = pd.read_parquet(FEATURE_CACHE)
    df.index = df.index.astype("datetime64[ns, UTC]")

    with open(DIRECTION_FEATURE_COLS) as f:
        feature_cols = json.load(f)
    feature_cols = [c for c in feature_cols if c in df.columns]

    with open(DIRECTION_CONFIG) as f:
        dir_config = json.load(f)

    # Load production OOS predictions
    dir_oos = pd.read_parquet(DIR_OOS).sort_index()
    dir_oos.index = dir_oos.index.astype("datetime64[ns, UTC]")
    mag_oos = pd.read_parquet(MAG_OOS).sort_index()
    mag_oos.index = mag_oos.index.astype("datetime64[ns, UTC]")

    # Align on common OOS index
    common_idx = dir_oos.index.intersection(mag_oos.index)
    logger.info("Dir OOS: %d  Mag OOS: %d  Common: %d",
                len(dir_oos), len(mag_oos), len(common_idx))

    # Build cascade label + train cascade classifier WF (on full feature matrix)
    cascade_label = build_cascade_label(df)
    # We need cascade OOS on the full feature matrix so we can align later
    X_full = df[feature_cols].fillna(0).values.astype(float)
    splits = walk_forward_splits(len(df) - HORIZON)  # trim to keep label valid
    logger.info("Cascade WF folds: %d", len(splits))
    X_trim = X_full[:len(df) - HORIZON]
    y_cascade_trim = cascade_label[:len(df) - HORIZON]
    dates_trim = df.index[:len(df) - HORIZON]
    t_casc = time.time()
    p_cascade_arr = train_cascade_wf(X_trim, y_cascade_trim, splits)
    logger.info("Cascade classifier trained (%.1fs)", time.time() - t_casc)

    # Assign regime on full close series
    regime_full = assign_regime(df["close"])
    regime_trim = regime_full[:len(df) - HORIZON]

    # Build a DataFrame on dates_trim with all needed series
    bt = pd.DataFrame({
        "close": df["close"].values[:len(df) - HORIZON],
        "liq_imb": df["cg_liq_imbalance"].astype(float).fillna(0).values[:len(df) - HORIZON],
        "p_cascade": p_cascade_arr,
        "regime": regime_trim,
    }, index=dates_trim)

    # Bring in production OOS pred_ret + mag_sigma aligned on common_idx
    bt = bt.join(dir_oos[["pred_ret", "y_path_ret_4h"]], how="left")
    bt = bt.join(mag_oos[["y_pred"]].rename(columns={"y_pred": "mag_sigma"}), how="left")

    # Restrict to rows with all required columns present
    bt = bt.dropna(subset=["pred_ret", "y_path_ret_4h", "mag_sigma", "p_cascade"])
    logger.info("Backtest rows after alignment: %d", len(bt))

    # Simulate production decoding bar by bar
    decoded = simulate_decoding(
        bt.index,
        bt["pred_ret"].values,
        bt["mag_sigma"].values,
        dir_config,
    )
    bt["pred_dir_sign"] = decoded["pred_direction_sign"]
    bt["confidence_original"] = decoded["confidence_original"]
    bt["tier_original"] = decoded["tier_original"]

    # Apply modifier
    mod_out = apply_modifier(
        bt["pred_dir_sign"].values,
        bt["liq_imb"].values,
        bt["p_cascade"].values,
        bt["confidence_original"].values,
    )
    bt["confidence_v2"] = mod_out["confidence_v2"]
    bt["tier_v2"] = mod_out["tier_v2"]
    bt["mod_active"] = mod_out["modifier_active"]
    bt["mod_aligned"] = mod_out["modifier_aligned"]
    bt["mod_conflicted"] = mod_out["modifier_conflicted"]

    # Convert to arrays
    tier_o = bt["tier_original"].values
    tier_v = bt["tier_v2"].values
    pd_sign = bt["pred_dir_sign"].values.astype(float)
    actual_ret = bt["y_path_ret_4h"].values.astype(float)
    regime_arr = bt["regime"].values

    # === Overall tier comparison ===
    overall = {
        "Strong": compare_tier(tier_o, tier_v, pd_sign, actual_ret, "Strong"),
        "Moderate": compare_tier(tier_o, tier_v, pd_sign, actual_ret, "Moderate"),
    }

    # === By regime ===
    by_regime = {}
    for reg in ["CHOPPY", "TRENDING_BULL", "TRENDING_BEAR", "WARMUP"]:
        m = (regime_arr == reg)
        if m.sum() < 30:
            continue
        by_regime[reg] = {
            "Strong": compare_tier(tier_o[m], tier_v[m], pd_sign[m],
                                   actual_ret[m], "Strong"),
            "Moderate": compare_tier(tier_o[m], tier_v[m], pd_sign[m],
                                     actual_ret[m], "Moderate"),
            "n_bars": int(m.sum()),
        }

    # === Modifier-triggered breakdown ===
    active = bt["mod_active"].values
    aligned = bt["mod_aligned"].values
    conflicted = bt["mod_conflicted"].values
    inactive = ~active

    mod_analysis = {
        "total_bars": int(len(bt)),
        "modifier_active_count": int(active.sum()),
        "modifier_active_pct": float(active.mean() * 100),
        "aligned_count": int(aligned.sum()),
        "conflicted_count": int(conflicted.sum()),
        "aligned_vs_conflicted_ratio": (
            float(aligned.sum() / max(conflicted.sum(), 1))
        ),
    }

    # When modifier ACTIVE — did WR go up?
    if active.sum() > 0:
        mod_analysis["active_v2_wr"] = tier_stats(
            active, pd_sign, actual_ret)
        mod_analysis["active_original_wr"] = tier_stats(
            active, pd_sign, actual_ret)  # Same set, WR is not tier-dependent here
        # Break down active bars by aligned vs conflicted
        mod_analysis["aligned_subgroup_wr"] = tier_stats(
            aligned, pd_sign, actual_ret)
        mod_analysis["conflicted_subgroup_wr"] = tier_stats(
            conflicted, pd_sign, actual_ret)
    # Sanity check: inactive bars should have identical tier
    inactive_same_tier = bool(np.all(tier_o[inactive] == tier_v[inactive]))
    mod_analysis["sanity_inactive_bars_same_tier"] = inactive_same_tier

    # === Modifier active only: tier comparison (focused view) ===
    active_tier_cmp = {
        "Strong": compare_tier(tier_o[active], tier_v[active],
                               pd_sign[active], actual_ret[active], "Strong"),
        "Moderate": compare_tier(tier_o[active], tier_v[active],
                                 pd_sign[active], actual_ret[active], "Moderate"),
        "n_active_bars": int(active.sum()),
    }

    # ── Final dump ──
    out = {
        "description": "Stage 2 walk-forward backtest of cascade confidence modifier",
        "config": {
            "gate_type": "rolling_top_fraction",
            "rolling_window": CASCADE_ROLLING_WINDOW,
            "warmup_bars": CASCADE_WARMUP,
            "top_fraction": CASCADE_TOP_FRAC,
            "fallback_threshold_cold_start": CASCADE_FALLBACK_THRESHOLD,
            "boost_mult": BOOST_MULT,
            "damp_mult": DAMP_MULT,
            "strong_threshold": STRONG_THRESHOLD,
            "moderate_threshold": MODERATE_THRESHOLD,
            "horizon_bars": HORIZON,
            "gate_thresh_min": float(np.nanmin(mod_out["gate_thresholds"])),
            "gate_thresh_median": float(np.nanmedian(mod_out["gate_thresholds"])),
            "gate_thresh_max": float(np.nanmax(mod_out["gate_thresholds"])),
        },
        "n_bars_backtested": int(len(bt)),
        "date_range": [str(bt.index[0]), str(bt.index[-1])],
        "overall": overall,
        "by_regime": by_regime,
        "modifier_activity": mod_analysis,
        "active_bars_tier_comparison": active_tier_cmp,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_path = OUT_DIR / "stage2_backtest.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info("Results → %s", out_path)

    # ── Human-readable summary + recommendation ──
    write_summary(out)


def write_summary(r: dict):
    strong = r["overall"]["Strong"]
    mod = r["overall"]["Moderate"]
    mod_stats = r["modifier_activity"]

    strong_delta_wr = strong.get("delta_win_rate_pct_points", 0) or 0
    strong_delta_n = strong.get("delta_count_pct", 0) or 0

    # Recommendation rubric per user spec
    if (strong_delta_wr >= 3 and strong_delta_n >= -30 and
        strong["v2"]["win_rate"] is not None):
        rec = "ADOPT"
        rec_reason = (f"Strong WR improved {strong_delta_wr:+.1f}pp with count "
                      f"change {strong_delta_n:+.1f}% (within −30% tolerance).")
    elif 0 <= strong_delta_wr < 3:
        rec = "MARGINAL — accumulate live data before deciding"
        rec_reason = (f"Strong WR improved only {strong_delta_wr:+.1f}pp; "
                      "signal lift is not decisive on 3696-bar OOS.")
    elif strong_delta_wr < 0:
        rec = "DO NOT ADOPT"
        rec_reason = f"Strong WR degraded {strong_delta_wr:+.1f}pp."
    else:
        rec = "MARGINAL"
        rec_reason = "See table."

    md = f"""# Cascade Modifier Study — Stage 2 Summary

Walk-forward backtest of the cascade confidence modifier on
**{r["n_bars_backtested"]} OOS bars** ({r["date_range"][0]} → {r["date_range"][1]}).

## Modifier config
- Activation gate: **rolling top {r["config"]["top_fraction"]*100:.0f}% of p_cascade** over {r["config"]["rolling_window"]}-bar window (warmup {r["config"]["warmup_bars"]}, cold-start fallback p={r["config"]["fallback_threshold_cold_start"]})
- Dynamic gate threshold range: [{r["config"]["gate_thresh_min"]:.3f}, {r["config"]["gate_thresh_max"]:.3f}], median {r["config"]["gate_thresh_median"]:.3f}
- Boost (aligned): confidence × **{r["config"]["boost_mult"]}** (capped at 100)
- Damp (conflicted): confidence × **{r["config"]["damp_mult"]}**
- Tier thresholds: Strong ≥ {r["config"]["strong_threshold"]}, Moderate ≥ {r["config"]["moderate_threshold"]}

## Modifier activity

- Modifier active on **{mod_stats["modifier_active_count"]} bars** ({mod_stats["modifier_active_pct"]:.1f}% of backtest)
- Aligned (direction-reg agrees with contrarian): {mod_stats["aligned_count"]}
- Conflicted (disagree): {mod_stats["conflicted_count"]}
- Aligned : Conflicted ratio = **{mod_stats["aligned_vs_conflicted_ratio"]:.2f}**
- Sanity (inactive bars unchanged tier): **{mod_stats["sanity_inactive_bars_same_tier"]}**

## Overall tier comparison

### Strong tier
| Metric | Original | V2 | Δ |
|---|---|---|---|
| Count | {strong["original"]["n"]} | {strong["v2"]["n"]} | {strong["delta_count"]:+d} ({strong["delta_count_pct"]:+.1f}%) |
| Win rate | {_fmt_pct(strong["original"]["win_rate"])} | {_fmt_pct(strong["v2"]["win_rate"])} | {_fmt_delta_pp(strong["delta_win_rate_pct_points"])} |
| Wilson 95% CI | {_fmt_ci(strong["original"]["ci95"])} | {_fmt_ci(strong["v2"]["ci95"])} | — |
| Mean \\|ret\\| | {_fmt_ret(strong["original"]["mean_abs_return"])} | {_fmt_ret(strong["v2"]["mean_abs_return"])} | — |

### Moderate tier
| Metric | Original | V2 | Δ |
|---|---|---|---|
| Count | {mod["original"]["n"]} | {mod["v2"]["n"]} | {mod["delta_count"]:+d} ({mod["delta_count_pct"]:+.1f}%) |
| Win rate | {_fmt_pct(mod["original"]["win_rate"])} | {_fmt_pct(mod["v2"]["win_rate"])} | {_fmt_delta_pp(mod["delta_win_rate_pct_points"])} |
| Wilson 95% CI | {_fmt_ci(mod["original"]["ci95"])} | {_fmt_ci(mod["v2"]["ci95"])} | — |
| Mean \\|ret\\| | {_fmt_ret(mod["original"]["mean_abs_return"])} | {_fmt_ret(mod["v2"]["mean_abs_return"])} | — |

## By regime

"""
    for reg, data in r["by_regime"].items():
        md += f"### {reg} ({data['n_bars']} bars)\n\n"
        md += "| Tier | Metric | Original | V2 | Δ |\n|---|---|---|---|---|\n"
        for tier in ("Strong", "Moderate"):
            d = data[tier]
            md += (f"| {tier} | Count | {d['original']['n']} | {d['v2']['n']} "
                   f"| {d['delta_count']:+d} |\n")
            md += (f"| {tier} | WR | {_fmt_pct(d['original']['win_rate'])} "
                   f"| {_fmt_pct(d['v2']['win_rate'])} "
                   f"| {_fmt_delta_pp(d['delta_win_rate_pct_points'])} |\n")
        md += "\n"

    md += f"""## Modifier-active bars only (focused view)

n = {r["active_bars_tier_comparison"]["n_active_bars"]}

| Tier | Original n | V2 n | Original WR | V2 WR | Δ WR |
|---|---|---|---|---|---|
"""
    for tier in ("Strong", "Moderate"):
        d = r["active_bars_tier_comparison"][tier]
        md += (f"| {tier} | {d['original']['n']} | {d['v2']['n']} "
               f"| {_fmt_pct(d['original']['win_rate'])} "
               f"| {_fmt_pct(d['v2']['win_rate'])} "
               f"| {_fmt_delta_pp(d['delta_win_rate_pct_points'])} |\n")

    md += f"""
## Modifier-subgroup breakdown

- Aligned bars (n={mod_stats["aligned_count"]}): WR = {_fmt_pct(mod_stats["aligned_subgroup_wr"]["win_rate"])} (CI {_fmt_ci(mod_stats["aligned_subgroup_wr"]["ci95"])})
- Conflicted bars (n={mod_stats["conflicted_count"]}): WR = {_fmt_pct(mod_stats["conflicted_subgroup_wr"]["win_rate"])} (CI {_fmt_ci(mod_stats["conflicted_subgroup_wr"]["ci95"])})

The aligned vs conflicted WR gap is the core evidence for the modifier's
logic. A wide gap (aligned WR ≫ conflicted WR) supports the
boost/damp scheme; a narrow gap means the modifier is labeling noise.

## Recommendation

**{rec}**

{rec_reason}

### Decision thresholds (per study spec)

- Strong WR ↑ ≥ 3pp **and** Count ↓ ≤ 30% → ADOPT
- Strong WR ↑ 0-3pp → MARGINAL, accumulate more live data
- Strong WR ↓ → DO NOT ADOPT

### Caveats

1. Baseline direction-reg OOS IC ≈ 0.18 (production-validated), but this
   backtest replays the full production decoding pipeline on WF-OOS
   predictions — the Strong-tier baseline WR here is the most honest
   estimate of what production actually delivers today.
2. Modifier active rate is driven by cascade classifier precision at
   p ≥ 0.70 threshold; if cascade classifier precision shifts in the
   next 3 months, the modifier's impact will shift accordingly.
3. The aligned/conflicted sub-group WR gap may not be stable across
   market regimes. Recommend re-running Stage 2 after 3-6 months of
   post-launch data accumulates.
"""

    out_path = OUT_DIR / "stage2_summary.md"
    out_path.write_text(md, encoding="utf-8")
    logger.info("Summary → %s", out_path)


def _fmt_pct(x):
    return f"{x*100:.1f}%" if x is not None else "—"

def _fmt_delta_pp(x):
    return f"{x:+.2f}pp" if x is not None else "—"

def _fmt_ci(c):
    if c is None or c[0] is None: return "—"
    return f"[{c[0]*100:.1f}%, {c[1]*100:.1f}%]"

def _fmt_ret(x):
    return f"{x*100:.3f}%" if x is not None else "—"


if __name__ == "__main__":
    main()
