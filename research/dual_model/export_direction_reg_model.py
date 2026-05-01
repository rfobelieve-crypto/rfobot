"""
Export the winning direction-regression model to production artifacts.

Production decoding is ROLLING PERCENTILE (not fixed ± threshold):
  - At inference time, the engine maintains a deque of the most recent
    signed predictions (dir_pred_history).
  - Strong signal fires when the current prediction is in the top
    strong_top_frac (default 5%) of the deque, split symmetrically between
    upper and lower tails (so ±2.5% each side).
  - Moderate tier uses moderate_top_frac (default 15%).

This script:
  1. Backs up the binary direction model (rollback path).
  2. Trains the regression model on FULL history (no WF hold-out).
  3. Loads the walk-forward OOS parquet and calibrates fallback ± thresholds
     (= top-strong_top_frac symmetric quantiles of all WF OOS predictions)
     for cold-start use when dir_pred_history < warmup_bars.
  4. Seeds training_stats.json's "dir_pred_history" with the last 500
     IN-SAMPLE predictions of the freshly-trained PRODUCTION MODEL on its
     own training data, so the rolling-percentile decoder starts warm with
     a buffer that lives on the same scale as live predictions.

     ⚠ Do NOT seed from WF OOS predictions: WF fold models train on a small
     subset and produce ~3.5x narrower variance than the production model,
     which biases the rolling-percentile thresholds and causes runaway
     same-side signals (mistake.md 2026-04-19).

  5. Does NOT touch magnitude_xgb.json / magnitude_feature_cols.json /
     magnitude_config.json.

Usage:
    python -m research.dual_model.export_direction_reg_model
    python -m research.dual_model.export_direction_reg_model --objective huber
"""
from __future__ import annotations

import sys
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import (
    load_and_cache_data, RESULTS_DIR,
)
from research.dual_model.build_direction_reg_labels import build_direction_reg_labels
from research.dual_model.direction_features_v2 import (
    FULL_DIRECTION, filter_available,
)
from research.dual_model.train_direction_reg_4h import (
    BASE_PARAMS, OBJECTIVE_MAP,
)

ARTIFACT_DIR = PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model"


def _calibrate_fallback(wf_oos_path: Path, strong_frac: float,
                        mod_frac: float) -> dict:
    """
    Read WF OOS predictions and derive fallback ± thresholds (symmetric
    quantiles).

    The fallback thresholds are what the engine falls back to when its live
    dir_pred_history buffer has fewer samples than warmup_bars. WF OOS is
    the right source for thresholds because it represents how the model
    behaves on truly unseen data — fallback should be conservative.
    """
    oos = pd.read_parquet(wf_oos_path)
    pred = oos["pred_ret"].values.astype(float)

    fallback = dict(
        strong_up=float(np.quantile(pred, 1.0 - strong_frac / 2.0)),
        strong_dn=float(np.quantile(pred, strong_frac / 2.0)),
        moderate_up=float(np.quantile(pred, 1.0 - mod_frac / 2.0)),
        moderate_dn=float(np.quantile(pred, mod_frac / 2.0)),
    )
    return fallback


def _build_warmup_buffer(model: xgb.XGBRegressor, X: pd.DataFrame,
                         y_index: pd.Index, n_bars: int = 500) -> list[float]:
    """
    Seed the rolling-percentile warmup buffer with the freshly-trained
    PRODUCTION model's in-sample predictions on its own training data,
    time-ordered, last `n_bars` rows.

    Why in-sample (not WF OOS): the runtime decoder compares each live
    prediction against the buffer's percentile thresholds. WF OOS fold
    models train on a small subset and produce ~3.5x narrower variance
    than the production model — using their predictions as the buffer
    biases the thresholds and pushes nearly every live prediction into a
    Strong tail (mistake.md 2026-04-19).

    Caller must verify: ratio of buffer_std to recent-live-pred std should
    sit in [0.5, 2.0].
    """
    in_sample = pd.Series(
        model.predict(X).astype(float),
        index=y_index,
    ).sort_index()
    warmup = in_sample.tail(n_bars).tolist()
    return warmup


def export(objective: str, strong_frac: float, mod_frac: float,
           pct_window: int, warmup_bars: int, n_estimators: int):
    if objective not in OBJECTIVE_MAP:
        raise ValueError(f"Unknown objective: {objective}")

    # ── 1. Backup current direction artifacts ──────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = ARTIFACT_DIR.parent / f"dual_model_backup_direction_binary_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("direction_xgb.json", "direction_feature_cols.json"):
        src = ARTIFACT_DIR / fname
        if src.exists():
            shutil.copy2(src, backup_dir / fname)
    print(f"[backup] binary direction artifacts -> {backup_dir}")

    # ── 2. Load data, build labels ─────────────────────────────────────
    df = load_and_cache_data()
    labels = build_direction_reg_labels(df)
    df = df.copy()
    df["y_path_ret_4h"] = labels["y_path_ret_4h"]

    features = filter_available(FULL_DIRECTION, list(df.columns))
    print(f"[data] features: {len(features)}/{len(FULL_DIRECTION)} available")

    mask = df["y_path_ret_4h"].notna()
    X = df.loc[mask, features].fillna(0)
    y = df.loc[mask, "y_path_ret_4h"].astype(float).values
    y_index = df.loc[mask].index
    print(f"[data] training rows: {len(y)}   "
          f"target mu/sigma = {y.mean():+.5f} / {y.std():.5f}")

    # ── 3. Fit on full history ─────────────────────────────────────────
    params = BASE_PARAMS.copy()
    params.update(OBJECTIVE_MAP[objective])
    params.pop("early_stopping_rounds", None)
    params["n_estimators"] = n_estimators

    model = xgb.XGBRegressor(**params)
    model.fit(X, y, verbose=False)
    print(f"[train] fit complete: {objective}  n_estimators={n_estimators}")

    in_sample_pred = model.predict(X)
    in_sample_mae = float(np.mean(np.abs(in_sample_pred - y)))
    in_sample_corr = float(np.corrcoef(in_sample_pred, y)[0, 1])
    in_sample_std = float(np.std(in_sample_pred))
    print(f"[train] in-sample MAE={in_sample_mae:.5f}  "
          f"corr={in_sample_corr:+.4f}  std={in_sample_std:.5f}  "
          f"(diagnostic only, NOT OOS)")

    # ── 4a. Calibrate fallback thresholds from WF OOS ──────────────────
    wf_path = RESULTS_DIR / f"direction_reg_oos_{objective}.parquet"
    if not wf_path.exists():
        raise FileNotFoundError(
            f"{wf_path} not found — run train_direction_reg_4h first")
    fallback = _calibrate_fallback(wf_path, strong_frac, mod_frac)
    print(f"[calib] fallback strong ±  up={fallback['strong_up']:+.5f}  "
          f"dn={fallback['strong_dn']:+.5f}")
    print(f"[calib] fallback mod    ±  up={fallback['moderate_up']:+.5f}  "
          f"dn={fallback['moderate_dn']:+.5f}")

    # ── 4b. Seed warmup buffer from PRODUCTION model in-sample preds ──
    warmup_buffer = _build_warmup_buffer(model, X, y_index, n_bars=500)
    buf_std = float(np.std(warmup_buffer))
    wf_pred_std = float(np.std(pd.read_parquet(wf_path)["pred_ret"].values))
    scale_ratio = buf_std / wf_pred_std if wf_pred_std > 0 else float("inf")
    print(f"[calib] warmup buffer: {len(warmup_buffer)} signed preds  "
          f"std={buf_std:.5f}")
    print(f"[calib] buffer/WF-OOS std ratio = {scale_ratio:.2f}x  "
          "(prod model typically 2-4x wider than WF folds; this is expected)")
    if scale_ratio < 1.5:
        print("[WARN] buffer std ratio < 1.5x — production model may be "
              "underfitting OR WF folds were trained on too-similar splits. "
              "Verify manually before deploying.")

    # ── 5. Write artifacts ─────────────────────────────────────────────
    model_path = ARTIFACT_DIR / "direction_xgb.json"
    feats_path = ARTIFACT_DIR / "direction_feature_cols.json"
    cfg_path = ARTIFACT_DIR / "direction_reg_config.json"
    stats_path = ARTIFACT_DIR / "training_stats.json"

    model.save_model(str(model_path))
    with open(feats_path, "w") as f:
        json.dump(features, f, indent=2)

    config = {
        "model_type": "regression",
        "objective": OBJECTIVE_MAP[objective]["objective"],
        "target": "y_path_ret_4h = mean(close[t+1..t+4])/close[t] - 1 (TWAP)",
        "horizon_bars": 4,
        "decoding": "rolling_percentile",
        "percentile_window": pct_window,
        "strong_top_frac": strong_frac,
        "moderate_top_frac": mod_frac,
        "warmup_bars": warmup_bars,
        "fallback": fallback,
        "use_mag_gate": False,
        "n_train_samples": int(len(y)),
        "n_features": len(features),
        "n_estimators": n_estimators,
        "trained_at": datetime.now().isoformat(),
        "in_sample_mae": in_sample_mae,
        "in_sample_corr": in_sample_corr,
    }
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    # Merge warmup buffer into training_stats.json (preserve existing keys)
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
    else:
        stats = {}
    stats["dir_pred_history"] = warmup_buffer
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[export] wrote {model_path.name}")
    print(f"[export] wrote {feats_path.name}  ({len(features)} features)")
    print(f"[export] wrote {cfg_path.name}")
    print(f"[export] seeded {stats_path.name}::dir_pred_history "
          f"({len(warmup_buffer)} bars)")
    print("[export] magnitude model UNTOUCHED")
    print(f"\nRollback: delete {cfg_path.name} + restore from "
          f"{backup_dir.name}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Export direction-regression model to production")
    p.add_argument("--objective", default="mse", choices=["mse", "huber"])
    p.add_argument("--strong-top-frac", type=float, default=0.05,
                   help="Strong tier: top X symmetric percentile (default 0.05)")
    p.add_argument("--moderate-top-frac", type=float, default=0.15,
                   help="Moderate tier: top X symmetric percentile (default 0.15)")
    p.add_argument("--percentile-window", type=int, default=500,
                   help="Rolling buffer size (default 500)")
    p.add_argument("--warmup-bars", type=int, default=100,
                   help="Min buffer size before percentile mode kicks in")
    p.add_argument("--n-estimators", type=int, default=250,
                   help="Full-fit tree count (default 250)")
    args = p.parse_args()
    export(
        objective=args.objective,
        strong_frac=args.strong_top_frac,
        mod_frac=args.moderate_top_frac,
        pct_window=args.percentile_window,
        warmup_bars=args.warmup_bars,
        n_estimators=args.n_estimators,
    )
