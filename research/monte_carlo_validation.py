"""
Monte Carlo Validation Suite — three independent simulations.

1. Permutation Test:  Shuffle labels N times, re-run walk-forward → p-value for IC/AUC
2. Future Path Sim:   Given current signal distribution, simulate N future equity paths
3. Bootstrap Robustness: Resample training data with replacement, retrain → prediction variance

Uses the current v7 Direction Regressor (136 features, MSE objective).
"""
from __future__ import annotations

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import (
    load_and_cache_data, walk_forward_splits, get_available_features, ensure_dirs,
    RESULTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ── Production model config ──────────────────────────────────────────
ARTIFACT_DIR = PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model"

def _load_prod_feature_cols() -> list[str]:
    with open(ARTIFACT_DIR / "direction_feature_cols.json") as f:
        return json.load(f)

XGB_PARAMS = dict(
    objective="reg:squarederror",
    max_depth=4,
    learning_rate=0.05,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=10,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0,
    n_jobs=-1,
)

HORIZON = 4
TARGET_COL = "y_path_ret_4h"


def _build_target(df: pd.DataFrame) -> pd.Series:
    """Build 4h TWAP path return target."""
    closes = df["close"].values.astype(float)
    n = len(closes)
    y = np.full(n, np.nan)
    for i in range(n - HORIZON):
        y[i] = np.mean(closes[i + 1 : i + 1 + HORIZON]) / closes[i] - 1
    return pd.Series(y, index=df.index, name=TARGET_COL)


# ═══════════════════════════════════════════════════════════════════════
# 1. PERMUTATION TEST — statistical significance of model signal
# ═══════════════════════════════════════════════════════════════════════

def _wf_regression_eval(df, feats, y, splits):
    """Run walk-forward regression, return (IC, directional_accuracy)."""
    all_pred, all_actual = [], []
    for train_idx, test_idx in splits:
        X_tr, y_tr = df[feats].iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = df[feats].iloc[test_idx], y.iloc[test_idx]
        mask_tr = y_tr.notna()
        mask_te = y_te.notna()
        if mask_tr.sum() < 50 or mask_te.sum() < 5:
            continue
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr[mask_tr], y_tr[mask_tr])
        pred = model.predict(X_te[mask_te])
        all_pred.extend(pred)
        all_actual.extend(y_te[mask_te].values)

    all_pred = np.array(all_pred)
    all_actual = np.array(all_actual)
    ic = stats.spearmanr(all_pred, all_actual).correlation
    dir_acc = np.mean(np.sign(all_pred) == np.sign(all_actual))
    return ic, dir_acc


def run_permutation_test(df, feats, y, splits, n_perm=200, seed=42):
    """Permutation test: shuffle y, re-run WF, build null distribution."""
    logger.info("═══ Permutation Test (%d iterations) ═══", n_perm)

    # Real performance
    real_ic, real_acc = _wf_regression_eval(df, feats, y, splits)
    logger.info("Real IC=%.4f  DirAcc=%.3f", real_ic, real_acc)

    rng = np.random.RandomState(seed)
    null_ics, null_accs = [], []

    for i in range(n_perm):
        # Shuffle y (break predictive relationship, keep distribution)
        y_shuffled = y.copy()
        valid_mask = y_shuffled.notna()
        vals = y_shuffled[valid_mask].values.copy()
        rng.shuffle(vals)
        y_shuffled[valid_mask] = vals

        ic_i, acc_i = _wf_regression_eval(df, feats, y_shuffled, splits)
        null_ics.append(ic_i)
        null_accs.append(acc_i)
        if (i + 1) % 20 == 0:
            logger.info("  perm %d/%d done (null IC mean=%.4f)", i + 1, n_perm, np.mean(null_ics))

    null_ics = np.array(null_ics)
    null_accs = np.array(null_accs)

    p_ic = np.mean(null_ics >= real_ic)
    p_acc = np.mean(null_accs >= real_acc)
    z_ic = (real_ic - null_ics.mean()) / (null_ics.std() + 1e-9)
    z_acc = (real_acc - null_accs.mean()) / (null_accs.std() + 1e-9)

    result = {
        "test": "permutation",
        "n_permutations": n_perm,
        "real_ic": float(real_ic),
        "real_dir_acc": float(real_acc),
        "null_ic_mean": float(null_ics.mean()),
        "null_ic_std": float(null_ics.std()),
        "null_acc_mean": float(null_accs.mean()),
        "null_acc_std": float(null_accs.std()),
        "p_ic": float(p_ic),
        "p_acc": float(p_acc),
        "z_ic": float(z_ic),
        "z_acc": float(z_acc),
        "ic_significant": bool(p_ic < 0.05),
        "acc_significant": bool(p_acc < 0.05),
        "null_ics": null_ics.tolist(),
        "null_accs": null_accs.tolist(),
    }

    logger.info("  IC  p=%.4f z=%.2f %s", p_ic, z_ic, "✅ SIGNIFICANT" if p_ic < 0.05 else "❌")
    logger.info("  Acc p=%.4f z=%.2f %s", p_acc, z_acc, "✅ SIGNIFICANT" if p_acc < 0.05 else "❌")
    return result


# ═══════════════════════════════════════════════════════════════════════
# 2. FUTURE PATH SIMULATION — Monte Carlo equity paths
# ═══════════════════════════════════════════════════════════════════════

def run_future_path_sim(oos_df: pd.DataFrame, n_paths=10000, n_signals=100, seed=42):
    """
    Simulate future equity paths based on empirical signal outcomes.

    Uses the WF OOS results: for each simulated signal, sample a real
    outcome from the OOS distribution (preserving the correlation between
    prediction confidence and actual return).
    """
    logger.info("═══ Future Path Simulation (%d paths × %d signals) ═══", n_paths, n_signals)

    pred = oos_df["pred_ret"].values
    actual = oos_df["y_path_ret_4h"].values

    # Split into Strong/Moderate by percentile (matching production thresholds)
    abs_pred = np.abs(pred)
    p975 = np.percentile(abs_pred, 97.5)  # Strong threshold
    p925 = np.percentile(abs_pred, 92.5)  # Moderate threshold

    strong_mask = abs_pred >= p975
    moderate_mask = (abs_pred >= p925) & (abs_pred < p975)
    signal_mask = strong_mask | moderate_mask

    strong_returns = actual[strong_mask]
    moderate_returns = actual[moderate_mask]
    signal_returns = actual[signal_mask]
    signal_pred_signs = np.sign(pred[signal_mask])
    signal_actual_signs = np.sign(actual[signal_mask])

    # Directional returns: positive if prediction direction was correct
    directional_returns = np.abs(signal_returns) * np.where(
        signal_pred_signs == signal_actual_signs, 1, -1
    )

    logger.info("  Strong signals: %d (WR %.1f%%)",
                len(strong_returns),
                np.mean(np.sign(pred[strong_mask]) == np.sign(actual[strong_mask])) * 100)
    logger.info("  Moderate signals: %d (WR %.1f%%)",
                len(moderate_returns),
                np.mean(np.sign(pred[moderate_mask]) == np.sign(actual[moderate_mask])) * 100 if len(moderate_returns) > 0 else 0)

    rng = np.random.RandomState(seed)
    paths = np.zeros((n_paths, n_signals + 1))

    for p in range(n_paths):
        # Sample with replacement from directional returns
        sampled = rng.choice(directional_returns, size=n_signals, replace=True)
        paths[p, 1:] = np.cumsum(sampled)

    # Statistics
    final_returns = paths[:, -1]
    percentiles = {
        "p5": float(np.percentile(final_returns, 5)),
        "p25": float(np.percentile(final_returns, 25)),
        "p50": float(np.percentile(final_returns, 50)),
        "p75": float(np.percentile(final_returns, 75)),
        "p95": float(np.percentile(final_returns, 95)),
    }

    # Max drawdown distribution
    max_drawdowns = []
    for p in range(n_paths):
        cummax = np.maximum.accumulate(paths[p])
        dd = paths[p] - cummax
        max_drawdowns.append(float(dd.min()))
    max_drawdowns = np.array(max_drawdowns)

    # Probability of profit
    prob_profit = float(np.mean(final_returns > 0))

    # Sharpe-like ratio (per signal)
    signal_mean = float(np.mean(directional_returns))
    signal_std = float(np.std(directional_returns))
    sharpe_per_signal = signal_mean / signal_std if signal_std > 0 else 0

    result = {
        "test": "future_path_simulation",
        "n_paths": n_paths,
        "n_signals_per_path": n_signals,
        "source_signals": int(len(signal_returns)),
        "source_strong": int(strong_mask.sum()),
        "source_moderate": int(moderate_mask.sum()),
        "empirical_wr": float(np.mean(signal_pred_signs == signal_actual_signs)),
        "empirical_mean_return": signal_mean,
        "empirical_std_return": signal_std,
        "sharpe_per_signal": sharpe_per_signal,
        "prob_profit_after_n_signals": prob_profit,
        "final_return_percentiles": percentiles,
        "max_drawdown_percentiles": {
            "p5": float(np.percentile(max_drawdowns, 5)),
            "p25": float(np.percentile(max_drawdowns, 25)),
            "p50": float(np.percentile(max_drawdowns, 50)),
        },
    }

    logger.info("  Prob(profit after %d signals): %.1f%%", n_signals, prob_profit * 100)
    logger.info("  Median cumulative return: %.2f%%", percentiles["p50"] * 100)
    logger.info("  5th-95th range: [%.2f%%, %.2f%%]",
                percentiles["p5"] * 100, percentiles["p95"] * 100)
    logger.info("  Median max drawdown: %.2f%%", np.percentile(max_drawdowns, 50) * 100)
    return result


# ═══════════════════════════════════════════════════════════════════════
# 3. BOOTSTRAP ROBUSTNESS — model prediction stability
# ═══════════════════════════════════════════════════════════════════════

def run_bootstrap_robustness(df, feats, y, n_boot=100, test_frac=0.2, seed=42):
    """
    Bootstrap resample training data, retrain model each time,
    predict on a held-out test set → measure prediction variance.

    High variance = model is sensitive to training data (fragile).
    Low variance = model is robust (stable signal).
    """
    logger.info("═══ Bootstrap Robustness (%d resamples) ═══", n_boot)

    valid = y.notna()
    X_valid = df[feats][valid]
    y_valid = y[valid]
    n = len(y_valid)

    # Fixed test set: last 20% (time-respecting)
    split_idx = int(n * (1 - test_frac))
    X_test = X_valid.iloc[split_idx:]
    y_test = y_valid.iloc[split_idx:]
    X_train_full = X_valid.iloc[:split_idx]
    y_train_full = y_valid.iloc[:split_idx]
    n_train = len(y_train_full)

    logger.info("  Train: %d bars, Test: %d bars", n_train, len(y_test))

    rng = np.random.RandomState(seed)
    all_preds = np.zeros((n_boot, len(y_test)))
    all_ics = []
    all_accs = []

    for i in range(n_boot):
        # Bootstrap resample training data (with replacement)
        boot_idx = rng.choice(n_train, size=n_train, replace=True)
        X_boot = X_train_full.iloc[boot_idx]
        y_boot = y_train_full.iloc[boot_idx]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_boot, y_boot)
        pred = model.predict(X_test)
        all_preds[i] = pred

        ic = stats.spearmanr(pred, y_test.values).correlation
        acc = np.mean(np.sign(pred) == np.sign(y_test.values))
        all_ics.append(ic)
        all_accs.append(acc)

        if (i + 1) % 20 == 0:
            logger.info("  boot %d/%d done", i + 1, n_boot)

    all_ics = np.array(all_ics)
    all_accs = np.array(all_accs)

    # Prediction stability: how much do individual predictions vary?
    pred_std_per_bar = all_preds.std(axis=0)  # std across boots for each test bar
    pred_mean_per_bar = all_preds.mean(axis=0)

    # Signal flip rate: how often does prediction sign change across boots?
    pred_signs = np.sign(all_preds)
    mode_sign = np.sign(pred_mean_per_bar)
    flip_rate = np.mean(pred_signs != mode_sign[np.newaxis, :])

    # Coefficient of variation of predictions
    mean_abs_pred = np.mean(np.abs(pred_mean_per_bar))
    mean_pred_std = np.mean(pred_std_per_bar)
    cv = mean_pred_std / mean_abs_pred if mean_abs_pred > 0 else float("inf")

    result = {
        "test": "bootstrap_robustness",
        "n_bootstrap": n_boot,
        "n_train": n_train,
        "n_test": len(y_test),
        "ic_mean": float(all_ics.mean()),
        "ic_std": float(all_ics.std()),
        "ic_ci_95": [float(np.percentile(all_ics, 2.5)), float(np.percentile(all_ics, 97.5))],
        "dir_acc_mean": float(all_accs.mean()),
        "dir_acc_std": float(all_accs.std()),
        "dir_acc_ci_95": [float(np.percentile(all_accs, 2.5)), float(np.percentile(all_accs, 97.5))],
        "pred_std_mean": float(mean_pred_std),
        "pred_cv": float(cv),
        "signal_flip_rate": float(flip_rate),
        "interpretation": {
            "flip_rate": "good" if flip_rate < 0.15 else ("acceptable" if flip_rate < 0.25 else "unstable"),
            "cv": "good" if cv < 0.5 else ("acceptable" if cv < 1.0 else "high_variance"),
            "ic_stability": "good" if all_ics.std() < 0.03 else ("acceptable" if all_ics.std() < 0.06 else "unstable"),
        },
    }

    logger.info("  IC: %.4f ± %.4f  [%.4f, %.4f]",
                all_ics.mean(), all_ics.std(),
                np.percentile(all_ics, 2.5), np.percentile(all_ics, 97.5))
    logger.info("  DirAcc: %.3f ± %.3f", all_accs.mean(), all_accs.std())
    logger.info("  Signal flip rate: %.1f%% (%s)", flip_rate * 100, result["interpretation"]["flip_rate"])
    logger.info("  Prediction CV: %.2f (%s)", cv, result["interpretation"]["cv"])
    return result


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Monte Carlo Validation Suite")
    ap.add_argument("--perm", type=int, default=200, help="Permutation iterations (default 200)")
    ap.add_argument("--paths", type=int, default=10000, help="Future path simulations (default 10000)")
    ap.add_argument("--boot", type=int, default=100, help="Bootstrap resamples (default 100)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-perm", action="store_true", help="Skip permutation test (slow)")
    ap.add_argument("--skip-path", action="store_true", help="Skip future path sim")
    ap.add_argument("--skip-boot", action="store_true", help="Skip bootstrap robustness")
    args = ap.parse_args()

    ensure_dirs()

    # Load data
    logger.info("Loading data...")
    df = load_and_cache_data(limit=4500)
    prod_feats = _load_prod_feature_cols()
    available = [f for f in prod_feats if f in df.columns]
    missing = [f for f in prod_feats if f not in df.columns]
    if missing:
        logger.warning("Missing %d features: %s", len(missing), missing[:5])
    logger.info("Using %d / %d production features", len(available), len(prod_feats))

    y = _build_target(df)
    df[TARGET_COL] = y

    results = {"timestamp": datetime.utcnow().isoformat(), "n_bars": len(df), "n_features": len(available)}

    # 1. Permutation test
    if not args.skip_perm:
        splits = list(walk_forward_splits(len(df)))
        logger.info("Walk-forward splits: %d folds", len(splits))
        results["permutation"] = run_permutation_test(
            df, available, y, splits, n_perm=args.perm, seed=args.seed
        )
    else:
        logger.info("Skipping permutation test")

    # 2. Future path simulation
    if not args.skip_path:
        oos_path = RESULTS_DIR / "direction_reg_oos_mse.parquet"
        if oos_path.exists():
            oos_df = pd.read_parquet(oos_path)
            results["future_paths"] = run_future_path_sim(
                oos_df, n_paths=args.paths, n_signals=100, seed=args.seed
            )
        else:
            logger.warning("OOS parquet not found: %s", oos_path)
    else:
        logger.info("Skipping future path simulation")

    # 3. Bootstrap robustness
    if not args.skip_boot:
        results["bootstrap"] = run_bootstrap_robustness(
            df, available, y, n_boot=args.boot, seed=args.seed
        )
    else:
        logger.info("Skipping bootstrap robustness")

    # Save results
    out_path = PROJECT_ROOT / "research" / "results" / "monte_carlo_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", out_path)

    # Print summary
    print("\n" + "=" * 60)
    print("MONTE CARLO VALIDATION SUMMARY")
    print("=" * 60)

    if "permutation" in results:
        p = results["permutation"]
        print(f"\n1. Permutation Test ({p['n_permutations']} iterations)")
        print(f"   IC:  {p['real_ic']:.4f}  p={p['p_ic']:.4f}  z={p['z_ic']:.2f}  {'✅' if p['ic_significant'] else '❌'}")
        print(f"   Acc: {p['real_dir_acc']:.3f}  p={p['p_acc']:.4f}  z={p['z_acc']:.2f}  {'✅' if p['acc_significant'] else '❌'}")
        print(f"   Null IC: {p['null_ic_mean']:.4f} ± {p['null_ic_std']:.4f}")

    if "future_paths" in results:
        f = results["future_paths"]
        pct = f["final_return_percentiles"]
        print(f"\n2. Future Path Simulation ({f['n_paths']} paths × {f['n_signals_per_path']} signals)")
        print(f"   Empirical WR: {f['empirical_wr']:.1%}  Mean ret: {f['empirical_mean_return']*100:+.3f}%")
        print(f"   Sharpe/signal: {f['sharpe_per_signal']:.3f}")
        print(f"   Prob(profit): {f['prob_profit_after_n_signals']:.1%}")
        print(f"   Median return: {pct['p50']*100:+.2f}%  [5th: {pct['p5']*100:+.2f}%, 95th: {pct['p95']*100:+.2f}%]")
        dd = f["max_drawdown_percentiles"]
        print(f"   Median max DD: {dd['p50']*100:.2f}%")

    if "bootstrap" in results:
        b = results["bootstrap"]
        print(f"\n3. Bootstrap Robustness ({b['n_bootstrap']} resamples)")
        print(f"   IC: {b['ic_mean']:.4f} ± {b['ic_std']:.4f}  [{b['ic_ci_95'][0]:.4f}, {b['ic_ci_95'][1]:.4f}]")
        print(f"   DirAcc: {b['dir_acc_mean']:.3f} ± {b['dir_acc_std']:.3f}")
        print(f"   Signal flip rate: {b['signal_flip_rate']:.1%} ({b['interpretation']['flip_rate']})")
        print(f"   Prediction CV: {b['pred_cv']:.2f} ({b['interpretation']['cv']})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
