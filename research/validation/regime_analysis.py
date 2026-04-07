"""
Layer 2 validation: Regime-conditional performance analysis.

Confirms the dual-model prediction system does not only work in
specific market conditions. Splits OOS predictions by volatility,
trend, funding, and OI regimes, then computes direction + magnitude
+ joint metrics for each regime bucket.

Usage
-----
    from research.validation.regime_analysis import run_regime_analysis
    df = run_regime_analysis(joint_oos, features_path, output_dir)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime computation helpers
# ---------------------------------------------------------------------------

def _compute_regimes(features: pd.DataFrame, joint_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Derive regime labels from feature columns, aligned to joint OOS index.

    Parameters
    ----------
    features : Full features DataFrame (from cache parquet).
    joint_index : DatetimeIndex of the joint OOS DataFrame.

    Returns
    -------
    DataFrame indexed like joint_index with columns for each regime type.
    Missing regimes (feature not found) are silently skipped.
    """
    feat = features.reindex(joint_index)
    regimes = pd.DataFrame(index=joint_index)

    # 1. Volatility: realized_vol_20b > median
    if "realized_vol_20b" in feat.columns:
        vol = feat["realized_vol_20b"]
        med = vol.median()
        regimes["volatility"] = np.where(vol > med, "high_vol", "low_vol")
        regimes.loc[vol.isna(), "volatility"] = np.nan
        logger.info("Volatility regime: median=%.6f, high=%d, low=%d",
                     med,
                     (regimes["volatility"] == "high_vol").sum(),
                     (regimes["volatility"] == "low_vol").sum())
    else:
        logger.warning("realized_vol_20b not found — skipping volatility regime")

    # 2. Trend: rolling 24h return
    if "log_return" in feat.columns:
        rolling_ret = feat["log_return"].rolling(24, min_periods=12).sum()
        regimes["trend"] = np.where(
            rolling_ret > 0.005, "bull",
            np.where(rolling_ret < -0.005, "bear", "range"),
        )
        regimes.loc[rolling_ret.isna(), "trend"] = np.nan
        logger.info("Trend regime: bull=%d, bear=%d, range=%d",
                     (regimes["trend"] == "bull").sum(),
                     (regimes["trend"] == "bear").sum(),
                     (regimes["trend"] == "range").sum())
    else:
        logger.warning("log_return not found — skipping trend regime")

    # 3. Funding: cg_funding_close percentiles
    if "cg_funding_close" in feat.columns:
        fr = feat["cg_funding_close"]
        p75 = fr.quantile(0.75)
        p25 = fr.quantile(0.25)
        regimes["funding"] = np.where(
            fr > p75, "high_funding",
            np.where(fr < p25, "low_funding", "neutral_funding"),
        )
        regimes.loc[fr.isna(), "funding"] = np.nan
        logger.info("Funding regime: high=%d, low=%d, neutral=%d (p25=%.6f, p75=%.6f)",
                     (regimes["funding"] == "high_funding").sum(),
                     (regimes["funding"] == "low_funding").sum(),
                     (regimes["funding"] == "neutral_funding").sum(),
                     p25, p75)
    else:
        logger.warning("cg_funding_close not found — skipping funding regime")

    # 4. OI change: cg_oi_close pct_change(4) percentiles
    if "cg_oi_close" in feat.columns:
        oi_chg = feat["cg_oi_close"].pct_change(4)
        p75 = oi_chg.quantile(0.75)
        p25 = oi_chg.quantile(0.25)
        regimes["oi_change"] = np.where(
            oi_chg > p75, "oi_expanding",
            np.where(oi_chg < p25, "oi_contracting", "oi_stable"),
        )
        regimes.loc[oi_chg.isna(), "oi_change"] = np.nan
        logger.info("OI change regime: expanding=%d, contracting=%d, stable=%d",
                     (regimes["oi_change"] == "oi_expanding").sum(),
                     (regimes["oi_change"] == "oi_contracting").sum(),
                     (regimes["oi_change"] == "oi_stable").sum())
    else:
        logger.warning("cg_oi_close not found — skipping OI change regime")

    return regimes


# ---------------------------------------------------------------------------
# Per-regime metrics
# ---------------------------------------------------------------------------

def _safe_auc(y_true: np.ndarray, prob_up: np.ndarray) -> float:
    """ROC AUC that returns NaN when undefined (single class or too few samples)."""
    mask = ~(np.isnan(y_true) | np.isnan(prob_up))
    y, p = y_true[mask], prob_up[mask]
    if len(y) < 10 or len(np.unique(y)) < 2:
        return np.nan
    try:
        return roc_auc_score(y, p)
    except ValueError:
        return np.nan


def _safe_accuracy(y_true: np.ndarray, prob_up: np.ndarray, threshold: float = 0.5) -> float:
    """Accuracy that returns NaN on insufficient data."""
    mask = ~(np.isnan(y_true) | np.isnan(prob_up))
    y, p = y_true[mask], prob_up[mask]
    if len(y) < 10:
        return np.nan
    return accuracy_score(y, (p >= threshold).astype(int))


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman IC that returns NaN on failure."""
    mask = ~(np.isnan(a) | np.isnan(b))
    a2, b2 = a[mask], b[mask]
    if len(a2) < 10:
        return np.nan
    try:
        corr, _ = spearmanr(a2, b2)
        return corr
    except Exception:
        return np.nan


def _compute_regime_metrics(
    joint: pd.DataFrame,
    mask: np.ndarray,
) -> dict:
    """
    Compute direction, magnitude, and joint metrics for a boolean mask.

    Parameters
    ----------
    joint : Joint OOS DataFrame with columns [prob_up, dir_true, mag_pred, mag_true, return_4h].
    mask : Boolean array aligned to joint index.

    Returns
    -------
    Dict of metric values.
    """
    sub = joint.loc[mask]
    n = len(sub)
    if n < 10:
        return {
            "n_samples": n,
            "dir_auc": np.nan,
            "dir_accuracy": np.nan,
            "mag_ic": np.nan,
            "joint_return": np.nan,
            "joint_winrate": np.nan,
        }

    y_true = sub["dir_true"].values.astype(float)
    prob_up = sub["prob_up"].values.astype(float)

    dir_auc = _safe_auc(y_true, prob_up)
    dir_acc = _safe_accuracy(y_true, prob_up)

    # Magnitude IC
    mag_pred = sub["mag_pred"].values.astype(float) if "mag_pred" in sub.columns else np.full(n, np.nan)
    mag_true = sub["mag_true"].values.astype(float) if "mag_true" in sub.columns else np.full(n, np.nan)
    mag_ic = _safe_spearman(mag_pred, mag_true)

    # Joint metrics (using return_4h)
    ret = sub["return_4h"].values.astype(float) if "return_4h" in sub.columns else np.full(n, np.nan)
    valid_ret = ret[~np.isnan(ret)]
    joint_return = float(np.mean(valid_ret)) if len(valid_ret) > 0 else np.nan
    joint_winrate = float(np.mean(valid_ret > 0)) if len(valid_ret) > 0 else np.nan

    return {
        "n_samples": n,
        "dir_auc": dir_auc,
        "dir_accuracy": dir_acc,
        "mag_ic": mag_ic,
        "joint_return": joint_return,
        "joint_winrate": joint_winrate,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_regime_analysis(
    joint: pd.DataFrame,
    features_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Run regime-conditional performance analysis on joint OOS predictions.

    Parameters
    ----------
    joint : Joint OOS DataFrame with columns [prob_up, dir_true, mag_pred,
            mag_true, return_4h]. Index must be DatetimeIndex (UTC).
    features_path : Path to features parquet (e.g. research/dual_model/.cache/features_all.parquet).
    output_dir : Directory to save results CSV.

    Returns
    -------
    DataFrame with columns [regime_type, regime_value, n_samples, dir_auc,
    dir_accuracy, mag_ic, joint_return, joint_winrate].
    Also saved to output_dir / validation_regime_breakdown.csv.
    """
    logger.info("=== Layer 2: Regime Analysis ===")
    logger.info("Joint OOS: %d rows, %s ~ %s",
                len(joint), joint.index[0], joint.index[-1])

    # Load features
    if not features_path.exists():
        raise FileNotFoundError(f"Features parquet not found: {features_path}")

    features = pd.read_parquet(features_path)
    if not isinstance(features.index, pd.DatetimeIndex):
        if "dt" in features.columns:
            features = features.set_index("dt")
        else:
            raise ValueError("Features must have DatetimeIndex or 'dt' column")

    logger.info("Features loaded: %d rows x %d cols", len(features), len(features.columns))

    # Compute regimes aligned to joint index
    regimes = _compute_regimes(features, joint.index)

    # Iterate over each regime type and value
    rows = []
    for regime_type in regimes.columns:
        col = regimes[regime_type]
        unique_vals = col.dropna().unique()
        logger.info("Regime '%s': %d unique values — %s", regime_type, len(unique_vals), list(unique_vals))

        for val in sorted(unique_vals):
            mask = (col == val).values
            metrics = _compute_regime_metrics(joint, mask)
            rows.append({
                "regime_type": regime_type,
                "regime_value": val,
                **metrics,
            })

    result = pd.DataFrame(rows)

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "validation_regime_breakdown.csv"
    result.to_csv(out_path, index=False)
    logger.info("Saved regime breakdown: %s (%d rows)", out_path, len(result))

    return result


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_regime_report(df: pd.DataFrame) -> None:
    """
    Pretty-print the regime breakdown DataFrame to console.

    Parameters
    ----------
    df : Output from run_regime_analysis().
    """
    print("\n" + "=" * 80)
    print("LAYER 2: REGIME-CONDITIONAL PERFORMANCE")
    print("=" * 80)

    for regime_type in df["regime_type"].unique():
        sub = df[df["regime_type"] == regime_type].copy()
        print(f"\n--- {regime_type.upper()} ---")
        print(f"{'Value':<20s} {'N':>6s} {'AUC':>7s} {'Acc':>7s} "
              f"{'Mag IC':>8s} {'MeanRet':>9s} {'WinRate':>8s}")
        print("-" * 68)
        for _, row in sub.iterrows():
            auc_str = f"{row['dir_auc']:.3f}" if pd.notna(row["dir_auc"]) else "  N/A"
            acc_str = f"{row['dir_accuracy']:.3f}" if pd.notna(row["dir_accuracy"]) else "  N/A"
            ic_str = f"{row['mag_ic']:.4f}" if pd.notna(row["mag_ic"]) else "   N/A"
            ret_str = f"{row['joint_return']:.5f}" if pd.notna(row["joint_return"]) else "    N/A"
            wr_str = f"{row['joint_winrate']:.3f}" if pd.notna(row["joint_winrate"]) else "  N/A"
            print(f"{row['regime_value']:<20s} {row['n_samples']:>6d} {auc_str:>7s} "
                  f"{acc_str:>7s} {ic_str:>8s} {ret_str:>9s} {wr_str:>8s}")

    # Summary: flag regimes with AUC < 0.52 or accuracy < 0.50
    print("\n--- WARNINGS ---")
    warn_count = 0
    for _, row in df.iterrows():
        issues = []
        if pd.notna(row["dir_auc"]) and row["dir_auc"] < 0.52:
            issues.append(f"AUC={row['dir_auc']:.3f}")
        if pd.notna(row["dir_accuracy"]) and row["dir_accuracy"] < 0.50:
            issues.append(f"Acc={row['dir_accuracy']:.3f}")
        if pd.notna(row["mag_ic"]) and abs(row["mag_ic"]) < 0.02:
            issues.append(f"IC={row['mag_ic']:.4f}")
        if issues:
            print(f"  [{row['regime_type']}={row['regime_value']}] weak: {', '.join(issues)}")
            warn_count += 1

    if warn_count == 0:
        print("  None — model performs reasonably across all regimes.")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    # Default paths
    joint_path = PROJECT_ROOT / "research" / "results" / "dual_model" / "joint_oos.parquet"
    features_path = PROJECT_ROOT / "research" / "dual_model" / ".cache" / "features_all.parquet"
    output_dir = PROJECT_ROOT / "research" / "results" / "validation"

    if not joint_path.exists():
        print(f"Joint OOS parquet not found: {joint_path}")
        sys.exit(1)

    joint = pd.read_parquet(joint_path)
    if not isinstance(joint.index, pd.DatetimeIndex):
        if "dt" in joint.columns:
            joint = joint.set_index("dt")

    result = run_regime_analysis(joint, features_path, output_dir)
    print_regime_report(result)
