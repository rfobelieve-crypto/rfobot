"""
Regime-Conditional XGBoost — Multi-Target Training.

Three separate training runs (C → B → A order):
  C. up_move_vol_adj   : upside potential (always >= 0)
  C. down_move_vol_adj : downside potential (always >= 0)
  C. strength_vol_adj  : net direction (up - down)
  B. Per-target: deep-dive regime TRENDING_BEAR feature importance
  A. strength_vol_adj  : flip TRENDING_BULL labels (negative IC → positive)

Architecture per target:
  - 1 Global baseline model
  - 3 Regime-specific models (TRENDING_BULL / TRENDING_BEAR / CHOPPY)
  - Walk-forward CV (no data leakage)

Models saved to: research/models/{target}/{regime}_xgb.json
feature_cols.txt saved once (shared across targets).

Usage:
    python research/model_training.py
    python research/model_training.py --save
"""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

PARQUET   = Path(__file__).parent / "ml_data" / "BTC_USD_15m_tabular.parquet"
MODEL_DIR = Path(__file__).parent / "models"
OUT_DIR   = Path(__file__).parent / "eda_charts"
N_FOLDS   = 5

TARGETS = ["up_move_vol_adj", "down_move_vol_adj", "strength_vol_adj"]

# Columns never used as features
EXCLUDE_BASE = {
    "ts_open",
    # all lookahead labels
    "future_high_4h", "future_low_4h",
    "up_move_4h", "down_move_4h",
    "up_move_vol_adj", "down_move_vol_adj",
    "strength_raw", "strength_vol_adj",
    "future_return_5m", "future_return_15m", "future_return_1h",
    "label_5m", "label_15m", "label_1h",
    # all NaN (OI not yet accumulated)
    "oi", "oi_delta", "oi_accel", "oi_divergence",
    "cvd_x_oi_delta", "cvd_oi_ratio",
    # price levels (scale-dependent)
    "open", "high", "low", "close",
    "vol_4h_proxy",
    # regime columns (routing only, not features)
    "regime", "regime_name",
}

XGB_PARAMS = dict(
    n_estimators          = 500,
    max_depth             = 4,
    learning_rate         = 0.03,
    subsample             = 0.8,
    colsample_bytree      = 0.7,
    min_child_weight      = 15,
    reg_alpha             = 0.1,
    reg_lambda            = 1.0,
    random_state          = 42,
    verbosity             = 0,
    early_stopping_rounds = 30,
)


# ─── Load ──────────────────────────────────────────────────────────────────────

def load() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(PARQUET)
    df["dt"] = pd.to_datetime(df["ts_open"], unit="ms", utc=True)
    df = df.set_index("dt").sort_index()

    # All targets must be present
    df = df.dropna(subset=TARGETS)

    feat_cols = [c for c in df.columns if c not in EXCLUDE_BASE]
    df[feat_cols] = df[feat_cols].ffill()

    nan_rate  = df[feat_cols].isnull().mean()
    drop_cols = list(nan_rate[nan_rate > 0.05].index)
    if drop_cols:
        print(f"  Dropping high-NaN features: {drop_cols}")
        feat_cols = [c for c in feat_cols if c not in drop_cols]

    df = df.dropna(subset=feat_cols)
    print(f"  {len(df)} rows x {len(feat_cols)} features")
    return df, feat_cols


# ─── Metrics ───────────────────────────────────────────────────────────────────

def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                 target: str, threshold: float = 0.1) -> dict:
    r2    = r2_score(y_true, y_pred)
    ic, _ = spearmanr(y_true, y_pred)

    # dir_acc only meaningful for net strength (has negative values)
    if target == "strength_vol_adj":
        mask = np.abs(y_true) > threshold
        dir_acc = (np.sign(y_pred[mask]) == np.sign(y_true[mask])).mean() \
                  if mask.sum() > 10 else np.nan
    else:
        dir_acc = np.nan   # up/down moves are always positive — direction meaningless

    return {"r2": r2, "ic": ic, "dir_acc": dir_acc, "n": len(y_true)}


# ─── Walk-forward CV ───────────────────────────────────────────────────────────

def walk_forward_cv(X: np.ndarray, y: np.ndarray, target: str,
                    n_folds: int = N_FOLDS,
                    label: str = "global") -> dict:
    n    = len(X)
    fold = n // (n_folds + 1)
    fold_metrics = []
    oos_pred     = np.full(n, np.nan)
    oos_true     = np.full(n, np.nan)

    for k in range(1, n_folds + 1):
        tr_end = fold * k
        te_end = min(fold * (k + 1), n)
        if te_end <= tr_end:
            break

        X_tr, y_tr = X[:tr_end],       y[:tr_end]
        X_te, y_te = X[tr_end:te_end], y[tr_end:te_end]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        pred = model.predict(X_te)
        m    = calc_metrics(y_te, pred, target)
        fold_metrics.append(m)
        oos_pred[tr_end:te_end] = pred
        oos_true[tr_end:te_end] = y_te

        dir_str = f"dir={m['dir_acc']:.1%}" if not np.isnan(m["dir_acc"]) else "dir=N/A"
        print(f"    [{label}] fold {k}/{n_folds}  "
              f"R2={m['r2']:+.4f}  IC={m['ic']:+.4f}  {dir_str}  n={m['n']}")

    r2s  = [m["r2"]      for m in fold_metrics]
    ics  = [m["ic"]      for m in fold_metrics]
    dirs = [m["dir_acc"] for m in fold_metrics if not np.isnan(m.get("dir_acc", np.nan))]

    return {
        "fold_metrics": fold_metrics,
        "mean_r2":      np.mean(r2s),
        "mean_ic":      np.mean(ics),
        "icir":         np.mean(ics) / np.std(ics) if np.std(ics) > 0 else 0.0,
        "mean_dir_acc": np.mean(dirs) if dirs else np.nan,
        "oos_pred":     oos_pred,
        "oos_true":     oos_true,
    }


# ─── Train final model ─────────────────────────────────────────────────────────

def train_final(X: np.ndarray, y: np.ndarray) -> xgb.XGBRegressor:
    params = {k: v for k, v in XGB_PARAMS.items()
              if k != "early_stopping_rounds"}
    model = xgb.XGBRegressor(**params)
    model.fit(X, y, verbose=False)
    return model


# ─── Run one target ────────────────────────────────────────────────────────────

def run_target(df: pd.DataFrame, feat_cols: list[str],
               target: str) -> dict:
    """Train global + regime-specific models for one target."""
    X_all = df[feat_cols].fillna(0).values
    y_all = df[target].values
    results = {}

    print(f"\n  [global]")
    results["global"] = walk_forward_cv(X_all, y_all, target, label="global")

    regimes = sorted(df["regime_name"].dropna().unique()) \
              if "regime_name" in df.columns else []

    for regime in regimes:
        mask = df["regime_name"] == regime
        n    = mask.sum()
        if n < 500:
            print(f"\n  [{regime}]: only {n} bars, skipping")
            continue
        print(f"\n  [{regime}]  ({n} bars)")
        X_r = df.loc[mask, feat_cols].fillna(0).values
        y_r = df.loc[mask, target].values
        results[regime] = walk_forward_cv(X_r, y_r, target, label=regime)

    return results


# ─── Save models ───────────────────────────────────────────────────────────────

def save_models(df: pd.DataFrame, feat_cols: list[str], target: str):
    tdir = MODEL_DIR / target
    tdir.mkdir(parents=True, exist_ok=True)

    # Global
    m = train_final(df[feat_cols].fillna(0).values, df[target].values)
    m.save_model(str(tdir / "global_xgb.json"))
    print(f"  [{target}] saved global_xgb.json")

    for regime in df["regime_name"].dropna().unique():
        mask = df["regime_name"] == regime
        if mask.sum() < 500:
            continue
        X_r = df.loc[mask, feat_cols].fillna(0).values
        y_r = df.loc[mask, target].values
        mr  = train_final(X_r, y_r)
        fname = regime.lower().replace(" ", "_") + "_xgb.json"
        mr.save_model(str(tdir / fname))
        print(f"  [{target}] saved {fname}")


# ─── Plots ─────────────────────────────────────────────────────────────────────

def plot_target_summary(results: dict, target: str, save: bool = False):
    """Bar chart: R2 / IC / ICIR / Dir-Acc for one target across models."""
    labels = list(results.keys())
    metrics_keys = ["mean_r2", "mean_ic", "icir", "mean_dir_acc"]
    titles        = ["Mean R2", "Mean IC", "ICIR", "Dir Accuracy"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    colors = ["#1976d2", "#26a69a", "#ff9800", "#e91e63", "#9c27b0"]

    for ax, key, title in zip(axes, metrics_keys, titles):
        vals = [results[l].get(key, np.nan) for l in labels]
        bc   = [colors[i % len(colors)] for i in range(len(labels))]
        bars = ax.bar(labels, vals, color=bc, alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(title)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{val:.3f}", ha="center", fontsize=8)

    plt.suptitle(f"Target: {target} — Global vs Regime Models", fontsize=12)
    plt.tight_layout()
    _save(fig, f"model_{target}.png", save)


def plot_cross_target_ic(all_results: dict[str, dict], save: bool = False):
    """IC and ICIR comparison across all targets × models."""
    model_keys  = sorted({k for r in all_results.values() for k in r})
    target_labels = {
        "up_move_vol_adj":   "Upside",
        "down_move_vol_adj": "Downside",
        "strength_vol_adj":  "Net Strength",
    }
    colors = ["#1976d2", "#26a69a", "#ff9800", "#e91e63"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel in zip(axes,
                                   ["mean_ic", "icir"],
                                   ["Mean IC (Spearman)", "ICIR"]):
        x     = np.arange(len(model_keys))
        width = 0.25
        for i, (tgt, res) in enumerate(all_results.items()):
            vals = [res.get(mk, {}).get(metric, np.nan) for mk in model_keys]
            ax.bar(x + i * width, vals,
                   width=width, label=target_labels.get(tgt, tgt),
                   color=colors[i % len(colors)], alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x + width)
        ax.set_xticklabels(model_keys, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} — All Targets")
        ax.legend(fontsize=8)

    plt.suptitle("Cross-Target Comparison: IC and ICIR per Regime", fontsize=12)
    plt.tight_layout()
    _save(fig, "model_cross_target.png", save)


def plot_feature_importance(df: pd.DataFrame, feat_cols: list[str],
                             target: str, save: bool = False):
    """Top-20 XGBoost gain importance for one target (full dataset)."""
    m   = train_final(df[feat_cols].fillna(0).values, df[target].values)
    imp = pd.Series(m.feature_importances_, index=feat_cols).sort_values(ascending=False)
    top = imp.head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top.index[::-1], top.values[::-1], color="#1976d2", alpha=0.85)
    ax.set_title(f"Feature Importance (global model) — {target}")
    ax.set_xlabel("Gain")
    plt.tight_layout()
    _save(fig, f"model_importance_{target}.png", save)


def plot_bear_regime_importance(df: pd.DataFrame, feat_cols: list[str],
                                 save: bool = False):
    """B: Feature importance specifically for TRENDING_BEAR × each target."""
    mask = df["regime_name"] == "TRENDING_BEAR"
    if mask.sum() < 200:
        print("  TRENDING_BEAR: not enough bars for importance plot")
        return

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(18, 7))
    colors_map = {
        "up_move_vol_adj":   "#26a69a",
        "down_move_vol_adj": "#ef5350",
        "strength_vol_adj":  "#1976d2",
    }

    for ax, target in zip(axes, TARGETS):
        X_r = df.loc[mask, feat_cols].fillna(0).values
        y_r = df.loc[mask, target].values
        m   = train_final(X_r, y_r)
        imp = pd.Series(m.feature_importances_, index=feat_cols).sort_values(ascending=False)
        top = imp.head(15)
        c   = colors_map.get(target, "#1976d2")
        ax.barh(top.index[::-1], top.values[::-1], color=c, alpha=0.85)
        ax.set_title(f"TRENDING_BEAR\n{target}", fontsize=10)
        ax.set_xlabel("Gain")
        ax.tick_params(axis="y", labelsize=8)

    plt.suptitle("B: Feature Importance — TRENDING_BEAR Regime × Each Target",
                 fontsize=12)
    plt.tight_layout()
    _save(fig, "model_bear_importance.png", save)


def _save(fig, name: str, save: bool):
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        p = OUT_DIR / name
        fig.savefig(p, bbox_inches="tight")
        print(f"  [saved] {p.name}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


# ─── Summary table ─────────────────────────────────────────────────────────────

def print_summary(all_results: dict[str, dict]):
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — IC / ICIR across targets × models")
    print("=" * 70)
    rows = []
    for target, res in all_results.items():
        for model_key, r in res.items():
            rows.append({
                "target":    target,
                "model":     model_key,
                "mean_IC":   round(r["mean_ic"],      4),
                "ICIR":      round(r["icir"],         4),
                "mean_R2":   round(r["mean_r2"],      4),
                "dir_acc":   round(r["mean_dir_acc"], 4)
                             if not np.isnan(r["mean_dir_acc"]) else "—",
            })
    df_sum = pd.DataFrame(rows).set_index(["target", "model"])
    print(df_sum.to_string())


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    print("Loading data...")
    df, feat_cols = load()

    if "regime_name" not in df.columns:
        print("WARNING: regime_name missing — run regime_detection.py first")

    print(f"Regime distribution:")
    if "regime_name" in df.columns:
        print(df["regime_name"].value_counts().to_string())

    all_results: dict[str, dict] = {}

    # ── C: Train all three targets separately ─────────────────────────────────
    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"TARGET: {target}")
        print(f"{'='*60}")
        results = run_target(df, feat_cols, target)
        all_results[target] = results

        # Per-target summary
        rows = []
        for k, r in results.items():
            dir_str = f"{r['mean_dir_acc']:.4f}" \
                      if not np.isnan(r["mean_dir_acc"]) else "—"
            rows.append({"model": k,
                         "mean_IC": round(r["mean_ic"], 4),
                         "ICIR":    round(r["icir"],    4),
                         "mean_R2": round(r["mean_r2"], 4),
                         "dir_acc": dir_str})
        print(f"\n  {target} — CV results:")
        print(pd.DataFrame(rows).set_index("model").to_string())

    # ── Save trained models ───────────────────────────────────────────────────
    print("\nSaving models...")
    for target in TARGETS:
        save_models(df, feat_cols, target)
    (MODEL_DIR / "feature_cols.txt").write_text("\n".join(feat_cols))
    print(f"  Saved feature_cols.txt ({len(feat_cols)} features)")

    # ── Summary across all targets ────────────────────────────────────────────
    print_summary(all_results)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nPlotting...")
    for target in TARGETS:
        plot_target_summary(all_results[target], target, save=args.save)
        plot_feature_importance(df, feat_cols, target, save=args.save)

    # C: cross-target IC comparison
    plot_cross_target_ic(all_results, save=args.save)

    # B: TRENDING_BEAR deep-dive
    plot_bear_regime_importance(df, feat_cols, save=args.save)

    print("\nDone.")


if __name__ == "__main__":
    main()
