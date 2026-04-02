"""
Feature Validation — which features predict 4h upside / downside / net strength?

Three targets:
  up_move_vol_adj   : upside potential (vol-adjusted)
  down_move_vol_adj : downside potential (vol-adjusted, always positive)
  strength_vol_adj  : net direction (up - down)

Three lenses:
  1. IC / ICIR  (Spearman, rolling window=200)
  2. Mutual Information
  3. XGBoost Permutation Importance (walk-forward CV)

Usage:
    python research/feature_validation.py
    python research/feature_validation.py --save
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
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

PARQUET = Path(__file__).parent / "ml_data" / "BTC_USD_15m_tabular.parquet"
OUT_DIR = Path(__file__).parent / "eda_charts"
TARGETS = ["up_move_vol_adj", "down_move_vol_adj", "strength_vol_adj"]

EXCLUDE = {
    "ts_open",
    # lookahead targets
    "future_high_4h", "future_low_4h",
    "up_move_4h", "down_move_4h",
    "up_move_vol_adj", "down_move_vol_adj",
    "strength_raw", "strength_vol_adj",
    "future_return_5m", "future_return_15m", "future_return_1h",
    "label_5m", "label_15m", "label_1h",
    # all NaN
    "oi", "oi_delta", "oi_accel", "oi_divergence",
    "cvd_x_oi_delta", "cvd_oi_ratio",
    # price levels (scale-dependent, not order flow)
    "open", "high", "low", "close",
    # vol_4h_proxy: computed from same window as label -> exclude
    "vol_4h_proxy",
}


# ─── Load ──────────────────────────────────────────────────────────────────────

def load() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(PARQUET)
    df["dt"] = pd.to_datetime(df["ts_open"], unit="ms", utc=True)
    df = df.set_index("dt").sort_index()

    feat_cols = [c for c in df.columns if c not in EXCLUDE and c not in set(TARGETS)]
    df = df.dropna(subset=TARGETS)
    df[feat_cols] = df[feat_cols].ffill()

    nan_rate = df[feat_cols].isnull().mean()
    drop     = list(nan_rate[nan_rate > 0.05].index)
    if drop:
        print(f"Dropping high-NaN features: {drop}")
        feat_cols = [c for c in feat_cols if c not in drop]

    df = df.dropna(subset=feat_cols)
    print(f"Dataset: {len(df)} rows x {len(feat_cols)} features")
    print(f"Features: {feat_cols}\n")
    return df, feat_cols


# ─── 1. IC / ICIR ──────────────────────────────────────────────────────────────

def compute_icir(df: pd.DataFrame, feat_cols: list[str],
                 target: str, window: int = 200) -> pd.DataFrame:
    y = df[target].values
    records = []
    for col in feat_cols:
        x   = df[col].values
        n   = len(x)
        ics = []
        for i in range(window, n):
            xw = x[i - window: i]
            yw = y[i - window: i]
            mask = ~(np.isnan(xw) | np.isnan(yw))
            if mask.sum() < 20:
                continue
            ic, _ = spearmanr(xw[mask], yw[mask])
            ics.append(ic)
        if not ics:
            continue
        ics    = np.array(ics)
        mean   = ics.mean()
        std    = ics.std()
        icir   = mean / std if std > 0 else 0.0
        records.append({
            "feature": col,
            "mean_ic": round(mean, 5),
            "std_ic":  round(std,  5),
            "icir":    round(icir, 4),
            "pct_pos": round((ics > 0).mean(), 3),
        })
    return (pd.DataFrame(records)
              .sort_values("icir", key=abs, ascending=False)
              .reset_index(drop=True))


# ─── 2. Mutual Information ─────────────────────────────────────────────────────

def compute_mi(df: pd.DataFrame, feat_cols: list[str],
               target: str) -> pd.Series:
    X  = df[feat_cols].fillna(0).values
    y  = df[target].values
    mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
    return pd.Series(mi, index=feat_cols).sort_values(ascending=False)


# ─── 3. XGBoost Permutation Importance ────────────────────────────────────────

def compute_perm_importance(df: pd.DataFrame, feat_cols: list[str],
                             target: str, n_splits: int = 5) -> pd.Series:
    n      = len(df)
    fold   = n // (n_splits + 1)
    X_all  = df[feat_cols].fillna(0).values
    y_all  = df[target].values
    perm   = np.zeros(len(feat_cols))
    r2s    = []

    for k in range(1, n_splits + 1):
        tr_end = fold * k
        te_end = fold * (k + 1)
        if te_end > n:
            break
        X_tr, y_tr = X_all[:tr_end],      y_all[:tr_end]
        X_te, y_te = X_all[tr_end:te_end], y_all[tr_end:te_end]

        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        pi   = permutation_importance(model, X_te, y_te,
                                      n_repeats=10, random_state=42, scoring="r2")
        perm += pi.importances_mean / n_splits
        r2s.append(r2_score(y_te, model.predict(X_te)))

    print(f"    mean val R2={np.mean(r2s):.4f}  (folds: {[round(r,4) for r in r2s]})")
    return pd.Series(perm, index=feat_cols).sort_values(ascending=False)


# ─── Combined ranking ──────────────────────────────────────────────────────────

def combined_rank(icir_df: pd.DataFrame, mi: pd.Series,
                  perm: pd.Series) -> pd.DataFrame:
    rank_ic   = icir_df.set_index("feature")["icir"].abs().rank(ascending=False)
    rank_mi   = mi.rank(ascending=False)
    rank_perm = perm.rank(ascending=False)

    df = pd.DataFrame({
        "icir":      icir_df.set_index("feature")["icir"],
        "mi":        mi,
        "perm_imp":  perm,
        "rank_ic":   rank_ic,
        "rank_mi":   rank_mi,
        "rank_perm": rank_perm,
    }).dropna()
    df["avg_rank"] = df[["rank_ic","rank_mi","rank_perm"]].mean(axis=1)
    return df.sort_values("avg_rank")[["icir","mi","perm_imp","avg_rank"]]


# ─── Plotting ──────────────────────────────────────────────────────────────────

def plot_combined(results: dict[str, pd.DataFrame], top_n: int = 15,
                  save: bool = False):
    """One chart comparing top features across all three targets."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    target_labels = {
        "up_move_vol_adj":   "Upside Potential",
        "down_move_vol_adj": "Downside Potential",
        "strength_vol_adj":  "Net Strength",
    }
    colors = {
        "up_move_vol_adj":   "#26a69a",
        "down_move_vol_adj": "#ef5350",
        "strength_vol_adj":  "#1976d2",
    }

    for ax, target in zip(axes, TARGETS):
        df = results[target].head(top_n)
        c  = colors[target]
        ax.barh(df.index[::-1], df["avg_rank"][::-1].apply(lambda x: top_n + 1 - x), color=c, alpha=0.8)
        ax.set_title(target_labels[target], fontsize=11)
        ax.set_xlabel("Combined Score (higher = better)")
        ax.tick_params(axis="y", labelsize=8)

    plt.suptitle("Top Features by Target (IC + MI + Permutation Importance)", fontsize=12)
    plt.tight_layout()
    _save(fig, "feature_validation_combined.png", save)


def plot_icir_heatmap(all_icir: dict[str, pd.DataFrame],
                      feat_cols: list[str], save: bool = False):
    """Heatmap: features × targets, coloured by ICIR."""
    data = {}
    for target in TARGETS:
        s = all_icir[target].set_index("feature")["icir"]
        data[target] = s
    hm = pd.DataFrame(data).reindex(feat_cols).dropna(how="all")

    # Sort by abs ICIR of strength_vol_adj
    hm = hm.reindex(hm["strength_vol_adj"].abs().sort_values(ascending=False).index)

    import matplotlib.colors as mcolors
    fig, ax = plt.subplots(figsize=(8, max(6, len(hm) * 0.35)))
    im = ax.imshow(hm.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels(["Upside", "Downside", "Net Strength"], fontsize=10)
    ax.set_yticks(range(len(hm)))
    ax.set_yticklabels(hm.index, fontsize=8)
    plt.colorbar(im, ax=ax, label="ICIR")
    ax.set_title("ICIR Heatmap: Feature x Target", fontsize=12)
    plt.tight_layout()
    _save(fig, "feature_validation_icir_heatmap.png", save)


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


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    df, feat_cols = load()

    all_icir    = {}
    all_results = {}

    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"TARGET: {target}")
        print(f"{'='*60}")

        print("  IC/ICIR...")
        icir_df = compute_icir(df, feat_cols, target)
        all_icir[target] = icir_df

        print("  Mutual Information...")
        mi = compute_mi(df, feat_cols, target)

        print("  XGBoost Permutation Importance...")
        perm = compute_perm_importance(df, feat_cols, target)

        rank = combined_rank(icir_df, mi, perm)
        all_results[target] = rank

        print(f"\n  Top 10 combined ranking:")
        print(rank.head(10).round(4).to_string())

    # ── Cross-target summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CROSS-TARGET SUMMARY")
    print(f"{'='*60}")
    print("\nICIR comparison (features that appear in top 10 of any target):")

    top_features: set[str] = set()
    for target in TARGETS:
        top_features.update(all_results[target].head(10).index)

    summary_rows = []
    for feat in sorted(top_features):
        row = {"feature": feat}
        for target in TARGETS:
            icir_row = all_icir[target].set_index("feature")
            row[target] = icir_row.loc[feat, "icir"] if feat in icir_row.index else np.nan
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).set_index("feature")
    summary.columns = ["ICIR_up", "ICIR_down", "ICIR_net"]
    print(summary.round(4).sort_values("ICIR_net", key=abs, ascending=False).to_string())

    # Save results
    for target in TARGETS:
        out = Path(__file__).parent / f"feature_validation_{target}.csv"
        all_results[target].reset_index().to_csv(out, index=False)
    print("\n[saved] feature_validation_*.csv")

    # Plots
    plot_icir_heatmap(all_icir, feat_cols, save=args.save)
    plot_combined(all_results, save=args.save)

    print("\nDone.")


if __name__ == "__main__":
    main()
