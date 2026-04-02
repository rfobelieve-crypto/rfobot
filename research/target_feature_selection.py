"""
Target-specific feature selection: per-feature ablation + combination search.
Tests each new feature individually per target, then builds optimal subsets.
"""
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

PARQUET = Path("research/ml_data/BTC_USD_1h_enhanced.parquet")
N_FOLDS = 5
HORIZON = 4

TARGETS = ["up_move_vol_adj", "down_move_vol_adj", "strength_vol_adj"]

NEW_FEATURES = [
    "cg_oi_close_pctchg_4h", "cg_oi_close_pctchg_8h",
    "cg_oi_close_pctchg_12h", "cg_oi_close_pctchg_24h",
    "cg_oi_range", "cg_oi_range_zscore", "cg_oi_range_pct",
    "cg_oi_upper_shadow",
    "cg_oi_binance_share_zscore",
    "quote_vol_zscore", "quote_vol_ratio",
]

EXCLUDE = {
    "open", "high", "low", "close", "volume",
    "log_return", "price_change_pct",
    "taker_buy_vol", "taker_buy_quote", "trade_count",
    "volume_ma_4h", "volume_ma_24h",
    "taker_delta_ma_4h", "taker_delta_std_4h",
    "return_skew",
    "y_return_4h",
    "up_move_vol_adj", "down_move_vol_adj", "strength_vol_adj",
    "future_high_4h", "future_low_4h",
    "regime_name",
}

XGB_PARAMS = dict(
    n_estimators=600, max_depth=5, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
    reg_alpha=0.05, reg_lambda=0.8, gamma=0.1,
    random_state=42, verbosity=0, early_stopping_rounds=40,
)


def load_data():
    df = pd.read_parquet(PARQUET)
    df = df.sort_index()

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)

    future_high = np.full(len(df), np.nan)
    future_low = np.full(len(df), np.nan)
    for i in range(len(df) - HORIZON):
        future_high[i] = np.max(high[i + 1: i + 1 + HORIZON])
        future_low[i] = np.min(low[i + 1: i + 1 + HORIZON])

    df["future_high_4h"] = future_high
    df["future_low_4h"] = future_low

    up_move = np.maximum(future_high / close - 1, 0)
    down_move = np.maximum(1 - future_low / close, 0)
    rvol = df["realized_vol_20b"].values
    rvol_safe = np.where(rvol > 1e-6, rvol, np.nan)

    df["up_move_vol_adj"] = up_move / rvol_safe
    df["down_move_vol_adj"] = down_move / rvol_safe
    df["strength_vol_adj"] = df["up_move_vol_adj"] - df["down_move_vol_adj"]

    for t in TARGETS:
        p01, p99 = df[t].quantile(0.01), df[t].quantile(0.99)
        df[t] = df[t].clip(p01, p99)

    log_ret = np.log(df["close"] / df["close"].shift(1))
    ret_24h = df["close"].pct_change(24)
    vol_24h = log_ret.rolling(24).std()
    vol_pct = vol_24h.expanding(min_periods=168).rank(pct=True)
    regime = pd.Series("CHOPPY", index=df.index)
    regime[(vol_pct > 0.6) & (ret_24h > 0.005)] = "TRENDING_BULL"
    regime[(vol_pct > 0.6) & (ret_24h < -0.005)] = "TRENDING_BEAR"
    regime.iloc[:168] = "WARMUP"
    df["regime_name"] = regime

    all_cols = sorted([c for c in df.columns if c not in EXCLUDE])
    df = df.dropna(subset=TARGETS)
    nan_rate = df[all_cols].isnull().mean()
    drop = list(nan_rate[nan_rate > 0.10].index)
    all_cols = [c for c in all_cols if c not in drop]
    df = df.dropna(subset=all_cols)
    df = df[df["regime_name"] != "WARMUP"]

    base_cols = [c for c in all_cols if c not in set(NEW_FEATURES)]
    new_avail = [c for c in NEW_FEATURES if c in all_cols]

    return df, base_cols, new_avail


def run_cv(df, feat_cols, target):
    """Walk-forward CV, returns per-fold ICs."""
    X = df[feat_cols].fillna(0).values
    y = df[target].values
    n = len(X)
    fold = n // (N_FOLDS + 1)
    fold_ics = []

    for k in range(1, N_FOLDS + 1):
        tr_end = fold * k
        te_end = min(fold * (k + 1), n)
        if te_end <= tr_end:
            break
        X_tr, y_tr = X[:tr_end], y[:tr_end]
        X_te, y_te = X[tr_end:te_end], y[tr_end:te_end]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        pred = model.predict(X_te)
        ic, _ = spearmanr(y_te, pred)
        fold_ics.append(ic)

    return fold_ics


def calc_icir(ics):
    m = np.mean(ics)
    s = np.std(ics)
    return m / s if s > 0 else 0.0


def summarize(ics):
    return {
        "mean_ic": np.mean(ics),
        "icir": calc_icir(ics),
        "pos_folds": sum(1 for ic in ics if ic > 0),
        "ics": ics,
    }


def main():
    df, base_cols, new_feats = load_data()
    print(f"Data: {len(df)} rows")
    print(f"Base features: {len(base_cols)}")
    print(f"New candidates: {new_feats}")
    print()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: COLLINEARITY CHECK
    # ══════════════════════════════════════════════════════════════════
    print("=" * 75)
    print("PHASE 1: COLLINEARITY AMONG NEW FEATURES")
    print("=" * 75)
    corr_df = df[new_feats].corr(method="spearman")
    print("\nHigh correlations (|r| > 0.7):")
    seen = set()
    for i, f1 in enumerate(new_feats):
        for j, f2 in enumerate(new_feats):
            if i >= j:
                continue
            r = corr_df.loc[f1, f2]
            if abs(r) > 0.7:
                print(f"  {f1:40s} x {f2:40s}  r={r:+.3f}")
                seen.add((f1, f2))
    print()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: SINGLE FEATURE ABLATION PER TARGET
    # ══════════════════════════════════════════════════════════════════
    results = {}  # {target: {feature: {delta_ic, delta_icir, folds_improved}}}

    for target in TARGETS:
        print("=" * 75)
        print(f"PHASE 2: SINGLE FEATURE TEST — {target}")
        print("=" * 75)

        base_ics = run_cv(df, base_cols, target)
        base_s = summarize(base_ics)
        print(f"\n  BASELINE: IC={base_s['mean_ic']:+.4f}  ICIR={base_s['icir']:+.4f}  "
              f"pos_folds={base_s['pos_folds']}/5")
        print(f"  Per-fold: {['%+.4f' % ic for ic in base_ics]}")

        target_results = {}
        print(f"\n  {'Feature':42s} {'ΔIC':>7s} {'ΔICIR':>7s} {'Folds+':>7s} {'Verdict':>10s}")
        print(f"  {'-'*80}")

        for feat in new_feats:
            test_cols = base_cols + [feat]
            test_ics = run_cv(df, test_cols, target)
            test_s = summarize(test_ics)

            d_ic = test_s["mean_ic"] - base_s["mean_ic"]
            d_icir = test_s["icir"] - base_s["icir"]
            folds_improved = sum(1 for a, b in zip(base_ics, test_ics) if b > a)

            # Classify
            if d_icir > 0.05 and folds_improved >= 3:
                verdict = "BENEFICIAL"
            elif d_icir < -0.2 or folds_improved <= 1:
                verdict = "HARMFUL"
            else:
                verdict = "neutral"

            target_results[feat] = {
                "d_ic": d_ic, "d_icir": d_icir,
                "folds_improved": folds_improved,
                "verdict": verdict,
                "ics": test_ics,
                "mean_ic": test_s["mean_ic"],
                "icir": test_s["icir"],
            }

            print(f"  {feat:42s} {d_ic:+.4f}  {d_icir:+.4f}  {folds_improved:>5d}/5  {verdict:>10s}")

        results[target] = {"base": base_s, "features": target_results}
        print()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: COMBINATION SEARCH (TOP FEATURES PER TARGET)
    # ══════════════════════════════════════════════════════════════════
    for target in TARGETS:
        print("=" * 75)
        print(f"PHASE 3: COMBINATION SEARCH — {target}")
        print("=" * 75)

        base_s = results[target]["base"]
        feat_results = results[target]["features"]

        # Rank by ΔICIR
        ranked = sorted(feat_results.items(), key=lambda x: x[1]["d_icir"], reverse=True)
        beneficial = [f for f, r in ranked if r["verdict"] == "BENEFICIAL"]
        non_harmful = [f for f, r in ranked if r["verdict"] != "HARMFUL"]

        print(f"\n  Beneficial: {beneficial}")
        print(f"  Non-harmful: {non_harmful[:5]}")

        if not beneficial:
            print(f"\n  No beneficial features found. Keep baseline.")
            print(f"  Baseline ICIR: {base_s['icir']:+.4f}")
            continue

        # Test top-1, top-2, top-3 combinations from beneficial
        best_icir = base_s["icir"]
        best_combo = []
        best_ics = base_s["ics"]

        candidates = beneficial[:5]  # cap at 5 to limit combos

        for size in range(1, min(4, len(candidates) + 1)):
            for combo in itertools.combinations(candidates, size):
                test_cols = base_cols + list(combo)
                test_ics = run_cv(df, test_cols, target)
                test_s = summarize(test_ics)
                folds_better = sum(1 for a, b in zip(base_s["ics"], test_ics) if b > a)

                marker = ""
                if test_s["icir"] > best_icir:
                    best_icir = test_s["icir"]
                    best_combo = list(combo)
                    best_ics = test_ics
                    marker = " ← BEST"

                print(f"  {str(combo):65s}  "
                      f"IC={test_s['mean_ic']:+.4f}  "
                      f"ICIR={test_s['icir']:+.4f}  "
                      f"folds+={folds_better}/5{marker}")

        print(f"\n  >>> OPTIMAL for {target}:")
        print(f"      Features: {best_combo if best_combo else 'BASELINE (no additions)'}")
        print(f"      ICIR: {base_s['icir']:+.4f} → {best_icir:+.4f} (Δ={best_icir - base_s['icir']:+.4f})")
        print(f"      Per-fold: {['%+.4f' % ic for ic in best_ics]}")
        print()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: CROSS-TARGET CONFLICT ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    print("=" * 75)
    print("PHASE 4: CROSS-TARGET CONFLICTS")
    print("=" * 75)

    for feat in new_feats:
        verdicts = {}
        for target in TARGETS:
            r = results[target]["features"][feat]
            verdicts[target] = (r["verdict"], r["d_icir"])

        conflict = False
        v_list = [v[0] for v in verdicts.values()]
        if "BENEFICIAL" in v_list and "HARMFUL" in v_list:
            conflict = True

        if conflict:
            print(f"\n  ⚠️  CONFLICT: {feat}")
            for t, (v, d) in verdicts.items():
                print(f"     {t:25s}  {v:12s}  ΔICIR={d:+.4f}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 5: FINAL RECOMMENDATION
    # ══════════════════════════════════════════════════════════════════
    print()
    print("=" * 75)
    print("PHASE 5: FINAL RECOMMENDATION")
    print("=" * 75)

    for target in TARGETS:
        base_s = results[target]["base"]
        feat_results = results[target]["features"]
        ranked = sorted(feat_results.items(), key=lambda x: x[1]["d_icir"], reverse=True)

        print(f"\n  {target}:")
        print(f"    Baseline ICIR: {base_s['icir']:+.4f}")
        print(f"    Feature ranking by ΔICIR:")
        for f, r in ranked:
            print(f"      {r['verdict']:12s}  {f:42s}  ΔICIR={r['d_icir']:+.4f}  folds+={r['folds_improved']}/5")


if __name__ == "__main__":
    main()
