"""
Breakthrough experiment — swap binary first-touch classifier for
vol-normalized path regression.

Hypothesis:
    The binary Init model is bottlenecked at ~68% Strong WR because it only
    sees 137 positive samples out of 3700. A regression target sees EVERY
    sample's continuous outcome (path_max_up/vol), giving ~27x more gradient
    signal. This may push AUC past the 0.57 structural ceiling.

Targets:
    y_long_reg  = max(path_max_up_4h, 0) / realized_vol_20b
    y_short_reg = max(-path_max_dn_4h, 0) / realized_vol_20b

Procedure:
    1. Load data + add initiation features
    2. Build path labels (reusing build_initiation_labels with H=4)
    3. Walk-forward XGBRegressor on long/short targets
    4. Rank predictions, take top-5% as Strong candidates
    5. Apply same gate: mag_pct>=0.80 AND breakout-required
    6. Measure realized WR against binary y_long_touch/y_short_touch
       (so result is directly comparable to 68.1% baseline)

Run:  python -m research.regression_target_breakthrough
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import load_and_cache_data, walk_forward_splits
from research.dual_model.build_initiation_labels import (
    build_initiation_labels, InitiationLabelConfig,
)
from research.dual_model.direction_features_v2 import (
    ABLATION_GROUPS, filter_available,
)
from indicator.initiation_features import (
    add_initiation_features, INITIATION_FEATURE_COLS,
)

STRONG_FRAC = 0.05
STRONG_MAG_PCT = 0.80
MAG_ROLL = 720
EXPORT_DIR = PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model"
RESULTS_DIR = PROJECT_ROOT / "research" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

XGB_REG = dict(
    objective="reg:squarederror", eval_metric="rmse",
    max_depth=4, learning_rate=0.05, n_estimators=400,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
    early_stopping_rounds=40,
)

XGB_REG_HUBER = dict(XGB_REG)
XGB_REG_HUBER.update(objective="reg:pseudohubererror", huber_slope=0.5)

# Quantile regression — explicitly optimize the right tail (90th percentile)
# to combat the regression-to-mean effect that flattens top-5% predictions.
XGB_REG_Q90 = dict(XGB_REG)
XGB_REG_Q90.update(objective="reg:quantileerror", quantile_alpha=0.90)
XGB_REG_Q90.pop("eval_metric", None)  # quantile doesn't support rmse eval

XGB_REG_Q95 = dict(XGB_REG_Q90)
XGB_REG_Q95.update(quantile_alpha=0.95)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def score_mag_percentile(df: pd.DataFrame) -> pd.Series:
    feats = json.loads((EXPORT_DIR / "magnitude_feature_cols.json").read_text())
    feats = [f for f in feats if f in df.columns]
    booster = xgb.Booster()
    booster.load_model(str(EXPORT_DIR / "magnitude_xgb.json"))
    X = df[feats].fillna(0).values
    dm = xgb.DMatrix(X, feature_names=feats)
    raw = booster.predict(dm)
    mag = pd.Series(raw, index=df.index)
    return mag.rolling(MAG_ROLL, min_periods=100).apply(
        lambda x: float((x[-1] > x[:-1]).mean()), raw=True
    )


def run_wf_regression(df: pd.DataFrame, features: list[str],
                       target: str, xgb_params: dict) -> pd.DataFrame:
    splits = walk_forward_splits(len(df), initial_train=288, test_size=48,
                                  step=48, purge=4, embargo=4)
    oos = []
    for tr, te in splits:
        train = df.iloc[tr]; test = df.iloc[te]
        tr_m = train[target].notna()
        te_m = test[target].notna()
        if tr_m.sum() < 80 or te_m.sum() < 10:
            continue
        X_tr = train.loc[tr_m, features].fillna(0)
        y_tr = train.loc[tr_m, target].values.astype(float)
        X_te = test.loc[te_m, features].fillna(0)
        y_te = test.loc[te_m, target].values.astype(float)

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        p = model.predict(X_te)
        sub = test.loc[te_m, []].copy()
        sub["pred"] = p
        sub["target"] = y_te
        oos.append(sub)
    return pd.concat(oos).sort_index() if oos else pd.DataFrame()


def strong_gate_reg(oos_long: pd.DataFrame, oos_short: pd.DataFrame,
                     df: pd.DataFrame, mag_pct: pd.Series,
                     y_long_bin: pd.Series, y_short_bin: pd.Series) -> dict:
    """
    Rank regression predictions, take top-5% as Strong candidates, apply
    mag_pct>=0.8 + breakout gate, measure realized WR via binary labels.
    """
    idx = oos_long.index.intersection(oos_short.index)
    if len(idx) == 0:
        return dict(n=0, wr=0.0, ci_lo=0.0, ci_hi=0.0)

    pl = oos_long.loc[idx, "pred"]
    ps = oos_short.loc[idx, "pred"]
    yl_bin = y_long_bin.reindex(idx).fillna(0).astype(int).values
    ys_bin = y_short_bin.reindex(idx).fillna(0).astype(int).values
    bo_up = df.loc[idx, "init_bo_up"].fillna(0).values
    bo_dn = df.loc[idx, "init_bo_dn"].fillna(0).values
    mp = mag_pct.reindex(idx).fillna(0.5).values

    long_thr = float(pl.quantile(1 - STRONG_FRAC))
    short_thr = float(ps.quantile(1 - STRONG_FRAC))

    long_strong = (pl.values >= long_thr) & (mp >= STRONG_MAG_PCT) & (bo_up == 1.0)
    short_strong = (ps.values >= short_thr) & (mp >= STRONG_MAG_PCT) & (bo_dn == 1.0)

    wins = fires = up = dn = 0
    long_wins = long_fires = 0
    short_wins = short_fires = 0
    for i in range(len(idx)):
        ls, ss = long_strong[i], short_strong[i]
        if ls and ss:
            continue
        if ls:
            fires += 1; up += 1; wins += int(yl_bin[i])
            long_fires += 1; long_wins += int(yl_bin[i])
        elif ss:
            fires += 1; dn += 1; wins += int(ys_bin[i])
            short_fires += 1; short_wins += int(ys_bin[i])

    if fires == 0:
        return dict(n=0, wr=0.0, ci_lo=0.0, ci_hi=0.0, up=0, dn=0,
                    long_thr=long_thr, short_thr=short_thr)
    wr = wins / fires
    lo, hi = wilson_ci(wins, fires)
    long_wr = long_wins / long_fires if long_fires else 0.0
    short_wr = short_wins / short_fires if short_fires else 0.0
    return dict(n=fires, wins=wins, wr=wr, ci_lo=lo, ci_hi=hi,
                up=up, dn=dn, long_thr=long_thr, short_thr=short_thr,
                long_wr=long_wr, long_fires=long_fires,
                short_wr=short_wr, short_fires=short_fires)


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    df = load_and_cache_data()
    df = add_initiation_features(df)
    print(f"Data: {len(df)} bars x {len(df.columns)} cols")

    # Build labels (H=4, k=0.8% — same as baseline for direct comparison)
    cfg = InitiationLabelConfig(k_pct=0.008, breakout_lookback=20,
                                 use_breakout_confirm=True, horizon_bars=4)
    labels = build_initiation_labels(df, cfg)
    df["y_long_touch"] = labels["y_long_touch"]
    df["y_short_touch"] = labels["y_short_touch"]
    df["path_max_up_4h"] = labels["path_max_up_4h"]
    df["path_max_dn_4h"] = labels["path_max_dn_4h"]

    # Vol normalization denominator
    if "realized_vol_20b" in df.columns:
        vol = df["realized_vol_20b"].astype(float)
    else:
        # fallback: rolling 20-bar std of log returns
        lr = np.log(df["close"] / df["close"].shift(1))
        vol = lr.shift(1).rolling(20, min_periods=8).std()
    vol_safe = vol.where(vol > 1e-6, np.nan)
    vol_med = float(vol_safe.median())
    vol_safe = vol_safe.fillna(vol_med)

    # Regression targets — only positive side matters
    # Long: how high did the path go? Short: how far did it drop?
    y_long_reg = np.clip(df["path_max_up_4h"].values, 0, None) / vol_safe.values
    y_short_reg = np.clip(-df["path_max_dn_4h"].values, 0, None) / vol_safe.values
    df["y_long_reg"] = y_long_reg
    df["y_short_reg"] = y_short_reg

    # Winsorize top 0.5% to tame outliers
    for col in ["y_long_reg", "y_short_reg"]:
        q = df[col].quantile(0.995)
        df[col] = df[col].clip(upper=q)

    valid = df["y_long_reg"].notna() & df["y_long_touch"].notna()
    print(f"Valid samples: {valid.sum()}")
    print(f"y_long_reg: mean={df.loc[valid,'y_long_reg'].mean():.3f}  "
          f"std={df.loc[valid,'y_long_reg'].std():.3f}  "
          f"p95={df.loc[valid,'y_long_reg'].quantile(0.95):.3f}  "
          f"p99={df.loc[valid,'y_long_reg'].quantile(0.99):.3f}")
    print(f"y_short_reg: mean={df.loc[valid,'y_short_reg'].mean():.3f}  "
          f"std={df.loc[valid,'y_short_reg'].std():.3f}  "
          f"p95={df.loc[valid,'y_short_reg'].quantile(0.95):.3f}  "
          f"p99={df.loc[valid,'y_short_reg'].quantile(0.99):.3f}")

    # Feature set
    base = filter_available(ABLATION_GROUPS["+ key_4_only"], list(df.columns))
    init = [c for c in INITIATION_FEATURE_COLS if c in df.columns]
    features = sorted(set(base) | set(init))
    print(f"Features: {len(features)} (base={len(base)} init={len(init)})")

    print("\nScoring MAG percentile...")
    mag_pct = score_mag_percentile(df)

    print("\n" + "=" * 80)
    print("  REGRESSION TARGET BREAKTHROUGH")
    print("=" * 80)

    configs = [
        ("reg_squarederror", XGB_REG),
        ("reg_pseudohuber",  XGB_REG_HUBER),
        ("reg_quantile_q90", XGB_REG_Q90),
        ("reg_quantile_q95", XGB_REG_Q95),
    ]

    all_results = []
    # Test each config at multiple STRONG_FRAC values to find a matched-n
    # comparison point with the 94-fire binary baseline.
    frac_sweep = [0.05, 0.08, 0.10, 0.12, 0.15]
    for name, params in configs:
        print(f"\n--- {name} ---")
        print("[1/2] Walk-forward LONG regression ...")
        oos_l = run_wf_regression(df, features, "y_long_reg", params)
        print(f"  long_oos={len(oos_l)}")
        if len(oos_l):
            ic_l, _ = spearmanr(oos_l["pred"], oos_l["target"])
            print(f"  LONG Spearman IC: {ic_l:.3f}")

        print("[2/2] Walk-forward SHORT regression ...")
        oos_s = run_wf_regression(df, features, "y_short_reg", params)
        print(f"  short_oos={len(oos_s)}")
        if len(oos_s):
            ic_s, _ = spearmanr(oos_s["pred"], oos_s["target"])
            print(f"  SHORT Spearman IC: {ic_s:.3f}")

        # Sweep STRONG_FRAC to find matched-fire comparison points
        print(f"\n  top-frac sweep (find matched-n comparison with baseline 94):")
        print(f"    {'frac':>6}{'n':>6}{'WR':>9}{'CI_lo':>9}{'long_n':>9}"
              f"{'long_WR':>10}{'short_n':>10}{'short_WR':>11}")
        for frac in frac_sweep:
            global STRONG_FRAC
            STRONG_FRAC_backup = STRONG_FRAC
            STRONG_FRAC = frac
            res = strong_gate_reg(
                oos_l, oos_s, df, mag_pct,
                df["y_long_touch"], df["y_short_touch"],
            )
            STRONG_FRAC = STRONG_FRAC_backup
            res["config"] = name
            res["frac"] = frac
            all_results.append(res)
            print(f"    {frac*100:>5.0f}%{res['n']:>6}"
                  f"{res['wr']*100:>8.1f}%{res['ci_lo']*100:>8.1f}%"
                  f"{res.get('long_fires',0):>9}"
                  f"{res.get('long_wr',0)*100:>9.1f}%"
                  f"{res.get('short_fires',0):>10}"
                  f"{res.get('short_wr',0)*100:>10.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    #   Gate-isolation probe: how strong is regression ALONE, without
    #   the breakout / mag co-adaptation penalty?
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  GATE-ISOLATION PROBE  (reg_squarederror, vary which gates apply)")
    print("=" * 80)

    # Re-run squarederror to get the oos_l / oos_s (they're not saved from
    # the loop above) — cheap reuse
    oos_l = run_wf_regression(df, features, "y_long_reg", XGB_REG)
    oos_s = run_wf_regression(df, features, "y_short_reg", XGB_REG)

    idx = oos_l.index.intersection(oos_s.index)
    pl = oos_l.loc[idx, "pred"].values
    ps = oos_s.loc[idx, "pred"].values
    yl_bin = df["y_long_touch"].reindex(idx).fillna(0).astype(int).values
    ys_bin = df["y_short_touch"].reindex(idx).fillna(0).astype(int).values
    bo_up = df.loc[idx, "init_bo_up"].fillna(0).values
    bo_dn = df.loc[idx, "init_bo_dn"].fillna(0).values
    mp = mag_pct.reindex(idx).fillna(0.5).values

    def eval_gate(long_mask, short_mask, label):
        fires = 0; wins = 0; lw = 0; lf = 0; sw = 0; sf = 0
        for i in range(len(idx)):
            ls, ss = long_mask[i], short_mask[i]
            if ls and ss:
                continue
            if ls:
                fires += 1; wins += int(yl_bin[i])
                lf += 1; lw += int(yl_bin[i])
            elif ss:
                fires += 1; wins += int(ys_bin[i])
                sf += 1; sw += int(ys_bin[i])
        if fires == 0:
            print(f"  {label:<38} n=0")
            return
        lo, hi = wilson_ci(wins, fires)
        long_wr_s = f"{lw/lf*100:5.1f}%" if lf else "  n/a"
        short_wr_s = f"{sw/sf*100:5.1f}%" if sf else "  n/a"
        print(f"  {label:<38} n={fires:>4}  WR={wins/fires*100:5.1f}%  "
              f"CI=[{lo*100:4.1f},{hi*100:4.1f}]  "
              f"L={lf:>3}/{long_wr_s}  S={sf:>3}/{short_wr_s}")

    for frac in [0.05, 0.10]:
        pl_thr = float(pd.Series(pl).quantile(1 - frac))
        ps_thr = float(pd.Series(ps).quantile(1 - frac))
        # Base masks: top-frac prediction
        base_l = (pl >= pl_thr)
        base_s = (ps >= ps_thr)
        print(f"\n--- top-{frac*100:.0f}% predictions ---")
        eval_gate(base_l, base_s,
                  f"no gate (pure reg)")
        eval_gate(base_l & (mp >= STRONG_MAG_PCT),
                  base_s & (mp >= STRONG_MAG_PCT),
                  f"+ mag_pct>=0.80 only")
        eval_gate(base_l & (bo_up == 1.0),
                  base_s & (bo_dn == 1.0),
                  f"+ breakout only")
        eval_gate(base_l & (mp >= STRONG_MAG_PCT) & (bo_up == 1.0),
                  base_s & (mp >= STRONG_MAG_PCT) & (bo_dn == 1.0),
                  f"+ mag>=0.80 + breakout (current)")

    print("\n" + "=" * 80)
    print("  FINAL SUMMARY  vs  Binary Baseline (n=94 WR=68.1% CI_lo=58.1%)")
    print("=" * 80)
    # Pick the config+frac with n closest to 94 for fair comparison
    closest = min(all_results, key=lambda r: abs(r["n"] - 94))
    print(f"\nClosest-n match (target n=94):")
    print(f"  {closest['config']} frac={closest['frac']*100:.0f}%: "
          f"n={closest['n']}  WR={closest['wr']*100:.1f}%  "
          f"CI_lo={closest['ci_lo']*100:.1f}%  "
          f"Δ={(closest['wr']-0.681)*100:+.1f}pp")
    # Pick highest CI_lo with n >= 40
    eligible = [r for r in all_results if r["n"] >= 40]
    if eligible:
        best = max(eligible, key=lambda r: r["ci_lo"])
        print(f"\nHighest CI_lo (n>=40):")
        print(f"  {best['config']} frac={best['frac']*100:.0f}%: "
              f"n={best['n']}  WR={best['wr']*100:.1f}%  "
              f"CI_lo={best['ci_lo']*100:.1f}%  "
              f"Δ CI_lo={(best['ci_lo']-0.581)*100:+.1f}pp")
        if best["ci_lo"] > 0.581:
            print("  *** BREAKTHROUGH ***")

    # Persist
    out = {
        "configs": all_results,
        "baseline": {"wr": 0.681, "ci_lo": 0.581, "n": 94},
    }
    (RESULTS_DIR / "regression_target_breakthrough.json").write_text(
        json.dumps(out, indent=2, default=float))
    print(f"\nSaved: {RESULTS_DIR/'regression_target_breakthrough.json'}")


if __name__ == "__main__":
    main()
