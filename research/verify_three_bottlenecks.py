"""
Empirically verify the three claimed Direction Model bottlenecks:
  1. Regime adaptation insufficient (same 29 features all regimes)
  2. Meta-information missing (no feature reliability awareness)
  3. dir_prob distribution too concentrated (mostly 0.4-0.6)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HIST = Path("indicator/model_artifacts/indicator_history.parquet")
CACHE = Path("research/dual_model/.cache/features_all.parquet")
DIR_FEATS = Path("indicator/model_artifacts/dual_model/direction_feature_cols.json")
TRAIN_STATS = Path("indicator/model_artifacts/dual_model/training_stats.json")


def main():
    df = pd.read_parquet(HIST)
    print(f"Indicator history: {df.shape}, range {df.index.min()} -> {df.index.max()}")

    # Filter to actual predictions (drop WARMUP / NaN)
    pred = df[df["dir_prob_up"].notna()].copy()
    print(f"Valid predictions: {len(pred)}")

    # ────────────────────────────────────────────────────────────────────
    # Claim 3: dir_prob distribution too concentrated (test first — easiest)
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CLAIM 3: dir_prob distribution too concentrated in [0.4, 0.6]")
    print("=" * 70)

    p = pred["dir_prob_up"]
    print(f"\nFull distribution:")
    print(f"  mean = {p.mean():.4f}   std = {p.std():.4f}")
    print(f"  min  = {p.min():.4f}   max = {p.max():.4f}")
    print(f"  q05  = {p.quantile(0.05):.4f}   q95 = {p.quantile(0.95):.4f}")

    bins = [0.0, 0.30, 0.35, 0.40, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70, 1.0]
    hist = pd.cut(p, bins=bins).value_counts().sort_index()
    print(f"\nBucket counts:")
    for b, c in hist.items():
        bar = "█" * int(c / max(hist) * 40)
        print(f"  {str(b):20s}  {c:5d}  {bar}")

    in_dead = ((p >= 0.40) & (p <= 0.60)).mean()
    in_strong = ((p < 0.30) | (p > 0.70)).mean()
    print(f"\n→ Fraction in [0.40, 0.60]: {in_dead:.1%}")
    print(f"→ Fraction in extremes (<0.30 or >0.70): {in_strong:.1%}")
    print(f"→ Fraction at signal threshold (<0.42 or >0.58): {((p < 0.42) | (p > 0.58)).mean():.1%}")

    # Verdict
    if in_dead > 0.70:
        print("VERDICT: TRUE — distribution is concentrated, claim is valid")
    elif in_dead > 0.50:
        print("VERDICT: PARTIAL — moderately concentrated")
    else:
        print("VERDICT: FALSE — distribution is well-spread")

    # ────────────────────────────────────────────────────────────────────
    # Claim 1: Regime adaptation — does the same model perform differently
    # in different regimes? If yes, regime-aware feature selection MIGHT help.
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CLAIM 1: Regime adaptation — same 29 features across all regimes")
    print("=" * 70)

    if "regime" in pred.columns and "actual_return_4h" in pred.columns:
        valid = pred[pred["actual_return_4h"].notna() & pred["regime"].notna()].copy()
        valid["dir_signed"] = valid["dir_prob_up"] - 0.5  # signed conviction
        print(f"\nValid samples with realized return: {len(valid)}")
        print(f"\nPer-regime IC (dir_prob_up - 0.5) vs actual_return_4h:")
        for regime, grp in valid.groupby("regime"):
            if len(grp) < 30:
                print(f"  {regime:18s}  n={len(grp):4d}  (too small)")
                continue
            rho, p_val = spearmanr(grp["dir_signed"], grp["actual_return_4h"])
            # accuracy for non-neutral
            non_neutral = grp[grp["pred_direction"].isin(["UP", "DOWN"])]
            if len(non_neutral) > 0:
                hit = (
                    ((non_neutral["pred_direction"] == "UP") & (non_neutral["actual_return_4h"] > 0)) |
                    ((non_neutral["pred_direction"] == "DOWN") & (non_neutral["actual_return_4h"] < 0))
                ).mean()
            else:
                hit = float("nan")
            print(f"  {regime:18s}  n={len(grp):4d}  IC={rho:+.4f} p={p_val:.2e}  "
                  f"dir_acc={hit:.1%}  signals_n={len(non_neutral)}")

        # Test specific feature ICs per regime — compute regime from cache directly
        print(f"\nFeature IC by regime (test if some features only work in certain regimes):")
        cache = pd.read_parquet(CACHE)
        from research.dual_model.build_direction_labels import build_direction_labels
        labels = build_direction_labels(cache, horizon_bars=4, k=0.5)
        cache["y_return_4h"] = labels["return_4h"]

        # Reproduce production regime logic: vol_pct > 0.6 + ret_24h direction
        log_ret = np.log(cache["close"] / cache["close"].shift(1))
        vol_24h = log_ret.rolling(24, min_periods=12).std()
        vol_pct = vol_24h.expanding(min_periods=72).rank(pct=True)
        ret_24h = cache["close"].pct_change(24)

        regime = pd.Series("CHOPPY", index=cache.index)
        regime[(vol_pct > 0.6) & (ret_24h > 0.005)] = "TRENDING_BULL"
        regime[(vol_pct > 0.6) & (ret_24h < -0.005)] = "TRENDING_BEAR"
        regime[vol_pct.isna()] = "WARMUP"
        cache_with_reg = cache.copy()
        cache_with_reg["regime"] = regime
        print(f"\nRegime distribution from cache:")
        print(cache_with_reg["regime"].value_counts().to_string())

        with open(DIR_FEATS) as f:
            feats = json.load(f)
        # Pick top 5 by full-sample IC
        full_ics = []
        for c in feats[:30]:
            if c not in cache_with_reg.columns:
                continue
            m = cache_with_reg[c].notna() & cache_with_reg["y_return_4h"].notna()
            if m.sum() < 100:
                continue
            r, _ = spearmanr(cache_with_reg.loc[m, c], cache_with_reg.loc[m, "y_return_4h"])
            full_ics.append((c, r))
        full_ics.sort(key=lambda x: abs(x[1]), reverse=True)
        top5 = [c for c, _ in full_ics[:5]]

        print(f"\n{'Feature':28s}  " + "  ".join(f"{r:>15s}" for r in ["CHOPPY", "T_BULL", "T_BEAR"]))
        print("-" * 80)
        for feat in top5:
            row = [feat]
            for regime in ["CHOPPY", "TRENDING_BULL", "TRENDING_BEAR"]:
                grp = cache_with_reg[cache_with_reg["regime"] == regime]
                m = grp[feat].notna() & grp["y_return_4h"].notna()
                if m.sum() < 50:
                    row.append(f"  n={int(m.sum())}")
                    continue
                r, _ = spearmanr(grp.loc[m, feat], grp.loc[m, "y_return_4h"])
                row.append(f"{r:+.3f} (n={int(m.sum())})")
            print(f"{row[0]:28s}  " + "  ".join(f"{x:>15s}" for x in row[1:]))

    # ────────────────────────────────────────────────────────────────────
    # Claim 2: Meta-information missing — does the model already have
    # regime/structural features? Check the 29 pruned features.
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CLAIM 2: Meta-information / regime awareness missing")
    print("=" * 70)
    with open(DIR_FEATS) as f:
        feats = json.load(f)

    structural = [
        "hour_sin", "hour_cos", "vol_regime", "squeeze_proxy",
        "vol_acceleration", "vol_kurtosis", "realized_vol_20b",
        "cg_oi_close_zscore", "cg_funding_close_zscore",
    ]
    has_structural = [f for f in structural if f in feats]
    print(f"\nStructural / meta-information features in 29 pruned set:")
    for f in has_structural:
        print(f"  [Y]{f}")
    missing_structural = [f for f in structural if f not in feats]
    if missing_structural:
        print(f"\nNot in pruned set:")
        for f in missing_structural:
            print(f"  [N]{f}")

    print(f"\nTotal structural features: {len(has_structural)} / {len(feats)} = "
          f"{len(has_structural)/len(feats):.0%}")

    # Training stats
    if TRAIN_STATS.exists():
        with open(TRAIN_STATS) as f:
            ts = json.load(f)
        print(f"\nTraining stats:")
        print(json.dumps(ts, indent=2)[:600])


if __name__ == "__main__":
    main()
