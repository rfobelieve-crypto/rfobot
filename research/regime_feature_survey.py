"""
Survey all 29 pruned features for cross-regime IC variance.
Identify features whose predictive power varies significantly by regime,
which become candidates for regime-interaction features.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.dual_model.build_direction_labels import build_direction_labels

CACHE = Path("research/dual_model/.cache/features_all.parquet")
DIR_FEATS = Path("indicator/model_artifacts/dual_model/direction_feature_cols.json")


def add_regime(df: pd.DataFrame) -> pd.DataFrame:
    log_ret = np.log(df["close"] / df["close"].shift(1))
    vol_24h = log_ret.rolling(24, min_periods=12).std()
    vol_pct = vol_24h.expanding(min_periods=72).rank(pct=True)
    ret_24h = df["close"].pct_change(24)
    regime = pd.Series("CHOPPY", index=df.index)
    regime[(vol_pct > 0.6) & (ret_24h > 0.005)] = "TRENDING_BULL"
    regime[(vol_pct > 0.6) & (ret_24h < -0.005)] = "TRENDING_BEAR"
    regime[vol_pct.isna()] = "WARMUP"
    df = df.copy()
    df["regime"] = regime
    return df


def regime_ic(feat: pd.Series, target: pd.Series) -> tuple[float, int]:
    m = feat.notna() & target.notna()
    if m.sum() < 50:
        return float("nan"), int(m.sum())
    rho, _ = spearmanr(feat[m], target[m])
    return rho, int(m.sum())


def main():
    df = pd.read_parquet(CACHE)
    df = add_regime(df)
    labels = build_direction_labels(df, horizon_bars=4, k=0.5)
    df["y_return_4h"] = labels["return_4h"]

    with open(DIR_FEATS) as f:
        feats = json.load(f)
    feats = [c for c in feats if c in df.columns]

    print(f"Surveying {len(feats)} pruned features across regimes")
    print(f"Regime counts: {df['regime'].value_counts().to_dict()}\n")

    results = []
    for f in feats:
        ic_full, n_full = regime_ic(df[f], df["y_return_4h"])
        row = {"feature": f, "ic_full": ic_full}
        for regime in ["CHOPPY", "TRENDING_BULL", "TRENDING_BEAR"]:
            grp = df[df["regime"] == regime]
            ic, n = regime_ic(grp[f], grp["y_return_4h"])
            row[f"ic_{regime}"] = ic
            row[f"n_{regime}"] = n
        results.append(row)

    res = pd.DataFrame(results)

    # Cross-regime variance: how much does the IC change?
    res["ic_range"] = res[["ic_CHOPPY", "ic_TRENDING_BULL", "ic_TRENDING_BEAR"]].max(axis=1) - \
                      res[["ic_CHOPPY", "ic_TRENDING_BULL", "ic_TRENDING_BEAR"]].min(axis=1)

    # Sign flip: does IC change sign across regimes?
    def has_flip(row):
        ics = [row["ic_CHOPPY"], row["ic_TRENDING_BULL"], row["ic_TRENDING_BEAR"]]
        ics = [x for x in ics if not np.isnan(x)]
        if len(ics) < 2:
            return False
        return any(np.sign(a) != np.sign(b) for a in ics for b in ics if abs(a) > 0.02 and abs(b) > 0.02)
    res["sign_flip"] = res.apply(has_flip, axis=1)

    # Sort by ic_range (biggest cross-regime variation)
    res = res.sort_values("ic_range", ascending=False)

    print("=" * 100)
    print("Features ranked by cross-regime IC range (biggest = most regime-dependent)")
    print("=" * 100)
    print(f"{'Feature':32s}  {'Full':>7s}  {'CHOPPY':>8s}  {'BULL':>8s}  {'BEAR':>8s}  {'Range':>7s}  Flip")
    print("-" * 100)
    for _, r in res.head(15).iterrows():
        flip = "FLIP" if r["sign_flip"] else ""
        print(f"{r['feature']:32s}  {r['ic_full']:+7.4f}  "
              f"{r['ic_CHOPPY']:+8.4f}  {r['ic_TRENDING_BULL']:+8.4f}  {r['ic_TRENDING_BEAR']:+8.4f}  "
              f"{r['ic_range']:7.4f}  {flip}")

    print("\n=== Sign-flip features (highest priority for interaction terms) ===")
    flips = res[res["sign_flip"]].copy()
    print(f"Found {len(flips)} sign-flipping features")
    for _, r in flips.iterrows():
        print(f"  {r['feature']:32s}  CHOPPY={r['ic_CHOPPY']:+.4f}  "
              f"BULL={r['ic_TRENDING_BULL']:+.4f}  BEAR={r['ic_TRENDING_BEAR']:+.4f}")

    print("\n=== Features dead in one regime (|IC|<0.02 in some regime, strong elsewhere) ===")
    def dead_pattern(row):
        ics = [row["ic_CHOPPY"], row["ic_TRENDING_BULL"], row["ic_TRENDING_BEAR"]]
        if any(abs(i) < 0.02 for i in ics) and any(abs(i) > 0.05 for i in ics):
            return True
        return False
    deads = res[res.apply(dead_pattern, axis=1)]
    for _, r in deads.iterrows():
        dead_in = []
        if abs(r["ic_CHOPPY"]) < 0.02: dead_in.append("CHOPPY")
        if abs(r["ic_TRENDING_BULL"]) < 0.02: dead_in.append("BULL")
        if abs(r["ic_TRENDING_BEAR"]) < 0.02: dead_in.append("BEAR")
        print(f"  {r['feature']:32s}  dead in: {','.join(dead_in)}")

    print("\n=== Recommended interaction candidates ===")
    # Pick top candidates: high range OR flip OR dead-in-one
    candidates = res[
        (res["ic_range"] > 0.08) | (res["sign_flip"]) |
        (res.apply(dead_pattern, axis=1))
    ].head(10)
    print(f"\nTotal: {len(candidates)} features warrant regime-interaction terms")
    for _, r in candidates.iterrows():
        print(f"  {r['feature']}")

    # Save for next step
    out = Path("research/results/regime_feature_survey.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
