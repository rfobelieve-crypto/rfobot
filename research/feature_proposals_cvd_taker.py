"""
IC validation for two proposed features:
  #1 cvd_price_divergence_4h  — 4h CVD vs price return divergence z-score
  #2 aggressive_buy_ratio_24h_z — 24h rolling taker buy ratio z-score

Compares against y_dir (direction label) and y_return_4h (raw forward return).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.dual_model.build_direction_labels import build_direction_labels

CACHE = Path("research/dual_model/.cache/features_all.parquet")


def zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std()
    return (s - mu) / sd.replace(0, np.nan)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # ---- #1: 4h CVD vs price return divergence ----
    # Combined futures + spot CVD delta (per-bar net taker flow)
    cvd_delta = df["cg_fcvd_delta"].fillna(0) + df["cg_scvd_delta"].fillna(0)
    cvd_4h = cvd_delta.rolling(4, min_periods=4).sum()
    ret_4h = df["close"].pct_change(4)

    # Z-score each over 168h (1 week) then take difference
    cvd_z = zscore(cvd_4h, 168)
    ret_z = zscore(ret_4h, 168)
    out["cvd_price_divergence_4h"] = cvd_z - ret_z

    # ---- #2: 24h rolling taker buy ratio z-score ----
    buy = df["cg_taker_buy"].fillna(0)
    sell = df["cg_taker_sell"].fillna(0)
    # Rolling sums to smooth
    buy_24 = buy.rolling(24, min_periods=12).sum()
    sell_24 = sell.rolling(24, min_periods=12).sum()
    ratio_24 = buy_24 / (buy_24 + sell_24).replace(0, np.nan)
    out["aggressive_buy_ratio_24h_z"] = zscore(ratio_24, 168)

    return out


def ic_report(feat: pd.Series, target: pd.Series, name: str, target_name: str):
    mask = feat.notna() & target.notna()
    if mask.sum() < 100:
        print(f"  {name} vs {target_name}: insufficient samples ({mask.sum()})")
        return
    rho, p = spearmanr(feat[mask], target[mask])
    n = int(mask.sum())
    print(f"  {name:30s} vs {target_name:14s}  IC={rho:+.4f}  p={p:.2e}  n={n}")


def rolling_ic(feat: pd.Series, target: pd.Series, window: int = 168) -> pd.Series:
    """Rolling IC over `window` bars to check stability."""
    df = pd.concat([feat, target], axis=1).dropna()
    if len(df) < window:
        return pd.Series(dtype=float)
    out = []
    idx = []
    for i in range(window, len(df) + 1):
        chunk = df.iloc[i - window:i]
        rho, _ = spearmanr(chunk.iloc[:, 0], chunk.iloc[:, 1])
        out.append(rho)
        idx.append(df.index[i - 1])
    return pd.Series(out, index=idx)


def main():
    print(f"Loading features from {CACHE}...")
    df = pd.read_parquet(CACHE)
    print(f"  shape={df.shape}  range={df.index.min()} -> {df.index.max()}")

    # Build labels
    labels = build_direction_labels(df, horizon_bars=4, k=0.5)
    df["y_dir"] = labels["y_dir"]
    df["y_return_4h"] = labels["return_4h"]

    # Build new features
    new_feats = build_features(df)
    print(f"\nNew features built: {list(new_feats.columns)}")
    print(f"  cvd_price_divergence_4h: nan_pct={new_feats['cvd_price_divergence_4h'].isna().mean():.2%}")
    print(f"  aggressive_buy_ratio_24h_z: nan_pct={new_feats['aggressive_buy_ratio_24h_z'].isna().mean():.2%}")

    # ---- Full-sample IC ----
    print("\n=== Full-sample IC ===")
    for col in new_feats.columns:
        ic_report(new_feats[col], df["y_return_4h"], col, "y_return_4h")
        ic_report(new_feats[col], df["y_dir"], col, "y_dir")

    # ---- Compare with existing top direction features ----
    print("\n=== Reference: existing top direction features ===")
    for ref in ["cg_oi_close_zscore", "cg_oi_agg_close", "cg_bfx_margin_ratio",
                "cg_fcvd_delta_zscore", "cg_taker_delta_zscore",
                "cg_spot_futures_cvd_divergence"]:
        if ref in df.columns:
            ic_report(df[ref], df["y_return_4h"], ref, "y_return_4h")

    # ---- Stability: split into train/test halves ----
    print("\n=== Train/Test split IC (50/50) ===")
    mid = len(df) // 2
    for col in new_feats.columns:
        f = new_feats[col]
        for label, sl in [("train", slice(None, mid)), ("test", slice(mid, None))]:
            ic_report(f.iloc[sl], df["y_return_4h"].iloc[sl], col, f"y_return_4h[{label}]")

    # ---- Rolling IC stats ----
    print("\n=== Rolling IC (168h window) stats ===")
    for col in new_feats.columns:
        ric = rolling_ic(new_feats[col], df["y_return_4h"], window=168)
        if len(ric) == 0:
            print(f"  {col}: insufficient data")
            continue
        print(f"  {col:30s}  mean={ric.mean():+.4f}  std={ric.std():.4f}  "
              f"min={ric.min():+.4f}  max={ric.max():+.4f}  "
              f"|IC|>0.05 frac={(ric.abs() > 0.05).mean():.2%}")

    # ---- Correlation with existing features (avoid redundancy) ----
    print("\n=== Correlation with existing related features ===")
    related = ["cg_fcvd_delta_zscore", "cg_scvd_delta_zscore",
               "cg_taker_delta_zscore", "cg_taker_ratio",
               "cg_spot_futures_cvd_divergence"]
    for col in new_feats.columns:
        print(f"\n  {col}:")
        for ref in related:
            if ref in df.columns:
                corr = new_feats[col].corr(df[ref])
                print(f"    vs {ref:35s}  corr={corr:+.3f}")

    # Save for later inspection
    out_path = Path("research/results/feature_proposals_cvd_taker.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_feats.to_parquet(out_path)
    print(f"\nFeatures saved to {out_path}")


if __name__ == "__main__":
    main()
