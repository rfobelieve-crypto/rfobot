"""
Improved CVD persistence variants.
The naive 4h CVD-price divergence had IC ~0. Test conditional + longer-window forms:

  V1 cvd_persistence_6h           — 6h CVD z-score gated by low-vol regime
  V2 cvd_buildup_no_breakout      — 6h CVD z * (1 - |6h ret z|), reward divergence
  V3 cvd_persistence_lowvol_only  — V1 but hard-zeroed when |ret_z| > 1
  V4 cvd_persistence_12h          — same as V1 but 12h window (longer accumulation)
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


def zscore(s, w):
    mu = s.rolling(w, min_periods=w // 2).mean()
    sd = s.rolling(w, min_periods=w // 2).std()
    return (s - mu) / sd.replace(0, np.nan)


def build(df):
    out = pd.DataFrame(index=df.index)
    cvd_delta = df["cg_fcvd_delta"].fillna(0) + df["cg_scvd_delta"].fillna(0)

    # 6h cumulative CVD, z-scored over 168h
    cvd_6h = cvd_delta.rolling(6, min_periods=3).sum()
    cvd_6h_z = zscore(cvd_6h, 168)

    ret_6h = df["close"].pct_change(6)
    ret_6h_z = zscore(ret_6h, 168)

    # 12h cumulative CVD
    cvd_12h = cvd_delta.rolling(12, min_periods=6).sum()
    cvd_12h_z = zscore(cvd_12h, 168)

    # V1: pure 6h CVD persistence
    out["cvd_persistence_6h"] = cvd_6h_z

    # V2: divergence-weighted (high CVD + low ret = strong signal)
    out["cvd_buildup_no_breakout"] = cvd_6h_z * (1.0 - ret_6h_z.abs().clip(0, 2) / 2)

    # V3: hard-gated low-vol only (zero out when ret has already moved)
    gate = (ret_6h_z.abs() < 1.0).astype(float)
    out["cvd_persistence_lowvol_only"] = cvd_6h_z * gate

    # V4: longer 12h window
    out["cvd_persistence_12h"] = cvd_12h_z

    return out


def ic_report(feat, target, name, target_name):
    m = feat.notna() & target.notna()
    if m.sum() < 100:
        print(f"  {name} vs {target_name}: insufficient ({m.sum()})")
        return None
    rho, p = spearmanr(feat[m], target[m])
    print(f"  {name:32s} vs {target_name:14s}  IC={rho:+.4f}  p={p:.2e}  n={int(m.sum())}")
    return rho


def rolling_ic(feat, target, window=168):
    df = pd.concat([feat, target], axis=1).dropna()
    if len(df) < window:
        return pd.Series(dtype=float)
    out, idx = [], []
    for i in range(window, len(df) + 1):
        chunk = df.iloc[i - window:i]
        rho, _ = spearmanr(chunk.iloc[:, 0], chunk.iloc[:, 1])
        out.append(rho)
        idx.append(df.index[i - 1])
    return pd.Series(out, index=idx)


def main():
    print(f"Loading {CACHE}...")
    df = pd.read_parquet(CACHE)
    labels = build_direction_labels(df, horizon_bars=4, k=0.5)
    df["y_dir"] = labels["y_dir"]
    df["y_return_4h"] = labels["return_4h"]

    feats = build(df)
    print(f"\nFeatures: {list(feats.columns)}")
    for c in feats.columns:
        print(f"  {c:32s}  nan={feats[c].isna().mean():.2%}  std={feats[c].std():.3f}")

    print("\n=== Full-sample IC ===")
    for c in feats.columns:
        ic_report(feats[c], df["y_return_4h"], c, "y_return_4h")
        ic_report(feats[c], df["y_dir"], c, "y_dir")

    print("\n=== Train/Test split (50/50) ===")
    mid = len(df) // 2
    for c in feats.columns:
        rt = ic_report(feats[c].iloc[:mid], df["y_return_4h"].iloc[:mid], c, "y_ret[train]")
        rs = ic_report(feats[c].iloc[mid:], df["y_return_4h"].iloc[mid:], c, "y_ret[test]")
        if rt is not None and rs is not None:
            print(f"    -> {'STABLE' if np.sign(rt) == np.sign(rs) else 'FLIP'}")

    print("\n=== Rolling IC (168h) ===")
    for c in feats.columns:
        ric = rolling_ic(feats[c], df["y_return_4h"])
        if len(ric):
            print(f"  {c:32s}  mean={ric.mean():+.4f}  std={ric.std():.4f}  "
                  f"|IC|>0.05 frac={(ric.abs() > 0.05).mean():.2%}")


if __name__ == "__main__":
    main()
