"""
IC validation for OI/funding/liq shock-type features:
  #3 oi_8h_accum_z              — 8h OI accumulation strength (trend precursor)
  #4 funding_extreme_persistence — bars of sustained |funding_z| > 1.0 (mean reversion)
  #5 long_liq_exhaustion_4h     — 4h cumulative long liq z-score (bottom reversal)
  #6 oi_price_disagreement_4h   — fraction of bars with OI/price sign mismatch (divergence)
  #7 binance_oi_share_extreme   — |Binance OI share z-score| (cross-exchange flow)
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
    log_ret = np.log(df["close"] / df["close"].shift(1))

    # ---- #3: 8h OI accumulation z-score ----
    oi_delta = df["cg_oi_agg_delta"].fillna(0)
    out["oi_8h_accum_z"] = zscore(oi_delta.rolling(8, min_periods=4).sum(), 168)

    # ---- #4: funding extreme persistence ----
    fund_z = df["cg_funding_close_zscore"].fillna(0)
    extreme = (fund_z.abs() > 1.0).astype(float)
    # Rolling sum over 12h (how many of last 12 bars were extreme)
    persistence = extreme.rolling(12, min_periods=6).sum()
    # Sign convention: positive funding extreme = bearish (overcrowded long)
    out["funding_extreme_persistence"] = -np.sign(fund_z) * persistence

    # ---- #5: 4h long-liq exhaustion ----
    long_liq = df["cg_liq_agg_long"].fillna(0)
    long_liq_4h = long_liq.rolling(4, min_periods=2).sum()
    out["long_liq_exhaustion_4h"] = zscore(long_liq_4h, 168)

    # ---- #6: OI/price sign disagreement over 4h ----
    sign_oi = np.sign(oi_delta)
    sign_ret = np.sign(log_ret)
    mismatch = (sign_oi != sign_ret).astype(float)
    # Weighted by magnitude: large OI move + opposite price = stronger signal
    weighted = mismatch * sign_oi * (oi_delta.abs() / oi_delta.abs().rolling(168, min_periods=72).mean())
    out["oi_price_disagreement_4h"] = weighted.rolling(4, min_periods=2).mean()

    # ---- #7: Binance OI share extreme ----
    if "cg_oi_binance_share_zscore" in df.columns:
        out["binance_oi_share_extreme"] = df["cg_oi_binance_share_zscore"].abs()
    else:
        out["binance_oi_share_extreme"] = np.nan

    return out


def ic_report(feat: pd.Series, target: pd.Series, name: str, target_name: str):
    mask = feat.notna() & target.notna()
    if mask.sum() < 100:
        print(f"  {name} vs {target_name}: insufficient samples ({mask.sum()})")
        return None
    rho, p = spearmanr(feat[mask], target[mask])
    n = int(mask.sum())
    print(f"  {name:32s} vs {target_name:14s}  IC={rho:+.4f}  p={p:.2e}  n={n}")
    return rho


def rolling_ic(feat: pd.Series, target: pd.Series, window: int = 168) -> pd.Series:
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

    labels = build_direction_labels(df, horizon_bars=4, k=0.5)
    df["y_dir"] = labels["y_dir"]
    df["y_return_4h"] = labels["return_4h"]

    new_feats = build_features(df)
    print(f"\nNew features built: {list(new_feats.columns)}")
    for c in new_feats.columns:
        print(f"  {c:32s}  nan={new_feats[c].isna().mean():.2%}  "
              f"std={new_feats[c].std():.3f}")

    print("\n=== Full-sample IC ===")
    ics = {}
    for col in new_feats.columns:
        rho_ret = ic_report(new_feats[col], df["y_return_4h"], col, "y_return_4h")
        rho_dir = ic_report(new_feats[col], df["y_dir"], col, "y_dir")
        ics[col] = (rho_ret, rho_dir)

    print("\n=== Train/Test split IC (50/50) ===")
    mid = len(df) // 2
    for col in new_feats.columns:
        f = new_feats[col]
        rt = ic_report(f.iloc[:mid], df["y_return_4h"].iloc[:mid], col, "y_ret[train]")
        rs = ic_report(f.iloc[mid:], df["y_return_4h"].iloc[mid:], col, "y_ret[test]")
        if rt is not None and rs is not None:
            stable = "STABLE" if np.sign(rt) == np.sign(rs) else "FLIP"
            print(f"    -> {stable}")

    print("\n=== Rolling IC (168h window) stats ===")
    for col in new_feats.columns:
        ric = rolling_ic(new_feats[col], df["y_return_4h"], window=168)
        if len(ric) == 0:
            continue
        print(f"  {col:32s}  mean={ric.mean():+.4f}  std={ric.std():.4f}  "
              f"|IC|>0.05 frac={(ric.abs() > 0.05).mean():.2%}")

    print("\n=== Reference: top existing features ===")
    for ref in ["cg_oi_close_zscore", "cg_oi_agg_close", "cg_bfx_margin_ratio",
                "vol_acceleration", "cg_funding_close_zscore",
                "cg_liq_imbalance_zscore", "oi_price_divergence"]:
        if ref in df.columns:
            ic_report(df[ref], df["y_return_4h"], ref, "y_return_4h")

    print("\n=== Correlation with most-related existing features ===")
    related_map = {
        "oi_8h_accum_z": ["cg_oi_agg_delta_zscore", "cg_oi_close_pctchg_8h", "cg_oi_close_zscore"],
        "funding_extreme_persistence": ["cg_funding_close_zscore", "funding_extreme", "funding_zscore"],
        "long_liq_exhaustion_4h": ["cg_liq_agg_imbalance_zscore", "cg_liq_long_cum_4h", "cg_liq_total_zscore"],
        "oi_price_disagreement_4h": ["oi_price_divergence", "cg_oi_agg_delta_zscore"],
        "binance_oi_share_extreme": ["cg_oi_binance_share_zscore", "cg_oi_binance_share"],
    }
    for col, refs in related_map.items():
        print(f"\n  {col}:")
        for ref in refs:
            if ref in df.columns:
                corr = new_feats[col].corr(df[ref])
                print(f"    vs {ref:35s}  corr={corr:+.3f}")

    out_path = Path("research/results/feature_proposals_oi_funding.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_feats.to_parquet(out_path)
    print(f"\nFeatures saved to {out_path}")


if __name__ == "__main__":
    main()
