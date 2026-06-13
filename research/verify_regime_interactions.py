"""
Validate IC of proposed regime-interaction features before adding to production.
Verifies they're additive (not degenerate copies of base features).
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


def main():
    df = pd.read_parquet(CACHE)

    # Build production regime
    log_ret = np.log(df["close"] / df["close"].shift(1))
    vol_24h = log_ret.rolling(24, min_periods=12).std()
    vol_pct = vol_24h.expanding(min_periods=72).rank(pct=True)
    ret_24h = df["close"].pct_change(24)

    is_bull = ((vol_pct > 0.6) & (ret_24h > 0.005)).astype(float)
    is_bear = ((vol_pct > 0.6) & (ret_24h < -0.005)).astype(float)
    is_choppy = (1 - is_bull - is_bear).clip(0, 1)
    # Set NaN for warmup so they don't pollute IC
    is_bull[vol_pct.isna()] = np.nan
    is_bear[vol_pct.isna()] = np.nan
    is_choppy[vol_pct.isna()] = np.nan

    labels = build_direction_labels(df, horizon_bars=4, k=0.5)
    df["y_return_4h"] = labels["return_4h"]

    # Build proposed interaction features
    proposed = pd.DataFrame(index=df.index)
    proposed["is_trending_bull"] = is_bull
    proposed["is_trending_bear"] = is_bear
    proposed["is_choppy"] = is_choppy

    # Sign-flipper: bfx_margin_ratio
    proposed["bfx_margin_x_bull"]   = df["cg_bfx_margin_ratio"] * is_bull
    proposed["bfx_margin_x_bear"]   = df["cg_bfx_margin_ratio"] * is_bear

    # Dead-in-bear: vol_kurtosis (works only in CHOPPY+BULL)
    proposed["vol_kurt_non_bear"]   = df["vol_kurtosis"] * (1 - is_bear)

    # Dead-in-bull: cg_oi_close_pctchg_8h (works only in CHOPPY+BEAR)
    proposed["oi_8h_non_bull"]      = df["cg_oi_close_pctchg_8h"] * (1 - is_bull)

    # Magnitude amplifier: cg_oi_agg_close in BEAR (3x stronger)
    proposed["oi_agg_close_x_bear"] = df["cg_oi_agg_close"] * is_bear

    # Sign flipper: cg_ls_ratio (positive in CHOPPY/BULL, negative in BEAR)
    proposed["ls_ratio_x_bear"]     = df["cg_ls_ratio"] * is_bear

    # vol_regime is dead in CHOPPY (only works in trending)
    proposed["vol_regime_trending"] = df["vol_regime"] * (is_bull + is_bear)

    print("=" * 80)
    print("IC of proposed regime-interaction features")
    print("=" * 80)
    print(f"\n{'Feature':32s}  {'IC':>9s}  {'p':>9s}  {'n':>5s}  vs base feature")
    print("-" * 90)

    base_map = {
        "bfx_margin_x_bull": "cg_bfx_margin_ratio",
        "bfx_margin_x_bear": "cg_bfx_margin_ratio",
        "vol_kurt_non_bear": "vol_kurtosis",
        "oi_8h_non_bull": "cg_oi_close_pctchg_8h",
        "oi_agg_close_x_bear": "cg_oi_agg_close",
        "ls_ratio_x_bear": "cg_ls_ratio",
        "vol_regime_trending": "vol_regime",
    }

    for col in proposed.columns:
        f = proposed[col]
        m = f.notna() & df["y_return_4h"].notna()
        if m.sum() < 100:
            print(f"{col:32s}  insufficient ({m.sum()})")
            continue
        rho, p = spearmanr(f[m], df.loc[m, "y_return_4h"])
        # Compare with base feature
        if col in base_map:
            base = base_map[col]
            base_m = df[base].notna() & df["y_return_4h"].notna()
            base_rho, _ = spearmanr(df.loc[base_m, base], df.loc[base_m, "y_return_4h"])
            comparison = f"vs {base} ({base_rho:+.4f})"
        else:
            comparison = ""
        print(f"{col:32s}  {rho:+9.4f}  {p:9.1e}  {int(m.sum()):5d}  {comparison}")

    # Train/test stability for the most important ones
    print("\n=== Train/Test stability ===")
    mid = len(df) // 2
    for col in proposed.columns:
        f = proposed[col]
        y = df["y_return_4h"]
        m_tr = f.iloc[:mid].notna() & y.iloc[:mid].notna()
        m_te = f.iloc[mid:].notna() & y.iloc[mid:].notna()
        if m_tr.sum() < 50 or m_te.sum() < 50:
            continue
        ic_tr, _ = spearmanr(f.iloc[:mid][m_tr], y.iloc[:mid][m_tr])
        ic_te, _ = spearmanr(f.iloc[mid:][m_te], y.iloc[mid:][m_te])
        flip = "FLIP" if (np.sign(ic_tr) != np.sign(ic_te) and abs(ic_tr) > 0.02 and abs(ic_te) > 0.02) else "STABLE"
        print(f"  {col:32s}  train={ic_tr:+.4f}  test={ic_te:+.4f}  {flip}")

    # Correlation matrix between new features (avoid redundancy)
    print("\n=== Inter-feature correlation (avoid redundancy) ===")
    interaction_cols = [c for c in proposed.columns if c not in ("is_trending_bull", "is_trending_bear", "is_choppy")]
    corr = proposed[interaction_cols].corr()
    high_corr = []
    for i, c1 in enumerate(interaction_cols):
        for c2 in interaction_cols[i + 1:]:
            v = corr.loc[c1, c2]
            if abs(v) > 0.5:
                high_corr.append((c1, c2, v))
    if high_corr:
        for a, b, v in high_corr:
            print(f"  WARN: {a} <-> {b}  corr={v:+.3f}")
    else:
        print("  All pairs |corr| < 0.5 (good — no redundancy)")


if __name__ == "__main__":
    main()
