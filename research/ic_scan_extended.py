"""
Extended IC scan — test new feature candidates from user's feature list.

New features (computable from 1h klines + Coinglass API):

Category 1 - Basic trade:
  - avg_trade_size (volume / trade_count)

Category 2 - Order flow (CVD/VPIN from taker_buy_vol):
  - cvd_rolling_24h (cumulative taker delta, rolling 24h)
  - cvd_zscore
  - cvd_delta (ΔCVD = change in CVD)
  - cvd_accel (CVD acceleration)
  - vpin (|buy - sell| / volume, rolling)

Category 3 - Volume dynamics:
  - vol_acceleration (short/long MA ratio)
  - price_impact (|return| / volume)
  - vol_entropy (rolling entropy of volume distribution)

Category 4 - OI dynamics:
  - oi_change_pct (ΔOI / OI)
  - oi_taker_ratio (ΔOI / taker_delta)

Category 5 - Funding interactions:
  - cumul_funding_8h (rolling 8h sum)
  - funding_x_taker (funding × taker_delta)
  - funding_x_oi_delta (funding × ΔOI)
  - funding_extreme (binary: |funding_zscore| > 2)

Category 6 - Cross-variable:
  - squeeze_proxy (high funding + high OI + CVD reversal)
  - price_new_high_oi_div (price makes high but OI declining)
  - liq_oi_acceleration (liq spike + OI acceleration)

Category 7 - Statistical:
  - vol_kurtosis (rolling kurtosis of volume)
  - delta_entropy (entropy of taker delta distribution)

Category 8 - Multi-scale CG:
  - CG features slope/momentum at multiple windows
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, entropy as sp_entropy

RAW = "market_data/raw_data"


def zscore(s, win=24):
    mu = s.rolling(win, min_periods=4).mean()
    sd = s.rolling(win, min_periods=4).std().replace(0, np.nan)
    return (s - mu) / sd


def rolling_entropy(s, win=24):
    """Rolling entropy of a series (binned into 10 quantiles)."""
    result = pd.Series(np.nan, index=s.index)
    for i in range(win, len(s)):
        window = s.iloc[i - win:i].dropna()
        if len(window) < 10:
            continue
        try:
            counts, _ = np.histogram(window, bins=10)
            counts = counts + 1e-10  # avoid log(0)
            probs = counts / counts.sum()
            result.iloc[i] = sp_entropy(probs)
        except Exception:
            pass
    return result


def build_extended_dataset():
    """Build dataset with all existing + new candidate features."""

    # ── Load base dataset (reuse ic_scan_1h logic) ──
    from research.ic_scan_1h import build_1h_dataset
    df = build_1h_dataset()

    # ── Raw series needed for new features ──
    kdf = pd.read_parquet(f"{RAW}/binance_klines_1h.parquet")
    taker_buy = kdf["taker_buy_vol"].reindex(df.index)
    taker_sell = (kdf["volume"] - kdf["taker_buy_vol"]).reindex(df.index)
    taker_delta = (taker_buy - taker_sell).reindex(df.index)
    volume = kdf["volume"].reindex(df.index)
    trade_count = kdf["trade_count"].reindex(df.index)

    # ════════════════════════════════════════════════════════════════════
    # Category 1: Basic trade features
    # ════════════════════════════════════════════════════════════════════
    df["avg_trade_size"] = volume / trade_count.replace(0, np.nan)
    df["avg_trade_size_zscore"] = zscore(df["avg_trade_size"])

    # ════════════════════════════════════════════════════════════════════
    # Category 2: CVD & VPIN (from taker_buy_vol)
    # ════════════════════════════════════════════════════════════════════
    # CVD = rolling cumulative taker delta (24h window)
    df["cvd_24h"] = taker_delta.rolling(24, min_periods=4).sum()
    df["cvd_zscore"] = zscore(df["cvd_24h"])
    df["cvd_delta"] = df["cvd_24h"].diff()
    df["cvd_accel"] = df["cvd_delta"].diff()
    df["cvd_delta_zscore"] = zscore(df["cvd_delta"])

    # VPIN approximation = |buy - sell| / total volume, rolling average
    abs_imbalance = (taker_buy - taker_sell).abs()
    df["vpin_raw"] = abs_imbalance / volume.replace(0, np.nan)
    df["vpin_24h"] = df["vpin_raw"].rolling(24, min_periods=4).mean()
    df["vpin_zscore"] = zscore(df["vpin_24h"])
    # VPIN extreme flag
    df["vpin_extreme"] = (df["vpin_zscore"].abs() > 2).astype(float)

    # ════════════════════════════════════════════════════════════════════
    # Category 3: Volume dynamics
    # ════════════════════════════════════════════════════════════════════
    vol_ma_4h = volume.rolling(4, min_periods=2).mean()
    vol_ma_24h = volume.rolling(24, min_periods=4).mean()
    df["vol_acceleration"] = vol_ma_4h / vol_ma_24h.replace(0, np.nan)

    # Price impact = |return| / log(volume)
    df["price_impact"] = df["log_return"].abs() / np.log1p(volume).replace(0, np.nan)
    df["price_impact_zscore"] = zscore(df["price_impact"])

    # Volume entropy (rolling 24h)
    df["vol_entropy"] = rolling_entropy(volume, win=24)

    # ════════════════════════════════════════════════════════════════════
    # Category 4: OI dynamics (extended)
    # ════════════════════════════════════════════════════════════════════
    if "cg_oi_delta" in df.columns and "cg_oi_close" in df.columns:
        df["oi_change_pct"] = df["cg_oi_delta"] / df["cg_oi_close"].shift(1).replace(0, np.nan)
        df["oi_change_pct_zscore"] = zscore(df["oi_change_pct"])

    if "cg_oi_delta" in df.columns:
        # OI / taker_delta ratio (unit position pressure)
        df["oi_taker_ratio"] = df["cg_oi_delta"] / taker_delta.replace(0, np.nan)
        df["oi_taker_ratio"] = df["oi_taker_ratio"].clip(-10, 10)
        df["oi_taker_ratio_zscore"] = zscore(df["oi_taker_ratio"])

        # OI + CVD divergence (OI up but CVD down = squeeze setup)
        if "cvd_zscore" in df.columns and "cg_oi_delta_zscore" in df.columns:
            df["oi_cvd_divergence"] = df["cg_oi_delta_zscore"] - df["cvd_zscore"]

    # ════════════════════════════════════════════════════════════════════
    # Category 5: Funding interactions
    # ════════════════════════════════════════════════════════════════════
    if "cg_funding_close" in df.columns:
        # Cumulative funding (8h rolling sum)
        df["cumul_funding_8h"] = df["cg_funding_close"].rolling(8, min_periods=2).sum()
        df["cumul_funding_8h_zscore"] = zscore(df["cumul_funding_8h"])

        # Funding × taker_delta (directional pressure amplified by crowding)
        df["funding_x_taker"] = df["cg_funding_close"] * taker_delta
        df["funding_x_taker_zscore"] = zscore(df["funding_x_taker"])

        # Funding × ΔOI (crowding acceleration)
        if "cg_oi_delta" in df.columns:
            df["funding_x_oi_delta"] = df["cg_funding_close"] * df["cg_oi_delta"]
            df["funding_x_oi_delta_zscore"] = zscore(df["funding_x_oi_delta"])

        # Funding extreme flag
        if "cg_funding_close_zscore" in df.columns:
            df["funding_extreme"] = (df["cg_funding_close_zscore"].abs() > 2).astype(float)

    # ════════════════════════════════════════════════════════════════════
    # Category 6: Cross-variable composites
    # ════════════════════════════════════════════════════════════════════
    # Squeeze proxy: extreme funding + rising OI + CVD reversal
    if all(c in df.columns for c in ["cg_funding_close_zscore", "cg_oi_delta_zscore", "cvd_delta_zscore"]):
        df["squeeze_proxy"] = (
            df["cg_funding_close_zscore"].abs() *
            df["cg_oi_delta_zscore"].abs() *
            np.sign(-df["cg_funding_close_zscore"] * df["cvd_delta_zscore"])
        )

    # Price new high but OI declining divergence
    if "cg_oi_delta" in df.columns:
        price_high_4h = df["close"].rolling(4, min_periods=2).max()
        price_at_high = (df["close"] >= price_high_4h * 0.999).astype(float)
        oi_declining = (df["cg_oi_delta"].rolling(4, min_periods=2).mean() < 0).astype(float)
        df["price_high_oi_div"] = price_at_high * oi_declining  # 1 when diverging

        # Same for lows
        price_low_4h = df["close"].rolling(4, min_periods=2).min()
        price_at_low = (df["close"] <= price_low_4h * 1.001).astype(float)
        oi_rising = (df["cg_oi_delta"].rolling(4, min_periods=2).mean() > 0).astype(float)
        df["price_low_oi_div"] = price_at_low * oi_rising

    # Liquidation + OI acceleration composite
    if "cg_liq_total" in df.columns and "cg_oi_accel" in df.columns:
        liq_z = zscore(df["cg_liq_total"])
        df["liq_oi_accel_composite"] = liq_z * df["cg_oi_accel"]

    # ════════════════════════════════════════════════════════════════════
    # Category 7: Statistical features
    # ════════════════════════════════════════════════════════════════════
    df["vol_kurtosis"] = volume.rolling(24, min_periods=10).apply(
        lambda x: x.kurtosis(), raw=False
    )

    df["delta_entropy"] = rolling_entropy(taker_delta, win=24)

    # ════════════════════════════════════════════════════════════════════
    # Category 8: Multi-scale CG features
    # ════════════════════════════════════════════════════════════════════
    for col in ["cg_oi_delta", "cg_taker_delta", "cg_funding_close", "cg_ls_ratio"]:
        if col not in df.columns:
            continue
        # 12h slope (medium term)
        df[f"{col}_slope_12h"] = df[col].rolling(12, min_periods=4).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0,
            raw=False,
        )
        # 24h momentum
        df[f"{col}_mom_24h"] = df[col] - df[col].shift(24)

    # Funding cumulative cost (rolling 24h)
    if "cg_funding_close" in df.columns:
        df["cumul_funding_24h"] = df["cg_funding_close"].rolling(24, min_periods=4).sum()

    return df


def ic_scan_new_only(df):
    """IC scan focusing on NEW features (not in original ic_scan_1h)."""
    NEW_FEATURES = [
        # Cat 1
        "avg_trade_size", "avg_trade_size_zscore",
        # Cat 2
        "cvd_24h", "cvd_zscore", "cvd_delta", "cvd_accel", "cvd_delta_zscore",
        "vpin_raw", "vpin_24h", "vpin_zscore", "vpin_extreme",
        # Cat 3
        "vol_acceleration", "price_impact", "price_impact_zscore", "vol_entropy",
        # Cat 4
        "oi_change_pct", "oi_change_pct_zscore", "oi_taker_ratio",
        "oi_taker_ratio_zscore", "oi_cvd_divergence",
        # Cat 5
        "cumul_funding_8h", "cumul_funding_8h_zscore",
        "funding_x_taker", "funding_x_taker_zscore",
        "funding_x_oi_delta", "funding_x_oi_delta_zscore",
        "funding_extreme",
        # Cat 6
        "squeeze_proxy", "price_high_oi_div", "price_low_oi_div",
        "liq_oi_accel_composite",
        # Cat 7
        "vol_kurtosis", "delta_entropy",
        # Cat 8
        "cg_oi_delta_slope_12h", "cg_taker_delta_slope_12h",
        "cg_funding_close_slope_12h", "cg_ls_ratio_slope_12h",
        "cg_oi_delta_mom_24h", "cg_taker_delta_mom_24h",
        "cg_funding_close_mom_24h", "cg_ls_ratio_mom_24h",
        "cumul_funding_24h",
    ]

    target = "y_return_4h"
    feat_cols = [c for c in NEW_FEATURES if c in df.columns]

    print(f"Testing {len(feat_cols)} NEW feature candidates")
    print(f"Dataset: {len(df)} bars, target std={df[target].std():.4f}")
    print()

    results = []
    for col in feat_cols:
        valid = df[[col, target]].dropna()
        if len(valid) < 100:
            continue
        ic, p = spearmanr(valid[col], valid[target])
        results.append((col, ic, p, len(valid)))

    results.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"{'Feature':45s} {'IC':>8s} {'p-value':>10s} {'n':>6s}")
    print("=" * 75)

    good = []
    for col, ic, p, n in results:
        sig = "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "   "
        bar_len = int(abs(ic) * 200)
        bar = "#" * min(bar_len, 30)
        marker = " <-- KEEP" if abs(ic) >= 0.025 and p < 0.05 else ""
        print(f"{col:45s} {ic:+.4f}   {p:.2e}  {n:5d} {sig} {bar}{marker}")
        if abs(ic) >= 0.025 and p < 0.05:
            good.append(col)

    print()
    print(f"Features with |IC| >= 0.025 & p < 0.05: {len(good)}")
    for f in good:
        print(f"  + {f}")

    return good


if __name__ == "__main__":
    print("Building extended dataset with new features...")
    df = build_extended_dataset()
    print(f"Total columns: {len(df.columns)}")
    print()
    good = ic_scan_new_only(df)
