"""
Single-feature IC scan on 1h Coinglass + Binance data (166 days).
Diagnostic: find which features actually predict 4h returns.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, entropy as sp_entropy

RAW = "market_data/raw_data"


def _entropy_10bin(x):
    counts, _ = np.histogram(x, bins=10)
    counts = counts + 1e-10
    probs = counts / counts.sum()
    return sp_entropy(probs)


def zscore(s, win=24):
    mu = s.rolling(win, min_periods=4).mean()
    sd = s.rolling(win, min_periods=4).std().replace(0, np.nan)
    return (s - mu) / sd


def build_1h_dataset():
    # ── Load klines ──
    kdf = pd.read_parquet(f"{RAW}/binance_klines_1h.parquet")
    df = kdf[["open", "high", "low", "close", "volume",
              "taker_buy_vol", "taker_buy_quote", "trade_count"]].copy()

    # ── Kline features ──
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["realized_vol_20b"] = df["log_return"].rolling(20, min_periods=4).std()
    df["return_skew"] = df["log_return"].rolling(24, min_periods=10).skew()
    df["return_kurtosis"] = df["log_return"].rolling(24, min_periods=10).apply(
        lambda x: x.kurtosis(), raw=False
    )

    sell_vol = df["volume"] - df["taker_buy_vol"]
    taker_delta = df["taker_buy_vol"] - sell_vol
    df["taker_delta_ratio"] = taker_delta / df["volume"].replace(0, np.nan)

    for w, suffix in [(4, "_4h"), (24, "_24h")]:
        df[f"taker_delta_ma{suffix}"] = taker_delta.rolling(w, min_periods=2).mean()
        df[f"taker_delta_std{suffix}"] = taker_delta.rolling(w, min_periods=2).std()
        df[f"volume_ma{suffix}"] = df["volume"].rolling(w, min_periods=2).mean()

    for lag in range(1, 11):
        df[f"return_lag_{lag}"] = df["log_return"].shift(lag)

    # Time features
    hour = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # ── Merge Coinglass ──
    def merge_cg(cg_df, col_map):
        for src, tgt in col_map.items():
            if src in cg_df.columns:
                merged = pd.merge_asof(
                    df[["close"]].reset_index(),
                    cg_df[[src]].reset_index().rename(columns={src: tgt}),
                    left_on="dt", right_on="dt", direction="backward",
                ).set_index("dt")
                df[tgt] = merged[tgt]
            else:
                df[tgt] = np.nan

    # OI
    oi = pd.read_parquet(f"{RAW}/cg_oi_1h.parquet")
    merge_cg(oi, {"close": "cg_oi_close"})
    df["cg_oi_delta"] = df["cg_oi_close"].diff()
    df["cg_oi_accel"] = df["cg_oi_delta"].diff()

    # OI aggregated
    oi_agg = pd.read_parquet(f"{RAW}/cg_oi_agg_1h.parquet")
    merge_cg(oi_agg, {"close": "cg_oi_agg_close"})
    df["cg_oi_agg_delta"] = df["cg_oi_agg_close"].diff()
    df["cg_oi_binance_share"] = (
        df["cg_oi_close"] / df["cg_oi_agg_close"].replace(0, np.nan)
    )

    # Liquidation
    liq = pd.read_parquet(f"{RAW}/cg_liquidation_1h.parquet")
    liq_map = {}
    for src, tgt in [
        ("long_liquidation_usd", "cg_liq_long"),
        ("short_liquidation_usd", "cg_liq_short"),
    ]:
        if src in liq.columns:
            liq_map[src] = tgt
    merge_cg(liq, liq_map)
    total_liq = (df.get("cg_liq_long", 0) + df.get("cg_liq_short", 0)).replace(0, np.nan)
    df["cg_liq_total"] = total_liq
    df["cg_liq_ratio"] = df.get("cg_liq_long", 0) / total_liq
    df["cg_liq_imbalance"] = (
        (df.get("cg_liq_long", 0) - df.get("cg_liq_short", 0)) / total_liq
    )

    # Long/Short (top trader)
    ls = pd.read_parquet(f"{RAW}/cg_long_short_1h.parquet")
    ls_map = {}
    for src, tgt in [
        ("top_account_long_percent", "cg_ls_long_pct"),
        ("top_account_short_percent", "cg_ls_short_pct"),
    ]:
        if src in ls.columns:
            ls_map[src] = tgt
    merge_cg(ls, ls_map)
    df["cg_ls_ratio"] = df.get("cg_ls_long_pct", 0) / df.get(
        "cg_ls_short_pct", pd.Series(np.nan)
    ).replace(0, np.nan)

    # Global Long/Short (NEW)
    gls = pd.read_parquet(f"{RAW}/cg_global_ls_1h.parquet")
    gls_map = {}
    for src, tgt in [
        ("global_account_long_percent", "cg_gls_long_pct"),
        ("global_account_short_percent", "cg_gls_short_pct"),
    ]:
        if src in gls.columns:
            gls_map[src] = tgt
    merge_cg(gls, gls_map)
    df["cg_gls_ratio"] = df.get("cg_gls_long_pct", 0) / df.get(
        "cg_gls_short_pct", pd.Series(np.nan)
    ).replace(0, np.nan)
    # Top vs Global divergence
    if "cg_ls_ratio" in df.columns and "cg_gls_ratio" in df.columns:
        df["cg_ls_divergence"] = df["cg_ls_ratio"] - df["cg_gls_ratio"]

    # Funding
    funding = pd.read_parquet(f"{RAW}/cg_funding_1h.parquet")
    merge_cg(funding, {"close": "cg_funding_close"})
    if "high" in funding.columns and "low" in funding.columns:
        merge_cg(funding, {"high": "_fh", "low": "_fl"})
        df["cg_funding_range"] = df.get("_fh", 0) - df.get("_fl", 0)
        df.drop(columns=["_fh", "_fl"], errors="ignore", inplace=True)

    # Taker
    taker = pd.read_parquet(f"{RAW}/cg_taker_1h.parquet")
    taker_map = {}
    for src, tgt in [
        ("taker_buy_volume_usd", "cg_taker_buy"),
        ("taker_sell_volume_usd", "cg_taker_sell"),
    ]:
        if src in taker.columns:
            taker_map[src] = tgt
    merge_cg(taker, taker_map)
    df["cg_taker_delta"] = df.get("cg_taker_buy", 0) - df.get("cg_taker_sell", 0)
    total_taker = (
        df.get("cg_taker_buy", 0) + df.get("cg_taker_sell", 0)
    ).replace(0, np.nan)
    df["cg_taker_ratio"] = df.get("cg_taker_buy", 0) / total_taker

    # Z-scores (window=24 for 1h = 24h lookback)
    for col in [
        "cg_oi_delta", "cg_oi_agg_delta", "cg_liq_imbalance",
        "cg_ls_ratio", "cg_taker_delta", "cg_funding_close",
        "cg_oi_close", "cg_liq_total", "cg_gls_ratio", "cg_ls_divergence",
    ]:
        if col in df.columns:
            df[f"{col}_zscore"] = zscore(df[col])

    # Cross features
    df["cg_liq_x_oi"] = df.get("cg_liq_total", 0) / df["cg_oi_close"].replace(0, np.nan)
    df["cg_crowding"] = df.get("cg_funding_close", 0) * df.get("cg_ls_ratio", 0)
    df["cg_conviction"] = (
        df.get("cg_taker_delta_zscore", 0) * df.get("cg_oi_delta_zscore", 0)
    )

    # Momentum
    for col in ["cg_oi_delta", "cg_taker_delta", "cg_funding_close"]:
        if col in df.columns:
            df[f"{col}_slope_4h"] = df[col].rolling(4, min_periods=2).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0,
                raw=False,
            )
            df[f"{col}_mom_1h"] = df[col] - df[col].shift(1)

    # OI-price divergence
    if "cg_oi_delta_zscore" in df.columns:
        ret_cum = df["log_return"].rolling(4, min_periods=2).sum()
        ret_z = zscore(ret_cum)
        df["oi_price_divergence"] = df["cg_oi_delta_zscore"] - ret_z

    # Funding-taker alignment
    if "cg_funding_close_zscore" in df.columns and "cg_taker_delta_zscore" in df.columns:
        df["funding_taker_align"] = (
            df["cg_funding_close_zscore"] * df["cg_taker_delta_zscore"]
        )

    # Vol regime
    df["vol_regime"] = df["realized_vol_20b"] / df[
        "realized_vol_20b"
    ].rolling(24, min_periods=10).mean().replace(0, np.nan)

    # ── Volume dynamics (IC-validated) ──
    vol_ma_4h = df["volume"].rolling(4, min_periods=2).mean()
    vol_ma_24h = df["volume"].rolling(24, min_periods=4).mean()
    df["vol_acceleration"] = vol_ma_4h / vol_ma_24h.replace(0, np.nan)
    df["vol_kurtosis"] = df["volume"].rolling(24, min_periods=10).apply(
        lambda x: x.kurtosis(), raw=False
    )
    df["vol_entropy"] = df["volume"].rolling(24, min_periods=10).apply(
        lambda x: _entropy_10bin(x), raw=False
    )

    # Squeeze proxy
    if "cg_funding_close_zscore" in df.columns and "cg_oi_delta_zscore" in df.columns:
        cvd_delta_z = zscore(taker_delta.diff())
        df["squeeze_proxy"] = (
            df["cg_funding_close_zscore"].abs() *
            df["cg_oi_delta_zscore"].abs() *
            np.sign(-df["cg_funding_close_zscore"] * cvd_delta_z)
        )

    # ── Target ──
    HORIZON = 4  # 4h = 4 × 1h bars
    df["y_return_4h"] = df["close"].shift(-HORIZON) / df["close"] - 1
    df = df.iloc[:-HORIZON]
    df = df.dropna(subset=["y_return_4h"])

    return df


def ic_scan(df):
    EXCLUDE = {
        "open", "high", "low", "close", "volume",
        "taker_buy_vol", "taker_buy_quote", "trade_count",
        "ts_open", "y_return_4h",
    }

    feat_cols = [
        c for c in df.columns
        if c not in EXCLUDE and df[c].dtype in ["float64", "float32", "int64"]
    ]

    print(f"Dataset: {len(df)} bars x {len(feat_cols)} features")
    print(f"Target std: {df['y_return_4h'].std():.4f}")
    print(f"Range: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print()

    results = []
    for col in feat_cols:
        valid = df[[col, "y_return_4h"]].dropna()
        if len(valid) < 100:
            continue
        ic, p = spearmanr(valid[col], valid["y_return_4h"])
        results.append((col, ic, p, len(valid)))

    results.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"{'Feature':45s} {'IC':>8s} {'p-value':>10s} {'n':>6s}")
    print("=" * 75)

    count_sig = 0
    for col, ic, p, n in results:
        sig = "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "   "
        if abs(ic) >= 0.02:
            count_sig += 1
        bar_len = int(abs(ic) * 200)
        bar = "#" * min(bar_len, 30)
        print(f"{col:45s} {ic:+.4f}   {p:.2e}  {n:5d} {sig} {bar}")

    print()
    print(f"Features with |IC| >= 0.02: {count_sig}")
    print(f"Features with |IC| >= 0.03: {sum(1 for _, ic, _, _ in results if abs(ic) >= 0.03)}")
    print(f"Features with |IC| >= 0.05: {sum(1 for _, ic, _, _ in results if abs(ic) >= 0.05)}")


if __name__ == "__main__":
    df = build_1h_dataset()
    ic_scan(df)
