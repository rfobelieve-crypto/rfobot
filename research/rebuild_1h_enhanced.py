"""
Rebuild BTC_USD_1h_enhanced.parquet with all features.

Features include:
  - Kline-derived (return, vol, taker delta, trade count)
  - Coinglass (OI, liquidation, funding, taker, L/S ratios)
  - Liquidation enhanced (surge, cascade, cumulative, asymmetry slope)
  - Large trade proxy (avg_trade_size, trade_intensity, concentration)
  - Cross-features, momentum, divergence, time encoding
"""
import pandas as pd
import numpy as np
from scipy.stats import entropy as sp_entropy

RAW = "market_data/raw_data"
OUT = "research/ml_data/BTC_USD_1h_enhanced.parquet"


def _entropy_10bin(x):
    counts, _ = np.histogram(x, bins=10)
    counts = counts + 1e-10
    probs = counts / counts.sum()
    return sp_entropy(probs)


def zscore(s, win=24):
    mu = s.rolling(win, min_periods=4).mean()
    sd = s.rolling(win, min_periods=4).std().replace(0, np.nan)
    return (s - mu) / sd


def build():
    # ── Load klines ──
    kdf = pd.read_parquet(f"{RAW}/binance_klines_1h.parquet")
    df = kdf[["open", "high", "low", "close", "volume",
              "taker_buy_vol", "taker_buy_quote", "trade_count"]].copy()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── Kline features ──
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["price_change_pct"] = df["close"].pct_change()
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

    # ── Binance quote_vol (NEW) ──
    quote_vol = pd.to_numeric(kdf.get("quote_vol", pd.Series(dtype=float)), errors="coerce")
    quote_vol = quote_vol.reindex(df.index)
    df["quote_vol_zscore"] = zscore(quote_vol)
    df["quote_vol_ratio"] = quote_vol / quote_vol.rolling(24, min_periods=4).mean().replace(0, np.nan)

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

    # OI (with ALL OHLC — NEW)
    oi = pd.read_parquet(f"{RAW}/cg_oi_1h.parquet")
    merge_cg(oi, {"close": "cg_oi_close", "open": "_oi_open",
                   "high": "_oi_high", "low": "_oi_low"})
    df["cg_oi_delta"] = df["cg_oi_close"].diff()
    df["cg_oi_accel"] = df["cg_oi_delta"].diff()

    # NEW: OI pct change at multiple windows
    for w in [4, 8, 12, 24]:
        df[f"cg_oi_close_pctchg_{w}h"] = df["cg_oi_close"].pct_change(w)

    # NEW: OI range (intra-bar OI volatility)
    df["cg_oi_range"] = df["_oi_high"] - df["_oi_low"]
    df["cg_oi_range_zscore"] = zscore(df["cg_oi_range"])
    df["cg_oi_range_pct"] = df["cg_oi_range"] / df["cg_oi_close"].replace(0, np.nan)

    # NEW: OI upper shadow (OI tried to go higher but pulled back)
    df["cg_oi_upper_shadow"] = df["_oi_high"] - np.maximum(df["_oi_open"], df["cg_oi_close"])

    # Clean up temp cols
    df.drop(columns=["_oi_open", "_oi_high", "_oi_low"], errors="ignore", inplace=True)

    # OI aggregated
    oi_agg = pd.read_parquet(f"{RAW}/cg_oi_agg_1h.parquet")
    merge_cg(oi_agg, {"close": "cg_oi_agg_close"})
    df["cg_oi_agg_delta"] = df["cg_oi_agg_close"].diff()
    df["cg_oi_binance_share"] = (
        df["cg_oi_close"] / df["cg_oi_agg_close"].replace(0, np.nan)
    )
    # NEW: Binance share z-score
    df["cg_oi_binance_share_zscore"] = zscore(df["cg_oi_binance_share"])

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

    # Global Long/Short
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

    # Z-scores (24h)
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

    # Volume dynamics
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

    # ── Absorption / Delta-Price Divergence ──
    if "realized_vol_20b" in df.columns:
        expected_move = df["realized_vol_20b"] * 0.5
        price_return_abs = df["log_return"].abs()
        under_reaction = (1 - price_return_abs / expected_move.replace(0, np.nan)).clip(-2, 2)
        taker_delta_usd = taker_delta * df["close"]
        df["absorption_score"] = taker_delta_usd * under_reaction
        df["absorption_zscore"] = zscore(df["absorption_score"])
        df["absorption_ma_4h"] = df["absorption_score"].rolling(4, min_periods=2).mean()
        df["absorption_cumsum_4h"] = df["absorption_score"].rolling(4, min_periods=1).sum()

    # ── Liquidation enhanced features ──
    if "cg_liq_total" in df.columns:
        liq_ma = df["cg_liq_total"].rolling(24, min_periods=4).mean().replace(0, np.nan)
        df["cg_liq_surge"] = df["cg_liq_total"] / liq_ma

        df["cg_liq_long_cum_4h"] = df.get("cg_liq_long", 0).rolling(4, min_periods=1).sum()
        df["cg_liq_short_cum_4h"] = df.get("cg_liq_short", 0).rolling(4, min_periods=1).sum()
        df["cg_liq_long_cum_8h"] = df.get("cg_liq_long", 0).rolling(8, min_periods=1).sum()
        df["cg_liq_short_cum_8h"] = df.get("cg_liq_short", 0).rolling(8, min_periods=1).sum()

        # Cascade: consecutive bars with above-average liquidation
        above_avg = (df["cg_liq_total"] > liq_ma).astype(int)
        reset = above_avg.eq(0).cumsum()
        df["cg_liq_cascade"] = above_avg.groupby(reset).cumsum()

        # Imbalance slope
        if "cg_liq_imbalance" in df.columns:
            df["cg_liq_imbalance_slope_4h"] = df["cg_liq_imbalance"].rolling(
                4, min_periods=2
            ).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0, raw=False)

        # Liquidation vs volume
        vol_usd = df["close"] * df["volume"]
        df["cg_liq_vs_vol"] = df["cg_liq_total"] / vol_usd.replace(0, np.nan)

        # Surge z-score
        df["cg_liq_surge_zscore"] = zscore(df["cg_liq_surge"])

    # ── Large trade proxy (from kline trade_count / volume) ──
    if "trade_count" in df.columns and "volume" in df.columns:
        tc = df["trade_count"].replace(0, np.nan)
        df["avg_trade_size"] = df["volume"] / tc
        df["avg_trade_size_zscore"] = zscore(df["avg_trade_size"])

        if "quote_vol" in kdf.columns:
            qv = pd.to_numeric(kdf["quote_vol"], errors="coerce").reindex(df.index)
            df["avg_trade_notional"] = qv / tc
            df["avg_trade_notional_zscore"] = zscore(df["avg_trade_notional"])

        df["trade_intensity"] = tc / df["volume"].replace(0, np.nan)
        df["trade_intensity_zscore"] = zscore(df["trade_intensity"])

        tc_ma_4h = tc.rolling(4, min_periods=2).mean()
        tc_ma_24h = tc.rolling(24, min_periods=4).mean()
        df["trade_count_accel"] = tc_ma_4h / tc_ma_24h.replace(0, np.nan)

        if "taker_delta_ratio" in df.columns:
            df["large_taker_signal"] = (
                df["avg_trade_size_zscore"] * df["taker_delta_ratio"]
            )

    print(f"Built: {len(df)} bars x {len(df.columns)} columns")
    print(f"Range: {df.index[0]} ~ {df.index[-1]}")

    # Feature summary
    new_cols = [c for c in df.columns if any(tag in c for tag in [
        "pctchg_", "oi_range", "oi_upper_shadow", "binance_share_zscore",
        "quote_vol_zscore", "quote_vol_ratio",
        "liq_surge", "liq_long_cum", "liq_short_cum", "liq_cascade",
        "liq_imbalance_slope", "liq_vs_vol",
        "avg_trade_size", "avg_trade_notional", "trade_intensity",
        "trade_count_accel", "large_taker_signal",
    ])]
    print(f"Enhanced features: {new_cols}")

    df.to_parquet(OUT)
    print(f"Saved to {OUT}")
    return df


if __name__ == "__main__":
    build()
