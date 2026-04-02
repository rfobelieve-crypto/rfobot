"""
Extended IC scan — test ALL unused Coinglass fields + interaction features.
Focus: OI OHLC, funding open, interaction combos.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

RAW = "market_data/raw_data"


def zscore(s, win=24):
    mu = s.rolling(win, min_periods=4).mean()
    sd = s.rolling(win, min_periods=4).std().replace(0, np.nan)
    return (s - mu) / sd


def build_extended():
    # ── Base klines ──
    kdf = pd.read_parquet(f"{RAW}/binance_klines_1h.parquet")
    df = kdf[["open", "high", "low", "close", "volume",
              "taker_buy_vol", "taker_buy_quote", "quote_vol"]].copy()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["realized_vol_20b"] = df["log_return"].rolling(20, min_periods=4).std()

    # ── Binance unused: quote_vol ──
    df["quote_vol_zscore"] = zscore(df["quote_vol"])
    df["quote_vol_ratio"] = df["quote_vol"] / df["quote_vol"].rolling(24, min_periods=4).mean()

    # ── Helper: merge CG data ──
    def merge_cg(cg_df, col_map):
        for src, tgt in col_map.items():
            if src in cg_df.columns:
                merged = pd.merge_asof(
                    df[["close"]].reset_index(),
                    cg_df[[src]].reset_index().rename(columns={src: tgt}),
                    left_on="dt", right_on="dt", direction="backward",
                ).set_index("dt")
                df[tgt] = merged[tgt]

    # ══════════════════════════════════════════════════════════════
    # OI — USE ALL OHLC
    # ══════════════════════════════════════════════════════════════
    oi = pd.read_parquet(f"{RAW}/cg_oi_1h.parquet")
    merge_cg(oi, {"open": "oi_open", "high": "oi_high", "low": "oi_low", "close": "oi_close"})

    # OI range (intra-bar volatility of open interest)
    df["oi_range"] = df["oi_high"] - df["oi_low"]
    df["oi_range_pct"] = df["oi_range"] / df["oi_close"].replace(0, np.nan)
    df["oi_range_zscore"] = zscore(df["oi_range"])

    # OI bar change (open→close within bar)
    df["oi_bar_change"] = df["oi_close"] - df["oi_open"]
    df["oi_bar_change_pct"] = df["oi_bar_change"] / df["oi_open"].replace(0, np.nan)

    # OI close (existing strong feature)
    df["oi_close_zscore"] = zscore(df["oi_close"])
    df["oi_delta"] = df["oi_close"].diff()
    df["oi_delta_zscore"] = zscore(df["oi_delta"])

    # OI upper/lower shadow (similar to candle wicks)
    df["oi_upper_shadow"] = df["oi_high"] - np.maximum(df["oi_open"], df["oi_close"])
    df["oi_lower_shadow"] = np.minimum(df["oi_open"], df["oi_close"]) - df["oi_low"]
    df["oi_shadow_ratio"] = df["oi_upper_shadow"] / (df["oi_lower_shadow"] + 1e-10)

    # OI momentum (different windows)
    for w in [4, 8, 12, 24]:
        df[f"oi_close_pctchg_{w}h"] = df["oi_close"].pct_change(w)

    # ══════════════════════════════════════════════════════════════
    # OI AGGREGATED — USE ALL OHLC
    # ══════════════════════════════════════════════════════════════
    oi_agg = pd.read_parquet(f"{RAW}/cg_oi_agg_1h.parquet")
    merge_cg(oi_agg, {"open": "oi_agg_open", "high": "oi_agg_high",
                       "low": "oi_agg_low", "close": "oi_agg_close"})

    df["oi_agg_range"] = df["oi_agg_high"] - df["oi_agg_low"]
    df["oi_agg_range_pct"] = df["oi_agg_range"] / df["oi_agg_close"].replace(0, np.nan)
    df["oi_agg_bar_change"] = df["oi_agg_close"] - df["oi_agg_open"]
    df["oi_agg_close_zscore"] = zscore(df["oi_agg_close"])

    # Binance share
    df["oi_binance_share"] = df["oi_close"] / df["oi_agg_close"].replace(0, np.nan)
    df["oi_binance_share_zscore"] = zscore(df["oi_binance_share"])
    df["oi_binance_share_delta"] = df["oi_binance_share"].diff()

    # ══════════════════════════════════════════════════════════════
    # FUNDING — USE open (funding open→close change)
    # ══════════════════════════════════════════════════════════════
    funding = pd.read_parquet(f"{RAW}/cg_funding_1h.parquet")
    merge_cg(funding, {"open": "funding_open", "high": "funding_high",
                        "low": "funding_low", "close": "funding_close"})

    df["funding_bar_change"] = df["funding_close"] - df["funding_open"]
    df["funding_range"] = df["funding_high"] - df["funding_low"]
    df["funding_range_zscore"] = zscore(df["funding_range"])
    df["funding_close_zscore"] = zscore(df["funding_close"])
    df["funding_bar_change_zscore"] = zscore(df["funding_bar_change"])

    # Funding extremes
    df["funding_abs"] = df["funding_close"].abs()
    df["funding_abs_zscore"] = zscore(df["funding_abs"])

    # ══════════════════════════════════════════════════════════════
    # LIQUIDATION
    # ══════════════════════════════════════════════════════════════
    liq = pd.read_parquet(f"{RAW}/cg_liquidation_1h.parquet")
    for src, tgt in [("long_liquidation_usd", "liq_long"), ("short_liquidation_usd", "liq_short")]:
        if src in liq.columns:
            merge_cg(liq, {src: tgt})
    df["liq_total"] = df.get("liq_long", 0) + df.get("liq_short", 0)
    df["liq_imbalance"] = (df.get("liq_long", 0) - df.get("liq_short", 0)) / df["liq_total"].replace(0, np.nan)

    # ══════════════════════════════════════════════════════════════
    # L/S RATIO
    # ══════════════════════════════════════════════════════════════
    ls = pd.read_parquet(f"{RAW}/cg_long_short_1h.parquet")
    for src, tgt in [("top_account_long_percent", "ls_long_pct"),
                     ("top_account_short_percent", "ls_short_pct")]:
        if src in ls.columns:
            merge_cg(ls, {src: tgt})
    df["ls_ratio"] = df.get("ls_long_pct", 0) / df.get("ls_short_pct", pd.Series(np.nan)).replace(0, np.nan)

    gls = pd.read_parquet(f"{RAW}/cg_global_ls_1h.parquet")
    for src, tgt in [("global_account_long_percent", "gls_long_pct"),
                     ("global_account_short_percent", "gls_short_pct")]:
        if src in gls.columns:
            merge_cg(gls, {src: tgt})
    df["gls_ratio"] = df.get("gls_long_pct", 0) / df.get("gls_short_pct", pd.Series(np.nan)).replace(0, np.nan)
    df["ls_divergence"] = df["ls_ratio"] - df["gls_ratio"]

    # ══════════════════════════════════════════════════════════════
    # TAKER
    # ══════════════════════════════════════════════════════════════
    taker = pd.read_parquet(f"{RAW}/cg_taker_1h.parquet")
    for src, tgt in [("taker_buy_volume_usd", "taker_buy"), ("taker_sell_volume_usd", "taker_sell")]:
        if src in taker.columns:
            merge_cg(taker, {src: tgt})
    df["taker_delta"] = df.get("taker_buy", 0) - df.get("taker_sell", 0)
    df["taker_delta_zscore"] = zscore(df["taker_delta"])

    # ══════════════════════════════════════════════════════════════
    # INTERACTION FEATURES (CROSS)
    # ══════════════════════════════════════════════════════════════

    # OI level × funding (high OI + high funding = extreme leverage)
    df["oi_x_funding"] = df["oi_close_zscore"] * df["funding_close_zscore"]

    # OI level × volume (high OI + high volume = potential squeeze)
    vol_z = zscore(df["volume"])
    df["oi_x_volume"] = df["oi_close_zscore"] * vol_z

    # OI range × price range (OI volatile when price volatile?)
    price_range = df["high"] - df["low"]
    price_range_z = zscore(price_range)
    df["oi_range_x_price_range"] = zscore(df["oi_range"]) * price_range_z

    # OI level × liquidation (high OI + high liq = cascade risk)
    liq_z = zscore(df["liq_total"])
    df["oi_x_liq"] = df["oi_close_zscore"] * liq_z

    # OI level × L/S ratio (high OI + crowded long = short squeeze risk)
    ls_z = zscore(df["ls_ratio"])
    df["oi_x_ls"] = df["oi_close_zscore"] * ls_z

    # Funding × L/S (crowding metric, improved)
    df["funding_x_ls"] = df["funding_close_zscore"] * ls_z

    # Funding × OI delta (funding rising + OI increasing = leveraged chase)
    df["funding_x_oi_delta"] = df["funding_close_zscore"] * df["oi_delta_zscore"]

    # Taker × OI level (aggressive buying into high OI = dangerous)
    df["taker_x_oi"] = df["taker_delta_zscore"] * df["oi_close_zscore"]

    # Liq imbalance × OI change (liquidation direction when OI drops)
    liq_imb_z = zscore(df["liq_imbalance"])
    df["liq_imb_x_oi_delta"] = liq_imb_z * df["oi_delta_zscore"]

    # Triple: OI high + funding extreme + volume spike
    df["leverage_stress"] = (
        df["oi_close_zscore"].abs() *
        df["funding_abs_zscore"] *
        vol_z.clip(lower=0)  # only when volume above average
    )

    # OI-price divergence (improved: use pct_change not just delta)
    ret_4h = df["close"].pct_change(4)
    ret_4h_z = zscore(ret_4h)
    oi_4h_z = zscore(df["oi_close"].pct_change(4))
    df["oi_price_div_v2"] = oi_4h_z - ret_4h_z

    # Funding reversal signal (funding flips sign)
    df["funding_sign_change"] = (
        np.sign(df["funding_close"]) != np.sign(df["funding_close"].shift(1))
    ).astype(float)

    # OI washout (large OI drop = forced liquidation)
    df["oi_washout"] = df["oi_close"].pct_change(1).clip(upper=0)  # only negative
    df["oi_washout_zscore"] = zscore(df["oi_washout"])

    # ══════════════════════════════════════════════════════════════
    # TARGET
    # ══════════════════════════════════════════════════════════════
    HORIZON = 4
    df["y_return_4h"] = df["close"].shift(-HORIZON) / df["close"] - 1
    df = df.iloc[:-HORIZON]
    df = df.dropna(subset=["y_return_4h"])

    return df


def ic_scan(df):
    EXCLUDE = {
        "open", "high", "low", "close", "volume", "quote_vol",
        "taker_buy_vol", "taker_buy_quote",
        "oi_open", "oi_high", "oi_low", "oi_close",
        "oi_agg_open", "oi_agg_high", "oi_agg_low", "oi_agg_close",
        "funding_open", "funding_high", "funding_low", "funding_close",
        "liq_long", "liq_short", "taker_buy", "taker_sell",
        "ls_long_pct", "ls_short_pct", "gls_long_pct", "gls_short_pct",
        "y_return_4h",
    }

    feat_cols = [
        c for c in df.columns
        if c not in EXCLUDE and df[c].dtype in ["float64", "float32", "int64"]
    ]

    print(f"Extended IC Scan: {len(df)} bars x {len(feat_cols)} features")
    print(f"Target: y_return_4h, std={df['y_return_4h'].std():.4f}")
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
    print("=" * 80)

    for col, ic, p, n in results:
        sig = "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "   "
        bar_len = int(abs(ic) * 200)
        bar = "#" * min(bar_len, 30)
        # Mark NEW features
        new = " [NEW]" if any(tag in col for tag in [
            "oi_range", "oi_bar_change", "oi_upper", "oi_lower", "oi_shadow",
            "oi_close_pctchg", "oi_agg_range", "oi_agg_bar",
            "oi_binance_share_zscore", "oi_binance_share_delta",
            "funding_bar_change", "funding_range_zscore", "funding_abs",
            "funding_sign", "oi_washout",
            "oi_x_", "taker_x_", "funding_x_", "liq_imb_x_",
            "leverage_stress", "oi_price_div_v2",
            "quote_vol",
        ]) else ""
        print(f"{col:45s} {ic:+.4f}   {p:.2e}  {n:5d} {sig} {bar}{new}")

    print()
    new_feats = [r for r in results if any(tag in r[0] for tag in [
        "oi_range", "oi_bar_change", "oi_upper", "oi_lower", "oi_shadow",
        "oi_close_pctchg", "oi_agg_range", "oi_agg_bar",
        "oi_binance_share_zscore", "oi_binance_share_delta",
        "funding_bar_change", "funding_range_zscore", "funding_abs",
        "funding_sign", "oi_washout",
        "oi_x_", "taker_x_", "funding_x_", "liq_imb_x_",
        "leverage_stress", "oi_price_div_v2",
        "quote_vol",
    ])]

    print("=" * 80)
    print(f"NEW features tested: {len(new_feats)}")
    print(f"NEW features with |IC| >= 0.03:")
    for col, ic, p, n in new_feats:
        if abs(ic) >= 0.03:
            sig = "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "   "
            print(f"  {col:43s} {ic:+.4f} {sig}")


if __name__ == "__main__":
    df = build_extended()
    ic_scan(df)
