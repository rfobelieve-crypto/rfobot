"""
Strict walk-forward validation for all candidate features.
70/30 chronological split, NO shuffle.
Train IC vs Test IC comparison — FLIP/COLLAPSE = REJECT.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE = Path(__file__).resolve().parent / "dual_model" / ".cache" / "features_all.parquet"


def _zscore(s, win=168):
    mu = s.rolling(win, min_periods=win // 2).mean()
    sd = s.rolling(win, min_periods=win // 2).std().replace(0, np.nan)
    return (s - mu) / sd


def _zscore_short(s, win=24):
    mu = s.rolling(win, min_periods=4).mean()
    sd = s.rolling(win, min_periods=4).std().replace(0, np.nan)
    return (s - mu) / sd


def main():
    df = pd.read_parquet(CACHE)
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"])
        df = df.set_index("dt").sort_index()

    close = df["close"].astype(float)
    twap_4h = close.shift(-1).rolling(4).mean()
    y_dir = twap_4h / close - 1
    y_mag = y_dir.abs()

    # ── Build all candidate features ──
    features = {}

    fund_col = next((c for c in ["cg_funding_close", "cg_funding_rate"] if c in df.columns), None)
    oi_col = next((c for c in ["cg_oi_close", "cg_oi_agg_close"] if c in df.columns), None)
    ls_col = next((c for c in ["cg_ls_ratio", "cg_pos_ls_ratio"] if c in df.columns), None)
    taker_col = next((c for c in ["cg_taker_delta"] if c in df.columns), None)
    dvol_col = next((c for c in ["bvol_zscore_72h", "deribit_dvol"] if c in df.columns), None)
    oi_z_col = next((c for c in ["cg_oi_close_zscore", "cg_oi_delta_zscore"] if c in df.columns), None)
    fng_col = next((c for c in ["fear_greed_value", "fng_value"] if c in df.columns), None)

    if fund_col and oi_col:
        f = df[fund_col].fillna(0).astype(float)
        oi = df[oi_col].ffill().astype(float)
        oi_chg = oi.diff(1)
        oi_pct_8h = oi.pct_change(8)
        f_signed = f * np.sign(f)

        pain_24 = f_signed.rolling(24, min_periods=12).sum()
        features["funding_pain_24h"] = _zscore(pain_24 * oi_chg.rolling(24, min_periods=12).mean())

        pain_48 = f_signed.rolling(48, min_periods=24).sum()
        features["funding_pain_48h"] = _zscore(pain_48 * oi_chg.rolling(48, min_periods=24).mean())

        features["funding_squeeze"] = _zscore(f.abs() * oi_pct_8h)

    if ls_col:
        ls = df[ls_col].ffill().astype(float)
        vel = ls - ls.shift(4)
        features["ls_acceleration_8h"] = vel - vel.shift(4)

    if ls_col and fund_col:
        ls2 = df[ls_col].ffill().astype(float)
        f2 = df[fund_col].fillna(0).astype(float)
        features["ls_funding_divergence"] = _zscore((ls2 - 1.0) * (-f2) * 1000)

    if taker_col and oi_col:
        tk = df[taker_col].fillna(0).astype(float)
        oi3 = df[oi_col].ffill().astype(float)
        features["taker_oi_divergence"] = _zscore(tk * (-oi3.pct_change(1).clip(-0.1, 0.1)))

    if oi_col and "volume" in df.columns:
        oi4 = df[oi_col].ffill().astype(float)
        vol4 = df["volume"].astype(float)
        oi_chg2 = oi4.diff(1)
        vol_z = _zscore_short(vol4, 24)
        flow = np.sign(oi_chg2) * vol_z
        features["exchange_flow_8h"] = flow.rolling(8, min_periods=4).sum()
        features["oi_to_volume_zscore"] = _zscore(oi4 / vol4.replace(0, np.nan))

    if dvol_col and oi_z_col:
        dz = df[dvol_col].fillna(0).astype(float)
        oz = df[oi_z_col].fillna(0).astype(float)
        if dvol_col == "deribit_dvol":
            dz = _zscore(dz, 72)
        features["dvol_oi_interaction"] = dz * oz

    if fng_col:
        features["fng_value"] = df[fng_col].ffill().astype(float)

    # Cross-market
    try:
        import yfinance as yf
        tickers = {"SPX": "^GSPC", "DXY": "DX-Y.NYB", "US10Y": "^TNX"}
        start = df.index.min().strftime("%Y-%m-%d")
        end = (df.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        for name, ticker in tickers.items():
            try:
                data = yf.download(ticker, start=start, end=end, progress=False)
                if data.empty:
                    continue
                daily_close = data["Close"].squeeze()
                if daily_close.index.tz is None:
                    daily_close.index = daily_close.index.tz_localize("UTC")
                hourly = daily_close.reindex(df.index, method="ffill")
                features[f"{name}_return_1d"] = hourly.pct_change(24)
                mu = hourly.rolling(20 * 24, min_periods=5 * 24).mean()
                sd = hourly.rolling(20 * 24, min_periods=5 * 24).std().replace(0, np.nan)
                features[f"{name}_zscore_20d"] = (hourly - mu) / sd
                if name == "US10Y":
                    features["US10Y_return_5d"] = hourly.pct_change(5 * 24)
            except Exception as e:
                print(f"  SKIP {name}: {e}")
        spx_r = features.get("SPX_return_1d")
        dxy_r = features.get("DXY_return_1d")
        if spx_r is not None and dxy_r is not None:
            features["risk_on_composite"] = spx_r.fillna(0) - dxy_r.fillna(0)
    except ImportError:
        print("yfinance not available, skipping cross-market")

    # ── Walk-forward split ──
    n = len(df)
    split = int(n * 0.7)
    print(f"Data: {n} bars")
    print(f"Train: {df.index[0].date()} to {df.index[split].date()} ({split} bars)")
    print(f"Test:  {df.index[split].date()} to {df.index[-1].date()} ({n - split} bars)")
    print()

    # Direction
    print("=" * 110)
    print("DIRECTION TARGET (signed return)")
    print("=" * 110)
    print(f"{'Feature':<30} {'Train IC':>10} {'Test IC':>10} {'Ratio':>8} {'Status':>10} {'Full IC':>10} {'Verdict':>10}")
    print("-" * 110)

    dir_pass = []
    dir_reject = []

    for fname in sorted(features.keys()):
        feat = features[fname]
        mask_all = feat.notna() & y_dir.notna()
        idx = np.arange(n)
        mask_train = mask_all & (idx < split)
        mask_test = mask_all & (idx >= split)

        if mask_train.sum() < 100 or mask_test.sum() < 50:
            continue

        ic_train, _ = spearmanr(feat[mask_train], y_dir[mask_train])
        ic_test, _ = spearmanr(feat[mask_test], y_dir[mask_test])
        ic_full, _ = spearmanr(feat[mask_all], y_dir[mask_all])

        if np.isnan(ic_train) or np.isnan(ic_test):
            continue

        ratio = ic_test / ic_train if abs(ic_train) > 1e-6 else 0

        if np.sign(ic_train) != np.sign(ic_test):
            status = "FLIP"
        elif abs(ic_test) < abs(ic_train) * 0.3:
            status = "COLLAPSE"
        elif abs(ic_test) >= abs(ic_train) * 0.7:
            status = "STABLE"
        else:
            status = "DECAY"

        if status in ("FLIP", "COLLAPSE"):
            verdict = "REJECT"
            dir_reject.append(fname)
        elif abs(ic_test) >= 0.02:
            verdict = "PASS"
            dir_pass.append(fname)
        else:
            verdict = "WEAK"

        print(f"{fname:<30} {ic_train:>+10.4f} {ic_test:>+10.4f} {ratio:>8.2f} {status:>10} {ic_full:>+10.4f} {verdict:>10}")

    # Magnitude
    print()
    print("=" * 110)
    print("MAGNITUDE TARGET (|return_4h|)")
    print("=" * 110)
    print(f"{'Feature':<30} {'Train IC':>10} {'Test IC':>10} {'Ratio':>8} {'Status':>10} {'Full IC':>10} {'Verdict':>10}")
    print("-" * 110)

    mag_pass = []
    mag_candidates = ["oi_to_volume_zscore", "exchange_flow_8h", "fng_value",
                      "dvol_oi_interaction", "funding_squeeze"]

    for fname in mag_candidates:
        if fname not in features:
            print(f"{fname:<30} NOT COMPUTED")
            continue
        feat = features[fname]
        mask_all = feat.notna() & y_mag.notna()
        idx = np.arange(n)
        mask_train = mask_all & (idx < split)
        mask_test = mask_all & (idx >= split)

        if mask_train.sum() < 100 or mask_test.sum() < 50:
            continue

        ic_train, _ = spearmanr(feat[mask_train], y_mag[mask_train])
        ic_test, _ = spearmanr(feat[mask_test], y_mag[mask_test])
        ic_full, _ = spearmanr(feat[mask_all], y_mag[mask_all])

        ratio = ic_test / ic_train if abs(ic_train) > 1e-6 else 0

        if np.sign(ic_train) != np.sign(ic_test):
            status = "FLIP"
        elif abs(ic_test) < abs(ic_train) * 0.3:
            status = "COLLAPSE"
        elif abs(ic_test) >= abs(ic_train) * 0.7:
            status = "STABLE"
        else:
            status = "DECAY"

        if status in ("FLIP", "COLLAPSE"):
            verdict = "REJECT"
        elif abs(ic_test) >= 0.02:
            verdict = "PASS"
            mag_pass.append(fname)
        else:
            verdict = "WEAK"

        print(f"{fname:<30} {ic_train:>+10.4f} {ic_test:>+10.4f} {ratio:>8.2f} {status:>10} {ic_full:>+10.4f} {verdict:>10}")

    print()
    print("=" * 110)
    print(f"Direction PASS ({len(dir_pass)}): {dir_pass}")
    print(f"Direction REJECT ({len(dir_reject)}): {dir_reject}")
    print(f"Magnitude PASS ({len(mag_pass)}): {mag_pass}")
    print("=" * 110)


if __name__ == "__main__":
    main()
