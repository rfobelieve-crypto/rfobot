"""
Initiation label builder for Long/Short initiation model.

Label semantics (strict, trailing-only where relevant):

    long_init(t)  = 1  iff
        close[t+4] / close[t] - 1  >=  +k          (forward outcome, 4h)
        AND close[t] > max(close[t-lookback : t])  (trailing breakout confirm)

    short_init(t) = 1  iff
        close[t+4] / close[t] - 1  <=  -k
        AND close[t] < min(close[t-lookback : t])

The forward-return uses t..t+4 only (no look-ahead past t+4) (no look-ahead past t+8).
The breakout-confirm uses strictly t-lookback..t-1 (no look-ahead).

k defaults to 1.0% per user spec. k-sweep helper exposes {0.8, 1.0, 1.2, 1.5}.

Output columns:
    ret_4h          - forward cumulative return (4h)
    ret_vol_adj_4h  - ret_4h / realized_vol_20b (reference only)
    breakout_up     - 1/0 trailing-only breakout above rolling high
    breakout_dn     - 1/0 trailing-only breakout below rolling low
    y_long_init     - final binary label (with breakout gate if enabled)
    y_short_init    - final binary label (with breakout gate if enabled)
    y_long_raw      - same but without breakout gate (for ablation)
    y_short_raw     - same but without breakout gate (for ablation)
    y_any_init      - y_long_init OR y_short_init

NaN in the last HORIZON_BARS rows (no forward return available).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HORIZON_BARS = 4            # 4h forward horizon on 1h bars
DEFAULT_K_PCT = 0.008       # +/- 0.8%
DEFAULT_BREAKOUT_LOOKBACK = 20
K_SWEEP_VALUES = (0.008, 0.010, 0.012, 0.015)


@dataclass
class InitiationLabelConfig:
    k_pct: float = DEFAULT_K_PCT
    breakout_lookback: int = DEFAULT_BREAKOUT_LOOKBACK
    use_breakout_confirm: bool = True
    horizon_bars: int = HORIZON_BARS


def _trailing_breakout(close: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (breakout_up, breakout_dn) as float arrays, trailing-only.
    breakout_up[t] = 1 iff close[t] > max(close[t-lookback:t])  (excludes t)
    breakout_dn[t] = 1 iff close[t] < min(close[t-lookback:t])  (excludes t)
    First `lookback` bars are NaN.
    """
    s = pd.Series(close)
    prior_high = s.shift(1).rolling(lookback, min_periods=lookback).max()
    prior_low = s.shift(1).rolling(lookback, min_periods=lookback).min()
    bo_up = (s > prior_high).astype(float)
    bo_dn = (s < prior_low).astype(float)
    bo_up[prior_high.isna()] = np.nan
    bo_dn[prior_low.isna()] = np.nan
    return bo_up.values, bo_dn.values


def build_initiation_labels(
    df: pd.DataFrame,
    config: InitiationLabelConfig | None = None,
) -> pd.DataFrame:
    """
    Build initiation labels on the input dataframe.

    Requires columns:
        close               (required)
        realized_vol_20b    (optional; only for ret_vol_adj_4h diagnostic)
    """
    if config is None:
        config = InitiationLabelConfig()
    if "close" not in df.columns:
        raise ValueError("build_initiation_labels: df must have 'close' column")

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float) if "high" in df.columns else close
    low = df["low"].values.astype(float) if "low" in df.columns else close
    n = len(close)
    H = config.horizon_bars
    k = config.k_pct

    ret_fwd = np.full(n, np.nan)
    # First-touch forward path stats: does high/low over t+1..t+H exceed +/-k?
    max_up_fwd = np.full(n, np.nan)
    max_dn_fwd = np.full(n, np.nan)
    for i in range(n - H):
        ret_fwd[i] = close[i + H] / close[i] - 1
        window_high = high[i + 1 : i + 1 + H].max()
        window_low = low[i + 1 : i + 1 + H].min()
        max_up_fwd[i] = window_high / close[i] - 1
        max_dn_fwd[i] = window_low / close[i] - 1

    vol = (df["realized_vol_20b"].values.astype(float)
           if "realized_vol_20b" in df.columns else np.full(n, np.nan))
    vol_safe = np.where(vol > 0, vol, np.nan)
    ret_vol_adj = ret_fwd / vol_safe

    bo_up, bo_dn = _trailing_breakout(close, config.breakout_lookback)

    # --- Endpoint label (legacy): close[t+H]/close[t] crosses +/-k ---
    long_raw = (ret_fwd >= +k).astype(float)
    short_raw = (ret_fwd <= -k).astype(float)
    long_raw[np.isnan(ret_fwd)] = np.nan
    short_raw[np.isnan(ret_fwd)] = np.nan

    # --- First-touch label (new): forward path high/low exceeds +/-k ---
    long_touch_raw = (max_up_fwd >= +k).astype(float)
    short_touch_raw = (max_dn_fwd <= -k).astype(float)
    long_touch_raw[np.isnan(max_up_fwd)] = np.nan
    short_touch_raw[np.isnan(max_dn_fwd)] = np.nan

    if config.use_breakout_confirm:
        y_long = np.where((long_raw == 1.0) & (bo_up == 1.0), 1.0, 0.0)
        y_short = np.where((short_raw == 1.0) & (bo_dn == 1.0), 1.0, 0.0)
        y_long = np.where(np.isnan(long_raw) | np.isnan(bo_up), np.nan, y_long)
        y_short = np.where(np.isnan(short_raw) | np.isnan(bo_dn), np.nan, y_short)

        y_long_touch = np.where((long_touch_raw == 1.0) & (bo_up == 1.0), 1.0, 0.0)
        y_short_touch = np.where((short_touch_raw == 1.0) & (bo_dn == 1.0), 1.0, 0.0)
        y_long_touch = np.where(np.isnan(long_touch_raw) | np.isnan(bo_up),
                                 np.nan, y_long_touch)
        y_short_touch = np.where(np.isnan(short_touch_raw) | np.isnan(bo_dn),
                                  np.nan, y_short_touch)
    else:
        y_long = long_raw
        y_short = short_raw
        y_long_touch = long_touch_raw
        y_short_touch = short_touch_raw

    y_any = np.where(np.isnan(y_long) | np.isnan(y_short), np.nan,
                     np.maximum(y_long, y_short))

    out = pd.DataFrame({
        "ret_4h": ret_fwd,
        "ret_vol_adj_4h": ret_vol_adj,
        "path_max_up_4h": max_up_fwd,
        "path_max_dn_4h": max_dn_fwd,
        "breakout_up": bo_up,
        "breakout_dn": bo_dn,
        "y_long_raw": long_raw,
        "y_short_raw": short_raw,
        "y_long_init": y_long,
        "y_short_init": y_short,
        "y_any_init": y_any,
        "y_long_touch": y_long_touch,
        "y_short_touch": y_short_touch,
    }, index=df.index)

    valid = ~np.isnan(y_long) & ~np.isnan(y_short)
    if valid.sum() > 0:
        logger.info(
            "initiation labels: k=%.2f%%, lookback=%d, confirm=%s | "
            "valid=%d long=%.2f%% short=%.2f%% any=%.2f%% (raw: long=%.2f%% short=%.2f%%)",
            k * 100, config.breakout_lookback, config.use_breakout_confirm,
            int(valid.sum()),
            100 * out.loc[valid, "y_long_init"].mean(),
            100 * out.loc[valid, "y_short_init"].mean(),
            100 * out.loc[valid, "y_any_init"].mean(),
            100 * out.loc[valid, "y_long_raw"].mean(),
            100 * out.loc[valid, "y_short_raw"].mean(),
        )
    return out


def sweep_k_distribution(
    df: pd.DataFrame,
    k_values: tuple[float, ...] = K_SWEEP_VALUES,
    breakout_lookback: int = DEFAULT_BREAKOUT_LOOKBACK,
) -> pd.DataFrame:
    """
    Report positive-class rates at multiple k thresholds, with and without
    breakout confirmation, broken down by calendar month.

    Returns a tidy DataFrame with columns:
        k, month, n_valid, long_rate, short_rate, any_rate,
        long_rate_raw, short_rate_raw
    """
    rows = []
    months = df.index.strftime("%Y-%m")
    month_keys = sorted(set(months))
    for k in k_values:
        cfg = InitiationLabelConfig(k_pct=k, breakout_lookback=breakout_lookback,
                                     use_breakout_confirm=True)
        lbl = build_initiation_labels(df, cfg)
        for m in month_keys:
            mask = (months == m) & lbl["y_long_init"].notna().values
            if mask.sum() < 30:
                continue
            sub = lbl.loc[mask]
            rows.append({
                "k": k,
                "month": m,
                "n_valid": int(mask.sum()),
                "long_rate": float(sub["y_long_init"].mean()),
                "short_rate": float(sub["y_short_init"].mean()),
                "any_rate": float(sub["y_any_init"].mean()),
                "long_rate_raw": float(sub["y_long_raw"].mean()),
                "short_rate_raw": float(sub["y_short_raw"].mean()),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from research.dual_model.shared_data import load_and_cache_data

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_and_cache_data()
    print(f"Loaded {len(df)} bars")

    sweep = sweep_k_distribution(df)
    print("\n=== k-sweep monthly distribution (with breakout confirm) ===")
    pivot = sweep.pivot_table(index="month", columns="k",
                               values="any_rate", aggfunc="first")
    print((pivot * 100).round(2).to_string())

    print("\n=== k=1.0% baseline (monthly long/short split) ===")
    base = sweep[sweep["k"] == DEFAULT_K_PCT].copy()
    base["long_pct"] = (base["long_rate"] * 100).round(2)
    base["short_pct"] = (base["short_rate"] * 100).round(2)
    base["any_pct"] = (base["any_rate"] * 100).round(2)
    print(base[["month", "n_valid", "long_pct", "short_pct", "any_pct"]].to_string(index=False))
