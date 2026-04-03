"""
Direction label construction for BTC direction prediction research.

Provides three labeling strategies:
  1. raw_sign    — sign of future return (baseline, noisy)
  2. deadzone    — drop samples where |return| < threshold
  3. triple_barrier — first-hit of upper/lower barrier

All methods strictly avoid look-ahead bias: labels are computed from
future price data that would not be available at prediction time,
but the *features* used alongside these labels must be trailing-only.

Usage:
    from research.direction_labels import build_direction_labels

    labels = build_direction_labels(
        df, method="triple_barrier", horizon_bars=4,
        vol_col="realized_vol_20b", k=0.5,
    )
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Raw sign label ───────────────────────────────────────────────────────

def label_raw_sign(
    close: np.ndarray,
    horizon: int,
    neutral_value: float = np.nan,
) -> np.ndarray:
    """
    Label = sign of close[t+horizon] / close[t] - 1.

    Parameters
    ----------
    close : 1-D array of close prices
    horizon : bars to look forward
    neutral_value : value for exactly-zero returns (rare)

    Returns
    -------
    labels : 1-D array, 1=UP, 0=DOWN, neutral_value for zero/NaN
    """
    n = len(close)
    labels = np.full(n, np.nan)
    for i in range(n - horizon):
        ret = close[i + horizon] / close[i] - 1
        if np.isnan(ret):
            continue
        if ret > 0:
            labels[i] = 1.0
        elif ret < 0:
            labels[i] = 0.0
        else:
            labels[i] = neutral_value
    return labels


# ── Deadzone label ───────────────────────────────────────────────────────

def label_deadzone(
    close: np.ndarray,
    horizon: int,
    threshold: float | np.ndarray,
    neutral_value: float = np.nan,
) -> np.ndarray:
    """
    Label = sign of future return, but NaN if |return| < threshold.

    Parameters
    ----------
    close : 1-D array of close prices
    horizon : bars to look forward
    threshold : scalar or per-bar array (e.g. k * realized_vol)
                If scalar, applied uniformly.
                If array, must have same length as close.
    neutral_value : label for |return| < threshold (default NaN = drop)

    Returns
    -------
    labels : 1-D array, 1=UP, 0=DOWN, neutral_value for deadzone/NaN
    """
    n = len(close)
    labels = np.full(n, np.nan)
    thr = np.broadcast_to(threshold, n) if np.isscalar(threshold) else threshold

    for i in range(n - horizon):
        ret = close[i + horizon] / close[i] - 1
        if np.isnan(ret) or np.isnan(thr[i]):
            continue
        if abs(ret) < thr[i]:
            labels[i] = neutral_value
        elif ret > 0:
            labels[i] = 1.0
        else:
            labels[i] = 0.0
    return labels


def compute_vol_threshold(
    vol: np.ndarray,
    k: float = 0.5,
    min_thr: float = 0.001,
) -> np.ndarray:
    """
    Volatility-adaptive threshold: thr_t = k * vol_t, floored at min_thr.

    Parameters
    ----------
    vol : per-bar realized volatility (e.g. realized_vol_20b)
    k : multiplier (0.3~1.0 typical; 0.5 = half a vol move)
    min_thr : absolute floor to avoid zero-threshold in low-vol

    Returns
    -------
    threshold : per-bar array
    """
    thr = k * np.abs(vol)
    thr = np.where(np.isnan(thr), min_thr, thr)
    return np.maximum(thr, min_thr)


# ── Triple-barrier label ─────────────────────────────────────────────────

def label_triple_barrier(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    horizon: int,
    upper_k: float = 1.0,
    lower_k: float = 1.0,
    vol: np.ndarray | None = None,
    fixed_upper: float | None = None,
    fixed_lower: float | None = None,
    neutral_value: float = np.nan,
) -> np.ndarray:
    """
    Triple-barrier labeling using intra-bar high/low for first-hit detection.

    For each bar t:
      upper_barrier = close[t] * (1 + upper_thr)
      lower_barrier = close[t] * (1 - lower_thr)
    Scan bars t+1 .. t+horizon:
      - If high[j] >= upper_barrier FIRST → label = 1 (UP)
      - If low[j] <= lower_barrier FIRST → label = 0 (DOWN)
      - If both hit same bar → use close direction in that bar
      - If neither hit within horizon → neutral_value

    Parameters
    ----------
    high, low, close : 1-D arrays (same length)
    horizon : max bars to scan forward
    upper_k, lower_k : multiplier for vol-based barriers
        barrier = k * vol[t]; ignored if fixed_upper/lower given
    vol : per-bar volatility array (required if fixed not given)
    fixed_upper, fixed_lower : fixed absolute return thresholds
        (e.g. 0.01 = 1%). Takes precedence over vol-based.
    neutral_value : label when no barrier hit

    Returns
    -------
    labels : 1-D array, 1=UP, 0=DOWN, neutral_value for no-hit
    """
    n = len(close)
    labels = np.full(n, np.nan)

    for i in range(n - 1):
        c = close[i]
        if np.isnan(c) or c <= 0:
            continue

        # Compute barriers
        if fixed_upper is not None:
            u_thr = fixed_upper
        elif vol is not None and not np.isnan(vol[i]):
            u_thr = upper_k * vol[i]
        else:
            continue

        if fixed_lower is not None:
            l_thr = fixed_lower
        elif vol is not None and not np.isnan(vol[i]):
            l_thr = lower_k * vol[i]
        else:
            continue

        upper_barrier = c * (1 + u_thr)
        lower_barrier = c * (1 - l_thr)

        max_j = min(i + 1 + horizon, n)
        hit = False
        for j in range(i + 1, max_j):
            hit_upper = high[j] >= upper_barrier
            hit_lower = low[j] <= lower_barrier

            if hit_upper and hit_lower:
                # Both barriers hit in same bar — use close direction
                if close[j] >= c:
                    labels[i] = 1.0
                else:
                    labels[i] = 0.0
                hit = True
                break
            elif hit_upper:
                labels[i] = 1.0
                hit = True
                break
            elif hit_lower:
                labels[i] = 0.0
                hit = True
                break

        if not hit:
            labels[i] = neutral_value

    return labels


# ── Unified interface ────────────────────────────────────────────────────

def build_direction_labels(
    df: pd.DataFrame,
    method: str = "deadzone",
    horizon_bars: int = 4,
    vol_col: str = "realized_vol_20b",
    k: float = 0.5,
    fixed_threshold: float | None = None,
    neutral_value: float = np.nan,
) -> pd.Series:
    """
    Unified direction label builder.

    Parameters
    ----------
    df : DataFrame with at least 'close' column.
         For triple_barrier, also needs 'high' and 'low'.
         For vol-adjusted methods, needs vol_col.
    method : "raw_sign" | "deadzone" | "triple_barrier"
    horizon_bars : number of bars to look forward
    vol_col : column name for realized volatility
    k : volatility multiplier for threshold / barrier width
    fixed_threshold : if set, use fixed threshold instead of vol-adjusted
    neutral_value : label for neutral / no-hit samples

    Returns
    -------
    pd.Series with labels (1=UP, 0=DOWN, neutral_value=neutral)
    """
    close = df["close"].values.astype(float)

    if method == "raw_sign":
        labels = label_raw_sign(close, horizon_bars, neutral_value)

    elif method == "deadzone":
        if fixed_threshold is not None:
            thr = fixed_threshold
        elif vol_col in df.columns:
            vol = df[vol_col].values.astype(float)
            thr = compute_vol_threshold(vol, k=k)
        else:
            raise ValueError(f"Need vol_col='{vol_col}' or fixed_threshold for deadzone")
        labels = label_deadzone(close, horizon_bars, thr, neutral_value)

    elif method == "triple_barrier":
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)

        if fixed_threshold is not None:
            labels = label_triple_barrier(
                high, low, close, horizon_bars,
                fixed_upper=fixed_threshold, fixed_lower=fixed_threshold,
                neutral_value=neutral_value,
            )
        elif vol_col in df.columns:
            vol = df[vol_col].values.astype(float)
            labels = label_triple_barrier(
                high, low, close, horizon_bars,
                upper_k=k, lower_k=k, vol=vol,
                neutral_value=neutral_value,
            )
        else:
            raise ValueError(f"Need vol_col='{vol_col}' or fixed_threshold for triple_barrier")
    else:
        raise ValueError(f"Unknown method: {method}. Use raw_sign/deadzone/triple_barrier")

    return pd.Series(labels, index=df.index, name=f"dir_{method}_{horizon_bars}b")
