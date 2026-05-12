"""
Python port of jdehorty's Machine Learning: Lorentzian Classification
(TradingView indicator), tuned to user's settings (LDC參數.docx, 2026-05-10):

    Source: close
    Neighbors Count: 8
    Max Bars Back: 2000
    Feature Count: 3   (RSI 14/1, WT 10/11, CCI 20/1)   ← user only uses 3
    Volatility filter: ON
    Regime filter:     ON, threshold -0.1
    ADX filter:        OFF
    EMA filter:        ON, period 200    ← non-default
    SMA filter:        OFF
    Kernel filter:     ON (rationalQuadratic h=8, r=8, x=25)
    Enhance Smoothing: OFF
    Use Dynamic Exits: ON  (we DON'T port exits — only entry signals)

Data alignment: user trades BINANCE:BTCUSDT.P which matches our local
binance_klines_1h.parquet (Binance USDT perp). No data-source mismatch.

Pinescript reference: https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/
The PineScript is in the repo at LDC原始碼.txt-equivalent (provided 2026-05-10).

Note on faithfulness:
    - n_rsi / n_wt / n_cci use jdehorty/MLExtensions's `normalize` helper,
      which is an EXPANDING-WINDOW min-max rescale.  Python equivalent
      tracked here as cumulative min/max.
    - ANN search reproduces the i%4 stride + chronological neighbor
      replacement logic exactly.
    - Filters reproduce the published PineScript bodies.
    - Some PineScript indicators use Pine's own ta.* implementations
      which can differ from pandas/TA-Lib by ±1 bar of warmup.  Per-bar
      values may differ by < 1% from what TradingView shows; cumulative
      signal sequence on a 5-month window is expected to be 95%+ identical.
"""
from __future__ import annotations
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

KLINES_PATH = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"
OUT_DIR = PROJECT_ROOT / "research" / "results"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─── Indicators (Python equivalents of pinescript ta.*) ─────────────────────

def rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI (matches pinescript ta.rsi)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period, min_periods=period).mean()


def cci(close: pd.Series, period: int) -> pd.Series:
    """Commodity Channel Index — pinescript ta.cci(src, n) uses sma(src,n) and mean abs deviation."""
    tp = close  # pinescript ta.cci with single source uses src directly, NOT (h+l+c)/3
    ma = sma(tp, period)
    md = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Pinescript ta.atr — RMA of true range."""
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    # ta.atr uses RMA = ewm(alpha=1/period)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


# ─── Normalization helpers (jdehorty/MLExtensions) ──────────────────────────

def rescale(s: pd.Series, old_min: float, old_max: float,
            new_min: float = 0, new_max: float = 1) -> pd.Series:
    return new_min + (new_max - new_min) * (s - old_min) / max(old_max - old_min, 1e-10)


def normalize_expanding(s: pd.Series, new_min: float = 0, new_max: float = 1) -> pd.Series:
    """Expanding-window min-max rescale (tracks historical min/max as we go)."""
    h_min = s.expanding().min()
    h_max = s.expanding().max()
    rng = (h_max - h_min).replace(0, 1e-10)
    return new_min + (new_max - new_min) * (s - h_min) / rng


# ─── jdehorty's normalized indicators ───────────────────────────────────────

def n_rsi(close: pd.Series, n1: int, n2: int) -> pd.Series:
    """ema(rsi(close, n1), n2) rescaled 0..100 → 0..1."""
    return rescale(ema(rsi(close, n1), n2), 0, 100, 0, 1)


def n_wt(hlc3: pd.Series, n1: int, n2: int) -> pd.Series:
    """Wave Trend, expanding-normalized."""
    e1 = ema(hlc3, n1)
    e2 = ema((hlc3 - e1).abs(), n1)
    ci = (hlc3 - e1) / (0.015 * e2.replace(0, np.nan))
    wt1 = ema(ci, n2)
    wt2 = sma(wt1, 4)
    return normalize_expanding(wt1 - wt2)


def n_cci(close: pd.Series, n1: int, n2: int) -> pd.Series:
    return normalize_expanding(ema(cci(close, n1), n2))


# ─── Filters ────────────────────────────────────────────────────────────────

def filter_volatility(high: pd.Series, low: pd.Series, close: pd.Series,
                      min_len: int = 1, max_len: int = 10) -> pd.Series:
    """ATR(min_len) > ATR(max_len)."""
    return atr(high, low, close, min_len) > atr(high, low, close, max_len)


def regime_filter(open_: pd.Series, high: pd.Series, low: pd.Series,
                  close: pd.Series, threshold: float = -0.1) -> pd.Series:
    """jdehorty's regime filter — KLMF-based slope-decline measure ≥ threshold."""
    src = (open_ + high + low + close) / 4

    n = len(src)
    val1 = np.zeros(n)
    val2 = np.zeros(n)
    klmf = np.zeros(n)
    s_arr = src.values
    h_arr = high.values
    l_arr = low.values

    for i in range(1, n):
        val1[i] = 0.2 * (s_arr[i] - s_arr[i - 1]) + 0.8 * val1[i - 1]
        val2[i] = 0.1 * (h_arr[i] - l_arr[i]) + 0.8 * val2[i - 1]
        if val2[i] != 0:
            omega = abs(val1[i] / val2[i])
        else:
            omega = 0
        alpha = (-omega ** 2 + np.sqrt(omega ** 4 + 16 * omega ** 2)) / 8 if omega > 0 else 0
        klmf[i] = alpha * s_arr[i] + (1 - alpha) * klmf[i - 1]

    klmf_s = pd.Series(klmf, index=src.index)
    abs_slope = klmf_s.diff().abs()
    exp_avg_abs_slope = ema(abs_slope, 200)  # 1.0 *
    norm_slope_decline = (abs_slope - exp_avg_abs_slope) / exp_avg_abs_slope.replace(0, np.nan)
    return norm_slope_decline >= threshold


# ─── Kernel: rationalQuadratic ──────────────────────────────────────────────

def kernel_gaussian(close: pd.Series, lookback: int, start_at_bar: int) -> pd.Series:
    """
    Nadaraya-Watson Gaussian kernel regression — exact port of
    jdehorty/KernelFunctions/2 :: gaussian.

    PineScript:
        for i = 0 to _size + startAtBar
            w = math.exp(-(i**2) / (2 * lookback**2))
            sumWY += src[i] * w; sumW += w
        yhat = sumWY / sumW

    Used as yhat2 in jdehorty's dynamic-exit logic, with lookback = h - lag
    (lag default = 2; user's setting h=8, so yhat2 lookback = 6).
    """
    n_weights = start_at_bar + 2
    h = lookback
    weights = np.array([
        np.exp(-(i ** 2) / (2 * h * h))
        for i in range(n_weights)
    ])
    weights = weights / weights.sum()
    arr = close.values
    out = np.full(len(arr), np.nan)
    for t in range(n_weights - 1, len(arr)):
        window = arr[t - n_weights + 1: t + 1][::-1]
        out[t] = np.dot(window, weights)
    return pd.Series(out, index=close.index)


def kernel_rational_quadratic(close: pd.Series, lookback: int, relative_weight: float,
                              start_at_bar: int) -> pd.Series:
    """
    Nadaraya-Watson Rational Quadratic kernel regression — exact port of
    jdehorty/KernelFunctions/2 :: rationalQuadratic.

    PineScript:
        _size = array.size(array.from(_src))    // always 1 (single-element array)
        for i = 0 to _size + startAtBar         // i = 0 to 26 (when startAtBar=25)
            w = (1 + i² / (lookback² × 2 × relativeWeight))^(-relativeWeight)
            sumWY += src[i] × w
            sumW  += w
        yhat = sumWY / sumW

    Note: PineScript denominator is `lookback² × 2 × relativeWeight`, NOT
    `relativeWeight² × 2 × lookback`.  These are numerically identical only
    when relativeWeight == lookback (user's case: both = 8).
    """
    n_weights = start_at_bar + 2  # i = 0..startAtBar+1 inclusive
    r = relative_weight
    h = lookback
    weights = np.array([
        (1 + (i ** 2) / (h * h * 2 * r)) ** (-r)
        for i in range(n_weights)
    ])
    weights = weights / weights.sum()  # normalize once

    arr = close.values
    out = np.full(len(arr), np.nan)
    for t in range(n_weights - 1, len(arr)):
        # window[0] = close[t] (current bar, weight idx 0)
        # window[1] = close[t-1] (1 bar ago)
        # window[n_weights-1] = close[t - n_weights + 1] (oldest)
        window = arr[t - n_weights + 1: t + 1][::-1]
        out[t] = np.dot(window, weights)
    return pd.Series(out, index=close.index)


# ─── ANN Lorentzian KNN (faithful port of jdehorty's loop) ──────────────────

def ann_lorentzian_signals(features: np.ndarray, labels: np.ndarray,
                            neighbors_count: int = 8,
                            max_bars_back: int = 2000) -> np.ndarray:
    """
    For each bar t, compute prediction = sum of k=8 ANN labels.

    Reproduces:
        for i = 0 to sizeLoop:
            d = lorentzian_dist(current, history[i])
            if d >= lastDistance and i % 4:   # i not divisible by 4
                lastDistance = d
                push d to distances; push label[i] to predictions
                if size > k:
                    lastDistance = distances[round(k*3/4)]  # 6th element when k=8
                    shift first
        prediction = sum(predictions)

    features: (n, n_features), chronologically ordered (oldest at 0)
    labels:   (n,) — y_train_series at each bar
    """
    n, n_feat = features.shape
    out = np.zeros(n)

    for t in range(n):
        # PineScript at bar T (after array.push of current features):
        #   array.size = T+1, size = min(maxBarsBack-1, T)
        #   sizeLoop = size = min(maxBarsBack-1, T)
        #   for i = 0 to sizeLoop  (INCLUSIVE)
        #
        # When T < maxBarsBack:   loop covers i = 0..T (includes current bar T)
        # When T >= maxBarsBack:  loop covers i = 0..maxBarsBack-1 (excludes T)
        #
        # featureSeries = features[t] (current);
        # featureArrays.fX[i] = features[i] (historical).
        if t == 0:
            continue
        if t < max_bars_back:
            i_lo, i_hi = 0, t  # inclusive: bars 0..t
        else:
            i_lo, i_hi = t - max_bars_back, t - max_bars_back + (max_bars_back - 1)  # 0..maxBarsBack-1 in array space → bars (t-maxBarsBack)..(t-1)
        # Note: in PineScript array space, i is the index into featureArrays,
        # which corresponds to the bar index (since arrays are push-only and
        # never trimmed when iterating).  So i % 4 in PineScript == bar_index % 4.

        last_distance = -1.0
        distances = []
        predictions = []

        for i in range(i_lo, i_hi + 1):
            # PineScript: `if d >= lastDistance and i%4`
            # i here is array index = bar index in our setup.
            if (i % 4) == 0:
                continue

            # Lorentzian distance between current bar features (t) and historical (i)
            diff = features[t] - features[i]
            d = float(np.log1p(np.abs(diff)).sum())

            if d >= last_distance:
                last_distance = d
                distances.append(d)
                predictions.append(labels[i])
                if len(predictions) > neighbors_count:
                    # shift oldest out
                    distances.pop(0)
                    predictions.pop(0)
                    # PineScript: lastDistance := distances[round(k*3/4)]
                    # for k=8: round(6) = 6 → 7th element (0-indexed 6)
                    idx_75 = round(neighbors_count * 3 / 4)
                    last_distance = distances[idx_75]

        out[t] = float(np.sum(predictions))

    return out


# ─── Main pipeline ──────────────────────────────────────────────────────────

def compute_ldc_signals(
    klines: pd.DataFrame,
    *,
    neighbors_count: int = 8,
    max_bars_back: int = 2000,
    feature_spec: tuple = (("rsi", 14, 1), ("wt", 10, 11), ("cci", 20, 1)),
    use_volatility_filter: bool = True,
    use_regime_filter: bool = True,
    regime_threshold: float = -0.1,
    use_ema_filter: bool = True,
    ema_period: int = 200,
    kernel_h: int = 8,
    kernel_r: float = 8.0,
    kernel_x: int = 25,
    kernel_lag: int = 2,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute LDC signals with configurable parameters.

    feature_spec : tuple of (kind, n1, n2) tuples.  kind in {"rsi","wt","cci"}.
        n1/n2 are the indicator's two main periods (see n_rsi/n_wt/n_cci).
        1-5 features supported.  Default = jdehorty's 3 features.

    Set use_*_filter=False to drop that filter (more entries).
    Set ema_period=None or use_ema_filter=False to drop EMA filter.

    Returns DataFrame with columns:
        prediction, signal, is_bullish_kernel, is_bearish_kernel,
        is_bullish_cross, is_bearish_cross, yhat1, yhat2,
        ema_uptrend, ema_downtrend, vol_filter_pass, regime_filter_pass,
        start_long_trade, start_short_trade
    """
    log = logger.info if verbose else logger.debug
    close = klines["close"]
    high = klines["high"]
    low = klines["low"]
    open_ = klines["open"]
    hlc3 = (high + low + close) / 3

    feat_parts = []
    feat_names = []
    for i, (kind, n1, n2) in enumerate(feature_spec, start=1):
        if kind == "rsi":
            f = n_rsi(close, n1, n2)
            feat_names.append(f"f{i}_rsi{n1}_{n2}")
        elif kind == "wt":
            f = n_wt(hlc3, n1, n2)
            feat_names.append(f"f{i}_wt{n1}_{n2}")
        elif kind == "cci":
            f = n_cci(close, n1, n2)
            feat_names.append(f"f{i}_cci{n1}_{n2}")
        else:
            raise ValueError(f"unknown feature kind {kind}")
        feat_parts.append(f)
    log("Computing %d normalized features: %s", len(feat_parts), feat_names)
    feature_df = pd.concat(feat_parts, axis=1)
    feature_df.columns = feat_names

    # Drop bars where any feature is NaN (warmup)
    valid = feature_df.dropna().index
    feature_df = feature_df.loc[valid]
    klines_v = klines.loc[valid]
    close_v = klines_v["close"]
    high_v = klines_v["high"]
    low_v = klines_v["low"]
    open_v = klines_v["open"]
    log("After feature warmup drop: %d / %d rows", len(feature_df), len(klines))

    # Training labels — exact port of PineScript:
    #   `src[4] < src[0] ? direction.short : src[4] > src[0] ? direction.long : neutral`
    # In PineScript src[N] = N bars ago, so:
    #   close[t-4] < close[t]  (past 4 bars went UP)   → label = -1 (SHORT)
    #   close[t-4] > close[t]  (past 4 bars went DOWN) → label = +1 (LONG)
    # This is REVERSAL-style labelling: "current bar similar to past bars whose
    # past-4 went up → predict short (bet on mean reversion down)".  The
    # contrarian label sign is jdehorty's design intent — DO NOT flip back to
    # the intuitive trend-following form.  Verified by re-reading library
    # source 2026-05-10.
    y_train = pd.Series(0, index=close_v.index, dtype=int)
    y_train[close_v.shift(4) < close_v] = -1  # past 4 went up → SHORT label
    y_train[close_v.shift(4) > close_v] = 1   # past 4 went down → LONG label

    log("Running ANN Lorentzian KNN (k=%d, max_bars_back=%d)...", neighbors_count, max_bars_back)
    pred = ann_lorentzian_signals(
        feature_df.values, y_train.values,
        neighbors_count=neighbors_count, max_bars_back=max_bars_back,
    )
    pred_s = pd.Series(pred, index=feature_df.index, name="prediction")

    log("Computing filters...")
    vol_pass = (filter_volatility(high_v, low_v, close_v, 1, 10).fillna(False)
                if use_volatility_filter else pd.Series(True, index=close_v.index))
    reg_pass = (regime_filter(open_v, high_v, low_v, close_v, regime_threshold).fillna(False)
                if use_regime_filter else pd.Series(True, index=close_v.index))
    filter_all = vol_pass & reg_pass

    if use_ema_filter:
        ema_val = ema(close_v, ema_period)
        ema_uptrend = close_v > ema_val
        ema_downtrend = close_v < ema_val
    else:
        ema_uptrend = pd.Series(True, index=close_v.index)
        ema_downtrend = pd.Series(True, index=close_v.index)

    log("Computing kernel filter (rationalQuadratic h=%d, r=%.1f, x=%d)...",
        kernel_h, kernel_r, kernel_x)
    yhat1 = kernel_rational_quadratic(close_v, kernel_h, kernel_r, kernel_x)
    is_bullish_rate = yhat1 > yhat1.shift(1)
    is_bearish_rate = yhat1 < yhat1.shift(1)

    # Secondary kernel for dynamic exits (gaussian, lookback = h - lag)
    yhat2 = kernel_gaussian(close_v, max(2, kernel_h - kernel_lag), kernel_x)
    # Cross signals (yhat1 = rationalQuadratic, yhat2 = gaussian)
    is_bullish_cross = (yhat1 > yhat2) & (yhat1.shift(1) <= yhat2.shift(1))
    is_bearish_cross = (yhat1 < yhat2) & (yhat1.shift(1) >= yhat2.shift(1))

    # Signal carry-over (pinescript: signal := pred>0 and filter_all ? long : pred<0 and filter_all ? short : nz(signal[1]))
    raw_dir = np.where(filter_all & (pred_s > 0), 1,
                        np.where(filter_all & (pred_s < 0), -1, 0))
    signal = np.zeros(len(raw_dir), dtype=int)
    last = 0
    for i, d in enumerate(raw_dir):
        if d != 0:
            last = d
        signal[i] = last
    signal_s = pd.Series(signal, index=feature_df.index, name="signal")

    is_different = signal_s.diff().fillna(0) != 0

    is_buy = (signal_s == 1) & ema_uptrend
    is_sell = (signal_s == -1) & ema_downtrend
    is_new_buy = is_buy & is_different
    is_new_sell = is_sell & is_different

    start_long = is_new_buy & is_bullish_rate & ema_uptrend
    start_short = is_new_sell & is_bearish_rate & ema_downtrend

    out = pd.DataFrame({
        "prediction": pred_s,
        "signal": signal_s,
        "vol_filter_pass": vol_pass,
        "regime_filter_pass": reg_pass,
        "ema_uptrend": ema_uptrend,
        "ema_downtrend": ema_downtrend,
        "is_bullish_kernel": is_bullish_rate,
        "is_bearish_kernel": is_bearish_rate,
        "is_bullish_cross": is_bullish_cross,
        "is_bearish_cross": is_bearish_cross,
        "yhat1": yhat1,
        "yhat2": yhat2,
        "start_long_trade": start_long,
        "start_short_trade": start_short,
    })
    return out


def main():
    logger.info("Loading klines from %s", KLINES_PATH)
    klines = pd.read_parquet(KLINES_PATH)
    klines = klines[["open", "high", "low", "close"]].copy()
    klines = klines[~klines.index.duplicated(keep="last")].sort_index()
    klines = klines.dropna()
    if klines.index.tz is not None:
        klines.index = klines.index.tz_convert("UTC").tz_localize(None)
    logger.info("Klines: n=%d, range %s ~ %s",
                len(klines), klines.index.min(), klines.index.max())

    signals = compute_ldc_signals(klines)

    # Sanity stats
    n = len(signals)
    n_long = signals["start_long_trade"].sum()
    n_short = signals["start_short_trade"].sum()
    avg_per_day = (n_long + n_short) / (n / 24)
    print(f"\n=== LDC entry signals ===")
    print(f"  n bars: {n}")
    print(f"  long entries:  {n_long}")
    print(f"  short entries: {n_short}")
    print(f"  total entries: {n_long + n_short}")
    print(f"  avg per day:   {avg_per_day:.2f}")

    # Filter waterfall — see which condition drops the most signals
    print(f"\n=== Entry filter waterfall ===")
    sig_change = signals["signal"].diff().fillna(0) != 0
    is_buy_sig = (signals["signal"] == 1)
    is_sell_sig = (signals["signal"] == -1)
    n_changes = sig_change.sum()
    print(f"  signal flips (any direction):  {n_changes}")
    print(f"  flips → long (signal=+1):      {(is_buy_sig & sig_change).sum()}")
    print(f"  flips → long + EMA up:         {(is_buy_sig & sig_change & signals['ema_uptrend']).sum()}")
    print(f"  flips → long + EMA up + kernel up:  {(is_buy_sig & sig_change & signals['ema_uptrend'] & signals['is_bullish_kernel']).sum()}")
    print(f"  flips → short (signal=-1):     {(is_sell_sig & sig_change).sum()}")
    print(f"  flips → short + EMA down:      {(is_sell_sig & sig_change & signals['ema_downtrend']).sum()}")
    print(f"  flips → short + EMA dn + kernel dn: {(is_sell_sig & sig_change & signals['ema_downtrend'] & signals['is_bearish_kernel']).sum()}")

    # Filter pass-through stats
    print(f"\n=== Filter activation rates ===")
    print(f"  volatility filter pass: {signals['vol_filter_pass'].mean()*100:.1f}%")
    print(f"  regime filter pass:     {signals['regime_filter_pass'].mean()*100:.1f}%")
    print(f"  EMA uptrend:            {signals['ema_uptrend'].mean()*100:.1f}%")
    print(f"  EMA downtrend:          {signals['ema_downtrend'].mean()*100:.1f}%")
    print(f"  Bullish kernel:         {signals['is_bullish_kernel'].mean()*100:.1f}%")

    # Per-month signal distribution
    print(f"\n=== Per-month entry distribution ===")
    sig_only = signals[signals["start_long_trade"] | signals["start_short_trade"]].copy()
    sig_only["dir"] = np.where(sig_only["start_long_trade"], "long", "short")
    by_month = sig_only.groupby([sig_only.index.to_period("M"), "dir"]).size().unstack(fill_value=0)
    print(by_month.to_string())

    # Save full signals + entry-only summary
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full_path = OUT_DIR / "ldc_signals_full.parquet"
    signals.to_parquet(full_path)
    logger.info("Saved full signals → %s", full_path)

    entries = signals[signals["start_long_trade"] | signals["start_short_trade"]].copy()
    entries["direction"] = np.where(entries["start_long_trade"], "long", "short")
    entries = entries[["prediction", "direction"]]
    entries_path = OUT_DIR / "ldc_entries.parquet"
    entries.to_parquet(entries_path)
    logger.info("Saved entry-only → %s (%d rows)", entries_path, len(entries))


if __name__ == "__main__":
    main()
