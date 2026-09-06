# -*- coding: utf-8 -*-
"""Stage 1 — volume-profile engine.  Independent of everything else.

Pure computation over an array of bars.  It knows nothing about sweeps, levels
or labels; Stage 4 is the only caller that joins it to events.  Tested on
synthetic data first (tests/test_profile.py), then measured against real data.

Frozen definitions
------------------
bins touched by one bar
    range(floor(low / bin_size), ceil(high / bin_size))      -- HALF-OPEN
    so low=100, high=110, bin=1 touches 10 bins (100..109), each getting
    volume/10.  A bar with low == high (or a range narrower than one bin)
    touches exactly the single bin floor(low / bin_size).
method
    'uniform'  the volume is split equally across the touched bins
    'close'    the whole volume goes to floor(close / bin_size)
POC
    argmax of the histogram.  **Ties: the median price of the tied bins**
    (pre-registered; without it the answer depends on dict ordering).
HVN / LVN
    bins whose volume is >= the 80th / <= the 20th percentile of the
    non-empty bin volumes.
VWAP
    sum(v * (h+l+c)/3) / sum(v) over the same bars -- no binning, so it
    carries no discretisation error.  That asymmetry against POC is real and
    must be stated wherever the two are compared.
lookback
    ('time', hours)      bars with t_ref - hours*3600e3 <= open_ms < t_ref
    ('volume', frac)     walk back from the newest bar until the cumulative
                         volume reaches frac * avg_daily_volume_30d

Look-ahead
    every bar entering a profile satisfies open_ms + 60_000 <= t_ref, i.e. it
    is CLOSED at t_ref.  Asserted on every call, not checked by eye.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_MS = 60_000


@dataclass
class Profile:
    bin_size: float
    bin_ids: np.ndarray          # int64, ascending, non-empty bins only
    volumes: np.ndarray          # float64, aligned with bin_ids
    total_volume: float
    poc: float                   # price
    vwap: float                  # price
    hvn: np.ndarray              # prices, ascending
    lvn: np.ndarray              # prices, ascending
    n_bars: int
    first_ms: int
    last_ms: int

    def bin_price(self, bin_id):
        return (np.asarray(bin_id) + 0.5) * self.bin_size

    def next_hvn(self, side_sign):
        """HVN nearest to the POC on the far side of the sweep.

        side_sign +1 (sellside) looks BELOW the POC, -1 (buyside) ABOVE.
        Returns None when the profile has no HVN on that side.
        """
        if side_sign > 0:
            c = self.hvn[self.hvn < self.poc]
            return float(c[-1]) if len(c) else None
        c = self.hvn[self.hvn > self.poc]
        return float(c[0]) if len(c) else None


def _touched(low, high, bin_size):
    """Half-open bin range per bar; guarantees at least one bin."""
    b0 = np.floor(low / bin_size).astype(np.int64)
    b1 = np.ceil(high / bin_size).astype(np.int64)
    n = np.maximum(b1 - b0, 1)
    return b0, n


def _histogram(low, high, close, vol, bin_size, method):
    if method == "close":
        ids = np.floor(close / bin_size).astype(np.int64)
        w = vol
    elif method == "uniform":
        b0, n = _touched(low, high, bin_size)
        share = vol / n
        total = int(n.sum())
        # expand each bar into its n contiguous bins, vectorised
        ends = np.cumsum(n)
        starts = ends - n
        pos = np.arange(total, dtype=np.int64)
        offs = pos - np.repeat(starts, n)
        ids = np.repeat(b0, n) + offs
        w = np.repeat(share, n)
    else:
        raise ValueError("method must be 'uniform' or 'close'")
    uniq, inv = np.unique(ids, return_inverse=True)
    return uniq, np.bincount(inv, weights=w, minlength=len(uniq))


def build_profile(bars, t_ref, lookback, method, bin_size,
                  avg_daily_volume=None, hvn_q=0.80, lvn_q=0.20):
    """bars: object with numpy arrays ts (open ms), high, low, close, volume.

    Returns a Profile, or None when the window holds no traded volume.
    """
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")
    ts = bars.ts
    # every bar must be CLOSED strictly before t_ref
    hi = int(np.searchsorted(ts, t_ref - MIN_MS, side="right"))
    if hi <= 0:
        return None

    kind, arg = lookback
    if kind == "time":
        lo = int(np.searchsorted(ts, t_ref - int(arg * 3600_000), side="left"))
    elif kind == "volume":
        if not avg_daily_volume or avg_daily_volume <= 0:
            return None
        target = arg * avg_daily_volume
        v = np.nan_to_num(bars.volume[:hi])
        # cumulative from the newest bar backwards
        cum = np.cumsum(v[::-1])
        k = int(np.searchsorted(cum, target, side="left"))
        if k >= len(cum):
            return None                      # not enough history to reach it
        lo = hi - (k + 1)
    else:
        raise ValueError("lookback must be ('time', hours) or ('volume', frac)")
    if lo >= hi:
        return None

    sl = slice(lo, hi)
    low, high, close = bars.low[sl], bars.high[sl], bars.close[sl]
    vol = bars.volume[sl]
    good = np.isfinite(low) & np.isfinite(high) & np.isfinite(close) \
        & np.isfinite(vol) & (vol > 0)
    if not good.any():
        return None
    low, high, close, vol = low[good], high[good], close[good], vol[good]
    used_ts = ts[sl][good]

    # --- look-ahead guard: not a comment, an assertion -------------------
    assert used_ts.max() + MIN_MS <= t_ref, (
        "look-ahead: a bar closing at/after t_ref entered the profile")

    ids, w = _histogram(low, high, close, vol, bin_size, method)
    total = float(w.sum())
    if total <= 0:
        return None

    top = w.max()
    tied = ids[w >= top - 1e-12]
    poc = float(np.median((tied + 0.5) * bin_size))      # frozen tie rule
    vwap = float((vol * (high + low + close) / 3.0).sum() / vol.sum())
    hv = float(np.quantile(w, hvn_q))
    lv = float(np.quantile(w, lvn_q))
    return Profile(
        bin_size=float(bin_size),
        bin_ids=ids,
        volumes=w,
        total_volume=total,
        poc=poc,
        vwap=vwap,
        hvn=np.sort((ids[w >= hv] + 0.5) * bin_size),
        lvn=np.sort((ids[w <= lv] + 0.5) * bin_size),
        n_bars=int(good.sum()),
        first_ms=int(used_ts.min()),
        last_ms=int(used_ts.max()),
    )


def depth_between(prof, price_a, price_b):
    """Share of the profile's volume strictly between two prices."""
    lo, hi = (price_a, price_b) if price_a <= price_b else (price_b, price_a)
    px = (prof.bin_ids + 0.5) * prof.bin_size
    m = (px > lo) & (px < hi)
    return float(prof.volumes[m].sum() / prof.total_volume)


class Bars:
    """Minimal column container so Stage 1 has no pandas dependency."""

    __slots__ = ("ts", "open", "high", "low", "close", "volume")

    def __init__(self, ts, open_, high, low, close, volume):
        self.ts = np.asarray(ts, dtype=np.int64)
        self.open = np.asarray(open_, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.low = np.asarray(low, dtype=np.float64)
        self.close = np.asarray(close, dtype=np.float64)
        self.volume = np.asarray(volume, dtype=np.float64)

    def __len__(self):
        return len(self.ts)

    @classmethod
    def from_parquet(cls, path, cols=("ts", "open", "high", "low", "close", "volume")):
        import pandas as pd
        d = pd.read_parquet(path, columns=list(cols))
        return cls(d["ts"].to_numpy(), d["open"].to_numpy(), d["high"].to_numpy(),
                   d["low"].to_numpy(), d["close"].to_numpy(), d["volume"].to_numpy())
