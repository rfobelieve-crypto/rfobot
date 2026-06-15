"""
Breakout Gate Variant Sweep

Gate isolation probe showed that breakout alone contributes ~39pp to Strong
WR (46->50% WR from model-only 11%). Improving the breakout definition is
the highest-leverage lever remaining.

This script tests 12+ breakout variants against FIXED OOS predictions from
initiation_replay_oos.parquet. The Init model and its predictions stay
constant — only the gate mask changes. This isolates "how much of the 39pp
gate lift can be extended by a better breakout definition?"

Variants tested:
    V1. Lookback sweep: 10 / 20 (current) / 40 / 80 / 120
    V2. Donchian high/low (high > max(prior_high), low < min(prior_low))
    V3. ATR-normalized magnitude filter: require (close-prior_high)/ATR>=k
    V4. Volume-confirmed breakout: breakout AND vol > 1.5*vol_ma20
    V5. Consolidated: close>prior_high AND open<prior_high (closes above after opening inside)
    V6. Combinations: lookback 40 + ATR>=0.25, lookback 40 + volume surge

For each variant, apply the Strong gate:
    p_long >= top-5% AND mag_pct >= 0.80 AND breakout_up_variant
Measure WR vs baseline (current V1 lookback=20, no magnitude threshold).

Run:  python -m research.breakout_gate_sweep
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import load_and_cache_data
from indicator.initiation_features import add_initiation_features

RESULTS_DIR = PROJECT_ROOT / "research" / "results"
OOS_PATH = RESULTS_DIR / "initiation_replay_oos.parquet"

STRONG_FRAC = 0.05
STRONG_MAG_PCT = 0.80


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _trailing_max(s: pd.Series, w: int) -> pd.Series:
    return s.shift(1).rolling(w, min_periods=max(2, w // 2)).max()


def _trailing_min(s: pd.Series, w: int) -> pd.Series:
    return s.shift(1).rolling(w, min_periods=max(2, w // 2)).min()


def _atr(high, low, close, w=14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev).abs(),
        (low - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.shift(1).rolling(w, min_periods=max(2, w // 2)).mean()


def build_variants(df: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Return dict of variant_name -> (bo_up, bo_dn) boolean Series."""
    close = df["close"].astype(float)
    high = df["high"].astype(float) if "high" in df.columns else close
    low = df["low"].astype(float) if "low" in df.columns else close
    vol = df["volume"].astype(float) if "volume" in df.columns else None

    atr = _atr(high, low, close, 14)
    vol_ma = (vol.shift(1).rolling(20, min_periods=5).mean()
              if vol is not None else None)

    variants: dict[str, tuple[pd.Series, pd.Series]] = {}

    # ── V1: Lookback sweep (close vs close-based prior high/low) ─────────
    for lb in [10, 20, 40, 80, 120]:
        ph = _trailing_max(close, lb)
        pl = _trailing_min(close, lb)
        bo_up = (close > ph).astype(float)
        bo_dn = (close < pl).astype(float)
        variants[f"V1_close_lb{lb}"] = (bo_up, bo_dn)

    # ── V2: Donchian (high/low based) ─────────────────────────────────────
    for lb in [20, 40]:
        ph = _trailing_max(high, lb)
        pl = _trailing_min(low, lb)
        bo_up = (high > ph).astype(float)
        bo_dn = (low < pl).astype(float)
        variants[f"V2_donch_lb{lb}"] = (bo_up, bo_dn)

    # ── V3: ATR-normalized magnitude filter (close-based lb=20) ───────────
    ph20 = _trailing_max(close, 20)
    pl20 = _trailing_min(close, 20)
    for k in [0.1, 0.25, 0.5, 1.0]:
        bo_up = ((close > ph20) & ((close - ph20) / atr >= k)).astype(float)
        bo_dn = ((close < pl20) & ((pl20 - close) / atr >= k)).astype(float)
        variants[f"V3_atr_k{k}"] = (bo_up, bo_dn)

    # ── V4: Volume-confirmed breakout ─────────────────────────────────────
    if vol_ma is not None:
        for k in [1.2, 1.5, 2.0]:
            vol_surge = (vol > vol_ma * k)
            bo_up = ((close > ph20) & vol_surge).astype(float)
            bo_dn = ((close < pl20) & vol_surge).astype(float)
            variants[f"V4_vol_k{k}"] = (bo_up, bo_dn)

    # ── V5: Close-from-inside (no gap opens above level) ──────────────────
    open_ = df["open"].astype(float) if "open" in df.columns else close
    bo_up_v5 = ((close > ph20) & (open_ <= ph20)).astype(float)
    bo_dn_v5 = ((close < pl20) & (open_ >= pl20)).astype(float)
    variants["V5_close_from_inside"] = (bo_up_v5, bo_dn_v5)

    # ── V6: Combos — strongest pairs ──────────────────────────────────────
    ph40 = _trailing_max(close, 40)
    pl40 = _trailing_min(close, 40)
    bo_up = ((close > ph40) & ((close - ph40) / atr >= 0.25)).astype(float)
    bo_dn = ((close < pl40) & ((pl40 - close) / atr >= 0.25)).astype(float)
    variants["V6_lb40_atr0.25"] = (bo_up, bo_dn)

    if vol_ma is not None:
        bo_up = ((close > ph40) & (vol > vol_ma * 1.5)).astype(float)
        bo_dn = ((close < pl40) & (vol > vol_ma * 1.5)).astype(float)
        variants["V6_lb40_vol1.5"] = (bo_up, bo_dn)

        # Triple filter: lb40 + ATR + vol
        bo_up = (
            (close > ph40)
            & ((close - ph40) / atr >= 0.25)
            & (vol > vol_ma * 1.3)
        ).astype(float)
        bo_dn = (
            (close < pl40)
            & ((pl40 - close) / atr >= 0.25)
            & (vol > vol_ma * 1.3)
        ).astype(float)
        variants["V6_triple_filter"] = (bo_up, bo_dn)

    return variants


def eval_variant(oos: pd.DataFrame, df: pd.DataFrame,
                 bo_up_v: pd.Series, bo_dn_v: pd.Series,
                 name: str) -> dict:
    idx = oos.index
    pl = oos["p_long"].values
    ps = oos["p_short"].values
    yl = oos["y_long"].values.astype(int)
    ys = oos["y_short"].values.astype(int)
    mp = oos["mag_pct"].fillna(0.5).values

    bu = bo_up_v.reindex(idx).fillna(0).values
    bd = bo_dn_v.reindex(idx).fillna(0).values

    # top-5% quantile thresholds from this OOS sample
    long_thr = float(pd.Series(pl).quantile(1 - STRONG_FRAC))
    short_thr = float(pd.Series(ps).quantile(1 - STRONG_FRAC))

    long_strong = (pl >= long_thr) & (mp >= STRONG_MAG_PCT) & (bu == 1.0)
    short_strong = (ps >= short_thr) & (mp >= STRONG_MAG_PCT) & (bd == 1.0)

    fires = wins = lw = lf = sw = sf = 0
    for i in range(len(idx)):
        ls, ss = long_strong[i], short_strong[i]
        if ls and ss:
            continue
        if ls:
            fires += 1; wins += int(yl[i])
            lf += 1; lw += int(yl[i])
        elif ss:
            fires += 1; wins += int(ys[i])
            sf += 1; sw += int(ys[i])

    if fires == 0:
        return dict(name=name, n=0, wr=0.0, ci_lo=0.0, ci_hi=0.0,
                    long_n=0, long_wr=0.0, short_n=0, short_wr=0.0,
                    fire_rate=0.0)

    wr = wins / fires
    lo, hi = wilson_ci(wins, fires)
    fire_rate_up = float(bo_up_v.reindex(idx).fillna(0).mean())
    fire_rate_dn = float(bo_dn_v.reindex(idx).fillna(0).mean())

    return dict(
        name=name, n=fires, wins=wins, wr=wr, ci_lo=lo, ci_hi=hi,
        long_n=lf, long_wins=lw,
        long_wr=lw / lf if lf else 0.0,
        short_n=sf, short_wins=sw,
        short_wr=sw / sf if sf else 0.0,
        fire_rate_bo_up=fire_rate_up,
        fire_rate_bo_dn=fire_rate_dn,
    )


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    # Load OOS predictions
    if not OOS_PATH.exists():
        print(f"ERROR: {OOS_PATH} not found. Run initiation_replay.py first.")
        return
    oos = pd.read_parquet(OOS_PATH)
    print(f"Loaded OOS: {len(oos)} bars  "
          f"(p_long/p_short/y_long/y_short/mag_pct)")

    df = load_and_cache_data()
    df = add_initiation_features(df)
    print(f"Loaded main df: {len(df)} bars x {len(df.columns)} cols")

    variants = build_variants(df)
    print(f"Built {len(variants)} breakout variants\n")

    # Evaluate each variant
    results = []
    for name, (bu, bd) in variants.items():
        res = eval_variant(oos, df, bu, bd, name)
        results.append(res)

    # Sort by Wilson CI lower bound (require n>=30)
    sorted_res = sorted(results, key=lambda r: -r["ci_lo"])

    print("=" * 105)
    print(f"  BREAKOUT GATE VARIANT RESULTS")
    print(f"  Baseline (current V1_close_lb20): see first row")
    print("=" * 105)
    print(f"{'variant':<25}{'n':>5}{'wins':>6}{'WR':>9}{'CI_lo':>8}{'CI_hi':>8}"
          f"{'long_n':>8}{'long_WR':>10}{'short_n':>9}{'short_WR':>11}"
          f"{'bo_up%':>9}")
    print("-" * 105)

    # Show baseline first, then sorted
    baseline = next(r for r in results if r["name"] == "V1_close_lb20")
    def row(r):
        return (f"{r['name']:<25}{r['n']:>5}{r['wins']:>6}{r['wr']*100:>8.1f}%"
                f"{r['ci_lo']*100:>7.1f}%{r['ci_hi']*100:>7.1f}%"
                f"{r['long_n']:>8}{r['long_wr']*100:>9.1f}%"
                f"{r['short_n']:>9}{r['short_wr']*100:>10.1f}%"
                f"{r['fire_rate_bo_up']*100:>8.1f}%")
    print(row(baseline))
    print("-" * 105)
    for r in sorted_res:
        if r["name"] == "V1_close_lb20":
            continue
        print(row(r))

    # Declare winner
    eligible = [r for r in results if r["n"] >= 30]
    if eligible:
        winner = max(eligible, key=lambda r: r["ci_lo"])
        base = baseline
        print("\n" + "=" * 75)
        print("  WINNER")
        print("=" * 75)
        print(f"  Baseline (V1_close_lb20): n={base['n']}  WR={base['wr']*100:.1f}%  "
              f"CI_lo={base['ci_lo']*100:.1f}%")
        print(f"  Winner   ({winner['name']}): n={winner['n']}  "
              f"WR={winner['wr']*100:.1f}%  CI_lo={winner['ci_lo']*100:.1f}%")
        d_wr = (winner["wr"] - base["wr"]) * 100
        d_ci = (winner["ci_lo"] - base["ci_lo"]) * 100
        print(f"  Δ WR: {d_wr:+.1f}pp   Δ CI_lo: {d_ci:+.1f}pp")
        if winner["name"] == "V1_close_lb20":
            print("\n  ==> Baseline still wins. Current breakout is optimal "
                  "among these variants.")
        elif d_ci > 2.0:
            print("\n  *** BREAKTHROUGH *** (>2pp CI_lo improvement)")
        elif d_ci > 0:
            print(f"\n  Marginal improvement. Consider if worth the change.")

    # Save
    with open(RESULTS_DIR / "breakout_gate_sweep.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {RESULTS_DIR/'breakout_gate_sweep.json'}")


if __name__ == "__main__":
    main()
