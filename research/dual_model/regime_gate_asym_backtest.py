"""Backtest: does regime-gating UP (BULL_CONTRA_PENALTY) or an asymmetric
UP/DOWN floor fix the "too many losing LONG signals in TRENDING_BEAR" problem?

Context (2026-06-05): live UP signals ran 24% win over the last bear leg
(DOWN ran 89%). The production decode already has a wired-but-disabled knob
`BULL_CONTRA_PENALTY` (inference.py:90, =1.0) that widens the UP threshold in
TRENDING_BEAR (and the DOWN threshold in TRENDING_BULL). This script sweeps it
on historical walk-forward OOS predictions and compares against an always-on
asymmetric UP floor — WITHOUT touching live.

Decode logic is copied verbatim from indicator/inference.py:320-405 so the
backtest matches production. Metrics follow mistake.md / feedback_retrain
discipline: tier-filtered sign-acc + net bps, broken down by regime AND
direction, with per-fold consistency and bootstrap CI. Crucially we check the
gate does NOT hurt CHOPPY/BULL (no overfitting to the current bear).

Caveat: this evaluates a fixed-4h-hold proxy (sign(pred) vs y_path_ret_4h),
not the live signal-exit + 3xATR trailing. Fine for comparing *which signals
to fire*; absolute bps are a proxy, the RELATIVE variant ranking is the point.
"""
from __future__ import annotations

import sys
from collections import deque

import numpy as np
import pandas as pd

# ── Production constants (indicator/inference.py + direction_reg_config.json) ──
STRONG_FRAC = 0.05
MOD_FRAC = 0.15
WARMUP_BARS = 100
PCTILE_WINDOW = 500
ABS_FLOOR_STRONG = 0.0008
ABS_FLOOR_MODERATE = 0.0005
FALLBACK_STRONG_UP = 0.0021502008312381804
FALLBACK_STRONG_DN = -0.0015994133136700839
FALLBACK_MOD_UP = 0.0011227245995542035
FALLBACK_MOD_DN = -0.0010095790785271674

ROUND_TRIP_COST = 0.0008 * 2  # okx/config.py taker_cost 0.0008 round-trip = 16 bps

OOS_PATH = "research/results/dual_model/direction_reg_oos_mse.parquet"
KLINES_PATH = "market_data/raw_data/binance_klines_1h.parquet"


def assign_regime(close: pd.Series) -> pd.Series:
    """Verbatim from inference.py:471-487 (trailing-only)."""
    WB = 72
    log_ret = np.log(close / close.shift(1))
    ret_24h = close.pct_change(24)
    vol_24h = log_ret.rolling(24).std()
    vol_pct = vol_24h.expanding(min_periods=WB).rank(pct=True)
    regime = np.full(len(close), "CHOPPY", dtype=object)
    regime[(vol_pct > 0.6).values & (ret_24h > 0.005).values] = "TRENDING_BULL"
    regime[(vol_pct > 0.6).values & (ret_24h < -0.005).values] = "TRENDING_BEAR"
    regime[:WB] = "WARMUP"
    return pd.Series(regime, index=close.index)


def decode(pred: np.ndarray, regime: np.ndarray, *, penalty: float = 1.0,
           up_floor_mult: float = 1.0, dn_floor_mult: float = 1.0):
    """Replicates inference.py:333-391 decode.

    penalty       : BULL_CONTRA_PENALTY — widens UP cutoff in BEAR, DOWN in BULL.
    up/dn_floor_mult : regime-INDEPENDENT asymmetric floor multiplier (tests
                       "UP is just structurally weak, tighten it everywhere").
    """
    n = len(pred)
    direction = np.full(n, "NEUTRAL", dtype=object)
    tier = np.full(n, "Weak", dtype=object)
    buf: deque = deque(maxlen=PCTILE_WINDOW)
    for i in range(n):
        p = float(pred[i])
        buf.append(p)
        if len(buf) < WARMUP_BARS:
            up_s, dn_s = FALLBACK_STRONG_UP, FALLBACK_STRONG_DN
            up_m, dn_m = FALLBACK_MOD_UP, FALLBACK_MOD_DN
        else:
            a = np.fromiter(buf, dtype=float)
            up_s = float(np.quantile(a, 1.0 - STRONG_FRAC / 2.0))
            dn_s = float(np.quantile(a, STRONG_FRAC / 2.0))
            up_m = float(np.quantile(a, 1.0 - MOD_FRAC / 2.0))
            dn_m = float(np.quantile(a, MOD_FRAC / 2.0))
        # absolute floor (+ optional asymmetric multiplier on the floor)
        up_s_f = max(up_s, ABS_FLOOR_STRONG * up_floor_mult)
        dn_s_f = min(dn_s, -ABS_FLOOR_STRONG * dn_floor_mult)
        up_m_f = max(up_m, ABS_FLOOR_MODERATE * up_floor_mult)
        dn_m_f = min(dn_m, -ABS_FLOOR_MODERATE * dn_floor_mult)
        # regime contra-trend penalty
        reg = regime[i]
        up_mul = penalty if reg == "TRENDING_BEAR" else 1.0
        dn_mul = penalty if reg == "TRENDING_BULL" else 1.0
        up_s_e, up_m_e = up_s_f * up_mul, up_m_f * up_mul
        dn_s_e, dn_m_e = dn_s_f * dn_mul, dn_m_f * dn_mul
        if p >= up_s_e:
            direction[i], tier[i] = "UP", "Strong"
        elif p <= dn_s_e:
            direction[i], tier[i] = "DOWN", "Strong"
        elif p >= up_m_e:
            direction[i], tier[i] = "UP", "Moderate"
        elif p <= dn_m_e:
            direction[i], tier[i] = "DOWN", "Moderate"
    return direction, tier


def per_signal_net_bps(direction: np.ndarray, y: np.ndarray) -> np.ndarray:
    """+y for UP (long), -y for DOWN (short), minus round-trip cost. In bps."""
    sgn = np.where(direction == "UP", 1.0, np.where(direction == "DOWN", -1.0, np.nan))
    pnl = sgn * y - ROUND_TRIP_COST
    return pnl * 1e4  # bps


def boot_ci(x: np.ndarray, n=2000, seed=42):
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def evaluate(df: pd.DataFrame, direction: np.ndarray, label: str) -> dict:
    d = df.copy()
    d["direction"] = direction
    d = d[(d["regime"] != "WARMUP") & (d["direction"] != "NEUTRAL")].copy()
    if d.empty:
        print(f"\n### {label}: NO SIGNALS")
        return {"label": label, "n": 0}
    d["correct"] = (np.sign(np.where(d["direction"] == "UP", 1, -1))
                    == np.sign(d["y_path_ret_4h"])).astype(int)
    d["net_bps"] = per_signal_net_bps(d["direction"].values,
                                      d["y_path_ret_4h"].values)
    n = len(d)
    acc = d["correct"].mean() * 100
    net = d["net_bps"].mean()
    lo, hi = boot_ci(d["net_bps"].values)
    # per-fold net bps consistency
    fold_net = d.groupby("fold")["net_bps"].mean()
    frac_pos = (fold_net > 0).mean() * 100
    print(f"\n### {label}")
    print(f"  signals={n}  sign_acc={acc:.1f}%  net={net:+.1f}bps "
          f"CI[{lo:+.1f},{hi:+.1f}]  total={d['net_bps'].sum()/1e2:+.1f}%  "
          f"per-fold net>0: {frac_pos:.0f}%")
    # by regime x direction
    print("  by regime × dir (n | sign_acc | net_bps):")
    for reg in ["TRENDING_BEAR", "TRENDING_BULL", "CHOPPY"]:
        for dr in ["UP", "DOWN"]:
            s = d[(d["regime"] == reg) & (d["direction"] == dr)]
            if len(s):
                print(f"    {reg:14s} {dr:4s}: {len(s):4d} | "
                      f"{s['correct'].mean()*100:5.1f}% | "
                      f"{s['net_bps'].mean():+7.1f}")
    return {"label": label, "n": n, "sign_acc": acc, "net_bps": net,
            "net_ci": (lo, hi), "total_pct": d["net_bps"].sum() / 1e2,
            "frac_pos_folds": frac_pos}


def main():
    oos = pd.read_parquet(OOS_PATH)
    kl = pd.read_parquet(KLINES_PATH)
    close_full = kl["close"].astype(float)
    regime_full = assign_regime(close_full)
    oos["regime"] = regime_full.reindex(oos.index).values
    before = len(oos)
    oos = oos[oos["regime"].notna()].copy()
    print(f"OOS rows: {before} -> {len(oos)} after regime join "
          f"(dropped {before - len(oos)} with no close)")
    rdist = oos["regime"].value_counts().to_dict()
    print("regime dist:", rdist)
    print(f"round-trip cost: {ROUND_TRIP_COST*1e4:.0f} bps")

    pred = oos["pred_ret"].values
    reg = oos["regime"].values

    variants = [
        ("V0 baseline (penalty=1.0, production now)", dict(penalty=1.0)),
        ("V1 penalty=1.5", dict(penalty=1.5)),
        ("V2 penalty=2.0", dict(penalty=2.0)),
        ("V3 penalty=2.5 (CLAUDE.md original)", dict(penalty=2.5)),
        ("V4 penalty=3.0", dict(penalty=3.0)),
        ("V5 full gate (penalty=1e9, no contra-trend at all)", dict(penalty=1e9)),
        ("V6 asym floor UP×2.0 (regime-independent)", dict(up_floor_mult=2.0)),
        ("V7 asym floor UP×3.0", dict(up_floor_mult=3.0)),
        ("V8 penalty=2.5 + asym floor UP×1.5", dict(penalty=2.5, up_floor_mult=1.5)),
    ]
    rows = []
    for label, kw in variants:
        direction, _ = decode(pred, reg, **kw)
        rows.append(evaluate(oos, direction, label))

    print("\n" + "=" * 78)
    print("SUMMARY (sorted by net_bps)")
    print("=" * 78)
    rows = [r for r in rows if r.get("n", 0) > 0]
    for r in sorted(rows, key=lambda x: -x["net_bps"]):
        print(f"  {r['label']:48s} n={r['n']:4d} "
              f"acc={r['sign_acc']:5.1f}% net={r['net_bps']:+6.1f}bps "
              f"CI[{r['net_ci'][0]:+.1f},{r['net_ci'][1]:+.1f}] "
              f"folds>0={r['frac_pos_folds']:.0f}%")


if __name__ == "__main__":
    main()
