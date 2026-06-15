"""Calibration -> Sizing study (2026-06-06).

Thesis from the @RuujSs piece: confidence must be a CALIBRATED probability, not a
"feeling". And (ch.3) Kelly needs an accurate p. So: is our confidence_score
calibrated, and does sizing on a CALIBRATED edge beat the flat / tier sizing we
found earlier?

Pipeline:
 1. Production decode -> direction, tier, confidence per OOS bar.
 2. Fired signals: win = (dir matches realized sign), net_r = dir*y - cost.
 3. Chronological 60/40 split (calibrate on train, evaluate on TEST — avoids
    in-sample calibration optimism, per mistake.md discipline).
 4. RAW miscalibration on test: reliability table + Brier, overall + per-regime.
 5. Isotonic calibrate confidence->P(win) on train, apply to test. Brier before/after.
 6. SIZING on test (all rules normalized to avg leverage 1.0x — fair compare):
    flat / tier / raw-confidence / calibrated-edge(2p-1) / calibrated-Kelly.
    Metric: Sharpe, MDD, terminal. Does calibrated sizing win?

Caveat: fixed-4h-hold proxy (OOS target), not live trailing-stop. Relative
ranking is the signal.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

STRONG_FRAC, MOD_FRAC, WARMUP, WINDOW = 0.05, 0.15, 100, 500
ABS_FLOOR_STRONG, ABS_FLOOR_MOD = 0.0008, 0.0005
FB_SU, FB_SD = 0.0021502008312381804, -0.0015994133136700839
FB_MU, FB_MD = 0.0011227245995542035, -0.0010095790785271674
COST = 0.0008 * 2
VOL_WIN = 168
OOS_PATH = "research/results/dual_model/direction_reg_oos_mse.parquet"
KLINES_PATH = "market_data/raw_data/binance_klines_1h.parquet"


def assign_regime(close):
    WB = 72
    lr = np.log(close / close.shift(1))
    r24 = close.pct_change(24)
    vp = lr.rolling(24).std().expanding(min_periods=WB).rank(pct=True)
    reg = np.full(len(close), "CHOPPY", dtype=object)
    reg[(vp > 0.6).values & (r24 > 0.005).values] = "TRENDING_BULL"
    reg[(vp > 0.6).values & (r24 < -0.005).values] = "TRENDING_BEAR"
    reg[:WB] = "WARMUP"
    return pd.Series(reg, index=close.index)


def decode(pred):
    n = len(pred)
    direction = np.full(n, "NEUTRAL", dtype=object)
    tier = np.full(n, "Weak", dtype=object)
    conf = np.zeros(n)
    buf = deque(maxlen=WINDOW)
    for i in range(n):
        p = float(pred[i]); buf.append(p)
        if len(buf) < WARMUP:
            us, ds, um, dm = FB_SU, FB_SD, FB_MU, FB_MD
        else:
            a = np.fromiter(buf, dtype=float)
            us = float(np.quantile(a, 1 - STRONG_FRAC / 2)); ds = float(np.quantile(a, STRONG_FRAC / 2))
            um = float(np.quantile(a, 1 - MOD_FRAC / 2)); dm = float(np.quantile(a, MOD_FRAC / 2))
        us_f, ds_f = max(us, ABS_FLOOR_STRONG), min(ds, -ABS_FLOOR_STRONG)
        um_f, dm_f = max(um, ABS_FLOOR_MOD), min(dm, -ABS_FLOOR_MOD)
        if p >= us_f: direction[i], tier[i] = "UP", "Strong"
        elif p <= ds_f: direction[i], tier[i] = "DOWN", "Strong"
        elif p >= um_f: direction[i], tier[i] = "UP", "Moderate"
        elif p <= dm_f: direction[i], tier[i] = "DOWN", "Moderate"
        ref = max(abs(us), abs(ds), 1e-6)
        conf[i] = float(np.clip(min(abs(p) / ref, 1.0) ** 0.6 * 100, 0, 100))
    return direction, tier, conf


def reliability(conf, win, label):
    print(f"  {label}: conf-decile  pred%   actual%   n")
    df = pd.DataFrame({"c": conf, "w": win})
    df["bin"] = pd.qcut(df["c"], 5, duplicates="drop")
    for b, g in df.groupby("bin", observed=True):
        print(f"    {str(b):24s} {g['c'].mean():5.0f}   {g['w'].mean()*100:5.0f}    {len(g)}")
    print(f"    Brier(conf/100 as P): {brier_score_loss(win, np.clip(conf/100,0,1)):.4f}")


def sim(L, r, span_days):
    step = np.clip(1 + L * r, 1e-9, None)
    eq = np.cumprod(step)
    pnl = L * r
    sd = pnl.std(ddof=1)
    sh = (pnl.mean() / sd * np.sqrt(len(r) / (span_days / 365))) if sd > 0 else np.nan
    peak = np.maximum.accumulate(eq)
    mdd = float(((eq - peak) / peak).min() * 100)
    return dict(term=float(eq[-1]), sharpe=sh, mdd=mdd, maxlev=float(L.max()))


def norm(w):
    w = np.clip(np.asarray(w, float), 0, None)
    m = w.mean()
    return w * (1.0 / m) if m > 0 else w


def main():
    oos = pd.read_parquet(OOS_PATH)
    kl = pd.read_parquet(KLINES_PATH)
    close = kl["close"].astype(float)
    oos["regime"] = assign_regime(close).reindex(oos.index).values
    oos["rvol"] = close.pct_change().rolling(VOL_WIN).std().reindex(oos.index).values
    oos = oos[(oos["regime"].notna()) & (oos["rvol"] > 0)].copy().sort_index()
    d, t, c = decode(oos["pred_ret"].values)
    oos["dir"], oos["tier"], oos["conf"] = d, t, c
    sig = oos[oos["dir"] != "NEUTRAL"].copy()
    dsign = np.where(sig["dir"].values == "UP", 1.0, -1.0)
    sig["win"] = (dsign == np.sign(sig["y_path_ret_4h"].values)).astype(int)
    sig["net_r"] = dsign * sig["y_path_ret_4h"].values - COST

    # chronological 60/40
    cut = int(len(sig) * 0.6)
    tr, te = sig.iloc[:cut], sig.iloc[cut:]
    span = (te.index[-1] - te.index[0]).days or 1
    print(f"signals={len(sig)} | train={len(tr)} test={len(te)} (chrono split) "
          f"| test span={span}d")

    print("\n=== 1. RAW confidence calibration on TEST (is it a 'feeling'?) ===")
    reliability(te["conf"].values, te["win"].values, "ALL test")
    for rg in ["TRENDING_BEAR", "TRENDING_BULL", "CHOPPY"]:
        s = te[te["regime"] == rg]
        if len(s) > 20:
            reliability(s["conf"].values, s["win"].values, rg)

    print("\n=== 2. Isotonic calibrate confidence->P(win) (fit train, apply test) ===")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(tr["conf"].values, tr["win"].values)
    p_cal = iso.predict(te["conf"].values)
    b_raw = brier_score_loss(te["win"].values, np.clip(te["conf"].values / 100, 0, 1))
    b_cal = brier_score_loss(te["win"].values, p_cal)
    print(f"  Brier raw(conf/100)={b_raw:.4f}  ->  calibrated={b_cal:.4f}  "
          f"({'better' if b_cal < b_raw else 'worse'} by {abs(b_raw-b_cal):.4f})")
    print(f"  calibrated p range: [{p_cal.min():.2f}, {p_cal.max():.2f}] "
          f"mean {p_cal.mean():.2f}  (vs raw conf/100 mean {te['conf'].mean()/100:.2f})")

    # also calibrate confidence -> E[net_r] for a Kelly-style edge
    iso_r = IsotonicRegression(out_of_bounds="clip")
    iso_r.fit(tr["conf"].values, tr["net_r"].values)
    er_cal = iso_r.predict(te["conf"].values)

    print("\n=== 3. SIZING on TEST (avg lev=1.0x, fair) — does calibration help? ===")
    r = te["net_r"].values
    rv = te["rvol"].values
    is_strong = (te["tier"].values == "Strong").astype(float)
    rules = {
        "flat": np.ones(len(te)),
        "tier (S1.0/M0.5)": np.where(is_strong > 0, 1.0, 0.5),
        "raw-confidence": te["conf"].values / 100,
        "calibrated-edge (2p-1)": np.clip(2 * p_cal - 1, 0, None),
        "calibrated-Kelly (E[r]/rvol^2)": np.clip(er_cal, 0, None) / (rv ** 2),
    }
    print(f"  {'rule':32s} {'term':>6s} {'Sharpe':>7s} {'MDD%':>7s} {'maxLev':>7s}")
    res = {}
    for name, w in rules.items():
        m = sim(norm(w), r, span); res[name] = m
        print(f"  {name:32s} {m['term']:6.2f} {m['sharpe']:7.2f} {m['mdd']:7.1f} {m['maxlev']:7.2f}")

    best = max(res, key=lambda k: res[k]["sharpe"])
    print(f"\n  best Sharpe on test: '{best}' ({res[best]['sharpe']:.2f}) "
          f"vs flat ({res['flat']['sharpe']:.2f}) vs tier ({res['tier (S1.0/M0.5)']['sharpe']:.2f})")
    print("  NOTE: test n is small; treat as directional, confirm before any live use.")


if __name__ == "__main__":
    main()
