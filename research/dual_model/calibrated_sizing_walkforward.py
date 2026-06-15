"""Walk-forward calibrated-edge sizing vs tier — full robustness gauntlet (2026-06-06).

Follow-up to calibration_for_sizing.py, which on ONE 60/40 split showed
calibrated-edge sizing (size ∝ 2*p_cal-1) beating tier (Strong1.0/Mod0.5).
Single-split wins are how you get fooled (this session's recurring lesson), so:

 1. WALK-FORWARD calibration: for each fold k, fit isotonic(conf->win) on signals
    from folds < k only, apply to fold k. Calibration is itself out-of-sample.
 2. Per-signal leverage cap (winsorize) — the prior run's calibrated-Kelly hit
    ~10x on one signal; cap stops single-bet concentration.
 3. GAUNTLET for "calibrated-edge > tier" (mistake.md 2026-06-02 4 conditions):
    aggregate Sharpe/term, per-fold mean-PnL diff, frac folds positive,
    bootstrap CI on Sharpe diff (must not span 0).

All sizing normalized to avg leverage 1.0x (fair: allocation shape only).
Caveat: fixed-4h-hold proxy; relative ranking is the signal.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

STRONG_FRAC, MOD_FRAC, WARMUP, WINDOW = 0.05, 0.15, 100, 500
ABS_FLOOR_STRONG, ABS_FLOOR_MOD = 0.0008, 0.0005
FB_SU, FB_SD = 0.0021502008312381804, -0.0015994133136700839
FB_MU, FB_MD = 0.0011227245995542035, -0.0010095790785271674
COST = 0.0008 * 2
MIN_CAL_TRAIN = 80     # min prior signals before a fold is evaluable
LEV_CAP_PCTILE = 95    # winsorize per-signal weight at this percentile
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


def norm(w):
    w = np.clip(np.asarray(w, float), 0, None)
    m = w.mean()
    return w * (1.0 / m) if m > 0 else w


def sim(L, r, span_days):
    step = np.clip(1 + L * r, 1e-9, None)
    eq = np.cumprod(step)
    pnl = L * r
    sd = pnl.std(ddof=1)
    sh = (pnl.mean() / sd * np.sqrt(len(r) / (span_days / 365))) if sd > 0 else np.nan
    peak = np.maximum.accumulate(eq)
    return dict(term=float(eq[-1]), sharpe=sh,
                mdd=float(((eq - peak) / peak).min() * 100), maxlev=float(L.max()))


def main():
    oos = pd.read_parquet(OOS_PATH)
    kl = pd.read_parquet(KLINES_PATH)
    close = kl["close"].astype(float)
    oos["regime"] = assign_regime(close).reindex(oos.index).values
    oos = oos[oos["regime"].notna()].copy().sort_index()
    d, t, c = decode(oos["pred_ret"].values)
    oos["dir"], oos["tier"], oos["conf"] = d, t, c
    sig = oos[oos["dir"] != "NEUTRAL"].copy()
    dsign = np.where(sig["dir"].values == "UP", 1.0, -1.0)
    sig["win"] = (dsign == np.sign(sig["y_path_ret_4h"].values)).astype(int)
    sig["net_r"] = dsign * sig["y_path_ret_4h"].values - COST

    # ── walk-forward isotonic calibration by fold ──
    folds = sorted(sig["fold"].unique())
    p_cal = pd.Series(np.nan, index=sig.index)
    for k in folds:
        prior = sig[sig["fold"] < k]
        cur = sig[sig["fold"] == k]
        if len(prior) < MIN_CAL_TRAIN or len(cur) == 0:
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(prior["conf"].values, prior["win"].values)
        p_cal.loc[cur.index] = iso.predict(cur["conf"].values)
    sig["p_cal"] = p_cal
    ev = sig[sig["p_cal"].notna()].copy()   # evaluable (post calibration burn-in)
    span = (ev.index[-1] - ev.index[0]).days or 1
    print(f"signals={len(sig)}  evaluable(after WF-cal burn-in)={len(ev)}  "
          f"folds={ev['fold'].nunique()}  span={span}d")
    print(f"WF-calibrated p: mean {ev['p_cal'].mean():.2f} "
          f"(raw conf/100 mean {ev['conf'].mean()/100:.2f}) "
          f"range [{ev['p_cal'].min():.2f},{ev['p_cal'].max():.2f}]")

    r = ev["net_r"].values
    is_strong = (ev["tier"].values == "Strong").astype(float)
    # calibrated-edge weight, winsorized to stop single-bet concentration
    w_cal = np.clip(2 * ev["p_cal"].values - 1, 0, None)
    cap = np.percentile(w_cal[w_cal > 0], LEV_CAP_PCTILE) if (w_cal > 0).any() else 1.0
    w_cal = np.minimum(w_cal, cap)

    rules = {
        "flat": np.ones(len(ev)),
        "tier (S1.0/M0.5)": np.where(is_strong > 0, 1.0, 0.5),
        "raw-confidence": ev["conf"].values / 100,
        "calibrated-edge (cap)": w_cal,
    }
    L = {name: norm(w) for name, w in rules.items()}
    print(f"\n  {'rule':24s} {'term':>6s} {'Sharpe':>7s} {'MDD%':>7s} {'maxLev':>7s}")
    for name in rules:
        m = sim(L[name], r, span)
        print(f"  {name:24s} {m['term']:6.2f} {m['sharpe']:7.2f} {m['mdd']:7.1f} {m['maxlev']:7.2f}")

    # ── GAUNTLET: calibrated-edge vs tier ──
    pnl_cal = L["calibrated-edge (cap)"] * r
    pnl_tier = L["tier (S1.0/M0.5)"] * r
    diff = pnl_cal - pnl_tier
    ev2 = ev.copy(); ev2["diff"] = diff
    fold_diff = ev2.groupby("fold")["diff"].mean()
    fold_diff = fold_diff[ev2.groupby("fold").size() >= 3]
    # bootstrap on Sharpe diff
    rng = np.random.default_rng(42)
    sh_d = []
    for _ in range(3000):
        idx = rng.integers(0, len(r), len(r))
        a, b = pnl_cal[idx], pnl_tier[idx]
        sa = a.mean() / a.std(ddof=1) if a.std(ddof=1) > 0 else 0
        sb = b.mean() / b.std(ddof=1) if b.std(ddof=1) > 0 else 0
        sh_d.append(sa - sb)
    lo, hi = np.percentile(sh_d, [2.5, 97.5])

    print("\n=== GAUNTLET: calibrated-edge vs tier (4 conditions) ===")
    agg = (sim(L["calibrated-edge (cap)"], r, span)["sharpe"]
           - sim(L["tier (S1.0/M0.5)"], r, span)["sharpe"])
    print(f"  1. aggregate Sharpe lift:        {agg:+.2f}  "
          f"({'PASS' if agg > 0 else 'FAIL'})")
    print(f"  2. per-fold mean-PnL diff:       {fold_diff.mean()*1e4:+.2f}bps  "
          f"({'PASS' if fold_diff.mean() > 0 else 'FAIL'})")
    print(f"  3. frac folds positive:          {(fold_diff>0).mean()*100:.0f}% "
          f"of {len(fold_diff)}  ({'PASS' if (fold_diff>0).mean() > 0.55 else 'FAIL'})")
    print(f"  4. bootstrap Sharpe-diff 95% CI: [{lo:+.3f},{hi:+.3f}]  "
          f"({'PASS (excl 0)' if lo > 0 else 'FAIL (spans 0)'})")
    n_pass = sum([agg > 0, fold_diff.mean() > 0,
                  (fold_diff > 0).mean() > 0.55, lo > 0])
    print(f"\n  VERDICT: {n_pass}/4 conditions -> "
          f"{'calibrated-edge beats tier (robust)' if n_pass == 4 else 'NOT robustly better than tier — keep tier'}")


if __name__ == "__main__":
    main()
