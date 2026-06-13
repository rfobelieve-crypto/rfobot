"""OOS position-SIZING rule comparison (Stage-4 ladder prep, 2026-06-05).

Holds the SIGNAL logic fixed (production decode on WF-OOS preds) and varies only
HOW MUCH to bet per signal. Answers the @RuujSs-article question for our system:
does edge/vol-aware sizing (Kelly / confidence / vol-target) allocate better than
flat sizing — on historical OOS that actually contains drawdowns?

METHOD (the important part): every rule is normalized to the SAME average
deployed leverage (avg L = 1.0x). Otherwise "more leverage -> more return until
ruin" wins trivially. Normalizing isolates ALLOCATION SHAPE — does putting more
size on high-edge / low-vol bars beat betting flat?

Per-signal base return r_i = dir_i * y_path_ret_4h - round_trip_cost (unlevered
4h hold). Equity compounds: E *= (1 + L_i * r_i).

Caveat: uses fixed-4h-hold proxy (OOS target), not the live trailing-stop exit.
Fine for RELATIVE sizing-rule comparison; absolute numbers are a proxy.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

STRONG_FRAC, MOD_FRAC, WARMUP, WINDOW = 0.05, 0.15, 100, 500
ABS_FLOOR_STRONG, ABS_FLOOR_MOD = 0.0008, 0.0005
FB_SU, FB_SD = 0.0021502008312381804, -0.0015994133136700839
FB_MU, FB_MD = 0.0011227245995542035, -0.0010095790785271674
COST = 0.0008 * 2
VOL_WIN = 168  # 7d hourly realized-vol window for vol-target / Kelly variance
TARGET_AVG_LEV = 1.0

OOS_PATH = "research/results/dual_model/direction_reg_oos_mse.parquet"
KLINES_PATH = "market_data/raw_data/binance_klines_1h.parquet"


def decode(pred):
    """Production decode -> (direction, tier, confidence). penalty=1.0 baseline."""
    n = len(pred)
    direction = np.full(n, "NEUTRAL", dtype=object)
    tier = np.full(n, "Weak", dtype=object)
    conf = np.zeros(n)
    buf = deque(maxlen=WINDOW)
    for i in range(n):
        p = float(pred[i])
        buf.append(p)
        if len(buf) < WARMUP:
            us, ds, um, dm = FB_SU, FB_SD, FB_MU, FB_MD
        else:
            a = np.fromiter(buf, dtype=float)
            us = float(np.quantile(a, 1 - STRONG_FRAC / 2))
            ds = float(np.quantile(a, STRONG_FRAC / 2))
            um = float(np.quantile(a, 1 - MOD_FRAC / 2))
            dm = float(np.quantile(a, MOD_FRAC / 2))
        us_f, ds_f = max(us, ABS_FLOOR_STRONG), min(ds, -ABS_FLOOR_STRONG)
        um_f, dm_f = max(um, ABS_FLOOR_MOD), min(dm, -ABS_FLOOR_MOD)
        if p >= us_f:
            direction[i], tier[i] = "UP", "Strong"
        elif p <= ds_f:
            direction[i], tier[i] = "DOWN", "Strong"
        elif p >= um_f:
            direction[i], tier[i] = "UP", "Moderate"
        elif p <= dm_f:
            direction[i], tier[i] = "DOWN", "Moderate"
        ref = max(abs(us), abs(ds), 1e-6)
        conf[i] = float(np.clip(min(abs(p) / ref, 1.0) ** 0.6 * 100, 0, 100))
    return direction, tier, conf


def metrics(L, r, span_days):
    """Equity sim + risk metrics for a normalized leverage vector."""
    step = 1 + L * r
    ruin = bool(np.any(step <= 0))
    step = np.clip(step, 1e-9, None)
    eq = np.cumprod(step)
    pnl = L * r
    mu, sd = pnl.mean(), pnl.std(ddof=1)
    n = len(r)
    sig_per_yr = n / (span_days / 365.0)
    sharpe = (mu / sd * np.sqrt(sig_per_yr)) if sd > 0 else float("nan")
    peak = np.maximum.accumulate(eq)
    mdd = float(((eq - peak) / peak).min() * 100)
    return dict(n=n, term=float(eq[-1]), sharpe=sharpe, mdd=mdd,
                avg_lev=float(L.mean()), max_lev=float(L.max()),
                ruin=ruin, worst=float(pnl.min() * 100))


def normalize(w):
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0, None)
    m = w.mean()
    return w * (TARGET_AVG_LEV / m) if m > 0 else w


def main():
    oos = pd.read_parquet(OOS_PATH)
    kl = pd.read_parquet(KLINES_PATH)
    close = kl["close"].astype(float)
    ret = close.pct_change()
    rvol = ret.rolling(VOL_WIN).std()
    oos["rvol"] = rvol.reindex(oos.index).values
    oos = oos[oos["rvol"].notna() & (oos["rvol"] > 0)].copy()

    direction, tier, conf = decode(oos["pred_ret"].values)
    oos["dir"], oos["tier"], oos["conf"] = direction, tier, conf

    sig = oos[oos["dir"] != "NEUTRAL"].copy().sort_index()
    span_days = (sig.index[-1] - sig.index[0]).days or 1
    dsign = np.where(sig["dir"].values == "UP", 1.0, -1.0)
    r = dsign * sig["y_path_ret_4h"].values - COST
    p_abs = np.abs(sig["pred_ret"].values)
    c = sig["conf"].values / 100.0
    rv = sig["rvol"].values
    is_strong = (sig["tier"].values == "Strong").astype(float)

    print(f"signals={len(sig)} span={span_days}d cost={COST*1e4:.0f}bps "
          f"target_avg_lev={TARGET_AVG_LEV}x (all rules normalized to this)")

    rules = {
        "flat (every signal equal)": np.ones(len(sig)),
        "tier (Strong 1.0 / Mod 0.5)": np.where(is_strong > 0, 1.0, 0.5),
        "confidence-scaled": c,
        "confidence^2 (aggressive edge)": c ** 2,
        "vol-target (1/rvol)": 1.0 / rv,
        "half-Kelly (|pred|/rvol^2)": p_abs / (rv ** 2),
        "conf x vol-target (c/rvol)": c / rv,
    }
    rows = []
    for name, w in rules.items():
        L = normalize(w)
        m = metrics(L, r, span_days)
        m["rule"] = name
        rows.append(m)

    print(f"\n{'rule':32s} {'term':>6s} {'Sharpe':>7s} {'MDD%':>7s} "
          f"{'maxLev':>7s} {'worst%':>7s} ruin")
    base = next(x for x in rows if x["rule"].startswith("flat"))
    for m in sorted(rows, key=lambda x: -x["sharpe"]):
        d = "  <= flat baseline" if m is base else ""
        print(f"  {m['rule']:30s} {m['term']:6.2f} {m['sharpe']:7.2f} "
              f"{m['mdd']:7.1f} {m['max_lev']:7.2f} {m['worst']:7.1f} "
              f"{'YES' if m['ruin'] else 'no':>4s}{d}")

    print("\nRead: same avg leverage (1.0x) for all -> differences are pure")
    print("ALLOCATION SHAPE. Higher Sharpe + shallower MDD = smarter sizing.")
    print("term = equity multiple over the OOS window at 1.0x avg leverage.")

    # bootstrap: is the best rule's Sharpe edge over flat real, or noise?
    best = max(rows, key=lambda x: x["sharpe"])
    if best is not base:
        rng = np.random.default_rng(42)
        wb = normalize(rules[best["rule"]])
        diffs = []
        pnl_b = wb * r
        pnl_f = normalize(rules["flat (every signal equal)"]) * r
        for _ in range(3000):
            idx = rng.integers(0, len(r), len(r))
            sb, sf = pnl_b[idx], pnl_f[idx]
            sh_b = sb.mean() / sb.std(ddof=1) if sb.std(ddof=1) > 0 else 0
            sh_f = sf.mean() / sf.std(ddof=1) if sf.std(ddof=1) > 0 else 0
            diffs.append(sh_b - sh_f)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        frac = float(np.mean(np.array(diffs) > 0))
        print(f"\nBest='{best['rule']}' vs flat, per-signal Sharpe diff "
              f"95% CI [{lo:+.3f},{hi:+.3f}], P(better)={frac*100:.0f}% "
              f"-> {'SIGNIFICANT' if lo > 0 else 'NOT significant (CI spans 0)'}")

        # per-fold consistency (mistake.md 2026-06-02): is the edge broad or
        # driven by a few folds? Compare mean per-signal PnL per fold.
        sig2 = sig.copy()
        sig2["pnl_best"] = pnl_b
        sig2["pnl_flat"] = pnl_f
        fold_win = []
        for f, g in sig2.groupby("fold"):
            if len(g) >= 3:
                fold_win.append(g["pnl_best"].mean() - g["pnl_flat"].mean())
        fold_win = np.array(fold_win)
        print(f"per-fold mean-PnL diff (best-flat): {np.mean(fold_win)*1e4:+.1f}bps "
              f"mean, {(fold_win > 0).mean()*100:.0f}% of {len(fold_win)} folds "
              f"positive, median {np.median(fold_win)*1e4:+.1f}bps")


if __name__ == "__main__":
    main()
