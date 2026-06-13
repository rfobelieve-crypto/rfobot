"""
Phase-1 validation for the "微观阻力清算图" (micro-resistance liquidation map).

The Notion product doc describes a directional *resistance asymmetry* signal:
green bar  = downside resistance huge + upside vacuum -> price floats up easy
red   bar  = upside resistance huge + downside collapse -> price drops easy

Two resistance components in the doc:
  (1) 摩擦力 / 筹码阻力  = dense historical traded volume = friction
  (2) 引力 / 清算真空    = clustered trapped leveraged liquidation levels = suction

Component (2) needs OI+leverage reverse-engineering of liquidation PRICE levels
(the doc itself calls the mechanical version reflexive/failure-prone) -> deferred.

Component (1) -- a true trailing VOLUME PROFILE (VPVR) -- is the part that is
genuinely NEW data versus V7. V7's existing cg_liq_* / depth / swing-distance
features were already tested and found redundant (mistake.md 2026-06-01/02), but
chip-distribution (volume-at-price) has never been implemented in this system
(B_vpoc_dist was flagged NOT IMPLEMENTED). So it is worth a cheap first pass.

This script does NOT deploy anything. It runs the two gates from mistake.md:
  GATE 1  raw walk-forward IC vs y_path_ret_4h  (is there ANY signal?)
          + block-bootstrap CI + monthly stability
  GATE 2  conditional IC vs V7 OOS residual     (does V7 already absorb it?)

Only if GATE 2 is significant is a full ensemble A/B worth running (phase 2).

Output: research/results/dual_model/resistance_map_phase1.csv
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import load_and_cache_data, RESULTS_DIR
from research.dual_model.build_direction_reg_labels import build_direction_reg_labels
from research.dual_model.direction_features_v2 import FULL_DIRECTION
from research.dual_model.train_direction_reg_4h import train_direction_reg_walk_forward

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# Resistance-asymmetry grid. Keep small to limit multiple-comparison inflation.
# band = +/- price fraction around current close; lookback = trailing bars (1h).
BANDS = [0.005, 0.010, 0.020]
LOOKBACKS = [168, 720]          # 7d, 30d
EPS = 1e-9


# ----------------------------------------------------------------------------
# Volume-profile resistance asymmetry
# ----------------------------------------------------------------------------
def build_vp_resistance_features(
    klines: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Trailing volume-profile resistance asymmetry.

    For each bar t (price P = close[t]) and each past bar i in (t-L, t-1]:
      distribute bar i's notional (quote_vol) uniformly across [low_i, high_i].
      mass_up = volume mass landing in (P, P+band*P]   (chips ABOVE = up friction)
      mass_dn = volume mass landing in [P-band*P, P)   (chips BELOW = down friction)

    asym = (mass_dn - mass_up) / (mass_dn + mass_up)   in [-1, 1]
        > 0  -> more chips below, vacuum above -> easy UP  (green, doc semantics)
        < 0  -> more chips above, collapse below -> easy DOWN (red)

    lr   = log((mass_dn+eps)/(mass_up+eps))            unbounded variant

    Strictly trailing: bar t uses only bars < t. klines extends earlier than
    target_index so even the first target bar gets a full lookback.
    """
    k = klines.sort_index()
    low = k["low"].values.astype(float)
    high = k["high"].values.astype(float)
    close = k["close"].values.astype(float)
    # notional traded; quote_vol is in USDT (price*size), best mass proxy
    qv = (k["quote_vol"].values.astype(float)
          if "quote_vol" in k.columns
          else (k["volume"].values.astype(float) * close))
    rng = np.maximum(high - low, EPS)
    n = len(k)
    max_L = max(LOOKBACKS)

    cols: dict[str, np.ndarray] = {}
    for band in BANDS:
        for L in LOOKBACKS:
            cols[f"vp_asym_b{int(band*1e4):04d}_l{L}"] = np.full(n, np.nan)
            cols[f"vp_lr_b{int(band*1e4):04d}_l{L}"] = np.full(n, np.nan)

    for t in range(max_L, n):
        P = close[t]
        for L in LOOKBACKS:
            s = t - L
            lo_w = low[s:t]
            hi_w = high[s:t]
            m_w = qv[s:t]
            rng_w = rng[s:t]
            for band in BANDS:
                up_lo, up_hi = P, P * (1.0 + band)
                dn_lo, dn_hi = P * (1.0 - band), P
                # overlap of each bar's [lo,hi] with the up / down band
                ov_up = np.clip(np.minimum(hi_w, up_hi) - np.maximum(lo_w, up_lo),
                                0.0, None)
                ov_dn = np.clip(np.minimum(hi_w, dn_hi) - np.maximum(lo_w, dn_lo),
                                0.0, None)
                mass_up = float(np.sum(m_w * ov_up / rng_w))
                mass_dn = float(np.sum(m_w * ov_dn / rng_w))
                denom = mass_up + mass_dn + EPS
                bk = f"b{int(band*1e4):04d}_l{L}"
                cols[f"vp_asym_{bk}"][t] = (mass_dn - mass_up) / denom
                cols[f"vp_lr_{bk}"][t] = np.log((mass_dn + EPS) / (mass_up + EPS))

    out = pd.DataFrame(cols, index=k.index)
    return out.reindex(target_index)


# ----------------------------------------------------------------------------
# Stats helpers
# ----------------------------------------------------------------------------
def block_bootstrap_ic_ci(x: np.ndarray, y: np.ndarray,
                          block: int = 24, n_boot: int = 2000,
                          seed: int = 42) -> tuple[float, float, float]:
    """Moving-block bootstrap CI for Spearman IC (respects serial correlation)."""
    rng = np.random.default_rng(seed)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < block * 3:
        return float("nan"), float("nan"), float("nan")
    point = float(spearmanr(x, y).correlation)
    n_blocks = int(np.ceil(n / block))
    starts_pool = np.arange(0, n - block + 1)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.choice(starts_pool, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        r = spearmanr(x[idx], y[idx]).correlation
        boots[b] = r if np.isfinite(r) else 0.0
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def monthly_ic(feat: pd.Series, y: pd.Series) -> dict:
    df = pd.DataFrame({"f": feat, "y": y}).dropna()
    out = {}
    for mlabel, g in df.groupby(df.index.to_period("M")):
        if len(g) >= 30:
            out[str(mlabel)] = float(spearmanr(g["f"], g["y"]).correlation)
    return out


# ----------------------------------------------------------------------------
def main() -> int:
    logger.info("Loading cached features…")
    df = load_and_cache_data(limit=4000)
    logger.info("Loaded: %d bars × %d cols (%s ~ %s)",
                *df.shape, df.index[0], df.index[-1])

    # Extended klines for full lookback warmup at the start of the cache window
    raw_klines = pd.read_parquet(
        PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet")
    if raw_klines.index.name != "dt":
        raw_klines = raw_klines.copy()
        raw_klines.index = pd.to_datetime(raw_klines["ts_open"], unit="ms", utc=True)
    raw_klines = raw_klines.sort_index()

    logger.info("Building volume-profile resistance features "
                "(%d bands × %d lookbacks)…", len(BANDS), len(LOOKBACKS))
    vp = build_vp_resistance_features(raw_klines, df.index)
    feat_cols = list(vp.columns)
    cov = {c: int(vp[c].notna().sum()) for c in feat_cols}
    logger.info("VP feature coverage (of %d bars): %s", len(df), cov)

    # Labels (same TWAP path target the production model uses)
    labels = build_direction_reg_labels(df)
    y = labels["y_path_ret_4h"]

    # ---------------- GATE 1: raw IC ----------------
    logger.info("=" * 70)
    logger.info("GATE 1: raw Spearman IC vs y_path_ret_4h (block-bootstrap CI)")
    logger.info("=" * 70)
    rows = []
    for c in feat_cols:
        x = vp[c]
        point, lo, hi = block_bootstrap_ic_ci(x.values, y.values)
        m_ic = monthly_ic(x, y)
        mvals = list(m_ic.values())
        frac_same = (np.mean([np.sign(v) == np.sign(point) for v in mvals])
                     if mvals else float("nan"))
        ci_excl0 = np.isfinite(lo) and np.isfinite(hi) and (lo * hi > 0)
        rows.append(dict(
            feature=c, raw_ic=point, ci_lo=lo, ci_hi=hi,
            ci_excludes_0=ci_excl0,
            n_months=len(mvals), frac_months_same_sign=frac_same,
            monthly=";".join(f"{k}:{v:+.3f}" for k, v in m_ic.items()),
        ))
        logger.info("  %-22s IC=%+.4f  CI[%+.4f,%+.4f] %s  months=%d same=%.0f%%",
                    c, point, lo, hi, "EXCL0" if ci_excl0 else "incl0",
                    len(mvals), 100 * frac_same if mvals else 0)

    gate1 = pd.DataFrame(rows)
    # candidates worth the (expensive) conditional test: |IC|>=0.03 & CI excl 0
    g1_pass = gate1[(gate1["raw_ic"].abs() >= 0.03) & gate1["ci_excludes_0"]]
    logger.info("GATE 1 survivors (|IC|>=0.03 & CI excludes 0): %d / %d",
                len(g1_pass), len(gate1))

    # ---------------- GATE 2: conditional IC vs V7 residual ----------------
    logger.info("=" * 70)
    logger.info("GATE 2: V7 baseline walk-forward → conditional IC vs residual")
    logger.info("=" * 70)
    oos_base, metrics_base, _ = train_direction_reg_walk_forward(
        df, FULL_DIRECTION, objective="mse")
    logger.info("V7 baseline OOS: IC=%+.4f  AUC=%.4f  (sanity vs canonical 0.17/0.59)",
                metrics_base["spearman_ic"], metrics_base["auc_sign"])

    resid = (oos_base["y_path_ret_4h"] - oos_base["pred_ret"])
    resid.name = "resid"
    cond_rows = []
    for c in feat_cols:
        xc = vp[c].reindex(oos_base.index)
        point, lo, hi = block_bootstrap_ic_ci(xc.values, resid.values)
        ci_excl0 = np.isfinite(lo) and np.isfinite(hi) and (lo * hi > 0)
        cond_rows.append(dict(
            feature=c, cond_ic=point, cond_ci_lo=lo, cond_ci_hi=hi,
            cond_ci_excludes_0=ci_excl0,
        ))
        logger.info("  %-22s cond_IC=%+.4f  CI[%+.4f,%+.4f] %s",
                    c, point, lo, hi, "EXCL0" if ci_excl0 else "incl0")

    gate2 = pd.DataFrame(cond_rows)
    g2_pass = gate2[(gate2["cond_ic"].abs() >= 0.03) & gate2["cond_ci_excludes_0"]]

    # ---------------- merge + verdict ----------------
    report = gate1.merge(gate2, on="feature")
    report["n_oos"] = len(oos_base)
    out_csv = RESULTS_DIR / "resistance_map_phase1.csv"
    report.to_csv(out_csv, index=False)

    print("\n" + "=" * 92)
    print("RESISTANCE-MAP PHASE 1  —  VERDICT")
    print("=" * 92)
    print(f"V7 baseline OOS IC={metrics_base['spearman_ic']:+.4f} "
          f"AUC={metrics_base['auc_sign']:.4f}  (n_oos={len(oos_base)})")
    print(f"\n{'feature':22s} {'raw_IC':>8s} {'rawCI0':>7s} "
          f"{'cond_IC':>8s} {'condCI0':>8s}")
    print("-" * 92)
    for _, r in report.iterrows():
        print(f"{r['feature']:22s} {r['raw_ic']:>+8.4f} "
              f"{'excl' if r['ci_excludes_0'] else 'incl':>7s} "
              f"{r['cond_ic']:>+8.4f} "
              f"{'excl' if r['cond_ci_excludes_0'] else 'incl':>8s}")
    print("-" * 92)
    print(f"GATE 1 pass (raw |IC|>=0.03 & CI excl 0): {len(g1_pass)}/{len(gate1)}")
    print(f"GATE 2 pass (cond |IC|>=0.03 & CI excl 0): {len(g2_pass)}/{len(gate2)}")
    if len(g2_pass) > 0:
        print("\n>>> GATE 2 SURVIVORS — worth a full ensemble A/B (phase 2):")
        for f in g2_pass["feature"]:
            print(f"      {f}")
        print("    (A/B must still pass per-fold mean + frac_pos + bootstrap"
              " CI — mistake.md 2026-06-02)")
    else:
        print("\n>>> NO-GO: no feature carries marginal info beyond V7."
              "\n    V7 already absorbs the volume-profile resistance signal."
              "\n    Do NOT build the chart panel as a model input on this basis.")
    print("=" * 92)
    print(f"\nWrote → {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
