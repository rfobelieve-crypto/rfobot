"""
Phase 1A — IC screen for Direction Feature Pack (DFP) candidates.

Builds ~20 trailing-only candidate features targeting the direction-ranking
weakness of the current Init model (top-5% WR ~48% means probability ranking
has no real edge). Screens each against:
    - y_long_touch / y_short_touch  (binary first-touch labels)
    - path_max_up_4h / path_max_dn_4h  (continuous forward path stats)

Reports:
    1. Spearman IC (full sample)
    2. Monthly IC stability (6 months)
    3. Inter-correlation vs existing init_* features (flag redundancy > 0.85)
    4. Survivor list: |IC| >= 0.03 AND same-sign in >= 4/6 months

No model training here. Pure feature screening.
Run: python -m research.direction_feature_ic_screen
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import load_and_cache_data
from research.dual_model.build_initiation_labels import (
    build_initiation_labels, InitiationLabelConfig,
)
from indicator.initiation_features import (
    add_initiation_features, INITIATION_FEATURE_COLS,
)

logger = logging.getLogger(__name__)
RESULTS_DIR = PROJECT_ROOT / "research" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IC_THRESHOLD = 0.03           # minimum |IC| to survive screen
MONTH_STABILITY_MIN = 4       # require same-sign IC in >= 4/6 months
CORR_REDUNDANCY_MAX = 0.85    # drop features >0.85 correlated with existing init_*


# ═══════════════════════════════════════════════════════════════════════
#   Helpers
# ═══════════════════════════════════════════════════════════════════════

def signed_streak(s: pd.Series, cap: int = 24) -> pd.Series:
    """
    Consecutive same-sign run length (positive = bullish streak, negative
    = bearish streak). Uses shift(1) so no look-ahead.

    Example input:  [+, +, +, -, -, +, +]
            output: [1, 2, 3, -1, -2, 1, 2]  shifted by 1
    """
    prev = s.shift(1)
    sign = np.sign(prev).fillna(0).astype(int)
    # New group starts whenever sign changes
    group_id = (sign != sign.shift()).cumsum()
    # cumcount within each group (0-indexed) → +1 for natural streak length
    run = prev.groupby(group_id).cumcount() + 1
    # Make bearish runs negative so XGB sees signed direction
    return (run * sign).clip(-cap, cap)


def flag_streak(flag: pd.Series, cap: int = 24) -> pd.Series:
    """Consecutive bars where a boolean flag is True. Trailing-only."""
    f = flag.shift(1).fillna(0).astype(int)
    group_id = (f != f.shift()).cumsum()
    run = (f.groupby(group_id).cumcount() + 1) * f
    return run.clip(0, cap)


def pct_rank_trailing(s: pd.Series, window: int) -> pd.Series:
    """Percentile rank of s[t] within trailing window [t-window, t-1]."""
    return s.shift(1).rolling(window, min_periods=max(4, window // 2)).apply(
        lambda x: float((x[-1] >= x[:-1]).mean()) if len(x) > 1 else np.nan,
        raw=True,
    )


# ═══════════════════════════════════════════════════════════════════════
#   Feature builders (all trailing-only)
# ═══════════════════════════════════════════════════════════════════════

def build_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all ~20 DFP candidates. Returns a DataFrame indexed same as df
    with new columns (not modifying df). Every column is strictly trailing.
    """
    out = pd.DataFrame(index=df.index)

    # ── Group A: Flow Persistence (signed streak) ──────────────────────────
    if "cg_fcvd_delta" in df.columns:
        out["dfp_fcvd_sign_streak"] = signed_streak(df["cg_fcvd_delta"], cap=24)
    if "cg_scvd_delta" in df.columns:
        out["dfp_scvd_sign_streak"] = signed_streak(df["cg_scvd_delta"], cap=24)
    if "cg_taker_delta" in df.columns:
        out["dfp_taker_sign_streak"] = signed_streak(df["cg_taker_delta"], cap=24)
    if "cg_oi_delta" in df.columns:
        out["dfp_oi_sign_streak"] = signed_streak(df["cg_oi_delta"], cap=24)
    if "cg_pos_ls_ratio" in df.columns:
        pos_d = df["cg_pos_ls_ratio"].diff()
        out["dfp_pos_ls_sign_streak"] = signed_streak(pos_d, cap=24)

    # ── Group B: Multi-Source Flow Consensus ───────────────────────────────
    flow_sources = []
    for c in ["cg_fcvd_delta", "cg_scvd_delta", "cg_taker_delta", "large_bar_delta"]:
        if c in df.columns:
            flow_sources.append(np.sign(df[c].shift(1).fillna(0)))
    if len(flow_sources) >= 2:
        # signed sum: -N (all bearish) to +N (all bullish)
        consensus_signed = sum(flow_sources)
        out["dfp_flow_consensus_signed"] = consensus_signed
        out["dfp_flow_consensus_persist_8h"] = consensus_signed.rolling(
            8, min_periods=2).mean()
        # Magnitude-weighted consensus: weight each source by its zscore
        weighted = pd.Series(0.0, index=df.index)
        for c, zcol in [
            ("cg_fcvd_delta", "cg_fcvd_delta_zscore"),
            ("cg_scvd_delta", "cg_scvd_delta_zscore"),
            ("cg_taker_delta", "cg_taker_delta_zscore"),
        ]:
            if zcol in df.columns:
                weighted = weighted + df[zcol].shift(1).fillna(0).clip(-3, 3)
        if weighted.abs().sum() > 0:
            out["dfp_flow_consensus_zweighted"] = weighted

    # Spot vs futures CVD leadership
    if "cg_scvd_delta" in df.columns and "cg_fcvd_delta" in df.columns:
        sd = df["cg_scvd_delta"].shift(1)
        fd = df["cg_fcvd_delta"].shift(1)
        # signed ratio: + means spot dominates in same direction as futures,
        # - means spot leading opposite (divergence)
        denom = fd.abs() + sd.abs() + 1e-9
        out["dfp_spot_lead_ratio"] = (sd - fd) / denom

    # Retail vs whale divergence
    if "cg_ls_ratio" in df.columns and "cg_pos_ls_ratio" in df.columns:
        retail_d = df["cg_ls_ratio"].diff().shift(1)
        whale_d = df["cg_pos_ls_ratio"].diff().shift(1)
        out["dfp_retail_whale_div"] = retail_d - whale_d

    # ── Group C: Coinbase Premium (US institutional flow) ──────────────────
    if "cg_cb_premium_rate" in df.columns:
        cbp = df["cg_cb_premium_rate"]
        out["dfp_cb_premium_sign_streak"] = signed_streak(cbp, cap=24)
        out["dfp_cb_premium_slope_4h"] = cbp.shift(1).diff(4)
        if "cg_funding_close" in df.columns:
            # Divergence: funding says crowd is long but institutions (CB) are flat/short
            fund_sign = np.sign(df["cg_funding_close"].shift(1))
            cbp_sign = np.sign(cbp.shift(1))
            out["dfp_cb_funding_divergence"] = (fund_sign != cbp_sign).astype(float) * fund_sign

    # ── Group D: OI × Price Regime (position build/cover persistence) ──────
    if "cg_oi_delta" in df.columns:
        d_close = df["close"].diff().shift(1)
        d_oi = df["cg_oi_delta"].shift(1)

        long_build_flag = ((d_close > 0) & (d_oi > 0)).astype(float)
        short_build_flag = ((d_close < 0) & (d_oi > 0)).astype(float)
        long_cover_flag = ((d_close < 0) & (d_oi < 0)).astype(float)
        short_cover_flag = ((d_close > 0) & (d_oi < 0)).astype(float)

        out["dfp_long_build_persist_8h"] = long_build_flag.rolling(
            8, min_periods=2).sum()
        out["dfp_short_build_persist_8h"] = short_build_flag.rolling(
            8, min_periods=2).sum()
        out["dfp_net_build_8h"] = (
            out["dfp_long_build_persist_8h"] - out["dfp_short_build_persist_8h"]
        )
        # Reversal setup: short_cover = bullish reversal, long_cover = bearish reversal
        out["dfp_short_cover_persist_8h"] = short_cover_flag.rolling(
            8, min_periods=2).sum()
        out["dfp_long_cover_persist_8h"] = long_cover_flag.rolling(
            8, min_periods=2).sum()

    # ── Group E: Flow Extreme (24h percentile rank) ────────────────────────
    if "cg_fcvd_cum" in df.columns:
        out["dfp_fcvd_cum_24h_rank"] = pct_rank_trailing(df["cg_fcvd_cum"], 24)
    if "cg_scvd_cum" in df.columns:
        out["dfp_scvd_cum_24h_rank"] = pct_rank_trailing(df["cg_scvd_cum"], 24)
    if "cg_taker_delta" in df.columns:
        taker_4h = df["cg_taker_delta"].rolling(4, min_periods=2).sum()
        out["dfp_taker_4h_24h_rank"] = pct_rank_trailing(taker_4h, 24)

    # ── Group F: Liquidation Cascade Direction ─────────────────────────────
    if "cg_liq_long" in df.columns and "cg_liq_short" in df.columns:
        ll = df["cg_liq_long"]
        ls = df["cg_liq_short"]
        long_dom = (ll > ls).astype(float)
        short_dom = (ls > ll).astype(float)
        out["dfp_liq_long_dom_streak"] = flag_streak(long_dom, cap=12)
        out["dfp_liq_short_dom_streak"] = flag_streak(short_dom, cap=12)

        # Post-cascade reversal setup
        if "cg_taker_delta" in df.columns:
            ll_4h = ll.rolling(4, min_periods=2).sum().shift(1)
            ls_4h = ls.rolling(4, min_periods=2).sum().shift(1)
            taker_streak = signed_streak(df["cg_taker_delta"], cap=12)
            ll_extreme = (ll_4h > ll_4h.rolling(48, min_periods=12).quantile(0.9))
            ls_extreme = (ls_4h > ls_4h.rolling(48, min_periods=12).quantile(0.9))
            out["dfp_post_long_liq_rev"] = ll_extreme.astype(float) * (
                taker_streak.clip(lower=0))
            out["dfp_post_short_liq_rev"] = ls_extreme.astype(float) * (
                (-taker_streak).clip(lower=0))

    return out


# ═══════════════════════════════════════════════════════════════════════
#   Scoring
# ═══════════════════════════════════════════════════════════════════════

def ic_single(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Spearman IC and p-value (NaN-safe)."""
    m = x.notna() & y.notna()
    if m.sum() < 50:
        return (np.nan, np.nan)
    try:
        r, p = spearmanr(x[m], y[m])
        return (float(r), float(p))
    except Exception:
        return (np.nan, np.nan)


def monthly_ic(x: pd.Series, y: pd.Series) -> dict:
    """Per-month IC. Returns dict month->IC and stability score."""
    months = x.index.strftime("%Y-%m")
    unique_months = sorted(set(months))
    out = {}
    for m in unique_months:
        mask = (months == m)
        if mask.sum() < 30:
            continue
        r, _ = ic_single(x[mask], y[mask])
        if not np.isnan(r):
            out[m] = r
    return out


def screen_features(feats: pd.DataFrame, labels: pd.DataFrame,
                    existing_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each candidate feature, compute IC vs 4 labels + monthly stability
    + redundancy vs existing init_* features. Returns scored DataFrame.
    """
    label_cols = {
        "y_long_touch": labels["y_long_touch"],
        "y_short_touch": labels["y_short_touch"],
        "path_max_up_4h": labels["path_max_up_4h"],
        "path_max_dn_4h": labels["path_max_dn_4h"],
    }

    existing_init = [c for c in INITIATION_FEATURE_COLS if c in existing_df.columns]
    existing_vals = existing_df[existing_init]

    rows = []
    for feat_name in feats.columns:
        x = feats[feat_name]
        coverage = x.notna().mean()

        row = {"feature": feat_name, "coverage": coverage}

        # Full-sample IC on 4 labels
        for lbl_name, y in label_cols.items():
            r, _ = ic_single(x, y)
            row[f"ic_{lbl_name}"] = r

        # Monthly stability on primary target (y_long_touch AND y_short_touch combined)
        # We track long-side stability (months where IC sign matches full IC)
        long_full = row["ic_y_long_touch"]
        if not np.isnan(long_full):
            mic = monthly_ic(x, label_cols["y_long_touch"])
            if mic:
                same_sign = sum(
                    1 for v in mic.values()
                    if (v >= 0) == (long_full >= 0) and abs(v) > 0.01
                )
                row["months_stable_long"] = same_sign
                row["n_months"] = len(mic)
            else:
                row["months_stable_long"] = 0
                row["n_months"] = 0
        else:
            row["months_stable_long"] = 0
            row["n_months"] = 0

        # Inter-correlation with existing init_* features (max abs)
        max_redundancy = 0.0
        redundant_with = ""
        for c in existing_init:
            try:
                m = x.notna() & existing_vals[c].notna()
                if m.sum() < 100:
                    continue
                corr = x[m].corr(existing_vals[c][m])
                if not np.isnan(corr) and abs(corr) > max_redundancy:
                    max_redundancy = abs(corr)
                    redundant_with = c
            except Exception:
                continue
        row["max_init_corr"] = max_redundancy
        row["redundant_with"] = redundant_with

        rows.append(row)

    return pd.DataFrame(rows)


def apply_survival_rules(scored: pd.DataFrame) -> pd.DataFrame:
    """Apply the 3 survival rules and mark each feature PASS/FAIL/REDUNDANT."""
    scored = scored.copy()

    # Rule 1: at least one label IC has |value| >= threshold
    ic_cols = [c for c in scored.columns if c.startswith("ic_")]
    scored["max_abs_ic"] = scored[ic_cols].abs().max(axis=1)
    scored["primary_label"] = scored[ic_cols].abs().idxmax(axis=1)

    rule1 = scored["max_abs_ic"] >= IC_THRESHOLD
    # Rule 2: monthly stability on long-side IC
    rule2 = scored["months_stable_long"] >= MONTH_STABILITY_MIN
    # Rule 3: not redundant with existing init_*
    rule3 = scored["max_init_corr"] < CORR_REDUNDANCY_MAX

    status = []
    for r1, r2, r3 in zip(rule1, rule2, rule3):
        if not r3:
            status.append("REDUNDANT")
        elif r1 and r2:
            status.append("PASS")
        elif r1 and not r2:
            status.append("FAIL_unstable")
        else:
            status.append("FAIL_weak_ic")
    scored["status"] = status
    return scored


# ═══════════════════════════════════════════════════════════════════════
#   Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_and_cache_data()
    df = add_initiation_features(df)

    # Build labels (first-touch, k=0.8%)
    cfg = InitiationLabelConfig(k_pct=0.008, breakout_lookback=20,
                                 use_breakout_confirm=True, horizon_bars=4)
    labels = build_initiation_labels(df, cfg)
    print(f"Data: {len(df)} bars, labels built")
    print(f"y_long_touch pos rate: {labels['y_long_touch'].mean()*100:.2f}%")
    print(f"y_short_touch pos rate: {labels['y_short_touch'].mean()*100:.2f}%\n")

    # Build candidate features
    feats = build_candidate_features(df)
    print(f"Built {len(feats.columns)} candidate features\n")

    # Score
    scored = screen_features(feats, labels, df)
    scored = apply_survival_rules(scored)
    scored = scored.sort_values("max_abs_ic", ascending=False).reset_index(drop=True)

    # ── Report ────────────────────────────────────────────────────────────
    print("=" * 110)
    print(f"  IC SCREEN RESULTS (threshold |IC|>={IC_THRESHOLD}, "
          f"monthly stable>={MONTH_STABILITY_MIN}/6, "
          f"max init corr<{CORR_REDUNDANCY_MAX})")
    print("=" * 110)
    cols_show = [
        "feature", "status", "max_abs_ic",
        "ic_y_long_touch", "ic_y_short_touch",
        "ic_path_max_up_4h", "ic_path_max_dn_4h",
        "months_stable_long", "max_init_corr", "redundant_with",
    ]
    with pd.option_context("display.max_rows", 100,
                            "display.max_colwidth", 30,
                            "display.width", 200):
        print(scored[cols_show].to_string(index=False))

    print("\n" + "=" * 75)
    print("  SURVIVORS by status")
    print("=" * 75)
    for status, group in scored.groupby("status"):
        print(f"\n  [{status}] {len(group)} features")
        for _, r in group.iterrows():
            print(f"    {r['feature']:<36}  "
                  f"max|IC|={r['max_abs_ic']:+.3f}  "
                  f"months_stable={int(r['months_stable_long'])}/6")

    # Save
    out_path = RESULTS_DIR / "direction_feature_ic_screen.csv"
    scored.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")

    # Save survivors list for Phase 1B
    survivors = scored[scored["status"] == "PASS"]["feature"].tolist()
    surv_path = RESULTS_DIR / "direction_feature_survivors.json"
    import json
    surv_path.write_text(json.dumps(survivors, indent=2))
    print(f"Survivor list ({len(survivors)} features) → {surv_path}")


if __name__ == "__main__":
    main()
