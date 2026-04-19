"""
Initiation-model-specific features.

All features are strictly TRAILING-ONLY (use shift(1) before any rolling
window that would otherwise include the current bar's close).

Groups:
    A. Breakout confirmation        — close vs trailing rolling high/low
                                      normalized by ATR.
    B. Funding acceleration         — 1st and 2nd derivative of funding rate,
                                      plus sign-persistence.
    C. OI-price divergence          — trailing rolling corr(d_oi, d_close) and
                                      a directional divergence flag.
    D. Liquidation cluster proximity — trailing cumulative long/short liq
                                      pressure normalized by rolling baseline.
    E. Breakout strength composite  — combines A+C to grade init quality.

Design constraints:
    - No look-ahead: every rolling window excludes the current bar.
    - No new raw API columns — reuses columns already in the cache parquet
      (close, high, low, cg_oi_*, cg_funding_*, cg_liq_*).
    - Safe fillna=0 only on truly derivative columns; raw features preserve
      NaN so the caller can mask them out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BREAKOUT_LOOKBACK = 20
ATR_LOOKBACK = 14
FUNDING_DIFF_LAG = 1
FUNDING_ACCEL_LAG = 1
OI_DIVERGENCE_WINDOW = 12
LIQ_BASELINE_WINDOW = 48


def _trailing_roll(s: pd.Series, window: int, how: str) -> pd.Series:
    r = s.shift(1).rolling(window, min_periods=max(2, window // 2))
    if how == "max":
        return r.max()
    if how == "min":
        return r.min()
    if how == "mean":
        return r.mean()
    if how == "std":
        return r.std()
    raise ValueError(how)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         window: int = ATR_LOOKBACK) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.shift(1).rolling(window, min_periods=max(2, window // 2)).mean()


def add_initiation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append initiation-specific features to df IN PLACE and return df.

    Required columns:
        close, high, low
        cg_funding_close               (optional — skipped if missing)
        cg_oi_close                    (optional)
        cg_liq_long, cg_liq_short      (optional)
    """
    out = df

    close = out["close"].astype(float)
    high = out["high"].astype(float) if "high" in out.columns else close
    low = out["low"].astype(float) if "low" in out.columns else close
    open_ = out["open"].astype(float) if "open" in out.columns else close

    # --- A. Breakout confirmation -------------------------------------------
    prior_high = _trailing_roll(close, BREAKOUT_LOOKBACK, "max")
    prior_low = _trailing_roll(close, BREAKOUT_LOOKBACK, "min")
    atr = _atr(high, low, close, ATR_LOOKBACK)

    # Normalized distance above prior high / below prior low.
    out["init_break_up_atr"] = (close - prior_high) / atr.replace(0, np.nan)
    out["init_break_dn_atr"] = (prior_low - close) / atr.replace(0, np.nan)

    # V5 "close-from-inside" breakout: require the bar to OPEN inside the prior
    # range and CLOSE outside it. This filters out gap-through breakouts which
    # are historically lower-quality (see research/breakout_gate_sweep.py:
    # V5 lifted Strong WR 64.9% -> 69.1%, CI_lo +3.6pp vs naive close>prior_high).
    out["init_bo_up"] = ((close > prior_high) & (open_ <= prior_high)).astype(float)
    out["init_bo_dn"] = ((close < prior_low) & (open_ >= prior_low)).astype(float)
    out.loc[prior_high.isna(), "init_bo_up"] = np.nan
    out.loc[prior_low.isna(), "init_bo_dn"] = np.nan

    # Breakout streak: how many bars has close stayed above prior_high?
    # Computed on past bars only (shift(1) on the flag).
    bo_up_prev = out["init_bo_up"].shift(1).fillna(0)
    bo_dn_prev = out["init_bo_dn"].shift(1).fillna(0)
    out["init_bo_up_streak"] = (
        bo_up_prev.groupby((bo_up_prev != bo_up_prev.shift()).cumsum()).cumcount() + 1
    ) * bo_up_prev
    out["init_bo_dn_streak"] = (
        bo_dn_prev.groupby((bo_dn_prev != bo_dn_prev.shift()).cumsum()).cumcount() + 1
    ) * bo_dn_prev

    # --- B. Funding acceleration --------------------------------------------
    if "cg_funding_close" in out.columns:
        fund = out["cg_funding_close"].astype(float)
        fund_d1 = fund - fund.shift(FUNDING_DIFF_LAG)
        fund_d2 = fund_d1 - fund_d1.shift(FUNDING_ACCEL_LAG)
        out["init_funding_d1"] = fund_d1.shift(1)     # trailing only
        out["init_funding_d2"] = fund_d2.shift(1)

        # Sign persistence: # bars of same-sign funding change in last 8 bars
        sign = np.sign(fund_d1).shift(1)
        roll_sign = sign.rolling(8, min_periods=2).sum()
        out["init_funding_sign_persist_8h"] = roll_sign

    # --- C. OI-price divergence ---------------------------------------------
    if "cg_oi_close" in out.columns:
        oi = out["cg_oi_close"].astype(float)
        d_oi = oi.diff().shift(1)
        d_close = close.diff().shift(1)

        # Rolling Spearman is expensive; use Pearson on ranks of past window.
        def _roll_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
            return a.rolling(window, min_periods=max(4, window // 2)).corr(b)

        out["init_oi_close_corr_12h"] = _roll_corr(d_oi, d_close, OI_DIVERGENCE_WINDOW)

        # Divergence magnitudes:
        # + bullish init: price up AND oi up (both > 0)
        # + bearish init: price down AND oi up (shorts building)
        sign_close = np.sign(d_close)
        sign_oi = np.sign(d_oi)
        out["init_oi_bullish_build"] = ((sign_close > 0) & (sign_oi > 0)).astype(float)
        out["init_oi_bearish_build"] = ((sign_close < 0) & (sign_oi > 0)).astype(float)

    # --- D. Liquidation cluster proximity -----------------------------------
    if "cg_liq_long" in out.columns and "cg_liq_short" in out.columns:
        liq_long = out["cg_liq_long"].astype(float)
        liq_short = out["cg_liq_short"].astype(float)

        # Trailing 8h cumulative
        ll_cum_8h = liq_long.shift(1).rolling(8, min_periods=2).sum()
        ls_cum_8h = liq_short.shift(1).rolling(8, min_periods=2).sum()

        # Baseline: 48h rolling mean, strictly trailing
        ll_base = liq_long.shift(1).rolling(LIQ_BASELINE_WINDOW,
                                            min_periods=LIQ_BASELINE_WINDOW // 2).mean()
        ls_base = liq_short.shift(1).rolling(LIQ_BASELINE_WINDOW,
                                             min_periods=LIQ_BASELINE_WINDOW // 2).mean()

        out["init_liq_long_cluster"] = ll_cum_8h / (ll_base * 8 + 1e-9)
        out["init_liq_short_cluster"] = ls_cum_8h / (ls_base * 8 + 1e-9)

        # Directional cluster imbalance (normalized pressure)
        total = ll_cum_8h + ls_cum_8h
        out["init_liq_cluster_imb"] = np.where(
            total > 0, (ls_cum_8h - ll_cum_8h) / total, 0.0
        )

    # --- E. Composite breakout strength -------------------------------------
    # Simple orthogonal combo: sign of breakout × funding_d1 alignment
    if "init_funding_d1" in out.columns:
        up_align = (out["init_break_up_atr"].clip(lower=0)
                    * out["init_funding_d1"].fillna(0).clip(lower=0))
        dn_align = (out["init_break_dn_atr"].clip(lower=0)
                    * (-out["init_funding_d1"].fillna(0)).clip(lower=0))
        out["init_break_funding_align_up"] = up_align
        out["init_break_funding_align_dn"] = dn_align

    # --- F. Direction Feature Pack (Phase 1B, IC-screened) -----------------
    _add_direction_feature_pack(out)

    return out


# ═══════════════════════════════════════════════════════════════════════════
#   Direction Feature Pack (20 IC-screen survivors, Phase 1B)
#   Source: research/direction_feature_ic_screen.py
#   All features are strictly trailing-only.
# ═══════════════════════════════════════════════════════════════════════════

def _signed_streak(s: pd.Series, cap: int = 24) -> pd.Series:
    """Consecutive same-sign run length, signed. Trailing-only via shift(1)."""
    prev = s.shift(1)
    sign = np.sign(prev).fillna(0).astype(int)
    group_id = (sign != sign.shift()).cumsum()
    run = prev.groupby(group_id).cumcount() + 1
    return (run * sign).clip(-cap, cap)


def _flag_streak(flag: pd.Series, cap: int = 24) -> pd.Series:
    """Consecutive bars where a boolean flag is True. Trailing-only."""
    f = flag.shift(1).fillna(0).astype(int)
    group_id = (f != f.shift()).cumsum()
    run = (f.groupby(group_id).cumcount() + 1) * f
    return run.clip(0, cap)


def _pct_rank_trailing(s: pd.Series, window: int) -> pd.Series:
    """Percentile rank of s[t] within trailing window [t-window, t-1]."""
    return s.shift(1).rolling(window, min_periods=max(4, window // 2)).apply(
        lambda x: float((x[-1] >= x[:-1]).mean()) if len(x) > 1 else np.nan,
        raw=True,
    )


def _add_direction_feature_pack(out: pd.DataFrame) -> None:
    """Append the 20 IC-screen-surviving DFP features to `out` in place."""
    close = out["close"].astype(float) if "close" in out.columns else None

    # ── Group A: Flow persistence (signed streak) ─────────────────────────
    if "cg_fcvd_delta" in out.columns:
        out["dfp_fcvd_sign_streak"] = _signed_streak(out["cg_fcvd_delta"], cap=24)
    if "cg_scvd_delta" in out.columns:
        out["dfp_scvd_sign_streak"] = _signed_streak(out["cg_scvd_delta"], cap=24)
    if "cg_taker_delta" in out.columns:
        out["dfp_taker_sign_streak"] = _signed_streak(out["cg_taker_delta"], cap=24)
    if "cg_oi_delta" in out.columns:
        out["dfp_oi_sign_streak"] = _signed_streak(out["cg_oi_delta"], cap=24)
    if "cg_pos_ls_ratio" in out.columns:
        pos_d = out["cg_pos_ls_ratio"].diff()
        out["dfp_pos_ls_sign_streak"] = _signed_streak(pos_d, cap=24)

    # ── Group B: Multi-source flow consensus ──────────────────────────────
    flow_sources = []
    for c in ["cg_fcvd_delta", "cg_scvd_delta", "cg_taker_delta", "large_bar_delta"]:
        if c in out.columns:
            flow_sources.append(np.sign(out[c].shift(1).fillna(0)))
    if len(flow_sources) >= 2:
        consensus_signed = sum(flow_sources)
        out["dfp_flow_consensus_signed"] = consensus_signed
        out["dfp_flow_consensus_persist_8h"] = consensus_signed.rolling(
            8, min_periods=2).mean()
        weighted = pd.Series(0.0, index=out.index)
        has_z = False
        for c, zcol in [
            ("cg_fcvd_delta", "cg_fcvd_delta_zscore"),
            ("cg_scvd_delta", "cg_scvd_delta_zscore"),
            ("cg_taker_delta", "cg_taker_delta_zscore"),
        ]:
            if zcol in out.columns:
                weighted = weighted + out[zcol].shift(1).fillna(0).clip(-3, 3)
                has_z = True
        if has_z:
            out["dfp_flow_consensus_zweighted"] = weighted

    # Spot vs futures leadership
    if "cg_scvd_delta" in out.columns and "cg_fcvd_delta" in out.columns:
        sd = out["cg_scvd_delta"].shift(1)
        fd = out["cg_fcvd_delta"].shift(1)
        denom = fd.abs() + sd.abs() + 1e-9
        out["dfp_spot_lead_ratio"] = (sd - fd) / denom

    # Retail vs whale divergence
    if "cg_ls_ratio" in out.columns and "cg_pos_ls_ratio" in out.columns:
        retail_d = out["cg_ls_ratio"].diff().shift(1)
        whale_d = out["cg_pos_ls_ratio"].diff().shift(1)
        out["dfp_retail_whale_div"] = retail_d - whale_d

    # ── Group D: OI × Price regime (build / cover persistence) ────────────
    if "cg_oi_delta" in out.columns and close is not None:
        d_close = close.diff().shift(1)
        d_oi = out["cg_oi_delta"].shift(1)

        long_build_flag = ((d_close > 0) & (d_oi > 0)).astype(float)
        short_build_flag = ((d_close < 0) & (d_oi > 0)).astype(float)
        long_cover_flag = ((d_close < 0) & (d_oi < 0)).astype(float)
        short_cover_flag = ((d_close > 0) & (d_oi < 0)).astype(float)

        out["dfp_long_build_persist_8h"] = long_build_flag.rolling(
            8, min_periods=2).sum()
        short_build_8h = short_build_flag.rolling(8, min_periods=2).sum()
        out["dfp_net_build_8h"] = out["dfp_long_build_persist_8h"] - short_build_8h
        out["dfp_long_cover_persist_8h"] = long_cover_flag.rolling(
            8, min_periods=2).sum()
        out["dfp_short_cover_persist_8h"] = short_cover_flag.rolling(
            8, min_periods=2).sum()

    # ── Group E: Flow extreme (24h percentile rank) ───────────────────────
    if "cg_fcvd_cum" in out.columns:
        out["dfp_fcvd_cum_24h_rank"] = _pct_rank_trailing(out["cg_fcvd_cum"], 24)
    if "cg_scvd_cum" in out.columns:
        out["dfp_scvd_cum_24h_rank"] = _pct_rank_trailing(out["cg_scvd_cum"], 24)
    if "cg_taker_delta" in out.columns:
        taker_4h = out["cg_taker_delta"].rolling(4, min_periods=2).sum()
        out["dfp_taker_4h_24h_rank"] = _pct_rank_trailing(taker_4h, 24)

    # ── Group F: Liquidation cascade direction ────────────────────────────
    if "cg_liq_long" in out.columns and "cg_liq_short" in out.columns:
        ll = out["cg_liq_long"]
        ls = out["cg_liq_short"]
        long_dom = (ll > ls).astype(float)
        short_dom = (ls > ll).astype(float)
        out["dfp_liq_long_dom_streak"] = _flag_streak(long_dom, cap=12)
        out["dfp_liq_short_dom_streak"] = _flag_streak(short_dom, cap=12)

        if "cg_taker_delta" in out.columns:
            ll_4h = ll.rolling(4, min_periods=2).sum().shift(1)
            taker_streak = _signed_streak(out["cg_taker_delta"], cap=12)
            ll_extreme = (ll_4h > ll_4h.rolling(48, min_periods=12).quantile(0.9))
            out["dfp_post_long_liq_rev"] = ll_extreme.astype(float) * (
                taker_streak.clip(lower=0))


INITIATION_FEATURE_COLS = [
    # Original init_* set
    "init_break_up_atr",
    "init_break_dn_atr",
    "init_bo_up",
    "init_bo_dn",
    "init_bo_up_streak",
    "init_bo_dn_streak",
    "init_funding_d1",
    "init_funding_d2",
    "init_funding_sign_persist_8h",
    "init_oi_close_corr_12h",
    "init_oi_bullish_build",
    "init_oi_bearish_build",
    "init_liq_long_cluster",
    "init_liq_short_cluster",
    "init_liq_cluster_imb",
    "init_break_funding_align_up",
    "init_break_funding_align_dn",
    # Direction Feature Pack (Phase 1B, IC-screen survivors)
    "dfp_fcvd_cum_24h_rank",
    "dfp_scvd_cum_24h_rank",
    "dfp_flow_consensus_persist_8h",
    "dfp_long_build_persist_8h",
    "dfp_taker_4h_24h_rank",
    "dfp_liq_short_dom_streak",
    "dfp_liq_long_dom_streak",
    "dfp_fcvd_sign_streak",
    "dfp_flow_consensus_signed",
    "dfp_taker_sign_streak",
    "dfp_flow_consensus_zweighted",
    "dfp_net_build_8h",
    "dfp_long_cover_persist_8h",
    "dfp_scvd_sign_streak",
    "dfp_oi_sign_streak",
    "dfp_spot_lead_ratio",
    "dfp_short_cover_persist_8h",
    "dfp_pos_ls_sign_streak",
    "dfp_retail_whale_div",
    "dfp_post_long_liq_rev",
]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from research.dual_model.shared_data import load_and_cache_data

    df = load_and_cache_data()
    df = add_initiation_features(df)
    present = [c for c in INITIATION_FEATURE_COLS if c in df.columns]
    print(f"Added {len(present)} initiation features on {len(df)} bars")
    print(df[present].describe().T[["count", "mean", "std", "min", "max"]])
