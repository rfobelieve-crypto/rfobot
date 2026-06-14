"""
Feature family: xchan_div  (Cross-channel divergence & lead/lag)
2026-06-14 feature search.

Idea (per task): the V7 model already carries SINGLE-WINDOW, INSTANTANEOUS
cross-channel divergence features:
  - cg_spot_futures_cvd_divergence = scvd_delta_z - fcvd_delta_z   (1 bar)
  - cvd_persistence_12h           = z(12h sum of that divergence)
  - dfp_spot_lead_ratio           = (scvd_d - fcvd_d)/(|scvd_d|+|fcvd_d|)  (1 bar)
  - dfp_*_sign_streak             = single-channel signed run length
  - oi_price_divergence / funding_taker_align / cg_margin_funding_align

So to find MARGINAL info I must NOT rebuild those.  This family targets things
V7 does NOT have:
  (A) MULTI-WINDOW divergence accumulation / acceleration of spot-vs-perp CVD
      flow (not the 1-bar z-diff, not just the 12h sum).
  (B) Lead/lag dynamics: rolling cross-correlation & cumulative-lead sign of
      spot CVD vs perp CVD (who is dragging whom), trailing-only.
  (C) CROSS-channel sign-AGREEMENT streaks (two or three channels agreeing /
      disagreeing) -- not the existing single-channel sign streaks.
  (D) Z-scored GAPS between channel pairs that currently have NO divergence
      feature at all: funding vs coinbase-premium (leverage crowd vs spot
      allocators), ETF-net-flow vs funding (allocator vs leverage), OI-build
      vs spot-CVD (trapped positioning vs real spot demand).
  (E) Trapped-positioning: OI rising while spot CVD falling, accumulated.

HARD RULES honoured:
  1. Trailing-only.  Every rolling op is past-anchored; lead/lag uses shift(+k)
     of the LAGGING channel (past values), never shift(-k).  Streaks use the
     project's shift(1)-anchored _signed_streak pattern.
  2. No TA indicators (no MACD/EMA/RSI/Bollinger).  Only order-flow channel
     algebra + z-scores + sign streaks + raw return behaviour.
  3. No feat*sparse_indicator with <30% nonzero.  Sign-agreement features are
     dense (defined every bar); we check nonzero_frac in the screen.
  4. Daily alt-data (etf_*, fear_greed_*) already lagged in parquet -- we use
     etf_net_flow_usd / etf_flow_zscore_30d AS-IS, no re-shift, and build the
     ETF-vs-funding gap at the (already-lagged) daily resolution it lives on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.featsearch_lib import load_features, screen_candidates  # noqa: E402

RESULTS = PROJECT_ROOT / "research" / "results"
OUT_PARQUET = RESULTS / "newfeat_xchan_div.parquet"


# ----------------------------------------------------------------------------
# trailing-only primitives (past-anchored, no forward windows)
# ----------------------------------------------------------------------------
def zscore(s: pd.Series, win: int) -> pd.Series:
    mu = s.rolling(win, min_periods=max(4, win // 3)).mean()
    sd = s.rolling(win, min_periods=max(4, win // 3)).std().replace(0, np.nan)
    return (s - mu) / sd


def signed_streak(s: pd.Series, cap: int = 24) -> pd.Series:
    """Consecutive same-sign run length, signed. Trailing-only via shift(1)
    (the project's canonical _signed_streak)."""
    prev = s.shift(1)
    sign = np.sign(prev).fillna(0).astype(int)
    group_id = (sign != sign.shift()).cumsum()
    run = prev.groupby(group_id).cumcount() + 1
    return (run * sign).clip(-cap, cap)


def roll_corr(a: pd.Series, b: pd.Series, win: int) -> pd.Series:
    return a.rolling(win, min_periods=max(6, win // 2)).corr(b)


def safe_norm(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / (den.abs() + 1e-9)


def main() -> None:
    df = load_features()
    idx = df.index
    cand = pd.DataFrame(index=idx)

    # --- raw channels (all dense, nonnan ~1.0 verified) --------------------
    scvd_d = df["cg_scvd_delta"].astype(float)        # spot CVD per-bar delta
    fcvd_d = df["cg_fcvd_delta"].astype(float)        # perp CVD per-bar delta
    scvd_cum = df["cg_scvd_cum"].astype(float)        # spot CVD cumulative
    fcvd_cum = df["cg_fcvd_cum"].astype(float)        # perp CVD cumulative
    oi_d = df["cg_oi_delta"].astype(float)            # OI per-bar delta
    funding = df["cg_funding_close"].astype(float)
    cbprem = df["cg_cb_premium_rate"].astype(float)   # coinbase premium rate
    etf_z = df["etf_flow_zscore_30d"].astype(float)   # already-lagged daily
    logret = df["log_return"].astype(float)

    # robust per-channel normalisation so different $ scales are comparable.
    # rolling-rank-free: z-score each delta on a 168h (1w) trailing window.
    scvd_dz = zscore(scvd_d, 168)
    fcvd_dz = zscore(fcvd_d, 168)
    oi_dz = zscore(oi_d, 168)
    fund_z = zscore(funding, 168)
    cb_z = zscore(cbprem, 168)

    # the per-bar spot-vs-perp CVD gap (in comparable z-units).  NOTE this is
    # NOT the same as cg_spot_futures_cvd_divergence which uses the *_delta_zscore
    # columns (a 96h z); we use a 168h z and then build DYNAMICS on top.
    gap = scvd_dz - fcvd_dz   # >0 = spot buying harder than perp

    # =====================================================================
    # (A) MULTI-WINDOW divergence accumulation & acceleration
    #     V7 has the 1-bar gap and one 12h sum; here multiple horizons + the
    #     ACCELERATION (change of the accumulated gap), which V7 lacks.
    # =====================================================================
    gap_acc_8h = gap.rolling(8, min_periods=4).sum()
    gap_acc_24h = gap.rolling(24, min_periods=12).sum()
    # acceleration: how fast the 8h accumulated gap is itself changing vs 8h ago
    cand["xc_gap_accel_8h"] = gap_acc_8h - gap_acc_8h.shift(8)
    # span between short- and long-horizon accumulated gap (regime of divergence)
    cand["xc_gap_span_8_24h"] = zscore(gap_acc_8h, 168) - zscore(gap_acc_24h, 168)

    # =====================================================================
    # (B) LEAD/LAG dynamics of spot CVD vs perp CVD
    #     Who leads?  Trailing-only: compare corr(spot_now, perp_lagged) vs
    #     corr(perp_now, spot_lagged) using PAST bars only.
    # =====================================================================
    L = 3  # lead/lag horizon in bars (3h)
    # corr(spot delta, perp delta k-bars-ago): spot reacts to past perp
    corr_spot_on_perplag = roll_corr(scvd_dz, fcvd_dz.shift(L), 48)
    # corr(perp delta, spot delta k-bars-ago): perp reacts to past spot
    corr_perp_on_spotlag = roll_corr(fcvd_dz, scvd_dz.shift(L), 48)
    # >0 => spot is the LEADER (perp follows spot more than spot follows perp)
    cand["xc_spot_leadlag_3h"] = corr_perp_on_spotlag - corr_spot_on_perplag
    # contemporaneous coupling strength (decoupling = divergence regime)
    cand["xc_cvd_coupling_48h"] = roll_corr(scvd_dz, fcvd_dz, 48)

    # =====================================================================
    # (C) CROSS-channel sign-AGREEMENT streaks (multi-channel, not single)
    #     V7 has single-channel sign streaks (dfp_*_sign_streak).  Here we
    #     streak the AGREEMENT/DISAGREEMENT between channel pairs.
    # =====================================================================
    # spot vs perp CVD agreement: +1 same direction, -1 opposite
    sp_agree = np.sign(scvd_d) * np.sign(fcvd_d)
    cand["xc_spot_perp_agree_streak"] = signed_streak(
        pd.Series(sp_agree, index=idx), cap=20)
    # OI-build vs spot-CVD agreement (trapped positioning when they disagree)
    oi_scvd_agree = np.sign(oi_d) * np.sign(scvd_d)
    cand["xc_oi_spotcvd_agree_streak"] = signed_streak(
        pd.Series(oi_scvd_agree, index=idx), cap=20)
    # funding-sign vs perp-CVD agreement (do leveraged longs pay AND buy?)
    fund_fcvd_agree = np.sign(funding) * np.sign(fcvd_d)
    cand["xc_fund_perpcvd_agree_streak"] = signed_streak(
        pd.Series(fund_fcvd_agree, index=idx), cap=20)

    # =====================================================================
    # (D) Z-scored GAPS between channel pairs with NO existing divergence feat
    # =====================================================================
    # funding (leverage crowd cost) vs coinbase premium (spot allocator bid).
    # Both z-scored to comparable units; gap = leverage-vs-spot dislocation.
    cand["xc_fund_minus_premium_z"] = fund_z - cb_z
    # multi-window persistence of that dislocation (is it building?)
    cand["xc_fund_premium_persist_8h"] = (fund_z - cb_z).rolling(
        8, min_periods=4).mean()

    # ETF net flow (allocator) vs funding (leverage).  etf_z is daily, already
    # lagged in parquet -> we do NOT re-shift.  Resample funding to a daily
    # trailing mean so the gap is computed at the (lagged) daily resolution
    # the alt-data lives on, avoiding sub-daily peeking.
    fund_daily = funding.rolling(24, min_periods=12).mean()
    fund_daily_z = zscore(fund_daily, 24 * 14)   # ~14d daily-equivalent window
    cand["xc_etf_minus_funding_z"] = etf_z - fund_daily_z

    # OI-build vs spot-CVD: trapped positioning gap.  OI up + spot CVD down =
    # leverage building into no real spot demand (fragile).  Distinct from
    # oi_price_divergence (OI vs PRICE) -- this is OI vs SPOT FLOW.
    cand["xc_oi_minus_spotcvd_z"] = oi_dz - scvd_dz
    cand["xc_oi_spotcvd_gap_24h"] = (oi_dz - scvd_dz).rolling(
        24, min_periods=12).mean()

    # =====================================================================
    # (E) Trapped-positioning intensity & spot-conviction divergence
    # =====================================================================
    # cumulative spot-vs-perp CVD ratio momentum: are spot allocators
    # cumulatively building relative to perp over a week, and is that turning?
    cum_ratio = safe_norm(scvd_cum - fcvd_cum, scvd_cum.abs() + fcvd_cum.abs())
    cand["xc_cum_spotperp_ratio_z"] = zscore(cum_ratio, 168)
    # its 8h change = spot allocators gaining/losing ground vs perp
    cand["xc_cum_spotperp_ratio_chg8h"] = cum_ratio - cum_ratio.shift(8)

    # price-confirmed divergence: when spot leads (gap>0) AND price moving with
    # spot, that's real demand; built as gap * sign-agreement with return,
    # accumulated.  (dense; not a sparse indicator multiply.)
    gap_ret_align = np.sign(gap) * np.sign(logret)
    cand["xc_gap_return_align_streak"] = signed_streak(
        pd.Series(gap_ret_align, index=idx), cap=20)

    # ----------------------------------------------------------------------
    # leak-safety self-check: appending a future bar must not change the last
    # computed value of any candidate (pure trailing).  We verify by recomputing
    # the gap-based core on a truncated frame and comparing the overlap.
    # ----------------------------------------------------------------------
    _leak_ok = _leak_self_check(df)
    print(f"[leak-check] trailing-only recompute matches on overlap: {_leak_ok}")

    cand = cand.replace([np.inf, -np.inf], np.nan)
    print(f"[build] {cand.shape[1]} candidates built, index {cand.index[0]} .. {cand.index[-1]}")
    print("[coverage]")
    for c in cand.columns:
        s = cand[c]
        nz = float((s.abs() > 1e-12).mean())
        print(f"   {c:34s} nonnan={s.notna().mean():.2f} nonzero={nz:.2f} "
              f"std={np.nanstd(s):.3g}")

    # ----------------------------------------------------------------------
    # screen (single source of truth — do NOT reimplement IC/screen)
    # ----------------------------------------------------------------------
    res = screen_candidates(cand, existing_df=df)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    cols = ["feature", "cond_ic", "raw_ic", "frac_fold_consist", "n_folds",
            "nonzero_frac", "max_corr_existing", "SCREEN_PASS"]
    print("\n========== SCREEN RESULTS (ranked by |cond_ic|) ==========")
    print(res[cols].to_string(index=False))

    passers = res[res["SCREEN_PASS"]]["feature"].tolist()
    print(f"\n[verdict] {len(passers)} / {len(res)} pass full screen: {passers}")

    if passers:
        out = cand[passers].copy()
        RESULTS.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUT_PARQUET)
        print(f"[saved] {OUT_PARQUET}  shape={out.shape}")
    else:
        print("[saved] nothing — no candidate passed the screen")

    # machine-readable tail for the workflow harness
    print("\n===RESULT_JSON_BEGIN===")
    import json
    top5 = res.head(5)[["feature", "cond_ic", "frac_fold_consist",
                        "max_corr_existing", "nonzero_frac"]]
    payload = dict(
        n_built=int(cand.shape[1]),
        leak_ok=bool(_leak_ok),
        passers=[
            dict(name=r["feature"], cond_ic=float(r["cond_ic"]),
                 consist=float(r["frac_fold_consist"]),
                 max_corr=float(r["max_corr_existing"]),
                 nonzero=float(r["nonzero_frac"]))
            for _, r in res[res["SCREEN_PASS"]].iterrows()],
        top5=[dict(name=r["feature"], cond_ic=float(r["cond_ic"]),
                   consist=(None if pd.isna(r["frac_fold_consist"]) else float(r["frac_fold_consist"])),
                   max_corr=(None if pd.isna(r["max_corr_existing"]) else float(r["max_corr_existing"])),
                   nonzero=(None if pd.isna(r["nonzero_frac"]) else float(r["nonzero_frac"])))
              for _, r in top5.iterrows()],
        parquet=str(OUT_PARQUET) if passers else "",
    )
    print(json.dumps(payload, indent=2))
    print("===RESULT_JSON_END===")


def _leak_self_check(df: pd.DataFrame) -> bool:
    """Recompute one representative rolling feature on a frame truncated at
    bar T-200, then on the full frame, and assert the overlapping values are
    identical.  A forward/centered window would change historical values when
    future bars are added; trailing-only does not."""
    def core(frame: pd.DataFrame) -> pd.Series:
        scvd_dz = zscore(frame["cg_scvd_delta"].astype(float), 168)
        fcvd_dz = zscore(frame["cg_fcvd_delta"].astype(float), 168)
        gap = scvd_dz - fcvd_dz
        return gap.rolling(24, min_periods=12).sum()

    full = core(df)
    trunc = core(df.iloc[:-200])
    common = trunc.index
    a = full.reindex(common)
    b = trunc
    both = a.notna() & b.notna()
    if both.sum() < 100:
        return False
    return bool(np.allclose(a[both].values, b[both].values, atol=1e-9, rtol=1e-6))


if __name__ == "__main__":
    main()
