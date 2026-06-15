"""
Feature family: flow_accel — Flow acceleration & 2nd-order microstructure.

Goal: build trailing-only candidate features capturing 2nd-order order-flow
dynamics that the deployed 136-feature V7 may NOT already capture, then screen
for MARGINAL conditional information vs the clean V7 OOS residual.

Five genuinely distinct ideas (not transforms of one thing):
  1. CVD 2nd-difference  — acceleration of signed taker flow (jerk of buying pressure)
  2. Taker-imbalance regime persistence — dwell/run-length of imbalance sign, BEYOND imb_*
  3. trade_intensity * avg_trade_size — informed (few big) vs noise (many small) flow
  4. absorption-then-reversal — sequence: heavy flow absorbed at a level, then price reverts
  5. large-vs-small bar divergence persistence — smart-money (large) vs retail (small) split

HARD RULES (mistake.md):
  - Trailing-only: bar t uses ONLY bars <= t. State = cumulative-from-past.
    All rolling windows are backward; all diffs are x[t]-x[t-k]; NO shift(-k).
  - NO TA indicators (MACD/EMA/RSI/Bollinger/Stoch). Order-flow + raw price-behavior only.
  - NEVER feat*sparse_indicator with nonzero<30%. Regime conditioning via *(1-is_dead).
  - Daily alt-data already lagged; we build NONE of it here (pure intraday flow).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.featsearch_lib import load_features, screen_candidates  # noqa: E402

OUT_PARQUET = PROJECT_ROOT / "research" / "results" / "newfeat_flow_accel.parquet"


def _zscore(s: pd.Series, win: int) -> pd.Series:
    """Trailing rolling z-score (backward window, uses bars <= t only)."""
    m = s.rolling(win, min_periods=max(8, win // 3)).mean()
    sd = s.rolling(win, min_periods=max(8, win // 3)).std()
    return (s - m) / sd.replace(0.0, np.nan)


def _roll_rank(s: pd.Series, win: int) -> pd.Series:
    """Trailing percentile rank of the current value within the past `win` bars."""
    return s.rolling(win, min_periods=max(8, win // 3)).apply(
        lambda a: (a[:-1] < a[-1]).mean() if len(a) > 1 else np.nan, raw=True)


def build_candidates(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    cand = pd.DataFrame(index=idx)

    # ---- Core trailing flow primitives (all from raw OHLCV + taker columns) ----
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)
    tbv = df["taker_buy_vol"].astype(float)             # taker BUY base volume
    tsv = (vol - tbv).clip(lower=0.0)                    # taker SELL base volume
    tcount = df["trade_count"].astype(float)
    avg_sz = df["avg_trade_size"].astype(float)          # base/trade
    intensity = df["trade_intensity"].astype(float)
    ret = df["log_return"].astype(float)
    rng = (high - low).clip(lower=1e-9)

    # Per-bar signed taker delta (base units) and its CVD (cumulative-from-past)
    taker_delta = (tbv - tsv)                            # >0 net buying
    taker_imb = taker_delta / vol.replace(0.0, np.nan)   # in [-1,1], like imb_1b
    cvd = taker_delta.cumsum()                           # trailing cumulative, ok

    # ===========================================================================
    # IDEA 1 — CVD 2nd-difference (acceleration / jerk of signed flow)
    # imb_slope_4b exists; the *acceleration* (2nd diff) and its z are new angles.
    # ===========================================================================
    cvd_vel_4b = cvd - cvd.shift(4)                       # 1st diff over 4 bars = net delta
    cvd_vel_4b_prev = cvd.shift(4) - cvd.shift(8)
    cvd_accel_4b = cvd_vel_4b - cvd_vel_4b_prev           # 2nd diff: change in net-delta rate
    # normalise by trailing flow scale so it is regime-comparable
    flow_scale = vol.rolling(24, min_periods=8).mean().replace(0.0, np.nan)
    cand["fa_cvd_accel_4b"] = cvd_accel_4b / (4.0 * flow_scale)
    cand["fa_cvd_accel_z48"] = _zscore(cvd_accel_4b, 48)

    # Jerk: 3rd-order — is the acceleration itself accelerating? (1-bar deltas)
    delta_1b = taker_delta
    cand["fa_cvd_jerk_z48"] = _zscore(delta_1b.diff().diff(), 48)

    # ===========================================================================
    # IDEA 2 — Taker-imbalance regime PERSISTENCE (beyond imb_sign_persistence)
    # dwell time of the current imbalance sign + magnitude-weighted run strength.
    # ===========================================================================
    imb_sign = np.sign(taker_imb).fillna(0.0)
    # run length: bars since the sign last flipped (trailing, vectorised)
    grp = (imb_sign != imb_sign.shift(1)).cumsum()
    run_len = imb_sign.groupby(grp).cumcount() + 1
    # signed dwell, capped & shrunk so it isn't a sparse step function
    signed_dwell = (imb_sign * np.log1p(run_len.astype(float)))
    cand["fa_imb_signed_dwell"] = signed_dwell
    # persistence STRENGTH: avg |imbalance| over the current run (how forceful the run is).
    # expanding-mean WITHIN each sign-run = cumsum/count reset at every flip (trailing).
    abs_imb = taker_imb.abs()
    run_csum = abs_imb.groupby(grp).cumsum()
    run_cnt = abs_imb.groupby(grp).cumcount() + 1
    run_abs_mean = run_csum / run_cnt
    cand["fa_imb_run_force"] = (imb_sign * run_abs_mean).reindex(idx)
    # regime entropy: how choppy the sign has been over 12 bars (low=trending flow)
    flip_rate_12 = (imb_sign.diff().abs() > 0).rolling(12, min_periods=6).mean()
    cand["fa_imb_flip_rate_12"] = flip_rate_12

    # ===========================================================================
    # IDEA 3 — trade_intensity * avg_trade_size  (informed big vs noise small)
    # A few large trades (informed) vs many tiny ones (noise). Sign by flow dir.
    # ===========================================================================
    avg_sz_z = _zscore(avg_sz, 48)
    intensity_z = _zscore(intensity, 48)
    # informed-flow proxy: big-trade regime AND directional taker pressure
    cand["fa_informed_flow"] = avg_sz_z * taker_imb
    # "whale vs minnow" texture: big trades but LOW count = concentrated execution
    cand["fa_concentration"] = avg_sz_z - intensity_z
    # directional concentration: concentrated execution pushing one way
    cand["fa_dir_concentration"] = (avg_sz_z - intensity_z) * imb_sign

    # ===========================================================================
    # IDEA 4 — absorption-then-reversal sequence
    # Heavy one-sided flow that FAILS to move price (absorption), then price
    # reverts. Trailing: absorption measured at t-k, reversal realised by t.
    # ===========================================================================
    # absorption_t = strong taker imbalance but small same-direction return
    flow_push = taker_imb                                  # signed flow
    price_resp = ret / rng.div(close)                      # return scaled by bar range frac
    # absorption: big |flow| with price NOT following (resp small / opposite)
    absorb_signed = flow_push * (1.0 - np.tanh((price_resp * np.sign(flow_push)).clip(lower=0)))
    # reversal payoff: did the 3 bars AFTER a past absorption revert?  Use PAST
    # absorption (shift +3) correlated with realised 3-bar return ending now.
    ret_3b = close.pct_change(3)
    past_absorb = absorb_signed.shift(3)                  # absorption that happened 3 bars ago
    # signed so that "absorbed buying 3 bars ago -> now selling off" is positive structure
    cand["fa_absorb_then_rev"] = (-np.sign(past_absorb) * ret_3b
                                  * absorb_signed.abs().shift(3).clip(0, 1))
    # contemporaneous absorption intensity (z) — pressure without follow-through
    cand["fa_absorb_intensity_z"] = _zscore(absorb_signed.abs(), 48) * imb_sign

    # ===========================================================================
    # IDEA 5 — large-vs-small bar divergence PERSISTENCE
    # large_bar_delta / small_bar_delta exist as point values; the PERSISTENT
    # divergence (smart vs retail disagreeing for several bars) is the new angle.
    # ===========================================================================
    lbd = df["large_bar_delta"].astype(float)
    sbd = df["small_bar_delta"].astype(float)
    # normalise each by trailing scale, then their disagreement
    lbd_n = lbd / lbd.abs().rolling(48, min_periods=12).mean().replace(0.0, np.nan)
    sbd_n = sbd / sbd.abs().rolling(48, min_periods=12).mean().replace(0.0, np.nan)
    div = lbd_n - sbd_n                                    # large minus small (signed)
    # persistence: trailing mean of the disagreement over 8 bars (sustained split)
    cand["fa_lgsm_div_persist_8b"] = div.rolling(8, min_periods=4).mean()
    # smart-money lead: large-bar delta z (informed) standing apart from price
    cand["fa_smartmoney_lead"] = _zscore(lbd, 48) - _roll_rank(ret, 48).mul(2).sub(1)

    # ---- coverage / hygiene ----
    cand = cand.replace([np.inf, -np.inf], np.nan)
    return cand


def main() -> None:
    df = load_features()
    print(f"[load] features frame: {df.shape}")

    cand = build_candidates(df)
    print(f"[build] {cand.shape[1]} candidate features built")

    # ---- leak-safety self-check: trailing-only invariance ----
    # TWO independent tests; both must pass for a clean bill of health.
    #
    # (A) TAIL TRUNCATION: drop last 50 bars and rebuild. Overlapping bars must be
    #     identical — catches forward/centered windows that reach the tail edge.
    df_trunc = df.iloc[:-50].copy()
    cand_trunc = build_candidates(df_trunc)
    check_idx = cand_trunc.index
    a, b = cand.reindex(check_idx), cand_trunc.reindex(check_idx)
    max_abs_diff_A, worst_A = 0.0, None
    for c in cand.columns:
        both = a[c].notna() & b[c].notna()
        if both.sum() == 0:
            continue
        d = (a[c][both] - b[c][both]).abs().max()
        if pd.notna(d) and d > max_abs_diff_A:
            max_abs_diff_A, worst_A = float(d), c

    # (B) INTERIOR FORWARD-PERTURBATION: corrupt the raw inputs of one future bar
    #     (position p+30) and rebuild; an EARLIER bar's features (positions <= p)
    #     must be byte-identical. Any forward-peek (e.g. an interior shift(-k))
    #     would propagate the corruption backward and be caught here — this is the
    #     definitive trailing-only test (the truncation test alone can miss it).
    p = len(df) - 60
    df_pert = df.copy()
    raw_in = ["open", "high", "low", "close", "volume", "taker_buy_vol",
              "trade_count", "avg_trade_size", "trade_intensity", "log_return",
              "large_bar_delta", "small_bar_delta"]
    for col in raw_in:
        df_pert.iloc[p + 30, df_pert.columns.get_loc(col)] *= 3.0  # gross corruption
    cand_pert = build_candidates(df_pert)
    past_idx = df.index[:p + 1]            # bars at/ before the past anchor
    ap, bp = cand.reindex(past_idx), cand_pert.reindex(past_idx)
    max_abs_diff_B, worst_B = 0.0, None
    for c in cand.columns:
        both = ap[c].notna() & bp[c].notna()
        if both.sum() == 0:
            continue
        d = (ap[c][both] - bp[c][both]).abs().max()
        if pd.notna(d) and d > max_abs_diff_B:
            max_abs_diff_B, worst_B = float(d), c

    leak_ok = (max_abs_diff_A < 1e-9) and (max_abs_diff_B < 1e-9)
    if leak_ok:
        leak_note = (f"trailing-only CONFIRMED — tail-truncation max|Δ|="
                     f"{max_abs_diff_A:.1e}, interior future-perturbation max|Δ|="
                     f"{max_abs_diff_B:.1e} (future bar's corruption did NOT leak "
                     f"back into past features)")
    else:
        leak_note = (f"LEAK SUSPECT — truncation Δ={max_abs_diff_A:.2e} on {worst_A}; "
                     f"interior-perturb Δ={max_abs_diff_B:.2e} on {worst_B}")
    print(f"[leak-check] {leak_note}")

    # ---- coverage report ----
    print("\n[coverage] nonzero / non-nan fraction per candidate:")
    for c in cand.columns:
        s = cand[c]
        print(f"  {c:28s} nz={float((s.abs()>1e-12).mean()):.3f} "
              f"nonnan={float(s.notna().mean()):.3f}")

    # ---- SCREEN ----
    res = screen_candidates(cand, existing_df=df)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    show = ["feature", "cond_ic", "raw_ic", "frac_fold_consist", "n_folds",
            "nonzero_frac", "max_corr_existing", "SCREEN_PASS"]
    print("\n[screen] ranked by |cond_ic|:")
    print(res[show].to_string(index=False))

    passers = res[res["SCREEN_PASS"] == True]  # noqa: E712
    print(f"\n[result] {len(passers)} / {len(res)} candidates PASS the full screen")

    if len(passers) > 0:
        keep = list(passers["feature"])
        OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        cand[keep].to_parquet(OUT_PARQUET)
        print(f"[save] wrote {len(keep)} passing cols -> {OUT_PARQUET}")
    else:
        print("[save] no passers — parquet NOT written (family FAILS the screen)")


if __name__ == "__main__":
    main()
