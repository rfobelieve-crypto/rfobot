"""
Feature family: liq_aftermath  (Liquidation-cascade AFTERMATH state)
2026-06-14 feature search.

Idea: a liquidation cascade (forced de-leveraging) is an event; the *information*
is in what the market does in the bars AFTER it. We encode purely-trailing state:
  - bars-since-last cascade / surge (recency of the last forced-deleveraging event)
  - the SIGN of the last cascade (long-liq blow-off vs short-liq squeeze) carried
    forward as state
  - post-cascade trailing price/flow behavior (return & absorption accumulated only
    over bars that come AFTER the cascade) -> "did the market reclaim / follow through"
  - liq-imbalance regime transitions (flip from long-dominant to short-dominant liq)
  - cascade magnitude interacted with SUBSEQUENT absorption_net (trailing)

HARD RULES honored:
  1. Trailing-only: every feature at bar t uses bars <= t. Bars-since counters,
     cumulative-since-event sums, .shift(+1) regime compares. No forward windows.
  2. No TA indicators. Only order-flow (liq, absorption, taker) + raw price
     behavior (returns, vol, wicks).
  3. No feat*sparse_indicator with nonzero<0.30. cg_liq_cascade>0 fires 27.5% of
     bars -> we DO NOT multiply a feature by that raw flag. Instead we build dense
     state (bars-since, decaying recency, carried sign) which is defined on ~100%
     of bars. Where we condition we use a DENSE mask or (1 - dead) form.
  4. Daily alt-data untouched.

Screen via research.featsearch_lib.screen_candidates (identical methodology across
families). Save only passers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.featsearch_lib import (  # noqa: E402
    load_features,
    screen_candidates,
)

RESULTS = PROJECT_ROOT / "research" / "results"
OUT_PARQUET = RESULTS / "newfeat_liq_aftermath.parquet"


# ----------------------------------------------------------------------------
# trailing-only helpers
# ----------------------------------------------------------------------------
def bars_since_true(mask: pd.Series) -> pd.Series:
    """Bars since `mask` was last True, counted using bars <= t only.
    At the event bar value = 0; one bar later = 1; etc. Before the first event
    the count keeps rising from the start (NaN-free, monotone). Trailing-only."""
    m = mask.fillna(False).to_numpy().astype(bool)
    out = np.empty(len(m), dtype=float)
    last = -1
    for i in range(len(m)):
        if m[i]:
            last = i
        out[i] = (i - last) if last >= 0 else np.nan
    return pd.Series(out, index=mask.index)


def carry_forward_at_event(value: pd.Series, mask: pd.Series) -> pd.Series:
    """At each bar, the value of `value` sampled at the LAST bar where mask was
    True (<= t). i.e. carry the event's attribute forward as state. Trailing."""
    v = value.to_numpy(dtype=float)
    m = mask.fillna(False).to_numpy().astype(bool)
    out = np.full(len(m), np.nan, dtype=float)
    cur = np.nan
    for i in range(len(m)):
        if m[i]:
            cur = v[i]
        out[i] = cur
    return pd.Series(out, index=value.index)


def cumsum_since_event(value: pd.Series, mask: pd.Series, include_event: bool = False) -> pd.Series:
    """Cumulative sum of `value` accumulated only over bars strictly AFTER (or
    including) the last event bar, up to and including t. Resets at each event.
    Trailing-only (only uses bars <= t)."""
    v = value.fillna(0.0).to_numpy(dtype=float)
    m = mask.fillna(False).to_numpy().astype(bool)
    out = np.full(len(m), np.nan, dtype=float)
    acc = np.nan
    seen = False
    for i in range(len(m)):
        if m[i]:
            seen = True
            acc = v[i] if include_event else 0.0
        elif seen:
            acc = (0.0 if np.isnan(acc) else acc) + v[i]
        out[i] = acc if seen else np.nan
    return pd.Series(out, index=value.index)


def main() -> None:
    df = load_features()
    idx = df.index
    n = len(df)
    print(f"[load] features {df.shape}")

    # ---- raw inputs (all already trailing in the production frame) ----
    cascade = pd.to_numeric(df["cg_liq_cascade"], errors="coerce").fillna(0.0)   # int 0..9, fires 27.5%
    surge = pd.to_numeric(df["cg_liq_surge"], errors="coerce")                    # continuous surge ratio
    surge_z = pd.to_numeric(df["cg_liq_surge_zscore"], errors="coerce")
    liq_imb = pd.to_numeric(df["cg_liq_imbalance"], errors="coerce")             # +1 long-liq dominant
    liq_imb_z = pd.to_numeric(df["cg_liq_imbalance_zscore"], errors="coerce")
    liq_total = pd.to_numeric(df["cg_liq_total"], errors="coerce")
    liq_long = pd.to_numeric(df["cg_liq_long"], errors="coerce")
    liq_short = pd.to_numeric(df["cg_liq_short"], errors="coerce")
    absorb_net = pd.to_numeric(df["absorption_net"], errors="coerce")
    log_ret = pd.to_numeric(df["log_return"], errors="coerce")
    rvol = pd.to_numeric(df["realized_vol_20b"], errors="coerce").replace(0, np.nan)
    taker_delta = pd.to_numeric(df["taker_delta_ratio"], errors="coerce")

    # event definitions (trailing). cascade>0 fires 27.5% -> a real, not-too-sparse
    # event. surge_z high tail = strong forced-deleverage burst.
    cascade_event = cascade > 0
    big_surge_event = surge_z > 1.0      # ~top-of-tail surge burst

    cand = pd.DataFrame(index=idx)

    # 1) RECENCY: exponentially-decaying "freshness" of the last cascade. Dense
    #    (defined every bar once a cascade has happened). High right after a
    #    cascade, decays toward 0. Pure state, not the raw sparse flag.
    bsc = bars_since_true(cascade_event)
    cand["la_cascade_recency"] = np.exp(-bsc / 6.0)

    # 2) bars since the last BIG surge burst (forced-deleverage shock recency),
    #    log-compressed so it's a smooth dense state.
    bss = bars_since_true(big_surge_event)
    cand["la_surge_recency_log"] = -np.log1p(bss)

    # 3) SIGNED last-cascade direction carried forward, weighted by its surge
    #    magnitude AT the event: + = last shock was long-liq (longs blown out,
    #    contrarian-up pressure), - = short-liq. Carried as state until next event.
    cascade_dir_mag = np.sign(liq_imb) * surge_z.clip(lower=0)
    cand["la_last_cascade_signed_mag"] = carry_forward_at_event(cascade_dir_mag, cascade_event)

    # 4) POST-CASCADE cumulative return: how much price has moved since the last
    #    cascade bar (reclaim / follow-through). Reset each cascade. Trailing.
    cand["la_post_cascade_cumret"] = cumsum_since_event(log_ret, cascade_event)

    # 5) POST-CASCADE cumulative return SIGNED by cascade direction: did price move
    #    WITH the contrarian thesis (long-liq -> price recovers up)? Positive =
    #    mean-reversion playing out; negative = continuation of the liquidation.
    last_dir = carry_forward_at_event(np.sign(liq_imb), cascade_event)
    cand["la_post_cascade_reclaim"] = cand["la_post_cascade_cumret"] * last_dir

    # 6) POST-CASCADE absorption accumulation: cumulative absorption_net since the
    #    last cascade, normalized by trailing |absorption| scale -> who is absorbing
    #    the aftermath. Magnitude(cascade) x subsequent absorption, all trailing.
    absorb_scale = absorb_net.abs().rolling(48, min_periods=12).median().replace(0, np.nan)
    post_absorb = cumsum_since_event(absorb_net, cascade_event)
    cand["la_post_cascade_absorb"] = (post_absorb / (absorb_scale * 4.0)).clip(-8, 8)

    # 7) cascade MAGNITUDE (size of last shock) x subsequent absorption sign:
    #    big last cascade absorbed in the contrarian direction = stronger setup.
    last_surge_mag = carry_forward_at_event(surge_z.clip(lower=0), cascade_event)
    post_absorb_sign = np.sign(post_absorb)
    cand["la_cascade_mag_x_absorb"] = (last_surge_mag * post_absorb_sign).fillna(0.0)

    # 8) LIQ-IMBALANCE regime transition: did the dominant liquidation side FLIP
    #    this bar vs last bar (long-liq regime -> short-liq regime)? Signed by the
    #    NEW regime. Trailing (uses t and t-1 only). Dense-ish (sign of imbalance).
    imb_sign = np.sign(liq_imb)
    imb_sign_prev = imb_sign.shift(1)
    flip = (imb_sign != imb_sign_prev) & imb_sign.notna() & imb_sign_prev.notna()
    cand["la_imb_regime_flip"] = (flip.astype(float) * imb_sign).fillna(0.0)

    # 9) bars since the liq-imbalance regime last flipped (persistence of current
    #    forced-deleverage regime), decayed. Dense.
    bars_since_flip = bars_since_true(flip)
    cand["la_imb_regime_age"] = np.exp(-bars_since_flip / 8.0)

    # 10) post-cascade VOLATILITY expansion: trailing realized-vol now vs the vol
    #     at the last cascade bar -> is the aftermath calming or still violent.
    rvol_at_event = carry_forward_at_event(rvol, cascade_event)
    cand["la_post_cascade_vol_ratio"] = np.log((rvol / rvol_at_event).clip(0.1, 10))

    # 11) ASYMMETRIC post-cascade taker pressure: cumulative taker_delta since the
    #     last cascade, signed by cascade direction -> is taker flow confirming the
    #     contrarian reclaim or fading it. Trailing.
    post_taker = cumsum_since_event(taker_delta, cascade_event)
    cand["la_post_cascade_taker_confirm"] = (post_taker * last_dir).clip(-20, 20)

    # 12) long-vs-short liq AFTERMATH skew: trailing ratio of cumulative long-liq
    #     to total-liq over the bars since last cascade -> which side kept bleeding
    #     in the aftermath (continuation pressure). Dense once a cascade seen.
    post_long = cumsum_since_event(liq_long, cascade_event)
    post_short = cumsum_since_event(liq_short, cascade_event)
    denom = (post_long + post_short).replace(0, np.nan)
    cand["la_aftermath_longliq_share"] = (post_long / denom) - 0.5

    # 13) "double-tap" cascade clustering: count of cascade bars in the trailing
    #     8 bars (forced-deleverage clustering = capitulation phase). Dense rolling
    #     count, trailing window only.
    cand["la_cascade_cluster_8b"] = cascade_event.astype(float).rolling(8, min_periods=1).sum()

    # 14) surge-vs-cascade-recency interaction (1-dead form): current surge_z gated
    #     by how RECENTLY a cascade fired -> a fresh surge right after a cascade is
    #     a re-acceleration. Uses dense recency (not the sparse raw flag).
    cand["la_surge_x_fresh_cascade"] = surge_z * cand["la_cascade_recency"]

    print(f"[build] {cand.shape[1]} candidates")

    # ---- trailing-only self-checks (cheap leak guards) -----------------------
    # (a) no candidate may depend on a future bar: re-build on a truncated frame
    #     and assert the prefix is unchanged for a few representative dense cols.
    print("[leak-check] verifying prefix-stability under truncation ...")
    check_cols = [
        "la_cascade_recency", "la_post_cascade_cumret", "la_post_cascade_reclaim",
        "la_imb_regime_flip", "la_aftermath_longliq_share", "la_cascade_cluster_8b",
    ]
    cut = int(n * 0.7)
    leak_bad = []
    # rebuild the truncated versions of the path-dependent cols
    casc_t = cascade_event.iloc[:cut]
    rec_t = np.exp(-bars_since_true(casc_t) / 6.0)
    cr_t = cumsum_since_event(log_ret.iloc[:cut], casc_t)
    ld_t = carry_forward_at_event(np.sign(liq_imb.iloc[:cut]), casc_t)
    rcl_t = cr_t * ld_t
    isgn = np.sign(liq_imb.iloc[:cut]); isgnp = isgn.shift(1)
    flip_t = (isgn != isgnp) & isgn.notna() & isgnp.notna()
    flipf_t = (flip_t.astype(float) * isgn).fillna(0.0)
    pl_t = cumsum_since_event(liq_long.iloc[:cut], casc_t)
    ps_t = cumsum_since_event(liq_short.iloc[:cut], casc_t)
    den_t = (pl_t + ps_t).replace(0, np.nan)
    share_t = (pl_t / den_t) - 0.5
    clus_t = casc_t.astype(float).rolling(8, min_periods=1).sum()
    trunc_map = {
        "la_cascade_recency": rec_t,
        "la_post_cascade_cumret": cr_t,
        "la_post_cascade_reclaim": rcl_t,
        "la_imb_regime_flip": flipf_t,
        "la_aftermath_longliq_share": share_t,
        "la_cascade_cluster_8b": clus_t,
    }
    for c in check_cols:
        full_prefix = cand[c].iloc[:cut]
        tr = trunc_map[c]
        a = pd.to_numeric(full_prefix, errors="coerce").to_numpy()
        b = pd.to_numeric(tr, errors="coerce").to_numpy()
        both = np.isfinite(a) & np.isfinite(b)
        if both.sum() == 0:
            continue
        max_diff = np.max(np.abs(a[both] - b[both]))
        if max_diff > 1e-9:
            leak_bad.append((c, float(max_diff)))
    if leak_bad:
        print("  !! LEAK SUSPECT (prefix changed under truncation):", leak_bad)
    else:
        print("  OK: all checked cols are prefix-stable (trailing-only confirmed)")

    # coverage report
    print("[coverage] nonzero / nan per candidate:")
    for c in cand.columns:
        s = pd.to_numeric(cand[c], errors="coerce")
        print(f"  {c:34s} nan={s.isna().mean():.3f} nonzero={(s.abs()>1e-12).mean():.3f}")

    # ---- screen --------------------------------------------------------------
    print("[screen] running screen_candidates vs clean V7 OOS residual ...")
    res = screen_candidates(cand, existing_df=df)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    show = res[["feature", "cond_ic", "raw_ic", "frac_fold_consist", "n_folds",
                "nonzero_frac", "max_corr_existing", "SCREEN_PASS", "note"]]
    print(show.to_string(index=False))

    passers = res[res["SCREEN_PASS"]]["feature"].tolist()
    print(f"\n[result] {len(passers)} / {len(res)} candidates PASS the screen")

    parquet_path = ""
    if passers:
        RESULTS.mkdir(parents=True, exist_ok=True)
        cand[passers].to_parquet(OUT_PARQUET)
        parquet_path = str(OUT_PARQUET)
        print(f"[save] wrote {len(passers)} passing cols -> {parquet_path}")
    else:
        print("[save] no passers -> no parquet written")

    # machine-readable summary for the workflow
    top5 = res.head(5)[["feature", "cond_ic", "frac_fold_consist",
                        "max_corr_existing", "nonzero_frac"]]
    print("\n===TOP5===")
    print(top5.to_string(index=False))
    print("===END===")


if __name__ == "__main__":
    main()
