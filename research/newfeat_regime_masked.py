"""
Feature family: regime_masked  (2026-06-14 feature search)

Idea (mistake.md 2026-04-13 "sparse indicator is degenerate"):
    NEVER  feat * is_sparse_indicator   -- collapses to the sparsity pattern.
    INSTEAD mask only where a base flow feature plausibly DIES:
        feat * (1 - is_dead_regime)
    and add the regime indicators THEMSELVES as features so XGB learns the
    conditional split (point 2 of that mistake entry).

Hypothesis: funding / OI / CVD / taker-flow signals carry directional
information in "normal" conditions but DECOUPLE from 4h direction when:
  - implied vol is elevated (options/vega-driven tape, flow stops leading), or
  - sentiment is at an extreme-fear washout (forced de-risking dominates flow), or
  - realized vol regime is high (mechanical liquidation noise swamps order flow).
By zeroing those base features OUTSIDE the regime where they work (keeping
>70% of bars non-zero), the masked feature may carry marginal info the V7
ensemble's global splits did not isolate.

HARD RULES respected:
  1. Trailing-only. All masks use already-trailing levels (dvol z-score 72h,
     fear_greed level which is pre-lagged, trailing-bull/bear flags) or an
     EXPANDING (past-only) median for the realized-vol threshold. No forward
     windows, no .shift(-k).
  2. No TA indicators -- only order-flow + raw price-behavior stats.
  3. Masks are feat*(1 - is_dead_regime), NOT feat*sparse_indicator. Every
     mask verified to keep >70% non-zero (asserted at build time).
  4. Daily alt-data (fear_greed_*) already lagged in parquet -- used as-is,
     no re-shift, no same-day peek built.

Screening: research.featsearch_lib.screen_candidates (conditional IC vs the
CLEAN V7 OOS residual). We do NOT write our own IC/screen code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.featsearch_lib import load_features, screen_candidates  # noqa: E402

OUT_PARQUET = PROJECT_ROOT / "research" / "results" / "newfeat_regime_masked.parquet"

NONZERO_MASK_MIN = 0.70  # rule: each mask must keep >70% of bars non-zero


def _assert_trailing_threshold_ok():
    """Document why each mask is trailing-only (no run-time effect, just a note)."""
    return True


def build_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Build 8-15 genuinely distinct regime-masked candidates + regime indicators."""
    idx = df.index
    cand = pd.DataFrame(index=idx)

    # ------------------------------------------------------------------ #
    # 1. DEAD-REGIME BOOLEANS  (is_dead = 1 where the base flow plausibly  #
    #    stops leading direction).  All trailing-only.                     #
    # ------------------------------------------------------------------ #
    # (a) High implied vol: dvol_zscore_72h is an already-trailing z-score.
    #     Absolute threshold +1.0 -> ~22% dead, ~78% alive. Trailing-safe.
    dvz = pd.to_numeric(df["dvol_zscore_72h"], errors="coerce")
    is_high_dvol = (dvz > 1.0).astype(float).fillna(0.0)

    # (b) Extreme-fear washout: fear_greed_value is pre-lagged (rule 4).
    #     Absolute level < 10 -> ~14% dead. Trailing-safe (level, not stat).
    fg = pd.to_numeric(df["fear_greed_value"], errors="coerce")
    is_extreme_fear = (fg < 10).astype(float).fillna(0.0)

    # (c) High realized-vol regime: realized_vol_20b is trailing, but a FIXED
    #     quantile cut over the whole sample would peek. Use an EXPANDING
    #     (past-only) 70th-pct proxy via expanding median * 1.4 so the
    #     threshold itself only uses bars <= t. min_periods=200 for stability.
    rv = pd.to_numeric(df["realized_vol_20b"], errors="coerce")
    rv_thr = rv.expanding(min_periods=200).median() * 1.4  # trailing threshold
    is_high_rv = (rv > rv_thr).astype(float)
    is_high_rv = is_high_rv.fillna(0.0)  # warmup bars -> not flagged dead

    # (d) Trending regimes already trailing flags in the frame.
    is_bull = pd.to_numeric(df["is_trending_bull"], errors="coerce").fillna(0.0)
    is_bear = pd.to_numeric(df["is_trending_bear"], errors="coerce").fillna(0.0)

    masks = {
        "high_dvol": is_high_dvol,
        "extreme_fear": is_extreme_fear,
        "high_rv": is_high_rv,
        "bull": is_bull,
        "bear": is_bear,
    }
    print("\n--- dead-regime mask dead_frac (1-this = retained non-zero) ---")
    for nm, m in masks.items():
        dead = float(m.mean())
        print(f"  {nm:14s} dead_frac={dead:.3f}  retained~={1-dead:.3f}")

    # ------------------------------------------------------------------ #
    # 2. BASE FLOW FEATURES that plausibly die in a regime.               #
    #    Pick genuinely distinct channels: funding, OI, spot/futures CVD, #
    #    taker delta, perp-vs-spot divergence, liq imbalance.             #
    # ------------------------------------------------------------------ #
    def num(c):
        return pd.to_numeric(df[c], errors="coerce")

    base = {
        "funding_z": num("cg_funding_close_zscore"),     # funding pressure
        "oi_chg4h": num("cg_oi_close_pctchg_4h"),        # OI build/unwind
        "fcvd_delta": num("cg_fcvd_delta"),              # futures CVD delta
        "scvd_delta": num("cg_scvd_delta"),              # spot CVD delta
        "taker_z": num("cg_taker_delta_zscore"),         # perp taker imbalance
        "sf_cvd_div": num("cg_spot_futures_cvd_divergence"),  # spot vs futures
        "liq_imb": num("cg_liq_imbalance"),              # liquidation imbalance
        "cvd_persist": num("cvd_persistence_12h"),       # CVD trend persistence
    }

    # ------------------------------------------------------------------ #
    # 3. MASKED CANDIDATES  =  base * (1 - is_dead_regime)                #
    #    Each pairing chosen so the hypothesis is economically motivated, #
    #    not 8 transforms of one base.  Verify >70% non-zero per mask.    #
    # ------------------------------------------------------------------ #
    pairings = [
        # funding leads direction EXCEPT in extreme-fear washout (forced flows)
        ("funding_z_alive_norfear", "funding_z", "extreme_fear"),
        # OI build informative EXCEPT when implied vol elevated (vega tape)
        ("oi_chg4h_alive_lowdvol", "oi_chg4h", "high_dvol"),
        # futures CVD leads EXCEPT in high realized-vol liquidation noise
        ("fcvd_delta_alive_lowrv", "fcvd_delta", "high_rv"),
        # spot CVD leads EXCEPT in high implied vol (decoupling)
        ("scvd_delta_alive_lowdvol", "scvd_delta", "high_dvol"),
        # perp taker imbalance informative EXCEPT in extreme-fear washout
        ("taker_z_alive_norfear", "taker_z", "extreme_fear"),
        # spot/futures CVD divergence informative EXCEPT in high realized vol
        ("sf_cvd_div_alive_lowrv", "sf_cvd_div", "high_rv"),
        # liq imbalance informative EXCEPT during a confirmed bear leg
        ("liq_imb_alive_nobear", "liq_imb", "bear"),
        # CVD persistence informative EXCEPT during a confirmed bull leg
        ("cvd_persist_alive_nobull", "cvd_persist", "bull"),
        # cross-check: taker imbalance EXCEPT high implied vol (distinct mask)
        ("taker_z_alive_lowdvol", "taker_z", "high_dvol"),
        # cross-check: funding EXCEPT high realized vol (distinct mask)
        ("funding_z_alive_lowrv", "funding_z", "high_rv"),
    ]

    print("\n--- masked candidate non-zero fractions (must be >0.70) ---")
    for out_name, base_key, mask_key in pairings:
        alive = (1.0 - masks[mask_key])  # 1 where flow is alive
        feat = base[base_key] * alive
        nz = float((feat.abs() > 1e-12).mean())
        flag = "OK" if nz > NONZERO_MASK_MIN else "!! LOW"
        print(f"  {out_name:28s} nonzero={nz:.3f} {flag}")
        if nz <= NONZERO_MASK_MIN:
            # do not ship a mask that kills >30% of bars (rule violation)
            print(f"      SKIP {out_name}: non-zero {nz:.3f} <= {NONZERO_MASK_MIN}")
            continue
        cand[out_name] = feat.values

    # ------------------------------------------------------------------ #
    # 4. REGIME INDICATORS THEMSELVES as features (mistake.md 2026-04-13   #
    #    point 2: let XGB learn the conditional split directly).  These    #
    #    are continuous regime DESCRIPTORS, distinct from the binary masks #
    #    and from existing V7 cols -- screen will catch any redundancy.    #
    # ------------------------------------------------------------------ #
    # implied-vol vs realized-vol regime gap (vol risk premium proxy),
    # trailing: dvol_close is trailing, realized_vol_20b is trailing.
    dvol = num("dvol_close")
    rv_ann = rv  # already trailing realized vol per 20-bar window
    cand["regime_ivrv_gap"] = (dvol / 100.0 - rv_ann * 100.0).values  # scaled gap

    # dvol z-score itself (continuous regime descriptor, not the binary)
    cand["regime_dvol_z"] = dvz.values

    # fear-greed level as continuous regime descriptor (pre-lagged)
    cand["regime_fg_level"] = fg.values

    # trailing realized-vol "heat": rv relative to its expanding median
    cand["regime_rv_heat"] = (rv / (rv.expanding(min_periods=200).median())).values

    print(f"\nBuilt {cand.shape[1]} candidate columns.")
    return cand


def leak_safety_selfcheck(df: pd.DataFrame, cand: pd.DataFrame) -> str:
    """Verify trailing-only by recomputing the last value on a TRUNCATED frame:
    a trailing feature's value at bar t must be IDENTICAL whether or not bars
    after t exist.  We truncate to the last 500 bars' midpoint and compare."""
    # pick a probe bar well inside the series (so expanding windows are warm)
    probe_pos = len(df) - 50
    probe_idx = df.index[probe_pos]
    full = build_candidates(df)
    trunc = build_candidates(df.iloc[: probe_pos + 1])  # only bars <= probe
    cols = [c for c in full.columns if c in trunc.columns]
    bad = []
    for c in cols:
        a = full.loc[probe_idx, c]
        b = trunc.loc[probe_idx, c]
        if pd.isna(a) and pd.isna(b):
            continue
        if not np.isclose(float(a), float(b), rtol=1e-9, atol=1e-12, equal_nan=True):
            bad.append((c, float(a), float(b)))
    if bad:
        return f"LEAK DETECTED at {probe_idx}: " + "; ".join(
            f"{c}: full={a:.6g} trunc={b:.6g}" for c, a, b in bad)
    return f"PASS: {len(cols)} features identical full vs truncated@{probe_idx} (trailing-only confirmed)"


def main():
    df = load_features()
    print("loaded features:", df.shape, "index", df.index.min(), "->", df.index.max())

    # show the raw inputs this family draws on (per task instruction)
    raw_inputs = ["dvol_zscore_72h", "dvol_close", "fear_greed_value",
                  "realized_vol_20b", "is_trending_bull", "is_trending_bear",
                  "cg_funding_close_zscore", "cg_oi_close_pctchg_4h",
                  "cg_fcvd_delta", "cg_scvd_delta", "cg_taker_delta_zscore",
                  "cg_spot_futures_cvd_divergence", "cg_liq_imbalance",
                  "cvd_persistence_12h"]
    print("raw inputs present:",
          all(c in df.columns for c in raw_inputs), raw_inputs)

    cand = build_candidates(df)

    # leak-safety self-check (truncated-frame identity)
    selfcheck = leak_safety_selfcheck(df, cand)
    print("\n[LEAK-SAFETY]", selfcheck)

    print("\n--- screening vs CLEAN V7 OOS residual (conditional IC) ---")
    res = screen_candidates(cand, existing_df=df)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    show_cols = ["feature", "cond_ic", "raw_ic", "frac_fold_consist",
                 "n_folds", "nonzero_frac", "max_corr_existing", "SCREEN_PASS"]
    print(res[show_cols].to_string(index=False))

    passers = res[res["SCREEN_PASS"]]["feature"].tolist()
    print(f"\nPASSERS: {len(passers)} -> {passers}")

    if passers:
        OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        cand[passers].to_parquet(OUT_PARQUET)
        print(f"saved {len(passers)} passing cols -> {OUT_PARQUET}")
    else:
        print("no passers; parquet not written")

    # top-5 by |cond_ic| for the synthesis (even if failed)
    top5 = res.sort_values("abs_cond_ic", ascending=False).head(5)
    print("\n--- TOP 5 by |cond_ic| ---")
    print(top5[show_cols].to_string(index=False))

    # machine-readable footer for the harness
    print("\n===RESULT_JSON_START===")
    import json
    out = {
        "n_built": int(cand.shape[1]),
        "passers": [
            {
                "name": r["feature"],
                "cond_ic": None if pd.isna(r["cond_ic"]) else float(r["cond_ic"]),
                "consist": None if pd.isna(r["frac_fold_consist"]) else float(r["frac_fold_consist"]),
                "max_corr": None if pd.isna(r["max_corr_existing"]) else float(r["max_corr_existing"]),
                "nonzero": None if pd.isna(r["nonzero_frac"]) else float(r["nonzero_frac"]),
            }
            for _, r in res[res["SCREEN_PASS"]].iterrows()
        ],
        "parquet_path": str(OUT_PARQUET) if passers else "",
        "selfcheck": selfcheck,
        "top5": [
            {
                "name": r["feature"],
                "cond_ic": None if pd.isna(r["cond_ic"]) else round(float(r["cond_ic"]), 4),
                "consist": None if pd.isna(r["frac_fold_consist"]) else round(float(r["frac_fold_consist"]), 3),
                "max_corr": None if pd.isna(r["max_corr_existing"]) else round(float(r["max_corr_existing"]), 3),
                "nonzero": None if pd.isna(r["nonzero_frac"]) else round(float(r["nonzero_frac"]), 3),
            }
            for _, r in top5.iterrows()
        ],
    }
    print(json.dumps(out, indent=2))
    print("===RESULT_JSON_END===")


if __name__ == "__main__":
    main()
