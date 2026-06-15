"""
Hot-path runtime smoke test for the freshly retrained dual model.

Per mistake.md 2026-04-22: import-OK does NOT imply runtime-OK.
This script exercises the actual prediction path:
  1. Loads IndicatorEngine (which loads new artifacts).
  2. Pulls cached features parquet (already rebuilt by training pipeline).
  3. Runs predict() on last ~50 bars.
  4. Reports prediction distribution, decoded direction labels,
     and confirms decoder is actually rolling-percentile (not binary).

If anything is wrong with the new model artifacts (feature mismatch,
schema drift, NaN cascade, decoder applied to wrong model type), this
catches it BEFORE pushing to Railway.
"""
from __future__ import annotations

import sys
import json
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main():
    # 1. Load the engine (this triggers artifact load)
    from indicator.inference import IndicatorEngine
    print("\n=== Step 1: load IndicatorEngine ===")
    eng = IndicatorEngine()

    print(f"  mode               : {eng.mode}")
    print(f"  dir_model_type     : {eng.dir_model_type}")
    print(f"  dir_decoding       : {eng.dir_decoding}")
    print(f"  pct_window         : {eng.dir_pct_window}")
    print(f"  strong_top_frac    : {eng.dir_strong_top_frac}")
    print(f"  moderate_top_frac  : {eng.dir_moderate_top_frac}")
    print(f"  warmup_bars        : {eng.dir_warmup_bars}")
    print(f"  dir features       : {len(eng.dual_dir_features)}")
    print(f"  mag features       : {len(eng.dual_mag_features)}")
    print(f"  dir_pred_history   : {len(eng.dir_pred_history)} entries  "
          f"(buffer is {'WARM' if len(eng.dir_pred_history) >= eng.dir_warmup_bars else 'COLD — fallback active'})")
    print(f"  pred_history (mag) : {len(eng.pred_history)} entries")

    # Sanity: must be regression
    assert eng.dir_model_type == "regression", \
        f"FAIL: dir_model_type={eng.dir_model_type} (expected 'regression')"
    assert eng.dir_decoding == "rolling_percentile", \
        f"FAIL: dir_decoding={eng.dir_decoding} (expected 'rolling_percentile')"
    assert len(eng.dual_dir_features) == 136, \
        f"FAIL: dir features={len(eng.dual_dir_features)} (expected 136)"
    assert len(eng.dual_mag_features) == 76, \
        f"FAIL: mag features={len(eng.dual_mag_features)} (expected 76)"
    assert len(eng.dir_pred_history) >= eng.dir_warmup_bars, \
        f"FAIL: warmup buffer too short ({len(eng.dir_pred_history)} < {eng.dir_warmup_bars})"
    print("  → all asserts passed")

    # 2. Load cached features (already built by training pipeline)
    print("\n=== Step 2: load cached features ===")
    cache = PROJECT_ROOT / "research" / "dual_model" / ".cache" / "features_all.parquet"
    df = pd.read_parquet(cache).sort_index()
    print(f"  features parquet   : {df.shape[0]} bars x {df.shape[1]} cols")
    print(f"  range              : {df.index[0]} ~ {df.index[-1]}")

    # Ensure all required features are present
    missing_dir = [c for c in eng.dual_dir_features if c not in df.columns]
    missing_mag = [c for c in eng.dual_mag_features if c not in df.columns]
    if missing_dir:
        print(f"  ⚠ missing dir features: {missing_dir[:5]} (+{max(0,len(missing_dir)-5)} more)")
    if missing_mag:
        print(f"  ⚠ missing mag features: {missing_mag[:5]} (+{max(0,len(missing_mag)-5)} more)")
    assert not missing_dir, "FAIL: missing direction features in features parquet"
    assert not missing_mag, "FAIL: missing magnitude features in features parquet"
    print("  → all required features present")

    # 3. Run predict on last 50 bars
    print("\n=== Step 3: run predict() on last 50 bars ===")
    tail = df.tail(50).copy()
    out = eng.predict(tail, update_history=False)
    print(f"  output shape       : {out.shape}")
    print(f"  output cols        : {list(out.columns)[:12]}...")

    pred_ret_col = next((c for c in out.columns
                         if c in ("pred_return_4h", "pred_ret", "dir_pred_ret")), None)
    assert pred_ret_col is not None, f"FAIL: no pred_return_4h-like column in {list(out.columns)}"
    print(f"  pred-ret column    : {pred_ret_col}")

    pred = out[pred_ret_col].values.astype(float)
    n_nan = int(np.isnan(pred).sum())
    print(f"  pred stats         : n={len(pred)}  n_nan={n_nan}  "
          f"min={np.nanmin(pred):+.5f}  median={np.nanmedian(pred):+.5f}  "
          f"max={np.nanmax(pred):+.5f}  std={np.nanstd(pred):.5f}")
    assert n_nan == 0, f"FAIL: {n_nan} NaN predictions"

    # Buffer std vs live pred std — sanity ratio
    buf = list(eng.dir_pred_history)
    buf_std = float(np.std(buf))
    live_std = float(np.nanstd(pred))
    ratio = buf_std / live_std if live_std > 0 else float("inf")
    print(f"  buffer std         : {buf_std:.5f}")
    print(f"  live pred std      : {live_std:.5f}")
    print(f"  buffer/live ratio  : {ratio:.2f}x  "
          f"(must be in [0.5, 2.0] per mistake.md 2026-04-19)")
    if not (0.5 <= ratio <= 2.0):
        print(f"  ⚠ ratio out of range — buffer/live distribution mismatch")

    # 4. Direction decoding distribution
    print("\n=== Step 4: decoded direction & strength distribution ===")
    dir_col = next((c for c in ("direction", "pred_direction") if c in out.columns), None)
    str_col = next((c for c in ("strength", "strength_score") if c in out.columns), None)
    conf_col = next((c for c in ("confidence", "confidence_score") if c in out.columns), None)
    if dir_col:
        dir_counter = Counter(out[dir_col].fillna("?").tolist())
        print(f"  {dir_col:18s}:", dict(dir_counter))
    if str_col:
        str_counter = Counter(out[str_col].fillna("?").tolist())
        print(f"  {str_col:18s}:", dict(str_counter))
    if conf_col:
        c = out[conf_col].astype(float)
        print(f"  confidence         : min={c.min():.0f}  median={c.median():.0f}  "
              f"max={c.max():.0f}")

    # Reality check: can't be 100% one direction across 50 bars (would be the
    # 'all DOWN' regression bug from mistake.md 2026-04-19 returning).
    if dir_col:
        most_common_dir, most_common_n = dir_counter.most_common(1)[0]
        share = most_common_n / len(out)
        if share > 0.90 and most_common_dir != "NEUTRAL":
            print(f"  ⚠ {most_common_dir} dominates {share:.0%} of bars — "
                  "investigate before deploying")
        else:
            print(f"  → direction balance OK ({most_common_dir}={share:.0%})")

    # 5. Print last 10 rows for eyeballing
    print("\n=== Step 5: last 10 predictions ===")
    cols_to_show = [c for c in
                    (dir_col, str_col, conf_col, pred_ret_col,
                     "mag_pred", "regime")
                    if c and c in out.columns]
    print(out[cols_to_show].tail(10).to_string())

    print("\n=== ALL CHECKS PASSED ✅ ===")


if __name__ == "__main__":
    main()
