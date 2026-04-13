"""
Train/Serve Diff Monitor — detect data drift between research cache and production.

Catches bugs like 2026-04-12 backfill timestamp unit mismatch, where research
cache silently had corrupt data while production ran on different values.

Three levels of diff:
  1. Cache freshness — is features_all.parquet older than raw parquets?
  2. OHLC consistency — do close prices in cache match production indicator_history
     on overlapping timestamps? (should be identical; any drift = data source issue)
  3. Feature stability — rebuild features from current raw parquets and compare
     to cached features_all. Any column with correlation < 0.999 or large absolute
     diff gets flagged — it means feature_builder_live.py was edited after the
     cache was last built, OR a data source silently changed.

Run this weekly (or before any retrain) as a production data sanity check.

Usage:
    python research/train_serve_diff.py
    python research/train_serve_diff.py --rebuild      # force rebuild cache for comparison
    python research/train_serve_diff.py --tol 0.001    # custom max diff threshold

Output:
    research/results/train_serve_diff.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CACHE = Path("research/dual_model/.cache/features_all.parquet")
PROD_HISTORY = Path("indicator/model_artifacts/indicator_history.parquet")
RAW_DIR = Path("market_data/raw_data")
OUT = Path("research/results/train_serve_diff.json")

# Features that existed in production (from direction/magnitude feature_cols)
# Checking these matters most — if a model input drifts, predictions drift.
import json as _json
PROD_FEATS_DIR = Path("indicator/model_artifacts/dual_model")
_critical_features: set[str] = set()
for name in ("direction_feature_cols.json", "magnitude_feature_cols.json"):
    p = PROD_FEATS_DIR / name
    if p.exists():
        _critical_features.update(_json.loads(p.read_text()))
CRITICAL_FEATURES = sorted(_critical_features)


def check_cache_freshness() -> dict:
    """Stage 1: is the cache older than any raw parquet?"""
    if not CACHE.exists():
        return {"status": "missing", "cache_mtime": None, "raw_mtimes": {}}

    cache_mtime = CACHE.stat().st_mtime
    raw_mtimes = {}
    stale_sources = []

    for p in sorted(RAW_DIR.glob("*.parquet")):
        m = p.stat().st_mtime
        raw_mtimes[p.name] = m
        if m > cache_mtime:
            stale_sources.append(p.name)

    return {
        "status": "stale" if stale_sources else "fresh",
        "cache_mtime": cache_mtime,
        "cache_mtime_iso": pd.Timestamp(cache_mtime, unit="s").isoformat(),
        "stale_sources": stale_sources,
        "raw_count": len(raw_mtimes),
    }


def check_ohlc_consistency(tol_pct: float = 0.001) -> dict:
    """
    Stage 2: do cache close prices match production close prices on overlap?
    Any diff means either:
      - Backfill and live data use different sources
      - Timestamp alignment is off by one bar
      - One side has a bad ingestion run
    """
    if not CACHE.exists() or not PROD_HISTORY.exists():
        return {"status": "skipped", "reason": "missing files"}

    cache = pd.read_parquet(CACHE)
    prod = pd.read_parquet(PROD_HISTORY)

    # Strip tz for alignment
    if hasattr(cache.index, "tz") and cache.index.tz is not None:
        cache.index = cache.index.tz_convert("UTC").tz_localize(None)
    if hasattr(prod.index, "tz") and prod.index.tz is not None:
        prod.index = prod.index.tz_convert("UTC").tz_localize(None)

    overlap = cache.index.intersection(prod.index)
    if len(overlap) == 0:
        return {"status": "no_overlap"}

    cache_o = cache.loc[overlap]
    prod_o = prod.loc[overlap]

    results = {"status": "ok", "overlap_bars": len(overlap),
               "overlap_start": str(overlap.min()), "overlap_end": str(overlap.max()),
               "columns": {}}

    for col in ["close", "open", "high", "low"]:
        if col not in cache_o.columns or col not in prod_o.columns:
            continue
        a = cache_o[col].astype(float)
        b = prod_o[col].astype(float)
        mask = a.notna() & b.notna()
        if mask.sum() == 0:
            continue
        abs_diff = (a[mask] - b[mask]).abs()
        rel_diff = abs_diff / b[mask].abs().clip(lower=1e-9)
        results["columns"][col] = {
            "n": int(mask.sum()),
            "max_abs_diff": float(abs_diff.max()),
            "max_rel_diff": float(rel_diff.max()),
            "mean_rel_diff": float(rel_diff.mean()),
            "drift": bool(rel_diff.max() > tol_pct),
        }
        if rel_diff.max() > tol_pct:
            # Find worst offender timestamp
            worst_idx = rel_diff.idxmax()
            results["columns"][col]["worst_ts"] = str(worst_idx)
            results["columns"][col]["worst_cache"] = float(a.loc[worst_idx])
            results["columns"][col]["worst_prod"] = float(b.loc[worst_idx])
            results["status"] = "drift"

    return results


def rebuild_and_compare(tol_corr: float = 0.999, tol_abs_quantile: float = 0.99) -> dict:
    """
    Stage 3: rebuild features from current raw parquets and compare per-column
    to the cached features_all.parquet. Any feature that drifts is flagged.

    This catches the case where feature_builder_live.py was edited after the cache
    was last written — the production code now computes different values than the
    cached training data. Retraining on stale cache would embed the drift.
    """
    from research.dual_model.shared_data import load_and_cache_data

    print("  Rebuilding features from current raw parquets...")
    rebuilt = load_and_cache_data(limit=4000, force_refresh=True, max_stale_hours=48.0)
    print(f"  Rebuilt: {rebuilt.shape}")

    # Re-read cache (may be identical if rebuild overwrote it; use a snapshot instead)
    # Workaround: the above load_and_cache_data writes to CACHE. So snapshot BEFORE.
    return {"status": "N/A", "note": "rebuild overwrites cache; use --snapshot mode"}


def snapshot_then_rebuild_compare(tol_corr: float = 0.999) -> dict:
    """
    Safer Stage 3: make a snapshot of cache, force rebuild, then compare.
    """
    import shutil
    snapshot = CACHE.with_suffix(".snapshot.parquet")
    if not CACHE.exists():
        return {"status": "skipped", "reason": "no cache"}

    shutil.copy(CACHE, snapshot)
    old = pd.read_parquet(snapshot)

    try:
        from research.dual_model.shared_data import load_and_cache_data
        print("  Force-rebuilding features (this may take ~30s)...")
        new = load_and_cache_data(limit=4000, force_refresh=True, max_stale_hours=48.0)
    finally:
        # Restore from snapshot regardless
        shutil.copy(snapshot, CACHE)
        snapshot.unlink()

    # Align indices
    if hasattr(old.index, "tz") and old.index.tz is not None:
        old.index = old.index.tz_convert("UTC").tz_localize(None)
    if hasattr(new.index, "tz") and new.index.tz is not None:
        new.index = new.index.tz_convert("UTC").tz_localize(None)
    overlap = old.index.intersection(new.index)
    if len(overlap) == 0:
        return {"status": "no_overlap"}

    old_o = old.loc[overlap]
    new_o = new.loc[overlap]

    results = {"status": "ok", "overlap_bars": len(overlap), "drifted": [], "stable": 0}

    # Check critical features (those actually used by production models)
    cols_to_check = [c for c in CRITICAL_FEATURES if c in old_o.columns and c in new_o.columns]
    # Also check any new columns in either
    only_in_old = set(old_o.columns) - set(new_o.columns)
    only_in_new = set(new_o.columns) - set(old_o.columns)
    results["only_in_old_cache"] = sorted(only_in_old)
    results["only_in_new_rebuild"] = sorted(only_in_new)

    for col in cols_to_check:
        a = old_o[col].astype(float)
        b = new_o[col].astype(float)
        mask = a.notna() & b.notna()
        if mask.sum() < 10:
            continue
        if a[mask].std() < 1e-12 or b[mask].std() < 1e-12:
            # constant column, skip correlation
            max_abs = float((a[mask] - b[mask]).abs().max())
            if max_abs > 1e-9:
                results["drifted"].append({"feature": col, "type": "constant_drift",
                                           "max_abs_diff": max_abs})
            else:
                results["stable"] += 1
            continue
        corr = float(a[mask].corr(b[mask]))
        max_abs = float((a[mask] - b[mask]).abs().max())
        scale = float(a[mask].abs().mean()) or 1e-9
        rel_diff = max_abs / scale

        if corr < tol_corr or rel_diff > 0.01:
            results["drifted"].append({
                "feature": col,
                "correlation": corr,
                "max_abs_diff": max_abs,
                "relative_diff": rel_diff,
            })
            results["status"] = "drift"
        else:
            results["stable"] += 1

    results["n_critical_checked"] = len(cols_to_check)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="Force rebuild cache and compare (slow, ~30s)")
    ap.add_argument("--tol", type=float, default=0.001,
                    help="Max relative OHLC drift before flag (default 0.001 = 0.1%)")
    args = ap.parse_args()

    print("=" * 70)
    print("TRAIN/SERVE DIFF MONITOR")
    print("=" * 70)

    report = {}

    # Stage 1
    print("\n[1/3] Cache freshness check")
    fresh = check_cache_freshness()
    report["freshness"] = fresh
    print(f"  Cache mtime: {fresh.get('cache_mtime_iso', 'N/A')}")
    print(f"  Raw sources newer than cache: {len(fresh.get('stale_sources', []))}")
    for s in fresh.get("stale_sources", [])[:5]:
        print(f"    - {s}")
    print(f"  Status: {fresh['status']}")

    # Stage 2
    print("\n[2/3] OHLC consistency (cache vs production indicator_history)")
    ohlc = check_ohlc_consistency(tol_pct=args.tol)
    report["ohlc"] = ohlc
    if ohlc["status"] == "skipped":
        print(f"  Skipped: {ohlc['reason']}")
    elif ohlc["status"] == "no_overlap":
        print("  No overlapping timestamps between cache and production history.")
    else:
        print(f"  Overlap: {ohlc['overlap_bars']} bars "
              f"({ohlc['overlap_start']} -> {ohlc['overlap_end']})")
        for col, m in ohlc["columns"].items():
            status = "DRIFT" if m["drift"] else "ok"
            print(f"    {col}: max_rel_diff={m['max_rel_diff']:.6f}  "
                  f"mean={m['mean_rel_diff']:.6f}  [{status}]")
            if m["drift"] and "worst_ts" in m:
                print(f"         worst @ {m['worst_ts']}: "
                      f"cache={m['worst_cache']:.2f} prod={m['worst_prod']:.2f}")
        print(f"  Status: {ohlc['status']}")

    # Stage 3
    print("\n[3/3] Feature rebuild diff (critical model features only)")
    if args.rebuild:
        rebuild_result = snapshot_then_rebuild_compare()
        report["rebuild"] = rebuild_result
        if rebuild_result["status"] == "ok":
            print(f"  Stable: {rebuild_result['stable']}/{rebuild_result['n_critical_checked']}")
            if rebuild_result["only_in_old_cache"]:
                print(f"  Only in old cache (removed?): "
                      f"{rebuild_result['only_in_old_cache'][:5]}")
            if rebuild_result["only_in_new_rebuild"]:
                print(f"  Only in new rebuild (added?): "
                      f"{rebuild_result['only_in_new_rebuild'][:5]}")
        elif rebuild_result["status"] == "drift":
            print(f"  DRIFT on {len(rebuild_result['drifted'])} features:")
            for d in rebuild_result["drifted"][:10]:
                print(f"    {d['feature']}: corr={d.get('correlation', 'N/A')}  "
                      f"rel_diff={d.get('relative_diff', 'N/A')}")
    else:
        print("  Skipped (use --rebuild to run full feature rebuild comparison)")
        report["rebuild"] = {"status": "skipped"}

    # Summary verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    problems = []
    if fresh["status"] == "stale":
        problems.append(f"Cache stale — {len(fresh['stale_sources'])} raw sources newer")
    if report["ohlc"].get("status") == "drift":
        problems.append("OHLC drift between cache and production")
    if report.get("rebuild", {}).get("status") == "drift":
        n = len(report["rebuild"]["drifted"])
        problems.append(f"{n} critical features drift after rebuild")

    if problems:
        for p in problems:
            print(f"  [WARN] {p}")
    else:
        print("  [OK] No drift detected.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
