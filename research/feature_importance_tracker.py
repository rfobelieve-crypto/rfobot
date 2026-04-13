"""
Feature Importance Version Tracker — detect gradual model/data degradation.

Every retrain overwrites direction_importance.csv and magnitude_importance.csv,
so the historical importance trajectory is lost. This tool:

  snapshot:   copy current importance CSVs to history/ tagged with date + git SHA
  diff:       compare the most recent snapshot to an earlier one and flag any
              feature whose importance changed by more than a threshold (default 50%)
              — either the feature's data source broke, or the model is drifting

Typical failure modes this catches:
  1. API endpoint silently returns stale data → importance of that feature drops
     dramatically at next retrain (gradual degradation nobody noticed)
  2. Feature builder bug introduced in a recent commit → importance shifts abruptly
     (you can git bisect the snapshot dates)
  3. Structural market regime change → a feature that worked in bull suddenly
     becomes irrelevant in chop (not a bug, but actionable — consider retrain
     cadence)

Usage:
    python research/feature_importance_tracker.py snapshot
    python research/feature_importance_tracker.py diff
    python research/feature_importance_tracker.py diff --from 2026-03-01 --to 2026-04-13
    python research/feature_importance_tracker.py list     # show all snapshots

Output:
    indicator/model_artifacts/dual_model/history/
        direction_importance_YYYYMMDD_<sha>.csv
        magnitude_importance_YYYYMMDD_<sha>.csv
    research/results/feature_importance_diff.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ARTIFACT_DIR = Path("indicator/model_artifacts/dual_model")
HISTORY_DIR = ARTIFACT_DIR / "history"
OUT_JSON = Path("research/results/feature_importance_diff.json")

CURRENT_FILES = {
    "direction": ARTIFACT_DIR / "direction_importance.csv",
    "magnitude": ARTIFACT_DIR / "magnitude_importance.csv",
}


def git_sha_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def snapshot():
    """Copy current importance CSVs to history/ with date+sha tag."""
    if not CURRENT_FILES["direction"].exists() or not CURRENT_FILES["magnitude"].exists():
        print("ERROR: importance CSV files not found in", ARTIFACT_DIR)
        return 1

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y%m%d")
    sha = git_sha_short()
    tag = f"{date}_{sha}"

    for kind, src in CURRENT_FILES.items():
        dst = HISTORY_DIR / f"{kind}_importance_{tag}.csv"
        shutil.copy(src, dst)
        print(f"  saved: {dst.name}")

    # Also save a metadata sidecar (what model config produced this snapshot)
    meta = {
        "date": date,
        "sha": sha,
        "timestamp": datetime.now().isoformat(),
    }
    for kind in ("direction", "magnitude"):
        cfg = ARTIFACT_DIR / f"{kind}_config.json"
        if cfg.exists():
            meta[f"{kind}_config"] = json.loads(cfg.read_text())

    (HISTORY_DIR / f"meta_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"  saved: meta_{tag}.json")
    return 0


def list_snapshots():
    """List all snapshots in history."""
    if not HISTORY_DIR.exists():
        print("No snapshots yet. Run `snapshot` first.")
        return 0

    snapshots: dict[str, list[str]] = {}
    for p in sorted(HISTORY_DIR.glob("*_importance_*.csv")):
        # Parse filename: {kind}_importance_{YYYYMMDD}_{sha}.csv
        parts = p.stem.split("_")
        kind = parts[0]
        tag = "_".join(parts[2:])
        snapshots.setdefault(tag, []).append(kind)

    print(f"Snapshots in {HISTORY_DIR}:")
    for tag in sorted(snapshots.keys()):
        kinds = ", ".join(sorted(snapshots[tag]))
        print(f"  {tag}: [{kinds}]")
    return 0


def load_snapshot(tag: str, kind: str) -> pd.DataFrame:
    path = HISTORY_DIR / f"{kind}_importance_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # CSV format from export scripts: first column = feature name, second = importance
    df.columns = ["feature", "importance"] + list(df.columns[2:])
    return df[["feature", "importance"]].set_index("feature")


def find_tag(pattern: str) -> str | None:
    """Find snapshot tag matching a date prefix."""
    if not HISTORY_DIR.exists():
        return None
    matches = sorted({
        "_".join(p.stem.split("_")[2:])
        for p in HISTORY_DIR.glob("*_importance_*.csv")
    })
    if not matches:
        return None
    if pattern == "latest":
        return matches[-1]
    if pattern == "previous":
        return matches[-2] if len(matches) >= 2 else None
    # Try prefix match on date
    for m in matches:
        if m.startswith(pattern.replace("-", "")):
            return m
    return None


def diff(from_tag: str, to_tag: str, threshold: float = 0.5):
    """Compare two snapshots, flag features whose importance shifted > threshold."""
    report = {
        "from_tag": from_tag,
        "to_tag": to_tag,
        "threshold": threshold,
        "direction": {},
        "magnitude": {},
    }

    for kind in ("direction", "magnitude"):
        try:
            old = load_snapshot(from_tag, kind)
            new = load_snapshot(to_tag, kind)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            continue

        merged = old.join(new, how="outer", lsuffix="_old", rsuffix="_new").fillna(0)
        merged["delta"] = merged["importance_new"] - merged["importance_old"]
        merged["pct_change"] = (
            merged["delta"] / merged["importance_old"].clip(lower=1e-6)
        )

        # Significant changes
        flagged = merged[merged["pct_change"].abs() > threshold].copy()
        flagged = flagged.sort_values("pct_change", key=lambda x: x.abs(), ascending=False)

        # New features (present in new but not old)
        added = merged[(merged["importance_old"] == 0) & (merged["importance_new"] > 0)]
        removed = merged[(merged["importance_old"] > 0) & (merged["importance_new"] == 0)]

        print(f"\n── {kind.upper()} MODEL ──")
        print(f"Old ({from_tag}): {len(old)} features")
        print(f"New ({to_tag}): {len(new)} features")

        if len(added) > 0:
            print(f"\n  Added features ({len(added)}):")
            for feat, row in added.iterrows():
                print(f"    + {feat:32s}  importance={row['importance_new']:.4f}")

        if len(removed) > 0:
            print(f"\n  Removed features ({len(removed)}):")
            for feat, row in removed.iterrows():
                print(f"    - {feat:32s}  was={row['importance_old']:.4f}")

        if len(flagged) > 0:
            print(f"\n  Flagged (|change| > {threshold*100:.0f}%):")
            print(f"    {'Feature':32s}  {'Old':>8s}  {'New':>8s}  {'Δ':>9s}  {'% chg':>9s}")
            for feat, row in flagged.iterrows():
                if feat in added.index or feat in removed.index:
                    continue
                print(f"    {feat:32s}  {row['importance_old']:>8.4f}  "
                      f"{row['importance_new']:>8.4f}  "
                      f"{row['delta']:>+9.4f}  {row['pct_change']*100:>+8.1f}%")
        else:
            print(f"\n  No features with |change| > {threshold*100:.0f}%.")

        # Correlation of top-10 importance rank (sanity check: is the model using
        # the same top features in roughly the same order?)
        top_old = old.nlargest(10, "importance").index.tolist()
        top_new = new.nlargest(10, "importance").index.tolist()
        overlap = set(top_old) & set(top_new)
        print(f"\n  Top-10 importance overlap: {len(overlap)}/10 features")
        if len(overlap) < 7:
            print(f"    WARNING: top features changed significantly — model is drifting")

        report[kind] = {
            "n_old": len(old),
            "n_new": len(new),
            "n_added": len(added),
            "n_removed": len(removed),
            "n_flagged": len(flagged),
            "top10_overlap": len(overlap),
            "added": added.index.tolist(),
            "removed": removed.index.tolist(),
            "flagged": [
                {"feature": f, **flagged.loc[f].to_dict()}
                for f in flagged.index
            ],
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved: {OUT_JSON}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("snapshot", help="Snapshot current importance CSVs to history/")
    sub.add_parser("list", help="List all snapshots")

    d = sub.add_parser("diff", help="Compare two snapshots")
    d.add_argument("--from", dest="from_tag", default="previous",
                   help="Earlier snapshot tag, date prefix, or 'previous'")
    d.add_argument("--to", dest="to_tag", default="latest",
                   help="Later snapshot tag, date prefix, or 'latest'")
    d.add_argument("--threshold", type=float, default=0.5,
                   help="Percent-change threshold to flag (default 0.5 = 50%)")

    args = ap.parse_args()

    if args.cmd == "snapshot":
        return snapshot()
    if args.cmd == "list":
        return list_snapshots()
    if args.cmd == "diff":
        from_tag = find_tag(args.from_tag) or args.from_tag
        to_tag = find_tag(args.to_tag) or args.to_tag
        if not from_tag or not to_tag:
            print(f"ERROR: could not resolve snapshot tags "
                  f"(from={args.from_tag}, to={args.to_tag}).")
            print("Run `snapshot` at least twice first, or use `list` to see tags.")
            return 1
        if from_tag == to_tag:
            print(f"ERROR: from and to are the same snapshot ({from_tag}).")
            return 1
        return diff(from_tag, to_tag, args.threshold)


if __name__ == "__main__":
    sys.exit(main())
