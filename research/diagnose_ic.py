"""
Dual Model IC Diagnostic — connect to Railway MySQL and analyze
live predictions from the dual-model period only.

Usage:
    python research/diagnose_ic.py              # uses .env DB config
    python research/diagnose_ic.py --parquet    # fallback to local parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Dual model stable start (after WARMUP ended)
DUAL_MODEL_START = "2026-03-20"


def load_from_mysql() -> pd.DataFrame:
    """Load indicator_history from MySQL (Railway)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from shared.db import get_db_conn

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT dt, `close`, pred_return_4h, pred_direction_code,
                       confidence_score, dir_prob_up,
                       COALESCE(mag_pred, 0) as mag_pred,
                       COALESCE(regime_code, 0) as regime_code
                FROM indicator_history
                WHERE dt >= %s
                ORDER BY dt ASC
            """, (DUAL_MODEL_START,))
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"])
    return df


def load_from_parquet() -> pd.DataFrame:
    """Fallback: load from local parquet."""
    path = Path(__file__).resolve().parent.parent / "indicator" / "model_artifacts" / "indicator_history.parquet"
    df = pd.read_parquet(path)
    df = df.loc[DUAL_MODEL_START:].copy()
    df = df.reset_index().rename(columns={"index": "dt"})
    if "dt" not in df.columns and df.index.name == "dt":
        df = df.reset_index()
    return df


def analyze(df: pd.DataFrame):
    """Run full IC diagnostic."""
    dir_map = {1: "UP", -1: "DOWN", 0: "NEUTRAL"}
    regime_map = {2: "TRENDING_BULL", -2: "TRENDING_BEAR", 0: "CHOPPY", -99: "WARMUP"}

    if "pred_direction_code" in df.columns:
        df["pred_direction"] = df["pred_direction_code"].map(dir_map).fillna("NEUTRAL")
    if "regime_code" in df.columns:
        df["regime"] = df["regime_code"].map(regime_map).fillna("UNKNOWN")

    # Compute actual 4h return
    df["actual_4h"] = df["close"].shift(-4) / df["close"] - 1
    eval_df = df.dropna(subset=["actual_4h", "pred_return_4h"]).copy()

    print(f"{'='*60}")
    print(f"  Dual Model IC Diagnostic")
    print(f"  Period: {DUAL_MODEL_START} ~ {df['dt'].max()}")
    print(f"  Total rows: {len(df)}, with actuals: {len(eval_df)}")
    print(f"{'='*60}\n")

    if len(eval_df) < 10:
        print("Insufficient data with actuals for IC calculation.")
        print(f"(Last 4 bars always lack actuals; data may stop at {df['dt'].max()})")
        return

    # --- Overall IC ---
    ic, p = spearmanr(eval_df["pred_return_4h"], eval_df["actual_4h"])
    print(f"Overall IC: {ic:+.4f} (p={p:.4f}, n={len(eval_df)})")

    # --- Prediction variance check ---
    pred_std = eval_df["pred_return_4h"].std()
    actual_std = eval_df["actual_4h"].std()
    print(f"Pred std:   {pred_std:.6f}")
    print(f"Actual std: {actual_std:.6f}")
    if pred_std > 0:
        print(f"Ratio:      {pred_std / actual_std:.3f} (ideal ~0.3-1.0)")
    print()

    # --- Direction accuracy ---
    directional = eval_df[eval_df["pred_direction"].isin(["UP", "DOWN"])].copy()
    if len(directional) > 0:
        directional["dir_correct"] = (
            ((directional["pred_return_4h"] > 0) & (directional["actual_4h"] > 0)) |
            ((directional["pred_return_4h"] < 0) & (directional["actual_4h"] < 0))
        )
        overall_acc = directional["dir_correct"].mean()
        print(f"Direction accuracy: {overall_acc:.1%} ({len(directional)} directional bars)")

        # By strength tier
        if "confidence_score" in directional.columns:
            for label, lo, hi in [("Strong", 80, 101), ("Moderate", 65, 80), ("Weak", 0, 65)]:
                tier = directional[
                    (directional["confidence_score"] >= lo) &
                    (directional["confidence_score"] < hi)
                ]
                if len(tier) >= 3:
                    acc = tier["dir_correct"].mean()
                    print(f"  {label:>10}: {acc:.1%} (n={len(tier)})")
        print()

    # --- Rolling IC by 3-day windows ---
    print("3-day rolling IC:")
    eval_sorted = eval_df.sort_values("dt")
    window_size = 72  # 3 days * 24h
    for i in range(0, len(eval_sorted), window_size):
        chunk = eval_sorted.iloc[i:i + window_size]
        if len(chunk) >= 10:
            wic, _ = spearmanr(chunk["pred_return_4h"], chunk["actual_4h"])
            d0 = chunk["dt"].iloc[0].strftime("%m/%d")
            d1 = chunk["dt"].iloc[-1].strftime("%m/%d")

            non_neutral = chunk[chunk["pred_return_4h"] != 0]
            if len(non_neutral) > 0:
                dc = (
                    ((non_neutral["pred_return_4h"] > 0) & (non_neutral["actual_4h"] > 0)) |
                    ((non_neutral["pred_return_4h"] < 0) & (non_neutral["actual_4h"] < 0))
                ).mean()
            else:
                dc = float("nan")
            bar = "+" * max(0, int(wic * 20)) + "-" * max(0, int(-wic * 20))
            print(f"  {d0}~{d1}: IC={wic:+.4f} DirAcc={dc:.0%} n={len(chunk)} |{bar}|")
    print()

    # --- IC by regime ---
    if "regime" in eval_df.columns:
        print("IC by regime:")
        for regime, grp in eval_df.groupby("regime"):
            if len(grp) >= 10 and regime not in ("UNKNOWN", "WARMUP"):
                ric, _ = spearmanr(grp["pred_return_4h"], grp["actual_4h"])
                dir_sub = grp[grp["pred_return_4h"] != 0]
                if len(dir_sub) > 0:
                    dc = (
                        ((dir_sub["pred_return_4h"] > 0) & (dir_sub["actual_4h"] > 0)) |
                        ((dir_sub["pred_return_4h"] < 0) & (dir_sub["actual_4h"] < 0))
                    ).mean()
                else:
                    dc = float("nan")
                print(f"  {regime:>18}: IC={ric:+.4f}, DirAcc={dc:.0%}, n={len(grp)}")
        print()

    # --- Calibration: confidence quartiles ---
    if len(directional) >= 20:
        print("Calibration (confidence quartile -> accuracy):")
        directional = directional.copy()
        directional["conf_q"] = pd.qcut(
            directional["confidence_score"], 4,
            labels=["Q1_low", "Q2", "Q3", "Q4_high"],
            duplicates="drop"
        )
        for q, grp in directional.groupby("conf_q"):
            acc = grp["dir_correct"].mean()
            avg_c = grp["confidence_score"].mean()
            print(f"  {q:>8}: acc={acc:.1%}, avg_conf={avg_c:.0f}, n={len(grp)}")
        print()

    # --- Recent 10 predictions ---
    print("Last 10 predictions vs actuals:")
    for _, row in eval_sorted.tail(10).iterrows():
        pred = row["pred_return_4h"]
        actual = row["actual_4h"]
        ok = "OK" if (pred > 0 and actual > 0) or (pred < 0 and actual < 0) or pred == 0 else "X "
        regime = row.get("regime", "?")
        print(f"  {row['dt'].strftime('%m/%d %H:00')}: "
              f"pred={pred:+.5f} actual={actual:+.5f} [{ok}] {regime}")


def main():
    parser = argparse.ArgumentParser(description="Dual Model IC Diagnostic")
    parser.add_argument("--parquet", action="store_true", help="Use local parquet instead of MySQL")
    args = parser.parse_args()

    if args.parquet:
        print("Loading from local parquet...\n")
        df = load_from_parquet()
    else:
        try:
            print("Connecting to MySQL...\n")
            df = load_from_mysql()
        except Exception as e:
            print(f"MySQL connection failed: {e}")
            print("Falling back to local parquet...\n")
            df = load_from_parquet()

    if len(df) == 0:
        print("No data found.")
        return

    analyze(df)


if __name__ == "__main__":
    main()
