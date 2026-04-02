"""
OOS-only backtest: only evaluates bars AFTER the training data cutoff.
Automatically detects training end date from the enhanced parquet.
Appends results to research/oos_tracking.csv for longitudinal tracking.

Usage:
    python research/backtest_oos.py
"""
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from indicator.data_fetcher import fetch_binance_klines, fetch_coinglass
from indicator.feature_builder_live import build_live_features
from indicator.inference import IndicatorEngine, STRENGTH_DEADZONE, STRONG_THRESHOLD, MODERATE_THRESHOLD

TRACKING_FILE = Path("research/oos_tracking.csv")
ENHANCED_PARQUET = Path("research/ml_data/BTC_USD_1h_enhanced.parquet")

# ── Detect training cutoff ────────────────────────────────────────────────
if ENHANCED_PARQUET.exists():
    train_df = pd.read_parquet(ENHANCED_PARQUET, columns=[])
    TRAIN_END = train_df.index[-1]
    print(f"Training data ends: {TRAIN_END}")
else:
    # Fallback: hardcode known cutoff
    TRAIN_END = pd.Timestamp("2026-04-01 21:00:00", tz="UTC")
    print(f"Training parquet not found, using hardcoded cutoff: {TRAIN_END}")

# Need 4 bars after a bar to know the outcome
OOS_START = TRAIN_END + pd.Timedelta(hours=1)
print(f"OOS evaluation starts: {OOS_START}")
print()

# ── Fetch & predict ───────────────────────────────────────────────────────
klines = fetch_binance_klines(limit=500)
cg = fetch_coinglass(interval="1h", limit=500)
features = build_live_features(klines, cg)

engine = IndicatorEngine()
result = engine.predict(features)

# ── Actual 4h return ──────────────────────────────────────────────────────
close = features["close"].values.astype(float)
actual_4h = np.full(len(close), np.nan)
for i in range(len(close) - 4):
    actual_4h[i] = close[i + 4] / close[i] - 1
result["actual_4h_ret"] = actual_4h
result["actual_dir"] = np.where(actual_4h > 0, "UP",
                       np.where(actual_4h < 0, "DOWN", "NEUTRAL"))

# ── Filter OOS only ──────────────────────────────────────────────────────
oos = result[(result.index >= OOS_START) & ~np.isnan(result["actual_4h_ret"])].copy()

if len(oos) == 0:
    print("No OOS bars with completed 4h outcomes yet.")
    print(f"Need data after {OOS_START + pd.Timedelta(hours=4)} for first OOS outcome.")
    sys.exit(0)

print(f"=== OOS BACKTEST: {oos.index[0].strftime('%m/%d %H:%M')} ~ {oos.index[-1].strftime('%m/%d %H:%M')} ===")
print(f"Deadzone={STRENGTH_DEADZONE}, Strong>={STRONG_THRESHOLD}, Moderate>={MODERATE_THRESHOLD}")
print(f"OOS bars: {len(oos)}")
print()

# ── Stats ─────────────────────────────────────────────────────────────────
signals = oos[oos["pred_direction"].isin(["UP", "DOWN"])]
neutral = oos[oos["pred_direction"] == "NEUTRAL"]

if len(signals) == 0:
    print("No directional signals in OOS window.")
    sys.exit(0)

correct = (signals["pred_direction"] == signals["actual_dir"]).sum()
total_sig = len(signals)
acc = correct / total_sig

print(f"Signal bars: {total_sig} / {len(oos)} ({total_sig/len(oos):.0%})")
print(f"NEUTRAL bars: {len(neutral)} ({len(neutral)/len(oos):.0%})")
print(f"Direction accuracy: {acc:.1%} ({correct}/{total_sig})")

# Warn if sample too small
if total_sig < 50:
    se = np.sqrt(acc * (1 - acc) / total_sig)
    ci_low, ci_high = acc - 1.96 * se, acc + 1.96 * se
    print(f"  ** Small sample — 95% CI: [{ci_low:.1%}, {ci_high:.1%}] **")
print()

# ── By Tier ───────────────────────────────────────────────────────────────
print("=== BY CONFIDENCE TIER ===")
for tier in ["Strong", "Moderate", "Weak"]:
    sub = signals[signals["strength_score"] == tier]
    if len(sub) == 0:
        print(f"  {tier:10s}: 0 signals")
        continue
    c = (sub["pred_direction"] == sub["actual_dir"]).sum()

    dir_ret = []
    for _, row in sub.iterrows():
        if row["pred_direction"] == "UP":
            dir_ret.append(row["actual_4h_ret"])
        else:
            dir_ret.append(-row["actual_4h_ret"])
    avg_dir_ret = np.mean(dir_ret) * 100

    print(f"  {tier:10s}: {len(sub):3d} signals, accuracy={c/len(sub):.1%}, "
          f"avg directional return={avg_dir_ret:+.4f}%")

# ── Signal Detail ─────────────────────────────────────────────────────────
print()
print("=== OOS SIGNALS ===")
hdr = f"{'Time':20s}  {'Pred':5s}  {'Actual':6s}  {'OK':3s}  {'Tier':10s}  {'Conf':>5s}  {'Diff':>7s}  {'4h Ret':>8s}  Regime"
print(hdr)
print("-" * len(hdr))

for idx, row in signals.iterrows():
    d = row["pred_direction"]
    ad = row["actual_dir"]
    ok = "Y" if d == ad else "N"
    s = row["strength_score"]
    c = row["confidence_score"]
    up = row.get("up_pred", 0)
    down = row.get("down_pred", 0)
    diff = up - down
    ret = row["actual_4h_ret"] * 100
    regime = row["regime"]
    print(f"{str(idx)[:19]}  {d:5s}  {ad:6s}  {ok:>3s}  {s:10s}  {c:5.1f}  {diff:+7.3f}  {ret:+7.3f}%  {regime}")

# ── Append to tracking CSV ────────────────────────────────────────────────
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
new_rows = []
for idx, row in signals.iterrows():
    new_rows.append({
        "eval_time": now,
        "bar_time": str(idx)[:19],
        "pred_direction": row["pred_direction"],
        "actual_dir": row["actual_dir"],
        "correct": int(row["pred_direction"] == row["actual_dir"]),
        "strength_score": row["strength_score"],
        "confidence_score": round(row["confidence_score"], 1),
        "strength_raw": round(row.get("up_pred", 0) - row.get("down_pred", 0), 3),
        "actual_4h_ret": round(row["actual_4h_ret"], 6),
        "regime": row["regime"],
    })

new_df = pd.DataFrame(new_rows)

if TRACKING_FILE.exists():
    existing = pd.read_csv(TRACKING_FILE)
    # Deduplicate by bar_time (don't double-count)
    existing_bars = set(existing["bar_time"].values)
    new_df = new_df[~new_df["bar_time"].isin(existing_bars)]
    if len(new_df) > 0:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(TRACKING_FILE, index=False)
        print(f"\n+{len(new_df)} new OOS bars appended to {TRACKING_FILE}")
        print(f"Total tracked: {len(combined)} bars")
    else:
        print(f"\nNo new bars to append (all already tracked)")
        print(f"Total tracked: {len(existing)} bars")
else:
    new_df.to_csv(TRACKING_FILE, index=False)
    print(f"\nCreated {TRACKING_FILE} with {len(new_df)} OOS bars")

# ── Cumulative OOS stats ──────────────────────────────────────────────────
if TRACKING_FILE.exists():
    all_oos = pd.read_csv(TRACKING_FILE)
    total = len(all_oos)
    total_correct = all_oos["correct"].sum()
    print(f"\n=== CUMULATIVE OOS STATS ({total} bars) ===")
    print(f"Overall: {total_correct}/{total} ({total_correct/total:.1%})")
    for tier in ["Strong", "Moderate", "Weak"]:
        sub = all_oos[all_oos["strength_score"] == tier]
        if len(sub) > 0:
            c = sub["correct"].sum()
            print(f"  {tier:10s}: {c}/{len(sub)} ({c/len(sub):.1%})")
