"""
Backtest: new model + new parameters, bar-by-bar simulation.
Processes bars sequentially to keep pred_history consistent (same as live).
Compares predicted direction vs actual 4h return.
"""
import sys
import logging

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from indicator.data_fetcher import fetch_binance_klines, fetch_coinglass
from indicator.feature_builder_live import build_live_features
from indicator.inference import IndicatorEngine, STRENGTH_DEADZONE, STRONG_THRESHOLD, MODERATE_THRESHOLD

# Fetch data
klines = fetch_binance_klines(limit=500)
cg = fetch_coinglass(interval="1h", limit=500)
features = build_live_features(klines, cg)

# Run inference (sequential, same as live)
engine = IndicatorEngine()
result = engine.predict(features)

# Compute actual 4h return for each bar
close = features["close"].values.astype(float)
actual_4h = np.full(len(close), np.nan)
for i in range(len(close) - 4):
    actual_4h[i] = close[i + 4] / close[i] - 1
result["actual_4h_ret"] = actual_4h
result["actual_dir"] = np.where(actual_4h > 0, "UP", np.where(actual_4h < 0, "DOWN", "NEUTRAL"))

# Filter 3/25 onwards, exclude last 4 bars (no outcome yet)
start = pd.Timestamp("2026-03-25", tz="UTC")
r = result[(result.index >= start) & ~np.isnan(result["actual_4h_ret"])].copy()

print(f"=== BACKTEST: {r.index[0].strftime('%m/%d')} ~ {r.index[-1].strftime('%m/%d %H:%M')} ===")
print(f"Deadzone={STRENGTH_DEADZONE}, Strong>={STRONG_THRESHOLD}, Moderate>={MODERATE_THRESHOLD}")
print(f"Total bars: {len(r)}")
print()

# ── Overall Stats ──
signals = r[r["pred_direction"].isin(["UP", "DOWN"])]
neutral = r[r["pred_direction"] == "NEUTRAL"]

correct = (signals["pred_direction"] == signals["actual_dir"]).sum()
total_sig = len(signals)
acc = correct / total_sig if total_sig > 0 else 0

print(f"Signal bars: {total_sig} / {len(r)} ({total_sig/len(r):.0%})")
print(f"NEUTRAL bars: {len(neutral)} ({len(neutral)/len(r):.0%})")
print(f"Direction accuracy (all signals): {acc:.1%} ({correct}/{total_sig})")
print()

# ── By Direction ──
for d in ["UP", "DOWN"]:
    sub = signals[signals["pred_direction"] == d]
    if len(sub) == 0:
        continue
    c = (sub["actual_dir"] == d).sum()
    avg_ret = sub["actual_4h_ret"].mean() * 100
    print(f"  {d}: {len(sub)} signals, accuracy={c/len(sub):.1%}, avg 4h return={avg_ret:+.3f}%")

# ── By Confidence Tier ──
print()
print("=== BY CONFIDENCE TIER ===")
for tier in ["Strong", "Moderate", "Weak"]:
    sub = signals[signals["strength_score"] == tier]
    if len(sub) == 0:
        print(f"  {tier:10s}: 0 signals")
        continue
    c = (sub["pred_direction"] == sub["actual_dir"]).sum()
    avg_ret_correct = sub[sub["pred_direction"] == sub["actual_dir"]]["actual_4h_ret"].abs().mean() * 100
    avg_ret_wrong = sub[sub["pred_direction"] != sub["actual_dir"]]["actual_4h_ret"].abs().mean() * 100

    # Directional return: positive if signal was right
    dir_ret = []
    for _, row in sub.iterrows():
        if row["pred_direction"] == "UP":
            dir_ret.append(row["actual_4h_ret"])
        else:
            dir_ret.append(-row["actual_4h_ret"])
    avg_dir_ret = np.mean(dir_ret) * 100

    print(f"  {tier:10s}: {len(sub):3d} signals, accuracy={c/len(sub):.1%}, "
          f"avg directional return={avg_dir_ret:+.4f}%")

# ── By Regime ──
print()
print("=== BY REGIME ===")
for regime in ["CHOPPY", "TRENDING_BULL", "TRENDING_BEAR"]:
    sub = signals[signals["regime"] == regime]
    if len(sub) < 5:
        continue
    c = (sub["pred_direction"] == sub["actual_dir"]).sum()
    print(f"  {regime:20s}: {len(sub):3d} signals, accuracy={c/len(sub):.1%}")

# ── Signal Detail ──
print()
print("=== ALL SIGNALS WITH OUTCOMES ===")
print()
hdr = f"{'Time':20s}  {'Pred':5s}  {'Actual':6s}  {'OK':3s}  {'Tier':10s}  {'Conf':>5s}  {'Diff':>7s}  {'4h Ret':>8s}  Regime"
print(hdr)
print("-" * len(hdr))

for idx, row in r.iterrows():
    d = row["pred_direction"]
    if d == "NEUTRAL":
        continue

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

# ── Summary ──
print()
print("=== SUMMARY ===")
strong_sig = signals[signals["strength_score"] == "Strong"]
if len(strong_sig) > 0:
    s_correct = (strong_sig["pred_direction"] == strong_sig["actual_dir"]).sum()
    print(f"Strong signals: {s_correct}/{len(strong_sig)} correct ({s_correct/len(strong_sig):.1%})")

mod_sig = signals[signals["strength_score"] == "Moderate"]
if len(mod_sig) > 0:
    m_correct = (mod_sig["pred_direction"] == mod_sig["actual_dir"]).sum()
    print(f"Moderate signals: {m_correct}/{len(mod_sig)} correct ({m_correct/len(mod_sig):.1%})")

weak_sig = signals[signals["strength_score"] == "Weak"]
if len(weak_sig) > 0:
    w_correct = (weak_sig["pred_direction"] == weak_sig["actual_dir"]).sum()
    print(f"Weak signals: {w_correct}/{len(weak_sig)} correct ({w_correct/len(weak_sig):.1%})")

print(f"Overall: {correct}/{total_sig} correct ({acc:.1%})")
