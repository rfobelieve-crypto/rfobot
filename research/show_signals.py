"""Show all signals from 3/25 to now with current model settings."""
import sys, logging
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, ".")

import pandas as pd
from indicator.data_fetcher import fetch_binance_klines, fetch_coinglass
from indicator.feature_builder_live import build_live_features
from indicator.inference import IndicatorEngine

klines = fetch_binance_klines(limit=500)
cg = fetch_coinglass(interval="1h", limit=500)
features = build_live_features(klines, cg)
engine = IndicatorEngine()
result = engine.predict(features)

start = pd.Timestamp("2026-03-25", tz="UTC")
r = result[result.index >= start].copy()

print(f"=== SIGNALS 3/25 ~ NOW (deadzone=0.50, strong>=80) ===")
print(f"Total bars: {len(r)}")

signals = r[r["pred_direction"].isin(["UP", "DOWN"])]
neutral = r[r["pred_direction"] == "NEUTRAL"]
strong = r[r["strength_score"] == "Strong"]
moderate = r[r["strength_score"] == "Moderate"]

print(f"Signal bars: {len(signals)} / {len(r)} ({len(signals)/len(r):.0%})")
print(f"  UP: {(signals['pred_direction']=='UP').sum()}  DOWN: {(signals['pred_direction']=='DOWN').sum()}  NEUTRAL: {len(neutral)}")
print(f"  Strong: {len(strong)}  Moderate: {len(moderate)}")
print()

header = f"{'Time (UTC)':20s}  {'Dir':8s}  {'Tier':10s}  {'Conf':>5s}  {'Up':>6s}  {'Down':>6s}  {'Diff':>7s}  Regime"
print(header)
print("-" * len(header))

prev_dir = None
for idx, row in r.iterrows():
    d = row["pred_direction"]
    s = row.get("strength_score", "")
    c = row.get("confidence_score", 0)
    up = row.get("up_pred", 0)
    down = row.get("down_pred", 0)
    diff = up - down
    regime = row.get("regime", "")

    if d == "NEUTRAL":
        prev_dir = d
        continue

    flip = ""
    if prev_dir and prev_dir != d and prev_dir != "NEUTRAL":
        flip = "  << FLIP"
    prev_dir = d

    print(f"{str(idx)[:19]}  {d:8s}  {s:10s}  {c:5.1f}  {up:6.3f}  {down:6.3f}  {diff:+7.3f}  {regime}{flip}")
