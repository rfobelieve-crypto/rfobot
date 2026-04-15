"""
One-shot: rebuild chart PNG with the new Direction-Reg model using cached
features. Lets us eyeball the triangle distribution without waiting for
auto_update's next cycle.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from research.dual_model.shared_data import load_and_cache_data
from indicator.inference import IndicatorEngine
from indicator.chart_renderer import render_chart


def main():
    df = load_and_cache_data()
    print(f"[data] cached features: {len(df)} bars  "
          f"[{df.index[0]} -> {df.index[-1]}]")

    engine = IndicatorEngine()
    print(f"[engine] mode={engine.mode}  "
          f"dir_type={getattr(engine, 'dir_model_type', 'n/a')}")

    # Predict over the full history so rolling percentile cutoffs stabilise,
    # then slice last 200 for the chart.
    pred = engine.predict(df, context_features=df)
    print(f"[predict] rows: {len(pred)}   "
          f"Strong: {(pred['strength_score'] == 'Strong').sum()}  "
          f"Moderate: {(pred['strength_score'] == 'Moderate').sum()}")

    chart_df = pred.dropna(subset=["open", "high", "low", "close"])
    png = render_chart(chart_df, last_n=200)

    out_path = ROOT / "research" / "results" / "fresh_chart_direction_reg.png"
    out_path.write_bytes(png)
    print(f"[chart] wrote {out_path}  ({len(png)} bytes)")

    # Summary of last-200 window
    tail = chart_df.tail(200)
    print("\n  last-200 window triangle summary:")
    print(f"    Strong UP   : {((tail.strength_score=='Strong') & (tail.pred_direction=='UP')).sum()}")
    print(f"    Strong DOWN : {((tail.strength_score=='Strong') & (tail.pred_direction=='DOWN')).sum()}")
    print(f"    Moderate UP : {((tail.strength_score=='Moderate') & (tail.pred_direction=='UP')).sum()}")
    print(f"    Moderate DN : {((tail.strength_score=='Moderate') & (tail.pred_direction=='DOWN')).sum()}")


if __name__ == "__main__":
    main()
