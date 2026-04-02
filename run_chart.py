"""
Quick launcher for the research chart.
Usage:
    python run_chart.py                         # BTC-USD 1h 7 days
    python run_chart.py ETH-USD 4h 14           # custom
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research.storage.schema import ensure_schema
from research.bar_generator.runner import run_once
from research.viz.chart_builder import load_and_build
from research.config.settings import ChartConfig

symbol        = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
timeframe     = sys.argv[2] if len(sys.argv) > 2 else "1h"
lookback_days = int(sys.argv[3]) if len(sys.argv) > 3 else 7

print(f"[1/3] Ensuring schema...")
ensure_schema()

print(f"[2/3] Computing bars: {symbol} {timeframe} last {lookback_days}d ...")
n = run_once(symbol, timeframe, lookback_days=lookback_days)
print(f"      → {n} new bars computed")

print(f"[3/3] Building chart...")
config = ChartConfig(symbol=symbol, timeframe=timeframe, lookback_days=lookback_days)
fig = load_and_build(symbol, timeframe, lookback_days=lookback_days, config=config)

out = f"chart_{symbol.replace('-','_')}_{timeframe}.html"
fig.write_html(out)
print(f"\nChart saved: {out}")
print(f"Open in browser: file://{os.path.abspath(out)}")
