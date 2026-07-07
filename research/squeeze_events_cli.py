"""CLI for squeeze_events.py — offline event-table generation.

Adapter side: reuses the project's EXISTING historical OHLCV layer
(market_data/raw_data/binance_klines_1h.parquet — the same file every
research/* backtest loads; see e.g. research/v71_exit_variants.py:177).
No new downloader is introduced; symbols/timeframes without an existing
parquet raise a clear error instead of fetching.

Usage:
    python research/squeeze_events_cli.py \
        --symbol BTC-USD --timeframe 1h \
        --start 2026-01-01 --end 2026-06-30 \
        --out research/results/squeeze_events.parquet

    # SqueezeConfig overrides (all optional):
    python research/squeeze_events_cli.py --entry-mode close \
        --sl-anchor entry --tp-mults 1.5,3,5 --sl-mult 2.0 \
        --range-period 20 --max-open 1 --session-utc 8,20

Read-only with respect to everything except the --out parquet.
Not wired into any live/scheduled path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.squeeze_events import (
    EntryMode, SLAnchor, SqueezeConfig, TPGeometry, detect_events,
)

# Existing historical-bar sources, keyed by (canonical symbol, timeframe).
# Extend ONLY when a corresponding parquet already exists in the repo's
# data layer — this table maps, it never downloads.
_SOURCES: dict[tuple[str, str], Path] = {
    ("BTC-USD", "1h"): PROJECT_ROOT / "market_data" / "raw_data"
                        / "binance_klines_1h.parquet",
}


def load_ohlcv(symbol: str, timeframe: str,
               start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Adapter: existing parquet layer → squeeze_events input schema.

    Output contract (per squeeze_events.detect_events docstring):
    UTC DatetimeIndex + float open/high/low/close columns.
    """
    key = (symbol.upper(), timeframe.lower())
    path = _SOURCES.get(key)
    if path is None:
        known = ", ".join(f"{s}/{t}" for s, t in _SOURCES)
        raise SystemExit(
            f"No existing OHLCV source for {symbol}/{timeframe}. "
            f"Available: {known}. This CLI does not download data."
        )
    if not path.exists():
        raise SystemExit(
            f"Source parquet missing: {path}\n"
            "Regenerate locally via research/backfill_all_parquet.py "
            "(run on the machine that holds the data layer)."
        )

    df = pd.read_parquet(path)[["open", "high", "low", "close"]].dropna()
    df = df.astype(float)

    idx = pd.DatetimeIndex(df.index)
    # Repo convention stores naive-UTC; module contract wants explicit UTC.
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df.index = idx
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if start:
        df = df.loc[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df.loc[df.index <= pd.Timestamp(end, tz="UTC")]
    return df


def _parse_tp_mults(s: str) -> tuple[float, float, float]:
    parts = [float(x) for x in s.split(",") if x.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--tp-mults needs exactly 3 values, e.g. 2,3.5,6")
    return tuple(parts)  # type: ignore[return-value]


def _parse_session(s: str) -> tuple[int, int]:
    parts = [int(x) for x in s.split(",") if x.strip()]
    if len(parts) != 2 or not all(0 <= p <= 23 for p in parts):
        raise argparse.ArgumentTypeError("--session-utc needs start,end hours 0-23, e.g. 8,20")
    return (parts[0], parts[1])


def build_config(args: argparse.Namespace) -> SqueezeConfig:
    kwargs: dict = {}
    if args.range_period is not None:
        kwargs["range_period"] = args.range_period
    if args.atr_period is not None:
        kwargs["atr_period"] = args.atr_period
    if args.range_multiplier is not None:
        kwargs["range_multiplier"] = args.range_multiplier
    if args.entry_mode is not None:
        kwargs["entry_mode"] = EntryMode(args.entry_mode)
    if args.retest_expiry is not None:
        kwargs["retest_expiry"] = args.retest_expiry
    if args.sl_anchor is not None:
        kwargs["sl_anchor"] = SLAnchor(args.sl_anchor)
    if args.tp_geometry is not None:
        kwargs["tp_geometry"] = TPGeometry(args.tp_geometry)
    if args.tp_mults is not None:
        kwargs["tp_mults"] = args.tp_mults
    if args.sl_mult is not None:
        kwargs["sl_mult"] = args.sl_mult
    if args.horizon_bars is not None:
        kwargs["horizon_bars"] = args.horizon_bars
    if args.min_range_bars is not None:
        kwargs["min_range_bars"] = args.min_range_bars
    if args.session_utc is not None:
        kwargs["session_utc"] = args.session_utc
    if args.max_open is not None:
        kwargs["max_open"] = None if args.max_open <= 0 else args.max_open
    return SqueezeConfig(**kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbol", default="BTC-USD")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--out", default=str(
        PROJECT_ROOT / "research" / "results" / "squeeze_events.parquet"))
    # SqueezeConfig overrides
    ap.add_argument("--range-period", type=int, default=None)
    ap.add_argument("--atr-period", type=int, default=None)
    ap.add_argument("--range-multiplier", type=float, default=None)
    ap.add_argument("--entry-mode", choices=[m.value for m in EntryMode],
                    default=None)
    ap.add_argument("--retest-expiry", type=int, default=None)
    ap.add_argument("--sl-anchor", choices=[a.value for a in SLAnchor],
                    default=None)
    ap.add_argument("--tp-geometry", choices=[g.value for g in TPGeometry],
                    default=None)
    ap.add_argument("--tp-mults", type=_parse_tp_mults, default=None,
                    help="3 comma-separated multipliers, e.g. 2,3.5,6")
    ap.add_argument("--sl-mult", type=float, default=None)
    ap.add_argument("--horizon-bars", type=int, default=None)
    ap.add_argument("--min-range-bars", type=int, default=None)
    ap.add_argument("--session-utc", type=_parse_session, default=None,
                    help="UTC hour half-open window, e.g. 8,20 (or 22,4 overnight)")
    ap.add_argument("--max-open", type=int, default=None,
                    help="Concurrent-event cap; 0 or negative = unlimited")
    args = ap.parse_args()

    cfg = build_config(args)
    print(f"Loading {args.symbol} {args.timeframe} "
          f"[{args.start or 'begin'} → {args.end or 'end'}] …")
    ohlcv = load_ohlcv(args.symbol, args.timeframe, args.start, args.end)
    print(f"  {len(ohlcv)} bars  {ohlcv.index[0]} → {ohlcv.index[-1]}")
    print(f"Config: {cfg}")

    events = detect_events(ohlcv, cfg)
    if events.empty:
        print("No events detected over this window.")
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(out_path, index=False)

    n_fill = int(events["filled"].sum())
    n_res = int(events["resolved"].sum())
    print(f"\n{len(events)} events  (filled {n_fill}, resolved {n_res}, "
          f"cancelled {int(events['cancel_reason'].notna().sum())})")
    if n_res:
        res = events[events["resolved"]]
        print(f"  sl_first rate : {res['sl_first'].mean()*100:.1f}%")
        print(f"  avg r_scaleout: {res['r_scaleout'].mean():+.3f} R")
        print(f"  avg mfe/mae   : {res['mfe_R'].mean():.2f} / {res['mae_R'].mean():.2f} R")
    print(f"Wrote → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
