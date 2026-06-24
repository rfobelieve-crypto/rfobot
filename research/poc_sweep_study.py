"""POC × liquidity sweep continuation/reversal study (manual trading aid).

For each finished liquidity_events row:
  1. Build trailing 24h VPVR (POC, VAH, VAL) from 1h OHLCV bars
  2. Classify the sweep_ref_price into a zone vs that VPVR
  3. Aggregate continuation / reversal rates by zone × side × extra cuts

Why this is different from research/dual_model/resistance_map_phase1.py
(which concluded "V7 already absorbs the volume-profile resistance signal,
do NOT build a continuous feature on it"):
  · Phase 1 was a *continuous* feature fed into V7 — V7 absorbed it.
  · This is *event-conditional*: only inspects the subpopulation where a
    liquidity sweep already happened, and asks whether POC location shifts
    the continuation/reversal mix.  Not for model input — for the trader's
    eyes.

Outputs:
  · Section 1 — Zone × Side breakdown of cont% / rev% / neut% / avg_ret
  · Section 2 — Within near-POC events: CVD-alignment, session, sweep-size
    categorical cuts
  · Section 3 — Recent N near-POC events as study cases (open the event_time
    on TradingView and look at the chart pattern)

Read-only.  Requires research/dual_model/.cache/features_all.parquet
(gitignored — regenerate locally via research.dual_model.shared_data if
missing) and MySQL access for liquidity_events.

Usage:
    python research/poc_sweep_study.py
    python research/poc_sweep_study.py --lookback-h 4 --horizon 4h
    python research/poc_sweep_study.py --eps-pct 0.5 --recent-n 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.db import get_db_conn

FEATURES_PARQUET = (
    PROJECT_ROOT / "research" / "dual_model" / ".cache" / "features_all.parquet"
)


def load_ohlc() -> pd.DataFrame:
    if not FEATURES_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing {FEATURES_PARQUET}.\n"
            "Regenerate locally via research.dual_model.shared_data."
        )
    df = pd.read_parquet(FEATURES_PARQUET)
    cols = ["open", "high", "low", "close"]
    if "quote_vol" in df.columns:
        cols.append("quote_vol")
    elif "volume" in df.columns:
        cols.append("volume")
    df = df[cols].copy()
    if "quote_vol" not in df.columns:
        df["quote_vol"] = df["volume"].astype(float) * df["close"].astype(float)
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


def fetch_events() -> pd.DataFrame:
    sql = """
    SELECT id, event_uuid, event_time, trigger_ts, symbol, liquidity_side,
           entry_price, sweep_ref_price, sweep_size_pct,
           delta_15m, delta_1h, delta_4h,
           return_15m, return_1h, return_4h,
           first_hit_side,
           session, result_1h, result_4h
    FROM liquidity_events
    WHERE result_1h IS NOT NULL OR result_4h IS NOT NULL
    ORDER BY trigger_ts
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    finally:
        conn.close()
    return pd.DataFrame(rows)


def build_vpvr(win: pd.DataFrame, n_buckets: int = 200) -> tuple[float, float, float, float]:
    """Volume-at-price profile over the window.

    Each bar's notional (quote_vol) is distributed uniformly across [low, high].
    Returns (POC, VAH, VAL, total_volume).  NaN if the window is too thin.
    """
    if len(win) < 5:
        return (np.nan, np.nan, np.nan, 0.0)
    p_lo = float(win["low"].min())
    p_hi = float(win["high"].max())
    if p_hi <= p_lo:
        return (np.nan, np.nan, np.nan, 0.0)

    edges = np.linspace(p_lo, p_hi, n_buckets + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bucket_size = (p_hi - p_lo) / n_buckets

    volume = np.zeros(n_buckets, dtype=float)
    lows = win["low"].values.astype(float)
    highs = win["high"].values.astype(float)
    qvs = win["quote_vol"].values.astype(float)

    for lo, hi, qv in zip(lows, highs, qvs):
        if hi <= lo or qv <= 0:
            continue
        i_lo = max(0, int((lo - p_lo) / bucket_size))
        i_hi = min(n_buckets - 1, int((hi - p_lo) / bucket_size))
        n_bk = i_hi - i_lo + 1
        if n_bk <= 0:
            continue
        volume[i_lo:i_hi + 1] += qv / n_bk

    total = float(volume.sum())
    if total <= 0:
        return (np.nan, np.nan, np.nan, 0.0)
    poc_idx = int(np.argmax(volume))
    poc = float(centers[poc_idx])

    # 70% value-area envelope around POC
    target = 0.70 * total
    cumul = volume[poc_idx]
    lo_idx = hi_idx = poc_idx
    while cumul < target and (lo_idx > 0 or hi_idx < n_buckets - 1):
        next_lo = volume[lo_idx - 1] if lo_idx > 0 else -1.0
        next_hi = volume[hi_idx + 1] if hi_idx < n_buckets - 1 else -1.0
        if next_lo >= next_hi:
            lo_idx -= 1
            cumul += volume[lo_idx]
        else:
            hi_idx += 1
            cumul += volume[hi_idx]
    return (poc, float(centers[hi_idx]), float(centers[lo_idx]), total)


def classify_zone(price: float, poc: float, vah: float, val: float,
                  eps_pct: float = 0.3) -> str:
    """5-way categorical bucket: above_VAH / in_VA_upper / at_POC /
    in_VA_lower / below_VAL.

    eps_pct = how close to POC (in % of POC) counts as at_POC.
    """
    if any(np.isnan(x) for x in (poc, vah, val)):
        return "unknown"
    eps = poc * eps_pct / 100.0
    if abs(price - poc) <= eps:
        return "at_POC"
    if price > vah:
        return "above_VAH"
    if price < val:
        return "below_VAL"
    if price > poc:
        return "in_VA_upper"
    return "in_VA_lower"


def _to_float(x) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def summarize_group(g: pd.DataFrame, res_col: str, ret_col: str) -> dict:
    return {
        "n": len(g),
        "cont_pct": round((g[res_col] == "continuation").mean() * 100, 1),
        "rev_pct": round((g[res_col] == "reversal").mean() * 100, 1),
        "neut_pct": round((g[res_col] == "neutral").mean() * 100, 1),
        "avg_ret_pct": (round(float(g[ret_col].mean()), 3)
                        if g[ret_col].notna().any() else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback-h", type=int, default=24,
                    help="VPVR trailing window in hours (default 24)")
    ap.add_argument("--eps-pct", type=float, default=0.3,
                    help="|price - POC| <= eps%% × POC counts as at_POC")
    ap.add_argument("--horizon", choices=["1h", "4h"], default="1h",
                    help="Outcome horizon (default 1h)")
    ap.add_argument("--recent-n", type=int, default=20,
                    help="Sample events to list in Section 3 (default 20)")
    args = ap.parse_args()

    print(f"Loading OHLC from {FEATURES_PARQUET.name} …")
    ohlc = load_ohlc()
    print(f"  shape: {ohlc.shape},  range {ohlc.index[0]} → {ohlc.index[-1]}")

    print("Fetching liquidity_events from MySQL …")
    events = fetch_events()
    print(f"  {len(events)} finished events")
    if len(events) == 0:
        print("No events. Abort.")
        return 1

    # ── compute VPVR + zone per event
    print(f"Building VPVR ({args.lookback_h}h trailing) per event …")
    rows = []
    skipped = 0
    for _, ev in events.iterrows():
        ts_raw = ev["trigger_ts"]
        if ts_raw is None:
            skipped += 1
            continue
        ts = pd.to_datetime(int(ts_raw), unit="s", utc=True).tz_convert(None)
        window_end = ts
        window_start = ts - pd.Timedelta(hours=args.lookback_h)
        win = ohlc.loc[(ohlc.index > window_start) & (ohlc.index <= window_end)]
        if len(win) < 5:
            skipped += 1
            continue
        poc, vah, val, _ = build_vpvr(win)
        sweep_ref = _to_float(ev["sweep_ref_price"]) or _to_float(ev["entry_price"])
        if sweep_ref is None:
            skipped += 1
            continue
        zone = classify_zone(sweep_ref, poc, vah, val, eps_pct=args.eps_pct)
        delta_15m = _to_float(ev["delta_15m"]) or 0.0
        side_raw = ev["liquidity_side"] or ""
        side_label = "BSL" if side_raw == "buy" else ("SSL" if side_raw == "sell" else "?")
        # CVD aligned with sweep direction = a continuation expectation:
        #   BSL (buy-side sweep, expects upside continuation) → delta_15m > 0 = aligned
        #   SSL (sell-side sweep, expects downside continuation) → delta_15m < 0 = aligned
        cvd_aligned = ((side_raw == "buy" and delta_15m > 0)
                       or (side_raw == "sell" and delta_15m < 0))
        rows.append({
            "id": int(ev["id"]),
            "event_time": ev["event_time"],
            "trigger_ts": int(ts_raw),
            "side": side_label,
            "sweep_ref": round(sweep_ref, 2),
            "sweep_size_pct": _to_float(ev["sweep_size_pct"]) or 0.0,
            "POC": round(poc, 2) if not np.isnan(poc) else None,
            "VAH": round(vah, 2) if not np.isnan(vah) else None,
            "VAL": round(val, 2) if not np.isnan(val) else None,
            "dist_to_POC_pct": (round((sweep_ref - poc) / poc * 100, 3)
                                if not np.isnan(poc) else None),
            "zone": zone,
            "delta_15m": delta_15m,
            "delta_15m_sign": "+" if delta_15m > 0 else ("-" if delta_15m < 0 else "0"),
            "cvd_aligned": cvd_aligned,
            "session": ev["session"],
            "result_1h": ev["result_1h"],
            "result_4h": ev["result_4h"],
            "return_1h": _to_float(ev["return_1h"]),
            "return_4h": _to_float(ev["return_4h"]),
        })
    print(f"  {len(rows)} events with valid POC, skipped {skipped}")
    if not rows:
        return 1
    df = pd.DataFrame(rows)

    res_col = f"result_{args.horizon}"
    ret_col = f"return_{args.horizon}"
    df = df.dropna(subset=[res_col])
    print(f"  {len(df)} events have {res_col} populated")

    if len(df) < 30:
        print(f"\n⚠️  Sample size {len(df)} is low — cell-level stats will be noisy.")

    # ── Section 1: Zone × Side breakdown
    print("\n" + "=" * 92)
    print(f"SECTION 1 — Zone × Side outcome breakdown ({args.horizon} horizon)")
    print("=" * 92)
    sec1_rows = []
    for (zone, side), g in df.groupby(["zone", "side"], sort=False):
        sec1_rows.append({"zone": zone, "side": side, **summarize_group(g, res_col, ret_col)})
    overview = pd.DataFrame(sec1_rows).sort_values(["zone", "side"])
    print(overview.to_string(index=False))

    baseline = summarize_group(df, res_col, ret_col)
    print(f"\nBaseline (all {len(df)} events): "
          f"cont={baseline['cont_pct']}%  rev={baseline['rev_pct']}%  "
          f"neut={baseline['neut_pct']}%  avg_ret={baseline['avg_ret_pct']}%")

    # ── Section 2: near-POC subset, conditional cuts
    near = df[df["zone"].isin(["at_POC", "in_VA_upper", "in_VA_lower"])].copy()
    print("\n" + "=" * 92)
    print(f"SECTION 2 — Near-POC subset (n={len(near)}), conditional cuts "
          f"({args.horizon} horizon)")
    print("=" * 92)
    if len(near) >= 20:
        print("\n--- CVD alignment with sweep direction ---")
        cut_rows = []
        for v, g in near.groupby("cvd_aligned"):
            cut_rows.append({
                "cvd_aligned": bool(v),
                **summarize_group(g, res_col, ret_col),
            })
        print(pd.DataFrame(cut_rows).to_string(index=False))

        print("\n--- Session ---")
        cut_rows = []
        for s, g in near.groupby("session", sort=False):
            cut_rows.append({"session": s, **summarize_group(g, res_col, ret_col)})
        print(pd.DataFrame(cut_rows).to_string(index=False))

        near["sweep_size_bucket"] = pd.cut(
            near["sweep_size_pct"], bins=[-1e-9, 0.4, 0.8, 1000.0],
            labels=["small(<0.4%)", "mid(0.4-0.8%)", "big(>0.8%)"],
        )
        print("\n--- Sweep size bucket ---")
        cut_rows = []
        for b, g in near.groupby("sweep_size_bucket", observed=True):
            cut_rows.append({"sweep_size_bucket": str(b),
                             **summarize_group(g, res_col, ret_col)})
        print(pd.DataFrame(cut_rows).to_string(index=False))

        # Compound: CVD-alignment × sweep-size — the most actionable view
        print("\n--- Compound: CVD-alignment × sweep-size ---")
        cut_rows = []
        for (a, b), g in near.groupby(["cvd_aligned", "sweep_size_bucket"],
                                       observed=True):
            cut_rows.append({"cvd_aligned": bool(a), "sweep_size": str(b),
                             **summarize_group(g, res_col, ret_col)})
        print(pd.DataFrame(cut_rows).to_string(index=False))
    else:
        print(f"  insufficient near-POC samples ({len(near)} < 20) "
              f"for conditional cuts")

    # ── Section 3: recent N near-POC events
    print("\n" + "=" * 92)
    print(f"SECTION 3 — Recent {args.recent_n} near-POC events "
          f"(open these on TradingView)")
    print("=" * 92)
    recent = near.sort_values("trigger_ts", ascending=False).head(args.recent_n)
    show_cols = ["event_time", "side", "sweep_ref", "POC", "dist_to_POC_pct",
                 "zone", "sweep_size_pct", "delta_15m_sign", "cvd_aligned",
                 "session", res_col, ret_col]
    print(recent[show_cols].to_string(index=False))

    # ── Save full enriched table
    out_csv = PROJECT_ROOT / "research" / "results" / "poc_sweep_study.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nWrote full enriched event table → {out_csv}")

    print("\n--- HOW TO READ ---")
    print("  · Section 1: if a zone's cont% is well above the baseline cont%, ")
    print("    that zone has a continuation bias when a sweep happens there.")
    print("  · Section 2 'CVD alignment': aligned=True means the 15m flow is")
    print("    pushing in the same direction as the sweep — sanity check on ")
    print("    whether the move has follow-through demand or just stop hunt.")
    print("  · Section 2 'Compound': the cell with the highest n AND highest")
    print("    cont% delta vs baseline is your strongest categorical edge.")
    print("  · Section 3: take the event_time, paste into TradingView UTC+8 ")
    print("    and study the price/volume context. Pattern-match by eye —")
    print("    this is the part the data cannot do for you.")
    print("\n  ⚠️  This is a manual-trading aid, NOT a model input. The prior")
    print("       VPOC-as-continuous-feature study (resistance_map_phase1.py)")
    print("       found V7 already absorbs that signal — do not re-feed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
