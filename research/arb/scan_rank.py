# -*- coding: utf-8 -*-
"""§0.75b scanner ranking — the FROZEN promotion metric (2026-08-30).

Written before the scanner (../entropy-arb/tools/scanner.py) had produced
more than one cycle. It ranks every scanned pair by

    capturable_usd_per_day = fires_per_day x band_bps/1e4 x depth_usd

per side (sell A/buy B, buy A/sell B), keeps the larger side, where
  band    = p90 of the POSITIVE executable edge (analyze.py's methodology,
            identical to the recording family's gate — no sweep)
  fires   = samples with edge >= band, per day of scan span
  depth   = median over those fat samples of min(top-of-book USD on the two
            legs that the trade would hit)

Why money and not bps: SNDK's interim showed a pair can have a clean 5 bps
band on $200 books — cents per day. Ranking by bps would promote exactly
those.

Discipline (TODO §0.75b):
  * a pair is LISTED only after >= MIN_SPAN_DAYS of scanning and
    >= MIN_SAMPLES quotes — no one-cycle winners
  * listed < 48h on the Lighter leg -> excluded (book not built yet)
  * BTC pairs are always printed first as the CONTROL: their band is the
    instrument's noise floor. Anything whose band is within CONTROL_MULT x
    the BTC band is "not distinguishable from spread" and cannot be promoted
  * promotion = top PROMOTE_N by capturable_usd_per_day; the promoted pair's
    recording clock starts from ITS OWN first recorded minute, and the scan
    data that selected it never enters its verdict (selection window !=
    verification window)

Read-only. Output: research/results/arb_scan_rank.json
"""
from __future__ import annotations

import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SCAN = ROOT.parent / "entropy-arb" / "logs" / "scan"
OUT = ROOT / "research" / "results" / "arb_scan_rank.json"

MIN_SPAN_DAYS = 3.0
MIN_SAMPLES = 500
MIN_LISTED_H = 48.0
CONTROL_MULT = 2.0
PROMOTE_N = 3

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load() -> pd.DataFrame:
    files = sorted(glob.glob(str(SCAN / "scan_*.csv")))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.sort_values("ts").reset_index(drop=True)


def side_metric(g: pd.DataFrame, edge_col: str, depth_cols) -> dict:
    pos = g[g[edge_col] > 0]
    span_days = max((g.ts.max() - g.ts.min()) / 86400, 1e-9)
    if len(pos) < 10:
        return {"band_bps": None, "fires_per_day": 0.0, "depth_usd": None,
                "capturable_usd_per_day": 0.0}
    band = float(pos[edge_col].quantile(0.9))
    fat = g[g[edge_col] >= band]
    depth = float(fat[list(depth_cols)].min(axis=1).median()) if len(fat) else 0.0
    fires_per_day = len(fat) / span_days
    return {"band_bps": round(band, 3),
            "fires_per_day": round(fires_per_day, 1),
            "depth_usd": round(depth, 0),
            "capturable_usd_per_day": round(fires_per_day * band / 1e4 * depth, 2)}


def main() -> int:
    df = load()
    now = datetime.now(timezone.utc)
    print("§0.75b 掃描器排名（指標 2026-08-30 凍結：可捕獲美元／天 = 次數／天 × 帶寬 × 深度）")
    if df.empty:
        print("  尚無掃描資料")
        return 1
    span = (df.ts.max() - df.ts.min()) / 86400
    print(f"  掃描 {len(df):,} 筆報價｜{df.pair.nunique()} 個配對｜跨度 {span:.2f} 天"
          f"（名單門檻 ≥{MIN_SPAN_DAYS:.0f} 天、每配對 ≥{MIN_SAMPLES} 筆）")
    rows = []
    for pair, g in df.groupby("pair"):
        listed_h = None
        ca = g.b_created_at.dropna()
        if len(ca):
            try:
                listed_h = (now.timestamp() * 1000 - float(ca.iloc[-1])) / 3.6e6
            except Exception:
                listed_h = None
        sell = side_metric(g, "sell_edge_bps", ("a_bid_usd", "b_ask_usd"))
        buy = side_metric(g, "buy_edge_bps", ("b_bid_usd", "a_ask_usd"))
        best = max(sell, buy, key=lambda s: s["capturable_usd_per_day"])
        rows.append({"pair": pair, "n": int(len(g)),
                     "leg_b": g.leg_b.iloc[0], "listed_h": listed_h,
                     "sell": sell, "buy": buy,
                     "capturable_usd_per_day": best["capturable_usd_per_day"],
                     "band_bps": best["band_bps"], "depth_usd": best["depth_usd"],
                     "fires_per_day": best["fires_per_day"]})
    tab = pd.DataFrame(rows).sort_values("capturable_usd_per_day", ascending=False)

    ctrl = tab[tab.pair.str.startswith("BTC@")]
    ctrl_band = float(ctrl.band_bps.dropna().max()) if len(ctrl) else None
    print("\n  對照組（BTC，帶＝儀器的雜訊底）：")
    for _, r in ctrl.iterrows():
        print(f"    {r.pair:22s} band {r.band_bps} bps  depth ${r.depth_usd}  "
              f"≈ ${r.capturable_usd_per_day}/天")

    eligible = tab[(tab.n >= MIN_SAMPLES)
                   & ((tab.listed_h.isna()) | (tab.listed_h >= MIN_LISTED_H))
                   & (~tab.pair.str.startswith("BTC@"))]
    if ctrl_band is not None:
        eligible = eligible[eligible.band_bps > CONTROL_MULT * ctrl_band]
    gate_ok = span >= MIN_SPAN_DAYS

    print(f"\n  前 15（{'正式名單' if gate_ok else '期中觀察，跨度未達 — 不出名單'}）：")
    show = (eligible if gate_ok else tab[~tab.pair.str.startswith("BTC@")]).head(15)
    for _, r in show.iterrows():
        print(f"    {r.pair:22s} n={r.n:4d}  band {str(r.band_bps):>7s} bps  "
              f"{r.fires_per_day:6.1f}/天  depth ${str(r.depth_usd):>8s}  "
              f"≈ ${r.capturable_usd_per_day:>8}/天")
    promote = eligible.head(PROMOTE_N).pair.tolist() if gate_ok else []
    if gate_ok:
        print(f"\n  升格候選（前 {PROMOTE_N}）：{promote or '無'}")
        print("  升格後從該配對自己的首列起算 7 天；掃描期資料不進判決。")
    out = {"asof_utc": now.strftime("%Y-%m-%d %H:%M"), "span_days": round(span, 2),
           "quotes": int(len(df)), "pairs": int(df.pair.nunique()),
           "gate_ok": gate_ok, "control_band_bps": ctrl_band,
           "promote": promote,
           "top": tab.head(30).drop(columns=["sell", "buy"]).to_dict("records")}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str),
                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
