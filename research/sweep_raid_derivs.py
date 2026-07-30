# -*- coding: utf-8 -*-
"""Raid anatomy, round 2 — EVERY derivatives/state series at the raid moment.

User request (2026-07-30): 把所有能用的訂單流數據都加入判斷 — funding, OI,
CVD, liquidations, positioning, premium, DVOL. The horizon-decay result says
slow STATE variables are the only flow form that survives to hourly scale,
so state-at-the-event is the theoretically right place to look (unlike the
nine dead unconditional-IC hunts).

Scope: BTC only (the Coinglass series are BTC), 2025-10-22 -> now (~280d),
every raid of the four pool types, no position chain. Targets, same as
sweep_raid_anatomy: (a) resolution — BREAKOUT vs retest; (b) netR when
retested.

FEATURES (14, signed conventions documented; s=+1 means a HIGH was raided,
so "signed" = value x s = pressure IN the break direction):
  funding_signed   s*funding      crowding pays in the break direction
  funding_abs      |funding|      stress level regardless of side
  oi_chg_raid      OI %chg during the raid hour — the mechanism split:
                   OI DOWN = positions closing (stops = 獵殺 fuel);
                   OI UP = new positioning entering (breakout fuel)
  oi_chg_4h/24h    OI %chg over 4h/24h before the raid
  fut_taker_signed s*(futB-futS)/(futB+futS) in the raid hour (futures)
  spot_taker_signed same from spot CVD series
  spot_fut_div     spot_taker_signed - fut_taker_signed (spot leads?)
  premium_signed   s*coinbase premium_rate
  liq_burst        raid-hour liq total / trailing-24h mean
  stop_fuel        share of raid-hour liqs on the HUNTED side (raid a high
                   -> shorts get stopped; high = visible stop-run)
  top_ls_signed    s*(top_position_ls_ratio - 1)   whale tilt into break
  glob_ls_signed   s*(global_account_ls_ratio - 1) crowd tilt into break
  dvol_pct         DVOL percentile vs trailing 30d

MULTIPLE-COMPARISON BANNER, read before believing anything: 14 features x 2
targets = 28 looks; pure chance yields ~9 monotone patterns. Monotonicity
alone is NOISE here. The bar: monotone AND first/second-half consistent AND
material magnitude AND mechanically sensible. Descriptive; no registration
is touched.

Run: python research/sweep_raid_derivs.py
Out: research/results/sweep_raid_derivs.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import pandas as pd  # noqa: E402
import sweep_raid_anatomy as A  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/sweep_raid_derivs.json"
RAW = ROOT / "market_data/raw_data"


def hour_map(df: pd.DataFrame, col) -> dict[int, float]:
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    vals = df[col].astype(float).values
    return {int(t.value // 10**9) // 3600: float(v)
            for t, v in zip(idx, vals) if v == v}


def load_state():
    S = {}
    S["funding"] = hour_map(pd.read_parquet(RAW / "cg_funding_1h.parquet"), "close")
    oi = hour_map(pd.read_parquet(RAW / "cg_oi_agg_1h.parquet"), "close")
    S["oi"] = oi
    f = pd.read_parquet(RAW / "cg_futures_cvd_agg_1h.parquet")
    S["fut_b"] = hour_map(f, "agg_taker_buy_vol")
    S["fut_s"] = hour_map(f, "agg_taker_sell_vol")
    sp = pd.read_parquet(RAW / "cg_spot_cvd_agg_1h.parquet")
    S["spot_b"] = hour_map(sp, "agg_taker_buy_vol")
    S["spot_s"] = hour_map(sp, "agg_taker_sell_vol")
    S["prem"] = hour_map(pd.read_parquet(RAW / "cg_coinbase_premium_1h.parquet"),
                         "premium_rate")
    lq = pd.read_parquet(RAW / "cg_liq_agg_1h.parquet")
    S["liq_l"] = hour_map(lq, "aggregated_long_liquidation_usd")
    S["liq_s"] = hour_map(lq, "aggregated_short_liquidation_usd")
    S["top_ls"] = hour_map(pd.read_parquet(RAW / "cg_top_ls_position_1h.parquet"),
                           "top_position_long_short_ratio")
    S["glob_ls"] = hour_map(pd.read_parquet(RAW / "cg_global_ls_1h.parquet"),
                            "global_account_long_short_ratio")
    S["dvol"] = hour_map(pd.read_parquet(RAW / "deribit_dvol_1h.parquet"),
                         "dvol_close")
    return S


def attach(rows, S):
    out = []
    for r in rows:
        hh = r["ts"] // 3600
        s = r["side"]
        if hh not in S["funding"] or hh not in S["oi"]:
            continue
        f = dict(r)
        fu = S["funding"][hh]
        f["funding_signed"] = s * fu
        f["funding_abs"] = abs(fu)
        oi0, oi1 = S["oi"].get(hh - 1), S["oi"].get(hh)
        oi4, oi24 = S["oi"].get(hh - 4), S["oi"].get(hh - 24)
        f["oi_chg_raid"] = (oi1 / oi0 - 1) * 100 if oi0 and oi1 else None
        f["oi_chg_4h"] = (oi1 / oi4 - 1) * 100 if oi4 and oi1 else None
        f["oi_chg_24h"] = (oi1 / oi24 - 1) * 100 if oi24 and oi1 else None
        fb, fs = S["fut_b"].get(hh), S["fut_s"].get(hh)
        f["fut_taker_signed"] = (s * (fb - fs) / (fb + fs)
                                 if fb is not None and fs and (fb + fs) > 0 else None)
        sb, ss = S["spot_b"].get(hh), S["spot_s"].get(hh)
        f["spot_taker_signed"] = (s * (sb - ss) / (sb + ss)
                                  if sb is not None and ss and (sb + ss) > 0 else None)
        if f["fut_taker_signed"] is not None and f["spot_taker_signed"] is not None:
            f["spot_fut_div"] = f["spot_taker_signed"] - f["fut_taker_signed"]
        else:
            f["spot_fut_div"] = None
        pr = S["prem"].get(hh)
        f["premium_signed"] = s * pr if pr is not None else None
        ll, ls_ = S["liq_l"].get(hh), S["liq_s"].get(hh)
        base = [S["liq_l"].get(hh - k, 0) + S["liq_s"].get(hh - k, 0)
                for k in range(1, 25)]
        base = [b for b in base if b > 0]
        if ll is not None and ls_ is not None:
            tot = ll + ls_
            f["liq_burst"] = tot / (sum(base) / len(base)) if base else None
            hunted = ls_ if s == 1 else ll
            f["stop_fuel"] = hunted / tot if tot > 0 else None
        else:
            f["liq_burst"] = f["stop_fuel"] = None
        tl = S["top_ls"].get(hh)
        f["top_ls_signed"] = s * (tl - 1) if tl is not None else None
        gl = S["glob_ls"].get(hh)
        f["glob_ls_signed"] = s * (gl - 1) if gl is not None else None
        dv = S["dvol"].get(hh)
        dwin = [S["dvol"][hh - k] for k in range(1, 720) if hh - k in S["dvol"]]
        f["dvol_pct"] = (100 * sum(1 for x in dwin if x < dv) / len(dwin)
                         if dv is not None and len(dwin) > 300 else None)
        out.append(f)
    return out


FEATS = ["funding_signed", "funding_abs", "oi_chg_raid", "oi_chg_4h",
         "oi_chg_24h", "fut_taker_signed", "spot_taker_signed", "spot_fut_div",
         "premium_signed", "liq_burst", "stop_fuel", "top_ls_signed",
         "glob_ls_signed", "dvol_pct"]


def main() -> int:
    print("=" * 78)
    print("  RAID x DERIVATIVES STATE — 14 features, BTC, every raid")
    print("  (28 looks -> ~9 chance monotones; bar = monotone + halves + "
          "magnitude + mechanism)")
    print("=" * 78)
    S = load_state()
    flow, imb, canc = A.load_flow("BTC-USD")
    rows = attach(A.raids("BTC"), S)
    n = len(rows)
    br = sum(1 for r in rows if r["cls"] == "BREAKOUT")
    print(f"  BTC raids with CG coverage: {n}  (breakout base rate "
          f"{100*br/n:.0f}%)\n")

    res = {}
    print("  [terciles] 突破率 / 反轉率 / netR|回踩")
    for k in FEATS:
        rec, line = A.profile(rows, k)
        res[k] = rec
        print(line)

    # halves consistency for every feature whose breakout% OR netR is monotone
    def mono(rec, field):
        if not rec:
            return False
        v = [rec[b][field] for b in ("low", "mid", "high")]
        return (v[0] <= v[1] <= v[2]) or (v[0] >= v[1] >= v[2])

    cands = [k for k in FEATS if res.get(k)
             and (mono(res[k], "breakout_pct") or mono(res[k], "netR_if_retested"))]
    print(f"\n  monotone candidates ({len(cands)}): {', '.join(cands)}")
    rows_sorted = sorted(rows, key=lambda r: r["ts"])
    half = len(rows_sorted) // 2
    print("  [halves] 前半 vs 後半（單調方向要同向才算）")
    for k in cands:
        for tag, seg in (("H1", rows_sorted[:half]), ("H2", rows_sorted[half:])):
            _, line = A.profile(seg, k)
            print(f"  {tag} " + line)

    OUT.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
