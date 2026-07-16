"""Cancel-flow window analyzer — the forensic tool behind "叫 agent 分析".

Prints a minute-by-minute forensic table for any window (or the live right
edge): price/volume/taker delta + frozen v1 features (shock/毛偏斜/淨偏斜)
+ per-side gross & net flows, optional perp-book comparison, recorded
playbook events (with outcomes when已回填), and L20 wall snapshots.

Feature definitions are IMPORTED from market_data.tasks.cancel_playbook_watcher
(single source of truth — this file never redefines them).

Usage (times are TPE, UTC+8):
    python research/cancel_flow_analyze.py                      # 最近 90 分鐘
    python research/cancel_flow_analyze.py --mins 240
    python research/cancel_flow_analyze.py --from "2026-07-16 20:00" --to "2026-07-16 21:10"
    python research/cancel_flow_analyze.py --from "..." --to "..." --perp
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from shared.db import get_db_conn
from market_data.tasks.cancel_playbook_watcher import (
    compute_features, ZH, DEF_VERSION)

TPE = pd.Timedelta(hours=8)
WARMUP_MIN = 90   # feature rolling windows need history before the window


def _q(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params or None)
        return pd.DataFrame(cur.fetchall() or [])


def load_window(t0_ms: int, t1_ms: int, exchange: str = "binance"):
    lo = t0_ms - WARMUP_MIN * 60_000
    conn = get_db_conn()
    try:
        dd = _q(conn, "SELECT minute_start_ms ms, bid_add_qty ba, "
                      "bid_cancel_qty bc, ask_add_qty aa, ask_cancel_qty ac "
                      "FROM depth_deltas_1m WHERE canonical_symbol='BTC-USD' "
                      "AND exchange=%s AND minute_start_ms BETWEEN %s AND %s "
                      "ORDER BY minute_start_ms", (exchange, lo, t1_ms))
        fb = _q(conn, "SELECT window_start ms, volume_usd vol, delta_usd dlt "
                      "FROM flow_bars_1m WHERE canonical_symbol='BTC-USD' "
                      "AND exchange_scope='all' AND window_start BETWEEN %s AND %s "
                      "ORDER BY window_start", (lo, t1_ms))
        ob = _q(conn, "SELECT ts_ms, mid_price mid, bid_depth_usd_l20 bd, "
                      "ask_depth_usd_l20 ad, imbalance_l20 imb "
                      "FROM orderbook_snapshots_1m "
                      "WHERE canonical_symbol='BTC-USD' AND ts_ms BETWEEN %s AND %s "
                      "ORDER BY ts_ms", (lo, t1_ms))
        ev = _q(conn, "SELECT minute_start_ms ms, playbook, direction, px, shock, "
                      "skew15, net15, vshock, taker_ratio, fwd_ret_30m, "
                      "fwd_ret_60m, fwd_ret_120m, hit_60m, alerted "
                      "FROM cancel_playbook_events WHERE def_version=%s "
                      "AND minute_start_ms BETWEEN %s AND %s "
                      "ORDER BY minute_start_ms", (DEF_VERSION, t0_ms, t1_ms))
    finally:
        conn.close()
    if dd.empty:
        return pd.DataFrame(), ev
    for f in (dd, fb, ob):
        for c in f.columns:
            f[c] = pd.to_numeric(f[c])
    dd["m"] = dd["ms"] // 60_000
    df = dd.groupby("m").last()[["ba", "bc", "aa", "ac"]]
    if not fb.empty:
        fb["m"] = fb["ms"] // 60_000
        df = df.join(fb.groupby("m").last()[["vol", "dlt"]], how="left")
    else:
        df["vol"] = np.nan; df["dlt"] = np.nan
    if not ob.empty:
        ob["m"] = ob["ts_ms"] // 60_000
        df = df.join(ob.groupby("m").last()[["mid", "bd", "ad", "imb"]], how="left")
    else:
        df["mid"] = np.nan; df["bd"] = np.nan; df["ad"] = np.nan; df["imb"] = np.nan
    df["mid"] = df["mid"].ffill()
    return df, ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mins", type=int, default=90, help="最近 N 分鐘")
    ap.add_argument("--from", dest="t_from", help='TPE "YYYY-MM-DD HH:MM"')
    ap.add_argument("--to", dest="t_to", help='TPE "YYYY-MM-DD HH:MM"')
    ap.add_argument("--perp", action="store_true", help="附 perp 簿對照欄")
    args = ap.parse_args()

    if args.t_from and args.t_to:
        t0 = int((pd.Timestamp(args.t_from) - TPE).timestamp() * 1000)
        t1 = int((pd.Timestamp(args.t_to) - TPE).timestamp() * 1000)
    else:
        t1 = int(time.time() * 1000)
        t0 = t1 - args.mins * 60_000

    df, ev = load_window(t0, t1)
    if df.empty:
        print("視窗內無 depth 資料（collector 2026-07-09 起）")
        return 1
    feat = compute_features(df)
    vbase = feat["vol"].rolling(60, min_periods=30).median()
    feat["vsk"] = feat["vol"] / vbase.replace(0, np.nan)

    pfeat = None
    if args.perp:
        pdf, _ = load_window(t0, t1, exchange="binance_perp")
        if not pdf.empty:
            pfeat = compute_features(pdf)

    m0, m1 = t0 // 60_000, t1 // 60_000
    view = feat.loc[(feat.index >= m0) & (feat.index <= m1)]
    if view.empty:
        print("視窗內無資料")
        return 1

    last_t = pd.Timestamp(int(view.index.max()) * 60, unit="s") + TPE
    now_t = pd.Timestamp(time.time(), unit="s") + TPE  # tz-naive, matches last_t
    print(f"撤單流視窗分析  {pd.Timestamp(m0*60, unit='s') + TPE:%m-%d %H:%M} → "
          f"{last_t:%m-%d %H:%M} TPE  (n={len(view)}m, def {DEF_VERSION})")
    print(f"資料截止 {last_t:%H:%M}, 現在 {now_t:%H:%M}"
          + ("  ⚠ 資料落後 >3m" if (now_t - last_t).total_seconds() > 180 else ""))

    hdr = ("時間   |  價格    1m%   量x  takerΔ | shock  毛    淨   | "
           "bidC/askC  b淨/a淨 | L20imb")
    if pfeat is not None:
        hdr += " | perp:shock 毛 淨"
    print("\n" + hdr)
    prev_mid = None
    for m in view.index:
        r = view.loc[m]
        t = (pd.Timestamp(int(m) * 60, unit="s") + TPE).strftime("%H:%M")

        def f(v, spec):
            return format(float(v), spec) if v is not None and np.isfinite(v) \
                else "?".rjust(len(format(0, spec)))
        ret = ((r["mid"] / prev_mid - 1) * 100
               if prev_mid and np.isfinite(r["mid"]) else 0.0)
        prev_mid = float(r["mid"]) if np.isfinite(r["mid"]) else prev_mid
        dlt = r["dlt"] / 1e6 if np.isfinite(r["dlt"]) else None
        line = (f"{t} | {f(r['mid'], '7,.0f')} {ret:+.2f}% "
                f"{f(r['vsk'], '4.1f')} {f(dlt, '+6.2f')}M | "
                f"{f(r['shock'], '4.1f')}x {f(r['skew15'], '+.2f')} "
                f"{f(r['net15'], '+.2f')} | "
                f"{r['bc']:5.0f}/{r['ac']:5.0f} "
                f"{r['ba'] - r['bc']:+5.0f}/{r['aa'] - r['ac']:+5.0f} | "
                f"{f(r['imb'], '+.2f')}")
        if pfeat is not None and m in pfeat.index:
            p = pfeat.loc[m]
            line += (f" | {f(p['shock'], '4.1f')}x {f(p['skew15'], '+.2f')} "
                     f"{f(p['net15'], '+.2f')}")
        print(line)

    print(f"\n=== 視窗內已記錄事件: {len(ev)} 筆 ===")
    for _, e in ev.iterrows():
        t = (pd.Timestamp(int(e['ms']), unit='ms') + TPE).strftime("%m-%d %H:%M")
        outc = ""
        if pd.notna(e.get("fwd_ret_60m")):
            outc = f"  fwd60m={float(e['fwd_ret_60m']):+.3%}"
            if pd.notna(e.get("hit_60m")):
                outc += " ✅" if int(e["hit_60m"]) == 1 else " ❌"
        al = " 📣" if int(e.get("alerted") or 0) == 1 else ""
        print(f"  {t}  {ZH.get(e['playbook'], e['playbook'])} {e['direction']}"
              f"  px={float(e['px'] or 0):,.0f}{outc}{al}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
