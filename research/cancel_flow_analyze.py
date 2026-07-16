"""Cancel-flow window analyzer — the engine behind "叫 agent 分析".

One core (`analyze_window` → structured dict), three consumers:
  · CLI table  (default)         — human forensic minute table
  · CLI --summary                — compact deterministic text (TG command)
  · MCP agent  (indicator/agent) — imports analyze_window, LLM narrates

Feature definitions are IMPORTED from market_data.tasks.cancel_playbook_watcher
(single source of truth — never redefined here). Read-only on all tables.

Usage (times are TPE, UTC+8):
    python research/cancel_flow_analyze.py                      # 最近 90 分鐘
    python research/cancel_flow_analyze.py --mins 240 --perp
    python research/cancel_flow_analyze.py --from "2026-07-16 20:00" --to "2026-07-16 21:10"
    python research/cancel_flow_analyze.py --mins 90 --summary  # TG 版摘要
"""
from __future__ import annotations

import argparse
import json
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
    compute_features, ZH, DEF_VERSION, GATE_SHOCK)

TPE = pd.Timedelta(hours=8)
WARMUP_MIN = 90
MAX_MINUTE_ROWS = 240


def _q(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params or None)
        return pd.DataFrame(cur.fetchall() or [])


def _load(t0_ms: int, t1_ms: int, exchange: str = "binance"):
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
        ob = _q(conn, "SELECT ts_ms, mid_price mid, imbalance_l20 imb "
                      "FROM orderbook_snapshots_1m "
                      "WHERE canonical_symbol='BTC-USD' AND ts_ms BETWEEN %s AND %s "
                      "ORDER BY ts_ms", (lo, t1_ms))
        ev = _q(conn, "SELECT minute_start_ms ms, playbook, direction, px, shock, "
                      "vshock, fwd_ret_60m, hit_60m, alerted "
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
        df = df.join(ob.groupby("m").last()[["mid", "imb"]], how="left")
    else:
        df["mid"] = np.nan; df["imb"] = np.nan
    df["mid"] = df["mid"].ffill()
    return df, ev


def _t(m) -> str:
    return (pd.Timestamp(int(m) * 60, unit="s") + TPE).strftime("%m-%d %H:%M")


def _fv(v):
    return None if v is None or not np.isfinite(v) else round(float(v), 4)


def analyze_window(t0_ms: int, t1_ms: int, perp: bool = False) -> dict:
    """Structured, LLM-narratable analysis of one window. Read-only."""
    df, ev = _load(t0_ms, t1_ms)
    if df.empty:
        return {"error": "視窗內無 depth 資料（collector 2026-07-09 起）"}
    feat = compute_features(df)
    vbase = feat["vol"].rolling(60, min_periods=30).median()
    feat["vsk"] = feat["vol"] / vbase.replace(0, np.nan)
    m0, m1 = t0_ms // 60_000, t1_ms // 60_000
    v = feat.loc[(feat.index >= m0) & (feat.index <= m1)].copy()
    if v.empty:
        return {"error": "視窗內無資料"}
    v["ret"] = v["mid"].pct_change()
    v["bnet"] = v["ba"] - v["bc"]
    v["anet"] = v["aa"] - v["ac"]

    last = v.iloc[-1]
    lag_min = (time.time() - int(v.index.max()) * 60 - 60) / 60
    px0, px1 = float(v["mid"].iloc[0]), float(v["mid"].iloc[-1])

    gate = v[v["shock"] >= GATE_SHOCK]
    bursts = v.reindex(v["dlt"].abs().sort_values(ascending=False).index[:6])

    def tilt(s):
        s = s.dropna()
        if s.empty:
            return None
        return {"last": _fv(s.iloc[-1]), "mean": _fv(s.mean()),
                "min": _fv(s.min()), "max": _fv(s.max())}

    # deterministic lean score (display aid, NOT a signal):
    score = 0
    if last["net15"] == last["net15"]:
        score += (1 if last["net15"] >= 0.10 else -1 if last["net15"] <= -0.10 else 0)
    score += (1 if px1 > px0 * 1.0005 else -1 if px1 < px0 * 0.9995 else 0)
    dsum = v["dlt"].sum()
    score += (1 if dsum > 0 else -1 if dsum < 0 else 0)
    lean = "偏多" if score >= 2 else "偏空" if score <= -2 else "中性"

    step = max(1, int(np.ceil(len(v) / MAX_MINUTE_ROWS)))
    minutes = []
    for m in v.index[::step]:
        r = v.loc[m]
        minutes.append({
            "t": _t(m), "px": _fv(r["mid"]), "ret_pct": _fv(r["ret"] * 100),
            "vshock": _fv(r["vsk"]),
            "taker_musd": _fv(r["dlt"] / 1e6 if np.isfinite(r["dlt"]) else None),
            "shock": _fv(r["shock"]), "skew15": _fv(r["skew15"]),
            "net15": _fv(r["net15"]), "bid_net": _fv(r["bnet"]),
            "ask_net": _fv(r["anet"]), "l20_imb": _fv(r["imb"]),
        })

    events = []
    for _, e in ev.iterrows():
        events.append({
            "t": _t(int(e["ms"]) // 60_000),
            "playbook": ZH.get(e["playbook"], e["playbook"]),
            "direction": e["direction"], "px": _fv(pd.to_numeric(e["px"])),
            "shock": _fv(pd.to_numeric(e["shock"])),
            "vshock": _fv(pd.to_numeric(e["vshock"])),
            "fwd_ret_60m": _fv(pd.to_numeric(e["fwd_ret_60m"])),
            "hit_60m": None if pd.isna(e["hit_60m"]) else int(e["hit_60m"]),
            "alerted": int(e["alerted"] or 0),
        })

    res = {
        "meta": {"from_tpe": _t(m0), "to_tpe": _t(int(v.index.max())),
                 "n_min": int(len(v)), "lag_min": round(max(0.0, lag_min), 1),
                 "def_version": DEF_VERSION},
        "verdict": {
            "gate_minutes": int(len(gate)),
            "max_shock": _fv(v["shock"].max()),
            "lean": lean,
            "note": "無事件" if gate.empty else f"{len(gate)} 分鐘過鬧鐘門檻",
        },
        "right_edge": {"px": _fv(last["mid"]), "shock": _fv(last["shock"]),
                       "skew15": _fv(last["skew15"]), "net15": _fv(last["net15"]),
                       "vshock": _fv(last["vsk"])},
        "tilt": {"skew15": tilt(v["skew15"]), "net15": tilt(v["net15"]),
                 "deep_skew_min": int((v["skew15"].abs() >= 0.30).sum()),
                 "deep_net_min": int((v["net15"].abs() >= 0.30).sum()),
                 "bid_net_sum": _fv(v["bnet"].sum()),
                 "ask_net_sum": _fv(v["anet"].sum()),
                 "taker_delta_sum_musd": _fv(dsum / 1e6),
                 "px_change_pct": _fv((px1 / px0 - 1) * 100)},
        "gate_minutes": [{"t": _t(m), "shock": _fv(gate.loc[m, "shock"]),
                          "skew15": _fv(gate.loc[m, "skew15"]),
                          "net15": _fv(gate.loc[m, "net15"]),
                          "vshock": _fv(gate.loc[m, "vsk"])}
                         for m in gate.index],
        "taker_bursts": [{"t": _t(m), "px": _fv(bursts.loc[m, "mid"]),
                          "taker_musd": _fv(bursts.loc[m, "dlt"] / 1e6),
                          "vshock": _fv(bursts.loc[m, "vsk"]),
                          "ret_pct": _fv(bursts.loc[m, "ret"] * 100)}
                         for m in bursts.index if np.isfinite(bursts.loc[m, "dlt"])],
        "events": events,
        "minutes": minutes,
        "disclaimer": "研究覆盤工具, 非交易信號; edge 待 2026-08-10 判決",
    }
    if perp:
        pdf, _ = _load(t0_ms, t1_ms, exchange="binance_perp")
        if not pdf.empty:
            pf = compute_features(pdf)
            pv = pf.loc[(pf.index >= m0) & (pf.index <= m1)]
            res["perp_tilt"] = {"skew15": tilt(pv["skew15"]),
                                "net15": tilt(pv["net15"]),
                                "deep_skew_min": int((pv["skew15"].abs() >= 0.30).sum())}
    return res


# ── text renderers ───────────────────────────────────────────────────────────

def format_summary(r: dict) -> str:
    """Compact deterministic summary (fits a Telegram message)."""
    if "error" in r:
        return f"❌ {r['error']}"
    m, vd, t, re_ = r["meta"], r["verdict"], r["tilt"], r["right_edge"]

    def rng(x):
        return (f"last {x['last']:+.2f} [{x['min']:+.2f},{x['max']:+.2f}]"
                if x else "?")
    lines = [
        f"📋 撤單流摘要 {m['from_tpe']} → {m['to_tpe']} TPE "
        f"(n={m['n_min']}m, lag {m['lag_min']:.0f}m)",
        f"① 鬧鐘: 過門檻 {vd['gate_minutes']} 分鐘 (max {vd['max_shock']:.1f}x)"
        f" → {vd['note']}",
        f"② 毛偏斜: {rng(t['skew15'])} | 淨: {rng(t['net15'])}"
        f" | 深色 毛{t['deep_skew_min']}m/淨{t['deep_net_min']}m",
        f"③ 淨掛單: bid {t['bid_net_sum']:+,.0f} / ask {t['ask_net_sum']:+,.0f}"
        f" | takerΔ累計 {t['taker_delta_sum_musd']:+.1f}M",
        f"④ 價格: {t['px_change_pct']:+.2f}% → {re_['px']:,.0f}"
        f" | 右緣 shock {re_['shock']:.1f}x 毛 {re_['skew15']:+.2f}"
        f" 淨 {re_['net15']:+.2f}",
    ]
    if r["taker_bursts"]:
        b = r["taker_bursts"][0]
        lines.append(f"⑤ 最大砲彈: {b['t']} {b['taker_musd']:+.1f}M"
                     f" (量{b['vshock']:.1f}x, {b['ret_pct']:+.2f}%)")
    if r["events"]:
        evs = "; ".join(f"{e['t']} {e['playbook']}{e['direction']}"
                        + (f" fwd60={e['fwd_ret_60m']:+.3%}" if e["fwd_ret_60m"] is not None else "")
                        for e in r["events"][:5])
        lines.append(f"⑥ 事件({len(r['events'])}): {evs}")
    else:
        lines.append("⑥ 事件: 0 筆")
    lines.append(f"→ 底色傾斜: {vd['lean']}（機械判定, 非信號）")
    lines.append(r["disclaimer"])
    return "\n".join(lines)


def format_table(r: dict) -> str:
    if "error" in r:
        return f"❌ {r['error']}"
    m = r["meta"]
    out = [f"撤單流視窗分析  {m['from_tpe']} → {m['to_tpe']} TPE "
           f"(n={m['n_min']}m, def {m['def_version']}, lag {m['lag_min']:.0f}m)",
           "", "時間        |  價格    1m%   量x  takerΔ | shock  毛    淨   |"
           "  b淨/a淨  | L20imb"]
    for x in r["minutes"]:
        def g(k, spec, dash="    ?"):
            vv = x.get(k)
            return format(vv, spec) if vv is not None else dash
        out.append(f"{x['t']} | {g('px', '7,.0f')} {g('ret_pct', '+.2f')}% "
                   f"{g('vshock', '4.1f')} {g('taker_musd', '+6.2f')}M | "
                   f"{g('shock', '4.1f')}x {g('skew15', '+.2f')} {g('net15', '+.2f')} | "
                   f"{g('bid_net', '+6.0f')}/{g('ask_net', '+6.0f')} | {g('l20_imb', '+.2f')}")
    out.append("")
    out.append(f"=== 已記錄事件: {len(r['events'])} 筆 ===")
    for e in r["events"]:
        o = (f"  fwd60m={e['fwd_ret_60m']:+.3%}" if e["fwd_ret_60m"] is not None else "")
        o += " ✅" if e["hit_60m"] == 1 else (" ❌" if e["hit_60m"] == 0 else "")
        o += " 📣" if e["alerted"] else ""
        out.append(f"  {e['t']}  {e['playbook']} {e['direction']}  px={e['px'] or 0:,.0f}{o}")
    if "perp_tilt" in r:
        p = r["perp_tilt"]

        def rng(x):
            return (f"last {x['last']:+.2f} [{x['min']:+.2f},{x['max']:+.2f}]"
                    if x else "?")
        out.append(f"perp 對照: 毛 {rng(p['skew15'])} | 淨 {rng(p['net15'])}"
                   f" | 深色 {p['deep_skew_min']}m")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mins", type=int, default=90)
    ap.add_argument("--from", dest="t_from", help='TPE "YYYY-MM-DD HH:MM"')
    ap.add_argument("--to", dest="t_to", help='TPE "YYYY-MM-DD HH:MM"')
    ap.add_argument("--perp", action="store_true")
    ap.add_argument("--summary", action="store_true", help="TG 版壓縮摘要")
    ap.add_argument("--json", action="store_true", help="輸出完整 dict")
    args = ap.parse_args()

    if args.t_from and args.t_to:
        t0 = int((pd.Timestamp(args.t_from) - TPE).timestamp() * 1000)
        t1 = int((pd.Timestamp(args.t_to) - TPE).timestamp() * 1000)
    else:
        t1 = int(time.time() * 1000)
        t0 = t1 - args.mins * 60_000

    res = analyze_window(t0, t1, perp=args.perp)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
    elif args.summary:
        print(format_summary(res))
    else:
        print(format_table(res))
    return 0 if "error" not in res else 1


if __name__ == "__main__":
    raise SystemExit(main())
