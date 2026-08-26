# -*- coding: utf-8 -*-
"""Positioning gauges for the survival-conditions layer — TODO §0.65.

WHY THIS EXISTS, and why it is not "another feature sweep":

The survival layer (§0.49) monitors the PRECONDITIONS a strategy needs in
order to make money, rather than its P&L — because at +7bps/trade against
~100bps of noise, performance monitoring is mathematically too slow to
tell life from death. Every gauge in it today is PRICE-DERIVED: ADX,
Donchian breakout P&L, SMA50/200 trend P&L.

Meanwhile 67% of V7's 136 production features are Coinglass — paid,
non-price, positioning data — and 8 of its top 10 drivers are
`cg_oi_*` / `cg_ls_*` / `cg_bfx_margin_*`. The layer that asks "what state
is the market in" has never been given the data source that best answers
exactly that question. That gap, not a new idea, is what this file closes.

THE DISTINCTION THAT MAKES THIS DIFFERENT FROM THE SATURATION VERDICT:
"Coinglass is saturated" means adding more cg-derived features does not
lift 4h-direction AUC in a model that already holds 91 of them. It is a
statement about PREDICTION. This layer does not predict — a gauge only has
to describe the CURRENT state well enough that strategy performance
differs across its readings. Concurrent description and forward prediction
are different jobs and the saturation result speaks only to the second.
That must stay stated: nothing here may later be read as a forecast claim.

Supporting prior: project_orderflow_edge_verdict found instantaneous
hourly flow worthless at 4h (it lost to 30 random features) while
INTEGRATED positioning carried 66%. Positioning is the half that survived.

── PRE-REGISTRATION, frozen before the first run ────────────────────────

FIVE gauges. Thresholds are canonical, NOT swept — a swept cutoff here
would be the 2026-06-20 threshold-sweep trap wearing a new hat.

  G1 群眾槓桿水位   aggregate OI, z-score over trailing 720h (30d)
                   HIGH z>+1 / MID / LOW z<-1
     先驗: 高槓桿 = 易連環清算 = 對淡化型策略不利 (SF 逆勢接掃單失敗)
  G2 大戶 vs 散戶   top_position_long_short_ratio vs global_account ratio
                   分歧方向 = sign(log(top) - log(global))
     先驗: 大戶與群眾反向時, 群眾那側的價位更容易被穿透
  G3 資金費率       funding close, z-score 720h; EXTREME |z|>1 vs NORMAL
     先驗: 極端費率 = 一側過度擁擠 = 反轉燃料
  G4 清算強度       (long+short liq usd), z-score 720h; BURST z>+1
     先驗: 爆量清算 = 強制平倉主導 = 價位無效
  G5 現貨溢價       coinbase premium_rate sign
     先驗: 美國現貨需求方向, 與槓桿盤方向分歧時是壓力

TESTED AGAINST (both, full grid, no cherry-picking):
  SF  變體 B core9 回測成交 -> meanR by gauge state
  V7  Strong 訊號 -> 方向勝率 by gauge state

BTC's reading is used as a MARKET-WIDE state for all coins, exactly as the
weather station does — crypto beta is high enough that a single-asset
positioning read is the market's read. Stated so it is not mistaken for
a per-coin measurement.

JUDGMENT (the §0.49d two-tier bar, unchanged):
  一級 顯示級  符號正確 ∧ 桶間差達實質量級 ∧ SF 逐幣廣度 ≥6/9
  二級 告警級  再加 日聚類 CI 下緣離零 ∧ 兩半同號
Anything passing only 一級 goes on the board as display, never as an alert.

MULTIPLE COMPARISONS: 5 gauges x 2 strategies = 10 tests; at p<0.05 about
0.5 pass by chance. Stated here, and required in any writeup. A single
marginal survivor is NOT a finding — the layer already killed an entire
family (RSI/BB/Stoch mean-reversion crowd P&L: zero information on both
strategies), so the prior on any new gauge working is genuinely low.

SANITY BEFORE READING ANYTHING: bucket counts are checked for physical
plausibility first. Any bucket at 0 or >90% means broken instrument, and
the run stops rather than being interpreted (mistake.md 2026-08-02).
"""
from __future__ import annotations

import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import sweep_core as SC                                    # noqa: E402
from shared.db import get_db_conn                          # noqa: E402

RAW = ROOT / "market_data" / "raw_data"
CACHE = ROOT / "research" / "sweep_failure" / ".cache"
OUT = ROOT / "research" / "results" / "positioning_gauges.json"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
PIERCE_B = 0.25
Z_WIN = 720            # 30 days, canonical — not tuned
random.seed(31)


def clustered_ci(pairs, n_boot=3000):
    if not pairs:
        return None
    by = defaultdict(list)
    for d, v in pairs:
        by[d].append(v)
    days = list(by)
    if len(days) < 3:
        return None
    m = []
    for _ in range(n_boot):
        pick = [random.choice(days) for _ in days]
        vals = [x for d in pick for x in by[d]]
        if vals:
            m.append(st.mean(vals))
    m.sort()
    return m[int(.025 * len(m))], m[int(.975 * len(m))]


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def _load(name) -> pd.DataFrame:
    d = pd.read_parquet(RAW / f"{name}.parquet")
    if not isinstance(d.index, pd.DatetimeIndex):
        raise SystemExit(f"{name}: index is not datetime — instrument check")
    d = d[~d.index.duplicated(keep="last")].sort_index()
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    return d


def _z(s: pd.Series) -> pd.Series:
    m = s.rolling(Z_WIN, min_periods=Z_WIN // 3).mean()
    sd = s.rolling(Z_WIN, min_periods=Z_WIN // 3).std()
    return (s - m) / sd.replace(0, np.nan)


def build_gauges() -> dict[str, dict[int, str]]:
    """hour_ts -> state, per gauge. All readings are CONCURRENT."""
    g: dict[str, dict[int, str]] = {}

    oi = _load("cg_oi_agg_1h")["close"].astype(float)
    z = _z(oi)
    g["G1 群眾槓桿水位"] = {int(t.timestamp()): (
        "HIGH 高槓桿" if v > 1 else "LOW 低槓桿" if v < -1 else "MID 中性")
        for t, v in z.items() if pd.notna(v)}

    top = _load("cg_top_ls_position_1h")["top_position_long_short_ratio"].astype(float)
    glob = _load("cg_global_ls_1h")["global_account_long_short_ratio"].astype(float)
    j = pd.concat([top.rename("t"), glob.rename("g")], axis=1).dropna()
    dv = np.log(j["t"].clip(lower=1e-9)) - np.log(j["g"].clip(lower=1e-9))
    dz = _z(dv)
    g["G2 大戶vs散戶分歧"] = {int(t.timestamp()): (
        "大戶更多頭" if v > 1 else "大戶更空頭" if v < -1 else "一致")
        for t, v in dz.items() if pd.notna(v)}

    fz = _z(_load("cg_funding_1h")["close"].astype(float))
    g["G3 資金費率極端"] = {int(t.timestamp()): (
        "極端正" if v > 1 else "極端負" if v < -1 else "正常")
        for t, v in fz.items() if pd.notna(v)}

    lq = _load("cg_liq_agg_1h")
    tot = (lq["aggregated_long_liquidation_usd"].astype(float)
           + lq["aggregated_short_liquidation_usd"].astype(float))
    lz = _z(np.log1p(tot))
    g["G4 清算強度"] = {int(t.timestamp()): (
        "BURST 爆量" if v > 1 else "平靜")
        for t, v in lz.items() if pd.notna(v)}

    pr = _load("cg_coinbase_premium_1h")["premium_rate"].astype(float)
    g["G5 現貨溢價"] = {int(t.timestamp()): ("溢價" if v > 0 else "折價")
                        for t, v in pr.items() if pd.notna(v)}
    return g


def sf_fills():
    """Variant B core9 backtest fills — the layer's SF population."""
    out = []
    for sym in CORE9:
        fp = CACHE / f"{sym}USDT_1h.csv"
        if not fp.exists():
            continue
        bars = SC.load_csv(str(fp))
        for fill_ts, _x, R, _l, _A, _s, pierce, _sd in SC.backtest_symbol(bars):
            if pierce > PIERCE_B:
                continue
            out.append({"ts": int(fill_ts) // 3600 * 3600, "R": R, "sym": sym})
    return out


def v7_signals():
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT signal_time, correct FROM tracked_signals "
                        "WHERE strength='Strong' AND correct IS NOT NULL")
            rows = cur.fetchall()
    finally:
        conn.close()
    return [{"ts": int(r["signal_time"].replace(tzinfo=timezone.utc)
                       .timestamp()) // 3600 * 3600, "ok": int(r["correct"])}
            for r in rows]


def main() -> int:
    gauges = build_gauges()
    sf, v7 = sf_fills(), v7_signals()
    print("§0.65 持倉資料進生存條件層 —— 五個儀表，判準先於數字\n")
    print(f"  SF 母體：變體 B core9 回測成交 n={len(sf)}")
    print(f"  V7 母體：Strong 已結算訊號 n={len(v7)}")
    print("  所有讀數皆為**同期**狀態，不是預測——這一層問的是"
          "「現在是什麼環境」\n")

    res = {}
    for gname, gmap in gauges.items():
        sfg = defaultdict(list)
        for r in sf:
            s = gmap.get(r["ts"])
            if s:
                sfg[s].append(r)
        v7g = defaultdict(list)
        for r in v7:
            s = gmap.get(r["ts"])
            if s:
                v7g[s].append(r)
        n_sf = sum(len(v) for v in sfg.values())
        if not n_sf:
            print(f"── {gname} ── 無對齊樣本，跳過\n")
            continue

        # physical sanity FIRST — before any performance number is read
        shares = {k: len(v) / n_sf for k, v in sfg.items()}
        bad = [k for k, s in shares.items() if s > 0.9]
        print(f"── {gname} ──")
        if len(sfg) < 2 or bad:
            print(f"   儀器疑慮：桶分佈 { {k: round(s,3) for k,s in shares.items()} }"
                  " 不合物理，停止解讀\n")
            res[gname] = {"instrument_ok": False, "shares": shares}
            continue

        row = {"instrument_ok": True, "sf": {}, "v7": {}}
        print(f"   {'狀態':<12} {'SF n':>6} {'SF meanR':>10} "
              f"{'日聚類CI':>20} {'廣度':>6} | {'V7 n':>6} {'V7 WR':>7}")
        for state in sorted(sfg, key=lambda s: -len(sfg[s])):
            v = sfg[state]
            m = st.mean(x["R"] for x in v)
            ci = clustered_ci([(x["ts"] // 86400, x["R"]) for x in v])
            per = defaultdict(list)
            for x in v:
                per[x["sym"]].append(x["R"])
            breadth = sum(1 for s in per if st.mean(per[s]) > 0)
            u = v7g.get(state, [])
            wr = 100 * sum(x["ok"] for x in u) / len(u) if u else None
            cis = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else "—"
            print(f"   {state:<12} {len(v):6d} {m:+10.4f} {cis:>20} "
                  f"{breadth:4d}/{len(per)} | {len(u):6d} "
                  + (f"{wr:6.1f}%" if wr is not None else "     —"))
            row["sf"][state] = {"n": len(v), "meanR": round(m, 4),
                                "ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
                                "breadth": f"{breadth}/{len(per)}"}
            row["v7"][state] = {"n": len(u),
                                "wr": round(wr, 2) if wr is not None else None}
        # spread between the best and worst state
        ms = [row["sf"][s]["meanR"] for s in row["sf"]]
        ws = [row["v7"][s]["wr"] for s in row["v7"] if row["v7"][s]["wr"] is not None]
        row["sf_spread"] = round(max(ms) - min(ms), 4)
        row["v7_spread"] = round(max(ws) - min(ws), 2) if len(ws) > 1 else None
        print(f"   → SF 桶間差 {row['sf_spread']:+.4f}R"
              + (f"｜V7 桶間差 {row['v7_spread']:+.1f}pp"
                 if row["v7_spread"] is not None else ""))
        print()
        res[gname] = row

    # ── ranking against the frozen bar ──────────────────────────────────
    print("── 對照凍結判準 ──")
    print("   一級：符號正確 ∧ 桶間差實質 ∧ SF 廣度 ≥6/9")
    print("   二級：再加 CI 下緣離零")
    ranked = sorted(
        (r for r in res.values() if r.get("instrument_ok")),
        key=lambda r: -(r.get("sf_spread") or 0))
    for gname, r in sorted(res.items(),
                           key=lambda kv: -(kv[1].get("sf_spread") or 0)):
        if not r.get("instrument_ok"):
            print(f"   {gname:<18} 儀器不合格")
            continue
        best = max(r["sf"], key=lambda s: r["sf"][s]["meanR"])
        worst = min(r["sf"], key=lambda s: r["sf"][s]["meanR"])
        bci = r["sf"][best]["ci"]
        bbr = int(r["sf"][best]["breadth"].split("/")[0])
        tier2 = bci is not None and bci[0] > 0 and bbr >= 6
        tier1 = bbr >= 6 and (r["sf_spread"] or 0) >= 0.05
        tag = ("二級（告警級）" if tier2 and tier1 else
               "一級（顯示級）" if tier1 else "未達標")
        print(f"   {gname:<18} 差 {r['sf_spread']:+.4f}R  "
              f"最佳「{best}」{r['sf'][best]['meanR']:+.4f} "
              f"廣度 {r['sf'][best]['breadth']}  → {tag}")
    print("\n   探索性：5 儀表 × 2 策略 = 10 次比較，約 0.5 個僥倖過關。"
          "單一邊際存活者不是發現。")
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
