# -*- coding: utf-8 -*-
"""Sweep-failure SHADOW engine — prospective signal log, no money, no orders.

Purpose (2026-07-29): stop serialising engineering behind validation. Gate F
needs ~1.5 years of forward data either way; the execution path can be built
and debugged during that time instead of after it, so that the day the gate
passes the plumbing is already proven.

What it does, hourly:
  1. incrementally refresh the 1H kline cache (append-only, dedup by ts)
  2. run the FROZEN backtest function on full history — literally the same
     code Gate F scores, so the shadow log cannot drift from the gate
  3. write every post-freeze trade to a prospective CSV log, keyed by
     (symbol, fill_ts), stamping first_seen_utc the first time it appears
  4. re-derive outcomes each run; a trade stays OPEN until its full HOLD
     window has elapsed in the data

What it deliberately does NOT do: place orders, touch the OKX executor,
write to any production table. It is a recorder.

Why first_seen matters: sweep_forward recomputes forward trades from history
on every run, which is fine but leaves no proof the signal was known before
its outcome. first_seen_utc is that proof — a row whose first_seen precedes
its exit_ts is a genuinely prospective observation.

Universes: `core9` = the frozen Gate F basket. `added20` = the 2026-07-29
expansion (informational until the portfolio framework settles the
concurrency cap). Rows are tagged so Gate F can keep scoring core9 alone.

Run:      python research/sweep_failure/shadow_engine.py
Summary:  python research/sweep_failure/shadow_engine.py --summary
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
# repo root too: _regime_cells imports research.crowd_battery2, and when the
# scheduler runs `python research\sweep_failure\shadow_engine.py` the only
# path Python adds is the SCRIPT's directory — cwd is not on sys.path. That
# made the §0.59 regime annotation fail silently-but-logged on every run.
sys.path.insert(0, str(HERE.parents[1]))
os.environ["SLIP"] = "0"          # gross engine; bps costs applied here
import sweep_core as SC            # noqa: E402
import level_types as LT           # noqa: E402  (same trade fn = no drift)
# §0.94 variant M: the pool-lifecycle and distance definitions live in
# room_ahead and are imported, never re-implemented — a second copy of a
# definition disagrees silently (mistake.md 2026-08-26). Import-safe: both
# modules set SLIP=0 before importing sweep_core, and sweep_forward (pulled
# in transitively) is constants + defs only.
import room_ahead as RA            # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── 資料落點（2026-08-20 上雲）────────────────────────────────────────────
# SWEEP_DATA_DIR 設定時（Railway volume，例如 /data/sweep）：cache 與凍結帳本
# 都放 volume —— 容器層是暫存的，寫進去等於每次部署歸零。第一次啟動時從
# image 內建的 repo 副本播種，之後只 append。沒設環境變數 = 本機原行為。
#
# 帳本是「由 K 線決定性重算」的（同一根 bar 序列跑幾次都得到同一批 key），
# 所以播種副本比較舊也會自我補齊 —— 代價只是補齊列的 first_seen 較晚，
# 不計入 prospective 子集（誠實的一次性成本）。
_DATA_DIR = os.environ.get("SWEEP_DATA_DIR", "").strip()
if _DATA_DIR:
    _root = Path(_DATA_DIR)
    CACHE = _root / ".cache"
    LOG = _root / "sweep_shadow_log.csv"
    _repo_log = Path(__file__).resolve().parents[2] / "research/results/sweep_shadow_log.csv"
    _repo_cache = HERE / ".cache"
    try:
        _root.mkdir(parents=True, exist_ok=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        if not LOG.exists() and _repo_log.exists():
            import shutil
            shutil.copy2(_repo_log, LOG)
            print(f"seeded log from repo copy ({_repo_log.stat().st_size} bytes)")
        if _repo_cache.exists():
            import shutil
            for _f in _repo_cache.glob("*.csv"):
                _t = CACHE / _f.name
                if not _t.exists():
                    shutil.copy2(_f, _t)
    except Exception as _e:  # noqa: BLE001
        print(f"SWEEP_DATA_DIR seed failed: {_e}")
else:
    CACHE = HERE / ".cache"
    LOG = Path(__file__).resolve().parents[2] / "research/results/sweep_shadow_log.csv"
# 主源 + 鏡像：Railway 出口 IP 會被 Binance 擋（fapi 實測 418）；
# data-api.binance.vision 是官方公開資料域，回應形狀完全相同。
BASE = os.environ.get("SWEEP_KLINES_BASE", "https://api.binance.com/api/v3/klines")
BASE_FALLBACK = "https://data-api.binance.vision/api/v3/klines"
FREEZE_TS = int(datetime(2026, 7, 28, tzinfo=timezone.utc).timestamp())
SCEN_A = {"entry": 7.0, "texit": 3.0, "sexit": 10.0}

CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
ADDED20 = ["TRX", "DOT", "LTC", "UNI", "ATOM", "ETC", "NEAR", "APT", "FIL",
           "ARB", "OP", "INJ", "SUI", "AAVE", "ICP", "ALGO", "VET", "HBAR",
           "SAND", "AXS"]

# Variant B (pre-registered 2026-07-29, threshold fixed BEFORE any forward
# trade): take the signal ONLY when the sweep bar pierced the level by
# <= 0.25 ATR. Chosen as the round number nearest the 0.23 tercile boundary
# from the pre-declared tercile bucketing in winner_anatomy.py — not the
# t-peak, and not load-bearing: the whole 0.10-1.00 ATR sweep is positive
# and decays smoothly to the unfiltered mean, so there is no cliff to sit on.
PIERCE_MAX_B = 0.25
# AMENDED 2026-07-29, hours after registration and with 8 forward signals
# accrued (i.e. before any meaningful forward evidence): the liquidity SOURCE
# widens from swing pivots alone to all four pool types the LMSR map draws —
# swing, session extremes, PDH/PDL, PWH/PWL. Every type was tested and every
# type is reported; none were dropped (session is the weakest and is kept).
# This does not loosen the statistical bar — the gate arithmetic is untouched.
# It broadens the test surface, which takes the forward answer from ~8 months
# to ~2 (1327 filtered trades/month vs 245). The pierce filter generalising
# across all four independently is the reason to trust the widening at all.
LEVEL_KINDS = ("swing", "session", "pdh_pdl", "pwh_pwl")

FIELDS = ["symbol", "universe", "level_kind", "first_seen_utc", "fill_ts",
          "fill_utc", "entry_px", "atr", "pierce_atr", "variant_b", "status",
          "exit_ts", "exit_utc", "stopped", "gross_r", "net_r",
          # Order-flow annotation (2026-07-31, operator: "shadow 不整合我就
          # 沒辦法看到訊號的樣子"). PROSPECTIVE observation columns computed
          # from public 1m klines at record time — they change NOTHING in
          # the gate arithmetic (gate_stats reads net_r/variant_b/status
          # only; the registered Variant B track is untouched). Stored raw,
          # no thresholds baked in, so the October pre-registration can
          # apply its cuts to genuinely prospective values instead of
          # retro-computed ones. Definitions mirror the raid-anatomy round-4
          # attack-window features, sourced from klines so all 29 symbols
          # get them (taker delta = 2*taker_buy_base - volume):
          #   flow_reject   1m close crossed back inside before hour end
          #   flow_att_min  minutes spent beyond the level in the sweep hour
          #   flow_vshock   attack-minute vol / trailing-24h median 1m vol
          #   flow_taker    signed taker share INTO the break, attack mins
          #   flow_absorb   flow_taker / max(pierce, 0.1)
          "flow_reject", "flow_att_min", "flow_vshock", "flow_taker",
          "flow_absorb",
          # flow_vhigh (2026-08-01, variant D): 1 if this raid's vshock >=
          # the MEDIAN of this symbol's own strictly-earlier recorded raid
          # vshocks (>=5 priors required, else "na"). Causal, scale-free
          # (every symbol compared to itself), parameter-free (median).
          "flow_vhigh",
          # BTC-only survivor columns (2026-08-02, user: 存活的特徵就該列入
          # 紀錄) — prospective recording of every feature that EARNED a
          # seat in the research line, so October validates on genuinely
          # prospective values. Blank for non-BTC symbols by design
          # (Coinglass + V7 are BTC-scope). All causal at decision time:
          #   drv_q        raid-hour OI down AND taker with break (Q flag)
          #   drv_liqburst raid-hour liquidation total / trailing-24h mean
          #   drv_gap_oi   OI % change over complete hours between raid
          #                hour and fill ("na" for immediate fills) — the
          #                post-raid OI-bleed veto candidate
          #   v7_align     -side x V7 pred_return_4h at the raid bar
          #                (indicator_history, stored UTC — verified
          #                2026-08-02, max-dt lag vs UTC ~1.6h)
          #   drv_gap_cvd  s x futures taker share over the same gap hours
          #                ("na" for immediate fills) — the sequence
          #                hypothesis' second stage (post-raid CVD flip),
          #                added 2026-08-02 when the operator caught the
          #                recording gap
          "drv_q", "drv_liqburst", "drv_gap_oi", "v7_align", "drv_gap_cvd",
          # 2026-09-03: drv_q is the AND of two panels, so neither could be
          # scored on its own -- and the historical decomposition says the
          # CVD half is the weak one. Recorded separately from here on.
          "drv_oi_dn", "drv_cvd_with",
          # M2 additive (2026-08-18, TODO §0.5): trade side from the
          # frozen detection (swept high -> SHORT, swept low -> LONG).
          # Old rows carry "" and are deterministically backfilled on
          # later passes; gate arithmetic never reads it.
          "side",
          # §0.59 prospective columns (2026-08-26). §0.58 showed the forward
          # decay sits almost entirely in the NON-home regime cells while
          # RANGING held; the proposed filter (enter only in RANGING) and
          # the paired exit variant (fail_fast) therefore need to be scored
          # on genuinely NEW fills. Without these columns the verdict day
          # would have to retro-compute them — which is exactly the
          # "prospective values, not retro-computed" rule the flow columns
          # above were added under.
          #   regime_cell  frozen ADX(14) 25/20 label at the fill hour,
          #                TRENDING split by sign of the concurrent 24h
          #                return: RANGING / TREND_UP / TREND_DOWN /
          #                NEUTRAL (§0.49d + §0.54b, no new parameter)
          # fail_fast is deliberately NOT annotated here: exit_variants
          # .entries() omits the `last_exit` non-overlap constraint that
          # sweep_core enforces (it is internally paired, so overlap costs
          # it nothing), which means its fills are a DIFFERENT population.
          # Joining its R onto these rows by fill_ts silently mismatches
          # entries — first attempt produced -0.196R against +0.014R from
          # the paired run, i.e. a wrong sign. The verdict-day fail_fast
          # score comes from exit_by_regime.py, which stays paired.
          "regime_cell",
          # §0.71b prospective column (2026-08-28). How many OTHER pool
          # families had a live, unswept level within 0.10 ATR of this
          # level AT ITS SWEEP BAR. Counted at the sweep bar, not the fill
          # bar — the sweep itself takes out co-located levels, so a
          # fill-bar count excludes exactly what it is meant to measure
          # (the §0.71c instrument bug: 158 confluent events became 5).
          # Recorded, not judged: the candidate FAILED its pre-registration
          # on backtest data (pooled +0.0178 < 0.03, families 2/4) but the
          # three measurable families agreed in sign, so the verdict-day
          # question needs forward values — and forward values only exist
          # if they are recorded from today (the regime_cell rule).
          "confluence_kinds",
          # §0.94 prospective column (2026-09-03). Variant M = A ∧ 後方磁鐵
          # ≤ 1.00 ATR. The effect was found by EXPLORATION on backtest data
          # (the two pre-registered predictions both failed; the surviving
          # shape is the reverse of P2), so it may only be judged on rows
          # that did not exist when it was found -- hence a column recorded
          # from today rather than a retro-computed one, and hence NO
          # backfill of older rows: re-cutting the existing forward sample
          # is the C/D trap (§0.92 rule 3). Definition is imported from
          # room_ahead (all_pools + distances) so there is exactly ONE
          # implementation of it in the repo (mistake.md 2026-08-26).
          "magnet_atr"]

FLOW_BACKFILL_PER_RUN = 40      # cap 1m-kline fetch work per hourly run


def _confluence_counter(bars):
    """(kind, lvl, atr, fill_idx) -> count of OTHER families at the sweep bar.

    Degrades to None on any failure — this annotation must never break the
    frozen recorder (same contract as _regime_cells). All lookups are
    price-bisected; the batch first-hit is the tested one
    (tests/test_first_hits_batch.py, reverse-proven).
    """
    try:
        from bisect import bisect_left, bisect_right

        from research.confluence_all_families import first_hits_batch
        from research.liquidity_map_check import swing_levels
        fam = {k: list(v) for k, v in LT.build_levels(bars).items()}
        fam["swing"] = swing_levels(bars)
        live, own = {}, {}
        for k, items in fam.items():
            hits = first_hits_batch(bars, items)
            arr = sorted((p, est, sd, hh)
                         for (est, p, sd), hh in zip(items, hits))
            live[k] = (arr, [x[0] for x in arr])
            byp = {}
            for (est, pr, sd), hh in zip(items, hits):
                if hh is not None:
                    byp.setdefault(round(pr, 8), []).append(hh)
            own[k] = byp

        def count(kind, lvl, atr, fill_idx, side_long):
            cand = [hh for hh in own.get(kind, {}).get(round(lvl, 8), [])
                    if fill_idx - SC.W <= hh < fill_idx]
            if not cand or not atr or atr <= 0:
                return ""
            jsw = max(cand)
            want = 1 if not side_long else -1
            tol = 0.10 * atr
            c = 0
            for k2, (arr, keys) in live.items():
                if k2 == kind:
                    continue
                a0 = bisect_left(keys, lvl - tol)
                a1 = bisect_right(keys, lvl + tol)
                for p2, est2, s2, h2 in arr[a0:a1]:
                    if (s2 == want and est2 <= jsw
                            and (h2 is None or h2 >= jsw)):
                        c += 1
                        break
            return c
        return count
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] confluence annotation failed: {e}")
        return None


def _regime_cells(bars):
    """fill_ts -> frozen regime cell for §0.59.

    adx_state (§0.49d winner) with the §0.54b direction split. Pure
    time-indexed lookup — no entry population involved, so unlike a
    cross-file exit annotation this cannot mismatch fills. Failures
    degrade to blank; this must never break the frozen recorder.
    """
    cell = {}
    try:
        from research.crowd_battery2 import adx_state
        c = [b[SC.C] for b in bars]
        adx = adx_state(bars)
        LB = 24
        for i in range(LB, len(bars)):
            ts = bars[i][0]
            lab = adx.get(ts // 3600 * 3600)
            if lab is None:
                continue
            if lab == "RANGING":
                cell[ts] = "RANGING"
            elif lab != "TRENDING":
                cell[ts] = "NEUTRAL"
            else:
                cell[ts] = "TREND_UP" if c[i] / c[i - LB] - 1 > 0 else "TREND_DOWN"
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] regime annotation failed: {e}")
    return cell


def fetch_1m_window(sym: str, start_s: int, end_s: int) -> dict[int, tuple]:
    """minute -> (high, low, close, volume, taker_buy_base) from Binance
    spot 1m klines, [start_s, end_s). No disk cache: callers keep windows
    small (~25h) and the hourly run touches only new/blank rows."""
    out: dict[int, tuple] = {}
    cur = start_s * 1000
    while cur < end_s * 1000:
        req = urllib.request.Request(
            f"{BASE}?symbol={sym}USDT&interval=1m&startTime={int(cur)}"
            f"&endTime={end_s * 1000}&limit=1000",
            headers={"User-Agent": "sweep-shadow/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read().decode())
        if not d:
            break
        for k in d:
            out[int(k[0]) // 60_000] = (float(k[2]), float(k[3]), float(k[4]),
                                        float(k[5]), float(k[9]))
        cur = int(d[-1][0]) + 60_000
        if len(d) < 1000:
            break
        time.sleep(0.05)
    return out


def find_sweep(bars, fill_ts: int, lvl: float, atr: float, pierce: float):
    """Recover the sweep bar (and side) the log row came from: the nearest
    bar before the fill whose pierce of `lvl` matches the recorded
    pierce_atr. The log predates these columns, so this stays derivable for
    every historical row instead of only new ones."""
    idx = {b[0]: i for i, b in enumerate(bars)}
    j0 = idx.get(fill_ts)
    if j0 is None or atr <= 0:
        return None
    for j in range(j0 - 1, max(-1, j0 - 1 - SC.W), -1):
        hi, lo = bars[j][SC.H], bars[j][SC.L]
        if abs((hi - lvl) / atr - pierce) < 5e-4 and hi > lvl:
            return bars[j][0], 1
        if abs((lvl - lo) / atr - pierce) < 5e-4 and lo < lvl:
            return bars[j][0], -1
    return None


def flow_flags(sym: str, sweep_ts: int, side: int, lvl: float) -> dict | None:
    """The round-4 attack-window features for ONE sweep, minutes-only."""
    m0 = sweep_ts // 60
    m1 = fetch_1m_window(sym, sweep_ts - 24 * 3600, sweep_ts + 3600)
    mins = [(m, *m1[m]) for m in range(m0, m0 + 60) if m in m1]
    base = [m1[m][3] for m in range(m0 - 1440, m0) if m in m1]
    if len(mins) < 50 or len(base) < 500:
        return None
    beyond = [(m, hi, lo, c, v, tb) for (m, hi, lo, c, v, tb) in mins
              if (hi > lvl if side == 1 else lo < lvl)]
    if not beyond:
        return None
    last_beyond = beyond[-1][0]
    rej = 0
    for (m, _hi, _lo, c, _v, _tb) in mins:
        if m > last_beyond and ((c <= lvl) if side == 1 else (c >= lvl)):
            rej = 1
            break
    vol = sum(b[4] for b in beyond)
    med = sorted(base)[len(base) // 2]
    taker = (side * sum(2 * b[5] - b[4] for b in beyond) / vol
             if vol > 0 else None)
    return {
        "flow_reject": rej,
        "flow_att_min": len(beyond),
        "flow_vshock": (round((vol / len(beyond)) / med, 2)
                        if med > 0 else None),
        "flow_taker": round(taker, 4) if taker is not None else None,
    }


def annotate_flow(log: dict, sym: str, bars) -> int:
    """Fill blank flow_* columns for this symbol's rows, oldest first,
    respecting the per-run fetch cap. Values are immutable once written
    (the sweep hour is always closed by the time a fill exists)."""
    todo = sorted((k for k, r in log.items()
                   if k[0] == sym and r.get("flow_reject") in (None, "")
                   and r.get("pierce_atr")),
                  key=lambda k: k[2])[:FLOW_BACKFILL_PER_RUN]
    done = 0
    for k in todo:
        r = log[k]
        try:
            sw = find_sweep(bars, int(r["fill_ts"]), float(r["entry_px"]),
                            float(r["atr"]), float(r["pierce_atr"]))
            if sw is None:
                r["flow_reject"] = "na"
                continue
            sweep_ts, side = sw
            f = flow_flags(sym, sweep_ts, side, float(r["entry_px"]))
            if f is None:
                r["flow_reject"] = "na"
                continue
            pierce = float(r["pierce_atr"])
            f["flow_absorb"] = (round(f["flow_taker"] / max(pierce, 0.1), 3)
                                if f.get("flow_taker") is not None else "")
            for c in ("flow_reject", "flow_att_min", "flow_vshock",
                      "flow_taker", "flow_absorb"):
                r[c] = f.get(c, "")
            done += 1
        except Exception as e:  # noqa: BLE001 — leave blank, retry next run
            print(f"  {sym}: flow annotate failed ({type(e).__name__}: {e})")
            break
    return done


BOOTSTRAP_START_S = 1705795200   # 2024-01-21 00:00 UTC — 與操作者本機 cache 同窗


def _bootstrap(sym: str, p: Path) -> Path | None:
    """Container cold-start (2026-08-18): the cache used to exist only on the
    operator's machine, so in-image scheduler runs found nothing and skipped
    every symbol (/raid-signals stayed empty forever). Pull the full 1h window
    once — same start as the operator's cache, same CSV format — then the
    normal incremental refresh takes over. Purely data-layer; the FROZEN
    backtest and gate arithmetic are untouched."""
    CACHE.mkdir(parents=True, exist_ok=True)
    rows, start = [], BOOTSTRAP_START_S * 1000
    try:
        while True:
            d = None
            for _base in (BASE, BASE_FALLBACK):
                try:
                    req = urllib.request.Request(
                        f"{_base}?symbol={sym}USDT&interval=1h&startTime={start}&limit=1000",
                        headers={"User-Agent": "sweep-shadow/1.0"})
                    with urllib.request.urlopen(req, timeout=30) as r:
                        d = json.loads(r.read().decode())
                    break
                except Exception:  # noqa: BLE001
                    if _base == BASE_FALLBACK:
                        raise
            if not d:
                break
            now_ms = int(time.time() * 1000)
            for k in d:
                if int(k[6]) > now_ms:      # live bar
                    continue
                rows.append([int(k[0]) // 1000, float(k[1]), float(k[2]),
                             float(k[3]), float(k[4]), float(k[5])])
            if len(d) < 1000:
                break
            start = int(d[-1][0]) + 3_600_000
            time.sleep(0.15)                # 禮貌限速
    except Exception as e:  # noqa: BLE001
        print(f"  {sym}: bootstrap failed ({e})")
        if not rows:
            return None
    if not rows:
        return None
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        w.writerows(rows)
    print(f"  {sym}: bootstrap {len(rows)} bars")
    return p


def refresh(sym: str) -> Path | None:
    """Append new closed 1H bars to the cache. Never rewrites old bars —
    a revision by the exchange would show up as a shadow/backtest mismatch
    rather than being silently absorbed."""
    p = CACHE / f"{sym}USDT_1h.csv"
    if not p.exists():
        return _bootstrap(sym, p)
    rows = SC.load_csv(str(p))
    last = rows[-1][0]
    got = []
    try:
        d = None
        for _base in (BASE, BASE_FALLBACK):
            try:
                req = urllib.request.Request(
                    f"{_base}?symbol={sym}USDT&interval=1h&startTime={(last + 3600) * 1000}"
                    f"&limit=1000", headers={"User-Agent": "sweep-shadow/1.0"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    d = json.loads(r.read().decode())
                break
            except Exception as _e:  # noqa: BLE001
                if _base == BASE_FALLBACK:
                    raise
        if d is None:
            raise RuntimeError("no kline source reachable")
        now_ms = int(time.time() * 1000)
        for k in d:
            if int(k[6]) > now_ms:      # close_time in the future = live bar
                continue
            got.append([int(k[0]) // 1000, float(k[1]), float(k[2]),
                        float(k[3]), float(k[4]), float(k[5])])
    except Exception as e:  # noqa: BLE001
        print(f"  {sym}: refresh failed ({e})")
        return p
    if got:
        with p.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(got)
    return p


def net_r(r: float, lvl: float, atr: float, stopped: bool) -> float:
    legs = SCEN_A["entry"] + (SCEN_A["sexit"] if stopped else SCEN_A["texit"])
    return r - legs / 1e4 * lvl / (SC.DIS * atr)


def lt_to_scen_a(netr: float, lvl: float, atr: float, stopped: bool) -> tuple:
    """(gross, scenario-A net) recovered from an LT.trade_levels row.

    LT.net charges a flat 2 x 5 bps taker regardless of exit type; the Gate F
    cost spec (scenario A) charges 7 entry + 3 time-exit / 10 stop-exit. The
    two agree on time exits (10 = 10) but under-cost stop-outs by 7 bps.
    Found 2026-07-30 — until then the shadow log mixed the two models across
    level kinds. Un-net LT's flat cost exactly, then apply scenario A.
    """
    gross = netr + 2 * LT.TAKER / 1e4 * lvl / (SC.DIS * atr)
    return gross, net_r(gross, lvl, atr, stopped)


def read_log() -> dict[tuple[str, str, int], dict]:
    out = {}
    if LOG.exists():
        with LOG.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[(row["symbol"], row.get("level_kind", "swing"),
                     int(row["fill_ts"]))] = row
    return out


def write_log(log: dict[tuple[str, int], dict]) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(log, key=lambda x: (x[2], x[0], x[1])):
            w.writerow({c: log[k].get(c, "") for c in FIELDS})


def summary(log: dict) -> None:
    rows = list(log.values())
    if not rows:
        print("  (empty log)")
        return
    print(f"  logged rows: {len(rows)}")
    # freshly-built rows hold int 1; rows read back from CSV hold "1"
    def _isb(r):
        return str(r.get("variant_b", "")) == "1"
    groups = [("ALL", lambda r: True), ("B (pierce)", _isb),
              ("C (B∧收回)", is_variant_c), ("D (C∧量能高)", is_variant_d),
              ("E (BTC·OI↓∧CVD順破∧清算高)", variant_e_pred(log)),
              ("E' (BTC·OI↓∧清算高, 凍結 2026-09-03)", variant_e2_pred(log))]
    groups += [(f"B:{k}", (lambda k_: lambda r: _isb(r)
                           and r.get("level_kind") == k_)(k))
               for k in LEVEL_KINDS]
    for uni, pred in groups:
        sub = [r for r in rows if pred(r)]
        closed = [r for r in sub if r["status"] == "CLOSED"]
        if not sub:
            continue
        line = f"  {uni:<8} n={len(sub):>4}  open={len(sub)-len(closed):>3}"
        if closed:
            rs = [float(r["net_r"]) for r in closed]
            m = sum(rs) / len(rs)
            sd = (math.sqrt(sum((x - m) ** 2 for x in rs) / (len(rs) - 1))
                  if len(rs) > 1 else 0.0)
            t = m / (sd / math.sqrt(len(rs))) if sd > 0 else 0.0
            wr = 100 * sum(1 for x in rs if x > 0) / len(rs)
            line += (f"  closed={len(closed):>4}  meanR={m:+.4f}"
                     f"  WR={wr:.0f}%  t={t:+.2f}")
        print(line)
    # prospectiveness: rows whose outcome was still unknown when first seen
    pro = sum(1 for r in rows if r["status"] == "CLOSED" and r["exit_ts"]
              and r["first_seen_utc"] and
              datetime.strptime(r["first_seen_utc"], "%Y-%m-%d %H:%M").replace(
                  tzinfo=timezone.utc).timestamp() < int(r["exit_ts"]))
    print(f"  genuinely prospective (first_seen < exit): {pro}")


# ── VARIANT C (registered 2026-07-31, before any of its forward outcomes
# were examined as a cohort): the flow-confirmed subset the operator asked
# to see live instead of waiting for October ("為什麼不直接加進去").
#   C = variant_b AND flow_reject == 1  (raid hour closed back inside)
# Deliberately THRESHOLD-FREE: the only flow condition is the binary
# close-back flag — the strongest single resolution feature in the raid
# anatomy work. The volume-shock cut is NOT part of C: the shadow basket's
# vshock scale differs wildly across symbols (median 9.4x, tercile
# boundaries 5.9/16.4 vs ~3 in the BTC/ETH research), so any fixed number
# today would be either arbitrary or tuned. Raw vshock stays recorded; the
# October pre-registration decides whether a percentile-framed volume
# condition adds anything on top of C.
# C changes NOTHING for A/B: same trades, same gate arithmetic, one more
# label. It is an OBSERVATION cohort — promotion to a tradeable rule still
# requires its own forward evidence under the same clustered-CI bar.


def is_variant_c(row: dict) -> bool:
    return (str(row.get("variant_b", "")) == "1"
            and str(row.get("flow_reject", "")) == "1")


# ── VARIANT D (registered 2026-08-01): the ACTUAL order-flow combination
# from the raid anatomy — reversal recipe R∧V (close-back AND high attack
# volume). The operator's point stood: C alone is a 1m price-path confirm,
# not order flow. "High volume" is defined causally and scale-free:
#   flow_vhigh = vshock >= median of THIS symbol's own strictly-earlier
#   recorded raid vshocks (>=5 priors, else excluded) — every coin compared
#   to itself, median = parameter-free central cut. October's
#   pre-registration may test the tercile framing on top; BTC's OI x CVD
#   (Q) stays October-scope (Coinglass, BTC-only, separate data path).
# D ⊂ C ⊂ B ⊂ A. Observation cohort, same rules as C: gate arithmetic
# shared, A/B untouched, promotion needs its own forward evidence.


def is_variant_d(row: dict) -> bool:
    return is_variant_c(row) and str(row.get("flow_vhigh", "")) == "1"


# ── VARIANT E (registered 2026-08-02): the operator's own manual read,
# as a tracked cohort. When a raid happens they judge the next move from
# three panels — liquidations, CVD, open interest — so E encodes exactly
# that, from columns already recorded prospectively:
#   E = BTC raid ∧ drv_q = 1 (OI down AND taker with break: the OI+CVD
#       panels' stop-flush read) ∧ liquidation burst >= the causal median
#       of BTC's own earlier recorded raids (>=5 priors; zero tuned
#       numbers, same convention as flow_vhigh).
# E is deliberately NOT restricted to variant B — it tracks the manual
# read on every BTC raid, so its overlap with B/C/D stays measurable.
# BTC-only (Coinglass scope). Observation cohort, same rules as C/D.


def _liq_high_keys(log: dict) -> set:
    """BTC raids whose liq burst >= the causal median of EARLIER BTC raids.

    Causal by construction: the bar at each raid is what BTC's own past
    raids looked like up to that moment, never a global median. Shared by
    E and E' so the two cohorts cannot drift apart on this half.
    """
    from statistics import median
    rows = [(int(r["fill_ts"]), float(r["drv_liqburst"]),
             (r["symbol"], r.get("level_kind", "swing"), int(r["fill_ts"])))
            for r in log.values()
            if r["symbol"] == "BTC"
            and r.get("drv_liqburst") not in (None, "", "na")]
    rows.sort()
    eligible = set()
    for i, (fts, lb, key) in enumerate(rows):
        prior = [v for (ft2, v, _k) in rows[:i] if ft2 < fts]
        if len(prior) >= 5 and lb >= median(prior):
            eligible.add(key)
    return eligible


def variant_e_pred(log: dict):
    """Build the E-membership test (causal liq-burst median needs the
    whole log, so this returns a closure for gate_stats)."""
    eligible = _liq_high_keys(log)
    return lambda r: (str(r.get("drv_q", "")) == "1"
                      and (r["symbol"], r.get("level_kind", "swing"),
                           int(r["fill_ts"])) in eligible)


# ── VARIANT E' (registered 2026-09-03): E without the CVD panel.
# E = OI down AND taker-with-break AND liq burst; E' drops the middle one:
#   E' = BTC raid ∧ OI down at the raid hour ∧ liq burst >= causal median
# Why it exists: the historical decomposition (research/sweep_raid_variant_e.py,
# 3,068 BTC raids) says the CVD panel is the weak half -- dropping it scores
# HIGHER (+0.1723 vs +0.1616), and the pocket it excludes (OI down + burst but
# taker AGAINST the break) is the best cell in the table (n=37, +0.4382).
# That is a DISCOVERY-SAMPLE observation and cannot promote itself, so E'
# starts a clock of its own from zero today. E ⊂ E'; a failure of E' does NOT
# void E (E carries an extra condition), which is the opposite direction from
# the A⊃B⊃C⊃D chain and is deliberate.
def variant_e2_pred(log: dict):
    eligible = _liq_high_keys(log)
    return lambda r: (str(r.get("drv_oi_dn", "")) == "1"
                      and (r["symbol"], r.get("level_kind", "swing"),
                           int(r["fill_ts"])) in eligible)


def annotate_btc_survivors(log: dict, bars) -> int:
    """Fill the BTC-only survivor columns (drv_q / drv_liqburst /
    drv_gap_oi / v7_align) for rows still blank. Heavy deps (pandas for
    the Coinglass parquets, MySQL for V7 preds) are imported lazily and
    any failure leaves blanks for the next hourly run — the 29-symbol
    flow annotation must never depend on them. Values are immutable once
    written; CG parquets refresh daily, so recent raids may stay blank
    for up to a day (retried automatically)."""
    todo = [k for k, r in log.items()
            if k[0] == "BTC" and (r.get("drv_q") in (None, "")
                                  or r.get("drv_oi_dn") in (None, "")
                                  or r.get("v7_align") in (None, "")
                                  or r.get("drv_gap_cvd") in (None, ""))]
    if not todo:
        return 0
    root = Path(__file__).resolve().parents[2]
    try:
        import pandas as pd
        raw = root / "market_data" / "raw_data"

        def hmap(fname, col):
            df = pd.read_parquet(raw / fname)
            idx = pd.to_datetime(df.index)
            if idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            return {int(t.value // 10**9) // 3600: float(v)
                    for t, v in zip(idx, df[col].astype(float)) if v == v}

        oi = hmap("cg_oi_agg_1h.parquet", "close")
        fdf = pd.read_parquet(raw / "cg_futures_cvd_agg_1h.parquet")
        fb = hmap("cg_futures_cvd_agg_1h.parquet", "agg_taker_buy_vol")
        fs = hmap("cg_futures_cvd_agg_1h.parquet", "agg_taker_sell_vol")
        del fdf
        ll = hmap("cg_liq_agg_1h.parquet", "aggregated_long_liquidation_usd")
        ls = hmap("cg_liq_agg_1h.parquet", "aggregated_short_liquidation_usd")
    except Exception as e:  # noqa: BLE001
        print(f"  BTC survivors: CG parquets unavailable ({type(e).__name__})")
        return 0
    preds: dict[int, float] = {}
    try:
        import sys as _sys
        if str(root) not in _sys.path:
            _sys.path.insert(0, str(root))
        from shared.db import get_db_conn
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT dt, pred_return_4h FROM indicator_history "
                            "WHERE pred_return_4h IS NOT NULL")
                for row in cur.fetchall():
                    preds[int(row["dt"].replace(
                        tzinfo=timezone.utc).timestamp())] = float(
                            row["pred_return_4h"])
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        print(f"  BTC survivors: V7 preds unavailable ({type(e).__name__})")

    done = 0
    for k in todo:
        r = log[k]
        try:
            sw = find_sweep(bars, int(r["fill_ts"]), float(r["entry_px"]),
                            float(r["atr"]), float(r["pierce_atr"]))
            if sw is None:
                r["drv_q"] = "na"
                r["v7_align"] = "na"
                continue
            sweep_ts, side = sw
            hh = sweep_ts // 3600
            if r.get("drv_oi_dn") in (None, ""):
                # the two halves of drv_q, recorded separately (2026-09-03).
                # Same source, same hour, same sign convention -- this is a
                # NEW column from immutable data, not a rewrite of an old one.
                if all(x is not None for x in
                       (oi.get(hh), oi.get(hh - 1), fb.get(hh), fs.get(hh))):
                    r["drv_oi_dn"] = int(oi[hh] < oi[hh - 1])
                    r["drv_cvd_with"] = int(side * (fb[hh] - fs[hh]) > 0)
                else:
                    r["drv_oi_dn"] = r["drv_cvd_with"] = "na"
            if r.get("drv_q") in (None, ""):
                if all(x is not None for x in
                       (oi.get(hh), oi.get(hh - 1), fb.get(hh), fs.get(hh))):
                    r["drv_q"] = int(oi[hh] < oi[hh - 1]
                                     and side * (fb[hh] - fs[hh]) > 0)
                    tot = (ll.get(hh, 0) + ls.get(hh, 0))
                    base = [ll.get(hh - j, 0) + ls.get(hh - j, 0)
                            for j in range(1, 25)]
                    base = [b for b in base if b > 0]
                    r["drv_liqburst"] = (round(tot / (sum(base) / len(base)), 2)
                                         if base and tot > 0 else "na")
                    fill_hh = int(r["fill_ts"]) // 3600
                    if fill_hh >= hh + 2:
                        o0, o1 = oi.get(hh), oi.get(fill_hh - 1)
                        r["drv_gap_oi"] = (round((o1 / o0 - 1) * 100, 3)
                                           if o0 and o1 else "")
                    else:
                        r["drv_gap_oi"] = "na"
                    done += 1
            if r.get("drv_gap_cvd") in (None, ""):
                fill_hh = int(r["fill_ts"]) // 3600
                if fill_hh >= hh + 2:
                    num = den = 0.0
                    ok = True
                    for h2 in range(hh + 1, fill_hh):
                        b_, s_ = fb.get(h2), fs.get(h2)
                        if b_ is None or s_ is None:
                            ok = False
                            break
                        num += b_ - s_
                        den += b_ + s_
                    if ok and den > 0:
                        r["drv_gap_cvd"] = round(side * num / den, 4)
                else:
                    r["drv_gap_cvd"] = "na"
            if r.get("v7_align") in (None, "") and preds:
                p = preds.get(sweep_ts)
                if p is not None:          # missing -> stay blank, retry
                    r["v7_align"] = round(-side * p, 6)
        except Exception:  # noqa: BLE001 — leave blank, retry next run
            continue
    return done


def annotate_vhigh(log: dict) -> int:
    """Fill flow_vhigh for rows that have a vshock but no verdict yet.
    Uses only strictly-earlier fills of the same symbol — causal by
    construction, and immutable once written (priors never change)."""
    from statistics import median
    bysym: dict[str, list] = {}
    for k, r in log.items():
        if r.get("flow_vshock") not in (None, "", "na"):
            bysym.setdefault(k[0], []).append(
                (int(r["fill_ts"]), float(r["flow_vshock"]), k))
    done = 0
    for sym, rows in bysym.items():
        rows.sort()
        for i, (fts, vs, k) in enumerate(rows):
            r = log[k]
            if r.get("flow_vhigh") not in (None, ""):
                continue
            prior = [v for (ft2, v, _k2) in rows[:i] if ft2 < fts]
            if len(prior) < 5:
                r["flow_vhigh"] = "na"
            else:
                r["flow_vhigh"] = int(vs >= median(prior))
                done += 1
    return done


GATE_N = 1400          # same floor as Variant A's Gate F

# Variants C/D were pre-registered 2026-08-09 (TODO §0.44) with their own
# clock and their own floor: n >= 400 counted FROM THAT DAY.  Rows first seen
# before it are the "既有基礎" the section explicitly excludes — C/D were
# promoted to candidates *because* those rows looked good, so scoring them is
# self-grading, the same error combo_watchlist.forward_only() exists to stop.
# It matters here more than anywhere: on 2026-08-13 the unfiltered board read
# C +0.0965 / CI-low +0.0097 and D +0.1021 / +0.0062 — both apparently through
# the bar — while the true-forward cohorts were C +0.0068 / CI-low -0.0886
# (n=117) and D +0.0283 / -0.1178 (n=68).  All of the edge sat in the
# selection period, exactly like the 08-07 split (R∧Q +0.973 -> n=4 CI -0.206).
CD_CLOCK = "2026-08-09"
CD_GATE_N = 400


def _since_cd_clock(pred):
    """Score C/D on rows first seen after their registration day only."""
    return lambda r: pred(r) and (r.get("first_seen_utc") or "") >= CD_CLOCK


def gate_stats(log: dict, cohort=None) -> dict:
    """Structured gate progress — same arithmetic as Gate F: day-clustered
    bootstrap CI on net R plus the n>=1400 floor. Consumed by gate_progress
    (weekly report string) and the agent's /public/sweep-status JSON route
    (product-site strategy board). `cohort` defaults to variant B (the
    registered track); pass is_variant_c for the observation cohort — same
    arithmetic, so the two lines are always comparable. Without this the
    shadow log accumulates unwatched — a two-month clock nobody reads is a
    clock that does not exist."""
    import random
    from collections import defaultdict
    if cohort is None:
        def cohort(r):  # noqa: E306
            return str(r.get("variant_b", "")) == "1"
    rows = [r for r in log.values()
            if cohort(r) and r["status"] == "CLOSED" and r["net_r"] != ""]
    n_open = sum(1 for r in log.values()
                 if cohort(r) and r["status"] == "OPEN")
    if not rows:
        return {"n_closed": 0, "n_open": n_open, "floor": GATE_N,
                "mean_r": None, "ci_low": None, "wr_pct": None, "status": "empty"}
    byd = defaultdict(list)
    for r in rows:
        d = datetime.fromtimestamp(int(r["fill_ts"]), timezone.utc).date()
        byd[d].append(float(r["net_r"]))
    days = list(byd.values())
    rs = [float(r["net_r"]) for r in rows]
    n = len(rs)
    mean = sum(rs) / n
    rng = random.Random(7)
    means = []
    for _ in range(2000):
        acc = cnt = 0.0
        for _ in range(len(days)):
            g = days[rng.randrange(len(days))]
            acc += sum(g)
            cnt += len(g)
        means.append(acc / cnt)
    means.sort()
    lo = means[50]
    # Status is a classification of the frozen arithmetic above, nothing
    # more.  Once the floor is reached the clock has run out: CI-low > 0 is
    # PASS, otherwise FAIL (variant B: 2026-09-02, n=1428, CI-low -0.094,
    # TODO §0.92).  Below the floor it is still accumulating — the verdict
    # is only ever read at the floor, never "so far".
    if n >= GATE_N and lo > 0:
        status = "PASS"
    elif n >= GATE_N:
        status = "FAIL"
    else:
        status = "accumulating"
    return {"n_closed": n, "n_open": n_open, "floor": GATE_N,
            "mean_r": round(mean, 4), "ci_low": round(lo, 4),
            "wr_pct": round(100 * sum(1 for x in rs if x > 0) / n, 1),
            "status": status}


# ── E / E' decision rules, frozen 2026-09-03 (TODO §0.474b) ───────────
# E was registered 2026-08-02 with a definition and NO criteria ("純記錄").
# That is the shape that lets a good-looking number promote itself, so the
# bar is written down here, in code, where it gets evaluated -- and the
# numbers were derived BEFORE looking at anything but E's own variance
# (sd 0.65 on 22 post-registration rows -> n=41 for a CI that clears zero at
# the observed mean; 60 leaves margin for a weaker true mean).
#
# Four conditions, ALL required, read once at the floor:
#   1. n >= 60 closed rows dated on/after the cohort's own freeze
#   2. day-clustered bootstrap CI95 low > 0
#   3. meanR at least +0.08R above BTC raids OUTSIDE the cohort over the
#      same window -- without this, "BTC raids are good lately" passes as
#      "the three panels work"
#   4. both halves of the window positive (no single event carrying it)
E_CLOCK, E_GATE_N = "2026-08-02", 60
E2_CLOCK, E2_GATE_N = "2026-09-03", 60
E_CONTROL_EDGE = 0.08


def _since(pred, clock: str):
    ts0 = int(datetime.strptime(clock, "%Y-%m-%d")
              .replace(tzinfo=timezone.utc).timestamp())
    return lambda r: pred(r) and int(r["fill_ts"]) >= ts0


def e_clock(log: dict, which: str = "E") -> dict:
    """Frozen scorer for E / E'. Owns these numbers; boards must read it."""
    pred = variant_e_pred(log) if which == "E" else variant_e2_pred(log)
    clock, floor = ((E_CLOCK, E_GATE_N) if which == "E"
                    else (E2_CLOCK, E2_GATE_N))
    inside = _since(pred, clock)
    st = gate_stats(log, inside)
    btc_out = gate_stats(log, _since(
        lambda r: r["symbol"] == "BTC" and not pred(r), clock))
    rows = sorted((r for r in log.values()
                   if r["status"] == "CLOSED" and r["net_r"] != "" and inside(r)),
                  key=lambda r: int(r["exit_ts"]))
    rs = [float(r["net_r"]) for r in rows]
    h = len(rs) // 2
    halves = ((sum(rs[:h]) / h, sum(rs[h:]) / (len(rs) - h))
              if len(rs) >= 2 else (0.0, 0.0))
    gap = (st["mean_r"] - btc_out["mean_r"]) if st["n_closed"] and btc_out["n_closed"] else None
    at_floor = st["n_closed"] >= floor
    ok = (at_floor and st["ci_low"] > 0 and gap is not None
          and gap >= E_CONTROL_EDGE and min(halves) > 0)
    return {"which": which, "clock": clock, "floor": floor,
            "n": st["n_closed"], "mean_r": st["mean_r"], "ci_low": st["ci_low"],
            "wr_pct": st["wr_pct"], "control_mean": btc_out["mean_r"],
            "control_n": btc_out["n_closed"],
            "gap": round(gap, 4) if gap is not None else None,
            "halves": [round(halves[0], 4), round(halves[1], 4)],
            "status": ("PASS" if ok else "FAIL" if at_floor else "accumulating")}


# ── Variant M, frozen 2026-09-03 (TODO §0.94) ────────────────────────
# M = A ∧ back magnet <= 1.00 ATR. Floor 400 comes from the backtest cell's
# own variance BEFORE any forward row existed (n=2338, meanR +0.1015,
# sd 0.633, design effect 1.69 -> n≈253 clears zero at the observed mean;
# 400 leaves margin). core9 only: the registered breadth bar is >=6/9 and
# the backtest that produced the effect was core9.
# 連坐: A ⊃ M — if A is judged NO-GO, M is void. That was decided at
# registration, before any data, precisely so it cannot be revisited on
# the day A fails.
M_CLOCK, M_GATE_N = "2026-09-03", 400
M_CLOCK_TS = int(datetime(2026, 9, 3, tzinfo=timezone.utc).timestamp())
MAGNET_MAX_M = 1.00


def variant_m_pred(r) -> bool:
    v = r.get("magnet_atr")
    if v in (None, "", "na") or r.get("universe") != "core9":
        return False
    try:
        return float(v) <= MAGNET_MAX_M
    except (TypeError, ValueError):
        return False


def m_clock(log: dict) -> dict:
    """Frozen scorer for M. Owns these numbers; boards must read it.

    Four conditions, ALL required, read once at the floor (§0.94):
      1. n >= 400 closed rows with fill_ts on/after the freeze
      2. day-clustered bootstrap CI95 low > 0
      3. >= 6/9 core9 coins positive
      4. both halves of the window positive
    """
    inside = _since(variant_m_pred, M_CLOCK)
    st = gate_stats(log, inside)
    rows = sorted((r for r in log.values()
                   if r["status"] == "CLOSED" and r["net_r"] != "" and inside(r)),
                  key=lambda r: int(r["exit_ts"]))
    rs = [float(r["net_r"]) for r in rows]
    h = len(rs) // 2
    halves = ((sum(rs[:h]) / h, sum(rs[h:]) / (len(rs) - h))
              if len(rs) >= 2 else (0.0, 0.0))
    per: dict[str, list] = {}
    for r in rows:
        per.setdefault(r["symbol"], []).append(float(r["net_r"]))
    pos = sum(1 for v in per.values() if sum(v) / len(v) > 0)
    at_floor = st["n_closed"] >= M_GATE_N
    ok = (at_floor and st["ci_low"] is not None and st["ci_low"] > 0
          and pos >= 6 and min(halves) > 0)
    return {"clock": M_CLOCK, "floor": M_GATE_N, "n": st["n_closed"],
            "mean_r": st["mean_r"], "ci_low": st["ci_low"],
            "wr_pct": st["wr_pct"], "coins_pos": pos, "coins": len(per),
            "halves": [round(halves[0], 4), round(halves[1], 4)],
            "status": ("PASS" if ok else "FAIL" if at_floor else "accumulating")}


def gate_progress(log: dict) -> str:
    """Variant B gate line + the variant C observation line (same maths)."""
    s = gate_stats(log)
    if s["n_closed"] == 0:
        return "Variant B: 0/%d (no closed signals yet)" % GATE_N
    out = (f"Variant B: n={s['n_closed']}/{s['floor']} meanR={s['mean_r']:+.4f} "
           f"clustered-CI-low={s['ci_low']:+.4f} -> {s['status']}")
    # C/D on their own clock and their own floor (TODO §0.44).  The pre-clock
    # rows are shown only as `base`, never mixed into the scored figure —
    # printing the pooled number made both look through the bar while their
    # true-forward CI was deeply negative.
    # 連坐: A ⊃ B ⊃ C ⊃ D (TODO §0.43) — once B has failed at its floor the
    # C/D clocks are void.  Their numbers keep printing so they stay
    # auditable; only the label changes.
    void = " [作廢·連坐 B FAIL 2026-09-02]" if s["status"] == "FAIL" else ""
    for label, pred in (("C(B∧收回)", is_variant_c),
                        ("D(C∧量能高)", is_variant_d)):
        label += void
        st = gate_stats(log, _since_cd_clock(pred))
        base = gate_stats(log, pred)["n_closed"] - st["n_closed"]
        if st["n_closed"]:
            out += (f" | {label}: n={st['n_closed']}/{CD_GATE_N} since {CD_CLOCK} "
                    f"meanR={st['mean_r']:+.4f} CI-low={st['ci_low']:+.4f}"
                    f" (base {base} excluded)")
        else:
            out += f" | {label}: 0/{CD_GATE_N} since {CD_CLOCK} (base {base} excluded)"
    for which, zh in (("E", "E(BTC·OI↓∧CVD順破∧清算高)"),
                      ("E'", "E'(BTC·OI↓∧清算高)")):
        c = e_clock(log, which)

        def _n(x):
            return f"{x:+.4f}" if isinstance(x, (int, float)) else "—"

        out += (f" | {zh}: n={c['n']}/{c['floor']} since {c['clock']} "
                f"meanR={_n(c['mean_r'])} CI-low={_n(c['ci_low'])} "
                f"vs非本組BTC {_n(c['control_mean'])}"
                f"(差{_n(c['gap'])}) -> {c['status']}")
    m = m_clock(log)

    def _m(x):
        return f"{x:+.4f}" if isinstance(x, (int, float)) else "—"

    out += (f" | M(A∧後方磁鐵≤{MAGNET_MAX_M:.2f}ATR·core9): "
            f"n={m['n']}/{m['floor']} since {m['clock']} "
            f"meanR={_m(m['mean_r'])} CI-low={_m(m['ci_low'])} "
            f"幣{m['coins_pos']}/{m['coins']}正 -> {m['status']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true",
                    help="print the log summary without refreshing data")
    ap.add_argument("--gate", action="store_true",
                    help="one-line Variant B gate progress (for the weekly report)")
    ap.add_argument("--combos", action="store_true",
                    help="forward scoreboard for the frozen combo watchlist")
    args = ap.parse_args()
    log = read_log()
    if args.combos:
        import combo_watchlist as CW
        print(f"combo watchlist (registered {CW.REGISTERED}) — rows first seen "
              "AFTER registration only, clustered-CI arithmetic:")
        for name, pred in CW.combo_preds(log).items():
            st = gate_stats(log, CW.forward_only(pred))
            if st["n_closed"]:
                print(f"  {name:<10} closed={st['n_closed']:>4} open={st['n_open']:>3}"
                      f"  meanR={st['mean_r']:+.4f}  CI-low={st['ci_low']:+.4f}"
                      f"  WR={st['wr_pct']:.0f}%")
            else:
                print(f"  {name:<10} closed=0 (accumulating)")
        return 0
    if args.gate:
        print(gate_progress(log))
        return 0
    if args.summary:
        summary(log)
        return 0

    now = datetime.now(timezone.utc)
    stamp = f"{now:%Y-%m-%d %H:%M}"
    new = 0
    flow_done = 0
    drv_done = 0
    for uni, syms in (("core9", CORE9), ("added20", ADDED20)):
        for sym in syms:
            p = refresh(sym)
            if p is None:
                continue
            bars = SC.load_csv(str(p))
            last_ts = bars[-1][0]
            # §0.59 annotation tables, one pass per symbol. Pure reads of
            # frozen definitions — nothing below feeds gate arithmetic.
            cellmap = _regime_cells(bars)
            confcnt = _confluence_counter(bars)
            bar_idx = {b[0]: i for i, b in enumerate(bars)}
            pools = None          # §0.94: built on first row that needs it
            # one source of truth per kind: the frozen engine for swing,
            # the shared level engine for the time-defined pools. Tuples carry
            # (kind, fill_ts, exit_ts, gross, net, pierce, lvl, atr, stopped);
            # both paths are costed under scenario A (lt_to_scen_a un-nets
            # LT's flat taker model first).
            evts: list[tuple] = [
                ("swing", t[0], t[1], t[2],
                 net_r(t[2], t[3], t[4], t[5]), t[6], t[3], t[4], t[5], t[7])
                for t in SC.backtest_symbol(bars)]
            lv = LT.build_levels(bars)
            for kind in ("session", "pdh_pdl", "pwh_pwl"):
                for (f_ts, x_ts, netr, pc, lvl, atr, st_, sd) in LT.trade_levels(
                        bars, lv.get(kind, [])):
                    gross, neta = lt_to_scen_a(netr, lvl, atr, st_)
                    evts.append((kind, f_ts, x_ts, gross, neta, pc, lvl, atr,
                                 st_, sd))

            for (kind, fill_ts, exit_ts, gross, netv, pierce, lvl, atr,
                 stopped, side) in evts:
                if fill_ts < FREEZE_TS:
                    continue
                key = (sym, kind, fill_ts)
                done = fill_ts + SC.HOLD * 3600 <= last_ts
                row = log.get(key)
                if row is None:
                    row = {"symbol": sym, "universe": uni, "level_kind": kind,
                           "first_seen_utc": stamp, "fill_ts": fill_ts,
                           "fill_utc": f"{datetime.fromtimestamp(fill_ts, timezone.utc):%Y-%m-%d %H:%M}",
                           "entry_px": f"{lvl:.6f}", "atr": f"{atr:.6f}",
                           "pierce_atr": f"{pierce:.4f}",
                           "variant_b": int(pierce <= PIERCE_MAX_B),
                           "side": side}
                    log[key] = row
                    new += 1
                if not row.get("side"):
                    row["side"] = side   # deterministic backfill of old rows
                # §0.59: deterministic from klines, so old rows fill in on
                # later passes exactly like `side` did. Only rows whose
                # fill_ts is on/after the rule's registration date count as
                # evidence — the verdict filters on that, not on presence.
                if not row.get("regime_cell"):
                    row["regime_cell"] = cellmap.get(fill_ts, "")
                # 0 is a legitimate count, so test for absence, not falsiness
                if row.get("confluence_kinds") in (None, "") and confcnt:
                    fi = bar_idx.get(fill_ts)
                    if fi is not None:
                        row["confluence_kinds"] = confcnt(
                            kind, lvl, atr, fi, side == "LONG")
                # §0.94 magnet: only rows from M's own freeze forward. Older
                # rows are deliberately left blank — they are the data the
                # effect was FOUND on, so labelling them would manufacture
                # a "forward" sample out of the exploration set.
                if (row.get("magnet_atr") in (None, "")
                        and fill_ts >= M_CLOCK_TS):
                    fi = bar_idx.get(fill_ts)
                    if fi is not None:
                        if pools is None:
                            pools = RA.all_pools(bars)
                        _room, _mag = RA.distances(
                            pools, fi, lvl, 1 if side == "LONG" else -1, atr)
                        row["magnet_atr"] = f"{_mag:.4f}"
                row.update({
                    "status": "CLOSED" if done else "OPEN",
                    "exit_ts": exit_ts if done else "",
                    "exit_utc": (f"{datetime.fromtimestamp(exit_ts, timezone.utc):%Y-%m-%d %H:%M}"
                                 if done else ""),
                    "stopped": int(bool(stopped)) if done else "",
                    "gross_r": f"{gross:.6f}" if (done and gross is not None) else "",
                    "net_r": f"{netv:.6f}" if done else "",
                })
            flow_done += annotate_flow(log, sym, bars)
            if sym == "BTC":
                drv_done = annotate_btc_survivors(log, bars)
    annotate_vhigh(log)
    write_log(log)
    n_flow = sum(1 for r in log.values()
                 if r.get("flow_reject") not in (None, "", "na"))
    print(f"shadow run {stamp}  new signals: {new}  "
          f"flow annotated: +{flow_done} ({n_flow}/{len(log)} covered)  "
          f"BTC survivors: +{drv_done}")
    summary(log)
    print("  " + gate_progress(log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
