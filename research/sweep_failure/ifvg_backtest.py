"""F8 — IFVG(反轉公允價值缺口) × 掃單位置, three-cohort backtest.

PRE-REGISTERED 2026-08-13 (TODO §0.47) — every definition and every
prediction below was frozen and committed BEFORE this script produced a
single number.  Do not tune anything here against results; a changed
parameter is a new family with a new registration.

Origin: the user's discretionary combo — liquidity raid + IFVG retest entry
+ large prints inside the zone (5m/15m).  The question under test is NOT
"does IFVG work" but "is LOCATION the source of the edge": the terrain
campaign's dividing line was that pure price-structure dims all died
(S2 BOS/CHoCH, D6) while liquidity-location dims all survived (D1/D2/D3/D5).
Cohort design encodes exactly that:

    (a)  IFVG whose zone sits within 0.5 ATR of a level swept in the prior
         96 bars  — the user's manual setup
    (a2) (a) AND a large-print proxy on the retest bar
    (b)  every IFVG unconditionally
    (c)  random-timing control matched 1:1 to (b) (same coin, same
         direction, same exits, seed=7)

Frozen predictions: P1 (a)>(b), (a)>(c), (b)~(c); P2 verdict bar = cohort
(a) day-clustered bootstrap CI-low of NET R > 0; P3 (a2)>=(a); P4 gross>0
but net<=0 -> PARK (subhourly G2 lesson — 5/15m is where costs eat signals).

Costs: Scenario A (7 bps entry, 3 time-exit, 10 stop-exit), same R
conversion as shadow_engine.  Exits: sweep_core frozen (3.5 ATR disaster
stop, HOLD=8 native bars).  Same-bar entry+stop counts as stopped
(conservative).  Read-only research code — no production imports, no DB.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import random
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "results" / "ifvg_cache"
CACHE.mkdir(parents=True, exist_ok=True)

# ── frozen 2026-08-13 (TODO §0.47) ──────────────────────────────────────
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
TFS = {"5m": 300, "15m": 900}      # both scored, both reported
MONTHS = 12
MIN_GAP_ATR = 0.10                 # min FVG size
FVG_LIFE = 200                     # bars before an un-inverted FVG is dropped
RETEST_W = 96                      # bars after inversion to wait for retest
SWEEP_LOOKBACK = 96                # sweep must precede inversion by <= this
PROX_ATR = 0.5                     # zone-to-swept-level proximity
PIVOT = 10                         # same as sweep_core
PIERCE_ATR = 0.05                  # sweep pierce depth
BACK_IN = 3                        # bars to close back inside = sweep
HOLD = 8                           # native bars (sweep_core frozen)
DIS = 3.5                          # disaster stop, ATR mult (sweep_core)
VOL_WIN, VOL_P = 288, 0.80         # large-print proxy: volume percentile
TAKER_LONG, TAKER_SHORT = 0.55, 0.45
ENTRY_BPS, TEXIT_BPS, SEXIT_BPS = 7.0, 3.0, 10.0   # Scenario A
SEED = 7
BOOT_N = 2000
# ────────────────────────────────────────────────────────────────────────

BASE = "https://api.binance.com/api/v3/klines"
O, H, L, C, V, TB = 1, 2, 3, 4, 5, 6      # bar tuple layout


def fetch_bars(sym: str, tf: str, months: int) -> list[tuple]:
    """(ts_s, o, h, l, c, vol, taker_buy) ascending; gzip-csv cached."""
    fp = CACHE / f"{sym}_{tf}.csv.gz"
    if fp.exists():
        with gzip.open(fp, "rt", newline="") as f:
            return [tuple(float(x) if i else int(float(x))
                          for i, x in enumerate(row))
                    for row in csv.reader(f)]
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - months * 30 * 86_400_000
    out: list[tuple] = []
    cur = start_ms
    while cur < end_ms:
        req = urllib.request.Request(
            f"{BASE}?symbol={sym}USDT&interval={tf}&startTime={cur}"
            f"&endTime={end_ms}&limit=1000",
            headers={"User-Agent": "ifvg-backtest/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read().decode())
        if not d:
            break
        for k in d:
            out.append((int(k[0]) // 1000, float(k[1]), float(k[2]),
                        float(k[3]), float(k[4]), float(k[5]), float(k[9])))
        cur = int(d[-1][0]) + 1
        if len(d) < 1000:
            break
        time.sleep(0.12)
    with gzip.open(fp, "wt", newline="") as f:
        w = csv.writer(f)
        for b in out:
            w.writerow(b)
    return out


def atr14(bars: list[tuple]) -> list[float]:
    """Simple mean TR(14), same convention as sweep_core.atr14."""
    n = len(bars)
    atr = [0.0] * n
    trs: list[float] = []
    for i in range(n):
        if i == 0:
            tr = bars[i][H] - bars[i][L]
        else:
            pc = bars[i - 1][C]
            tr = max(bars[i][H] - bars[i][L],
                     abs(bars[i][H] - pc), abs(bars[i][L] - pc))
        trs.append(tr)
        if i >= 13:
            atr[i] = sum(trs[i - 13:i + 1]) / 14.0
    return atr


def sweep_events(bars: list[tuple], atr: list[float]) -> list[tuple[int, float]]:
    """(pierce_bar_index, level) — pivot swing H/L (k=PIVOT) + prior-day
    H/L; a sweep pierces the level by >= PIERCE_ATR and closes back inside
    within BACK_IN bars."""
    n = len(bars)
    levels: list[tuple[int, float, bool]] = []   # (born_i, price, is_high)
    for i in range(PIVOT, n - PIVOT):
        win = bars[i - PIVOT:i + PIVOT + 1]
        if bars[i][H] == max(b[H] for b in win):
            levels.append((i + PIVOT, bars[i][H], True))
        if bars[i][L] == min(b[L] for b in win):
            levels.append((i + PIVOT, bars[i][L], False))
    # prior-day high/low, usable from the first bar of the next UTC day
    by_day: dict[str, list[tuple]] = defaultdict(list)
    day_first: dict[str, int] = {}
    for i, b in enumerate(bars):
        d = datetime.fromtimestamp(b[0], tz=timezone.utc).strftime("%Y-%m-%d")
        by_day[d].append(b)
        day_first.setdefault(d, i)
    days = sorted(by_day)
    for prev, cur in zip(days, days[1:]):
        i0 = day_first[cur]
        levels.append((i0, max(b[H] for b in by_day[prev]), True))
        levels.append((i0, min(b[L] for b in by_day[prev]), False))

    out: list[tuple[int, float]] = []
    for born, lvl, is_high in levels:
        for j in range(born, min(born + 500, n)):
            a = atr[j]
            if a <= 0:
                continue
            pierced = (bars[j][H] >= lvl + PIERCE_ATR * a if is_high
                       else bars[j][L] <= lvl - PIERCE_ATR * a)
            if not pierced:
                continue
            back = any(
                (bars[k][C] < lvl if is_high else bars[k][C] > lvl)
                for k in range(j, min(j + BACK_IN + 1, n)))
            if back:
                out.append((j, lvl))
            break                      # first pierce decides; done with level
    return sorted(out)


def find_trades(bars: list[tuple], atr: list[float],
                sweeps: list[tuple[int, float]]) -> list[dict]:
    """All cohort-(b) trades, flagged for (a)/(a2) membership."""
    n = len(bars)
    sweep_by_i = sorted(sweeps)
    vols = [b[V] for b in bars]
    trades: list[dict] = []
    # open FVGs: (t_index, lo, hi, bullish)
    open_fvgs: list[tuple[int, float, float, bool]] = []
    for t in range(2, n):
        a = atr[t]
        if a > 0:
            if bars[t][L] > bars[t - 2][H] and \
                    bars[t][L] - bars[t - 2][H] >= MIN_GAP_ATR * a:
                open_fvgs.append((t, bars[t - 2][H], bars[t][L], True))
            if bars[t - 2][L] > bars[t][H] and \
                    bars[t - 2][L] - bars[t][H] >= MIN_GAP_ATR * a:
                open_fvgs.append((t, bars[t][H], bars[t - 2][L], False))
        still: list[tuple[int, float, float, bool]] = []
        for born, lo, hi, bull in open_fvgs:
            if t - born > FVG_LIFE:
                continue
            inverted = (bars[t][C] < lo) if bull else (bars[t][C] > hi)
            if not inverted:
                still.append((born, lo, hi, bull))
                continue
            tr = _retest_trade(bars, atr, t, lo, hi, bull)
            if tr is not None:
                inv_i = t
                near = any(
                    inv_i - SWEEP_LOOKBACK <= si <= inv_i
                    and abs(lvl - (lo + hi) / 2) <= PROX_ATR * atr[inv_i]
                    for si, lvl in sweep_by_i)
                tr["cohort_a"] = near
                e = tr["entry_i"]
                if near and e >= VOL_WIN:
                    p80 = sorted(vols[e - VOL_WIN:e])[
                        int(VOL_P * VOL_WIN)]
                    share = (bars[e][TB] / bars[e][V]) if bars[e][V] > 0 else 0.5
                    tr["cohort_a2"] = (
                        bars[e][V] >= p80
                        and (share >= TAKER_LONG if tr["dir"] > 0
                             else share <= TAKER_SHORT))
                else:
                    tr["cohort_a2"] = False
                trades.append(tr)
        open_fvgs = still
    return trades


def _retest_trade(bars, atr, inv_i, lo, hi, bull) -> dict | None:
    """User's manual entry: after the inversion close, first touch of the
    zone's near edge within RETEST_W bars fills a limit at the edge.
    Inverted bullish FVG -> SHORT at zone_lo; inverted bearish -> LONG at hi."""
    n = len(bars)
    direction = -1 if bull else 1
    edge = lo if bull else hi
    for j in range(inv_i + 1, min(inv_i + 1 + RETEST_W, n)):
        touched = (bars[j][H] >= edge) if bull else (bars[j][L] <= edge)
        if not touched:
            continue
        a = atr[j]
        if a <= 0:
            return None
        return _simulate(bars, j, edge, direction, a)
    return None


def _simulate(bars, entry_i, entry_px, direction, a) -> dict | None:
    stop = entry_px - direction * DIS * a
    n = len(bars)
    end = min(entry_i + HOLD, n - 1)
    stopped = False
    exit_px = bars[end][C]
    for j in range(entry_i, end + 1):     # entry bar included: conservative
        hit = (bars[j][L] <= stop) if direction > 0 else (bars[j][H] >= stop)
        if hit:
            stopped, exit_px = True, stop
            break
    gross = direction * (exit_px - entry_px) / (DIS * a)
    legs = ENTRY_BPS + (SEXIT_BPS if stopped else TEXIT_BPS)
    net = gross - legs / 1e4 * entry_px / (DIS * a)
    return {"entry_i": entry_i, "ts": bars[entry_i][0], "dir": direction,
            "gross": gross, "net": net, "stopped": stopped}


def random_control(bars, atr, real: list[dict], rng) -> list[dict]:
    out = []
    n = len(bars)
    for tr in real:
        for _ in range(50):
            i = rng.randrange(VOL_WIN, n - HOLD - 2)
            if atr[i] > 0:
                sim = _simulate(bars, i, bars[i][C], tr["dir"], atr[i])
                if sim:
                    out.append(sim)
                break
    return out


def clustered_ci(trades: list[dict], key: str = "net"):
    """Day-clustered bootstrap CI (same arithmetic family as gate_stats)."""
    if not trades:
        return 0.0, 0.0, 0.0
    by_day: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        d = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%Y-%m-%d")
        by_day[d].append(t[key])
    days = list(by_day.values())
    rng = random.Random(SEED)
    means = []
    for _ in range(BOOT_N):
        pick = [days[rng.randrange(len(days))] for _ in range(len(days))]
        flat = [x for grp in pick for x in grp]
        means.append(sum(flat) / len(flat))
    means.sort()
    mean = sum(t[key] for t in trades) / len(trades)
    return mean, means[int(0.025 * BOOT_N)], means[int(0.975 * BOOT_N)]


def report(label: str, trades: list[dict]) -> None:
    if not trades:
        print(f"  {label:<22} n=0")
        return
    g = sum(t["gross"] for t in trades) / len(trades)
    mean, lo, hi = clustered_ci(trades)
    wr = 100 * sum(1 for t in trades if t["net"] > 0) / len(trades)
    print(f"  {label:<22} n={len(trades):>5}  grossR={g:+.4f}  "
          f"netR={mean:+.4f}  CI95[{lo:+.4f},{hi:+.4f}]  WR={wr:.0f}%")


def main() -> None:
    rng = random.Random(SEED)
    for tf in TFS:
        print(f"\n════ {tf}  (core9, {MONTHS}mo, frozen §0.47) ════")
        pooled: dict[str, list[dict]] = {"a": [], "a2": [], "b": [], "c": []}
        pos: dict[str, int] = defaultdict(int)
        for sym in CORE9:
            bars = fetch_bars(sym, tf, MONTHS)
            if len(bars) < 1000:
                print(f"  {sym}: only {len(bars)} bars — skipped")
                continue
            atr = atr14(bars)
            sweeps = sweep_events(bars, atr)
            trades = find_trades(bars, atr, sweeps)
            a = [t for t in trades if t["cohort_a"]]
            a2 = [t for t in trades if t["cohort_a2"]]
            c = random_control(bars, atr, trades, rng)
            pooled["a"] += a
            pooled["a2"] += a2
            pooled["b"] += trades
            pooled["c"] += c
            if a:
                m = sum(t["net"] for t in a) / len(a)
                pos["a"] += 1 if m > 0 else 0
                print(f"  {sym:<5} b={len(trades):>4} a={len(a):>3} "
                      f"a2={len(a2):>3}  a_netR={m:+.4f}")
            else:
                print(f"  {sym:<5} b={len(trades):>4} a=  0")
        print(f"\n  ── pooled ({tf}) ──   [P1: a>b, a>c, b~c · "
              f"P2 verdict: a CI-low>0 · P3: a2>=a · P4: net decides]")
        report("(a) 位置條件", pooled["a"])
        report("(a2) a∧大單proxy", pooled["a2"])
        report("(b) 全部 IFVG", pooled["b"])
        report("(c) 隨機對照", pooled["c"])
        print(f"  (a) positive coins: {pos['a']}/9")


if __name__ == "__main__":
    main()
