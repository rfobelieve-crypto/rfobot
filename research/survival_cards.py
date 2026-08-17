"""Survival cards — does each strategy's edge die where its MECHANISM says
it should?  PRE-REGISTERED 2026-08-17 (TODO §0.49); definitions and
predictions frozen and committed before this script produced any number.

The premise: performance monitoring cannot answer "is the edge alive" on
any useful timescale (a +7bps edge under ~100bps/trade noise needs hundreds
of trades to distinguish from zero).  So we test the PRECONDITION instead:
each strategy card names the mechanism it harvests, the observable regime
variable that proxies the mechanism's health, and a frozen prediction of
where returns die.  Terrain-campaign ritual: predictions first, fixed
buckets, full-grid reporting, empty buckets treated as instrument failure.

Card 1  V7 (4h mean reversion, short-biased)  — dies in strong trends.
Card 2  Sweep-failure (exhaustion at pools)   — dies when pierces stop
        returning (one-way flow turns raids into genuine breakouts).

Frozen: trend_z(t) = (ln C_t - ln C_{t-720}) / (sigma_1h * sqrt(720)),
sigma_1h = std of trailing 720 1h log returns; buckets CALM |z|<1,
MID 1<=|z|<2, TREND |z|>=2 (absolute thresholds, no data-derived cuts).
Read-only research code.
"""
from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import sweep_core as SC                                    # noqa: E402

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
WINDOW = 720          # 30d of 1h bars
Z_MID, Z_TREND = 1.0, 2.0
BOOT_N = 2000
SEED = 7


def bucket_of(z: float) -> str:
    az = abs(z)
    if az >= Z_TREND:
        return "TREND"
    if az >= Z_MID:
        return "MID"
    return "CALM"


def trend_z_series(bars) -> dict[int, float]:
    """hour_ts(s) -> trailing trend_z.  Pure trailing; first WINDOW bars
    have no value."""
    lc = [math.log(b[SC.C]) for b in bars]
    rets = [lc[i] - lc[i - 1] for i in range(1, len(lc))]
    out: dict[int, float] = {}
    for i in range(WINDOW, len(bars)):
        w = rets[i - WINDOW:i]
        m = sum(w) / len(w)
        var = sum((x - m) ** 2 for x in w) / len(w)
        sd = math.sqrt(var)
        if sd <= 0:
            continue
        z = (lc[i] - lc[i - WINDOW]) / (sd * math.sqrt(WINDOW))
        out[bars[i][0] // 3600 * 3600] = z
    return out


def clustered_diff_ci(a: list[tuple[str, float]], b: list[tuple[str, float]]):
    """Day-clustered bootstrap CI on mean(a) - mean(b); items are
    (day, value)."""
    if not a or not b:
        return 0.0, 0.0, 0.0
    da, db = defaultdict(list), defaultdict(list)
    for d, v in a:
        da[d].append(v)
    for d, v in b:
        db[d].append(v)
    ka, kb = list(da.values()), list(db.values())
    rng = random.Random(SEED)
    diffs = []
    for _ in range(BOOT_N):
        fa = [x for _ in range(len(ka))
              for x in ka[rng.randrange(len(ka))]]
        fb = [x for _ in range(len(kb))
              for x in kb[rng.randrange(len(kb))]]
        diffs.append(sum(fa) / len(fa) - sum(fb) / len(fb))
    diffs.sort()
    point = (sum(v for _, v in a) / len(a)) - (sum(v for _, v in b) / len(b))
    return point, diffs[int(0.025 * BOOT_N)], diffs[int(0.975 * BOOT_N)]


def day_of(ts_s: int) -> str:
    return datetime.fromtimestamp(ts_s, tz=timezone.utc).strftime("%Y-%m-%d")


# ── Card 1: V7 ──────────────────────────────────────────────────────────

def card_v7() -> None:
    from shared.db import get_db_conn
    bars = SC.load_csv(str(CACHE / "BTCUSDT_1h.csv"))
    zmap = trend_z_series(bars)
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT signal_time, direction, strength, actual_return_4h, "
                "correct FROM tracked_signals "
                "WHERE actual_return_4h IS NOT NULL "
                "AND direction IN ('UP','DOWN') "
                "AND strength IN ('Strong','Moderate')")
            rows = cur.fetchall()
    finally:
        conn.close()

    print("\n════ 卡 1：V7（機制＝4h 均值回歸）════")
    for tier in ("Strong", "Moderate"):
        grid: dict[tuple[str, str], list[tuple[str, float, int]]] = \
            defaultdict(list)
        miss = 0
        for r in rows:
            if r["strength"] != tier:
                continue
            ts = int(r["signal_time"].replace(tzinfo=timezone.utc).timestamp())
            z = zmap.get(ts // 3600 * 3600)
            if z is None:
                miss += 1
                continue
            sgn = 1 if r["direction"] == "UP" else -1
            aligned = "順勢" if (sgn > 0) == (z > 0) else "逆勢"
            pnl = sgn * float(r["actual_return_4h"]) * 1e4     # bps
            grid[(bucket_of(z), aligned)].append(
                (day_of(ts), pnl, int(r["correct"] or 0)))
        n_all = sum(len(v) for v in grid.values())
        print(f"\n  {tier}  n={n_all}  (z 無值跳過 {miss})")
        print(f"  {'bucket':<7}{'側':<5}{'n':>5}{'WR':>7}{'mean bps':>10}")
        for bk in ("CALM", "MID", "TREND"):
            for al in ("順勢", "逆勢"):
                g = grid[(bk, al)]
                if g:
                    wr = 100 * sum(c for _, _, c in g) / len(g)
                    mb = sum(p for _, p, _ in g) / len(g)
                    print(f"  {bk:<7}{al:<5}{len(g):>5}{wr:>6.0f}%{mb:>+10.1f}")
                else:
                    print(f"  {bk:<7}{al:<5}{0:>5}   —  (空桶=儀器嫌疑)")
        # frozen predictions, pooled across alignment
        by_bucket = {bk: [(d, c) for al in ("順勢", "逆勢")
                          for d, _, c in grid[(bk, al)]]
                     for bk in ("CALM", "MID", "TREND")}
        wr = {bk: (100 * sum(c for _, c in v) / len(v) if v else None)
              for bk, v in by_bucket.items()}
        if wr["CALM"] is not None and wr["TREND"] is not None:
            pt, lo, hi = clustered_diff_ci(
                [(d, float(c)) for d, c in by_bucket["CALM"]],
                [(d, float(c)) for d, c in by_bucket["TREND"]])
            mono = (wr["CALM"] >= (wr["MID"] or 0) >= (wr["TREND"] or 0))
            print(f"  P1 CALM−TREND WR: {pt*100:+.1f}pp "
                  f"CI95[{lo*100:+.1f},{hi*100:+.1f}]pp "
                  f"({'≥5pp ✓' if pt*100 >= 5 else '<5pp ✗'})")
            print(f"  P2 單調性 CALM≥MID≥TREND: "
                  f"{wr['CALM']:.0f}/{(wr['MID'] or 0):.0f}"
                  f"/{wr['TREND']:.0f} {'✓' if mono else '✗'}")


# ── Card 2: sweep-failure ───────────────────────────────────────────────

def card_sf() -> None:
    print("\n════ 卡 2：掃單失敗（機制＝竭盡回歸）════")
    pooled: dict[str, list[tuple[str, float]]] = defaultdict(list)
    coin_diff: list[tuple[str, float]] = []
    fills: dict[str, list[int]] = defaultdict(list)     # bucket -> [filled?]
    print(f"  {'sym':<6}{'n':>5}{'CALM':>9}{'MID':>9}{'TREND':>9}"
          f"{'CALM−TREND':>12}")
    for sym in CORE9:
        fp = CACHE / f"{sym}USDT_1h.csv"
        if not fp.exists():
            print(f"  {sym:<6} no cache — skipped")
            continue
        bars = SC.load_csv(str(fp))
        zmap = trend_z_series(bars)
        trades = SC.backtest_symbol(bars)
        by_b: dict[str, list[float]] = defaultdict(list)
        for fill_ts, _exit_ts, R, *_ in trades:
            z = zmap.get(int(fill_ts) // 3600 * 3600)
            if z is None:
                continue
            bk = bucket_of(z)
            by_b[bk].append(R)
            pooled[bk].append((day_of(int(fill_ts)), R))
        # SF-P3 fill-rate: every sweep event, filled within W or not
        h = [b[SC.H] for b in bars]
        l = [b[SC.L] for b in bars]
        n = len(bars)
        for e in SC.detect_sweeps(bars):
            j, lvl = e["j"], e["level"]
            z = zmap.get(bars[j][0] // 3600 * 3600)
            if z is None:
                continue
            kd = 1 if e["kind"] == "buy" else -1
            filled = any(
                (l[f] <= lvl if kd == 1 else h[f] >= lvl)
                for f in range(j + 1, min(j + 1 + SC.W, n)))
            fills[bucket_of(z)].append(1 if filled else 0)
        m = {bk: (sum(v) / len(v) if v else None) for bk, v in by_b.items()}
        if m.get("CALM") is not None and m.get("TREND") is not None:
            coin_diff.append((sym, m["CALM"] - m["TREND"]))
        print(f"  {sym:<6}{sum(len(v) for v in by_b.values()):>5}"
              + "".join(f"{(m.get(bk) if m.get(bk) is not None else float('nan')):>+9.4f}"
                        if m.get(bk) is not None else f"{'—':>9}"
                        for bk in ("CALM", "MID", "TREND"))
              + (f"{m['CALM']-m['TREND']:>+12.4f}"
                 if m.get("CALM") is not None and m.get("TREND") is not None
                 else f"{'—':>12}"))

    print("\n  ── pooled ──")
    for bk in ("CALM", "MID", "TREND"):
        v = pooled[bk]
        if v:
            mr = sum(x for _, x in v) / len(v)
            print(f"  {bk:<7} n={len(v):>5}  meanR={mr:+.4f}")
        else:
            print(f"  {bk:<7} n=0  (空桶=儀器嫌疑)")
    pt, lo, hi = clustered_diff_ci(pooled["CALM"], pooled["TREND"])
    npos = sum(1 for _, d in coin_diff if d > 0)
    print(f"  SF-P1 CALM−TREND meanR: {pt:+.4f} CI95[{lo:+.4f},{hi:+.4f}] "
          f"({'✓' if pt > 0 else '✗'})")
    print(f"  SF-P2 逐幣同號 (CALM−TREND)>0: {npos}/{len(coin_diff)} "
          f"({'≥6/9 ✓' if npos >= 6 else '<6/9 ✗'})")
    print("  SF-P3 回填率 by bucket: "
          + "  ".join(f"{bk} {100*sum(v)/len(v):.0f}% (n={len(v)})"
                      for bk, v in ((b, fills[b]) for b in
                                    ("CALM", "MID", "TREND")) if v))


if __name__ == "__main__":
    card_v7()
    card_sf()
