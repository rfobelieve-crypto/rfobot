# -*- coding: utf-8 -*-
"""Stage 0b — turn raw 1-minute klines into the canonical bar table.

    data/bars/{COIN}.parquet
        ts int64 ms (UTC, bar OPEN, left-closed, CONTINUOUS 1-min index)
        open high low close volume n_trades      float64, NaN on missing bars
        atr_1h      mean true range over the last 60 1-min bars
        atr_h14     Wilder ATR(14) on 1-HOUR bars, through the last COMPLETED
                    hour before ts
        tick_size   constant column
    data/quality/{COIN}.md

Two ATR columns on purpose
--------------------------
The plan says `atr_1h (rolling 60 bar)`.  On a 1-minute index that is the mean
PER-MINUTE true range over the last hour -- roughly an order of magnitude
smaller than an hourly range, and `bin_size = atr/20` eats it directly.  The
frozen sweep engine and every effect size on this research line are expressed
in units of ATR(14) on 1-HOUR bars.  Rather than silently pick one, both are
carried and Stage 1 reports its diff distribution under both.

Look-ahead
----------
`atr_1h[i]` uses bars [i-59, i]; bar i is closed at ts[i]+60s, so this value is
usable at any t_ref >= ts[i]+60s -- which is exactly how Stage 4 consumes it.
`atr_h14[i]` uses only hours that CLOSED strictly before ts[i].  That matters:
the frozen engine reads ATR(14) at the event's own hourly bar, but this
pipeline timestamps the sweep INSIDE that hour, where the hour's own range is
not yet known.  Using it would be look-ahead (mistake.md 2026-09-03).

Missing bars are materialised as NaN rows, never skipped -- a silently absent
minute is indistinguishable from a quiet one, and the gap report needs them.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
OUT = HERE / "data" / "bars"
QUALITY = HERE / "data" / "quality"
MIN_MS = 60_000
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
ATR_MIN_VALID = 30          # of the 60, at least this many TRs must be real


def tick_sizes(syms):
    p = HERE / "data" / "ticks.json"
    if p.exists():
        d = json.loads(p.read_text())
        if all(s in d for s in syms):
            return d
    req = urllib.request.Request("https://api.binance.com/api/v3/exchangeInfo",
                                 headers={"User-Agent": "poc-stage0/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        info = json.loads(r.read().decode())
    d = {}
    for s in info["symbols"]:
        if not s["symbol"].endswith("USDT"):
            continue
        for f in s["filters"]:
            if f["filterType"] == "PRICE_FILTER":
                d[s["symbol"][:-4]] = float(f["tickSize"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({k: v for k, v in d.items()}, indent=1))
    return d


def true_range(high, low, close):
    prev = pd.Series(close).ffill().shift(1).to_numpy()
    a = high - low
    b = np.abs(high - prev)
    c = np.abs(low - prev)
    return np.nanmax(np.vstack([a, b, c]), axis=0)


def atr_minutes(tr, window=60, min_valid=ATR_MIN_VALID):
    s = pd.Series(tr)
    return s.rolling(window, min_periods=min_valid).mean().to_numpy()


def atr_hourly_wilder(df, period=14):
    """Wilder ATR(14) on 1h bars, mapped back to minutes with a one-hour lag.

    The value attached to minute t comes from hours that closed strictly
    before t, so it is knowable at t.
    """
    h = df.set_index(pd.to_datetime(df["ts"], unit="ms", utc=True))
    agg = h.resample("1h").agg(open=("open", "first"), high=("high", "max"),
                               low=("low", "min"), close=("close", "last"))
    tr = true_range(agg["high"].to_numpy(), agg["low"].to_numpy(),
                    agg["close"].to_numpy())
    a = pd.Series(tr, index=agg.index).ewm(alpha=1 / period, adjust=False,
                                           min_periods=period).mean()
    # the ATR of hour H is known at the END of hour H -> valid from H+1 onward
    a = a.shift(1)
    return a.reindex(h.index, method="ffill").to_numpy()


def build(sym, tick):
    raw = pd.read_csv(RAW / f"{sym}_1m.csv")
    raw = raw.drop_duplicates(subset="open_ms").sort_values("open_ms")
    lo, hi = int(raw.open_ms.iloc[0]), int(raw.open_ms.iloc[-1])
    full = np.arange(lo, hi + MIN_MS, MIN_MS, dtype=np.int64)
    df = pd.DataFrame({"ts": full}).merge(
        raw.rename(columns={"open_ms": "ts"}), on="ts", how="left")

    for c in ("open", "high", "low", "close", "volume", "n_trades",
              "taker_buy_base"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    tr = true_range(df["high"].to_numpy(), df["low"].to_numpy(),
                    df["close"].to_numpy())
    df["atr_1h"] = atr_minutes(tr)
    df["atr_h14"] = atr_hourly_wilder(df)
    # delta = taker buy - taker sell = 2*taker_buy - volume.  The maker/taker
    # flag is the matching engine's own, not a tick-rule guess: verified
    # 2026-09-06 against aggTrades `is_buyer_maker` over 1,440 minutes
    # (max relative error 4.8e-16, correlation 1.0000000000).
    df["delta"] = 2.0 * df["taker_buy_base"] - df["volume"]
    df["tick_size"] = float(tick)

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"{sym}.parquet", index=False)
    return df


# ------------------------------------------------------------------- gates
def run_asserts(df):
    """Stage 0 asserts.

    The plan's asserts are stated over the whole frame; once missing minutes
    are materialised as NaN rows (which the plan also requires) the OHLC ones
    can only hold over PRESENT rows -- NaN comparisons are False.  The two
    requirements conflict as written, so the OHLC checks are scoped to present
    rows and the gap count is reported separately.  Stated, not silently done.
    """
    fails = []
    ts = df["ts"].to_numpy()
    if not (np.diff(ts) > 0).all():
        fails.append("ts not strictly increasing")
    if not (np.diff(ts) == MIN_MS).all():
        fails.append("ts index not continuous at 1 min")
    p = df.dropna(subset=["open", "high", "low", "close"])
    if not (p["high"] >= p["low"]).all():
        fails.append("high < low on some present bar")
    if not (p["high"] >= p[["open", "close"]].max(axis=1)).all():
        fails.append("high < max(open, close)")
    if not (p["low"] <= p[["open", "close"]].min(axis=1)).all():
        fails.append("low > min(open, close)")
    v = df["volume"].dropna()
    if not (v >= 0).all():
        fails.append("negative volume")
    tb = df.dropna(subset=["taker_buy_base", "volume"])
    if len(tb) and not ((tb["taker_buy_base"] >= -1e-9)
                        & (tb["taker_buy_base"] <= tb["volume"] + 1e-9)).all():
        fails.append("taker_buy_base outside [0, volume]")
    if df["tick_size"].nunique() != 1:
        fails.append("tick_size not constant")
    return fails


def daily_volume_crosscheck(sym, df, n_days=5, seed=20260906):
    """Independent check: 1-min volume summed per day vs the exchange's own
    daily kline.  A different endpoint, so it catches a bad download rather
    than confirming it."""
    rng = np.random.default_rng(seed)
    d = df.dropna(subset=["volume"]).copy()
    d["day"] = pd.to_datetime(d["ts"], unit="ms", utc=True).dt.floor("1D")
    days = d["day"].unique()
    days = days[5:-2]                       # avoid partial first/last days
    pick = rng.choice(days, size=min(n_days, len(days)), replace=False)
    out = []
    for day in sorted(pick):
        start = int(pd.Timestamp(day).value // 1_000_000)
        url = (f"https://api.binance.com/api/v3/klines?symbol={sym}USDT"
               f"&interval=1d&startTime={start}&limit=1")
        req = urllib.request.Request(url, headers={"User-Agent": "poc-stage0/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            k = json.loads(r.read().decode())
        if not k:
            continue
        theirs = float(k[0][5])
        ours = float(d.loc[d["day"] == day, "volume"].sum())
        out.append((str(pd.Timestamp(day).date()), ours, theirs,
                    abs(ours - theirs) / theirs if theirs else np.nan))
    return out


def quality_report(sym, df, asserts, xcheck):
    ts = pd.to_datetime(df["ts"], unit="ms", utc=True)
    present = df["open"].notna()
    n = len(df)
    missing = int((~present).sum())
    # longest run of consecutive missing bars
    run = best = 0
    for ok in present.to_numpy():
        run = 0 if ok else run + 1
        best = max(best, run)
    zero_vol = int((df["volume"].fillna(-1) == 0).sum())
    by_day = pd.DataFrame({"day": ts.dt.floor("1D"), "present": present})
    per_day = by_day.groupby("day")["present"].agg(["size", "sum"])
    per_day["missing"] = per_day["size"] - per_day["sum"]
    worst = per_day.sort_values("missing", ascending=False).head(5)
    by_hour = by_day.assign(h=ts.dt.hour).groupby("h")["present"].apply(
        lambda s: 1 - s.mean())

    gap_pct = missing / n * 100
    zero_pct = zero_vol / max(1, int(present.sum())) * 100
    verdict = []
    verdict.append(("asserts", "PASS" if not asserts else "FAIL: " + "; ".join(asserts)))
    verdict.append(("gap <= 1%", f"{gap_pct:.4f}%  " + ("PASS" if gap_pct <= 1 else "FAIL -> DO NOT USE DOWNSTREAM")))
    verdict.append(("zero-volume bars < 0.5%", f"{zero_pct:.4f}%  " + ("PASS" if zero_pct < 0.5 else "FAIL")))
    if xcheck:
        worst_err = max(x[3] for x in xcheck)
        verdict.append(("daily volume vs exchange < 2%",
                        f"max {worst_err*100:.4f}%  " + ("PASS" if worst_err < 0.02 else "FAIL")))

    L = [f"# Stage 0 quality — {sym}", ""]
    L.append(f"- rows (continuous 1-min index): **{n:,}**")
    L.append(f"- coverage: {ts.iloc[0]} .. {ts.iloc[-1]}")
    L.append(f"- missing bars: **{missing:,}** ({gap_pct:.4f}%), longest run **{best}**")
    L.append(f"- zero-volume present bars: **{zero_vol:,}** ({zero_pct:.4f}% of present)")
    L.append("")
    L.append("## gate")
    L += [f"- {k}: {v}" for k, v in verdict]
    L.append("")
    L.append("## worst days by missing minutes")
    L.append("")
    L.append("| day | expected | present | missing |")
    L.append("|---|---|---|---|")
    for day, r in worst.iterrows():
        L.append(f"| {pd.Timestamp(day).date()} | {int(r['size'])} | {int(r['sum'])} | {int(r['missing'])} |")
    L.append("")
    L.append("## missing-rate by UTC hour")
    L.append("")
    L.append("| hour | missing rate |")
    L.append("|---|---|")
    for h, v in by_hour.items():
        L.append(f"| {h:02d} | {v*100:.3f}% |")
    if xcheck:
        L.append("")
        L.append("## daily volume cross-check (independent endpoint)")
        L.append("")
        L.append("| day | ours | exchange | rel err |")
        L.append("|---|---|---|---|")
        for day, ours, theirs, err in xcheck:
            L.append(f"| {day} | {ours:,.3f} | {theirs:,.3f} | {err*100:.5f}% |")
    QUALITY.mkdir(parents=True, exist_ok=True)
    (QUALITY / f"{sym}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    ok = not asserts and gap_pct <= 1 and zero_pct < 0.5 and \
        (not xcheck or max(x[3] for x in xcheck) < 0.02)
    return ok, gap_pct, zero_pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", default=",".join(CORE9))
    ap.add_argument("--no-xcheck", action="store_true")
    a = ap.parse_args()
    syms = [s.strip().upper() for s in a.syms.split(",") if s.strip()]
    ticks = tick_sizes(syms)
    allok = True
    for s in syms:
        if not (RAW / f"{s}_1m.csv").exists():
            print(f"{s:5s} raw missing, skipped")
            continue
        df = build(s, ticks[s])
        fails = run_asserts(df)
        xc = [] if a.no_xcheck else daily_volume_crosscheck(s, df)
        ok, gap, zero = quality_report(s, df, fails, xc)
        allok &= ok
        xs = f" volxcheck_max={max(x[3] for x in xc)*100:.4f}%" if xc else ""
        print(f"{s:5s} rows={len(df):>9,} gap={gap:.4f}% zerovol={zero:.4f}%"
              f"{xs}  asserts={'ok' if not fails else fails}  -> {'PASS' if ok else 'FAIL'}")
    print("\nStage 0 gate:", "ALL PASS" if allok else "FAILED")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
