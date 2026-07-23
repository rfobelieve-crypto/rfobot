"""Zero-cost prior check for multi-coin signal independence (user question:
"ETH is basically the same as BTC, is it worth porting?").

Uses only free Binance klines (no Coinglass quota). Computes, on the exact
V7 target (4h TWAP path return over 1h bars):
  1. return correlation BTC-ETH / BTC-SOL / ETH-SOL
  2. extreme-hour co-occurrence: of coin X's top-5% |target| hours (proxy for
     the Strong opportunity set), what fraction are also BTC top-5% hours
  3. among co-occurring extremes, same-sign fraction (directional overlap)

If ETH's extreme co-occurrence with BTC is already very high, the Step 3
overlap gate (<50%) is likely to fail before we spend quota on the full
ETH backfill; a lower number keeps ETH viable.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

BINANCE = "https://api.binance.com/api/v3/klines"
COINS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
DAYS = 730  # ~2 years of 1h bars
TOP_PCT = 0.05  # mirrors Strong tier selection ratio


def fetch_1h_closes(symbol: str, days: int) -> pd.Series:
    end = int(time.time() * 1000)
    start = end - days * 24 * 3600 * 1000
    rows = []
    cur = start
    while cur < end:
        resp = requests.get(BINANCE, params={
            "symbol": symbol, "interval": "1h",
            "startTime": cur, "limit": 1000,
        }, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1][0] + 3600_000
        time.sleep(0.15)
    df = pd.DataFrame(rows, columns=[
        "open_time", "o", "h", "l", "c", "v", "close_time",
        "qv", "n", "tbb", "tbq", "ig"])
    ts = pd.to_datetime(df["open_time"], unit="ms")
    closes = pd.Series(df["c"].astype(float).values, index=ts)
    return closes[~closes.index.duplicated(keep="first")]


def twap_path_ret_4h(close: pd.Series) -> pd.Series:
    """V7 target: mean(close[t+1..t+4]) / close[t] - 1."""
    fwd_mean = (close.shift(-1) + close.shift(-2) + close.shift(-3) + close.shift(-4)) / 4
    return fwd_mean / close - 1


def main():
    targets = {}
    for coin, sym in COINS.items():
        closes = fetch_1h_closes(sym, DAYS)
        targets[coin] = twap_path_ret_4h(closes)
        print(f"{coin}: {len(closes)} bars ({closes.index[0]} -> {closes.index[-1]})")

    df = pd.DataFrame(targets).dropna()
    print(f"\naligned hours: {len(df)}")

    print("\n== 4h TWAP path return correlation ==")
    print(df.corr().round(3).to_string())

    print(f"\n== extreme-hour co-occurrence (top {TOP_PCT:.0%} |target|) ==")
    extreme = {c: df[c].abs() >= df[c].abs().quantile(1 - TOP_PCT) for c in df}
    for a in df.columns:
        for b in df.columns:
            if a >= b:
                continue
            both = extreme[a] & extreme[b]
            co_a = both.sum() / extreme[a].sum()
            co_b = both.sum() / extreme[b].sum()
            same_sign = float((np.sign(df[a][both]) == np.sign(df[b][both])).mean()) if both.sum() else float("nan")
            print(f"{a}-{b}: P({b} extreme | {a} extreme)={co_a:.1%}  "
                  f"P({a} extreme | {b} extreme)={co_b:.1%}  "
                  f"same-sign among co-extreme={same_sign:.1%}  (n_both={both.sum()})")

    # directional version: top-5% signed UP set and bottom-5% DOWN set,
    # closer to how Strong UP / Strong DOWN are actually selected
    print(f"\n== directional co-occurrence (top {TOP_PCT:.0%} UP set / DOWN set) ==")
    for side, q in [("UP", 1 - TOP_PCT), ("DOWN", TOP_PCT)]:
        sets = {}
        for c in df.columns:
            thr = df[c].quantile(q)
            sets[c] = df[c] >= thr if side == "UP" else df[c] <= thr
        for a in df.columns:
            for b in df.columns:
                if a >= b:
                    continue
                both = sets[a] & sets[b]
                print(f"{side} {a}-{b}: P(both)={both.sum() / sets[a].sum():.1%} of {a}'s set  (n_both={both.sum()})")


if __name__ == "__main__":
    main()
