"""
Prototype: can we quantify "巨量換手" (volume churn at consolidation) as a
DAILY macro overlay using exchange volume only? (weak proxy, advisory only)

Definition of churn (大資金提前建倉 = 盤整放量):
  - high volume relative to history  (vol_z over rolling 90d)
  - price NOT trending = stuck in a range (low efficiency ratio over window)
  combine: churn = volume is heavy WHILE price goes nowhere.
  This is the exchange-data shadow of "籌碼堆積" — it CANNOT see holder cost
  basis (that needs on-chain URPD), but it can see "heavy trade, flat price".

Validation is eyeball-only by design: BTC has 2-3 cycle bottoms, no stats power.
We check: do churn spikes precede forward upside? + what is the reading NOW.
"""
import requests
import numpy as np
import pandas as pd

URL = "https://api.binance.com/api/v3/klines"


def fetch_daily(symbol="BTCUSDT", limit=1000):
    r = requests.get(URL, params={"symbol": symbol, "interval": "1d", "limit": limit}, timeout=20)
    r.raise_for_status()
    cols = ["ts_open", "open", "high", "low", "close", "volume", "close_time",
            "quote_vol", "trades", "tb_base", "tb_quote", "ignore"]
    df = pd.DataFrame(r.json(), columns=cols)
    for c in ["open", "high", "low", "close", "volume", "quote_vol"]:
        df[c] = df[c].astype(float)
    df["date"] = pd.to_datetime(df["ts_open"], unit="ms")
    return df.set_index("date")


def compute_churn(df, vol_win=90, eff_win=20):
    c = df["close"]
    # use quote_vol (USD turnover) — closer to "真金白銀" than base volume
    qv = df["quote_vol"]
    # volume z-score vs trailing 90d (trailing only, no look-ahead)
    vol_mean = qv.rolling(vol_win).mean()
    vol_std = qv.rolling(vol_win).std()
    df["vol_z"] = (qv - vol_mean) / vol_std
    # efficiency ratio over eff_win: |net move| / sum(|daily move|); low = choppy/range
    net = (c - c.shift(eff_win)).abs()
    path = c.diff().abs().rolling(eff_win).sum()
    df["efficiency"] = net / path
    # churn score: heavy turnover AND price stuck. rolling-mean vol_z over the window
    df["vol_z_smooth"] = df["vol_z"].rolling(eff_win).mean()
    df["churn"] = df["vol_z_smooth"] * (1.0 - df["efficiency"].clip(0, 1))
    # forward 60d return (for eyeball validation only — uses future, NOT a feature)
    df["fwd_60d"] = c.shift(-60) / c - 1.0
    return df


def main():
    df = fetch_daily()
    print(f"BTC daily: {df.index.min().date()} -> {df.index.max().date()}  ({len(df)} bars)")
    df = compute_churn(df)

    valid = df.dropna(subset=["churn"])
    thr = valid["churn"].quantile(0.85)  # top 15% churn days = "巨量換手區"
    print(f"\nchurn threshold (top 15%): {thr:.2f}")

    # group consecutive churn-spike days into episodes
    spike = valid["churn"] >= thr
    episodes = []
    cur = None
    for dt, on in spike.items():
        if on and cur is None:
            cur = [dt, dt]
        elif on:
            cur[1] = dt
        elif cur is not None:
            episodes.append(tuple(cur)); cur = None
    if cur:
        episodes.append(tuple(cur))

    print(f"\n=== 換手堆積 episodes (top-15% churn, merged) ===")
    print(f"{'start':>10} {'end':>10} {'days':>4} {'px_avg':>9} {'fwd60d@start':>12}")
    for s, e in episodes:
        seg = df.loc[s:e]
        px = seg["close"].mean()
        f60 = df.loc[s, "fwd_60d"]
        f60s = f"{f60*100:+.0f}%" if pd.notna(f60) else "n/a(recent)"
        print(f"{str(s.date()):>10} {str(e.date()):>10} {len(seg):>4} {px:>9,.0f} {f60s:>12}")

    # eyeball validation: avg forward-60d after churn-spike onset vs baseline
    onset = valid[spike & ~spike.shift(1).fillna(False)]
    fwd_churn = onset["fwd_60d"].dropna()
    fwd_all = valid["fwd_60d"].dropna()
    print(f"\n=== forward 60d return (eyeball only, NOT stat-significant) ===")
    print(f"  after churn-spike onset: mean {fwd_churn.mean()*100:+.1f}%  median {fwd_churn.median()*100:+.1f}%  (n={len(fwd_churn)})")
    print(f"  all days baseline:       mean {fwd_all.mean()*100:+.1f}%  median {fwd_all.median()*100:+.1f}%  (n={len(fwd_all)})")

    # CURRENT reading
    last = df.iloc[-1]
    p = (valid["churn"] < last["churn"]).mean() * 100
    print(f"\n=== CURRENT reading ({df.index[-1].date()}, close ${last['close']:,.0f}) ===")
    print(f"  vol_z={last['vol_z']:+.2f}  efficiency={last['efficiency']:.2f}  churn={last['churn']:+.2f}")
    print(f"  churn percentile vs 1000d history: {p:.0f}%")
    print(f"  -> {'巨量換手出現中' if last['churn']>=thr else '尚未出現明顯換手 (符合你的判斷)'}")


if __name__ == "__main__":
    main()
