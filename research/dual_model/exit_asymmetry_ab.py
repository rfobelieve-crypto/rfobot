"""Exit ASYMMETRY A/B (2026-06-07).

CONTEXT: Entry is maxed (AUC ceiling); exit is the one within-data lever (wall-2).
The live exit is SYMMETRIC long/short: 3xATR(14) trailing stop + 72h time cap +
opposite-signal flip. But payoff is ASYMMETRIC -- shorts ride big fast down-moves
(Strong DOWN +34bps), longs grind smaller (Strong UP +16bps; the move is real but
shallower, volatility leverage effect).

HYPOTHESIS (user): give longs a quicker/tighter exit (take the smaller move before
it mean-reverts) and let shorts run wider (capture the big leg). Test asymmetric
exits vs the symmetric live baseline on an EVENT-DRIVEN replay (one position at a
time, real exit rules on forward 1h OHLC paths) -- NOT the fixed-4h proxy.

DISCIPLINE: hypothesis-driven small rule set with round-number params (NOT a fine
sweep -> overfit). Report long/short split + per-month stability. Prefer robust
over optimal. Intra-bar: adverse (stop) checked before favorable (TP/extreme update)
= conservative. ATR fixed at entry. Costs: round-trip COST applied per trade.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.dual_model.position_sizing_oos_backtest import COST, OOS_PATH, decode

KL = "market_data/raw_data/binance_klines_1h.parquet"
ATR_PERIOD = 14
TIME_CAP_H = 72


def wilder_atr(h, l, c, period=ATR_PERIOD):
    pc = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def replay(bars, sig_dir, sig_trade, atr, rule):
    """Event-driven one-position replay. bars: df with open/high/low/close indexed
    by ts (aligned to sig arrays). Returns list of trade dicts."""
    n = len(bars)
    hi = bars["high"].values; lo = bars["low"].values; cl = bars["close"].values
    trades = []
    pos = None  # dict: side, entry_px, entry_i, extreme, atr0
    for i in range(n):
        if pos is not None:
            side = pos["side"]; a0 = pos["atr0"]; ext = pos["extreme"]
            exit_px = None; reason = None
            k = rule["k_long"] if side == 1 else rule["k_short"]
            stop = ext - k * a0 if side == 1 else ext + k * a0
            # 1) adverse: trailing stop hit this bar (conservative, before extreme update)
            if side == 1 and lo[i] <= stop:
                exit_px, reason = stop, "trail"
            elif side == -1 and hi[i] >= stop:
                exit_px, reason = stop, "trail"
            # 2) favorable: take-profit (asymmetric, optional per side)
            if exit_px is None:
                tp = rule.get("tp_long") if side == 1 else rule.get("tp_short")
                if tp is not None:
                    tp_px = pos["entry_px"] + side * tp * a0
                    if (side == 1 and hi[i] >= tp_px) or (side == -1 and lo[i] <= tp_px):
                        exit_px, reason = tp_px, "tp"
            # 3) opposite-signal flip (at close)
            if exit_px is None and rule.get("opp_exit", True):
                if (side == 1 and sig_dir[i] == -1) or (side == -1 and sig_dir[i] == 1):
                    exit_px, reason = cl[i], "opp"
            # 4) time cap
            if exit_px is None and (i - pos["entry_i"]) >= TIME_CAP_H:
                exit_px, reason = cl[i], "time"
            if exit_px is not None:
                r = side * (exit_px / pos["entry_px"] - 1.0) - COST
                trades.append(dict(side=side, ret=r, hold=i - pos["entry_i"],
                                   reason=reason, ts=bars.index[pos["entry_i"]]))
                pos = None
            else:
                pos["extreme"] = max(ext, hi[i]) if side == 1 else min(ext, lo[i])
        if pos is None and sig_trade[i] != 0:
            pos = dict(side=int(sig_trade[i]), entry_px=cl[i], entry_i=i,
                       extreme=hi[i] if sig_trade[i] == 1 else lo[i], atr0=atr[i])
    return trades


def stats(trades, label):
    if not trades:
        print(f"  {label:30s} no trades"); return None
    df = pd.DataFrame(trades)
    def m(sub):
        if len(sub) == 0: return "  -"
        eq = (1 + sub["ret"]).cumprod()
        mdd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
        return (f"n={len(sub):3d} hold={sub['hold'].mean():4.1f}h "
                f"win={ (sub['ret']>0).mean()*100:4.0f}% "
                f"bps={sub['ret'].mean()*1e4:+6.1f} "
                f"tot={eq.iloc[-1]:5.2f} mdd={mdd:5.1f}")
    print(f"  {label}")
    print(f"      ALL   {m(df)}")
    print(f"      LONG  {m(df[df.side==1])}")
    print(f"      SHORT {m(df[df.side==-1])}")
    return df


def main():
    oos = pd.read_parquet(OOS_PATH).sort_index()
    d, t, _ = decode(oos["pred_ret"].values)
    oos["dir"], oos["tier"] = d, t
    kl = pd.read_parquet(KL)[["open", "high", "low", "close"]].sort_index()
    kl["atr"] = wilder_atr(kl["high"], kl["low"], kl["close"])

    # align klines to the signal window (+ tail so late positions can close)
    start = oos.index[0]
    bars = kl[kl.index >= start].copy()
    sig = oos.reindex(bars.index)
    dir_map = {"UP": 1, "DOWN": -1, "NEUTRAL": 0}
    sig_dir = sig["dir"].map(dir_map).fillna(0).astype(int).values
    # tradeable entry = Strong or Moderate signal (non-neutral)
    sig_trade = np.where(sig["tier"].isin(["Strong", "Moderate"]).values,
                         sig_dir, 0)
    atr = bars["atr"].values
    print(f"bars={len(bars)} signals(entry-eligible)={int((sig_trade!=0).sum())} "
          f"span {bars.index[0].date()}..{bars.index[-1].date()} cost={COST*1e4:.0f}bps\n")

    rules = {
        "LIVE symmetric (trail3, opp, 72h)":
            dict(k_long=3.0, k_short=3.0, opp_exit=True),
        "asym: long TP@2ATR / short trail3":
            dict(k_long=3.0, k_short=3.0, tp_long=2.0, opp_exit=True),
        "asym: long TP@1.5 / short trail4 (run)":
            dict(k_long=3.0, k_short=4.0, tp_long=1.5, opp_exit=True),
        "asym trail: long2 / short4":
            dict(k_long=2.0, k_short=4.0, opp_exit=True),
        "no opp-exit, trail3 sym (let both run)":
            dict(k_long=3.0, k_short=3.0, opp_exit=False),
    }
    results = {}
    for name, rule in rules.items():
        tr = replay(bars, sig_dir, sig_trade, atr, rule)
        results[name] = stats(tr, name)
        print()

    # per-month stability: best asymmetric vs LIVE (LONG leg, the thing we changed)
    base = results["LIVE symmetric (trail3, opp, 72h)"]
    print("=" * 64)
    print("per-month LONG net_bps: LIVE vs each asym (does the long change help "
          "consistently, not one month?)")
    for name, df in results.items():
        if df is None or "asym" not in name: continue
        merged = []
        for src, lab in [(base, "live"), (df, "new")]:
            g = src[src.side == 1].copy()
            g["mon"] = pd.to_datetime(g["ts"]).dt.to_period("M").astype(str)
            merged.append(g.groupby("mon")["ret"].mean() * 1e4)
        cmp = pd.concat(merged, axis=1, keys=["live", "new"]).dropna()
        cmp["diff"] = cmp["new"] - cmp["live"]
        pos = (cmp["diff"] > 0).mean() * 100
        print(f"\n  [{name}]  long-leg per-month diff (new-live):")
        print("   " + "  ".join(f"{m}:{d:+.0f}" for m, d in cmp["diff"].items())
              + f"   | {pos:.0f}% months positive")


if __name__ == "__main__":
    main()
