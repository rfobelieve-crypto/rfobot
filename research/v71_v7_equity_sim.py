"""
V7 Pure Signal Exit + 3xATR Trailing Stop — Equity Simulation
=============================================================
Task spec (user, 2026-05-16): "用 V7 + 3x ATR trailing stop, 本金 1000u,
槓桿 5x, 跑看看績效如何."

SIMULATION MODEL — read this before reading numbers:
  - Account: 1000 USDT, ONE position at a time, sequential, COMPOUNDING.
    (This is the only well-defined reading of "1000u + 5x": V7 averages ~28h
     holds and overlaps heavily — 7 concurrent positions cannot share one
     1000u account. One-at-a-time is also V7's natural deployment.)
  - Each trade: margin = full current equity, leverage 5x → notional = 5×equity.
  - Per-trade equity multiplier = 1 + 5×(signed_price_return − 0.0008 fee).
    Fee 0.08% round-trip is charged on NOTIONAL (so 5× the unleveraged drag).
  - Liquidation modelled: if a trade's leveraged loss reaches −100% of margin
    (≈ −20% adverse price move) the position is wiped. With a 3×ATR (~3%)
    trailing stop this should essentially never fire — reported for honesty.
  - Exit = first of: 3×ATR(14) trailing stop / opposite v7.1 signal / 72h cap.

CAVEATS the reader must keep in mind:
  - n is small (~one-position subsample of the 1112 signals). Treat regime
    splits as indicative only.
  - Funding cost NOT modelled; real 28h holds at 5× notional pay ~3-4 funding
    intervals — a real drag this sim omits (kept off for comparability with
    the earlier variant backtests).
  - At 5x, a single 3% trailing-stop-out ≈ −15% of the account. WR is ~67%
    but losing streaks compound hard — see MaxDD.
  - Backtest is WF-OOS; live degrades.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
PROJECT_ROOT = _HERE.parent
from verify_kernel_method_c import decode_tiers, atr_wilder, FEE_RT  # noqa: E402

KLINES = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"
V71_OOS = PROJECT_ROOT / "research" / "results" / "dual_model" / "direction_reg_oos_mse.parquet"
FEATURES = PROJECT_ROOT / "research" / "dual_model" / ".cache" / "features_all.parquet"
REPORT = PROJECT_ROOT / "research" / "results" / "v71_v7_equity_sim.md"
PLOT = PROJECT_ROOT / "research" / "results" / "v71_v7_equity_sim.png"

CAPITAL0 = 1000.0
LEVERAGE = 5.0
ATR_PERIOD = 14
TRAIL_MULT = 3.0
TIME_CAP_H = 72

log_lines: list[str] = []


def log(msg: str = ""):
    log_lines.append(msg)
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def _strip_tz(idx):
    return idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx


# ─── trade simulation ───────────────────────────────────────────────────────

def sim_trade(d, entry_i, k_open, k_high, k_low, k_close, decoded_dir,
              decoded_tier, idx, trail_dist, mode):
    """
    Simulate one trade from entry_i (entry at open[entry_i]).
    mode = "V7"  -> trailing stop + opposite signal + 72h cap
    mode = "V0"  -> fixed 4h exit, no stop (reference)
    Returns (exit_i, exit_px, reason).
    """
    entry_px = k_open[entry_i]
    n = len(idx)

    if mode == "V0":
        exit_i = min(entry_i + 4, n - 1)
        return exit_i, k_close[exit_i], "fixed_4h"

    opp = "DOWN" if d == "UP" else "UP"
    max_exit_i = min(entry_i + TIME_CAP_H, n - 1)
    # trailing-stop anchor: running favourable extreme
    trail_extreme = entry_px
    for j in range(entry_i, max_exit_i + 1):
        # stop level from extreme observed through bar j-1
        if d == "UP":
            stop = trail_extreme - trail_dist
            if k_open[j] <= stop:                    # gapped through
                return j, k_open[j], "trail_stop"
            if k_low[j] <= stop:
                return j, stop, "trail_stop"
            trail_extreme = max(trail_extreme, k_high[j])
        else:
            stop = trail_extreme + trail_dist
            if k_open[j] >= stop:
                return j, k_open[j], "trail_stop"
            if k_high[j] >= stop:
                return j, stop, "trail_stop"
            trail_extreme = min(trail_extreme, k_low[j])
        # time cap
        if j == max_exit_i:
            return j, k_close[j], "time_cap"
        # opposite v7.1 signal at bar j close -> exit next bar open
        tj = idx[j]
        if decoded_dir.get(tj) == opp and decoded_tier.get(tj) != "None":
            return j + 1, k_open[j + 1], "opp_signal"
    return max_exit_i, k_close[max_exit_i], "time_cap"


def run_sim(sigs, k, decoded, atr, regime, mode):
    """Sequential one-position-at-a-time equity simulation."""
    idx = k.index
    pos = {ts: i for i, ts in enumerate(idx)}
    k_open = k["open"].to_numpy(float)
    k_high = k["high"].to_numpy(float)
    k_low = k["low"].to_numpy(float)
    k_close = k["close"].to_numpy(float)
    ddir, dtier = decoded["direction"], decoded["tier"]

    equity = CAPITAL0
    busy_until = -1
    trades = []
    # per-bar mark-to-market equity series
    mark = np.full(len(idx), np.nan)

    for sig_ts in sigs.index:
        i = pos.get(sig_ts)
        if i is None or i < busy_until or i + 1 >= len(idx):
            continue
        d = sigs.at[sig_ts, "direction"]
        entry_i = i + 1
        a = float(atr.iloc[i])
        if not np.isfinite(a) or a <= 0:
            continue
        trail_dist = TRAIL_MULT * a
        exit_i, exit_px, reason = sim_trade(
            d, entry_i, k_open, k_high, k_low, k_close, ddir, dtier, idx,
            trail_dist, mode)
        entry_px = k_open[entry_i]
        r = (exit_px / entry_px - 1.0) * (1.0 if d == "UP" else -1.0)
        net_r = r - FEE_RT                                  # price-return space
        lev_mult = 1.0 + LEVERAGE * net_r
        liquidated = lev_mult <= 0.0
        lev_mult = max(lev_mult, 0.0)
        entry_equity = equity
        # mark-to-market the trade bar by bar
        for b in range(entry_i, exit_i):
            mr = (k_close[b] / entry_px - 1.0) * (1.0 if d == "UP" else -1.0)
            mark[b] = entry_equity * max(1.0 + LEVERAGE * (mr - FEE_RT), 0.0)
        equity = entry_equity * lev_mult
        mark[exit_i] = equity
        trades.append(dict(
            signal_ts=sig_ts, entry_ts=idx[entry_i], exit_ts=idx[exit_i],
            direction=d, tier=sigs.at[sig_ts, "tier"],
            entry_px=entry_px, exit_px=exit_px, exit_reason=reason,
            hold_h=(idx[exit_i] - idx[entry_i]).total_seconds() / 3600.0,
            price_ret_pct=r * 100.0,
            equity_ret_pct=(lev_mult - 1.0) * 100.0,
            equity_after=equity, liquidated=liquidated,
            regime=regime.get(idx[entry_i], "UNKNOWN"),
        ))
        busy_until = exit_i
        if equity <= 0:
            break

    # forward-fill flat bars on the mark series
    mark_s = pd.Series(mark, index=idx).ffill()
    return pd.DataFrame(trades), mark_s


# ─── metrics ────────────────────────────────────────────────────────────────

def equity_metrics(trades: pd.DataFrame, mark: pd.Series) -> dict:
    if trades.empty:
        return dict(n=0)
    er = trades["equity_ret_pct"].to_numpy()
    daily = mark.dropna()
    daily = daily[daily.index >= trades["entry_ts"].min()]
    dret = daily.resample("1D").last().pct_change().dropna()
    sharpe = (float(dret.mean() / dret.std(ddof=1) * np.sqrt(365.0))
              if len(dret) > 1 and dret.std(ddof=1) > 0 else 0.0)
    eq = daily.resample("1D").last()
    mdd = float((1.0 - eq / eq.cummax()).max() * 100.0) if len(eq) else 0.0
    final = float(trades["equity_after"].iloc[-1])
    return dict(
        n=len(trades),
        wr=float((trades["equity_ret_pct"] > 0).mean()),
        avg_eq_ret=float(er.mean()),
        avg_price_bps=float(trades["price_ret_pct"].mean() * 100.0),
        med_eq_ret=float(np.median(er)),
        best=float(er.max()), worst=float(er.min()),
        avg_hold=float(trades["hold_h"].mean()),
        final_equity=final,
        roi=float((final / CAPITAL0 - 1.0) * 100.0),
        sharpe=sharpe, mdd=mdd,
        n_liq=int(trades["liquidated"].sum()),
    )


def fmt(m: dict) -> str:
    if m.get("n", 0) == 0:
        return "(no trades)"
    return (f"n={m['n']}  WR={m['wr']*100:.1f}%  "
            f"eq/trade={m['avg_eq_ret']:+.2f}%  "
            f"hold={m['avg_hold']:.1f}h  "
            f"final=${m['final_equity']:,.0f}  ROI={m['roi']:+.1f}%  "
            f"Sharpe={m['sharpe']:.2f}  MaxDD={m['mdd']:.1f}%  "
            f"liq={m['n_liq']}")


def main():
    log("# V7 + 3×ATR Trailing Stop — Equity Simulation (1000u, 5×)\n")
    log(f"Generated: {pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC\n")

    k = pd.read_parquet(KLINES)[["open", "high", "low", "close"]].dropna()
    k.index = _strip_tz(k.index)
    k = k[~k.index.duplicated(keep="last")].sort_index()

    v = pd.read_parquet(V71_OOS)
    v.index = _strip_tz(v.index)
    v = v[~v.index.duplicated(keep="last")].sort_index()

    feat = pd.read_parquet(FEATURES)
    feat.index = _strip_tz(feat.index)
    feat = feat[~feat.index.duplicated(keep="last")].sort_index()

    def _reg(row):
        if row.get("is_trending_bull", 0) == 1:
            return "BULL"
        if row.get("is_trending_bear", 0) == 1:
            return "BEAR"
        return "CHOPPY"
    regime = feat.apply(_reg, axis=1)

    atr = atr_wilder(k["high"], k["low"], k["close"], ATR_PERIOD)
    decoded = decode_tiers(v["pred_ret"])
    sigs = decoded[decoded["direction"] != "NEUTRAL"].copy()

    log("**Simulation model:** 1000 USDT, ONE position at a time, sequential, "
        "compounding, 5× leverage. Each trade margin = full current equity. "
        "Fee 0.08% round-trip on notional. Funding NOT modelled.")
    log(f"OOS span: {v.index[0]:%Y-%m-%d} → {v.index[-1]:%Y-%m-%d}")
    log(f"v7.1 signals available: {len(sigs)} (one-position sim consumes a "
        f"subset — see n below)\n")
    med_atr = float((TRAIL_MULT * atr).reindex(sigs.index).median())
    med_px = float(k["close"].reindex(sigs.index).median())
    log(f"Median 3×ATR trailing distance ≈ ${med_atr:,.0f} "
        f"({med_atr/med_px*100:.2f}% of price) → at 5× a stop-out ≈ "
        f"−{med_atr/med_px*100*LEVERAGE:.1f}% of equity.\n")

    # ---- run ----
    tr_v7, mark_v7 = run_sim(sigs, k, decoded, atr, regime, "V7")
    tr_v0, mark_v0 = run_sim(sigs, k, decoded, atr, regime, "V0")
    m_v7 = equity_metrics(tr_v7, mark_v7)
    m_v0 = equity_metrics(tr_v0, mark_v0)

    log("## Headline — V7 + 3×ATR trailing stop\n")
    log(f"    {fmt(m_v7)}\n")
    log("## Reference — V0 fixed 4h (same account model, no stop)\n")
    log(f"    {fmt(m_v0)}\n")

    log("## Comparison\n")
    log("| Metric | V7 + 3×ATR trail | V0 fixed 4h |")
    log("|---|--:|--:|")
    log(f"| Trades taken | {m_v7['n']} | {m_v0['n']} |")
    log(f"| Win rate | {m_v7['wr']*100:.1f}% | {m_v0['wr']*100:.1f}% |")
    log(f"| Avg equity return / trade | {m_v7['avg_eq_ret']:+.2f}% | "
        f"{m_v0['avg_eq_ret']:+.2f}% |")
    log(f"| Median equity return / trade | {m_v7['med_eq_ret']:+.2f}% | "
        f"{m_v0['med_eq_ret']:+.2f}% |")
    log(f"| Best / worst trade | {m_v7['best']:+.1f}% / {m_v7['worst']:+.1f}% | "
        f"{m_v0['best']:+.1f}% / {m_v0['worst']:+.1f}% |")
    log(f"| Avg holding | {m_v7['avg_hold']:.1f}h | {m_v0['avg_hold']:.1f}h |")
    log(f"| **Final equity** | **${m_v7['final_equity']:,.0f}** | "
        f"${m_v0['final_equity']:,.0f} |")
    log(f"| **ROI** | **{m_v7['roi']:+.1f}%** | {m_v0['roi']:+.1f}% |")
    log(f"| Sharpe (daily, ann.) | {m_v7['sharpe']:.2f} | {m_v0['sharpe']:.2f} |")
    log(f"| Max drawdown | {m_v7['mdd']:.1f}% | {m_v0['mdd']:.1f}% |")
    log(f"| Liquidations | {m_v7['n_liq']} | {m_v0['n_liq']} |")
    log("")

    # ---- exit reasons ----
    log("## V7 exit-reason breakdown\n")
    log("| Reason | n | share | avg equity ret/trade | WR |")
    log("|---|--:|--:|--:|--:|")
    for reason, grp in tr_v7.groupby("exit_reason"):
        log(f"| {reason} | {len(grp)} | {len(grp)/len(tr_v7)*100:.1f}% | "
            f"{grp['equity_ret_pct'].mean():+.2f}% | "
            f"{(grp['equity_ret_pct']>0).mean()*100:.1f}% |")
    log("")

    # ---- regime ----
    log("## V7 regime breakdown (regime at entry)\n")
    log("| Regime | n | WR | avg equity ret/trade | "
        "cum equity factor |")
    log("|---|--:|--:|--:|--:|")
    for reg in ("BULL", "BEAR", "CHOPPY"):
        grp = tr_v7[tr_v7["regime"] == reg]
        if len(grp) == 0:
            log(f"| {reg} | 0 | – | – | – |")
            continue
        cumf = float(np.prod(1.0 + grp["equity_ret_pct"].to_numpy() / 100.0))
        warn = " ⚠️low-n" if len(grp) < 25 else ""
        log(f"| {reg}{warn} | {len(grp)} | {(grp['equity_ret_pct']>0).mean()*100:.1f}% "
            f"| {grp['equity_ret_pct'].mean():+.2f}% | ×{cumf:.2f} |")
    log("")

    # ---- plot ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    e7 = mark_v7.dropna()
    e0 = mark_v0.dropna()
    ax[0].plot(e7.index, e7.values, color="#d62728", lw=1.1,
               label=f"V7+3×ATR  →${m_v7['final_equity']:,.0f}")
    ax[0].plot(e0.index, e0.values, color="#1f77b4", lw=1.1,
               label=f"V0 4h     →${m_v0['final_equity']:,.0f}")
    ax[0].axhline(CAPITAL0, ls="--", c="grey", lw=0.8)
    ax[0].set_title("Equity curve (1000 USDT, 5× leverage)")
    ax[0].set_ylabel("Account equity (USDT)")
    ax[0].legend(fontsize=8)
    # drawdown
    dd7 = (1.0 - e7 / e7.cummax()) * 100.0
    ax[1].fill_between(dd7.index, -dd7.values, 0, color="#d62728", alpha=0.4)
    ax[1].set_title(f"V7 drawdown (max {m_v7['mdd']:.1f}%)")
    ax[1].set_ylabel("Drawdown %")
    fig.suptitle("V7 pure signal exit + 3×ATR trailing stop — 5× equity sim",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT, dpi=110)
    log(f"Equity plot saved → {PLOT}\n")

    tr_v7.to_csv(PROJECT_ROOT / "research" / "results" / "v7_3atr_trades.csv",
                 index=False)
    REPORT.write_text("\n".join(log_lines), encoding="utf-8")
    log(f"Report saved → {REPORT}")


if __name__ == "__main__":
    main()
