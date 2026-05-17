"""
V7.1 paper-strategy WF-OOS trading backtest  (rebuilt 2026-05-17).

Reconstructs the backtest that produced indicator/paper_trading.py's
V7_BACKTEST_BASELINE (n=169, WR 56.2%, ROI 84.9%, Sharpe 5.10). The original
file with this name was referenced by v7_paper_executor.py and paper_trading.py
but was never committed to git — only the hard-coded result numbers survived.
This script lets the headline Sharpe be judged real vs. inflated.

What it does
------------
1. Loads genuine WF-OOS direction predictions
   (research/results/dual_model/direction_reg_oos_mse.parquet — 77 walk-forward
   folds, purge=4 + embargo=4; every test bar is predicted by a model trained
   only on prior data).
2. Decodes UP/DOWN/NEUTRAL bar-by-bar with the EXACT production rule
   (indicator/inference.py): 500-bar TRAILING rolling-percentile cutoffs
   (strong 5% / moderate 15%, two-tailed) + absolute |pred| floor, warmup=100
   fallback thresholds. Bar i's cutoff uses the buffer through bar i only.
3. Simulates v7.1 exactly as indicator/v7_paper_executor.run_cycle: one
   position at a time, enter at the signal-bar close, exit on 3xATR(14) Wilder
   trailing stop (intrabar, gap-through filled at bar open) / 72h time cap /
   opposite signal. 8 bps round-trip taker cost.
4. Fixed-fractional sizing: 2% equity risked per trade, 1x leverage cap,
   compounding from 1000 USDT.
5. Reports n / WR / ROI / MDD / Sharpe (three annualisations) / profit factor.
6. Runs integrity checks (backtest_audit.py style) + a Monte-Carlo block:
   trade-order bootstrap CI and a random-entry null (same exit mechanics).

Honest caveats — read before trusting the numbers
--------------------------------------------------
- WF-OOS fold predictions have std ~0.00088 vs the production model's ~0.003
  (.claude/rules/mistake.md 2026-04-19). The rolling-percentile decode is
  scale-invariant so signal SELECTION is faithful; the absolute floor is NOT
  scale-invariant — its binding rate on this scale is reported below.
- Live enters at the signal-bar close; the lost baseline reportedly entered at
  next-bar open. This script uses signal-bar close to match the live executor.
  Run with --entry next_open to compare.

Usage:
    python research/v71_v7_sizing_1x.py
    python research/v71_v7_sizing_1x.py --entry next_open --mc 2000
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OOS_PARQUET = PROJECT_ROOT / "research/results/dual_model/direction_reg_oos_mse.parquet"
FEATURES_PARQUET = PROJECT_ROOT / "research/dual_model/.cache/features_all.parquet"

# ── Strategy constants — imported 1:1 from indicator/v7_paper_executor.py ─────
ATR_PERIOD = 14
TRAIL_MULT = 3.0
TIME_CAP_HOURS = 72
TAKER_COST = 0.0008          # 8 bps round-trip
INITIAL_CAPITAL = 1000.0
RISK_FRAC = 0.02
MAX_LEVERAGE = 1.0

# ── Decode constants — from indicator/model_artifacts/dual_model/
#    direction_reg_config.json + indicator/inference.py ─────────────────────────
PCTILE_WINDOW = 500
WARMUP_BARS = 100
STRONG_FRAC = 0.05
MODERATE_FRAC = 0.15
ABS_FLOOR_STRONG = 0.0008
ABS_FLOOR_MODERATE = 0.0005
FALLBACK = dict(strong_up=0.0021502008312381804, strong_dn=-0.0015994133136700839,
                moderate_up=0.0011227245995542035, moderate_dn=-0.0010095790785271674)

PRODUCTION_PRED_STD = 0.003  # mistake.md 2026-04-19, for the scale caveat


# ── Data ──────────────────────────────────────────────────────────────────────

def _to_naive_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx


def load_data() -> pd.DataFrame:
    """Merge WF-OOS predictions with OHLC. Returns a dt-sorted frame.

    `in_oos` marks bars eligible to generate entries (decode is only defined
    there). Bars after the OOS window are kept so trades opened near the end
    can still be managed to a real exit.
    """
    oos = pd.read_parquet(OOS_PARQUET)
    feats = pd.read_parquet(FEATURES_PARQUET)
    oos.index = _to_naive_utc(pd.DatetimeIndex(oos.index))
    feats.index = _to_naive_utc(pd.DatetimeIndex(feats.index))
    oos = oos.sort_index()
    feats = feats.sort_index()

    ohlc = feats[["open", "high", "low", "close"]].copy()
    ohlc["atr"] = _atr_wilder(ohlc)

    df = ohlc.join(oos[["pred_ret", "fold"]], how="left")
    df = df.loc[df.index >= oos.index.min()].copy()
    df["in_oos"] = df["pred_ret"].notna()
    return df


def _atr_wilder(ohlc: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Wilder ATR — matches v7_paper_executor._atr_wilder (causal, no lookahead)."""
    prev_close = ohlc["close"].shift(1)
    tr = pd.concat([
        ohlc["high"] - ohlc["low"],
        (ohlc["high"] - prev_close).abs(),
        (ohlc["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ── Decode — exact replica of indicator/inference.py rolling-percentile path ──

def decode_signals(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (direction, tier, floor_binding_count) for OOS bars in order.

    NEUTRAL on non-OOS bars. Buffer is trailing-only: bar i's cutoff uses
    predictions through bar i (inference.py appends BEFORE computing the cut).
    """
    buf: deque[float] = deque(maxlen=PCTILE_WINDOW)
    n = len(df)
    direction = np.array(["NEUTRAL"] * n, dtype=object)
    tier = np.array(["Weak"] * n, dtype=object)
    preds = df["pred_ret"].values
    in_oos = df["in_oos"].values
    floor_binds = 0

    for i in range(n):
        if not in_oos[i]:
            continue
        p = float(preds[i])
        buf.append(p)

        if len(buf) < WARMUP_BARS:
            up_s, dn_s = FALLBACK["strong_up"], FALLBACK["strong_dn"]
            up_m, dn_m = FALLBACK["moderate_up"], FALLBACK["moderate_dn"]
        else:
            a = np.fromiter(buf, dtype=float)
            up_s = float(np.quantile(a, 1.0 - STRONG_FRAC / 2.0))
            dn_s = float(np.quantile(a, STRONG_FRAC / 2.0))
            up_m = float(np.quantile(a, 1.0 - MODERATE_FRAC / 2.0))
            dn_m = float(np.quantile(a, MODERATE_FRAC / 2.0))

        up_s_eff = max(up_s, ABS_FLOOR_STRONG)
        dn_s_eff = min(dn_s, -ABS_FLOOR_STRONG)
        up_m_eff = max(up_m, ABS_FLOOR_MODERATE)
        dn_m_eff = min(dn_m, -ABS_FLOOR_MODERATE)
        if (up_s_eff != up_s or dn_s_eff != dn_s or
                up_m_eff != up_m or dn_m_eff != dn_m):
            floor_binds += 1

        if p >= up_s_eff:
            direction[i], tier[i] = "UP", "Strong"
        elif p <= dn_s_eff:
            direction[i], tier[i] = "DOWN", "Strong"
        elif p >= up_m_eff:
            direction[i], tier[i] = "UP", "Moderate"
        elif p <= dn_m_eff:
            direction[i], tier[i] = "DOWN", "Moderate"

    return direction, tier, floor_binds


# ── Trade simulation — exact replica of v7_paper_executor.run_cycle ───────────

def simulate(df: pd.DataFrame, direction: np.ndarray, tier: np.ndarray,
             entry_mode: str = "signal_close",
             extra_slip: float = 0.0) -> pd.DataFrame:
    """One-position-at-a-time v7.1 sim. Returns a per-trade DataFrame.

    entry_mode : "signal_close" (live convention) | "next_open" (lost baseline).
    extra_slip : extra round-trip cost added ONLY to trailing-stop exits
                 (adverse-fill sensitivity; 0 = headline run).
    """
    o = df["open"].values
    h = df["high"].values
    lo = df["low"].values
    c = df["close"].values
    atr = df["atr"].values
    ts = df.index
    n = len(df)

    equity = INITIAL_CAPITAL
    pos = None
    trades = []

    for i in range(n):
        # ── Case A: manage an open position ──
        if pos is not None:
            bars_held = int((ts[i] - pos["entry_ts"]).total_seconds() / 3600)
            sd = pos["stop_dist"]
            prev_ext = pos["trail_extreme"]
            if pos["side"] == "LONG":
                cur_stop = prev_ext - sd
            else:
                cur_stop = prev_ext + sd

            exit_price = exit_reason = None
            # 1) trailing stop — intrabar, gap-through filled at bar open
            if pos["side"] == "LONG" and lo[i] <= cur_stop:
                exit_price, exit_reason = min(cur_stop, o[i]), "trail_stop"
            elif pos["side"] == "SHORT" and h[i] >= cur_stop:
                exit_price, exit_reason = max(cur_stop, o[i]), "trail_stop"
            # 2) time cap
            if exit_reason is None and bars_held >= TIME_CAP_HOURS:
                exit_price, exit_reason = c[i], "time_cap"
            # 3) opposite signal
            if exit_reason is None:
                opp = ((pos["side"] == "LONG" and direction[i] == "DOWN") or
                       (pos["side"] == "SHORT" and direction[i] == "UP"))
                if opp:
                    exit_price, exit_reason = c[i], "opp_signal"
            # 4) data end — force close (flagged, not a strategy exit)
            if exit_reason is None and i == n - 1:
                exit_price, exit_reason = c[i], "data_end"

            if exit_reason is None:
                if pos["side"] == "LONG":
                    pos["trail_extreme"] = max(prev_ext, h[i])
                else:
                    pos["trail_extreme"] = min(prev_ext, lo[i])
                continue

            cost = TAKER_COST + (extra_slip if exit_reason == "trail_stop" else 0.0)
            if pos["side"] == "LONG":
                gross = exit_price / pos["entry_price"] - 1.0
            else:
                gross = -(exit_price / pos["entry_price"] - 1.0)
            net = gross - cost
            equity_ret = pos["size_frac"] * net
            equity = max(equity * (1.0 + equity_ret), 0.0)
            trades.append(dict(
                entry_ts=pos["entry_ts"], exit_ts=ts[i], side=pos["side"],
                tier=pos["tier"], entry_price=pos["entry_price"],
                exit_price=exit_price, exit_reason=exit_reason,
                bars_held=bars_held, gross_pct=gross, net_pct=net,
                size_frac=pos["size_frac"], equity_ret_pct=equity_ret * 100.0,
                equity_after=equity, win=int(gross > 0)))
            pos = None
            continue  # no re-entry on the same bar (mirrors run_cycle)

        # ── Case B: flat — check for entry ──
        if not df["in_oos"].values[i]:
            continue
        if direction[i] not in ("UP", "DOWN"):
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue

        if entry_mode == "signal_close":
            entry_idx, entry_price = i, c[i]
        else:  # next_open
            if i + 1 >= n:
                continue
            entry_idx, entry_price = i + 1, o[i + 1]

        stop_dist = TRAIL_MULT * a
        stop_pct = stop_dist / entry_price
        size_frac = min(MAX_LEVERAGE, RISK_FRAC / stop_pct) if stop_pct > 0 else MAX_LEVERAGE
        pos = dict(
            side="LONG" if direction[i] == "UP" else "SHORT",
            tier=tier[i], entry_ts=ts[entry_idx], entry_price=entry_price,
            stop_dist=stop_dist, trail_extreme=entry_price, size_frac=size_frac)

    return pd.DataFrame(trades)


# ── Metrics ───────────────────────────────────────────────────────────────────

def _max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    return float(np.max((peak - equity_curve) / peak) * 100.0)


def summarize(trades: pd.DataFrame, span_days: float) -> dict:
    if trades.empty:
        return {"n": 0}
    eq = np.concatenate([[INITIAL_CAPITAL], trades["equity_after"].values])
    net = trades["net_pct"].values
    wins = net[net > 0]
    losses = net[net < 0]
    roi = (eq[-1] / INITIAL_CAPITAL - 1.0) * 100.0

    sharpe_pt = float(net.mean() / net.std()) if net.std() > 0 else 0.0
    trades_per_yr = len(trades) * 365.0 / span_days
    sharpe_tradecount = sharpe_pt * np.sqrt(trades_per_yr)

    # calendar Sharpe — daily equity-return series (the honest annualisation)
    daily = (pd.Series(trades["equity_after"].values, index=trades["exit_ts"])
             .resample("1D").last().ffill())
    daily = pd.concat([pd.Series([INITIAL_CAPITAL],
                                 index=[daily.index[0] - pd.Timedelta(days=1)]),
                       daily])
    dret = daily.pct_change().dropna()
    sharpe_cal = (float(dret.mean() / dret.std()) * np.sqrt(365.0)
                  if dret.std() > 0 else 0.0)

    return {
        "n": len(trades),
        "wr_pct": float(trades["win"].mean() * 100.0),
        "roi_pct": roi,
        "mdd_pct": _max_drawdown(eq),
        "sharpe_per_trade": sharpe_pt,
        "sharpe_tradecount_ann": float(sharpe_tradecount),
        "sharpe_calendar_ann": sharpe_cal,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        "avg_net_bps": float(net.mean() * 1e4),
        "avg_eq_ret_pct": float(trades["equity_ret_pct"].mean()),
        "avg_bars_held": float(trades["bars_held"].mean()),
        "final_equity": float(eq[-1]),
        "trades_per_yr": float(trades_per_yr),
    }


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def bootstrap_trades(trades: pd.DataFrame, n_iter: int, seed: int = 42) -> dict:
    """Resample per-trade equity returns with replacement; compound each path."""
    rng = np.random.default_rng(seed)
    eqr = trades["equity_ret_pct"].values / 100.0
    rois, mdds = [], []
    for _ in range(n_iter):
        s = rng.choice(eqr, size=len(eqr), replace=True)
        curve = INITIAL_CAPITAL * np.cumprod(1.0 + s)
        rois.append((curve[-1] / INITIAL_CAPITAL - 1.0) * 100.0)
        mdds.append(_max_drawdown(np.concatenate([[INITIAL_CAPITAL], curve])))
    rois = np.array(rois)
    return {
        "roi_p5": float(np.percentile(rois, 5)),
        "roi_p50": float(np.percentile(rois, 50)),
        "roi_p95": float(np.percentile(rois, 95)),
        "p_profitable": float((rois > 0).mean()),
        "mdd_p95": float(np.percentile(mdds, 95)),
    }


def random_entry_null(df: pd.DataFrame, direction: np.ndarray, real_summary: dict,
                      n_iter: int, span_days: float, seed: int = 7) -> dict:
    """Null model: same exit mechanics, but entries are random UP/DOWN drawn at
    the real strategy's per-bar entry rate. Tests whether the v7.1 signal picks
    better entries than the 3xATR trailing stop alone would on noise.
    """
    rng = np.random.default_rng(seed)
    in_oos = df["in_oos"].values
    n_oos = int(in_oos.sum())
    n_real_signals = int(np.sum([d in ("UP", "DOWN") for d in direction]))
    rate = n_real_signals / max(n_oos, 1)

    null_rois, null_sharpes = [], []
    for _ in range(n_iter):
        rand_dir = np.array(["NEUTRAL"] * len(df), dtype=object)
        draw = rng.random(len(df))
        side = rng.random(len(df))
        for i in range(len(df)):
            if in_oos[i] and draw[i] < rate:
                rand_dir[i] = "UP" if side[i] < 0.5 else "DOWN"
        t = simulate(df, rand_dir, rand_dir, entry_mode="signal_close")
        s = summarize(t, span_days)
        null_rois.append(s.get("roi_pct", 0.0))
        null_sharpes.append(s.get("sharpe_calendar_ann", 0.0))
    null_rois = np.array(null_rois)
    null_sharpes = np.array(null_sharpes)
    return {
        "entry_rate": rate,
        "null_roi_p50": float(np.percentile(null_rois, 50)),
        "null_roi_p95": float(np.percentile(null_rois, 95)),
        "null_sharpe_p95": float(np.percentile(null_sharpes, 95)),
        "p_roi": float((null_rois >= real_summary["roi_pct"]).mean()),
        "p_sharpe": float((null_sharpes >= real_summary["sharpe_calendar_ann"]).mean()),
    }


# ── Report ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", choices=["signal_close", "next_open"],
                    default="signal_close")
    ap.add_argument("--mc", type=int, default=10000, help="trade bootstrap iters")
    ap.add_argument("--null", type=int, default=500, help="random-entry null iters")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    df = load_data()
    span_days = (df.index.max() - df.index.min()).total_seconds() / 86400.0
    direction, tier, floor_binds = decode_signals(df)

    n_oos = int(df["in_oos"].sum())
    n_sig = int(np.sum([d in ("UP", "DOWN") for d in direction]))
    pred_std = float(df.loc[df["in_oos"], "pred_ret"].std())

    trades = simulate(df, direction, tier, entry_mode=args.entry)
    s = summarize(trades, span_days)

    print("=" * 72)
    print("  V7.1 WF-OOS TRADING BACKTEST  (rebuilt)")
    print("=" * 72)
    print(f"  OOS span      : {df.index.min()}  ->  {df.index.max()}")
    print(f"  OOS bars      : {n_oos}   ({span_days:.0f} days)")
    print(f"  entry mode    : {args.entry}")
    print(f"  decoded signals (UP/DOWN bars) : {n_sig}  "
          f"({n_sig / n_oos * 100:.1f}% of OOS bars)")
    print()
    print("  --- HEADLINE (compare to V7_BACKTEST_BASELINE: "
          "n=169 WR=56.2% ROI=84.9% MDD=5.3% Sharpe=5.10) ---")
    print(f"  trades            : {s['n']}")
    print(f"  win rate          : {s['wr_pct']:.1f}%")
    print(f"  ROI (compounded)  : {s['roi_pct']:+.1f}%   final ${s['final_equity']:.0f}")
    print(f"  max drawdown      : {s['mdd_pct']:.1f}%")
    print(f"  profit factor     : {s['profit_factor']:.2f}")
    print(f"  avg net / trade   : {s['avg_net_bps']:+.1f} bps")
    print(f"  avg bars held     : {s['avg_bars_held']:.1f} h")
    print()
    print("  --- SHARPE (audit point 4: which annualisation gives 5.10?) ---")
    print(f"  per-trade               : {s['sharpe_per_trade']:.3f}")
    print(f"  annualised (trade-count): {s['sharpe_tradecount_ann']:.2f}   "
          f"[= per-trade x sqrt({s['trades_per_yr']:.0f} trades/yr)]")
    print(f"  annualised (calendar)   : {s['sharpe_calendar_ann']:.2f}   "
          f"[daily equity returns x sqrt(365) — the honest one]")
    print()

    print("  --- INTEGRITY CHECKS ---")
    print("  [1] overlapping trades   : 0 by construction (one position at a time)")
    gap = trades[(trades["exit_reason"] == "trail_stop")]
    n_gap = 0
    if not gap.empty:
        for _, t in gap.iterrows():
            pass
    print(f"  [2] TP/SL ambiguity      : none — v7.1 has no TP, only a trailing "
          f"stop; stop level uses the PRIOR bar's extreme (no lookahead)")
    print("  [3] regime lookahead     : N/A — BULL_CONTRA_PENALTY=1.0 (no-op)")
    print("  [4] Sharpe periodicity   : see three figures above")
    print("  [5] parameter selection  : none swept here — all strategy constants "
          "imported 1:1 from v7_paper_executor.py; decode fracs from config json")
    print("  [6] slippage             : 8 bps round-trip taker; see sensitivity below")
    print("  [7] decode lookahead     : rolling buffer is trailing-only "
          "(append-then-cut, same as inference.py)")
    print(f"  [8] scale caveat         : OOS fold pred std={pred_std:.5f} vs "
          f"production ~{PRODUCTION_PRED_STD:.4f}")
    print(f"      abs-floor binding rate : {floor_binds}/{n_oos} bars "
          f"({floor_binds / n_oos * 100:.1f}%)")
    print()

    # exit-reason mix
    print("  --- exit-reason mix ---")
    for reason, cnt in trades["exit_reason"].value_counts().items():
        sub = trades[trades["exit_reason"] == reason]
        print(f"  {reason:<12} : {cnt:>3}  WR={sub['win'].mean() * 100:>5.1f}%  "
              f"avg net={sub['net_pct'].mean() * 1e4:+.0f} bps")
    print()

    # slippage sensitivity
    for slip in (0.0005, 0.0010):
        t2 = simulate(df, direction, tier, entry_mode=args.entry, extra_slip=slip)
        s2 = summarize(t2, span_days)
        print(f"  slippage sensitivity +{slip * 1e4:.0f}bps on stop exits : "
              f"ROI {s2['roi_pct']:+.1f}%  Sharpe(cal) {s2['sharpe_calendar_ann']:.2f}")
    print()

    # Monte Carlo
    print(f"  --- MONTE CARLO: trade-order bootstrap (n={args.mc}) ---")
    bs = bootstrap_trades(trades, args.mc)
    print(f"  ROI  p5 / p50 / p95   : {bs['roi_p5']:+.1f}% / "
          f"{bs['roi_p50']:+.1f}% / {bs['roi_p95']:+.1f}%")
    print(f"  P(profitable)         : {bs['p_profitable'] * 100:.1f}%")
    print(f"  MDD p95               : {bs['mdd_p95']:.1f}%")
    print()
    print(f"  --- MONTE CARLO: random-entry null (n={args.null}, same exit logic) ---")
    rn = random_entry_null(df, direction, s, args.null, span_days)
    print(f"  random-entry ROI    p50 / p95 : {rn['null_roi_p50']:+.1f}% / "
          f"{rn['null_roi_p95']:+.1f}%")
    print(f"  random-entry Sharpe       p95 : {rn['null_sharpe_p95']:.2f}")
    print(f"  p-value  ROI    (real >= random) : {rn['p_roi']:.3f}")
    print(f"  p-value  Sharpe (real >= random) : {rn['p_sharpe']:.3f}")
    verdict = ("SIGNAL ADDS EDGE over the trailing-stop mechanics"
               if rn["p_sharpe"] < 0.05
               else "CANNOT REJECT random-entry null — headline may be "
                    "driven by the exit mechanics, not the v7.1 signal")
    print(f"  verdict : {verdict}")
    print("=" * 72)

    if args.save:
        out = PROJECT_ROOT / "research/results/v71_v7_sizing_1x_trades.csv"
        trades.to_csv(out, index=False)
        print(f"  trades written -> {out}")


if __name__ == "__main__":
    main()
