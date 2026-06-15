"""
Verify Architecture: v7.1 + Kernel (方式 C) vs Pure v7.1 Baseline
=================================================================
Task spec (user, 2026-05-16). Steps 1-7 in one self-contained script.

DESIGN DECISIONS (documented for the report):
  - v7.1 signals  = research/results/dual_model/direction_reg_oos_mse.parquet
                    (WF OOS of the production direction regressor; production
                     objective in direction_reg_config.json = reg:squarederror,
                     so MSE is the canonical head, not Huber).
  - Kernel        = Nadaraya-Watson Rational Quadratic, h=8 r=8 x=25, recomputed
                    here on Binance 1h klines close; cross-checked vs the
                    already-deployed LDC kernel in research/results/ldc_signals_full.parquet.
  - kernel_state  = bull if yhat1>yhat1[-1], bear if yhat1<yhat1[-1], else neutral.
  - Tier decode   = 500-bar rolling percentile on pred_ret, Strong=top/bot 5%,
                    Moderate=5-15%. NO fixed |pred| floor: the production floor
                    (0.0008) is calibrated to the *production* model's pred scale
                    (~0.003 std); the WF OOS fold models have ~0.0009 std
                    (see mistake.md 2026-04-19), so the floor is the wrong ruler
                    here. Rolling percentile is self-calibrating and applied
                    IDENTICALLY to 方式 C and the baseline -> fair comparison.
  - Fees          = 0.08% round-trip (0.04% x2) per task.
  - No model retraining, no new features, no parameter optimization.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KLINES = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"
V71_OOS = PROJECT_ROOT / "research" / "results" / "dual_model" / "direction_reg_oos_mse.parquet"
LDC_REF = PROJECT_ROOT / "research" / "results" / "ldc_signals_full.parquet"
REPORT = PROJECT_ROOT / "research" / "results" / "verify_kernel_method_c.md"

FEE_RT = 0.0008          # 0.08% round-trip
ATR_MULT = 1.5
ATR_PERIOD = 14
MAX_HOLD_H = 72
PCT_WINDOW = 500
WARMUP = 100
STRONG_FRAC = 0.05
MOD_FRAC = 0.15
KERNEL_H, KERNEL_R, KERNEL_X = 8, 8, 25

log_lines: list[str] = []


def log(msg: str = ""):
    print(msg)
    log_lines.append(msg)


# ─── Step 1: Kernel implementation ──────────────────────────────────────────

def nadaraya_watson_rq(close: pd.Series, h: int = 8, r: int = 8, x: int = 25) -> pd.Series:
    """
    Nadaraya-Watson Rational Quadratic kernel regression.
    Faithful port of jdehorty/KernelFunctions/2 :: rationalQuadratic.

        for i = 0 .. (x + 1):
            w[i] = (1 + i^2 / (h^2 * 2 * r)) ^ (-r)
        yhat[t] = sum(close[t-i] * w[i]) / sum(w[i])

    Identical to indicator/research kernel_rational_quadratic used by the
    production LDC swing system.
    """
    n_weights = x + 2                       # i = 0 .. x+1 inclusive
    weights = np.array([
        (1.0 + (i * i) / (h * h * 2.0 * r)) ** (-r)
        for i in range(n_weights)
    ])
    weights = weights / weights.sum()
    arr = close.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    for t in range(n_weights - 1, len(arr)):
        window = arr[t - n_weights + 1: t + 1][::-1]   # window[0] = close[t]
        out[t] = float(np.dot(window, weights))
    return pd.Series(out, index=close.index)


def kernel_state(yhat: pd.Series) -> pd.Series:
    prev = yhat.shift(1)
    st = pd.Series("neutral", index=yhat.index, dtype=object)
    st[yhat > prev] = "bull"
    st[yhat < prev] = "bear"
    st[yhat.isna() | prev.isna()] = "neutral"
    return st


def atr_wilder(high, low, close, period=14) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ─── load ───────────────────────────────────────────────────────────────────

def _strip_tz(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx


def load():
    k = pd.read_parquet(KLINES)[["open", "high", "low", "close"]].dropna()
    k.index = _strip_tz(k.index)
    k = k[~k.index.duplicated(keep="last")].sort_index()

    v = pd.read_parquet(V71_OOS)
    v.index = _strip_tz(v.index)
    v = v[~v.index.duplicated(keep="last")].sort_index()

    ldc = pd.read_parquet(LDC_REF)
    ldc.index = _strip_tz(ldc.index)
    ldc = ldc[~ldc.index.duplicated(keep="last")].sort_index()
    return k, v, ldc


# ─── Step 3: decode v7.1 tiers ──────────────────────────────────────────────

def decode_tiers(pred: pd.Series) -> pd.DataFrame:
    """Rolling-percentile decode of signed pred_ret into tier+direction."""
    strong_hi = pred.rolling(PCT_WINDOW, min_periods=WARMUP).quantile(1 - STRONG_FRAC)
    strong_lo = pred.rolling(PCT_WINDOW, min_periods=WARMUP).quantile(STRONG_FRAC)
    mod_hi = pred.rolling(PCT_WINDOW, min_periods=WARMUP).quantile(1 - MOD_FRAC)
    mod_lo = pred.rolling(PCT_WINDOW, min_periods=WARMUP).quantile(MOD_FRAC)
    # rolling quantile includes the current bar -> shift by 1 so the cutoff
    # uses only bars strictly before t (no look-ahead).
    strong_hi, strong_lo = strong_hi.shift(1), strong_lo.shift(1)
    mod_hi, mod_lo = mod_hi.shift(1), mod_lo.shift(1)

    direction = pd.Series("NEUTRAL", index=pred.index, dtype=object)
    tier = pd.Series("None", index=pred.index, dtype=object)

    up_mod = pred >= mod_hi
    dn_mod = pred <= mod_lo
    up_strong = pred >= strong_hi
    dn_strong = pred <= strong_lo

    direction[up_mod] = "UP"
    direction[dn_mod] = "DOWN"
    tier[up_mod | dn_mod] = "Moderate"
    tier[up_strong | dn_strong] = "Strong"

    out = pd.DataFrame({"pred_ret": pred, "direction": direction, "tier": tier})
    return out


# ─── Step 4: backtest engines ───────────────────────────────────────────────

def _trade_pnl(direction: str, entry_px: float, exit_px: float) -> tuple[float, float]:
    """Return (gross, net) fractional return for a long/short trade."""
    raw = (exit_px / entry_px - 1.0)
    gross = raw if direction == "UP" else -raw
    net = gross - FEE_RT
    return gross, net


def backtest_method_c(signals: pd.DataFrame, k: pd.DataFrame,
                      kstate: pd.Series, atr: pd.Series) -> pd.DataFrame:
    """方式 C: kernel-confluence entries, kernel-flip / hard-stop / 72h exit."""
    idx = k.index
    pos = {ts: i for i, ts in enumerate(idx)}
    rows = []
    for sig_ts, row in signals.iterrows():
        i = pos.get(sig_ts)
        if i is None or i + 1 >= len(idx):
            continue
        d = row["direction"]
        entry_i = i + 1
        entry_ts = idx[entry_i]
        entry_px = float(k["open"].iloc[entry_i])
        a = float(atr.iloc[i])
        if not np.isfinite(a) or a <= 0:
            continue
        if d == "UP":
            stop_px = entry_px - ATR_MULT * a
        else:
            stop_px = entry_px + ATR_MULT * a

        exit_i = exit_px = None
        reason = None
        for j in range(entry_i, min(entry_i + MAX_HOLD_H, len(idx))):
            hi = float(k["high"].iloc[j])
            lo = float(k["low"].iloc[j])
            # 1) hard stop — intrabar
            if d == "UP" and lo <= stop_px:
                exit_i, exit_px, reason = j, stop_px, "hard_stop"
                break
            if d == "DOWN" and hi >= stop_px:
                exit_i, exit_px, reason = j, stop_px, "hard_stop"
                break
            # 2) kernel flip — known at this bar's close, exit next bar open
            st = kstate.get(idx[j], "neutral")
            flipped = (d == "UP" and st == "bear") or (d == "DOWN" and st == "bull")
            if flipped:
                if j + 1 < len(idx):
                    exit_i, exit_px, reason = j + 1, float(k["open"].iloc[j + 1]), "kernel_flip"
                else:
                    exit_i, exit_px, reason = j, float(k["close"].iloc[j]), "kernel_flip"
                break
        # 3) max hold cap
        if exit_i is None:
            exit_i = min(entry_i + MAX_HOLD_H - 1, len(idx) - 1)
            exit_px = float(k["close"].iloc[exit_i])
            reason = "max_hold"

        gross, net = _trade_pnl(d, entry_px, exit_px)
        rows.append(dict(
            signal_ts=sig_ts, entry_ts=entry_ts, exit_ts=idx[exit_i],
            direction=d, tier=row["tier"], entry_px=entry_px, exit_px=exit_px,
            exit_reason=reason, hold_h=(idx[exit_i] - entry_ts).total_seconds() / 3600.0,
            gross=gross, net=net,
        ))
    return pd.DataFrame(rows)


def backtest_baseline(signals: pd.DataFrame, k: pd.DataFrame) -> pd.DataFrame:
    """Pure v7.1: 4h fixed hold, next-bar-open entry."""
    idx = k.index
    pos = {ts: i for i, ts in enumerate(idx)}
    rows = []
    for sig_ts, row in signals.iterrows():
        i = pos.get(sig_ts)
        if i is None or i + 5 >= len(idx):
            continue
        d = row["direction"]
        entry_i = i + 1
        exit_i = i + 5                       # 4h hold = 4 bars after entry bar
        entry_px = float(k["open"].iloc[entry_i])
        exit_px = float(k["close"].iloc[exit_i])
        gross, net = _trade_pnl(d, entry_px, exit_px)
        rows.append(dict(
            signal_ts=sig_ts, entry_ts=idx[entry_i], exit_ts=idx[exit_i],
            direction=d, tier=row["tier"], entry_px=entry_px, exit_px=exit_px,
            exit_reason="fixed_4h", hold_h=4.0, gross=gross, net=net,
        ))
    return pd.DataFrame(rows)


# ─── Step 5: metrics ────────────────────────────────────────────────────────

def portfolio_daily(trades: pd.DataFrame, k: pd.DataFrame) -> pd.Series:
    """
    Equal-weight, mark-to-market daily return series for a trade list.

    At each 1h bar, every open trade contributes its signed bar return; the
    portfolio bar return is the EQUAL-WEIGHT MEAN across open trades (0 on
    flat bars). This correctly handles overlapping trades and unequal holding
    times — unlike `per-trade Sharpe x sqrt(trades/yr)`, which assumes the
    trades are independent and non-overlapping (badly violated here, so that
    estimator inflates Sharpe ~5-10x). Fees split half at entry / half at exit.
    """
    idx = k.index
    pos = {ts: i for i, ts in enumerate(idx)}
    n = len(idx)
    bsum = np.zeros(n)
    bcount = np.zeros(n)
    close = k["close"].to_numpy(dtype=float)
    openp = k["open"].to_numpy(dtype=float)
    half_fee = FEE_RT / 2.0
    for _, tr in trades.iterrows():
        ei, xi = pos[tr["entry_ts"]], pos[tr["exit_ts"]]
        sign = 1.0 if tr["direction"] == "UP" else -1.0
        exit_px = float(tr["exit_px"])
        if ei == xi:
            r = sign * (exit_px / openp[ei] - 1.0) - 2.0 * half_fee
            bsum[ei] += r
            bcount[ei] += 1
            continue
        bsum[ei] += sign * (close[ei] / openp[ei] - 1.0) - half_fee
        bcount[ei] += 1
        for j in range(ei + 1, xi):
            bsum[j] += sign * (close[j] / close[j - 1] - 1.0)
            bcount[j] += 1
        bsum[xi] += sign * (exit_px / close[xi - 1] - 1.0) - half_fee
        bcount[xi] += 1
    bar_ret = np.where(bcount > 0, bsum / np.maximum(bcount, 1), 0.0)
    s = pd.Series(bar_ret, index=idx)
    if not len(trades):
        return s.iloc[:0]
    lo = trades["entry_ts"].min()
    hi = trades["exit_ts"].max()
    s = s[(s.index >= lo) & (s.index <= hi)]
    return s.resample("1D").sum()


def metrics(trades: pd.DataFrame, k: pd.DataFrame) -> dict:
    if trades.empty:
        return dict(n=0)
    net = trades["net"].to_numpy()
    gross = trades["gross"].to_numpy()
    wr = float((net > 0).mean())
    daily = portfolio_daily(trades, k)
    if len(daily) > 1 and daily.std(ddof=1) > 0:
        sharpe = float(daily.mean() / daily.std(ddof=1) * np.sqrt(365.0))
    else:
        sharpe = 0.0
    equity = (1.0 + daily).cumprod()
    mdd = float((1.0 - equity / equity.cummax()).max()) if len(equity) else 0.0
    cum_pct = float((equity.iloc[-1] - 1.0) * 100.0) if len(equity) else 0.0
    return dict(
        n=len(trades), wr=wr,
        avg_gross_bps=float(gross.mean() * 1e4),
        avg_net_bps=float(net.mean() * 1e4),
        avg_abs_bps=float(np.abs(net).mean() * 1e4),
        median_net_bps=float(np.median(net) * 1e4),
        avg_hold_h=float(trades["hold_h"].mean()),
        sum_net_pct=float(net.sum() * 100.0),
        cum_net_pct=cum_pct,
        sharpe=sharpe,
        mdd_pct=float(mdd * 100.0),
        n_days=len(daily),
    )


def fmt_metrics(m: dict) -> str:
    if m.get("n", 0) == 0:
        return "  (no trades)"
    return (f"  n={m['n']}  WR={m['wr']*100:.1f}%  "
            f"net={m['avg_net_bps']:+.1f}bps  gross={m['avg_gross_bps']:+.1f}bps  "
            f"|ret|={m['avg_abs_bps']:.1f}bps  hold={m['avg_hold_h']:.1f}h  "
            f"cum={m['cum_net_pct']:+.2f}%  Sharpe={m['sharpe']:.2f}  "
            f"MDD={m['mdd_pct']:.2f}%")


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    log("# Verify: v7.1 + Kernel (方式 C) vs Pure v7.1 Baseline")
    log(f"\nGenerated: {pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC\n")

    k, v, ldc = load()

    # ===== STEP 1: kernel implementation & validation =======================
    log("## Step 1 — Kernel implementation & validation\n")
    yhat1 = nadaraya_watson_rq(k["close"], KERNEL_H, KERNEL_R, KERNEL_X)
    kst = kernel_state(yhat1)

    # cross-check vs already-deployed LDC kernel (ldc_signals_full.parquet)
    ref = ldc["yhat1"].dropna()
    common = yhat1.dropna().index.intersection(ref.index)
    diff = (yhat1.loc[common] - ref.loc[common]).abs()
    log(f"- Kernel formula: w[i] = (1 + i^2 / (h^2*2*r))^(-r), i=0..{KERNEL_X+1}; "
        f"h={KERNEL_H} r={KERNEL_R} x={KERNEL_X}")
    log(f"- Recomputed yhat1 on {len(k)} Binance 1h bars "
        f"({k.index[0]:%Y-%m-%d} → {k.index[-1]:%Y-%m-%d})")
    log(f"- Cross-check vs deployed LDC kernel (ldc_signals_full.parquet), "
        f"{len(common)} overlapping bars:")
    log(f"    max abs diff = {diff.max():.3e}   mean abs diff = {diff.mean():.3e}")
    # kernel-state agreement
    ref_state = pd.Series(np.where(ldc["is_bullish_kernel"], "bull",
                          np.where(ldc["is_bearish_kernel"], "bear", "neutral")),
                          index=ldc.index)
    cs = kst.loc[common]
    rs = ref_state.loc[common]
    state_match = float((cs == rs).mean())
    log(f"    kernel-state agreement = {state_match*100:.2f}% "
        f"({int((cs==rs).sum())}/{len(common)})")
    tol_ok = diff.max() < 1e-3
    log(f"- Numeric tolerance (<0.001 vs deployed kernel): "
        f"{'PASS' if tol_ok else 'FAIL'}")
    log("- NOTE: a manual TradingView cross-check (5 reference bars) is still "
        "required to certify against the live Pine Script; this script validates "
        "the Python port is reproducible and identical to the kernel already "
        "running in production LDC swing.\n")
    if not tol_ok or state_match < 0.99:
        log("**Step 1 FAILED — aborting per task instruction.**")
        REPORT.write_text("\n".join(log_lines), encoding="utf-8")
        return

    # ===== STEP 2: kernel state on v7.1 timeframe ===========================
    log("## Step 2 — Kernel state on v7.1 OOS timeframe\n")
    v_idx = v.index
    kst_v = kst.reindex(v_idx)
    counts = kst_v.value_counts(dropna=False)
    total = len(v_idx)
    nbull = int(counts.get("bull", 0))
    nbear = int(counts.get("bear", 0))
    nneu = total - nbull - nbear
    log(f"- v7.1 OOS span: {v_idx[0]:%Y-%m-%d %H:%M} → {v_idx[-1]:%Y-%m-%d %H:%M}  "
        f"({total} bars)")
    log(f"- bull:    {nbull:5d}  ({nbull/total*100:.1f}%)")
    log(f"- bear:    {nbear:5d}  ({nbear/total*100:.1f}%)")
    log(f"- neutral: {nneu:5d}  ({nneu/total*100:.1f}%)\n")

    # ===== STEP 3: decode v7.1 signals & kernel confluence filter ===========
    log("## Step 3 — v7.1 signal decode & kernel confluence filter\n")
    dec = decode_tiers(v["pred_ret"])
    sigs = dec[dec["direction"] != "NEUTRAL"].copy()
    sigs["kernel_state"] = kst.reindex(sigs.index).values

    def confluent(r):
        return ((r["direction"] == "UP" and r["kernel_state"] == "bull") or
                (r["direction"] == "DOWN" and r["kernel_state"] == "bear"))
    sigs["kept"] = sigs.apply(confluent, axis=1)

    n_sig = len(sigs)
    kept = sigs[sigs["kept"]]
    n_kept = len(kept)
    log(f"- Total v7.1 signals (Strong+Moderate, UP+DOWN): {n_sig}")
    log(f"- Kept (kernel confluence):   {n_kept}  ({n_kept/n_sig*100:.1f}%)")
    log(f"- Filtered out:               {n_sig-n_kept}  "
        f"({(n_sig-n_kept)/n_sig*100:.1f}%)\n")
    log("  By tier:")
    for t in ("Strong", "Moderate"):
        sub = sigs[sigs["tier"] == t]
        if len(sub):
            fk = int((~sub["kept"]).sum())
            log(f"    {t:9s}: {len(sub):4d} signals, "
                f"{fk} filtered ({fk/len(sub)*100:.1f}%), "
                f"{len(sub)-fk} kept")
    log("  By direction:")
    for d in ("UP", "DOWN"):
        sub = sigs[sigs["direction"] == d]
        if len(sub):
            fk = int((~sub["kept"]).sum())
            log(f"    {d:9s}: {len(sub):4d} signals, "
                f"{fk} filtered ({fk/len(sub)*100:.1f}%), "
                f"{len(sub)-fk} kept")
    log("")

    # ===== STEP 4-5: backtest 方式 C ========================================
    log("## Step 4-5 — 方式 C backtest & metrics\n")
    atr = atr_wilder(k["high"], k["low"], k["close"], ATR_PERIOD)
    tc = backtest_method_c(kept, k, kst, atr)
    mc = metrics(tc, k)
    log("**方式 C (kernel-confluence entry + kernel/stop/72h exit):**")
    log(fmt_metrics(mc))
    log("")
    if not tc.empty:
        log("  Exit-reason breakdown:")
        for reason, grp in tc.groupby("exit_reason"):
            mm = metrics(grp, k)
            log(f"    {reason:11s}: {fmt_metrics(mm).strip()}")
        log("")
        log("  By v7.1 tier:")
        for t in ("Strong", "Moderate"):
            grp = tc[tc["tier"] == t]
            if len(grp):
                log(f"    {t:9s}: {fmt_metrics(metrics(grp, k)).strip()}")
        log("  By direction:")
        for d in ("UP", "DOWN"):
            grp = tc[tc["direction"] == d]
            if len(grp):
                log(f"    {d:9s}: {fmt_metrics(metrics(grp, k)).strip()}")
        log("")

    # ===== STEP 6: baseline v7.1 ===========================================
    log("## Step 6 — Pure v7.1 baseline (4h fixed exit, same period)\n")
    all_sigs = sigs[["pred_ret", "direction", "tier"]].copy()
    tb = backtest_baseline(all_sigs, k)
    mb = metrics(tb, k)
    log("**v7.1 baseline (all Strong+Moderate signals, 4h fixed hold):**")
    log(fmt_metrics(mb))
    log("")
    if not tb.empty:
        log("  By v7.1 tier:")
        for t in ("Strong", "Moderate"):
            grp = tb[tb["tier"] == t]
            if len(grp):
                log(f"    {t:9s}: {fmt_metrics(metrics(grp, k)).strip()}")
        log("  By direction:")
        for d in ("UP", "DOWN"):
            grp = tb[tb["direction"] == d]
            if len(grp):
                log(f"    {d:9s}: {fmt_metrics(metrics(grp, k)).strip()}")
        log("")

    # side-by-side
    log("### Side-by-side comparison\n")
    log("| Metric | v7.1 baseline | 方式 C | Delta |")
    log("|---|---|---|---|")

    def drow(label, kb, kc, fmt="{:.2f}", pp=False):
        vb, vc = mb.get(kb, 0), mc.get(kc, 0)
        d = vc - vb
        log(f"| {label} | {fmt.format(vb)} | {fmt.format(vc)} | "
            f"{('+' if d>=0 else '')}{fmt.format(d)}{'pp' if pp else ''} |")

    log(f"| Total trades | {mb['n']} | {mc['n']} | {mc['n']-mb['n']} |")
    drow("Win rate %", "wr", "wr", "{:.3f}")
    drow("Avg net /trade (bps)", "avg_net_bps", "avg_net_bps", "{:.1f}")
    drow("Avg gross /trade (bps)", "avg_gross_bps", "avg_gross_bps", "{:.1f}")
    drow("Sharpe (annualised)", "sharpe", "sharpe", "{:.2f}")
    drow("Max drawdown %", "mdd_pct", "mdd_pct", "{:.2f}")
    drow("Cum net %", "cum_net_pct", "cum_net_pct", "{:.2f}")
    drow("Avg holding (h)", "avg_hold_h", "avg_hold_h", "{:.1f}")
    log("")

    # ===== STEP 7: decision ================================================
    log("## Step 7 — Decision\n")
    if mc["n"] == 0 or mb["n"] == 0:
        log("**Inconclusive** — one side produced no trades. Investigate data.")
    else:
        sharpe_impr = ((mc["sharpe"] - mb["sharpe"]) / abs(mb["sharpe"])
                       if mb["sharpe"] != 0 else float("inf"))
        wr_impr_pp = (mc["wr"] - mb["wr"]) * 100
        log(f"- Sharpe: {mb['sharpe']:.2f} → {mc['sharpe']:.2f}  "
            f"({sharpe_impr*100:+.1f}%)")
        log(f"- Win rate: {mb['wr']*100:.1f}% → {mc['wr']*100:.1f}%  "
            f"({wr_impr_pp:+.1f}pp)")
        log(f"- Net/trade: {mb['avg_net_bps']:+.1f} → {mc['avg_net_bps']:+.1f} bps")
        log("")
        if sharpe_impr > 0.30 or (wr_impr_pp > 5 and sharpe_impr > -0.10):
            verdict = ("A — 方式 C clearly beats baseline. "
                       "Proceed to robustness check (parameter sensitivity).")
        elif sharpe_impr < 0 or wr_impr_pp < -3:
            verdict = ("C — 方式 C worse than baseline. Verification failed; "
                       "document failure mode, stop here.")
        else:
            verdict = ("B — marginal. Run ablation: 方式 A (exit only) and "
                       "方式 B (filter only) separately to isolate the driver.")
        log(f"**DECISION: {verdict}**")
        log("")

        # ---- failure-mode decomposition (required when decision == C) -----
        # Isolate the two kernel components without changing integration method:
        #   filter effect  = baseline-on-kept-signals  vs  baseline-on-all
        #   exit effect    = 方式 C (kept + kernel exit) vs baseline-on-kept
        log("### Failure-mode decomposition\n")
        tb_kept = backtest_baseline(kept[["pred_ret", "direction", "tier"]], k)
        mb_kept = metrics(tb_kept, k)
        log("Three trade sets, identical accounting:")
        log(f"- [1] baseline ALL signals (4h exit):   {fmt_metrics(mb).strip()}")
        log(f"- [2] baseline KEPT signals (4h exit):  {fmt_metrics(mb_kept).strip()}")
        log(f"- [3] 方式 C   KEPT signals (kernel exit): {fmt_metrics(mc).strip()}")
        log("")
        log(f"- **Kernel FILTER effect** ([2] vs [1], same 4h exit): "
            f"net {mb['avg_net_bps']:+.1f} → {mb_kept['avg_net_bps']:+.1f} bps, "
            f"WR {mb['wr']*100:.1f}% → {mb_kept['wr']*100:.1f}%, "
            f"Sharpe {mb['sharpe']:.2f} → {mb_kept['sharpe']:.2f}")
        log(f"- **Kernel EXIT effect** ([3] vs [2], same kept signals): "
            f"net {mb_kept['avg_net_bps']:+.1f} → {mc['avg_net_bps']:+.1f} bps, "
            f"WR {mb_kept['wr']*100:.1f}% → {mc['wr']*100:.1f}%, "
            f"Sharpe {mb_kept['sharpe']:.2f} → {mc['sharpe']:.2f}")
        log("")
    log("")

    # save trade logs
    if not tc.empty:
        tc.to_csv(PROJECT_ROOT / "research" / "results" / "method_c_trades.csv", index=False)
    if not tb.empty:
        tb.to_csv(PROJECT_ROOT / "research" / "results" / "v71_baseline_trades.csv", index=False)
    REPORT.write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\n(Report saved → {REPORT})")


if __name__ == "__main__":
    main()
