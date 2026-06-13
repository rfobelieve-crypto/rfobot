"""
V7.1 Exit Logic Variant Backtest
================================
Task spec (user, 2026-05-16). Quantify which exit logic gives the best
net Sharpe / WR / MDD for v7.1 signals.

Methodology is IDENTICAL to verify_kernel_method_c.py — same v7.1 signal
decode, same fee model, same daily-MTM portfolio Sharpe — so the numbers
are directly comparable with the Method C verification and the V0 baseline
here reproduces that report's "v7.1 baseline".

Variants:
  V0  fixed 4h                       (current baseline)
  V1  fixed 8h
  V2  fixed 12h
  V3  fixed 24h
  V4  8h  + opposite-Strong early exit
  V5  8h  + opposite-Strong + wide disaster stop  max(3%, 4*ATR14)
  V6  12h + opposite-Strong early exit

Constraints honoured: no parameter optimisation, no extra variants, no model
change. "opposite Strong" = a bar that decodes to the OPPOSITE direction at
Strong tier (Strong tier == confidence >= 80 per CLAUDE.md decode spec).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
PROJECT_ROOT = _HERE.parent

# identical methodology — reuse the Method C verification primitives
from verify_kernel_method_c import (          # noqa: E402
    decode_tiers, atr_wilder, metrics, _trade_pnl, FEE_RT,
)

KLINES = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"
V71_OOS = PROJECT_ROOT / "research" / "results" / "dual_model" / "direction_reg_oos_mse.parquet"
FEATURES = PROJECT_ROOT / "research" / "dual_model" / ".cache" / "features_all.parquet"
REPORT = PROJECT_ROOT / "research" / "results" / "v71_exit_variants.md"

ATR_PERIOD = 14
DISASTER_PCT = 0.03      # 3% absolute
DISASTER_ATR = 4.0       # 4 x ATR(14)

VARIANTS = [
    # name, horizon_h, use_opp, use_stop, opp_tier
    ("V0  fixed 4h",                       4,  False, False, "Strong"),
    ("V1  fixed 8h",                       8,  False, False, "Strong"),
    ("V2  fixed 12h",                      12, False, False, "Strong"),
    ("V3  fixed 24h",                      24, False, False, "Strong"),
    ("V4  8h + opp-Strong",                8,  True,  False, "Strong"),
    ("V5  8h + opp-Strong + wide stop",    8,  True,  True,  "Strong"),
    ("V6  12h + opp-Strong",               12, True,  False, "Strong"),
    # pure signal-decay exit: no fixed horizon, 72h safety cap only.
    # exits on ANY opposite reading (Moderate or Strong) -> "model no longer
    # supports the position". Middle ~70% NEUTRAL bars = hold (band hysteresis).
    ("V7  pure signal exit (72h cap)",     72, True,  False, "any"),
    ("V8  pure signal exit + wide stop",   72, True,  True,  "any"),
]

log_lines: list[str] = []


def log(msg: str = ""):
    log_lines.append(msg)
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def _strip_tz(idx):
    return idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx


# ─── backtest engine ────────────────────────────────────────────────────────

def backtest(signals: pd.DataFrame, k: pd.DataFrame, decoded: pd.DataFrame,
             atr: pd.Series, regime: pd.Series,
             horizon_h: int, use_opp: bool, use_stop: bool,
             opp_tier: str = "Strong") -> pd.DataFrame:
    """
    One trade per signal. Entry = next bar open.
    Exit precedence on each live bar:
      1. disaster stop  (intrabar, if use_stop)   -> exit at stop price
      2. time exit      (at horizon bar)          -> exit at that bar close
      3. opposite signal (if use_opp)             -> exit at next bar open
         opp_tier="Strong": only an opposite Strong reading triggers.
         opp_tier="any":    any opposite reading (Moderate or Strong) triggers.
    """
    idx = k.index
    pos = {ts: i for i, ts in enumerate(idx)}
    openp = k["open"].to_numpy(float)
    high = k["high"].to_numpy(float)
    low = k["low"].to_numpy(float)
    close = k["close"].to_numpy(float)
    dir_arr = decoded["direction"]
    tier_arr = decoded["tier"]
    rows = []

    for sig_ts, srow in signals.iterrows():
        i = pos.get(sig_ts)
        if i is None or i + 1 >= len(idx):
            continue
        d = srow["direction"]
        entry_i = i + 1
        entry_px = openp[entry_i]
        max_exit_i = entry_i + horizon_h
        if max_exit_i >= len(idx):
            continue                                  # not enough forward data

        # disaster stop level
        stop_px = None
        if use_stop:
            a = float(atr.iloc[i])
            if not np.isfinite(a) or a <= 0:
                continue
            dist = max(DISASTER_PCT * entry_px, DISASTER_ATR * a)
            stop_px = entry_px - dist if d == "UP" else entry_px + dist

        exit_i = exit_px = reason = None
        for j in range(entry_i, max_exit_i + 1):
            # 1) disaster stop — intrabar, every live bar incl. entry & horizon bar
            if use_stop:
                if d == "UP" and low[j] <= stop_px:
                    exit_i, exit_px, reason = j, stop_px, "disaster_stop"
                    break
                if d == "DOWN" and high[j] >= stop_px:
                    exit_i, exit_px, reason = j, stop_px, "disaster_stop"
                    break
            # 2) time exit at the horizon bar
            if j == max_exit_i:
                exit_i, exit_px, reason = j, close[j], "time_exit"
                break
            # 3) opposite-Strong signal at bar j close -> exit next bar open
            if use_opp:
                opp = "DOWN" if d == "UP" else "UP"
                tj = idx[j]
                if dir_arr.get(tj) == opp and (
                        opp_tier == "any" or tier_arr.get(tj) == opp_tier):
                    exit_i, exit_px, reason = j + 1, openp[j + 1], "opp_signal"
                    break

        gross, net = _trade_pnl(d, entry_px, exit_px)
        rows.append(dict(
            signal_ts=sig_ts, entry_ts=idx[entry_i], exit_ts=idx[exit_i],
            direction=d, tier=srow["tier"], entry_px=entry_px, exit_px=exit_px,
            exit_reason=reason,
            hold_h=(idx[exit_i] - idx[entry_i]).total_seconds() / 3600.0,
            gross=gross, net=net,
            regime=regime.get(idx[entry_i], "UNKNOWN"),
        ))
    return pd.DataFrame(rows)


# ─── reporting helpers ──────────────────────────────────────────────────────

def mrow(name: str, m: dict) -> str:
    if m.get("n", 0) == 0:
        return f"| {name} | 0 | – | – | – | – | – | – | – | – |"
    return (f"| {name} | {m['n']} | {m['wr']*100:.1f}% | "
            f"{m['avg_gross_bps']:+.1f} | {m['avg_net_bps']:+.1f} | "
            f"{m['avg_abs_bps']:.0f} | {m['avg_hold_h']:.1f} | "
            f"{m['sharpe']:.2f} | {m['mdd_pct']:.2f}% | {m['cum_net_pct']:+.1f}% |")


def main():
    log("# V7.1 Exit Logic Variant Backtest\n")
    log(f"Generated: {pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC\n")

    # ---- load -------------------------------------------------------------
    k = pd.read_parquet(KLINES)[["open", "high", "low", "close"]].dropna()
    k.index = _strip_tz(k.index)
    k = k[~k.index.duplicated(keep="last")].sort_index()

    v = pd.read_parquet(V71_OOS)
    v.index = _strip_tz(v.index)
    v = v[~v.index.duplicated(keep="last")].sort_index()

    feat = pd.read_parquet(FEATURES)
    feat.index = _strip_tz(feat.index)
    feat = feat[~feat.index.duplicated(keep="last")].sort_index()

    # regime from v7.1's own trailing features (is_trending_bull/bear)
    def _regime(row):
        if row.get("is_trending_bull", 0) == 1:
            return "BULL"
        if row.get("is_trending_bear", 0) == 1:
            return "BEAR"
        return "CHOPPY"
    regime = feat.apply(_regime, axis=1)

    atr = atr_wilder(k["high"], k["low"], k["close"], ATR_PERIOD)

    decoded = decode_tiers(v["pred_ret"])
    sigs = decoded[decoded["direction"] != "NEUTRAL"].copy()

    log(f"OOS span: {v.index[0]:%Y-%m-%d %H:%M} → {v.index[-1]:%Y-%m-%d %H:%M}  "
        f"({len(v)} bars)")
    log(f"v7.1 signals (Strong+Moderate, UP+DOWN): {len(sigs)}  "
        f"(Strong {int((sigs.tier=='Strong').sum())}, "
        f"Moderate {int((sigs.tier=='Moderate').sum())})")
    rcounts = regime.reindex(sigs.index).value_counts(dropna=False).to_dict()
    log(f"Signal regime mix (at signal bar): {rcounts}")
    log(f"Disaster stop = max({DISASTER_PCT*100:.0f}% , {DISASTER_ATR:.0f}×ATR14)\n")

    # ---- run variants -----------------------------------------------------
    results = {}
    trades = {}
    for name, H, use_opp, use_stop, opp_tier in VARIANTS:
        tr = backtest(sigs, k, decoded, atr, regime, H, use_opp, use_stop,
                      opp_tier)
        trades[name] = tr
        results[name] = metrics(tr, k)

    # ---- main comparison table -------------------------------------------
    log("## Variant comparison\n")
    log("| Variant | n | WR | gross/tr (bps) | net/tr (bps) | |ret| (bps) | "
        "hold (h) | Sharpe | MaxDD | Cum net |")
    log("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for name, *_ in VARIANTS:
        log(mrow(name, results[name]))
    log("")
    log("Sharpe = annualised daily-MTM equal-weight portfolio (same as Method C "
        "verification). MaxDD / Cum net from the compounded daily MTM equity. "
        "net/tr is per-trade after 8bps round-trip fee.\n")

    # ---- WR by tier -------------------------------------------------------
    log("## Win rate by v7.1 tier\n")
    log("| Variant | Strong n | Strong WR | Strong net | Moderate n | "
        "Moderate WR | Moderate net |")
    log("|---|--:|--:|--:|--:|--:|--:|")
    for name, *_ in VARIANTS:
        tr = trades[name]
        st = metrics(tr[tr["tier"] == "Strong"], k)
        mo = metrics(tr[tr["tier"] == "Moderate"], k)
        log(f"| {name} | {st.get('n',0)} | "
            f"{st['wr']*100:.1f}% | {st['avg_net_bps']:+.1f} | "
            f"{mo.get('n',0)} | {mo['wr']*100:.1f}% | {mo['avg_net_bps']:+.1f} |"
            if st.get('n') and mo.get('n') else
            f"| {name} | {st.get('n',0)} | – | – | {mo.get('n',0)} | – | – |")
    log("")

    # ---- exit-reason breakdown + diagnostics -----------------------------
    log("## Exit-reason breakdown & trigger diagnostics\n")
    for name, H, use_opp, use_stop, opp_tier in VARIANTS:
        tr = trades[name]
        n = len(tr)
        if n == 0:
            continue
        parts = []
        for reason, grp in tr.groupby("exit_reason"):
            parts.append(f"{reason} {len(grp)} ({len(grp)/n*100:.1f}%, "
                          f"net {grp['net'].mean()*1e4:+.1f}bps)")
        log(f"- **{name}**: " + "; ".join(parts))
        if use_stop:
            sr = (tr["exit_reason"] == "disaster_stop").mean()
            flag = "OK" if 0.01 <= sr <= 0.05 else "OUT OF 1-5% TARGET"
            log(f"    disaster-stop trigger rate = {sr*100:.1f}%  [{flag}]")
        if use_opp:
            orr = (tr["exit_reason"] == "opp_signal").mean()
            log(f"    opposite-signal trigger rate = {orr*100:.1f}%")
    log("")

    # ---- regime breakdown -------------------------------------------------
    log("## Regime breakdown (regime at entry, from is_trending_bull/bear)\n")
    for reg in ("BULL", "BEAR", "CHOPPY"):
        log(f"### {reg}\n")
        log("| Variant | n | WR | net/tr (bps) | Sharpe | MaxDD |")
        log("|---|--:|--:|--:|--:|--:|")
        for name, *_ in VARIANTS:
            tr = trades[name]
            sub = tr[tr["regime"] == reg]
            m = metrics(sub, k)
            if m.get("n", 0) == 0:
                log(f"| {name} | 0 | – | – | – | – |")
            else:
                warn = "  ⚠️low-n" if m["n"] < 30 else ""
                log(f"| {name} | {m['n']}{warn} | {m['wr']*100:.1f}% | "
                    f"{m['avg_net_bps']:+.1f} | {m['sharpe']:.2f} | "
                    f"{m['mdd_pct']:.2f}% |")
        log("")

    # ---- decision ---------------------------------------------------------
    log("## Decision\n")
    r = results
    g = lambda nm, key: r[nm][key]
    n0, n1, n2, n3 = ("V0  fixed 4h", "V1  fixed 8h", "V2  fixed 12h",
                      "V3  fixed 24h")
    n4, n5, n6 = ("V4  8h + opp-Strong", "V5  8h + opp-Strong + wide stop",
                  "V6  12h + opp-Strong")

    log("**Q1 — does extending fixed horizon (4→8→12→24) monotonically "
        "improve net Sharpe?**")
    sh = [g(n0, "sharpe"), g(n1, "sharpe"), g(n2, "sharpe"), g(n3, "sharpe")]
    mono = all(sh[i] <= sh[i + 1] for i in range(3))
    log(f"- Sharpe 4h→8h→12h→24h: {sh[0]:.2f} → {sh[1]:.2f} → {sh[2]:.2f} "
        f"→ {sh[3]:.2f}  → {'monotone up' if mono else 'NOT monotone'}")
    nb = [g(x, "avg_net_bps") for x in (n0, n1, n2, n3)]
    log(f"- net/trade: {nb[0]:+.1f} → {nb[1]:+.1f} → {nb[2]:+.1f} → "
        f"{nb[3]:+.1f} bps\n")

    log("**Q2 — does opposite-signal early exit help or hurt? (V4 vs V1)**")
    log(f"- Sharpe {g(n1,'sharpe'):.2f} → {g(n4,'sharpe'):.2f}; "
        f"net/tr {g(n1,'avg_net_bps'):+.1f} → {g(n4,'avg_net_bps'):+.1f} bps; "
        f"WR {g(n1,'wr')*100:.1f}% → {g(n4,'wr')*100:.1f}%; "
        f"MaxDD {g(n1,'mdd_pct'):.2f}% → {g(n4,'mdd_pct'):.2f}%\n")

    log("**Q3 — does wide disaster stop cost edge or add insurance? "
        "(V5 vs V4)**")
    sr5 = (trades[n5]["exit_reason"] == "disaster_stop").mean()
    log(f"- Sharpe {g(n4,'sharpe'):.2f} → {g(n5,'sharpe'):.2f}; "
        f"net/tr {g(n4,'avg_net_bps'):+.1f} → {g(n5,'avg_net_bps'):+.1f} bps; "
        f"MaxDD {g(n4,'mdd_pct'):.2f}% → {g(n5,'mdd_pct'):.2f}%; "
        f"stop fired {sr5*100:.1f}% of trades\n")

    log("**Q4 — which variant is robust across regimes?**")
    for name, *_ in VARIANTS:
        tr = trades[name]
        cells = []
        for reg in ("BULL", "BEAR", "CHOPPY"):
            m = metrics(tr[tr["regime"] == reg], k)
            cells.append(f"{reg}={m['sharpe']:.1f}" if m.get("n") else f"{reg}=–")
        log(f"- {name}: " + "  ".join(cells))
    log("")

    n7, n8 = ("V7  pure signal exit (72h cap)",
              "V8  pure signal exit + wide stop")
    log("**Q5 — clock vs signal: does a pure signal-decay exit beat the "
        "time-based variants?**")
    log(f"- V7 (pure signal): Sharpe {g(n7,'sharpe'):.2f}, "
        f"WR {g(n7,'wr')*100:.1f}%, net {g(n7,'avg_net_bps'):+.1f}bps, "
        f"hold {g(n7,'avg_hold_h'):.1f}h, MaxDD {g(n7,'mdd_pct'):.2f}%, "
        f"cum {g(n7,'cum_net_pct'):+.1f}%")
    log(f"- vs V0 (clock 4h): Sharpe {g(n0,'sharpe'):.2f}, "
        f"net {g(n0,'avg_net_bps'):+.1f}bps, MaxDD {g(n0,'mdd_pct'):.2f}%")
    log(f"- vs V4 (8h + opp-Strong): Sharpe {g(n4,'sharpe'):.2f}, "
        f"net {g(n4,'avg_net_bps'):+.1f}bps, MaxDD {g(n4,'mdd_pct'):.2f}%")
    to7 = (trades[n7]["exit_reason"] == "time_exit").mean()
    log(f"- V7 safety-cap (72h) bind rate: {to7*100:.1f}% of trades "
        f"(low = signal genuinely drives the exit)")
    log(f"- V8 (V7 + wide stop): Sharpe {g(n8,'sharpe'):.2f}, "
        f"net {g(n8,'avg_net_bps'):+.1f}bps, MaxDD {g(n8,'mdd_pct'):.2f}%, "
        f"stop fired "
        f"{(trades[n8]['exit_reason']=='disaster_stop').mean()*100:.1f}%")
    verdict = ("signal beats clock" if g(n7, "sharpe") > g(n0, "sharpe")
               else "clock still wins")
    log(f"- **Verdict: {verdict}** "
        f"(V7 Sharpe {g(n7,'sharpe'):.2f} vs V0 {g(n0,'sharpe'):.2f})")
    log("")

    # auto-pick: best Sharpe among variants whose every regime sub-Sharpe >= 0
    robust = []
    for name, *_ in VARIANTS:
        tr = trades[name]
        ok = True
        for reg in ("BULL", "BEAR", "CHOPPY"):
            m = metrics(tr[tr["regime"] == reg], k)
            if m.get("n", 0) >= 30 and m["sharpe"] < 0:
                ok = False
        if ok:
            robust.append(name)
    ranked = sorted(robust, key=lambda nm: r[nm]["sharpe"], reverse=True)
    log("### Recommendation\n")
    if ranked:
        best = ranked[0]
        log(f"**Preferred variant: {best}**  "
            f"(Sharpe {r[best]['sharpe']:.2f}, WR {r[best]['wr']*100:.1f}%, "
            f"net {r[best]['avg_net_bps']:+.1f}bps, MaxDD {r[best]['mdd_pct']:.2f}%)")
        log("Selected as: highest portfolio Sharpe among variants with no "
            "negative-Sharpe regime (sample n≥30). Full reasoning written by "
            "the analyst on top of these numbers — see narrative below.\n")
    else:
        log("No variant passed the no-negative-regime-Sharpe screen — "
            "see regime tables.\n")

    # save
    for name, *_ in VARIANTS:
        safe = name.split()[0]
        trades[name].to_csv(
            PROJECT_ROOT / "research" / "results" / f"exit_variant_{safe}_trades.csv",
            index=False)
    REPORT.write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nReport saved → {REPORT}")


if __name__ == "__main__":
    main()
