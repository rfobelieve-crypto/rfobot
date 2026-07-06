"""Exit-variants sweep under the REAL live exit semantics + full gauntlet.

Motivation (2026-07-03 live exit autopsy, n=8): trail_stop exits gave back a
mean 2.56% of MFE (max 4.04%) and ran 33% WR, while opp_signal exits were
+2.45%/100%. n is far too small to touch live, so this sweep asks the same
question on hundreds of simulated trades: is 3xATR the wrong trail — and do
breakeven locks / MFE-triggered tightening / LONG-tighter asymmetry help?

Faithful to indicator/okx/executor.py TODAY:
  entry = STRONG only (strong_only_entry), next-bar open, 1-position occupancy;
  exits = 3xATR trailing stop (ratchet on completed bars, active next bar,
  intrabar hit) + opposite signal at close.  NO time cap (disabled 2026-06-10).
  Fees = REAL ruler: 10 bps round-trip (2026-07-06 fee fix; the legacy 8 bps
  constant under-counted taker+taker).

Gauntlet (multiple-comparison guards — mistake.md 2026-06-02):
  1. SELECT/CONFIRM split: challengers are ranked on the first SELECT_MONTHS
     entry-months only; the winner must then also beat baseline on the held-out
     CONFIRM window it never saw during selection.
  2. Per-month stability: challenger must beat baseline in >= 60% of months.
  3. Unpaired bootstrap CI (10k) of the net/trade difference vs baseline.
  4. MDD must not be materially worse (+2pp tolerance).
All four pass -> DEPLOY-CANDIDATE; anything less -> NO-GO, keep 3xATR.

Data: fresh walk-forward OOS (production trainer, features to today).  The
canonical direction_reg_oos_mse.parquet ends 2026-04-30, missing the BULL
regime + the live window — this script regenerates a fresh OOS when stale.

Usage:  python -m research.dual_model.exit_variants_sweep
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT))

from verify_kernel_method_c import decode_tiers, atr_wilder, _strip_tz, metrics

KLINES = ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"
FRESH_OOS = ROOT / "research" / "results" / "dual_model" / "direction_reg_oos_fresh.parquet"
OUT_CSV = ROOT / "research" / "results" / "dual_model" / "exit_variants_sweep.csv"

REAL_FEE_RT = 0.0010      # taker 5 bps/side x 2 — the post-fix ruler
LEGACY_FEE_RT = 0.0008    # old constant, reported once for reference
OOS_MAX_AGE_H = 48
SELECT_MONTHS = 4          # first N entry-months = selection window
BOOT_N = 10_000
SEED = 42


# ── fresh OOS (regenerate with the production trainer when stale) ──────────

def load_fresh_oos() -> pd.DataFrame:
    if FRESH_OOS.exists():
        oos = pd.read_parquet(FRESH_OOS)
        oos.index = _strip_tz(oos.index)
        age_h = (pd.Timestamp.utcnow().tz_localize(None)
                 - oos.index.max()).total_seconds() / 3600.0
        if age_h <= OOS_MAX_AGE_H:
            print(f"fresh OOS cache hit: {FRESH_OOS.name} "
                  f"({oos.index[0]:%Y-%m-%d} → {oos.index[-1]:%Y-%m-%d})")
            return oos
        print(f"fresh OOS is {age_h:.0f}h old → regenerating…")
    from research.dual_model.shared_data import load_and_cache_data
    from research.dual_model.direction_features_v2 import FULL_DIRECTION
    from research.dual_model.train_direction_reg_4h import (
        train_direction_reg_walk_forward,
    )
    df = load_and_cache_data(limit=4000)
    # Fail-loud staleness (same rule as quarterly_revalidation 2026-07-06):
    # an exit sweep graded on truncated data would repeat the June-IC artifact.
    data_age_h = (pd.Timestamp.now(tz=df.index[-1].tz)
                  - df.index[-1]).total_seconds() / 3600.0
    if data_age_h > OOS_MAX_AGE_H:
        raise RuntimeError(
            f"feature data ends {df.index[-1]} ({data_age_h:.0f}h old) — "
            f"backfill first; refusing to sweep on stale data")
    oos, m, _ = train_direction_reg_walk_forward(df, FULL_DIRECTION,
                                                 objective="mse")
    print(f"WF regenerated: AUC {m['auc_sign']:.4f} IC {m['spearman_ic']:+.4f} "
          f"n={len(oos)}")
    oos = oos[["pred_ret", "y_path_ret_4h"]].copy()
    oos.index = _strip_tz(oos.index)
    oos.to_parquet(FRESH_OOS)
    return oos


# ── simulation (harness-faithful, parameterised exits) ─────────────────────

def simulate(k: pd.DataFrame, decoded: pd.DataFrame, atr: pd.Series, *,
             trail_long: float, trail_short: float,
             be_trigger: float | None = None,
             tighten_trigger: float | None = None,
             tighten_mult: float | None = None,
             fee_rt: float = REAL_FEE_RT) -> pd.DataFrame:
    """STRONG-only, 1-position, next-bar-open entry.  Per completed bar:
    (1) intrabar stop hit at the PRIOR bar's stop, (2) opposite signal exits
    at close, (3) post-bar ratchet: update extreme/MFE, apply tighten/BE —
    all effective from the NEXT bar (no intrabar look-ahead)."""
    idx = k.index
    openp = k["open"].to_numpy(float)
    high = k["high"].to_numpy(float)
    low = k["low"].to_numpy(float)
    close = k["close"].to_numpy(float)
    dir_arr = decoded["direction"].reindex(idx).to_numpy()
    tier_arr = decoded["tier"].reindex(idx).to_numpy()
    atr_arr = atr.reindex(idx).to_numpy()
    n = len(idx)

    rows = []
    i = 0
    while i < n - 1:
        if not (dir_arr[i] != "NEUTRAL" and tier_arr[i] == "Strong"):
            i += 1
            continue
        d = dir_arr[i]
        a = atr_arr[i]
        entry_i = i + 1
        if entry_i >= n or not np.isfinite(a) or a <= 0:
            i += 1
            continue
        entry_px = openp[entry_i]
        up = d == "UP"
        stop_dist = (trail_long if up else trail_short) * a
        extreme = entry_px
        stop_px = entry_px - stop_dist if up else entry_px + stop_dist
        be_floor = None            # armed breakeven stop level
        tightened = False
        exit_i = exit_px = reason = None
        for j in range(entry_i, n):
            # 1) trailing stop — prior bar's level, intrabar hit
            if up and low[j] <= stop_px:
                exit_i, exit_px, reason = j, stop_px, "trail_stop"
                break
            if not up and high[j] >= stop_px:
                exit_i, exit_px, reason = j, stop_px, "trail_stop"
                break
            # 2) opposite signal (any tier) → exit at this close
            if dir_arr[j] == ("DOWN" if up else "UP"):
                exit_i, exit_px, reason = j, close[j], "opp_signal"
                break
            # 3) ratchet with this completed bar (active next bar)
            extreme = max(extreme, high[j]) if up else min(extreme, low[j])
            mfe = ((extreme - entry_px) / entry_px if up
                   else (entry_px - extreme) / entry_px)
            if (tighten_trigger is not None and not tightened
                    and mfe >= tighten_trigger):
                stop_dist = tighten_mult * a
                tightened = True
            if (be_trigger is not None and be_floor is None
                    and mfe >= be_trigger):
                # lock at breakeven + fees so a full retrace still nets >= 0
                be_floor = (entry_px * (1 + fee_rt) if up
                            else entry_px * (1 - fee_rt))
            stop_px = extreme - stop_dist if up else extreme + stop_dist
            if be_floor is not None:
                stop_px = max(stop_px, be_floor) if up else min(stop_px, be_floor)
        if exit_i is None:          # data ended while holding
            exit_i, exit_px, reason = n - 1, close[n - 1], "eod"
        raw = exit_px / entry_px - 1.0
        gross = raw if up else -raw
        net = gross - fee_rt
        mfe_final = ((extreme - entry_px) / entry_px if up
                     else (entry_px - extreme) / entry_px)
        rows.append(dict(
            signal_ts=idx[i], entry_ts=idx[entry_i], exit_ts=idx[exit_i],
            direction=d, tier="Strong", entry_px=entry_px, exit_px=exit_px,
            exit_reason=reason,
            hold_h=(idx[exit_i] - idx[entry_i]).total_seconds() / 3600.0,
            gross=gross, net=net,
            mfe=mfe_final, giveback=max(mfe_final - gross, 0.0),
        ))
        i = max(exit_i, entry_i)
    return pd.DataFrame(rows)


# ── variants ────────────────────────────────────────────────────────────────

VARIANTS: dict[str, dict] = {
    "BASE_3.0":        dict(trail_long=3.0, trail_short=3.0),
    "SYM_2.0":         dict(trail_long=2.0, trail_short=2.0),
    "SYM_2.5":         dict(trail_long=2.5, trail_short=2.5),
    "SYM_3.5":         dict(trail_long=3.5, trail_short=3.5),
    "BE@1.0":          dict(trail_long=3.0, trail_short=3.0, be_trigger=0.010),
    "BE@1.5":          dict(trail_long=3.0, trail_short=3.0, be_trigger=0.015),
    "TIGHT1.5@1.5":    dict(trail_long=3.0, trail_short=3.0,
                            tighten_trigger=0.015, tighten_mult=1.5),
    "TIGHT1.5@2.5":    dict(trail_long=3.0, trail_short=3.0,
                            tighten_trigger=0.025, tighten_mult=1.5),
    "BE1.5+TIGHT2.5":  dict(trail_long=3.0, trail_short=3.0, be_trigger=0.015,
                            tighten_trigger=0.025, tighten_mult=1.5),
    "ASYM_L2.0_S3.0":  dict(trail_long=2.0, trail_short=3.0),
    "ASYM_L2.5_S3.5":  dict(trail_long=2.5, trail_short=3.5),
}
BASE = "BASE_3.0"


def month_of(ts_series: pd.Series) -> pd.Series:
    return pd.to_datetime(ts_series).dt.to_period("M")


def summarize(tr: pd.DataFrame, k: pd.DataFrame) -> dict:
    m = metrics(tr, k)
    m["giveback_bps"] = float(tr["giveback"].mean() * 1e4) if len(tr) else np.nan
    m["mfe_bps"] = float(tr["mfe"].mean() * 1e4) if len(tr) else np.nan
    mix = tr.groupby("exit_reason")["net"].agg(["count", "mean"])
    m["mix"] = "  ".join(f"{r}:{int(c)}({v*1e4:+.0f}bps)"
                         for r, (c, v) in mix.iterrows())
    return m


def main() -> int:
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    k = pd.read_parquet(KLINES)[["open", "high", "low", "close"]].dropna()
    k.index = _strip_tz(k.index)
    k = k[~k.index.duplicated(keep="last")].sort_index()
    oos = load_fresh_oos()
    k = k.loc[oos.index[0]:oos.index[-1]]
    atr = atr_wilder(k["high"], k["low"], k["close"], 14)
    decoded = decode_tiers(oos["pred_ret"])
    n_strong = int(((decoded["tier"] == "Strong")
                    & (decoded["direction"] != "NEUTRAL")).sum())
    print(f"OOS {oos.index[0]:%Y-%m-%d} → {oos.index[-1]:%Y-%m-%d} "
          f"({len(oos)} bars, Strong bars={n_strong}, fee={REAL_FEE_RT*1e4:.0f}bps RT)\n")

    trades = {name: simulate(k, decoded, atr, **kw)
              for name, kw in VARIANTS.items()}

    # entry-month windows: first SELECT_MONTHS months = SELECT, rest = CONFIRM
    all_months = sorted(month_of(trades[BASE]["entry_ts"]).unique())
    sel_months = set(all_months[:SELECT_MONTHS])
    con_months = set(all_months[SELECT_MONTHS:])
    print(f"SELECT months: {sorted(str(m) for m in sel_months)}")
    print(f"CONFIRM months: {sorted(str(m) for m in con_months)}\n")

    def window(tr, months):
        return tr[month_of(tr["entry_ts"]).isin(months)]

    # ---- headline table (full period) --------------------------------------
    print("=" * 118)
    print(f"{'variant':16s} {'n':>4s} {'WR':>6s} {'net/tr':>8s} {'cum':>8s} "
          f"{'MDD':>7s} {'Sharpe':>7s} {'hold_h':>7s} {'MFE':>6s} {'giveback':>8s}")
    print("-" * 118)
    rows_csv = []
    for name in VARIANTS:
        m = summarize(trades[name], k)
        print(f"{name:16s} {m['n']:>4d} {m['wr']*100:>5.1f}% "
              f"{m['avg_net_bps']:>+7.1f} {m['cum_net_pct']:>+7.1f}% "
              f"{m['mdd_pct']:>6.2f}% {m['sharpe']:>7.2f} {m['avg_hold_h']:>7.1f} "
              f"{m['mfe_bps']:>5.0f} {m['giveback_bps']:>7.0f}")
        rows_csv.append(dict(variant=name, **{kk: vv for kk, vv in m.items()
                                              if kk != "mix"}))
    print("-" * 118)
    for name in VARIANTS:
        print(f"  {name:16s} exits: {summarize(trades[name], k)['mix']}")

    # legacy-fee reference: how much the old 8 bps ruler flattered the baseline
    base_tr = trades[BASE]
    legacy_net = base_tr["gross"].mean() - LEGACY_FEE_RT
    real_net = base_tr["net"].mean()
    print(f"\nruler check (BASE): net/tr {real_net*1e4:+.1f}bps @10bps real "
          f"vs {legacy_net*1e4:+.1f}bps @8bps legacy "
          f"(old ruler flattered by {(legacy_net-real_net)*1e4:.1f}bps/trade)")

    # ---- gauntlet -----------------------------------------------------------
    base_sel = window(base_tr, sel_months)
    base_con = window(base_tr, con_months)
    base_by_month = {mo: g["net"].sum()
                     for mo, g in base_tr.groupby(month_of(base_tr["entry_ts"]))}

    print("\n" + "=" * 118)
    print("GAUNTLET vs BASE_3.0  (1 SELECT-rank  2 CONFIRM holdout  "
          "3 per-month >=60%  4 bootstrap CI  5 MDD tolerance)")
    print("=" * 118)
    print(f"{'variant':16s} {'sel n/tr':>10s} {'con n/tr':>10s} "
          f"{'months>base':>11s} {'boot CI(bps)':>18s} {'p(<=0)':>7s} "
          f"{'dMDD':>6s} {'verdict':>10s}")
    print("-" * 118)

    results = []
    for name in VARIANTS:
        if name == BASE:
            continue
        tr = trades[name]
        sel, con = window(tr, sel_months), window(tr, con_months)
        d_sel = sel["net"].mean() - base_sel["net"].mean()
        d_con = (con["net"].mean() - base_con["net"].mean()
                 if len(con) and len(base_con) else np.nan)
        # per-month wins
        wins = tot = 0
        for mo, g in tr.groupby(month_of(tr["entry_ts"])):
            if mo in base_by_month:
                tot += 1
                wins += int(g["net"].sum() > base_by_month[mo])
        # unpaired bootstrap of full-period net/trade diff
        a = tr["net"].to_numpy()
        b = base_tr["net"].to_numpy()
        boots = (rng.choice(a, (BOOT_N, len(a)), replace=True).mean(axis=1)
                 - rng.choice(b, (BOOT_N, len(b)), replace=True).mean(axis=1))
        lo, hi = np.percentile(boots, [2.5, 97.5]) * 1e4
        p_le0 = float((boots <= 0).mean())
        d_mdd = summarize(tr, k)["mdd_pct"] - summarize(base_tr, k)["mdd_pct"]

        ok = (d_sel > 0
              and np.isfinite(d_con) and d_con > 0
              and tot > 0 and wins / tot >= 0.60
              and p_le0 < 0.05
              and d_mdd <= 2.0)
        verdict = "CANDIDATE" if ok else "no-go"
        results.append(dict(variant=name, d_sel_bps=d_sel * 1e4,
                            d_con_bps=(d_con * 1e4 if np.isfinite(d_con)
                                       else np.nan),
                            months_win=f"{wins}/{tot}", ci_lo=lo, ci_hi=hi,
                            p_le0=p_le0, d_mdd=d_mdd, verdict=verdict))
        print(f"{name:16s} {d_sel*1e4:>+9.1f} "
              f"{(d_con*1e4 if np.isfinite(d_con) else float('nan')):>+9.1f} "
              f"{wins:>6d}/{tot:<4d} [{lo:>+7.1f},{hi:>+7.1f}] "
              f"{p_le0:>6.3f} {d_mdd:>+5.1f}pp {verdict:>10s}")

    pd.DataFrame(rows_csv).to_csv(OUT_CSV, index=False)
    print(f"\nWrote → {OUT_CSV}   ({time.time()-t0:.0f}s)")

    cands = [r for r in results if r["verdict"] == "CANDIDATE"]
    print("\n" + "=" * 118)
    if cands:
        best = max(cands, key=lambda r: r["d_con_bps"])
        print(f"VERDICT: {len(cands)} candidate(s) survived the gauntlet; "
              f"best on CONFIRM = {best['variant']} "
              f"({best['d_con_bps']:+.1f}bps/trade vs base on held-out window).")
        print("Next: verify on live Gate-B trades before deploying "
              "(backtest fills stops at exact prices — live has slippage).")
    else:
        print("VERDICT: NO-GO — no exit variant beats 3xATR through the full "
              "gauntlet on this OOS. Keep the live exit unchanged; revisit "
              "with 30-50 clean live trades.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
