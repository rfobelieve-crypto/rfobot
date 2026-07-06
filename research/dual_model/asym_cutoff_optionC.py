"""Asymmetric LONG/SHORT Strong cutoff — Option C (no retrain, cutoff only).

The 34-day-old followup (memory asymmetric-ls-models-followup): DOWN edge >
UP edge is documented twice independently (bar-level sign-acc 59% vs 53% on
n=3700+; live shorts +115bps/70% vs longs +23bps/46%).  Option C asks the
cheapest version: keep the model, tighten ONLY the UP-Strong entry cutoff
(top 3% / 2% instead of 5%) while leaving DOWN at 5%.

Design notes:
  - Entry-side only.  Moderate tiers and the opposite-signal EXIT logic are
    untouched (exits fire on ANY opposite reading, so tightening UP-Strong
    does not change when a SHORT position exits).
  - Includes the mirrored control (tighten DOWN instead): if "tighten the
    weak side" is a real effect, the control should NOT beat baseline.
    Guards against "any restriction looks good in a good period".
  - Same gauntlet as exit_variants_sweep (SELECT/CONFIRM, per-month >= 60%,
    bootstrap CI, MDD tolerance) — bear-UP-gate NO-GO taught us slice-picked
    asymmetry hurts, so the verdict is portfolio-level, never slice-level.

Usage:  python -m research.dual_model.asym_cutoff_optionC
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT))

from verify_kernel_method_c import (
    atr_wilder, _strip_tz, metrics,
    PCT_WINDOW, WARMUP, STRONG_FRAC, MOD_FRAC,
)
from research.dual_model.exit_variants_sweep import (
    load_fresh_oos, simulate, month_of, KLINES, REAL_FEE_RT, BOOT_N, SEED,
    SELECT_MONTHS,
)

OUT_CSV = ROOT / "research" / "results" / "dual_model" / "asym_cutoff_optionC.csv"


def decode_tiers_asym(pred: pd.Series, *, strong_frac_up: float,
                      strong_frac_down: float) -> pd.DataFrame:
    """Rolling-percentile decode with per-side Strong fractions.

    Identical to verify_kernel_method_c.decode_tiers except the Strong
    quantile may differ between the UP tail and the DOWN tail.  Moderate
    stays symmetric at MOD_FRAC (it gates exits, not entries).
    """
    strong_hi = pred.rolling(PCT_WINDOW, min_periods=WARMUP).quantile(
        1 - strong_frac_up)
    strong_lo = pred.rolling(PCT_WINDOW, min_periods=WARMUP).quantile(
        strong_frac_down)
    mod_hi = pred.rolling(PCT_WINDOW, min_periods=WARMUP).quantile(1 - MOD_FRAC)
    mod_lo = pred.rolling(PCT_WINDOW, min_periods=WARMUP).quantile(MOD_FRAC)
    strong_hi, strong_lo = strong_hi.shift(1), strong_lo.shift(1)
    mod_hi, mod_lo = mod_hi.shift(1), mod_lo.shift(1)

    direction = pd.Series("NEUTRAL", index=pred.index, dtype=object)
    tier = pd.Series("None", index=pred.index, dtype=object)
    up_strong = pred >= strong_hi
    dn_strong = pred <= strong_lo
    up_mod = (pred >= mod_hi) & ~up_strong
    dn_mod = (pred <= mod_lo) & ~dn_strong
    direction[up_strong | up_mod] = "UP"
    direction[dn_strong | dn_mod] = "DOWN"
    tier[up_strong | dn_strong] = "Strong"
    tier[up_mod | dn_mod] = "Moderate"
    return pd.DataFrame({"direction": direction, "tier": tier})


def side_sign_acc(decoded: pd.DataFrame, oos: pd.DataFrame) -> dict:
    """Bar-level Strong sign-accuracy per side (4h path-return direction)."""
    out = {}
    j = decoded.join(oos["y_path_ret_4h"], how="inner")
    for side, sign in (("UP", 1), ("DOWN", -1)):
        sub = j[(j["tier"] == "Strong") & (j["direction"] == side)
                & np.isfinite(j["y_path_ret_4h"]) & (j["y_path_ret_4h"] != 0)]
        out[side] = (float((np.sign(sub["y_path_ret_4h"]) == sign).mean())
                     if len(sub) else np.nan, len(sub))
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    k = pd.read_parquet(KLINES)[["open", "high", "low", "close"]].dropna()
    k.index = _strip_tz(k.index)
    k = k[~k.index.duplicated(keep="last")].sort_index()
    oos = load_fresh_oos()
    k = k.loc[oos.index[0]:oos.index[-1]]
    atr = atr_wilder(k["high"], k["low"], k["close"], 14)

    variants = {
        "BASE_5/5":       dict(strong_frac_up=STRONG_FRAC, strong_frac_down=STRONG_FRAC),
        "UP3_DOWN5":      dict(strong_frac_up=0.03, strong_frac_down=STRONG_FRAC),
        "UP2_DOWN5":      dict(strong_frac_up=0.02, strong_frac_down=STRONG_FRAC),
        "CTRL_UP5_DOWN3": dict(strong_frac_up=STRONG_FRAC, strong_frac_down=0.03),
    }
    BASE = "BASE_5/5"

    print(f"OOS {oos.index[0]:%Y-%m-%d} → {oos.index[-1]:%Y-%m-%d} "
          f"({len(oos)} bars, fee={REAL_FEE_RT*1e4:.0f}bps RT)\n")

    # ---- bar-level: does tightening UP actually raise UP sign-acc? ---------
    print("Bar-level Strong sign-accuracy by side:")
    print(f"{'variant':16s} {'UP acc':>8s} {'nUP':>5s} {'DOWN acc':>9s} {'nDOWN':>6s}")
    decs = {}
    for name, kw in variants.items():
        dec = decode_tiers_asym(oos["pred_ret"], **kw)
        decs[name] = dec
        sa = side_sign_acc(dec, oos)
        print(f"{name:16s} {sa['UP'][0]*100:>7.1f}% {sa['UP'][1]:>5d} "
              f"{sa['DOWN'][0]*100:>8.1f}% {sa['DOWN'][1]:>6d}")

    # ---- trade-level under the REAL exit ------------------------------------
    trades = {name: simulate(k, decs[name], atr,
                             trail_long=3.0, trail_short=3.0)
              for name in variants}

    print("\nTrade-level (STRONG-only entry, real exit, 1-position):")
    print(f"{'variant':16s} {'n':>4s} {'nL':>4s} {'nS':>4s} {'WR':>6s} "
          f"{'net/tr':>8s} {'cum':>8s} {'MDD':>7s} {'L net':>7s} {'S net':>7s}")
    rows_csv = []
    for name in variants:
        tr = trades[name]
        m = metrics(tr, k)
        tl = tr[tr["direction"] == "UP"]
        tsh = tr[tr["direction"] == "DOWN"]
        lnet = tl["net"].mean() * 1e4 if len(tl) else np.nan
        snet = tsh["net"].mean() * 1e4 if len(tsh) else np.nan
        print(f"{name:16s} {m['n']:>4d} {len(tl):>4d} {len(tsh):>4d} "
              f"{m['wr']*100:>5.1f}% {m['avg_net_bps']:>+7.1f} "
              f"{m['cum_net_pct']:>+7.1f}% {m['mdd_pct']:>6.2f}% "
              f"{lnet:>+6.1f} {snet:>+6.1f}")
        rows_csv.append(dict(variant=name, n=m["n"], n_long=len(tl),
                             n_short=len(tsh), wr=m["wr"],
                             net_bps=m["avg_net_bps"],
                             cum_pct=m["cum_net_pct"], mdd=m["mdd_pct"],
                             long_net_bps=lnet, short_net_bps=snet))

    # ---- gauntlet vs BASE ----------------------------------------------------
    base_tr = trades[BASE]
    all_months = sorted(month_of(base_tr["entry_ts"]).unique())
    sel_months = set(all_months[:SELECT_MONTHS])
    con_months = set(all_months[SELECT_MONTHS:])

    def window(tr, months):
        return tr[month_of(tr["entry_ts"]).isin(months)]

    base_sel, base_con = window(base_tr, sel_months), window(base_tr, con_months)
    base_by_month = {mo: g["net"].sum()
                     for mo, g in base_tr.groupby(month_of(base_tr["entry_ts"]))}

    # NOTE on the comparison unit: tightening a cutoff mostly REMOVES trades.
    # net/trade can rise while cum falls (fewer trades).  A cutoff variant
    # must win on BOTH net/trade and cum to be deployable — otherwise it is
    # just trading less, not better.
    print("\nGAUNTLET vs BASE_5/5 (net/trade AND cum must improve):")
    print(f"{'variant':16s} {'sel d':>7s} {'con d':>7s} {'months':>7s} "
          f"{'boot CI(bps)':>18s} {'p(<=0)':>7s} {'dCum':>7s} {'dMDD':>6s} "
          f"{'verdict':>10s}")
    for name in variants:
        if name == BASE:
            continue
        tr = trades[name]
        sel, con = window(tr, sel_months), window(tr, con_months)
        d_sel = sel["net"].mean() - base_sel["net"].mean()
        d_con = (con["net"].mean() - base_con["net"].mean()
                 if len(con) and len(base_con) else np.nan)
        wins = tot = 0
        for mo, g in tr.groupby(month_of(tr["entry_ts"])):
            if mo in base_by_month:
                tot += 1
                wins += int(g["net"].sum() > base_by_month[mo])
        a, b = tr["net"].to_numpy(), base_tr["net"].to_numpy()
        boots = (rng.choice(a, (BOOT_N, len(a))).mean(axis=1)
                 - rng.choice(b, (BOOT_N, len(b))).mean(axis=1))
        lo, hi = np.percentile(boots, [2.5, 97.5]) * 1e4
        p_le0 = float((boots <= 0).mean())
        d_cum = metrics(tr, k)["cum_net_pct"] - metrics(base_tr, k)["cum_net_pct"]
        d_mdd = metrics(tr, k)["mdd_pct"] - metrics(base_tr, k)["mdd_pct"]
        ok = (d_sel > 0 and np.isfinite(d_con) and d_con > 0
              and tot > 0 and wins / tot >= 0.60 and p_le0 < 0.05
              and d_cum > 0 and d_mdd <= 2.0)
        verdict = "CANDIDATE" if ok else "no-go"
        print(f"{name:16s} {d_sel*1e4:>+6.1f} "
              f"{(d_con*1e4 if np.isfinite(d_con) else float('nan')):>+6.1f} "
              f"{wins:>4d}/{tot:<2d} [{lo:>+7.1f},{hi:>+7.1f}] {p_le0:>6.3f} "
              f"{d_cum:>+6.1f}% {d_mdd:>+5.1f}pp {verdict:>10s}")

    pd.DataFrame(rows_csv).to_csv(OUT_CSV, index=False)
    print(f"\nWrote → {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
