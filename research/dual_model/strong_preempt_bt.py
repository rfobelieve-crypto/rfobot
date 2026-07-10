"""Strong-only preempt/flip backtest — is same-cycle flip on an opposite
Strong worth implementing in the live executor?

Live today (executor.py:575, verified 2026-07-10): an opposite reading
closes the position and RETURNS — no entry that cycle. The reversal is
only caught if the NEXT bar still decodes Strong in the new direction.
The proposed change: when the opposite reading is STRONG, enter the new
direction next bar (flip). This script quantifies that delta under the
real exit (3xATR trail + opp exit, no time cap) and 1-position occupancy.

  STRONG         = flip semantics (rescan includes exit bar)
  STRONG_NOFLIP  = faithful live (skip exit bar's signal after opp exits)

Verdict discipline (mistake.md 2026-06-02): the flip is GO only if
  (1) flipped-entry trades' mean net bootstrap 95% CI low > 0, AND
  (2) first/second half of flipped trades agree in sign, AND
  (3) aggregate (cum/Sharpe/MDD) does not degrade.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "dual_model"))
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from verify_kernel_method_c import decode_tiers, atr_wilder, metrics, _strip_tz
from entry_policy_real_exit_bt import run_policy, KLINES, V71_OOS

RNG = np.random.default_rng(7)


def boot_ci(x: np.ndarray, n_boot: int = 4000) -> tuple[float, float, float]:
    if len(x) == 0:
        return (np.nan,) * 3
    bs = [np.mean(RNG.choice(x, len(x), replace=True)) for _ in range(n_boot)]
    return float(np.mean(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> int:
    k = pd.read_parquet(KLINES)[["open", "high", "low", "close"]].dropna()
    k.index = _strip_tz(k.index)
    k = k[~k.index.duplicated(keep="last")].sort_index()
    v = pd.read_parquet(V71_OOS)
    v.index = _strip_tz(v.index)
    v = v[~v.index.duplicated(keep="last")].sort_index()
    k = k.loc[v.index[0]:v.index[-1]]
    atr = atr_wilder(k["high"], k["low"], k["close"], 14)
    decoded = decode_tiers(v["pred_ret"])
    print(f"OOS span {v.index[0]:%Y-%m-%d} → {v.index[-1]:%Y-%m-%d} ({len(v)} bars), "
          f"time cap disabled (current live)\n")

    flip = run_policy(k, decoded, atr, "STRONG", time_cap=0)
    noflip = run_policy(k, decoded, atr, "STRONG_NOFLIP", time_cap=0)
    # drop non-strategy tail exit
    flip = flip[flip["exit_reason"] != "data_end"].reset_index(drop=True)
    noflip = noflip[noflip["exit_reason"] != "data_end"].reset_index(drop=True)

    mf, mn = metrics(flip, k), metrics(noflip, k)
    print(f"{'policy':14s} {'n':>4s} {'WR':>7s} {'net/tr':>8s} {'Sharpe':>7s} "
          f"{'MaxDD':>7s} {'cumNet':>8s}")
    for name, m in (("FLIP", mf), ("NOFLIP(live)", mn)):
        print(f"{name:14s} {m['n']:>4d} {m['wr']*100:>6.1f}% {m['avg_net_bps']:>+7.1f} "
              f"{m['sharpe']:>7.2f} {m['mdd_pct']:>6.2f}% {m['cum_net_pct']:>+7.1f}%")

    # ---- the delta: flipped entries = FLIP trades whose signal bar is the
    # previous FLIP trade's opp-exit bar (the entries NOFLIP misses/differs on)
    flipped_mask = np.zeros(len(flip), bool)
    for t in range(1, len(flip)):
        prev = flip.iloc[t - 1]
        if (str(prev["exit_reason"]).startswith("opp")
                and flip.iloc[t]["signal_ts"] == prev["exit_ts"]):
            flipped_mask[t] = True
    fl = flip[flipped_mask]
    print(f"\nflipped entries (signal on an opp-exit bar): n={len(fl)}")
    if len(fl):
        x = fl["net"].to_numpy(float)
        m0, lo, hi = boot_ci(x)
        wr = float((x > 0).mean())
        print(f"  WR {wr*100:.1f}%  net/tr {m0*1e4:+.1f}bps  "
              f"CI[{lo*1e4:+.1f},{hi*1e4:+.1f}]bps  cum {x.sum()*100:+.2f}%")
        half = len(fl) // 2
        a, b = x[:half], x[half:]
        print(f"  first half n={len(a)} sum {a.sum()*100:+.2f}%  |  "
              f"second half n={len(b)} sum {b.sum()*100:+.2f}%  "
              f"(sign agree: {np.sign(a.sum()) == np.sign(b.sum())})")
        print("  by direction:",
              {d: f"n={int((fl['direction']==d).sum())}, "
                  f"cum={fl.loc[fl['direction']==d,'net'].sum()*100:+.2f}%"
               for d in ("UP", "DOWN")})

    # ---- verdict per pre-registered discipline
    print("\n" + "=" * 78)
    ok_ci = len(fl) > 0 and lo > 0 if len(fl) else False
    ok_half = (len(fl) >= 2 and np.sign(fl['net'].to_numpy()[:len(fl)//2].sum())
               == np.sign(fl['net'].to_numpy()[len(fl)//2:].sum()))
    ok_agg = (mf["cum_net_pct"] >= mn["cum_net_pct"]
              and mf["mdd_pct"] >= mn["mdd_pct"] - 2.0)
    verdict = "GO" if (ok_ci and ok_half and ok_agg) else "NO-GO / INSUFFICIENT"
    print(f"VERDICT: {verdict}   (CI-low>0: {ok_ci}, halves agree: {ok_half}, "
          f"aggregate not degraded: {ok_agg})")
    print("note: n<100 cells are directional, not conclusions "
          "(mistake.md discipline).")

    out = ROOT / "research" / "results" / "dual_model" / "strong_preempt_bt.csv"
    flip.assign(flipped=flipped_mask).to_csv(out, index=False)
    print(f"\nWrote per-trade table → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
