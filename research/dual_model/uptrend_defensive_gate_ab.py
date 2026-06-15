"""Uptrend-defensive gate A/B (2026-06-06).

CONTEXT: This session confirmed (WF, fold-stable: 5/5 months negative) that the
direction model is a MEAN-REVERSION / contrarian long -- it earns going long into
weakness (TR_BEAR UP +18bps, falling-market UP +11bps) and BLEEDS going long into
strength (TR_BULL UP -15bps, rising-market UP -22bps). A sustained rally is its
worst regime.

HYPOTHESIS (wall-2, sizing): cutting UP exposure when the market is already in a
confirmed uptrend removes a robustly net-negative bucket and improves the kept
stream. DEFENSIVE only -- it reduces the bleed, it does NOT capture the rally (the
model has no trend-following edge to add).

DISCIPLINE (mistake.md 2026-06-02 + project_bear_up_gate_nogo):
- Compare vs TIER (the incumbent validated baseline), not just flat. The bear-UP
  gate looked sensible but HURT because it gated a GOOD bucket; a gate must beat
  the current baseline to deploy.
- All rules normalized to avg deployed = 1.0x (their convention) so differences are
  pure ALLOCATION SHAPE, not leverage. avg_dep column shows how much each gate
  actually sits out.
- Best gate vs tier must pass: per-fold mean-PnL diff > 0 in a majority of folds
  AND bootstrap 95% CI not spanning 0. Aggregate edge alone = suspected outlier.
- CONTROL: 'ALL-off@rising' gates BOTH directions in rising markets -- if the UP
  gate's benefit is real and UP-specific, the UP gate should beat this control.

Caveat: fixed-4h-hold OOS proxy (y_path_ret_4h - round-trip cost), not the live
trailing-stop exit. Fine for RELATIVE rule comparison; absolute numbers are a proxy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.dual_model.position_sizing_oos_backtest import (  # noqa: E402
    COST, OOS_PATH, decode, metrics, normalize,
)

CACHE = "research/dual_model/.cache/features_all.parquet"
RISE_THR = 0.005  # 24h return above this = "market already rising" (matches the
                  # is_trending_bull ret24 threshold)


def main():
    oos = pd.read_parquet(OOS_PATH).sort_index()
    feat = pd.read_parquet(CACHE)
    feat = feat.copy()
    feat["ret24"] = feat["close"].pct_change(24)

    d, t, _ = decode(oos["pred_ret"].values)
    oos["dir"], oos["tier"] = d, t
    df = oos.join(feat[["ret24", "is_trending_bull"]], how="inner").dropna(
        subset=["ret24"])
    sig = df[df["dir"] != "NEUTRAL"].copy().sort_index()

    dsign = np.where(sig["dir"].values == "UP", 1.0, -1.0)
    r = dsign * sig["y_path_ret_4h"].values - COST
    span_days = (sig.index[-1] - sig.index[0]).days or 1

    is_up = sig["dir"].values == "UP"
    is_strong = sig["tier"].values == "Strong"
    rising = sig["ret24"].values > RISE_THR
    tr_bull = sig["is_trending_bull"].values == 1
    w_tier = np.where(is_strong, 1.0, 0.5)

    print(f"signals={len(sig)} span={span_days}d cost={COST*1e4:.0f}bps "
          f"rise_thr={RISE_THR*100:.1f}%  (all normalized to avg dep 1.0x)")
    print(f"  UP signals in rising market: {int((is_up & rising).sum())}  "
          f"UP in TR_BULL: {int((is_up & tr_bull).sum())}")

    rules = {
        "flat": np.ones(len(sig)),
        "tier (S1.0/M0.5) [incumbent]": w_tier.copy(),
        "tier + UP-off @rising": w_tier * np.where(is_up & rising, 0.0, 1.0),
        "tier + UP-half @rising": w_tier * np.where(is_up & rising, 0.5, 1.0),
        "tier + UP-off @TR_BULL": w_tier * np.where(is_up & tr_bull, 0.0, 1.0),
        "tier + ALL-off @rising [control]": w_tier * np.where(rising, 0.0, 1.0),
    }

    rows = []
    for name, w in rules.items():
        avg_dep = float(np.asarray(w, float).mean())
        L = normalize(w)
        m = metrics(L, r, span_days)
        m["rule"] = name
        m["net_bps"] = float((L * r).mean() * 1e4)
        m["avg_dep"] = avg_dep
        rows.append(m)

    print(f"\n{'rule':34s} {'net_bps':>8s} {'Sharpe':>7s} {'MDD%':>7s} "
          f"{'term':>6s} {'avg_dep':>7s}")
    tier = next(x for x in rows if x["rule"].startswith("tier (S"))
    for m in sorted(rows, key=lambda x: -x["sharpe"]):
        tag = ""
        if m is tier:
            tag = "  <= incumbent"
        elif m["rule"].startswith("flat"):
            tag = "  (baseline)"
        print(f"  {m['rule']:32s} {m['net_bps']:+8.1f} {m['sharpe']:7.2f} "
              f"{m['mdd']:7.1f} {m['term']:6.2f} {m['avg_dep']:7.2f}{tag}")

    # Best GATE (exclude flat, tier, control) vs incumbent tier: gauntlet.
    gates = [x for x in rows if x["rule"].startswith("tier +")
             and "control" not in x["rule"]]
    best = max(gates, key=lambda x: x["sharpe"])
    print(f"\nbest gate = '{best['rule']}'  vs incumbent tier:")

    wb = normalize(rules[best["rule"]])
    wt = normalize(rules["tier (S1.0/M0.5) [incumbent]"])
    pnl_b, pnl_t = wb * r, wt * r
    rng = np.random.default_rng(42)
    diffs = np.array([
        (pnl_b[idx].mean() - pnl_t[idx].mean()) * 1e4
        for idx in (rng.integers(0, len(r), len(r)) for _ in range(3000))])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    pgt = float((diffs > 0).mean()) * 100

    sig2 = sig.copy()
    sig2["d"] = pnl_b - pnl_t
    fold_d = np.array([g["d"].mean() * 1e4 for f, g in sig2.groupby("fold")
                       if len(g) >= 3])
    frac_pos = float((fold_d > 0).mean()) * 100

    print(f"  per-signal mean-PnL diff: 95% CI [{lo:+.1f},{hi:+.1f}]bps  "
          f"P(better)={pgt:.0f}%")
    print(f"  per-fold: mean {fold_d.mean():+.1f}bps, {frac_pos:.0f}% of "
          f"{len(fold_d)} folds positive, median {np.median(fold_d):+.1f}bps")

    ci_ok = lo > 0
    fold_ok = frac_pos > 55 and fold_d.mean() > 0
    sharpe_ok = best["sharpe"] > tier["sharpe"]
    mdd_ok = best["mdd"] >= tier["mdd"]  # less negative = shallower = better
    ctrl = next(x for x in rows if "control" in x["rule"])
    up_specific = best["sharpe"] > ctrl["sharpe"]

    print("\n" + "=" * 66)
    print("GAUNTLET (all must pass to deploy the gate):")
    print(f"  Sharpe > incumbent      : {best['sharpe']:.2f} vs {tier['sharpe']:.2f}"
          f"  {'PASS' if sharpe_ok else 'FAIL'}")
    print(f"  MDD shallower           : {best['mdd']:.1f} vs {tier['mdd']:.1f}"
          f"  {'PASS' if mdd_ok else 'FAIL'}")
    print(f"  bootstrap CI excl. 0    : [{lo:+.1f},{hi:+.1f}]"
          f"  {'PASS' if ci_ok else 'FAIL (spans 0)'}")
    print(f"  per-fold >55% positive  : {frac_pos:.0f}%"
          f"  {'PASS' if fold_ok else 'FAIL'}")
    print(f"  UP-specific (beats ctrl): {best['sharpe']:.2f} vs ctrl "
          f"{ctrl['sharpe']:.2f}  {'PASS' if up_specific else 'FAIL'}")
    verdict = all([sharpe_ok, mdd_ok, ci_ok, fold_ok, up_specific])
    print(f"\n  VERDICT: {'DEPLOY-CANDIDATE' if verdict else 'NO-GO'}  "
          f"(NO-GO = leave system as-is, the gate is not a validated improvement)")


if __name__ == "__main__":
    main()
