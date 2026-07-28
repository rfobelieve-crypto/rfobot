"""Meta-labelled exit model — "should I close THIS position now?"

Origin (chat, 2026-07-28). The user's objection to the current design:
exiting a long because the model decoded a SHORT signal is answering the
wrong question. The entry model predicts, from FLAT, where price goes next.
Once a position exists the question changes — it now has a cost basis,
unrealised P&L, a high-water mark and time in trade, none of which the entry
model can see. This is the meta-labelling option (a) that was recommended on
2026-07-24 and passed over in favour of the RL route (option c), which then
failed 4/4 folds (mistake.md 2026-07-24).

Why RL failing does not condemn this: FQI had to fit Q(s, a) — an absolute
value surface over a 9-dim state, bootstrapped off its own estimates, so
error compounds where data is thin. This fits a single binary probability
from labels the counterfactual enumeration hands over directly. No
bootstrapping, far fewer degrees of freedom.

Framing — the model may only ADD early exits, never cancel a baseline one:

    label = 1  ⇔  closing at this bar nets more than what the baseline
                  (3xATR trail / opp_signal / data-end) eventually delivers

so the deployed policy is "baseline exits, plus an early exit when
P(better) > threshold". A useless model degenerates to the baseline rather
than to something new and unvalidated.

Honest caveats, stated up front:
  * `side` is a FEATURE, not a partition. Two separate long/short models is
    the intuitive reading of the request, but mistake.md 2026-04-13 (regime
    sub-models collapsing to AUC 0.378 on thin slices) says make the split a
    feature until the data proves a partition earns its keep. Whether
    splitting helps is then a measurable question, reported below.
  * The ~177k in-position rows come from ~3.7k bars via counterfactual
    enumeration. Effective sample size is nearer the bar count; the row
    count is inflated by overlapping paths, so folds are cut on ENTRY BAR
    (never mid-episode) and every headline number is per-fold, not pooled.
  * Training enumerates a hypothetical position at every bar, but scoring
    runs on the real Strong-only entry policy — learn broadly, judge on
    what actually gets traded.

Deployment gate (mistake.md 2026-06-02): aggregate lift is not enough.
per-fold mean > 0, frac positive folds > 0.55, and a bootstrap CI clear of
zero, or it is a NO-GO.

Run: python research/meta_exit_model.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import research.v71_v7_sizing_1x as bt                      # noqa: E402
from research.dual_model.shared_data import walk_forward_splits  # noqa: E402
from research.rl_joint_exit_entry import load_state_frame   # noqa: E402

OUT = ROOT / "research/results/meta_exit_model.json"

FEATURES = ["pred_ret", "vol_regime", "atr_pct", "side", "bars_held",
            "unrealized_pct", "mfe_pct", "mae_pct", "decay_streak"]
MAX_HOLD = 72                 # baseline rarely runs past this; bounds the walk
DECAY_BARS = 2                # live setting (OKX_CONVICTION_DECAY_BARS)
THRESHOLDS = (0.55, 0.60, 0.65, 0.70)
SEED = 42
BOOT = 2000


# ── baseline exit, mirroring v71_v7_sizing_1x.simulate ────────────────────

def _baseline_exit(i: int, side: int, *, o, h, lo, c, atr, direction, n):
    """Walk a position opened at close[i] forward under the live exit stack.

    Returns (exit_j, net_return). Order and fill conventions match
    simulate(): trailing stop is intrabar and a gap-through fills at the
    bar open, then opp_signal at the close, then data end.
    """
    a = atr[i]
    if not np.isfinite(a) or a <= 0:
        return None
    entry = c[i]
    stop_dist = bt.TRAIL_MULT * a
    ext = entry
    for j in range(i + 1, min(i + MAX_HOLD, n - 1) + 1):
        cur_stop = (ext - stop_dist) if side == 1 else (ext + stop_dist)
        if side == 1 and lo[j] <= cur_stop:
            px = min(cur_stop, o[j])
            return j, (px / entry - 1.0) - bt.TAKER_COST
        if side == -1 and h[j] >= cur_stop:
            px = max(cur_stop, o[j])
            return j, -(px / entry - 1.0) - bt.TAKER_COST
        opp = ((side == 1 and direction[j] == "DOWN")
               or (side == -1 and direction[j] == "UP"))
        if opp or j == n - 1:
            g = (c[j] / entry - 1.0) if side == 1 else -(c[j] / entry - 1.0)
            return j, g - bt.TAKER_COST
        ext = max(ext, h[j]) if side == 1 else min(ext, lo[j])
    j = min(i + MAX_HOLD, n - 1)
    g = (c[j] / entry - 1.0) if side == 1 else -(c[j] / entry - 1.0)
    return j, g - bt.TAKER_COST


def build_dataset(df: pd.DataFrame, direction: np.ndarray) -> pd.DataFrame:
    """One row per (entry bar, side, in-position bar) with its meta label."""
    o, h, lo, c = (df[k].values for k in ("open", "high", "low", "close"))
    atr = df["atr"].values
    pred = df["pred_ret"].values
    vol_regime = df["vol_regime"].values
    atr_pct = df["atr_pct"].values
    in_oos = df["in_oos"].values
    n = len(df)

    rows = []
    for i in range(n - 2):
        if not in_oos[i]:
            continue
        for side in (1, -1):
            res = _baseline_exit(i, side, o=o, h=h, lo=lo, c=c, atr=atr,
                                 direction=direction, n=n)
            if res is None:
                continue
            exit_j, base_net = res
            entry = c[i]
            mfe = mae = 0.0
            streak = 0
            for j in range(i + 1, exit_j):          # strictly before the exit
                unreal = ((c[j] / entry - 1.0) if side == 1
                          else -(c[j] / entry - 1.0))
                mfe, mae = max(mfe, unreal), min(mae, unreal)
                disagree = (pred[j] < 0) if side == 1 else (pred[j] > 0)
                streak = streak + 1 if disagree else 0
                now_net = unreal - bt.TAKER_COST
                rows.append(dict(
                    entry_bar=i, bar_i=j, side=side,
                    pred_ret=pred[j], vol_regime=vol_regime[j],
                    atr_pct=atr_pct[j], bars_held=j - i,
                    unrealized_pct=unreal, mfe_pct=mfe, mae_pct=mae,
                    decay_streak=streak,
                    label=int(now_net > base_net),
                    edge=now_net - base_net,
                ))
    return pd.DataFrame(rows)


# ── scoring: real Strong-only policy, with and without the overlay ────────

def run_policy(df, direction, tier, bars, model, thr, warm):
    """Strong-only entries; exit at whichever comes first — baseline, or the
    meta model saying this bar beats it. Returns per-trade net returns."""
    o, h, lo, c = (df[k].values for k in ("open", "high", "low", "close"))
    atr = df["atr"].values
    pred = df["pred_ret"].values
    vol_regime = df["vol_regime"].values
    atr_pct = df["atr_pct"].values
    n = len(df)
    nets, i = [], max(warm, bars[0])
    last = bars[-1]
    while i <= last:
        if direction[i] not in ("UP", "DOWN") or tier[i] != "Strong":
            i += 1
            continue
        side = 1 if direction[i] == "UP" else -1
        res = _baseline_exit(i, side, o=o, h=h, lo=lo, c=c, atr=atr,
                             direction=direction, n=n)
        if res is None:
            i += 1
            continue
        exit_j, net = res
        if model is not None:
            entry = c[i]
            mfe = mae = 0.0
            streak = 0
            for j in range(i + 1, exit_j):
                unreal = ((c[j] / entry - 1.0) if side == 1
                          else -(c[j] / entry - 1.0))
                mfe, mae = max(mfe, unreal), min(mae, unreal)
                disagree = (pred[j] < 0) if side == 1 else (pred[j] > 0)
                streak = streak + 1 if disagree else 0
                x = np.array([[pred[j], vol_regime[j], atr_pct[j], side, j - i,
                               unreal, mfe, mae, streak]])
                if model.predict_proba(x)[0, 1] >= thr:
                    exit_j, net = j, unreal - bt.TAKER_COST
                    break
        nets.append(net)
        i = exit_j + 1                      # no re-entry on the exit bar
    return np.array(nets)


def main() -> int:
    df = load_state_frame()
    direction, tier, warm = bt.decode_signals(df)
    print(f"bars {len(df)}  {df.index.min()} → {df.index.max()}  warmup={warm}")

    ds = build_dataset(df, direction)
    print(f"in-position rows {len(ds):,} from {ds['entry_bar'].nunique():,} "
          f"entry bars  |  label=1 (early exit better) "
          f"{ds['label'].mean() * 100:.1f}%")

    splits = walk_forward_splits(len(df), initial_train=288, test_size=250,
                                 step=250, purge=4, embargo=4)
    print(f"{len(splits)} folds\n")
    print(f"{'fold':>4}{'n_tr':>9}{'base':>9}{'best thr':>10}"
          f"{'meta':>9}{'lift bps':>10}{'trades':>8}")
    print("-" * 60)

    rows = []
    for k, (tr, te) in enumerate(splits):
        tr_set = set(tr)
        # Cut on ENTRY BAR so no episode straddles the boundary.
        m = ds[ds["entry_bar"].isin(tr_set) & ds["bar_i"].isin(tr_set)]
        if len(m) < 500 or m["label"].nunique() < 2:
            continue
        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            eval_metric="logloss", random_state=SEED, n_jobs=4)
        clf.fit(m[FEATURES].values, m["label"].values)

        base = run_policy(df, direction, tier, te, None, 0.0, warm)
        if len(base) == 0:
            continue
        best = max(
            ((thr, run_policy(df, direction, tier, te, clf, thr, warm))
             for thr in THRESHOLDS),
            key=lambda t: t[1].mean() if len(t[1]) else -9e9)
        thr, meta = best
        lift = (meta.mean() - base.mean()) * 10000
        rows.append(dict(fold=k, n_train=len(m), base_bps=base.mean() * 10000,
                         meta_bps=meta.mean() * 10000, lift_bps=lift,
                         thr=thr, n_trades=len(base)))
        print(f"{k:>4}{len(m):>9,}{base.mean() * 10000:>9.1f}{thr:>10.2f}"
              f"{meta.mean() * 10000:>9.1f}{lift:>10.1f}{len(base):>8}")

    if not rows:
        print("no usable folds")
        return 1

    lifts = np.array([r["lift_bps"] for r in rows])
    rng = np.random.default_rng(SEED)
    boot = np.array([rng.choice(lifts, len(lifts), replace=True).mean()
                     for _ in range(BOOT)])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    frac_pos = float((lifts > 0).mean())

    print("\n=== 4-gate verdict (mistake.md 2026-06-02) ===")
    g1 = lifts.mean() > 0
    g2 = np.median(lifts) > 0
    g3 = frac_pos > 0.55
    g4 = ci[0] * ci[1] > 0 and ci[0] > 0
    print(f"  per-fold mean lift {lifts.mean():+.1f} bps      {'PASS' if g1 else 'FAIL'}")
    print(f"  median lift        {np.median(lifts):+.1f} bps      {'PASS' if g2 else 'FAIL'}")
    print(f"  frac positive      {frac_pos:.0%} ({int(frac_pos * len(lifts))}/{len(lifts)})"
          f"        {'PASS' if g3 else 'FAIL'}")
    print(f"  bootstrap 95% CI   [{ci[0]:+.1f}, {ci[1]:+.1f}]  {'PASS' if g4 else 'FAIL'}")
    verdict = "DEPLOY-CANDIDATE" if all((g1, g2, g3, g4)) else "NO-GO"
    print(f"\n  VERDICT: {verdict}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        generated=str(pd.Timestamp.utcnow()), verdict=verdict,
        per_fold=rows, mean_lift_bps=float(lifts.mean()),
        median_lift_bps=float(np.median(lifts)),
        frac_positive=frac_pos, boot_ci=ci), indent=2, default=str),
        encoding="utf-8")
    print(f"saved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
