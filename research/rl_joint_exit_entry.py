"""
Joint entry+exit RL (offline, per-bar MDP) — 2026-07-24.

Origin (chat): user asked "why wasn't entry+exit trained jointly from the
start" and then explicitly chose the RL path over the lower-risk
meta-labeling-exit-model alternative, after being told the sample-size
mismatch (RL usually wants far more transitions than this project has).
This script is the first feasibility pass, not a production candidate.

Why OFFLINE RL, not online: we cannot let an untrained policy trade real
money to explore. Every transition here comes from a static historical
replay.

Why PER-BAR MDP, not per-trade episodes: framing "1 trade = 1 episode"
gives ~14-70 episodes (live/backtest trade counts) — nowhere near enough
for RL. Framing "1 hourly bar = 1 timestep" gives ~3700 timesteps, still
small by RL standards but the only framing with any chance of working.

Key structural trick that makes this tractable at all: our own trading
does not move the BTC market (small account). That means, unlike a real
RL environment, we can compute EXACT counterfactual transitions for every
action at every bar (not just the action actually taken historically) —
for every bar, "what if I entered LONG/SHORT/stayed flat here" and "what
if I held/exited a position that started at bar j" are both exactly
computable from the same fixed price path. This sidesteps the standard
offline-RL distributional-shift problem (no need for importance sampling
or a stochastic behavior policy) at the cost of an enumeration over
(entry_bar x max_hold_bars) instead of a single pass.

State (kept deliberately small — 282 raw features would blow the
sample-to-parameter ratio given ~3700 bars; only the info actually
available to the live executor at decision time):
  flat:      pred_ret, vol_regime, atr_pct
  in-pos:    pred_ret, vol_regime, atr_pct, side(+1/-1), bars_held,
             unrealized_pct, mfe_pct, mae_pct, decay_streak

Action spaces (masked by flat/in-position — two separate Q-heads,
avoids modeling an invalid action like "exit" while flat):
  flat:      {0: stay_flat, 1: enter_long, 2: enter_short}
  in-pos:    {0: hold, 1: exit}

Algorithm: Fitted Q-Iteration (FQI) with XGBoost regressors as the
Q-function approximator (consistent with the rest of this project's
tooling). Q_pos and Q_flat are fit jointly across outer iterations,
Q_flat's entry actions bootstrap off Q_pos at bars_held=0.

Validation: walk_forward_splits (purge+embargo, shared_data.py) — refit
FQI per fold on train, evaluate the GREEDY policy via full rollout
simulation on test (valid here specifically because price path is
action-independent — see note above). Compared against the existing
production baseline (trail_stop/opp_signal) and conviction_decay using
this project's own 4-gate discipline (mistake.md 2026-06-02):
  aggregate lift, per-fold mean lift, frac_positive folds, bootstrap CI.

Run: python research/rl_joint_exit_entry.py
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import research.v71_v7_sizing_1x as bt
from research.dual_model.shared_data import walk_forward_splits

RESULTS_DIR = PROJECT_ROOT / "research" / "results"
FEATURES_PARQUET = PROJECT_ROOT / "research/dual_model/.cache/features_all.parquet"

TAKER_COST = bt.TAKER_COST          # 8 bps round-trip, applied at exit
MAX_HOLD_BARS = 24                  # enumeration cap (1 day) — keeps the
                                     # transition dataset tractable; production
                                     # trades rarely run past this per
                                     # exit_decomposition.py's bars_held dist
GAMMA = 0.97                        # per-bar discount (~24-bar horizon still
                                     # weights early bars meaningfully)
FQI_ITERS = 12
N_FOLDS_TARGET_TEST_BARS = 250      # ~10 days per fold


# ── State construction ────────────────────────────────────────────────────

def load_state_frame() -> pd.DataFrame:
    """pred_ret / atr / OHLC from v71 loader, + vol_regime joined in."""
    df = bt.load_data()
    feats = pd.read_parquet(FEATURES_PARQUET)
    feats.index = bt._to_naive_utc(pd.DatetimeIndex(feats.index))
    df = df.join(feats[["vol_regime"]], how="left")
    df["vol_regime"] = df["vol_regime"].fillna(1.0)
    df["atr_pct"] = df["atr"] / df["close"]
    return df


FLAT_STATE_COLS = ["pred_ret", "vol_regime", "atr_pct"]
POS_STATE_COLS = ["pred_ret", "vol_regime", "atr_pct", "side", "bars_held",
                   "unrealized_pct", "mfe_pct", "mae_pct", "decay_streak"]


# ── Exact counterfactual transition enumeration ─────────────────────────────

def build_transitions(df: pd.DataFrame, lo_idx: int, hi_idx: int):
    """Enumerate exact transitions for bars in [lo_idx, hi_idx).

    Returns (flat_df, pos_df). Both carry a `bar_i` column (the CURRENT
    bar's positional index) so callers can slice by fold without redoing
    the O(n * MAX_HOLD_BARS) enumeration per fold.
    """
    c = df["close"].values
    o = df["open"].values
    h = df["high"].values
    lo = df["low"].values
    pred = df["pred_ret"].values
    vol_regime = df["vol_regime"].values
    atr_pct = df["atr_pct"].values
    in_oos = df["in_oos"].values
    n = len(df)

    flat_rows = []
    pos_rows = []

    for i in range(lo_idx, hi_idx):
        if not in_oos[i] or i + 1 >= n:
            continue

        # -- flat-state transitions at bar i --
        # stay_flat: no reward, next flat state at bar i+1
        flat_rows.append(dict(
            bar_i=i, action=0,
            pred_ret=pred[i], vol_regime=vol_regime[i], atr_pct=atr_pct[i],
            reward=0.0, next_bar_i=i + 1, next_kind="flat",
        ))
        # enter_long / enter_short: no reward at entry (cost deferred to
        # exit, matches bt.simulate convention), next state = in-position
        # at bars_held=0 (evaluated at bar i+1, entry price = close[i])
        for side_code, side_name, act in ((1, "LONG", 1), (-1, "SHORT", 2)):
            flat_rows.append(dict(
                bar_i=i, action=act,
                pred_ret=pred[i], vol_regime=vol_regime[i], atr_pct=atr_pct[i],
                reward=0.0, next_bar_i=i + 1, next_kind="pos",
                entry_bar=i, entry_price=c[i], side=side_code,
            ))

        # -- in-position transitions for a HYPOTHETICAL position opened
        #    at bar i (evaluated forward up to MAX_HOLD_BARS or data end) --
        entry_price = c[i]
        for side_code in (1, -1):
            mfe = 0.0
            mae = 0.0
            decay_streak = 0
            max_j = min(i + MAX_HOLD_BARS, n - 1)
            for j in range(i + 1, max_j + 1):
                bars_held = j - i
                if side_code == 1:
                    unreal = c[j] / entry_price - 1.0
                    bar_ret = c[j] / c[j - 1] - 1.0
                else:
                    unreal = -(c[j] / entry_price - 1.0)
                    bar_ret = -(c[j] / c[j - 1] - 1.0)
                mfe = max(mfe, unreal)
                mae = min(mae, unreal)
                disagreeing = (pred[j] < 0) if side_code == 1 else (pred[j] > 0)
                decay_streak = decay_streak + 1 if disagreeing else 0

                state = dict(
                    bar_i=j, entry_bar=i, side=side_code,
                    pred_ret=pred[j], vol_regime=vol_regime[j], atr_pct=atr_pct[j],
                    bars_held=bars_held, unrealized_pct=unreal,
                    mfe_pct=mfe, mae_pct=mae, decay_streak=decay_streak,
                )

                # action=1 (exit): terminal, realized net return this bar
                net = unreal - TAKER_COST
                pos_rows.append(dict(**state, action=1, reward=net,
                                      next_kind="terminal", next_bar_i=-1))

                # action=0 (hold): dense mark-to-market reward, bootstraps
                # to the next in-position state UNLESS we hit the hold cap,
                # in which case it's a forced exit (reward = full realized
                # net return this bar, terminal) — keeps every enumerated
                # trajectory finite and avoids an artificial "free" hold
                # value at the boundary.
                if bars_held >= MAX_HOLD_BARS or j >= n - 1:
                    pos_rows.append(dict(**state, action=0, reward=net,
                                          next_kind="terminal", next_bar_i=-1))
                    break
                else:
                    pos_rows.append(dict(**state, action=0, reward=bar_ret,
                                          next_kind="pos_next", next_bar_i=j + 1))

    return pd.DataFrame(flat_rows), pd.DataFrame(pos_rows)


# ── Fitted Q-Iteration ──────────────────────────────────────────────────────

def _xgb_fit(X: np.ndarray, y: np.ndarray) -> xgb.XGBRegressor:
    m = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.08,
                          subsample=0.8, colsample_bytree=0.8,
                          reg_lambda=1.0, n_jobs=-1, verbosity=0)
    m.fit(X, y)
    return m


def fit_fqi(flat_df: pd.DataFrame, pos_df: pd.DataFrame):
    """Iterate Q_pos and Q_flat. Returns (q_pos_model, q_flat_model) —
    each a dict {action: fitted XGBRegressor}."""
    pos_X = pos_df[POS_STATE_COLS].values
    flat_X = flat_df[FLAT_STATE_COLS].values

    # index pos_df by (entry_bar, side, next_bar_i) so hold-action targets
    # can look up next state's features to bootstrap Q_pos(next_state, *)
    # state features are identical between the action=0/action=1 rows that
    # share the same (entry_bar, side, bar_i) — dedupe before indexing, else
    # the lookup key is non-unique and .loc returns 2 stacked rows.
    pos_state_unique = pos_df.drop_duplicates(subset=["entry_bar", "side", "bar_i"])
    pos_lookup = pos_state_unique.set_index(["entry_bar", "side", "bar_i"])
    pos_lookup = pos_lookup.assign(
        side=pos_lookup.index.get_level_values("side"))[POS_STATE_COLS]
    pos_lookup = pos_lookup.sort_index()

    q_pos = {0: None, 1: None}
    q_flat = {0: None, 1: None, 2: None}

    for it in range(FQI_ITERS):
        # ---- Q_pos targets ----
        # action=1 (exit) target is just `reward` (terminal, no bootstrap)
        exit_mask = pos_df["action"] == 1
        hold_mask = pos_df["action"] == 0

        target_pos = pos_df["reward"].values.copy().astype(float)
        if q_pos[0] is not None and q_pos[1] is not None:
            hold_next_mask = hold_mask & (pos_df["next_kind"] == "pos_next")
            if hold_next_mask.any():
                nxt = pos_df.loc[hold_next_mask]
                nxt_X = nxt.apply(
                    lambda r: pos_lookup.loc[(r["entry_bar"], r["side"], r["next_bar_i"])].values
                    if (r["entry_bar"], r["side"], r["next_bar_i"]) in pos_lookup.index
                    else None, axis=1)
                valid = nxt_X.notna()
                if valid.any():
                    Xn = np.stack(nxt_X[valid].values)
                    q0 = q_pos[0].predict(Xn)
                    q1 = q_pos[1].predict(Xn)
                    boot = GAMMA * np.maximum(q0, q1)
                    idx = nxt.index[valid]
                    target_pos[pos_df.index.get_indexer(idx)] += boot

        q_pos_new = {
            0: _xgb_fit(pos_X[hold_mask.values], target_pos[hold_mask.values]),
            1: _xgb_fit(pos_X[exit_mask.values], target_pos[exit_mask.values]),
        }

        # ---- Q_flat targets ----
        target_flat = flat_df["reward"].values.copy().astype(float)
        stay_mask = (flat_df["action"] == 0).values
        enter_mask = (flat_df["action"] != 0).values

        if enter_mask.any():
            ent = flat_df.loc[enter_mask]
            # entry lands at bars_held=0 of a position opened at bar_i
            # (= entry_bar), evaluated at next_bar_i
            key = list(zip(ent["bar_i"], ent["side"], ent["next_bar_i"]))
            look = [pos_lookup.loc[k].values if k in pos_lookup.index else None
                    for k in key]
            valid = [i for i, v in enumerate(look) if v is not None]
            if valid:
                Xn = np.stack([look[i] for i in valid])
                q0 = q_pos_new[0].predict(Xn)
                q1 = q_pos_new[1].predict(Xn)
                boot = GAMMA * np.maximum(q0, q1)
                rows = ent.index[valid]
                target_flat[flat_df.index.get_indexer(rows)] += boot

        if q_flat[0] is not None:
            stay = flat_df.loc[stay_mask]
            nxt_X = stay[["pred_ret", "vol_regime", "atr_pct"]].copy()
            # next flat state = bar (bar_i+1)'s flat features; approximate
            # via a shifted lookup on the flat table restricted to action=0
            flat_by_bar = flat_df[flat_df["action"] == 0].set_index("bar_i")[FLAT_STATE_COLS]
            key = stay["next_bar_i"].values
            mask_in = np.isin(key, flat_by_bar.index.values)
            if mask_in.any():
                Xn = flat_by_bar.loc[key[mask_in]].values
                qvals = np.stack([q_flat[a].predict(Xn) for a in (0, 1, 2)], axis=1)
                boot = GAMMA * qvals.max(axis=1)
                rows = stay.index[mask_in]
                target_flat[flat_df.index.get_indexer(rows)] += boot

        q_flat_new = {a: _xgb_fit(flat_X[(flat_df["action"] == a).values],
                                   target_flat[(flat_df["action"] == a).values])
                      for a in (0, 1, 2)}

        q_pos, q_flat = q_pos_new, q_flat_new
        print(f"    FQI iter {it+1}/{FQI_ITERS} done "
              f"(pos n={len(pos_df)}, flat n={len(flat_df)})")

    return q_pos, q_flat


# ── Greedy rollout (= the actual backtest of the learned policy) ───────────

def rollout_policy(df: pd.DataFrame, lo_idx: int, hi_idx: int, q_pos, q_flat,
                    span_days_hint: float = None) -> pd.DataFrame:
    c = df["close"].values
    pred = df["pred_ret"].values
    vol_regime = df["vol_regime"].values
    atr_pct = df["atr_pct"].values
    in_oos = df["in_oos"].values
    ts = df.index

    trades = []
    pos = None
    i = lo_idx
    while i < hi_idx:
        if pos is None:
            if not in_oos[i]:
                i += 1
                continue
            x = np.array([[pred[i], vol_regime[i], atr_pct[i]]])
            qvals = [float(q_flat[a].predict(x)[0]) for a in (0, 1, 2)]
            a = int(np.argmax(qvals))
            if a == 0:
                i += 1
                continue
            side = 1 if a == 1 else -1
            pos = dict(entry_bar=i, side=side, entry_price=c[i],
                       entry_ts=ts[i], mfe=0.0, mae=0.0, decay_streak=0)
            i += 1
            continue

        bars_held = i - pos["entry_bar"]
        if pos["side"] == 1:
            unreal = c[i] / pos["entry_price"] - 1.0
        else:
            unreal = -(c[i] / pos["entry_price"] - 1.0)
        pos["mfe"] = max(pos["mfe"], unreal)
        pos["mae"] = min(pos["mae"], unreal)
        disagreeing = (pred[i] < 0) if pos["side"] == 1 else (pred[i] > 0)
        pos["decay_streak"] = pos["decay_streak"] + 1 if disagreeing else 0

        forced = bars_held >= MAX_HOLD_BARS or i >= hi_idx - 1
        if forced:
            a = 1
        else:
            x = np.array([[pred[i], vol_regime[i], atr_pct[i], pos["side"],
                           bars_held, unreal, pos["mfe"], pos["mae"],
                           pos["decay_streak"]]])
            q0 = float(q_pos[0].predict(x)[0])
            q1 = float(q_pos[1].predict(x)[0])
            a = 1 if q1 >= q0 else 0

        if a == 1:
            net = unreal - TAKER_COST
            trades.append(dict(
                entry_ts=pos["entry_ts"], exit_ts=ts[i],
                side="LONG" if pos["side"] == 1 else "SHORT",
                entry_price=pos["entry_price"], exit_price=c[i],
                bars_held=bars_held, gross_pct=unreal, net_pct=net,
                win=int(unreal > 0)))
            pos = None
        i += 1

    return pd.DataFrame(trades)


def summarize_rl(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return dict(n=0, wr_pct=0.0, avg_net_bps=0.0)
    wins = (trades["win"] == 1).mean() * 100
    avg_bps = trades["net_pct"].mean() * 1e4
    return dict(n=len(trades), wr_pct=round(float(wins), 1),
                avg_net_bps=round(float(avg_bps), 1))


def main():
    print("=" * 76)
    print("  JOINT ENTRY+EXIT RL (offline FQI, per-bar MDP) — feasibility pass")
    print("=" * 76)

    df = load_state_frame()
    n = len(df)
    oos_idx = np.where(df["in_oos"].values)[0]
    lo_all, hi_all = int(oos_idx.min()), int(oos_idx.max()) + 1
    span_days = (df.index[hi_all - 1] - df.index[lo_all]).total_seconds() / 86400.0
    print(f"\n  bars total={n}  in_oos bars={len(oos_idx)}  span={span_days:.0f}d")

    splits = walk_forward_splits(
        n_samples=hi_all - lo_all, initial_train=1200, test_size=N_FOLDS_TARGET_TEST_BARS,
        step=N_FOLDS_TARGET_TEST_BARS, purge=MAX_HOLD_BARS, embargo=MAX_HOLD_BARS)
    print(f"  {len(splits)} walk-forward folds (test~{N_FOLDS_TARGET_TEST_BARS}bars each)")

    direction, tier, _ = bt.decode_signals(df)
    direction = np.asarray(direction, dtype=object)
    tier = np.asarray(tier, dtype=object)

    fold_results = []
    for fi, (train_idx, test_idx) in enumerate(splits):
        train_lo, train_hi = lo_all + train_idx[0], lo_all + train_idx[-1] + 1
        test_lo, test_hi = lo_all + test_idx[0], lo_all + test_idx[-1] + 1

        print(f"\n  -- fold {fi+1}/{len(splits)}: "
              f"train[{df.index[train_lo]}..{df.index[train_hi-1]}] "
              f"test[{df.index[test_lo]}..{df.index[test_hi-1]}] --")

        flat_df, pos_df = build_transitions(df, train_lo, train_hi)
        if flat_df.empty or pos_df.empty:
            print("    skip: empty transition set")
            continue
        q_pos, q_flat = fit_fqi(flat_df, pos_df)

        rl_trades = rollout_policy(df, test_lo, test_hi, q_pos, q_flat)
        rl_s = summarize_rl(rl_trades)

        base_trades_full = bt.simulate(df.iloc[:test_hi].assign(
            in_oos=lambda d: d["in_oos"] & (np.arange(len(d)) >= test_lo)),
            direction[:test_hi], tier[:test_hi])
        base_s = summarize_rl(base_trades_full.rename(columns={"net_pct": "net_pct"}))

        print(f"    RL:       n={rl_s['n']:3d}  WR={rl_s['wr_pct']:5.1f}%  "
              f"avg_net_bps={rl_s['avg_net_bps']:+7.1f}")
        print(f"    baseline: n={base_s['n']:3d}  WR={base_s['wr_pct']:5.1f}%  "
              f"avg_net_bps={base_s['avg_net_bps']:+7.1f}")

        fold_results.append(dict(fold=fi, rl=rl_s, baseline=base_s))

    # ---- 4-gate aggregate ----
    lifts = [f["rl"]["avg_net_bps"] - f["baseline"]["avg_net_bps"] for f in fold_results
             if f["rl"]["n"] > 0]
    if lifts:
        lifts = np.array(lifts)
        rng = np.random.default_rng(42)
        boot = np.array([lifts[rng.integers(0, len(lifts), len(lifts))].mean()
                          for _ in range(5000)])
        print("\n  " + "=" * 60)
        print("  4-GATE VERDICT")
        print("  " + "=" * 60)
        print(f"  per-fold mean lift:   {lifts.mean():+.2f} bps")
        print(f"  frac positive folds:  {(lifts > 0).mean()*100:.1f}%")
        print(f"  bootstrap 95% CI:     [{np.percentile(boot,2.5):+.2f}, "
              f"{np.percentile(boot,97.5):+.2f}] bps")
        verdict = (lifts.mean() > 0 and (lifts > 0).mean() > 0.55
                   and np.percentile(boot, 2.5) > 0)
        print(f"  VERDICT: {'GO' if verdict else 'NO-GO'}")
    else:
        print("\n  NO-GO: RL policy never opened a trade in any fold")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "rl_joint_exit_entry_feasibility.json"
    out.write_text(json.dumps(dict(fold_results=fold_results,
                                    run_ts=pd.Timestamp.now(tz="UTC").isoformat()),
                               indent=2, default=str))
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
