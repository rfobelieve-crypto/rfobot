"""
Full-System Walk-Forward Validation

Validates the fully integrated production stack (Init + Mag + V5 breakout +
gate logic) end-to-end. Two views:

A) IN-SAMPLE (engine on full history):
   Calls IndicatorEngine.predict() on every bar. Models are pre-trained on
   the full data, so direction/strength is biased optimistic — this view
   tells us "does production code wire-up correctly" not "is it accurate".

B) TRUE OOS (replay-based):
   Loads research/results/initiation_replay_oos.parquet (walk-forward
   p_long/p_short/mag_pct from initiation_replay.py) and applies the
   production gate logic (same as IndicatorEngine._predict_dual). This is
   the honest end-to-end accuracy number.

Run:  python -m research.full_system_walkforward
"""
from __future__ import annotations

import sys
import json
import warnings
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import load_and_cache_data
from indicator.initiation_features import add_initiation_features
from indicator.inference import IndicatorEngine

RESULTS_DIR = PROJECT_ROOT / "research" / "results"
OOS_PATH = RESULTS_DIR / "initiation_replay_oos.parquet"
THRESH_PATH = (PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model"
               / "initiation_thresholds.json")

HORIZON = 4
TARGET_K = 0.008  # ±0.8% for first-touch / path-race


def compute_path_outcomes(close, high, low, horizon=HORIZON, k=TARGET_K):
    """
    For each bar t, scan [t+1, t+horizon] and determine:
      - fwd_ret        : close[t+h]/close[t] - 1           (4h closing return)
      - ft_long        : 1 if max(high) / close >= +k      (first-touch long)
      - ft_short       : 1 if min(low)  / close <= -k      (first-touch short)
      - race_long      : +0.8 hit before -0.8 within 4h (1=hit, -1=miss, 0=neither)
      - race_short     : -0.8 hit before +0.8 within 4h
    Returns dict of numpy arrays aligned to close.index.
    """
    c = close.values.astype(float)
    h = high.values.astype(float)
    lo = low.values.astype(float)
    n = len(c)
    fwd_ret = np.full(n, np.nan)
    ft_long = np.zeros(n, dtype=int)
    ft_short = np.zeros(n, dtype=int)
    race_long = np.zeros(n, dtype=int)
    race_short = np.zeros(n, dtype=int)

    for t in range(n - horizon):
        c0 = c[t]
        up_target = c0 * (1 + k)
        dn_target = c0 * (1 - k)
        fwd_ret[t] = c[t + horizon] / c0 - 1

        long_hit_bar = -1
        short_hit_bar = -1
        for j in range(1, horizon + 1):
            if long_hit_bar < 0 and h[t + j] >= up_target:
                long_hit_bar = j
            if short_hit_bar < 0 and lo[t + j] <= dn_target:
                short_hit_bar = j
            if long_hit_bar > 0 and short_hit_bar > 0:
                break

        ft_long[t] = 1 if long_hit_bar > 0 else 0
        ft_short[t] = 1 if short_hit_bar > 0 else 0

        # path race: who hit first
        if long_hit_bar > 0 and (short_hit_bar < 0 or long_hit_bar < short_hit_bar):
            race_long[t] = 1   # long target hit first → long win
            race_short[t] = -1 # short position would have been stopped out
        elif short_hit_bar > 0 and (long_hit_bar < 0 or short_hit_bar < long_hit_bar):
            race_short[t] = 1
            race_long[t] = -1
        elif long_hit_bar > 0 and short_hit_bar > 0 and long_hit_bar == short_hit_bar:
            # same bar — pessimistic: both stopped
            race_long[t] = -1
            race_short[t] = -1
        # else: neither touched → both stay 0 (no trigger)

    return dict(fwd_ret=fwd_ret, ft_long=ft_long, ft_short=ft_short,
                race_long=race_long, race_short=race_short)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _wr_block(mask_up, mask_dn, long_win, short_win,
              long_loss=None, short_loss=None):
    """Compute WR for one metric. If *_loss arrays given, excludes 'no trigger' bars."""
    if long_loss is None:
        n_up = int(mask_up.sum())
        n_dn = int(mask_dn.sum())
        w_up = int((mask_up & long_win).sum())
        w_dn = int((mask_dn & short_win).sum())
    else:
        # path-race: only count bars where some side triggered
        trig_up = mask_up & (long_win | long_loss)
        trig_dn = mask_dn & (short_win | short_loss)
        n_up = int(trig_up.sum())
        n_dn = int(trig_dn.sum())
        w_up = int((trig_up & long_win).sum())
        w_dn = int((trig_dn & short_win).sum())
    n = n_up + n_dn
    w = w_up + w_dn
    if n == 0:
        return 0, 0, 0.0, 0.0, 0.0
    wr = w / n
    lo, hi = wilson_ci(w, n)
    return n, w, wr, lo, hi


def tier_wr_table(tier_arr, dir_arr, path, label):
    print(f"\n  {label}")
    print(f"    {'tier':<10}{'metric':<18}{'n':>6}{'WR':>10}{'CI':>22}")
    ft_long_win = path["ft_long"].astype(bool)
    ft_short_win = path["ft_short"].astype(bool)
    race_long_win = path["race_long"] == 1
    race_long_loss = path["race_long"] == -1
    race_short_win = path["race_short"] == 1
    race_short_loss = path["race_short"] == -1
    fwd = path["fwd_ret"]

    for tier in ["Strong", "Moderate", "Weak"]:
        mask = tier_arr == tier
        up = mask & (dir_arr == "UP")
        dn = mask & (dir_arr == "DOWN")

        # 1. 4h closing return
        n, w, wr, lo, hi = _wr_block(up, dn, fwd > 0, fwd < 0)
        if n > 0:
            print(f"    {tier:<10}{'(1) 4h close':<18}{n:>6}{wr*100:>9.1f}%"
                  f"  [{lo*100:5.1f}%, {hi*100:5.1f}%]")

        # 2. first-touch ±0.8%
        n, w, wr, lo, hi = _wr_block(up, dn, ft_long_win, ft_short_win)
        if n > 0:
            print(f"    {'':<10}{'(2) first-touch':<18}{n:>6}{wr*100:>9.1f}%"
                  f"  [{lo*100:5.1f}%, {hi*100:5.1f}%]")

        # 3. path race ±0.8% (only counts triggered bars)
        n, w, wr, lo, hi = _wr_block(up, dn,
                                     race_long_win, race_short_win,
                                     race_long_loss, race_short_loss)
        if n > 0:
            print(f"    {'':<10}{'(3) race +/-0.8%':<18}{n:>6}{wr*100:>9.1f}%"
                  f"  [{lo*100:5.1f}%, {hi*100:5.1f}%]")
        print()


def apply_gate(p_long, p_short, mag_pct, bo_up, bo_dn, thr):
    """Replicates IndicatorEngine._predict_dual gate logic exactly."""
    sl = thr["strong"]
    md = thr["moderate"]
    n = len(p_long)
    direction = np.full(n, "NEUTRAL", dtype=object)
    tier = np.full(n, "Weak", dtype=object)
    for i in range(n):
        long_strong = (
            p_long[i] >= sl["prob_long_threshold"]
            and mag_pct[i] >= sl["mag_percentile_min"]
            and bo_up[i] == 1.0
        )
        short_strong = (
            p_short[i] >= sl["prob_short_threshold"]
            and mag_pct[i] >= sl["mag_percentile_min"]
            and bo_dn[i] == 1.0
        )
        if long_strong and short_strong:
            direction[i] = "UP" if p_long[i] >= p_short[i] else "DOWN"
            tier[i] = "Moderate"
        elif long_strong:
            direction[i] = "UP"; tier[i] = "Strong"
        elif short_strong:
            direction[i] = "DOWN"; tier[i] = "Strong"
        else:
            long_mod = (p_long[i] >= md["prob_long_threshold"]
                        and mag_pct[i] >= md["mag_percentile_min"]
                        and bo_up[i] == 1.0)
            short_mod = (p_short[i] >= md["prob_short_threshold"]
                         and mag_pct[i] >= md["mag_percentile_min"]
                         and bo_dn[i] == 1.0)
            if long_mod and not short_mod:
                direction[i] = "UP"; tier[i] = "Moderate"
            elif short_mod and not long_mod:
                direction[i] = "DOWN"; tier[i] = "Moderate"
            elif long_mod and short_mod:
                direction[i] = "UP" if p_long[i] >= p_short[i] else "DOWN"
                tier[i] = "Moderate"
            else:
                top = max(p_long[i], p_short[i])
                if top < 0.20:
                    direction[i] = "NEUTRAL"
                else:
                    direction[i] = "UP" if p_long[i] >= p_short[i] else "DOWN"
                tier[i] = "Weak"
    return direction, tier


def monthly_strong(df_out, label):
    print(f"\n  {label} — Strong monthly:")
    df_out = df_out.copy()
    df_out["month"] = pd.DatetimeIndex(df_out.index).strftime("%Y-%m")
    strong = df_out[df_out["tier"] == "Strong"]
    for m, g in strong.groupby("month"):
        ups = g[g["dir"] == "UP"]
        dns = g[g["dir"] == "DOWN"]
        wins = ((g["dir"] == "UP") & (g["fwd_ret"] > 0)).sum()
        wins += ((g["dir"] == "DOWN") & (g["fwd_ret"] < 0)).sum()
        n = len(g)
        wr = wins / n if n else 0
        print(f"    {m}: n={n:>3}  WR={wr*100:>5.1f}%  UP={len(ups)} DOWN={len(dns)}")


def main():
    print("=" * 78)
    print("  FULL-SYSTEM WALK-FORWARD VALIDATION")
    print("=" * 78)

    with open(THRESH_PATH) as f:
        thr = json.load(f)
    print(f"  Strong thresholds: long>={thr['strong']['prob_long_threshold']:.3f}  "
          f"short>={thr['strong']['prob_short_threshold']:.3f}  "
          f"mag>={thr['strong']['mag_percentile_min']}  "
          f"breakout={thr['strong']['require_breakout']}")

    # ─────────────────────────────────────────────────────────────
    # View A — IN-SAMPLE (production engine on full history)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  VIEW A — IN-SAMPLE (production engine on full feature history)")
    print("  Models trained on full data → optimistic; tests integration only")
    print("=" * 78)
    df = load_and_cache_data()
    feats = add_initiation_features(df)

    engine = IndicatorEngine()
    out = engine.predict(feats)

    close = feats["close"].astype(float)
    high = feats["high"].astype(float) if "high" in feats.columns else close
    low = feats["low"].astype(float) if "low" in feats.columns else close
    path_full = compute_path_outcomes(close, high, low)

    valid = ~np.isnan(path_full["fwd_ret"])
    out_v = out[valid].copy()
    path_v = {k: v[valid] for k, v in path_full.items()}

    print(f"\n  Bars: {len(out_v)}  "
          f"Distribution: {dict(pd.Series(out_v['strength_score']).value_counts())}")
    tier_wr_table(out_v["strength_score"].values,
                  out_v["pred_direction"].values,
                  path_v, "In-sample WR by tier (3 metrics):")

    df_a = pd.DataFrame({
        "tier": out_v["strength_score"].values,
        "dir":  out_v["pred_direction"].values,
        "fwd_ret": path_v["fwd_ret"],
    }, index=out_v.index)
    monthly_strong(df_a, "VIEW A")

    # ─────────────────────────────────────────────────────────────
    # View B — TRUE OOS (replay walk-forward + production gate)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  VIEW B — TRUE OOS (replay walk-forward p_long/p_short + gate)")
    print("  Models retrained per fold → honest accuracy")
    print("=" * 78)
    oos = pd.read_parquet(OOS_PATH)

    # Pull breakout flags from the same feature universe (V5 close-from-inside)
    feats_idx = feats.reindex(oos.index)
    bo_up = feats_idx["init_bo_up"].fillna(0).values
    bo_dn = feats_idx["init_bo_dn"].fillna(0).values

    direction, tier = apply_gate(
        oos["p_long"].values, oos["p_short"].values,
        oos["mag_pct"].fillna(0.5).values, bo_up, bo_dn, thr,
    )

    # path outcomes aligned to OOS index
    path_oos = {k: pd.Series(v, index=feats.index).reindex(oos.index).values
                for k, v in path_full.items()}
    valid = ~np.isnan(path_oos["fwd_ret"])
    direction_v = direction[valid]
    tier_v = tier[valid]
    path_v2 = {k: v[valid] for k, v in path_oos.items()}

    print(f"\n  Bars: {valid.sum()}  "
          f"Distribution: {dict(pd.Series(tier_v).value_counts())}")
    tier_wr_table(tier_v, direction_v, path_v2, "True OOS WR by tier (3 metrics):")

    df_b = pd.DataFrame({
        "tier": tier_v,
        "dir":  direction_v,
        "fwd_ret": path_v2["fwd_ret"],
    }, index=oos.index[valid])
    monthly_strong(df_b, "VIEW B")

    # ─────────────────────────────────────────────────────────────
    # Summary delta
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print("  View A (in-sample) shows the engine wires up correctly.")
    print("  View B (true OOS)  is the honest production accuracy.")
    print("  If A ≈ B → models generalize.  If A >> B → overfit.")


if __name__ == "__main__":
    main()
