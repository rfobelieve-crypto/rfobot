"""
Phase A: Triple-Barrier label + Meta-Labeling A/B test.

Goal: verify that the new target (triple-barrier, 8h horizon, vol-adaptive)
combined with meta-labeling (two-stage: conviction + direction) produces
higher-quality strong signals than the current binary endpoint label.

No production code is touched. Outputs JSON to
research/results/triple_barrier_phase_a.json.

Compared configs (same feature set, same walk-forward, same purge/embargo):
  A_old_binary     Current prod label: y_dir on close[t+4]/close[t]-1, k=0.5
  B_tb_single      Triple-barrier (8h, k=1.0) → binary UP/DOWN only (drop NEUTRAL in training)
  C_tb_meta        Triple-barrier + two-stage: Stage1 conviction, Stage2 direction

Reported metrics:
  - Label distribution
  - Stage-level AUC / accuracy
  - Strong-signal selectivity: at threshold 0.70, how many signals, what's the
    barrier-hit precision (UP prediction truly reached upper barrier etc)
  - Endpoint IC (so we can compare to current 4h world)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.dual_model.shared_data import load_and_cache_data, walk_forward_splits
from research.dual_model.build_direction_labels import build_direction_labels

ABLATION = Path("research/results/ablation_study.json")
OUT = Path("research/results/triple_barrier_phase_a.json")

DIR_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
    "verbosity": 0,
}

HORIZON_BARS = 8  # 8-hour time barrier
K_BARRIER = 1.0   # barrier = k * realized_vol_1h * sqrt(horizon)


def triple_barrier_labels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    vol_1h: np.ndarray,
    horizon: int = HORIZON_BARS,
    k: float = K_BARRIER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Triple-barrier label with vol-adaptive barriers.

    barrier_pct[i] = k * vol_1h[i] * sqrt(horizon)
        vol_1h = std of 1h log returns over a trailing window (already computed)

    For bar i, simulate [i+1, i+horizon]:
      - if bar's high reaches p0*(1+bp) first  → UP (1)
      - if bar's low  reaches p0*(1-bp) first  → DOWN (0)
      - both in same bar → use the one on the open→close direction (ambiguous → NEUTRAL)
      - time barrier reached → NEUTRAL (-1)

    Returns
    -------
    y_tri      : int8 array, 1 / 0 / -1 / nan
    y_conv     : float, 1 if decisive (UP or DOWN), 0 if NEUTRAL, nan if unlabelled
    y_dir      : float, 1 if UP, 0 if DOWN, nan if NEUTRAL or unlabelled
    touch_bars : number of bars until first touch (horizon if NEUTRAL)
    """
    n = len(close)
    y_tri = np.full(n, np.nan)
    y_conv = np.full(n, np.nan)
    y_dir = np.full(n, np.nan)
    touch_bars = np.full(n, np.nan)

    for i in range(n - horizon):
        p0 = close[i]
        bp = k * vol_1h[i] * np.sqrt(horizon) if not np.isnan(vol_1h[i]) else np.nan
        if np.isnan(bp) or bp <= 0 or p0 <= 0:
            continue
        up_th = p0 * (1 + bp)
        dn_th = p0 * (1 - bp)

        touched = False
        for j in range(1, horizon + 1):
            hi = high[i + j]
            lo = low[i + j]
            hit_up = hi >= up_th
            hit_dn = lo <= dn_th
            if hit_up and hit_dn:
                # Same-bar both touched → ambiguous; mark NEUTRAL
                y_tri[i] = -1.0
                y_conv[i] = 0.0
                touch_bars[i] = j
                touched = True
                break
            if hit_up:
                y_tri[i] = 1.0
                y_conv[i] = 1.0
                y_dir[i] = 1.0
                touch_bars[i] = j
                touched = True
                break
            if hit_dn:
                y_tri[i] = 0.0
                y_conv[i] = 1.0
                y_dir[i] = 0.0
                touch_bars[i] = j
                touched = True
                break
        if not touched:
            y_tri[i] = -1.0
            y_conv[i] = 0.0
            touch_bars[i] = horizon

    return y_tri, y_conv, y_dir, touch_bars


def get_feats(df: pd.DataFrame) -> list[str]:
    abl = json.loads(ABLATION.read_text())
    base = abl.get("baseline", {}).get("features", [])
    keep = abl.get("combined", {}).get("keep_features", [])
    seen: set[str] = set()
    out = []
    for f in base + keep:
        if f in df.columns and f not in seen:
            out.append(f)
            seen.add(f)
    return out


def make_weights(df: pd.DataFrame) -> np.ndarray:
    """Production-matched regime weighting."""
    log_ret = np.log(df["close"] / df["close"].shift(1))
    ret_24h = df["close"].pct_change(24)
    vol_24h = log_ret.rolling(24).std()
    vol_pct = vol_24h.expanding(min_periods=168).rank(pct=True)
    w = np.ones(len(df))
    bull = ((vol_pct > 0.6) & (ret_24h > 0.005)).fillna(False).values
    bear = ((vol_pct > 0.6) & (ret_24h < -0.005)).fillna(False).values
    w[bull] = 4.0
    w[bear] = 2.0
    return w


def wf_binary(df: pd.DataFrame, feats: list[str], y: np.ndarray,
              weights: np.ndarray, end_ret: np.ndarray,
              purge: int = 8, embargo: int = 8) -> dict:
    """Walk-forward binary classification; returns probs, labels, ends."""
    splits = walk_forward_splits(
        len(df), initial_train=288, test_size=48, step=48,
        purge=purge, embargo=embargo,
    )
    probs, labels, ends, test_idxs = [], [], [], []

    for tr_idx, te_idx in splits:
        y_tr_full = y[tr_idx]
        y_te_full = y[te_idx]
        tr_mask = ~np.isnan(y_tr_full)
        te_mask = ~np.isnan(y_te_full)
        if tr_mask.sum() < 50 or te_mask.sum() < 5:
            continue

        X_tr = df.iloc[tr_idx][feats].fillna(0).values[tr_mask]
        y_tr = y_tr_full[tr_mask].astype(int)
        X_te = df.iloc[te_idx][feats].fillna(0).values[te_mask]
        y_te = y_te_full[te_mask].astype(int)
        sw = weights[tr_idx][tr_mask]

        if len(np.unique(y_tr)) < 2:
            continue

        up_rate = y_tr.mean()
        p = DIR_PARAMS.copy()
        p["scale_pos_weight"] = (1 - up_rate) / up_rate if 0 < up_rate < 1 else 1.0
        m = xgb.XGBClassifier(**p)
        m.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
        pr = m.predict_proba(X_te)[:, 1]

        te_global = np.array(te_idx)[te_mask]
        probs.extend(pr.tolist())
        labels.extend(y_te.tolist())
        ends.extend(end_ret[te_global].tolist())
        test_idxs.extend(te_global.tolist())

    return {
        "probs": np.array(probs),
        "labels": np.array(labels),
        "ends": np.array(ends),
        "test_idxs": np.array(test_idxs),
    }


def summarize_binary(tag: str, out: dict) -> dict:
    probs = out["probs"]
    labels = out["labels"]
    ends = out["ends"]
    auc = float(roc_auc_score(labels, probs))
    acc = float(((probs >= 0.5).astype(int) == labels).mean())
    ic_end, _ = spearmanr(probs, ends)

    strong_up = probs >= 0.70
    strong_dn = probs <= 0.30
    strong_n = int(strong_up.sum() + strong_dn.sum())
    wins = int((ends[strong_up] > 0).sum() + (ends[strong_dn] < 0).sum())
    wr = wins / strong_n if strong_n > 0 else 0.0

    return {
        "tag": tag,
        "n_oos": int(len(probs)),
        "auc": auc,
        "acc": acc,
        "ic_endpoint": float(ic_end),
        "strong_n": strong_n,
        "strong_winrate": float(wr),
    }


def run() -> None:
    print("Loading data...")
    df = load_and_cache_data(limit=4000, force_refresh=False, max_stale_hours=12.0)
    feats = get_feats(df)
    print(f"Features: {len(feats)}")
    print(f"Rows: {len(df)}")

    weights = make_weights(df)

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    vol_1h = log_ret.rolling(20, min_periods=10).std().values.astype(float)

    # Old label (current prod)
    lab_old = build_direction_labels(df, k=0.5)
    y_old = lab_old["y_dir"].values.astype(float)
    end_ret_4h = lab_old["return_4h"].values.astype(float)

    # Triple-barrier label
    print(f"\nBuilding triple-barrier: horizon={HORIZON_BARS}h, k={K_BARRIER}, vol-adaptive")
    y_tri, y_conv, y_dir_tb, touch = triple_barrier_labels(
        close, high, low, vol_1h, horizon=HORIZON_BARS, k=K_BARRIER,
    )

    # Label distribution
    n_up_tb = int(np.nansum(y_tri == 1))
    n_dn_tb = int(np.nansum(y_tri == 0))
    n_nt_tb = int(np.nansum(y_tri == -1))
    n_valid = n_up_tb + n_dn_tb + n_nt_tb
    print(f"\n=== Triple-barrier distribution ===")
    print(f"  UP     : {n_up_tb} ({100*n_up_tb/n_valid:.1f}%)")
    print(f"  DOWN   : {n_dn_tb} ({100*n_dn_tb/n_valid:.1f}%)")
    print(f"  NEUTRAL: {n_nt_tb} ({100*n_nt_tb/n_valid:.1f}%)")
    avg_touch = float(np.nanmean(touch[y_conv == 1]))
    print(f"  avg bars-to-touch (decisive): {avg_touch:.2f}")

    # 8h endpoint return for IC comparison against new target
    end_ret_8h = pd.Series(close).pct_change(8).shift(-8).values

    # ── Config A: old binary label ────────────────────────────────
    print("\n[A] old binary label (k=0.5, 4h endpoint)")
    out_a = wf_binary(df, feats, y_old, weights, end_ret_4h, purge=4, embargo=4)
    sum_a = summarize_binary("A_old_binary", out_a)
    print(f"  AUC={sum_a['auc']:.4f} Acc={sum_a['acc']:.1%} IC_end={sum_a['ic_endpoint']:+.4f} "
          f"Strong n={sum_a['strong_n']} wr={sum_a['strong_winrate']:.1%}")

    # ── Config B: triple-barrier UP/DOWN only, single binary ──────
    print("\n[B] triple-barrier, single binary (drop NEUTRAL from training)")
    y_tb_bin = y_dir_tb.copy()  # already NaN for neutral
    out_b = wf_binary(df, feats, y_tb_bin, weights, end_ret_8h, purge=8, embargo=8)
    sum_b = summarize_binary("B_tb_single_binary", out_b)
    print(f"  AUC={sum_b['auc']:.4f} Acc={sum_b['acc']:.1%} IC_end={sum_b['ic_endpoint']:+.4f} "
          f"Strong n={sum_b['strong_n']} wr={sum_b['strong_winrate']:.1%}")

    # ── Config C: meta-labeling two stages ────────────────────────
    print("\n[C] meta-labeling: Stage1 conviction + Stage2 direction")

    # Stage 1 — conviction (everyone has a label: 0 or 1, no NaN except trailing horizon)
    y_stage1 = y_conv.copy()
    print("  Stage 1 training (conviction: decisive vs neutral)")
    out_s1 = wf_binary(df, feats, y_stage1, weights, end_ret_8h, purge=8, embargo=8)
    sum_s1 = summarize_binary("C_stage1_conviction", out_s1)
    print(f"    AUC={sum_s1['auc']:.4f} Acc={sum_s1['acc']:.1%}")

    # Stage 2 — direction (trained only on decisive samples; NEUTRAL → NaN)
    print("  Stage 2 training (direction | decisive)")
    out_s2 = wf_binary(df, feats, y_dir_tb, weights, end_ret_8h, purge=8, embargo=8)
    sum_s2 = summarize_binary("C_stage2_direction", out_s2)
    print(f"    AUC={sum_s2['auc']:.4f} Acc={sum_s2['acc']:.1%}")

    # Combined confidence on shared OOS indices
    s1_map = dict(zip(out_s1["test_idxs"], out_s1["probs"]))
    s2_map = dict(zip(out_s2["test_idxs"], out_s2["probs"]))
    shared = sorted(set(s1_map.keys()) & set(s2_map.keys()))

    p_conv = np.array([s1_map[i] for i in shared])
    p_dir = np.array([s2_map[i] for i in shared])
    # Combined confidence: both stages must agree
    # conviction_score = p_conv * |p_dir - 0.5| * 2
    conv_score = p_conv * np.abs(p_dir - 0.5) * 2
    # Predicted direction
    pred_up = p_dir >= 0.5

    # Actual 8h endpoint return
    shared_idx = np.array(shared)
    actual_end_8h = end_ret_8h[shared_idx]
    # Actual barrier outcome
    actual_tb = y_tri[shared_idx]  # 1 / 0 / -1

    # Strong signal: combined confidence > 0.40 (equivalent threshold to 0.70 single-model)
    strong_mask = conv_score >= 0.40
    strong_n = int(strong_mask.sum())

    # Hit rate by barrier outcome (the "realistic" success metric)
    if strong_n > 0:
        pred_strong = pred_up[strong_mask]
        actual_strong = actual_tb[strong_mask]
        # Correct: predicted UP and actual = UP (1), or predicted DOWN and actual = DOWN (0)
        correct_dir = (
            (pred_strong & (actual_strong == 1)) |
            (~pred_strong & (actual_strong == 0))
        )
        wrong_dir = (
            (pred_strong & (actual_strong == 0)) |
            (~pred_strong & (actual_strong == 1))
        )
        neutral_out = (actual_strong == -1)
        hit_rate_tb = float(correct_dir.sum()) / strong_n
        miss_rate_tb = float(wrong_dir.sum()) / strong_n
        neutral_rate_tb = float(neutral_out.sum()) / strong_n

        # Also report endpoint winrate (for apples-to-apples vs old)
        ends_strong = actual_end_8h[strong_mask]
        wins_end = (
            (pred_strong & (ends_strong > 0)).sum() +
            (~pred_strong & (ends_strong < 0)).sum()
        )
        wr_end = float(wins_end) / strong_n
    else:
        hit_rate_tb = miss_rate_tb = neutral_rate_tb = wr_end = 0.0

    print(f"  Combined (n={len(shared)}): strong_n={strong_n} "
          f"barrier_hit={hit_rate_tb:.1%} barrier_miss={miss_rate_tb:.1%} "
          f"neutral={neutral_rate_tb:.1%} end8h_wr={wr_end:.1%}")

    results = {
        "config": {
            "horizon_bars": HORIZON_BARS,
            "k_barrier": K_BARRIER,
            "vol_adaptive": True,
            "n_features": len(feats),
            "n_rows": len(df),
        },
        "label_distribution": {
            "up_pct": 100 * n_up_tb / n_valid,
            "down_pct": 100 * n_dn_tb / n_valid,
            "neutral_pct": 100 * n_nt_tb / n_valid,
            "avg_bars_to_touch": avg_touch,
        },
        "A_old_binary": sum_a,
        "B_tb_single_binary": sum_b,
        "C_stage1_conviction": sum_s1,
        "C_stage2_direction": sum_s2,
        "C_combined_strong": {
            "strong_n": strong_n,
            "barrier_hit_rate": hit_rate_tb,
            "barrier_miss_rate": miss_rate_tb,
            "neutral_rate": neutral_rate_tb,
            "endpoint_8h_winrate": wr_end,
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT (should we proceed to Phase B?)")
    print("=" * 70)
    old_strong_wr = sum_a["strong_winrate"]
    new_strong_hit = hit_rate_tb
    delta = new_strong_hit - old_strong_wr
    print(f"  Old binary strong winrate (4h endpoint):       {old_strong_wr:.1%}")
    print(f"  New meta-label strong barrier-hit (8h):        {new_strong_hit:.1%}")
    print(f"  New meta-label strong 8h endpoint winrate:     {wr_end:.1%}")
    print(f"  Delta (barrier-hit vs old endpoint):           {delta:+.1%}")
    if delta >= 0.03 and strong_n >= 20:
        print("  → PROCEED to Phase B (new target meaningfully stronger)")
    elif delta >= 0.0 and strong_n >= 20:
        print("  → MARGINAL. Consider sweeping k or horizon before Phase B.")
    else:
        print("  → DO NOT proceed. Revert to old target.")


if __name__ == "__main__":
    run()
