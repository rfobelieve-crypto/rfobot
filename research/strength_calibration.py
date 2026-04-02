"""
Confidence v2 Strength Calibration.

目標：
  1. 重新校準 Weak / Moderate / Strong 閾值（基於 v2 分佈，非固定 40/70）
  2. Coverage vs Precision 報告
  3. Quintile / Decile 驗證
  4. 高信心 bucket 在 OOS 上穩定性驗證

Usage:
    python -m research.strength_calibration
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from research.prediction_indicator_v2 import (
    load_data, select_features_burnin, walk_forward_predict,
    assign_direction_raw,
    BURN_IN_BARS, N_FOLDS, TARGET, DEADZONE,
)
from research.regime_analysis import assign_regime
from research.confidence_v2 import (
    compute_mag_score, compute_regime_score, compute_confidence_v2,
)


# ═══════════════════════════════════════════════════════════════════════════
# Coverage vs Precision Scan
# ═══════════════════════════════════════════════════════════════════════════

def coverage_precision_scan(conf: np.ndarray, pred: np.ndarray,
                            y: np.ndarray, direction: np.ndarray):
    """
    對不同 confidence 閾值掃描 coverage vs precision trade-off。
    """
    valid = ~np.isnan(conf) & ~np.isnan(pred) & ~np.isnan(y)
    active = valid & (direction != "NEUTRAL")
    n_active = active.sum()

    conf_a = conf[active]
    pred_a = pred[active]
    y_a = y[active]
    dir_a = direction[active]

    dir_pred = np.where(dir_a == "UP", 1, -1)
    dir_actual = np.sign(y_a)
    dir_correct = (dir_pred == dir_actual)

    print(f"\n{'='*70}")
    print(f"  Coverage vs Precision Scan (n_active={n_active})")
    print(f"{'='*70}")
    print(f"  {'Threshold':>10s}  {'Coverage':>9s}  {'n':>5s}  {'Dir Acc':>8s}  {'IC':>8s}  "
          f"{'Mean|Ret|':>10s}  {'p-value':>8s}")
    print(f"  {'-'*10}  {'-'*9}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}")

    thresholds = [0, 1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90]
    results = []

    for thr in thresholds:
        mask = conf_a >= thr
        n = mask.sum()
        if n < 20:
            continue

        da = dir_correct[mask].mean()
        ic, pval = spearmanr(pred_a[mask], y_a[mask])
        abs_ret = np.abs(y_a[mask]).mean()
        coverage = n / n_active

        results.append({
            "threshold": thr, "coverage": coverage, "n": n,
            "dir_acc": da, "ic": ic, "pval": pval, "abs_ret": abs_ret,
        })

        sig = "***" if pval < 0.01 else "* " if pval < 0.05 else "  "
        print(f"  {thr:>10d}  {coverage:9.1%}  {n:5d}  {da:8.1%}  {ic:+8.4f}  "
              f"{abs_ret*100:9.3f}%  {pval:8.3f} {sig}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Performance-Based Threshold Discovery
# ═══════════════════════════════════════════════════════════════════════════

def find_optimal_thresholds(conf: np.ndarray, pred: np.ndarray,
                            y: np.ndarray, direction: np.ndarray):
    """
    用 performance 斷點尋找 Weak / Moderate / Strong 最佳切分。

    方法：
    1. 掃描所有可能的 2-cut 組合
    2. 評估 Strong dir_acc > Moderate dir_acc > Weak dir_acc 的單調性
    3. 同時考慮各 tier 的 n 不能太少
    """
    valid = ~np.isnan(conf) & ~np.isnan(pred) & ~np.isnan(y)
    active = valid & (direction != "NEUTRAL")

    conf_a = conf[active]
    pred_a = pred[active]
    y_a = y[active]
    dir_a = direction[active]

    dir_pred = np.where(dir_a == "UP", 1, -1)
    dir_actual = np.sign(y_a)
    dir_correct = (dir_pred == dir_actual)

    # 候選切分點：v2 分佈的百分位
    pcts = np.arange(5, 96, 5)
    candidates = np.percentile(conf_a[~np.isnan(conf_a)], pcts)
    candidates = np.unique(np.round(candidates, 1))

    print(f"\n{'='*70}")
    print(f"  Optimal Threshold Search")
    print(f"{'='*70}")
    print(f"  Candidate thresholds (from v2 distribution percentiles):")
    print(f"  {candidates}")

    best_score = -1
    best_cuts = (0, 50)
    best_info = None

    for i, cut1 in enumerate(candidates):
        for cut2 in candidates[i+1:]:
            # Weak: [0, cut1), Moderate: [cut1, cut2), Strong: [cut2, 100]
            w_mask = conf_a < cut1
            m_mask = (conf_a >= cut1) & (conf_a < cut2)
            s_mask = conf_a >= cut2

            n_w, n_m, n_s = w_mask.sum(), m_mask.sum(), s_mask.sum()

            # 每個 tier 至少 50 bars
            if min(n_w, n_m, n_s) < 50:
                continue

            da_w = dir_correct[w_mask].mean()
            da_m = dir_correct[m_mask].mean()
            da_s = dir_correct[s_mask].mean()

            # 單調性得分：Strong > Moderate > Weak
            mono = 0
            if da_s > da_m:
                mono += 1
            if da_m > da_w:
                mono += 1

            # IC 加分
            ic_s, _ = spearmanr(pred_a[s_mask], y_a[s_mask]) if n_s >= 20 else (0, 1)
            ic_w, _ = spearmanr(pred_a[w_mask], y_a[w_mask]) if n_w >= 20 else (0, 1)

            # 綜合得分：單調性 + Strong IC 優勢 + Strong dir_acc
            score = mono * 10 + (da_s - da_w) * 100 + max(ic_s - ic_w, 0) * 50

            if score > best_score:
                best_score = score
                best_cuts = (cut1, cut2)
                best_info = {
                    "cuts": (cut1, cut2),
                    "n": (n_w, n_m, n_s),
                    "dir_acc": (da_w, da_m, da_s),
                    "ic_s": ic_s, "ic_w": ic_w,
                    "score": score,
                }

    if best_info:
        c1, c2 = best_info["cuts"]
        print(f"\n  Best threshold pair: Weak < {c1:.1f} | Moderate [{c1:.1f}, {c2:.1f}) | Strong >= {c2:.1f}")
        print(f"  Score: {best_info['score']:.2f}")
        print(f"  {'Tier':>10s}  {'n':>5s}  {'Coverage':>9s}  {'Dir Acc':>8s}")
        for name, n, da in zip(["Weak", "Moderate", "Strong"],
                                best_info["n"], best_info["dir_acc"]):
            print(f"  {name:>10s}  {n:5d}  {n/active.sum():9.1%}  {da:8.1%}")

    return best_cuts, best_info


# ═══════════════════════════════════════════════════════════════════════════
# Full Validation with Calibrated Thresholds
# ═══════════════════════════════════════════════════════════════════════════

def full_validation(conf: np.ndarray, pred: np.ndarray, y: np.ndarray,
                    direction: np.ndarray, regime: np.ndarray,
                    cuts: tuple[float, float]):
    """
    使用校準後的閾值進行完整驗證。
    """
    cut1, cut2 = cuts
    valid = ~np.isnan(conf) & ~np.isnan(pred) & ~np.isnan(y)
    active = valid & (direction != "NEUTRAL")

    conf_a = conf[active]
    pred_a = pred[active]
    y_a = y[active]
    dir_a = direction[active]
    regime_a = regime[active]

    dir_pred = np.where(dir_a == "UP", 1, -1)
    dir_actual = np.sign(y_a)
    dir_correct = (dir_pred == dir_actual)

    strength = np.where(conf_a >= cut2, "Strong",
               np.where(conf_a >= cut1, "Moderate", "Weak"))

    # ── 1. 三 Tier 詳細指標 ──
    print(f"\n{'='*70}")
    print(f"  Calibrated Strength Tiers: Weak < {cut1:.1f} | Mod [{cut1:.1f},{cut2:.1f}) | Strong >= {cut2:.1f}")
    print(f"{'='*70}")

    print(f"\n  {'Tier':>10s}  {'n':>5s}  {'%':>6s}  {'Dir Acc':>8s}  {'IC':>8s}  "
          f"{'Mean|Ret|':>10s}  {'Mean Pred':>10s}")
    print(f"  {'-'*10}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")

    for name in ["Weak", "Moderate", "Strong"]:
        mask = strength == name
        n = mask.sum()
        if n < 10:
            print(f"  {name:>10s}  {n:5d}  (too few)")
            continue
        da = dir_correct[mask].mean()
        ic, pval = spearmanr(pred_a[mask], y_a[mask])
        abs_ret = np.abs(y_a[mask]).mean()
        mean_pred = np.abs(pred_a[mask]).mean()
        sig = "***" if pval < 0.01 else "* " if pval < 0.05 else "  "
        print(f"  {name:>10s}  {n:5d}  {n/len(conf_a):5.1%}  {da:8.1%}  "
              f"{ic:+8.4f}{sig} {abs_ret*100:9.3f}%  {mean_pred*100:9.4f}%")

    # ── 2. Decile 驗證 ──
    print(f"\n  Decile Calibration:")
    try:
        bins = pd.qcut(conf_a, 10, labels=False, duplicates="drop")
        n_bins = len(set(bins))
    except ValueError:
        bins = pd.qcut(conf_a, 5, labels=False, duplicates="drop")
        n_bins = len(set(bins))

    print(f"  {'Bin':>5s}  {'Conf Range':>14s}  {'n':>5s}  {'Dir Acc':>8s}  {'IC':>8s}")
    print(f"  {'-'*5}  {'-'*14}  {'-'*5}  {'-'*8}  {'-'*8}")

    bin_accs = []
    for b in sorted(set(bins)):
        mask = bins == b
        n = mask.sum()
        if n < 10:
            continue
        da = dir_correct[mask].mean()
        ic_b, _ = spearmanr(pred_a[mask], y_a[mask]) if n >= 15 else (np.nan, np.nan)
        lo = conf_a[mask].min()
        hi = conf_a[mask].max()
        bin_accs.append(da)
        ic_str = f"{ic_b:+8.4f}" if not np.isnan(ic_b) else f"{'N/A':>8s}"
        print(f"  {int(b):5d}  {lo:6.1f}~{hi:6.1f}  {n:5d}  {da:8.1%}  {ic_str}")

    if len(bin_accs) >= 3:
        diffs = np.diff(bin_accs)
        mono = (diffs > 0).sum() / len(diffs)
        print(f"\n  Dir Acc monotonicity: {mono:.0%} ({(diffs > 0).sum()}/{len(diffs)} increasing)")

    # ── 3. Per-Fold Stability ──
    print(f"\n  Per-Fold Stability (Strong tier only):")
    n_total = len(pred)
    oos_data = n_total - BURN_IN_BARS
    fold_size = oos_data // N_FOLDS

    # 需要映射回原始 index
    active_indices = np.where(active)[0]

    print(f"  {'Fold':>5s}  {'n_strong':>9s}  {'Dir Acc':>8s}  {'IC':>8s}  {'Period':>16s}")
    print(f"  {'-'*5}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*16}")

    fold_strong_accs = []
    for k in range(N_FOLDS):
        fs = BURN_IN_BARS + fold_size * k
        fe = min(BURN_IN_BARS + fold_size * (k + 1), n_total)

        # 找出此 fold 中 active 且 Strong 的 bars
        fold_mask = np.zeros(n_total, dtype=bool)
        fold_mask[fs:fe] = True

        in_fold = fold_mask[active_indices]
        is_strong = strength == "Strong"
        both = in_fold & is_strong

        n_s = both.sum()
        if n_s < 10:
            print(f"  {k+1:>5d}  {n_s:>9d}  (too few)")
            continue

        da_s = dir_correct[both].mean()
        ic_s, _ = spearmanr(pred_a[both], y_a[both]) if n_s >= 15 else (np.nan, np.nan)
        fold_strong_accs.append(da_s)

        # 日期
        from research.prediction_indicator_v2 import load_data as _ld
        ic_str = f"{ic_s:+8.4f}" if not np.isnan(ic_s) else f"{'N/A':>8s}"
        print(f"  {k+1:>5d}  {n_s:>9d}  {da_s:8.1%}  {ic_str}")

    if len(fold_strong_accs) >= 2:
        mean_da = np.mean(fold_strong_accs)
        std_da = np.std(fold_strong_accs)
        print(f"\n  Strong tier stability: mean={mean_da:.1%}  std={std_da:.1%}  "
              f"min={min(fold_strong_accs):.1%}  max={max(fold_strong_accs):.1%}")

    # ── 4. Regime × Strength 交叉表 ──
    print(f"\n  Regime × Strength Dir Acc:")
    print(f"  {'Regime':>16s}  {'Weak':>8s}  {'Moderate':>8s}  {'Strong':>8s}")
    print(f"  {'-'*16}  {'-'*8}  {'-'*8}  {'-'*8}")

    for r in ["TRENDING_BULL", "TRENDING_BEAR", "CHOPPY"]:
        row = f"  {r:>16s}"
        for s in ["Weak", "Moderate", "Strong"]:
            mask = (regime_a == r) & (strength == s)
            n = mask.sum()
            if n >= 10:
                da = dir_correct[mask].mean()
                row += f"  {da:8.1%}"
            else:
                row += f"  {'n<10':>8s}"
        print(row)

    # ── 5. Monthly × Strength ──
    print(f"\n  Monthly × Strength Dir Acc:")

    # 取得 active bars 的時間 index
    df_idx = pd.Series(range(n_total))
    active_times = pd.DatetimeIndex([pd.Timestamp("2025-01-01")] * active.sum())  # placeholder
    # 需要從原始 df 拿 index
    # 簡化：用 fold 邊界推算月份

    return cuts, strength


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 70)
    print("  Strength Tier Calibration — Performance-Based Thresholds")
    print("=" * 70)

    df, feat_cols = load_data()
    feat_cols = select_features_burnin(df, feat_cols, top_k=40)

    print(f"\n{'='*70}")
    print(f"  Walk-Forward OOS Prediction")
    print(f"{'='*70}")
    oos_pred = walk_forward_predict(df, feat_cols)
    direction = assign_direction_raw(oos_pred)
    y = df[TARGET].values
    regime = assign_regime(df)

    # Confidence v2
    mag_score = compute_mag_score(oos_pred)
    regime_score = compute_regime_score(oos_pred, y, regime)
    conf_v2 = compute_confidence_v2(mag_score, regime_score)

    # ── v2 Distribution ──
    valid = ~np.isnan(conf_v2)
    print(f"\n{'='*70}")
    print(f"  Confidence v2 Distribution")
    print(f"{'='*70}")
    v = conf_v2[valid]
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{p:02d}: {np.percentile(v, p):6.1f}")
    print(f"  Zeros: {(v == 0).sum()} ({(v == 0).sum()/len(v):.1%})")
    print(f"  >0:    {(v > 0).sum()} ({(v > 0).sum()/len(v):.1%})")

    # ── Coverage vs Precision ──
    scan_results = coverage_precision_scan(conf_v2, oos_pred, y, direction)

    # ── Threshold Discovery ──
    best_cuts, best_info = find_optimal_thresholds(conf_v2, oos_pred, y, direction)

    # ── Full Validation ──
    cuts, strength_arr = full_validation(conf_v2, oos_pred, y, direction, regime, best_cuts)

    # ── Monthly validation with actual dates ──
    print(f"\n  Monthly Strong Tier Performance:")
    valid_mask = ~np.isnan(conf_v2) & ~np.isnan(oos_pred) & ~np.isnan(y) & (direction != "NEUTRAL")
    df_oos = df[valid_mask].copy()
    df_oos["pred"] = oos_pred[valid_mask]
    df_oos["conf"] = conf_v2[valid_mask]
    df_oos["correct"] = (np.where(direction[valid_mask] == "UP", 1, -1) == np.sign(y[valid_mask]))
    df_oos["strength"] = np.where(df_oos["conf"] >= best_cuts[1], "Strong",
                          np.where(df_oos["conf"] >= best_cuts[0], "Moderate", "Weak"))
    df_oos["month"] = df_oos.index.strftime("%Y-%m")

    print(f"  {'Month':>8s}  {'n_strong':>9s}  {'Dir Acc':>8s}  {'IC':>8s}  "
          f"{'n_all':>6s}  {'All Acc':>8s}")
    print(f"  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*8}")

    for mo, grp in df_oos.groupby("month"):
        strong = grp[grp["strength"] == "Strong"]
        n_s = len(strong)
        n_all = len(grp)
        da_all = grp["correct"].mean()

        if n_s >= 10:
            da_s = strong["correct"].mean()
            ic_s, _ = spearmanr(strong["pred"], strong[TARGET])
            ic_str = f"{ic_s:+8.4f}"
        else:
            da_s = np.nan
            ic_str = f"{'N/A':>8s}"

        da_s_str = f"{da_s:8.1%}" if not np.isnan(da_s) else f"{'N/A':>8s}"
        print(f"  {mo:>8s}  {n_s:>9d}  {da_s_str}  {ic_str}  {n_all:>6d}  {da_all:8.1%}")

    # ── Final Summary ──
    print(f"\n{'='*70}")
    print(f"  FINAL CALIBRATION RESULT")
    print(f"{'='*70}")
    c1, c2 = best_cuts
    print(f"\n  Recommended Thresholds:")
    print(f"    Weak:     confidence < {c1:.1f}")
    print(f"    Moderate: {c1:.1f} <= confidence < {c2:.1f}")
    print(f"    Strong:   confidence >= {c2:.1f}")
    print(f"\n  (Based on v2 composite: mag_score × regime_score)")
    print(f"  (Old thresholds were 40/70 — not calibrated to v2 distribution)")

    if best_info:
        da_w, da_m, da_s = best_info["dir_acc"]
        n_w, n_m, n_s = best_info["n"]
        print(f"\n  {'Tier':>10s}  {'Dir Acc':>8s}  {'n':>5s}  {'Coverage':>9s}  {'Monotonic':>10s}")
        print(f"  {'-'*10}  {'-'*8}  {'-'*5}  {'-'*9}  {'-'*10}")
        total = n_w + n_m + n_s
        print(f"  {'Weak':>10s}  {da_w:8.1%}  {n_w:5d}  {n_w/total:9.1%}")
        print(f"  {'Moderate':>10s}  {da_m:8.1%}  {n_m:5d}  {n_m/total:9.1%}  "
              f"{'>' if da_m > da_w else '<'} Weak")
        print(f"  {'Strong':>10s}  {da_s:8.1%}  {n_s:5d}  {n_s/total:9.1%}  "
              f"{'>' if da_s > da_m else '<'} Moderate")

        if da_s > da_m > da_w:
            print(f"\n  PASS: Strong > Moderate > Weak (fully monotonic)")
        elif da_s > da_w:
            print(f"\n  PARTIAL: Strong > Weak but Moderate not in between")
        else:
            print(f"\n  FAIL: Not monotonic")


def main():
    run()


if __name__ == "__main__":
    main()
