# -*- coding: utf-8 -*-
"""Key-driver confirmation — which raid features actually drive the future
path, measured once, on one universe, with one methodology.

User direction (2026-08-01): confirm WHICH order-flow features are the key
drivers of what happens after a raid FIRST, then evaluate scripts with
those. The four anatomy rounds screened features in separate passes; this
unifies them: same sample, univariate effect sizes with stability checks,
then a multivariate permutation-importance pass (allowed now under the
standing rule — >=2 univariate survivors exist; the meta-labeling ban was
on multivariate fishing BEFORE univariate survivors).

Features (attack window, causal, both symbols):
  pierce      穿越深度 (price behaviour)
  att_min     攻擊分鐘數 (price behaviour)
  reject      收回內側 (1m price path)
  att_vshock  量能倍數 (flow: volume)
  att_taker   追價佔比 (flow: aggression direction)
  absorption  吸收 (flow: aggression per ATR of progress)

Targets:
  A resolution — BREAKOUT vs retested (all raids)
  B netR       — net R when retested (fills only)

Method: univariate = full-sample AUC (A) / Spearman (B) + first/second
half sign agreement. Multivariate = XGBoost, time-ordered 70/30 split,
OOS metric + permutation importance (mean metric drop over 20 shuffles of
one column on the held-out tail). The model is a measuring device, not a
signal — its OOS number is reported so importance is read with the right
amount of trust.

Run: python research/sweep_raid_keydrivers.py
Out: research/results/sweep_raid_keydrivers.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
import sweep_raid_menu2 as M  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/sweep_raid_keydrivers.json"
FEATS = ["pierce", "att_min", "reject_in_hour", "att_vshock", "att_taker",
         "absorption"]
ZH = {"pierce": "穿越深度", "att_min": "攻擊分鐘", "reject_in_hour": "收回內側",
      "att_vshock": "量能倍數", "att_taker": "追價佔比", "absorption": "吸收"}


def auc(y, x):
    """Rank AUC of x for binary y (ties midranked)."""
    y = np.asarray(y)
    x = np.asarray(x, dtype=float)
    r = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort") + 1.0
    # midrank correction for ties
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty_like(r)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main() -> int:
    print("=" * 78)
    print("  KEY DRIVERS — 哪些特徵真正影響獵取後走向（單變量+多變量一次定案）")
    print("=" * 78)
    rows = M.build()
    rows = [r for r in rows
            if all(r.get(f) is not None for f in FEATS)]
    rows.sort(key=lambda r: r["ts"])
    n = len(rows)
    half = n // 2
    res = {}

    # ── univariate table ────────────────────────────────────────────────
    yA = np.array([1 if r["cls"] == "BREAKOUT" else 0 for r in rows])
    fills = [r for r in rows if r["netR"] is not None]
    yB = np.array([r["netR"] for r in fills])
    print(f"\n  [單變量] n={n}（fills={len(fills)}, 突破基準 {100*yA.mean():.0f}%）")
    print(f"  {'特徵':<6}{'解析AUC':>9}{'半1':>7}{'半2':>7} | {'netR-IC':>8}{'半1':>8}{'半2':>8}  判定")
    for f in FEATS:
        xA = np.array([r[f] for r in rows], dtype=float)
        a_full = auc(yA, xA)
        a_h1 = auc(yA[:half], xA[:half])
        a_h2 = auc(yA[half:], xA[half:])
        xB = np.array([r[f] for r in fills], dtype=float)
        halfb = len(fills) // 2
        ic, _ = spearmanr(xB, yB)
        ic1, _ = spearmanr(xB[:halfb], yB[:halfb])
        ic2, _ = spearmanr(xB[halfb:], yB[halfb:])
        stable_a = (a_h1 - 0.5) * (a_h2 - 0.5) > 0 and abs(a_full - 0.5) > 0.03
        stable_b = ic1 * ic2 > 0 and abs(ic) > 0.05
        verdict = ("解析+品質" if stable_a and stable_b
                   else "解析" if stable_a else "品質" if stable_b else "—")
        res[f] = {"auc": round(a_full, 3), "auc_h": [round(a_h1, 3), round(a_h2, 3)],
                  "ic": round(float(ic), 3),
                  "ic_h": [round(float(ic1), 3), round(float(ic2), 3)],
                  "verdict": verdict}
        print(f"  {ZH[f]:<6}{a_full:>9.3f}{a_h1:>7.3f}{a_h2:>7.3f} | "
              f"{ic:>+8.3f}{ic1:>+8.3f}{ic2:>+8.3f}  {verdict}")

    # ── multivariate permutation importance ─────────────────────────────
    from xgboost import XGBClassifier, XGBRegressor
    rng = np.random.default_rng(7)
    cut = int(n * 0.7)
    Xa = np.array([[r[f] for f in FEATS] for r in rows], dtype=float)
    clf = XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.05,
                        subsample=0.9, random_state=7, eval_metric="logloss")
    clf.fit(Xa[:cut], yA[:cut])
    pa = clf.predict_proba(Xa[cut:])[:, 1]
    base_auc = auc(yA[cut:], pa)
    print(f"\n  [多變量A·解析] OOS AUC={base_auc:.3f}（時間切分後30%）· permutation 重要度:")
    imp_a = {}
    for i, f in enumerate(FEATS):
        drops = []
        for _ in range(20):
            Xp = Xa[cut:].copy()
            rng.shuffle(Xp[:, i])
            drops.append(base_auc - auc(yA[cut:], clf.predict_proba(Xp)[:, 1]))
        imp_a[f] = float(np.mean(drops))
    for f, v in sorted(imp_a.items(), key=lambda x: -x[1]):
        print(f"    {ZH[f]:<6} AUC貢獻 {v:+.4f}")
    res["mv_resolution"] = {"oos_auc": round(float(base_auc), 3),
                            "perm": {k: round(v, 4) for k, v in imp_a.items()}}

    fillsX = np.array([[r[f] for f in FEATS] for r in fills], dtype=float)
    cutb = int(len(fills) * 0.7)
    reg = XGBRegressor(max_depth=3, n_estimators=200, learning_rate=0.05,
                       subsample=0.9, random_state=7)
    reg.fit(fillsX[:cutb], yB[:cutb])
    pb = reg.predict(fillsX[cutb:])
    base_ic, _ = spearmanr(pb, yB[cutb:])
    print(f"\n  [多變量B·品質] OOS Spearman IC={base_ic:+.3f} · permutation 重要度:")
    imp_b = {}
    for i, f in enumerate(FEATS):
        drops = []
        for _ in range(20):
            Xp = fillsX[cutb:].copy()
            rng.shuffle(Xp[:, i])
            icp, _ = spearmanr(reg.predict(Xp), yB[cutb:])
            drops.append(base_ic - icp)
        imp_b[f] = float(np.mean(drops))
    for f, v in sorted(imp_b.items(), key=lambda x: -x[1]):
        print(f"    {ZH[f]:<6} IC貢獻 {v:+.4f}")
    res["mv_quality"] = {"oos_ic": round(float(base_ic), 3),
                         "perm": {k: round(v, 4) for k, v in imp_b.items()}}

    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    print("  讀法: 單變量兩半同號才算穩; 多變量 OOS 接近 0.5/0 代表聯合模型")
    print("  沒有超出單變量的資訊——importance 只在 OOS 有肉時才可信。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
