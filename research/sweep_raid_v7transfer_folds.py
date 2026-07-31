# -*- coding: utf-8 -*-
"""V7-transfer fold gauntlet — the promotion test for the single-split lift.

The single 70/30 result (Δ AUC +0.047 / Δ IC +0.032) is exactly the kind
of number WQ101 taught us not to trust: aggregate, one split, and feature
selection done on the FULL sample (selection leak). This runs the honest
version:

  5 sequential test blocks (expanding train). PER FOLD: screen the 276
  V7 features on TRAIN ONLY (same stability rules), de-correlate, cap 5,
  fit keys-only vs keys+selected, score the untouched test block.
  Verdict needs the standing 4 conditions: per-fold mean Δ > 0, >=3/5
  folds positive, bootstrap CI of fold Δs not spanning deep negative,
  and the aggregate direction agreeing.

Run: python research/sweep_raid_v7transfer_folds.py
Out: research/results/sweep_raid_v7transfer_folds.json
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
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
import sweep_raid_menu2 as M  # noqa: E402
from sweep_raid_keydrivers import auc  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/sweep_raid_v7transfer_folds.json"
FEATS_PQ = ROOT / "research/dual_model/.cache/features_all.parquet"
RAW = {"open", "high", "low", "close", "volume", "taker_buy_vol"}
KEYS = ["pierce", "att_min", "reject_in_hour", "att_vshock"]
BLOCKS = [(0.30, 0.44), (0.44, 0.58), (0.58, 0.72), (0.72, 0.86), (0.86, 1.0)]


def screen_train(Fm, y, is_cls, idx):
    """Survivors on the training slice only (same rules as the T2 screen)."""
    half = len(idx) // 2
    out = []
    for j in range(Fm.shape[1]):
        x = Fm[idx, j]
        if is_cls:
            a_f = auc(y[idx], x)
            a1 = auc(y[idx[:half]], x[:half])
            a2 = auc(y[idx[half:]], x[half:])
            if abs(a_f - .5) >= 0.06 and (a1 - .5) * (a2 - .5) > 0:
                out.append((j, abs(a_f - .5)))
        else:
            ic, _ = spearmanr(x, y[idx])
            i1, _ = spearmanr(x[:half], y[idx[:half]])
            i2, _ = spearmanr(x[half:], y[idx[half:]])
            if not np.isnan(ic) and abs(ic) >= 0.06 and i1 * i2 > 0:
                out.append((j, abs(ic)))
    out.sort(key=lambda t: -t[1])
    picked = []
    for j, _s in out:
        if any(abs(np.corrcoef(Fm[idx, j], Fm[idx, p])[0, 1]) > 0.7
               for p in picked):
            continue
        picked.append(j)
        if len(picked) >= 5:
            break
    return picked


def main() -> int:
    print("=" * 78)
    print("  V7-TRANSFER FOLD GAUNTLET — 每折 train 內選特徵，5 折 OOS Δ")
    print("=" * 78)
    F = pd.read_parquet(FEATS_PQ)
    rows = [r for r in M.build() if r["sym"] == "BTC"]
    for r in rows:
        r["dt"] = pd.Timestamp(r["ts"], unit="s", tz="UTC")
    rows = [r for r in rows if r["dt"] in F.index]
    rows.sort(key=lambda r: r["ts"])
    n = len(rows)
    fcols = [c for c in F.columns if c not in RAW]
    Fm = F[fcols].reindex([r["dt"] for r in rows]).to_numpy(dtype=float)
    med = np.nanmedian(Fm, axis=0)
    Fm = np.where(np.isnan(Fm), med, Fm)
    Xk = np.array([[r[k] for k in KEYS] for r in rows], dtype=float)
    yA = np.array([1 if r["cls"] == "BREAKOUT" else 0 for r in rows])
    fills = np.array([i for i, r in enumerate(rows) if r["netR"] is not None])
    yB_all = np.full(n, np.nan)
    for i in fills:
        yB_all[i] = rows[i]["netR"]

    from xgboost import XGBClassifier, XGBRegressor
    res = {}
    for task in ("resolution", "quality"):
        deltas, lines = [], []
        for (a, b) in BLOCKS:
            lo, hi = int(n * a), int(n * b)
            tr = np.arange(0, lo)
            te = np.arange(lo, hi)
            if task == "quality":
                tr = tr[~np.isnan(yB_all[tr])]
                te = te[~np.isnan(yB_all[te])]
                if len(te) < 40 or len(tr) < 150:
                    continue
                sel = screen_train(Fm, yB_all, False, tr)
                def sc(X):
                    m = XGBRegressor(max_depth=3, n_estimators=200,
                                     learning_rate=0.05, subsample=0.9,
                                     random_state=7)
                    m.fit(X[tr], yB_all[tr])
                    ic, _ = spearmanr(m.predict(X[te]), yB_all[te])
                    return ic
            else:
                if len(te) < 60 or len(tr) < 200:
                    continue
                sel = screen_train(Fm, yA, True, tr)
                def sc(X):
                    m = XGBClassifier(max_depth=3, n_estimators=200,
                                      learning_rate=0.05, subsample=0.9,
                                      random_state=7, eval_metric="logloss")
                    m.fit(X[tr], yA[tr])
                    return auc(yA[te], m.predict_proba(X[te])[:, 1])
            base = sc(Xk)
            withv = sc(np.column_stack([Xk, Fm[:, sel]])) if sel else base
            d = withv - base
            deltas.append(float(d))
            lines.append(f"    fold[{a:.0%}-{b:.0%}] base={base:+.3f} "
                         f"+V7={withv:+.3f} Δ={d:+.3f} (選{len(sel)}特徵)")
        print(f"\n  [{task}]")
        for ln in lines:
            print(ln)
        if deltas:
            arr = np.array(deltas)
            rng = np.random.default_rng(7)
            boots = [float(np.mean(rng.choice(arr, len(arr), replace=True)))
                     for _ in range(2000)]
            lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
            npos = int((arr > 0).sum())
            print(f"    per-fold mean Δ={arr.mean():+.4f} · 正折 {npos}/{len(arr)}"
                  f" · bootstrap CI [{lo_ci:+.4f}, {hi_ci:+.4f}]")
            verdict = ("PASS" if arr.mean() > 0 and npos >= 3
                       and lo_ci > -0.005 else "NO-GO")
            print(f"    verdict: {verdict}")
            res[task] = {"deltas": deltas, "mean": round(float(arr.mean()), 4),
                         "n_pos": npos, "ci": [round(float(lo_ci), 4),
                                               round(float(hi_ci), 4)],
                         "verdict": verdict}

    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
