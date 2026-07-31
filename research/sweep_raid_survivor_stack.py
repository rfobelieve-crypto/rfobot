# -*- coding: utf-8 -*-
"""Survivor stack — the user's scoring suggestion, restricted to features
that EARNED their seat. The full-pool version (combo G) already failed the
fold gauntlet; this is the disciplined complement: stack ONLY the
survivors, score every raid, and judge by the HIGH-CONFIDENCE bucket
(user: 重點看高信心區間的表現，而不是整體平均).

Features (each with a documented survival record):
  pierce, att_min, reject, att_vshock   attack-window keys (keydrivers)
  q_flag                                during-raid OI down + taker with
                                        break (quadrants)
  liq_burst                             best real-flow univariate (0.731)
  pred_align                            V7's own OOS view (CI all-positive)

Universe: BTC fills with full coverage (CG + V7 OOS pred window).
Target: fade win (netR > 0); expectancy read on netR.
Method: 5 expanding-window folds, XGB, pooled+per-fold OOS AUC, then the
top-20% predicted-probability bucket per fold — WR / mean netR vs the
fold's own base. Verdict bar: majority of folds must show top-bucket
netR above the fold base, and pooled top-bucket bootstrap CI above base.

Run: python research/sweep_raid_survivor_stack.py
Out: research/results/sweep_raid_survivor_stack.json
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
import sweep_raid_menu2 as M  # noqa: E402
import sweep_raid_anatomy as A  # noqa: E402
import sweep_raid_derivs as D  # noqa: E402
from sweep_raid_keydrivers import auc  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/sweep_raid_survivor_stack.json"
OOS_PQ = ROOT / "research/results/dual_model/direction_reg_oos_mse.parquet"
FEATS = ["pierce", "att_min", "reject_in_hour", "att_vshock",
         "q_flag", "liq_burst", "pred_align"]
BLOCKS = [(0.30, 0.44), (0.44, 0.58), (0.58, 0.72), (0.72, 0.86), (0.86, 1.0)]


def main() -> int:
    print("=" * 78)
    print("  SURVIVOR STACK — 只堆疊存活特徵的評分模型 + 高信心區間（BTC）")
    print("=" * 78)
    P = pd.read_parquet(OOS_PQ)
    S = D.load_state()
    dmap = {r["ts"]: r for r in D.attach(A.raids("BTC"), S)}
    rows = []
    for r in M.build():
        if r["sym"] != "BTC" or r["netR"] is None:
            continue
        dt = pd.Timestamp(r["ts"], unit="s", tz="UTC")
        d = dmap.get(r["ts"])
        if d is None or dt not in P.index:
            continue
        if d.get("oi_chg_raid") is None or d.get("fut_taker_signed") is None \
                or d.get("liq_burst") is None:
            continue
        r2 = dict(r)
        r2["q_flag"] = int(d["oi_chg_raid"] < 0 and d["fut_taker_signed"] > 0)
        r2["liq_burst"] = d["liq_burst"]
        r2["pred_align"] = -r["side"] * float(P.loc[dt, "pred_ret"])
        rows.append(r2)
    rows.sort(key=lambda r: r["ts"])
    n = len(rows)
    X = np.array([[r[f] for f in FEATS] for r in rows], dtype=float)
    ynet = np.array([r["netR"] for r in rows])
    ywin = (ynet > 0).astype(int)
    base_wr = 100 * ywin.mean()
    base_net = ynet.mean()
    print(f"  universe: {n} BTC fills (全覆蓋) · 基準 WR {base_wr:.0f}% · "
          f"均 netR {base_net:+.3f}")

    from xgboost import XGBClassifier
    fold_lines, top_all, rest_all, aucs = [], [], [], []
    oof = np.full(n, np.nan)
    for (a, b) in BLOCKS:
        lo, hi = int(n * a), int(n * b)
        tr, te = np.arange(0, lo), np.arange(lo, hi)
        m = XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.05,
                          subsample=0.9, random_state=7, eval_metric="logloss")
        m.fit(X[tr], ywin[tr])
        p = m.predict_proba(X[te])[:, 1]
        oof[te] = p
        a_f = auc(ywin[te], p)
        aucs.append(float(a_f))
        cut = np.quantile(p, 0.8)
        top = te[p >= cut]
        rest = te[p < cut]
        top_all += list(top)
        rest_all += list(rest)
        t_net = ynet[top].mean()
        t_wr = 100 * ywin[top].mean()
        b_net = ynet[te].mean()
        fold_lines.append(
            f"    fold[{a:.0%}-{b:.0%}] AUC {a_f:.3f} · top20% netR {t_net:+.3f}"
            f"/WR {t_wr:.0f}% (n={len(top)}) vs 折基準 {b_net:+.3f}"
            f"  {'✓' if t_net > b_net else '✗'}")

    print("\n  [5 折 OOS]")
    for ln in fold_lines:
        print(ln)
    n_beat = sum(1 for ln in fold_lines if ln.endswith("✓"))
    top_all = np.array(top_all)
    rest_all = np.array(rest_all)
    t_net, t_wr = ynet[top_all].mean(), 100 * ywin[top_all].mean()
    r_net = ynet[rest_all].mean()
    rng = np.random.default_rng(7)
    boots = [float(np.mean(rng.choice(ynet[top_all], len(top_all), True)))
             for _ in range(2000)]
    lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
    print(f"\n  [pooled OOS] AUC mean {np.mean(aucs):.3f} · "
          f"top20% n={len(top_all)}: netR {t_net:+.3f} / WR {t_wr:.0f}%"
          f"  vs 其餘 {r_net:+.3f} · top-bucket bootstrap CI "
          f"[{lo_ci:+.3f}, {hi_ci:+.3f}]")
    verdict = ("PASS" if n_beat >= 3 and t_net > base_net and lo_ci > 0
               else "NO-GO")
    print(f"  verdict（≥3/5 折勝基準 + pooled top CI>0）: {verdict}")

    res = {"n": n, "base_wr": round(float(base_wr), 1),
           "base_net": round(float(base_net), 3),
           "fold_aucs": [round(a, 3) for a in aucs],
           "folds_beat_base": n_beat,
           "top20": {"n": int(len(top_all)), "netR": round(float(t_net), 3),
                     "wr": round(float(t_wr), 1),
                     "ci": [round(float(lo_ci), 3), round(float(hi_ci), 3)]},
           "rest_netR": round(float(r_net), 3), "verdict": verdict}
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
