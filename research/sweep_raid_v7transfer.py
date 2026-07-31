# -*- coding: utf-8 -*-
"""V7 feature transfer onto raids — do the 4h model's state features (and
its own OOS prediction) separate post-raid scripts?

TODO 0.469 item 2 (2026-08-02). Three layers, discipline pre-stated:

  T1 (ONE named look): pred_align = -side * V7 OOS pred_ret at the raid
     bar — "does the 4h model agree with the fade". OOS fold predictions
     (direction_reg_oos_mse.parquet), never in-sample production preds
     (mistake.md 2026-04-13). Causal: pred at label J exists at J close,
     fills start J+1.
  T2 screen: ~276 engineered state features at the raid bar vs
     (a) resolution AUC, (b) netR IC. Stability bar: material full-sample
     effect (|AUC-.5|>=0.06 / |IC|>=0.06) AND both halves same side. With
     ~550 looks, chance alone passes a handful — the count vs chance is
     printed, and only survivors that ALSO add incrementally (T3) matter.
  T3 incremental: confirmed keys [pierce, att_min, reject, vshock] alone
     vs keys + top de-correlated survivors — time-split 70/30 OOS AUC/IC
     delta + permutation. The standing lesson (WQ101, liq-features,
     2026-06-01/02): univariate pass without ensemble lift = redundancy.

Universe: BTC raids inside the feature parquet window (~5.5 months).
Run: python research/sweep_raid_v7transfer.py
Out: research/results/sweep_raid_v7transfer.json
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

OUT = ROOT / "research/results/sweep_raid_v7transfer.json"
FEATS_PQ = ROOT / "research/dual_model/.cache/features_all.parquet"
OOS_PQ = ROOT / "research/results/dual_model/direction_reg_oos_mse.parquet"
RAW = {"open", "high", "low", "close", "volume", "taker_buy_vol"}
KEYS = ["pierce", "att_min", "reject_in_hour", "att_vshock"]


def main() -> int:
    print("=" * 78)
    print("  V7 FEATURE TRANSFER — 4h 模型的狀態特徵/預測 能否分獵取劇本")
    print("=" * 78)
    F = pd.read_parquet(FEATS_PQ)
    P = pd.read_parquet(OOS_PQ)
    rows = [r for r in M.build() if r["sym"] == "BTC"]
    for r in rows:
        r["dt"] = pd.Timestamp(r["ts"], unit="s", tz="UTC")
    rows = [r for r in rows if r["dt"] in F.index]
    rows.sort(key=lambda r: r["ts"])
    n = len(rows)
    print(f"  BTC raids inside feature window: {n}")
    res = {}

    yA = np.array([1 if r["cls"] == "BREAKOUT" else 0 for r in rows])
    fills_i = [i for i, r in enumerate(rows) if r["netR"] is not None]
    yB = np.array([rows[i]["netR"] for i in fills_i])

    # ── T1: the model's own view at the raid bar (ONE look) ─────────────
    pr = []
    for r in rows:
        pr.append(float(P.loc[r["dt"], "pred_ret"]) if r["dt"] in P.index
                  else np.nan)
    pr = np.array(pr)
    mask = ~np.isnan(pr)
    align = np.array([-rows[i]["side"] * pr[i] for i in range(n)])
    mA = mask
    a_res = auc(yA[mA], align[mA])
    fm = [i for i in fills_i if mask[i]]
    icB, _ = spearmanr([align[i] for i in fm], [rows[i]["netR"] for i in fm])
    hb = len(fm) // 2
    ic1, _ = spearmanr([align[i] for i in fm[:hb]], [rows[i]["netR"] for i in fm[:hb]])
    ic2, _ = spearmanr([align[i] for i in fm[hb:]], [rows[i]["netR"] for i in fm[hb:]])
    print(f"\n  [T1] pred_align（V7 是否站在 fade 這邊, n={int(mA.sum())}, fills={len(fm)}）")
    print(f"    解析 AUC {a_res:.3f} · netR IC {icB:+.3f} (半 {ic1:+.3f}/{ic2:+.3f})")
    # tercile read for interpretability
    va = sorted(align[i] for i in fm)
    lo_c, hi_c = va[len(va) // 3], va[2 * len(va) // 3]
    for name, pred in (("模型反對 fade", lambda v: v <= lo_c),
                       ("中性", lambda v: lo_c < v < hi_c),
                       ("模型支持 fade", lambda v: v >= hi_c)):
        g = [rows[i]["netR"] for i in fm if pred(align[i])]
        wr = 100 * sum(1 for x in g if x > 0) / len(g)
        print(f"    {name:<10} netR {sum(g)/len(g):+.3f} / WR {wr:.0f}% (n={len(g)})")
    res["T1_pred_align"] = {"n": int(mA.sum()), "auc_res": round(float(a_res), 3),
                            "ic": round(float(icB), 3),
                            "ic_h": [round(float(ic1), 3), round(float(ic2), 3)]}

    # ── T2: screen all engineered features at the raid bar ──────────────
    fcols = [c for c in F.columns if c not in RAW]
    half = n // 2
    hbf = len(fills_i) // 2
    surv_res, surv_q = [], []
    for c in fcols:
        x = F[c].reindex([r["dt"] for r in rows]).to_numpy(dtype=float)
        ok = ~np.isnan(x)
        if ok.mean() < 0.8:
            continue
        xa = np.where(ok, x, np.nanmedian(x))
        a_f = auc(yA, xa)
        a1 = auc(yA[:half], xa[:half])
        a2 = auc(yA[half:], xa[half:])
        if abs(a_f - .5) >= 0.06 and (a1 - .5) * (a2 - .5) > 0:
            surv_res.append((c, float(a_f)))
        xb = np.array([xa[i] for i in fills_i])
        ic, _ = spearmanr(xb, yB)
        i1, _ = spearmanr(xb[:hbf], yB[:hbf])
        i2, _ = spearmanr(xb[hbf:], yB[hbf:])
        if abs(ic) >= 0.06 and i1 * i2 > 0:
            surv_q.append((c, float(ic)))
    print(f"\n  [T2] screen {len(fcols)} 特徵 × 2 目標（~{2*len(fcols)} looks）")
    print(f"    解析存活 {len(surv_res)} 個 · 品質存活 {len(surv_q)} 個")
    surv_res.sort(key=lambda t: -abs(t[1] - .5))
    surv_q.sort(key=lambda t: -abs(t[1]))
    for tag, lst, fmt in (("解析", surv_res[:10], lambda v: f"AUC {v:.3f}"),
                          ("品質", surv_q[:10], lambda v: f"IC {v:+.3f}")):
        print(f"    top {tag}:")
        for c, v in lst:
            print(f"      {c:<38} {fmt(v)}")
    res["T2_counts"] = {"screened": len(fcols), "res_surv": len(surv_res),
                        "q_surv": len(surv_q)}
    res["T2_top_res"] = surv_res[:10]
    res["T2_top_q"] = surv_q[:10]

    # ── T3: incremental over confirmed keys ─────────────────────────────
    def decorr(cands, cap=5):
        picked = []
        for c, _v in cands:
            x = F[c].reindex([r["dt"] for r in rows]).to_numpy(dtype=float)
            if any(abs(pd.Series(x).corr(pd.Series(
                    F[p].reindex([r["dt"] for r in rows]).to_numpy(dtype=float))))
                    > 0.7 for p in picked):
                continue
            picked.append(c)
            if len(picked) >= cap:
                break
        return picked

    from xgboost import XGBClassifier, XGBRegressor
    rng = np.random.default_rng(7)
    Xk = np.array([[r[k] for k in KEYS] for r in rows], dtype=float)
    cut = int(n * 0.7)
    for task, cands in (("resolution", surv_res), ("quality", surv_q)):
        add = decorr(cands)
        Xadd = np.column_stack([
            np.nan_to_num(F[c].reindex([r["dt"] for r in rows])
                          .to_numpy(dtype=float),
                          nan=float(np.nanmedian(F[c]))) for c in add]) \
            if add else np.empty((n, 0))
        if task == "resolution":
            def fit_score(X):
                m = XGBClassifier(max_depth=3, n_estimators=200,
                                  learning_rate=0.05, subsample=0.9,
                                  random_state=7, eval_metric="logloss")
                m.fit(X[:cut], yA[:cut])
                return auc(yA[cut:], m.predict_proba(X[cut:])[:, 1])
            b = fit_score(Xk)
            w = fit_score(np.column_stack([Xk, Xadd])) if add else b
        else:
            fi = np.array(fills_i)
            cutb = int(len(fi) * 0.7)
            def fit_score(X):
                m = XGBRegressor(max_depth=3, n_estimators=200,
                                 learning_rate=0.05, subsample=0.9,
                                 random_state=7)
                m.fit(X[fi[:cutb]], yB[:cutb])
                ic, _ = spearmanr(m.predict(X[fi[cutb:]]), yB[cutb:])
                return ic
            b = fit_score(Xk)
            w = fit_score(np.column_stack([Xk, Xadd])) if add else b
        print(f"\n  [T3·{task}] keys-only OOS={b:+.3f} → +V7({len(add)}特徵) OOS={w:+.3f}"
              f"  Δ={w-b:+.3f}   added={add}")
        res[f"T3_{task}"] = {"base": round(float(b), 3), "with": round(float(w), 3),
                             "added": add}

    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    print("  讀法: T2 存活數要跟機率基準比; T3 Δ 才是部署層級的證據"
          "（單變量過≠有增量, WQ101 教訓）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
