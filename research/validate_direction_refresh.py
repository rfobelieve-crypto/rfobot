"""Direction-model refresh — pre-deploy gate (G1-G4), written BEFORE results.

Context (2026-08-08): the deployed direction model (2026-05-01) drifted —
its output MEAN moved +0.0024 over four months while Spearman-IC-based
revalidation stayed green (rank metrics are blind to level shifts). The
two-tail rolling-percentile decode turned that level drift into direction
skew (July fired 14 UP : 1 DOWN Strong) and the executor traded almost
only its weak side (live LONG −27 bps vs SHORT +38 bps). Fix = retrain on
data through today (maintenance refresh: same features, same params, same
tier definition).

GATES (all four must pass or NO DEPLOY):
  G1 quality     WF sign_AUC in [0.55, 0.62] (monthly-revalidation band)
  G2 centring    new production model on the last 60d:
                   |mean(pred)| < 0.5 x ABS_FLOOR (0.0004)
                   AND each tail has >= 2% of preds beyond the floor
  G3 balance     offline two-tail decode replay of the last 60d:
                   Strong DOWN share in [30%, 70%]
  G4 buffer      seeded dir_pred_history std vs new model's recent-preds
                   std in [0.5, 2.0]  (mistake.md 2026-04-19)

Run AFTER export_direction_reg_model has written the new artifacts:
    python research/validate_direction_refresh.py
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ART = ROOT / "indicator" / "model_artifacts" / "dual_model"
RES = ROOT / "research" / "results" / "dual_model"
FLOOR = 0.0008
RECENT_BARS = 1440          # ~60d of 1h bars


def g1_quality() -> tuple[bool, str]:
    oos = pd.read_parquet(RES / "direction_reg_oos_mse.parquet")
    pred, y = oos["pred_ret"].to_numpy(), oos["y_path_ret_4h"].to_numpy()
    ok = np.isfinite(pred) & np.isfinite(y) & (y != 0)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score((y[ok] > 0).astype(int), pred[ok])
    passed = 0.55 <= auc <= 0.62
    return passed, f"WF sign_AUC={auc:.4f} (band [0.55,0.62], n={ok.sum()})"


def _recent_preds() -> np.ndarray:
    """New production model's predictions on the freshest RECENT_BARS bars."""
    import xgboost as xgb
    from research.dual_model.shared_data import load_and_cache_data

    feats = json.loads((ART / "direction_feature_cols.json").read_text())
    booster = xgb.XGBRegressor()
    booster.load_model(str(ART / "direction_xgb.json"))
    df = load_and_cache_data()
    X = df[feats].tail(RECENT_BARS)
    return booster.predict(X.to_numpy(dtype=np.float32)), X.index


def g2_centring(pred: np.ndarray) -> tuple[bool, str]:
    m = float(np.mean(pred))
    lo_tail = float((pred <= -FLOOR).mean())
    hi_tail = float((pred >= FLOOR).mean())
    passed = abs(m) < 0.5 * FLOOR and lo_tail >= 0.02 and hi_tail >= 0.02
    return passed, (f"mean={m:+.5f} (|.|<{0.5*FLOOR:.4f})  "
                    f"tail<=-floor {lo_tail:.1%}  tail>=+floor {hi_tail:.1%} (each >=2%)")


def g3_balance(pred: np.ndarray) -> tuple[bool, str]:
    buf: deque = deque(maxlen=500)
    up = dn = 0
    for v in pred:
        if len(buf) >= 100:
            uc = max(np.percentile(buf, 97.5), FLOOR)
            dc = min(np.percentile(buf, 2.5), -FLOOR)
            if v >= uc:
                up += 1
            elif v <= dc:
                dn += 1
        buf.append(v)
    tot = up + dn
    share = dn / tot if tot else float("nan")
    passed = tot >= 10 and 0.30 <= share <= 0.70
    return passed, f"replay fires UP {up} / DOWN {dn}  (DOWN share {share:.0%}, band [30%,70%])"


def g4_buffer(pred: np.ndarray) -> tuple[bool, str]:
    stats = json.loads((ROOT / "indicator" / "model_artifacts" / "dual_model"
                        / "training_stats.json").read_text())
    hist = stats.get("dir_pred_history") or []
    if len(hist) < 100:
        return False, f"dir_pred_history has only {len(hist)} entries"
    ratio = float(np.std(hist) / max(np.std(pred[-200:]), 1e-12))
    passed = 0.5 <= ratio <= 2.0
    return passed, f"buffer std / recent pred std = {ratio:.2f} (band [0.5,2.0], n={len(hist)})"


def main() -> int:
    pred, idx = _recent_preds()
    print(f"新模型近期預測：n={len(pred)}  {idx[0]} → {idx[-1]}\n")
    gates = [("G1 品質", g1_quality()),
             ("G2 對中", g2_centring(pred)),
             ("G3 平衡", g3_balance(pred)),
             ("G4 buffer", g4_buffer(pred))]
    all_ok = True
    for name, (ok, msg) in gates:
        all_ok &= ok
        print(f"  {name}: {'PASS' if ok else '** FAIL **'}  {msg}")
    print()
    print("VERDICT:", "DEPLOY OK — 四關全過" if all_ok
          else "NO DEPLOY — 有關卡未過，還原備份、記錄發現、B 案升級")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
