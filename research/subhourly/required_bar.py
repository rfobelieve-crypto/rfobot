# -*- coding: utf-8 -*-
"""Phase 0 後續 power analysis — 量化 PARK 復活 re-run 的「必要贏面」。

不是新的假設檢定：只把已判決的掃描結果換算成「撤單特徵 10 月 re-run
要達到的具體數字」，讓 G2 的 8bps 門檻有可操作的翻譯：
  · 最佳 taker 系 cell 的 top-5% 條件桶: 毛捕捉 / 條件 |move| / 隱含方向勝率
  · 反推: 毛捕捉 >= 8bps 需要的勝率與 IC 倍數

Usage: python research/subhourly/required_bar.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from research.subhourly.minute_ic_scan import (   # noqa: E402
    build_features, load_era_a, load_era_b)

COST = 8.0
TOP = 0.05
CELLS = (  # (era_loader, extras, feature, horizon, ic_sign) — Phase 0a 已判最佳格
    ("A", "ti_15", 30, -1),
    ("A", "ret_15", 30, -1),
    ("B", "dz_60", 60, -1),
    ("B", "ret_60", 60, -1),
)


def main() -> int:
    frames = {"A": build_features(load_era_a(), with_extras=False),
              "B": build_features(load_era_b(), with_extras=True)}
    print(f"{'cell':16s} {'n_top':>7s} {'E|y|條件':>9s} {'毛捕捉':>7s} "
          f"{'隱含p':>6s} {'需p(8bps)':>9s} {'IC放大需':>8s}")
    for era, feat_name, h, sgn in CELLS:
        feat = frames[era]
        px = feat["px"]
        y = (px.shift(-h) / px - 1) * 10_000
        sub = np.column_stack([feat[feat_name].to_numpy(), y.to_numpy()])
        sub = sub[~np.isnan(sub).any(axis=1)]
        x, yy = sub[:, 0], sub[:, 1]
        q = np.quantile(np.abs(x), 1 - TOP)
        pick = np.abs(x) >= q
        xs, ys = x[pick], yy[pick]
        pnl = sgn * np.sign(xs) * ys                     # 毛捕捉 (bps)
        gross = float(pnl.mean())
        e_abs = float(np.abs(ys).mean())                 # 條件 |move|
        # 若全捕捉條件 |move|: 毛 = (2p-1)*E|y|cond → 隱含/所需 p
        p_now = 0.5 + gross / (2 * e_abs)
        p_req = 0.5 + COST / (2 * e_abs)
        mult = COST / gross if gross > 0 else float("inf")
        print(f"{era}/{feat_name}@{h}m".ljust(16)
              + f" {pick.sum():>7,} {e_abs:>8.1f}b {gross:>+6.2f}b "
              f"{p_now:>5.1%} {p_req:>8.1%} {mult:>7.2f}x")
    print("\n讀法: 撤單特徵 re-run 要過 G2, top-5% 桶毛捕捉須 ≥8bps —")
    print("即最佳 taker 系 cell 的 1.3-4x, 或等價地把條件方向勝率從")
    print("~51-53% 推到 ~55-56% (條件|move| 35-45bps 的視野下)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
