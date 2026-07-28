# -*- coding: utf-8 -*-
"""Why does hourly order flow carry no standalone edge at 4h? Measure the decay.

Keep-only showed the 30 order-flow features alone score AUC 0.4995 — nothing.
The standard microstructure explanation is horizon mismatch: flow imbalance
explains the move it участвует in (same bar) and dies within minutes, while
what survives to a 4h horizon is slow STATE (positioning = integrated flow).

This measures that decay directly on our own features: Spearman IC of each
feature against returns at increasing horizons.

  same_bar*  — return of the bar the flow happened in (close[t]/close[t-1]-1).
               NOT tradeable; it is the "flow explains price" contemporaneous
               correlation. Row t's feature is built from minutes [t, t+60)
               and close[t] is that bar's close (alignment verified 400/400 in
               research/intrabar_volume_ic.py), so this pairing is exact.
  +1h/+2h/+4h/+8h — forward point returns from close[t] (tradeable horizons).
  4h_path    — the production target, mean(close[t+1..t+4])/close[t]-1.

Expected signature if horizon mismatch is the mechanism:
  per-bar flow deltas:   big |IC| at same_bar, collapse at +1h and beyond
  integrated flow (cum/MA): intermediate
  positioning state:     small at same_bar, flat-or-growing with horizon

Run: python research/orderflow_horizon_decay.py
Out: research/results/orderflow_horizon_decay.txt (verbatim table)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.dual_model.shared_data import load_and_cache_data  # noqa: E402
from research.dual_model.build_direction_reg_labels import build_direction_reg_labels  # noqa: E402

OUT = ROOT / "research/results/orderflow_horizon_decay.txt"

FAMILIES = [
    ("FLOW per-bar deltas", [
        "taker_delta_ratio", "cg_taker_delta", "cg_taker_delta_zscore",
        "cg_fcvd_delta", "cg_scvd_delta", "impact_asymmetry_zscore",
        "post_absorb_breakout",
    ]),
    ("FLOW integrated (cum / MA / persistence)", [
        "taker_delta_ma_24h", "cvd_persistence_12h",
        "cg_fcvd_cum", "cg_scvd_cum",
    ]),
    ("POSITIONING state", [
        "cg_bfx_margin_ratio", "cg_funding_close", "cg_pos_long_pct",
        "cg_crowding", "cg_oi_binance_share", "cg_pos_account_divergence",
        "cg_cb_premium",
    ]),
]


def main() -> int:
    df = load_and_cache_data()
    labels = build_direction_reg_labels(df)
    df = df.copy()
    df["y_path_ret_4h"] = labels["y_path_ret_4h"]
    c = df["close"].astype(float)

    horizons = {
        "same_bar*": c / c.shift(1) - 1,
        "+1h": c.shift(-1) / c - 1,
        "+2h": c.shift(-2) / c - 1,
        "+4h": c.shift(-4) / c - 1,
        "4h_path": df["y_path_ret_4h"],
        "+8h": c.shift(-8) / c - 1,
    }

    n = len(df)
    sig = 1.96 / np.sqrt(n)
    lines = []
    lines.append(f"n_bars={n}   |IC| significance yardstick (p<0.05) ~ {sig:.3f}")
    lines.append(f"{'feature':<28}" + "".join(f"{h:>10}" for h in horizons))
    for fam, feats in FAMILIES:
        lines.append(f"-- {fam}")
        for f in feats:
            if f not in df.columns:
                lines.append(f"{f:<28}{'(missing)':>10}")
                continue
            cells = []
            for r in horizons.values():
                m = df[f].notna() & r.notna()
                if m.sum() < 100:
                    cells.append(float("nan"))
                    continue
                cells.append(spearmanr(df.loc[m, f].values, r[m].values).correlation)
            lines.append(f"{f:<28}" + "".join(
                f"{v:+10.3f}" if np.isfinite(v) else f"{'nan':>10}" for v in cells))

    text = "\n".join(lines)
    print(text)
    OUT.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
