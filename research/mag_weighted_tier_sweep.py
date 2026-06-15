"""
Mag-weighted Strong tier sweep on production WF OOS.

Q: Does adding a magnitude gate to the Strong tier meaningfully lift
   top-k precision without killing too many signals?

Method
------
Replay production direction-reg rolling-percentile decoding on the
3696-row WF OOS from direction_reg_oos_mse.parquet, aligned with the
same-period magnitude OOS from magnitude_oos_phase1_voladj.parquet.

For each bar compute mag_pct_200 (rolling 200-bar percentile of
|mag_pred_sigma|) — this matches what inference.py now outputs in live
(`mag_pct_200` column written to tracked_signals from commit f56a5d2).

Sweep `mag_gate` ∈ {none, 70, 75, 80, 85, 90}. For each gate:
  Strong_v2 = Original Strong tier AND mag_pct_200 >= mag_gate
  Strongs that fail the gate are demoted to Moderate (not deleted).

Report per gate:
  - Strong count (vs baseline)
  - Strong precision (hit rate of directional signal vs realized sign)
  - Wilson 95% CI
  - Mean |realized return| on Strong signals
  - Signal frequency change vs baseline

Output: research/results/mag_weighted_tier_sweep.json
"""
from __future__ import annotations

import json
import logging
import math
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIR_OOS = Path("research/results/dual_model/direction_reg_oos_mse.parquet")
MAG_OOS = Path("research/results/dual_model/magnitude_oos_phase1_voladj.parquet")
DIRECTION_CONFIG = Path("indicator/model_artifacts/dual_model/direction_reg_config.json")
OUT = Path("research/results/mag_weighted_tier_sweep.json")

MAG_WINDOW = 200
MAG_GATES = [None, 70, 75, 80, 85, 90]


def wilson_ci(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    halfw = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (max(0.0, centre - halfw), min(1.0, centre + halfw))


def simulate(pred_ret: np.ndarray, mag_sigma: np.ndarray, cfg: dict,
             mag_gate_pct: float | None) -> dict:
    """Replay production decoding + optional mag gate on Strong tier.

    mag_gate_pct: None = no gate (baseline); otherwise e.g. 80 means
    Strong requires pred_ret in direction top-5% AND mag_pct_200 >= 80.
    Strongs failing the gate are demoted to Moderate.
    """
    warmup = cfg["warmup_bars"]
    strong_frac = cfg["strong_top_frac"]
    mod_frac = cfg["moderate_top_frac"]
    win = cfg["percentile_window"]
    fb = cfg["fallback"]

    n = len(pred_ret)
    buf_dir: deque = deque(maxlen=win)
    buf_mag: deque = deque(maxlen=MAG_WINDOW)

    tier = np.full(n, "Weak", dtype=object)
    direction_sign = np.zeros(n)
    mag_pct_arr = np.full(n, 50.0)

    for i in range(n):
        p = float(pred_ret[i])
        m = abs(float(mag_sigma[i])) if not np.isnan(mag_sigma[i]) else 0.0
        buf_dir.append(p)
        if m > 1e-9:
            buf_mag.append(m)

        if len(buf_dir) < warmup:
            up_s = fb["strong_up"]; dn_s = fb["strong_dn"]
            up_m = fb["moderate_up"]; dn_m = fb["moderate_dn"]
        else:
            arr = np.fromiter(buf_dir, dtype=float)
            up_s = float(np.quantile(arr, 1.0 - strong_frac / 2.0))
            dn_s = float(np.quantile(arr, strong_frac / 2.0))
            up_m = float(np.quantile(arr, 1.0 - mod_frac / 2.0))
            dn_m = float(np.quantile(arr, mod_frac / 2.0))

        # Mag percentile (rolling 200-bar)
        if len(buf_mag) >= 10 and m > 1e-9:
            arr_m = np.fromiter(buf_mag, dtype=float)
            mag_pct = float((arr_m <= m).sum() / len(arr_m) * 100)
        else:
            mag_pct = 50.0
        mag_pct_arr[i] = mag_pct

        # Direction tier (production logic)
        if p >= up_s:
            tier[i] = "Strong"; direction_sign[i] = 1
        elif p <= dn_s:
            tier[i] = "Strong"; direction_sign[i] = -1
        elif p >= up_m:
            tier[i] = "Moderate"; direction_sign[i] = 1
        elif p <= dn_m:
            tier[i] = "Moderate"; direction_sign[i] = -1

        # Mag gate on Strong tier
        if mag_gate_pct is not None and tier[i] == "Strong":
            if mag_pct < mag_gate_pct:
                tier[i] = "Moderate"  # demote, direction_sign preserved

    return {
        "tier": tier,
        "direction_sign": direction_sign,
        "mag_pct": mag_pct_arr,
    }


def evaluate_tier(tier: np.ndarray, direction_sign: np.ndarray,
                  actual_ret: np.ndarray, tier_label: str) -> dict:
    mask = tier == tier_label
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "precision": None, "ci95": [None, None],
                "mean_abs_return": None}
    # Hit = direction_sign predicted matches actual sign
    pd_ = direction_sign[mask]
    ar = actual_ret[mask]
    wins = ((pd_ > 0) & (ar > 0)) | ((pd_ < 0) & (ar < 0))
    n_wins = int(wins.sum())
    prec = n_wins / n
    ci_lo, ci_hi = wilson_ci(n_wins, n)
    return {
        "n": n,
        "n_wins": n_wins,
        "precision": float(prec),
        "ci95": [float(ci_lo), float(ci_hi)],
        "mean_abs_return": float(np.abs(ar).mean()),
    }


def main():
    logger.info("Loading WF OOS artifacts...")
    dir_oos = pd.read_parquet(DIR_OOS).sort_index()
    mag_oos = pd.read_parquet(MAG_OOS).sort_index()

    common = dir_oos.index.intersection(mag_oos.index)
    logger.info("Dir OOS: %d  Mag OOS: %d  Common: %d",
                len(dir_oos), len(mag_oos), len(common))

    with open(DIRECTION_CONFIG) as f:
        cfg = json.load(f)

    df = pd.DataFrame({
        "pred_ret": dir_oos.loc[common, "pred_ret"].astype(float),
        "y": dir_oos.loc[common, "y_path_ret_4h"].astype(float),
        "mag_sigma": mag_oos.loc[common, "y_pred"].astype(float),
    }).dropna()
    logger.info("Aligned + clean: %d bars", len(df))

    pred_ret = df["pred_ret"].values
    mag_sigma = df["mag_sigma"].values
    actual_ret = df["y"].values

    results = []
    baseline_strong_n = None

    for gate in MAG_GATES:
        out = simulate(pred_ret, mag_sigma, cfg, gate)
        strong_stats = evaluate_tier(out["tier"], out["direction_sign"],
                                      actual_ret, "Strong")
        moderate_stats = evaluate_tier(out["tier"], out["direction_sign"],
                                        actual_ret, "Moderate")
        if gate is None:
            baseline_strong_n = strong_stats["n"]
        signal_freq_delta = (
            (strong_stats["n"] - baseline_strong_n) / baseline_strong_n * 100
            if baseline_strong_n else 0.0
        )
        row = {
            "mag_gate": "baseline" if gate is None else f"p{gate}",
            "mag_gate_pct": gate,
            "strong": strong_stats,
            "moderate": moderate_stats,
            "strong_signal_freq_delta_pct": round(signal_freq_delta, 1),
        }
        results.append(row)

        s = strong_stats
        if s["precision"] is not None:
            logger.info(
                "gate=%-9s  Strong: n=%-3d  prec=%.3f  CI[%.2f,%.2f]  "
                "avg|ret|=%.4f  Δfreq=%+.1f%%",
                row["mag_gate"], s["n"], s["precision"],
                s["ci95"][0], s["ci95"][1], s["mean_abs_return"],
                signal_freq_delta,
            )

    # Summary table formatted as markdown-ish for copy/paste
    summary_lines = [
        "",
        f"{'Gate':<10} {'Strong n':<10} {'Precision':<12} {'95% CI':<16} {'avg|ret|':<10} {'Δfreq':<10}",
        "-" * 72,
    ]
    for r in results:
        s = r["strong"]
        if s["precision"] is None:
            summary_lines.append(f"{r['mag_gate']:<10} n=0 skipped")
            continue
        summary_lines.append(
            f"{r['mag_gate']:<10} "
            f"{s['n']:<10} "
            f"{s['precision']*100:5.1f}%      "
            f"[{s['ci95'][0]*100:4.1f}, {s['ci95'][1]*100:4.1f}]  "
            f"{s['mean_abs_return']*100:5.3f}%    "
            f"{r['strong_signal_freq_delta_pct']:+5.1f}%"
        )

    logger.info("")
    for line in summary_lines:
        logger.info(line)

    out = {
        "description": "Mag-weighted Strong tier sweep on production WF OOS",
        "n_bars": int(len(df)),
        "date_range": [str(df.index[0]), str(df.index[-1])],
        "mag_window_bars": MAG_WINDOW,
        "config": {
            "percentile_window": cfg["percentile_window"],
            "warmup_bars": cfg["warmup_bars"],
            "strong_top_frac": cfg["strong_top_frac"],
            "moderate_top_frac": cfg["moderate_top_frac"],
        },
        "results": results,
        "summary_table": summary_lines,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info("\nResults → %s", OUT)


if __name__ == "__main__":
    main()
