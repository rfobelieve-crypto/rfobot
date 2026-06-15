"""
Cascade Modifier Study — Stage 1: H2 Stability & Reversion Dynamics.

This is the gating stage before we commit to testing a confidence modifier
in production-like backtest. Based on prior research (label_feasibility_sweep
H2 finding): when cascade occurs, cg_liq_imbalance shows Spearman IC of
-0.252 with future 4h return — suggesting a strong contrarian signal via
liquidation exhaustion. Before integrating this as a confidence modifier,
we must verify:
  Task 1.1  — is the negative IC STABLE across months, or is the edge
              concentrated in a minority of the sample?
  Task 1.2  — what are the reversion magnitudes and at what horizon; does
              the contrarian trade survive round-trip fees?

Does NOT modify production code. All outputs go to
research/results/cascade_modifier_study/.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_CACHE = Path("research/dual_model/.cache/features_all.parquet")
OUT_DIR = Path("research/results/cascade_modifier_study")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 4
ROUND_TRIP_FEE = 0.0008  # 0.08% Binance taker round-trip
MIN_MONTHLY_CASCADES = 15  # minimum cascade events in a month to compute IC
N_BOOT = 500


def build_data():
    df = pd.read_parquet(FEATURE_CACHE)
    close = df["close"].astype(float)
    liq_total = df["cg_liq_total"].astype(float).fillna(0)
    liq_imb = df["cg_liq_imbalance"].astype(float)

    # Forward 4h TWAP path return (production target)
    fwd_mean = sum(close.shift(-i) for i in range(1, HORIZON + 1)) / HORIZON
    path_ret_4h = fwd_mean / close - 1

    # Endpoint 4h return (for reversion magnitude analysis)
    endpoint_ret_4h = close.shift(-HORIZON) / close - 1

    # Per-hour forward returns for dynamics plot
    per_hour = {
        f"ret_{h}h": close.shift(-h) / close - 1
        for h in range(1, HORIZON + 1)
    }

    # Cascade surge: forward 4h liquidation > trailing 500-bar 90th pct × HORIZON
    fwd_liq = sum(liq_total.shift(-i) for i in range(1, HORIZON + 1))
    trailing_p90 = liq_total.rolling(500, min_periods=100).quantile(0.90) * HORIZON
    surge = (fwd_liq > trailing_p90).astype(float)
    surge[(trailing_p90.isna()) | (fwd_liq.isna())] = np.nan

    out = pd.DataFrame({
        "close": close,
        "liq_imb": liq_imb,
        "path_ret_4h": path_ret_4h,
        "endpoint_ret_4h": endpoint_ret_4h,
        "surge": surge,
        **per_hour,
    }, index=df.index)
    out = out.iloc[:-HORIZON].copy()
    return out


def bootstrap_spearman_ci(p, a, n_boot=N_BOOT):
    mask = ~(np.isnan(p) | np.isnan(a))
    p, a = np.asarray(p[mask]), np.asarray(a[mask])
    n = len(p)
    if n < 20:
        return float("nan"), float("nan")
    rng = np.random.default_rng(42)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = spearmanr(p[idx], a[idx])
        boots[b] = r if r == r else 0.0
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def task_1_1_monthly_breakdown(data: pd.DataFrame) -> dict:
    """Monthly H2: per-month Spearman IC of liq_imb vs path_ret_4h, and
    contrarian precision (-sign(liq_imb) as direction predictor)."""
    cascade_data = data[data["surge"] == 1].copy()
    cascade_data = cascade_data.dropna(subset=["liq_imb", "path_ret_4h"])
    logger.info("Cascade bars total: %d", len(cascade_data))

    periods = cascade_data.index.to_period("M")
    rows = []
    for month in sorted(periods.unique()):
        m_mask = (periods == month)
        m_data = cascade_data[m_mask]
        n = len(m_data)
        if n < MIN_MONTHLY_CASCADES:
            rows.append({
                "month": str(month), "cascade_count": n,
                "ic": None, "ic_ci_lower": None, "ic_ci_upper": None,
                "contrarian_precision": None, "contrarian_n_signals": 0,
                "skipped_reason": f"< {MIN_MONTHLY_CASCADES} cascade events",
            })
            continue

        imb = m_data["liq_imb"].values
        ret = m_data["path_ret_4h"].values
        ic, _ = spearmanr(imb, ret)
        ci_lo, ci_hi = bootstrap_spearman_ci(imb, ret)

        # Contrarian: predict direction = -sign(imb), compare to sign(ret)
        pred_dir = -np.sign(imb)
        actual_dir = np.sign(ret)
        valid = (pred_dir != 0) & (actual_dir != 0)
        if valid.sum() > 0:
            contra_prec = float((pred_dir[valid] == actual_dir[valid]).mean())
            contra_n = int(valid.sum())
        else:
            contra_prec = None
            contra_n = 0

        rows.append({
            "month": str(month), "cascade_count": n,
            "ic": float(ic) if ic == ic else None,
            "ic_ci_lower": ci_lo if ci_lo == ci_lo else None,
            "ic_ci_upper": ci_hi if ci_hi == ci_hi else None,
            "contrarian_precision": contra_prec,
            "contrarian_n_signals": contra_n,
        })

    # Global stats (all cascade bars)
    global_ic, _ = spearmanr(cascade_data["liq_imb"], cascade_data["path_ret_4h"])
    g_lo, g_hi = bootstrap_spearman_ci(
        cascade_data["liq_imb"].values, cascade_data["path_ret_4h"].values
    )

    # Stability verdict
    months_with_data = [r for r in rows if r["ic"] is not None]
    n_months = len(months_with_data)
    months_negative = sum(1 for r in months_with_data if r["ic"] < 0)
    months_negative_significant = sum(
        1 for r in months_with_data
        if r["ic"] < 0 and r["ic_ci_upper"] is not None and r["ic_ci_upper"] < 0
    )
    months_near_zero = sum(
        1 for r in months_with_data
        if r["ic"] is not None and -0.05 < r["ic"] < 0.05
    )

    if n_months == 0:
        verdict = "NO_DATA"
    elif months_negative_significant == n_months:
        verdict = "STABLE"
    elif months_negative == n_months and months_near_zero <= 2:
        verdict = "MARGINAL"
    elif (months_negative / n_months) < 0.70:
        verdict = "UNSTABLE"
    elif months_near_zero > n_months * 0.30:
        verdict = "UNSTABLE"
    else:
        verdict = "MARGINAL"

    result = {
        "description": "Per-month Spearman IC of cg_liq_imbalance vs path_ret_4h, restricted to cascade bars (surge=1)",
        "min_monthly_cascades_threshold": MIN_MONTHLY_CASCADES,
        "total_cascade_bars": int(len(cascade_data)),
        "global_ic": float(global_ic),
        "global_ic_ci95": [g_lo, g_hi],
        "months": rows,
        "stability_counts": {
            "n_months_with_data": n_months,
            "months_negative_ic": months_negative,
            "months_negative_and_ci_ge_0": months_negative_significant,
            "months_near_zero_ic": months_near_zero,
        },
        "stability_verdict": verdict,
    }
    out_path = OUT_DIR / "h2_monthly.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("[Task 1.1] → %s", out_path)
    logger.info("  Verdict: %s  (global IC=%.3f CI[%.3f,%.3f])",
                verdict, global_ic, g_lo, g_hi)
    for r in rows:
        ic_str = f"{r['ic']:+.3f}" if r["ic"] is not None else "  —  "
        ci_str = (f"[{r['ic_ci_lower']:+.3f},{r['ic_ci_upper']:+.3f}]"
                  if r["ic_ci_lower"] is not None else "—")
        prec_str = (f"{r['contrarian_precision']:.3f}"
                    if r["contrarian_precision"] is not None else " —  ")
        logger.info("  %s  n=%3d  IC=%s  CI=%s  contra_prec=%s",
                    r["month"], r["cascade_count"], ic_str, ci_str, prec_str)
    return result


def task_1_2_reversion_dynamics(data: pd.DataFrame) -> dict:
    """Analyse cascade-bar reversion: magnitude, per-hour dynamics, fee
    sensitivity."""
    cascade = data[data["surge"] == 1].copy()
    cascade = cascade.dropna(subset=["liq_imb", "endpoint_ret_4h",
                                     "ret_1h", "ret_2h", "ret_3h", "ret_4h"])
    non_cascade = data[data["surge"] == 0].copy()
    non_cascade = non_cascade.dropna(subset=["endpoint_ret_4h"])

    # 1. abs return distributions
    casc_abs = cascade["endpoint_ret_4h"].abs()
    non_casc_abs = non_cascade["endpoint_ret_4h"].abs()
    magnitude = {
        "cascade_bars": {
            "n": int(len(casc_abs)),
            "abs_return_4h_mean": float(casc_abs.mean()),
            "abs_return_4h_median": float(casc_abs.median()),
            "abs_return_4h_p25": float(casc_abs.quantile(0.25)),
            "abs_return_4h_p75": float(casc_abs.quantile(0.75)),
            "abs_return_4h_p90": float(casc_abs.quantile(0.90)),
        },
        "non_cascade_bars": {
            "n": int(len(non_casc_abs)),
            "abs_return_4h_mean": float(non_casc_abs.mean()),
            "abs_return_4h_median": float(non_casc_abs.median()),
        },
        "cascade_over_non_cascade_ratio_median": float(
            casc_abs.median() / non_casc_abs.median()
        ) if non_casc_abs.median() > 0 else None,
    }

    # 2. Per-hour cumulative return, split by sign(liq_imb).
    # H2 IC = -0.252 ⇒ liq_imb high (long-dom) predicts NEGATIVE future
    # return. So the winning strategy is: pred_dir = -sign(liq_imb),
    # i.e. GO WITH the side that is doing the liquidating (the
    # opposite side from the liquidated positions). Terminology note:
    # "contrarian to the liquidated side", not "contrarian to price".
    long_dom = cascade[cascade["liq_imb"] > 0]   # longs getting liquidated
    short_dom = cascade[cascade["liq_imb"] < 0]  # shorts getting liquidated
    hours = [1, 2, 3, 4]

    per_hour = {"long_dominant": {}, "short_dominant": {}, "contrarian_combined": {}}

    for h in hours:
        col = f"ret_{h}h"
        # long_dom (longs liquidated) → expect DOWN → strategy-P&L = -ret
        l = long_dom[col].dropna()
        per_hour["long_dominant"][f"{h}h"] = {
            "n": int(len(l)),
            "mean_return": float(l.mean()) if len(l) else None,  # raw BTC return on these bars
            "median_return": float(l.median()) if len(l) else None,
            # "win rate" = fraction where predicted direction (DOWN) is correct, i.e. actual return < 0
            "win_rate": float((l < 0).mean()) if len(l) else None,
            "short_pnl_mean": float((-l).mean()) if len(l) else None,  # mean of -return
        }
        # short_dom (shorts liquidated) → expect UP → strategy-P&L = +ret
        s = short_dom[col].dropna()
        per_hour["short_dominant"][f"{h}h"] = {
            "n": int(len(s)),
            "mean_return": float(s.mean()) if len(s) else None,
            "median_return": float(s.median()) if len(s) else None,
            "win_rate": float((s > 0).mean()) if len(s) else None,
            "long_pnl_mean": float(s.mean()) if len(s) else None,
        }
        # Combined strategy P&L: short when long-dom, long when short-dom.
        # Mathematically: pnl = -sign(liq_imb) × ret = -ret if imb>0 else +ret
        combined = pd.concat([-l, s])
        per_hour["contrarian_combined"][f"{h}h"] = {
            "n": int(len(combined)),
            "mean_signed_return": float(combined.mean()) if len(combined) else None,
            "median_signed_return": float(combined.median()) if len(combined) else None,
            "win_rate": float((combined > 0).mean()) if len(combined) else None,
            "mean_after_fees": (
                float(combined.mean() - ROUND_TRIP_FEE) if len(combined) else None
            ),
        }

    # 3. Fee sensitivity summary — is the contrarian strategy profitable
    # after fees, at each horizon?
    fee_ok = {}
    for h in hours:
        mean_after = per_hour["contrarian_combined"][f"{h}h"]["mean_after_fees"]
        fee_ok[f"{h}h_profitable_after_fees"] = (
            mean_after is not None and mean_after > 0
        )

    # 4. Verdict on reversion
    # Look at horizon with best after-fee expectancy
    after_fees = {
        h: per_hour["contrarian_combined"][f"{h}h"]["mean_after_fees"] or -1
        for h in hours
    }
    best_h = max(after_fees, key=after_fees.get)
    best_after = after_fees[best_h]
    if best_after > 0.001:
        rev_verdict = "PROFITABLE"
    elif best_after > 0:
        rev_verdict = "MARGINAL"
    else:
        rev_verdict = "UNPROFITABLE"

    result = {
        "description": "Cascade-bar reversion dynamics (magnitude, per-hour, fee-sensitivity)",
        "round_trip_fee": ROUND_TRIP_FEE,
        "magnitude": magnitude,
        "per_hour": per_hour,
        "fee_profitable_by_horizon": fee_ok,
        "best_horizon": best_h,
        "best_mean_after_fees": best_after,
        "reversion_verdict": rev_verdict,
    }
    out_path = OUT_DIR / "h2_reversion_dynamics.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("[Task 1.2] → %s", out_path)
    logger.info("  Magnitude: cascade median |ret_4h|=%.4f  vs non=%.4f  (ratio %s)",
                magnitude["cascade_bars"]["abs_return_4h_median"],
                magnitude["non_cascade_bars"]["abs_return_4h_median"],
                f'{magnitude["cascade_over_non_cascade_ratio_median"]:.2f}x' if magnitude["cascade_over_non_cascade_ratio_median"] else "—")
    for h in hours:
        c = per_hour["contrarian_combined"][f"{h}h"]
        logger.info("  contrarian +%dh: mean_signed_ret=%+.4f  WR=%.3f  after_fee=%+.4f",
                    h, c["mean_signed_return"], c["win_rate"], c["mean_after_fees"])
    logger.info("  Reversion verdict: %s (best horizon=%s after-fee=%+.4f)",
                rev_verdict, best_h, best_after)

    # 5. Plot per-hour dynamics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Left panel: raw BTC return in each sub-group. With H2 negative-IC,
    # long-dominant bars should drift NEGATIVE and short-dominant POSITIVE.
    ax1.plot(hours, [per_hour["long_dominant"][f"{h}h"]["mean_return"] for h in hours],
             "o-", label="Long-dominant cascade (liq_imb>0) mean BTC return",
             color="#ef5350")
    ax1.plot(hours, [per_hour["short_dominant"][f"{h}h"]["mean_return"] for h in hours],
             "o-", label="Short-dominant cascade (liq_imb<0) mean BTC return",
             color="#26a69a")
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_xlabel("Hours after cascade bar"); ax1.set_ylabel("Mean BTC return")
    ax1.set_title("Post-cascade BTC return by liq_imbalance side")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # Right: strategy P&L (short when long-dom, long when short-dom) = H2's implied strategy
    contra = [per_hour["contrarian_combined"][f"{h}h"]["mean_signed_return"] for h in hours]
    after = [per_hour["contrarian_combined"][f"{h}h"]["mean_after_fees"] for h in hours]
    ax2.plot(hours, contra, "o-", label="Strategy P&L (raw)", color="#26a69a")
    ax2.plot(hours, after, "s--", label=f"After round-trip fees (-{ROUND_TRIP_FEE*100:.2f}%)",
             color="#f9a825")
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.axhline(ROUND_TRIP_FEE, color="red", linewidth=0.5, linestyle=":",
                label="Fee hurdle")
    ax2.set_xlabel("Hours after cascade bar"); ax2.set_ylabel("Strategy return")
    ax2.set_title("Strategy P&L: pred_dir = -sign(liq_imb) on cascade bars")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = OUT_DIR / "h2_reversion_plot.png"
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    logger.info("  Plot → %s", plot_path)
    return result


def write_verdict(task1: dict, task2: dict) -> str:
    stab = task1["stability_verdict"]
    rev = task2["reversion_verdict"]
    if stab == "STABLE" and rev == "PROFITABLE":
        overall = "PROCEED_TO_STAGE_2"
        reason = "Monthly IC is consistently negative & significant; contrarian return beats fees."
    elif stab == "STABLE" and rev == "MARGINAL":
        overall = "PROCEED_TO_STAGE_2"
        reason = "IC is stable; edge is tight but positive after fees. Stage 2 will quantify modifier lift."
    elif stab == "MARGINAL" and rev in ("PROFITABLE", "MARGINAL"):
        overall = "PROCEED_TO_STAGE_2"
        reason = f"Stability is borderline (MARGINAL) but reversion is {rev}; worth testing as confidence modifier (not primary signal)."
    elif stab == "UNSTABLE":
        overall = "STOP"
        reason = "Monthly IC is not consistently negative. Edge is concentrated in a minority of months — not production-ready."
    elif rev == "UNPROFITABLE":
        overall = "STOP"
        reason = "Contrarian return does not survive round-trip fees at any horizon."
    else:
        overall = "STOP"
        reason = f"Stability={stab}, reversion={rev} — insufficient evidence."

    # Extract key numbers for the verdict doc
    global_ic = task1["global_ic"]
    global_ic_ci = task1["global_ic_ci95"]
    n_months_total = task1["stability_counts"]["n_months_with_data"]
    n_months_neg = task1["stability_counts"]["months_negative_ic"]
    n_months_neg_sig = task1["stability_counts"]["months_negative_and_ci_ge_0"]
    n_months_near_zero = task1["stability_counts"]["months_near_zero_ic"]

    mag = task2["magnitude"]
    rev_per_hour = task2["per_hour"]["contrarian_combined"]
    best_h = task2["best_horizon"]

    md = f"""# Cascade Modifier Study — Stage 1 Verdict

Generated: run of `research/cascade_modifier_study/stage1.py`.

## Monthly stability (Task 1.1)

- **Verdict**: `{stab}`
- Global cascade-bar Spearman IC: **{global_ic:+.3f}** (95% CI [{global_ic_ci[0]:+.3f}, {global_ic_ci[1]:+.3f}])
- Months with ≥ {task1["min_monthly_cascades_threshold"]} cascade bars: **{n_months_total}**
- Months with negative IC: {n_months_neg} / {n_months_total}
- Months with negative IC AND 95% CI upper < 0: **{n_months_neg_sig} / {n_months_total}**
- Months with IC in (-0.05, +0.05): {n_months_near_zero} / {n_months_total}

## Reversion dynamics (Task 1.2)

- **Verdict**: `{rev}`
- Best horizon (max after-fee expected return): **+{best_h}h**
- Mean contrarian signed return at best horizon: {rev_per_hour[f"{best_h}h"]["mean_signed_return"]:+.4f}
- After round-trip fee ({ROUND_TRIP_FEE*100:.2f}%): **{rev_per_hour[f"{best_h}h"]["mean_after_fees"]:+.4f}**
- Per-hour contrarian win rate:
"""
    for h in [1, 2, 3, 4]:
        md += (f"  - +{h}h: WR={rev_per_hour[f'{h}h']['win_rate']:.3f}, "
               f"mean_signed={rev_per_hour[f'{h}h']['mean_signed_return']:+.4f}, "
               f"after_fees={rev_per_hour[f'{h}h']['mean_after_fees']:+.4f}\n")

    md += f"""
- Magnitude comparison: cascade median |ret_4h| = {mag["cascade_bars"]["abs_return_4h_median"]:.4f}, non-cascade median = {mag["non_cascade_bars"]["abs_return_4h_median"]:.4f}
- Cascade-vs-non magnitude ratio (median): {mag.get("cascade_over_non_cascade_ratio_median", "n/a")}

## Overall verdict

- **`{overall}`**
- Reason: {reason}

## Implications for Stage 2

"""
    if overall == "PROCEED_TO_STAGE_2":
        md += ("Stage 2 will backtest the cascade confidence modifier on the production-"
               "equivalent walk-forward pipeline. The modifier uses `p_cascade >= 0.7` "
               "as the activation gate, boosts confidence by 15% when direction-reg and "
               f"contrarian align, and damps by 30% when they conflict. Best horizon "
               f"({best_h}h) suggests the modifier should be scored against the "
               f"production `path_ret_4h` target (which averages over all 4 hours) "
               "rather than endpoint return.\n")
    else:
        md += ("Stage 2 is **not warranted** under this Stage 1 evidence. Re-run this "
               "study when more data has accumulated (recommend +6 months), or "
               "investigate whether a tighter cascade definition (higher percentile "
               "threshold) yields a more stable edge.\n")

    out_path = OUT_DIR / "stage1_verdict.md"
    out_path.write_text(md, encoding="utf-8")
    logger.info("[Verdict] → %s", out_path)
    return overall


def main():
    logger.info("Building data...")
    data = build_data()
    logger.info("Rows: %d (after label horizon cutoff)", len(data))
    logger.info("Cascade bars: %d (%.1f%%)",
                int((data["surge"] == 1).sum()),
                100 * (data["surge"] == 1).mean())

    logger.info("=== Task 1.1: Monthly breakdown of H2 ===")
    task1 = task_1_1_monthly_breakdown(data)

    logger.info("=== Task 1.2: Reversion magnitude and timing ===")
    task2 = task_1_2_reversion_dynamics(data)

    overall = write_verdict(task1, task2)
    logger.info("\n>>> STAGE 1 OVERALL VERDICT: %s <<<", overall)
    return overall


if __name__ == "__main__":
    main()
