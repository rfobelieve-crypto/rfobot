# Cascade Modifier Study

**Status: ABANDONED — do not integrate into production.**

## Question

Can the H2 finding (Spearman IC = −0.252 between `cg_liq_imbalance` and
future 4h return, restricted to cascade bars) be turned into a
confidence modifier that lifts the direction-reg Strong-tier win rate?

## Answer

**No.** The ex-post statistical edge does not transfer to an ex-ante
tradeable signal under the modifier framing.

## Summary

- **Stage 1** (`stage1.py`): H2 verified on monthly breakdown (6/6 months
  negative IC) and reversion dynamics (PROFITABLE after 0.08% round-trip
  fees at 4h horizon). Verdict: `PROCEED_TO_STAGE_2`.
- **Stage 2** (`stage2.py`): walk-forward backtest on 3515 OOS bars with
  rolling top-5% `p_cascade` gate. Modifier active on 32 bars. Strong
  tier WR went from 65.8% → 65.3% (−0.5pp). On active bars, the
  aligned-vs-conflicted WR gap was **reversed** (aligned 44.4% vs
  conflicted 64.3%) — modifier boosts the wrong side. Verdict:
  `DO NOT ADOPT`.

## Why it failed

H2's edge lives on bars where cascade *actually* occurred (ex-post).
The modifier activates on bars where the cascade classifier *predicts*
cascade in its top-5% confidence band (ex-ante). Classifier AUC is
0.708 — that gap between ex-post and ex-ante is enough to flip the
contrarian edge. Ex-post IC ≠ ex-ante tradeable IC.

## Artifacts

| File | Content |
|---|---|
| `stage1.py` | Code for H2 monthly stability & reversion dynamics |
| `stage2.py` | Code for WF backtest of the confidence modifier |
| `h2_monthly.json` | Per-month cascade IC table (stability analysis) |
| `h2_reversion_dynamics.json` | Per-hour reversion magnitude + fee-sensitivity |
| `h2_reversion_plot.png` | Visualization of +1h to +4h contrarian P&L |
| `stage1_verdict.md` | Human-readable Stage 1 verdict (MARGINAL / PROFITABLE → PROCEED) |
| `stage2_backtest.json` | All Stage 2 numbers (tier counts, WR, CIs, subgroup breakdown) |
| `stage2_summary.md` | Human-readable Stage 2 summary (→ DO NOT ADOPT) |

## Future directions (if someone revisits)

If this line is ever reopened, the failure mode points to two possible
paths that avoid ex-post/ex-ante mismatch:

1. Train a **cascade-conditional direction sub-model** — a dedicated
   regressor fit only on bars flagged by the cascade classifier, with
   its own features (not just `liq_imbalance`). That way the model
   directly learns what works on the ex-ante cascade distribution,
   rather than inheriting an ex-post rule.
2. Evaluate H2 again on a **higher-precision cascade definition** —
   e.g. cascade only when *realized* liquidation exceeds trailing p95
   rather than p90, so that the ex-ante/ex-post gap narrows. Smaller
   sample but cleaner signal.

Neither is on the current roadmap. Parking.
