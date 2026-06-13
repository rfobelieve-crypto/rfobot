# Cascade Modifier Study — Stage 1 Verdict

Generated: run of `research/cascade_modifier_study/stage1.py`.

## Monthly stability (Task 1.1)

- **Verdict**: `MARGINAL`
- Global cascade-bar Spearman IC: **-0.254** (95% CI [-0.331, -0.160])
- Months with ≥ 15 cascade bars: **6**
- Months with negative IC: 6 / 6
- Months with negative IC AND 95% CI upper < 0: **2 / 6**
- Months with IC in (-0.05, +0.05): 0 / 6

## Reversion dynamics (Task 1.2)

- **Verdict**: `PROFITABLE`
- Best horizon (max after-fee expected return): **+4h**
- Mean contrarian signed return at best horizon: +0.0043
- After round-trip fee (0.08%): **+0.0035**
- Per-hour contrarian win rate:
  - +1h: WR=0.578, mean_signed=+0.0022, after_fees=+0.0014
  - +2h: WR=0.604, mean_signed=+0.0031, after_fees=+0.0023
  - +3h: WR=0.614, mean_signed=+0.0040, after_fees=+0.0032
  - +4h: WR=0.614, mean_signed=+0.0043, after_fees=+0.0035

- Magnitude comparison: cascade median |ret_4h| = 0.0134, non-cascade median = 0.0040
- Cascade-vs-non magnitude ratio (median): 3.3377299468448838

## Overall verdict

- **`PROCEED_TO_STAGE_2`**
- Reason: Stability is borderline (MARGINAL) but reversion is PROFITABLE; worth testing as confidence modifier (not primary signal).

## Implications for Stage 2

Stage 2 will backtest the cascade confidence modifier on the production-equivalent walk-forward pipeline. The modifier uses `p_cascade >= 0.7` as the activation gate, boosts confidence by 15% when direction-reg and contrarian align, and damps by 30% when they conflict. Best horizon (4h) suggests the modifier should be scored against the production `path_ret_4h` target (which averages over all 4 hours) rather than endpoint return.
