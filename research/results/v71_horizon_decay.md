# V7.1 Horizon Decay Curve

Generated: 2026-05-16 12:15 UTC

v7.1 OOS span: 2025-11-27 -> 2026-04-30  (3696 bars)
v7.1 signals (Strong+Moderate, UP+DOWN): 1112  (Strong 417, Moderate 695)

## HORIZON view — pure close-to-close return at H (clean alpha decay)

| H (h) | n signals | WR | avg ret/trade (bps) | IC full-sample | p-value |
|---|---|---|---|---|---|
| 4 | 1112 | 64.2% | +32.0 | 0.1748 | 9.59e-27 |
| 6 | 1112 | 64.2% | +42.0 | 0.1825 | 4.78e-29 |
| 8 | 1112 | 61.5% | +42.3 | 0.1561 | 1.40e-21 |
| 12 | 1112 | 59.9% | +45.1 | 0.1290 | 3.43e-15 |
| 24 | 1112 | 62.4% | +72.1 | 0.1537 | 5.73e-21 |

IC = Spearman(pred_ret, signed forward H-bar return) over the full 3696-bar OOS sample. avg ret is GROSS (round-trip fee ~8bps for scale).

## BARRIER view — cached TP50/SL30 labels (confounded, see header)

| H (h) | barrier WR | barrier avg ret/trade (bps) |
|---|---|---|
| 4 | 42.4% | +2.6 |
| 6 | 41.7% | +2.7 |
| 8 | 41.5% | +2.8 |
| 12 | 41.2% | +2.8 |
| 24 | 40.6% | +2.5 |

Confounded: fixed TP50/SL30 means a longer window only changes how often a barrier is reached; with SL(30) closer than TP(50), WR drifts down with H for geometric reasons. NOT a clean alpha-decay signal.

## Signal-level conviction IC

| H (h) | IC(|pred|, signed realized return) |
|---|---|
| 4 | 0.1454 |
| 6 | 0.1350 |
| 8 | 0.1082 |
| 12 | 0.0841 |
| 24 | 0.1194 |

## Alpha time-structure

- Peak horizon (max |IC|): **H=6**, IC=0.1825
- IC at 4h (current fixed exit): 0.1748
- Half-life: |IC| never drops to 50% of peak within H<=24 (min |IC| past peak = 0.1290)
- Alpha past 4h: max |IC| for H>4 is 0.1825 = 104% of the 4h IC (persists/grows past 4h)

Plot saved -> C:\Users\rfo\Desktop\flowbot\flow_system\research\results\v71_horizon_decay.png
