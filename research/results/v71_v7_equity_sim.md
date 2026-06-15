# V7 + 3×ATR Trailing Stop — Equity Simulation (1000u, 5×)

Generated: 2026-05-16 12:58 UTC

**Simulation model:** 1000 USDT, ONE position at a time, sequential, compounding, 5× leverage. Each trade margin = full current equity. Fee 0.08% round-trip on notional. Funding NOT modelled.
OOS span: 2025-11-27 → 2026-04-30
v7.1 signals available: 1112 (one-position sim consumes a subset — see n below)

Median 3×ATR trailing distance ≈ $1,641 (2.13% of price) → at 5× a stop-out ≈ −10.7% of equity.

## Headline — V7 + 3×ATR trailing stop

    n=169  WR=56.2%  eq/trade=+2.52%  hold=13.8h  final=$30,240  ROI=+2924.0%  Sharpe=5.49  MaxDD=35.2%  liq=0

## Reference — V0 fixed 4h (same account model, no stop)

    n=324  WR=60.5%  eq/trade=+1.05%  hold=4.0h  final=$16,618  ROI=+1561.8%  Sharpe=4.77  MaxDD=34.0%  liq=0

## Comparison

| Metric | V7 + 3×ATR trail | V0 fixed 4h |
|---|--:|--:|
| Trades taken | 169 | 324 |
| Win rate | 56.2% | 60.5% |
| Avg equity return / trade | +2.52% | +1.05% |
| Median equity return / trade | +0.74% | +0.69% |
| Best / worst trade | +48.2% / -26.0% | +22.3% / -37.6% |
| Avg holding | 13.8h | 4.0h |
| **Final equity** | **$30,240** | $16,618 |
| **ROI** | **+2924.0%** | +1561.8% |
| Sharpe (daily, ann.) | 5.49 | 4.77 |
| Max drawdown | 35.2% | 34.0% |
| Liquidations | 0 | 0 |

## V7 exit-reason breakdown

| Reason | n | share | avg equity ret/trade | WR |
|---|--:|--:|--:|--:|
| opp_signal | 99 | 58.6% | +4.35% | 73.7% |
| time_cap | 3 | 1.8% | +33.62% | 100.0% |
| trail_stop | 67 | 39.6% | -1.58% | 28.4% |

## V7 regime breakdown (regime at entry)

| Regime | n | WR | avg equity ret/trade | cum equity factor |
|---|--:|--:|--:|--:|
| BULL | 25 | 48.0% | +2.74% | ×1.76 |
| BEAR | 34 | 61.8% | +6.13% | ×5.06 |
| CHOPPY | 110 | 56.4% | +1.35% | ×3.39 |

Equity plot saved → C:\Users\rfo\Desktop\flowbot\flow_system\research\results\v71_v7_equity_sim.png
