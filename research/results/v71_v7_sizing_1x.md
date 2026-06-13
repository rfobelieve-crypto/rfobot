# V7 + 3×ATR Trailing Stop — Deployable Sizing (2% risk, 1× cap)

Generated: 2026-05-16 13:03 UTC

**Sizing:** fixed-fractional, 2% equity risk per trade, leverage capped at 1×, compounding. 1000 USDT start. Fee 0.08% round-trip on notional; funding NOT modelled. One position at a time.

## Headline — V7 + 3×ATR trailing, 2%-risk 1× sizing

- Trades: 169   Win rate: 56.2%
- Avg equity return/trade: +0.38%   (median +0.12%)
- Best / worst trade: +6.0% / -2.0%
- Avg holding: 13.8h
- **Final equity: $1,849  →  ROI +84.9%** (5-month OOS)
- Sharpe (daily, annualised): 5.10
- **Max drawdown: 5.3%**
- Avg position size: 88% of equity   (1× cap bound on 45% of trades)
- Avg loss on losing trades: -0.82% (risk budget is −2%; trailing stop often exits tighter)

## Comparison (both 1× — V0 is a rough reference, flat 1×, no stop)

| Metric | V7 2%-risk 1× | V0 flat 1× |
|---|--:|--:|
| Trades | 169 | 324 |
| Win rate | 56.2% | 60.5% |
| Avg equity ret/trade | +0.38% | +0.21% |
| Best / worst trade | +6.0% / -2.0% | +4.5% / -7.5% |
| Final equity | $1,849 | $1,933 |
| ROI (5 months) | +84.9% | +93.3% |
| Sharpe | 5.10 | 4.88 |
| Max drawdown | 5.3% | 6.8% |

## V7 exit-reason breakdown

| Reason | n | share | avg equity ret/trade | WR |
|---|--:|--:|--:|--:|
| opp_signal | 99 | 58.6% | +0.72% | 73.7% |
| time_cap | 3 | 1.8% | +3.34% | 100.0% |
| trail_stop | 67 | 39.6% | -0.26% | 28.4% |

## V7 regime breakdown (regime at entry)

| Regime | n | WR | avg equity ret/trade |
|---|--:|--:|--:|
| BULL | 25 | 48.0% | +0.36% |
| BEAR | 34 | 61.8% | +0.86% |
| CHOPPY | 110 | 56.4% | +0.23% |

## Caveats

- WF-OOS backtest; live degrades (CLAUDE.md: live WR 69%→60%).
- n is small (one-position subsample); regime splits are indicative.
- Funding cost omitted — real ~14h holds pay ~1-2 funding intervals.
- Project is at Stage 1 (paper trading). This is a sizing study, not a deploy signal; live execution still goes through the staged plan.
- Worst trade reflects a gap through the trailing stop; at 1× this is survivable (unlike the 5× run where it cost −26% of the account).

Equity plot saved → C:\Users\rfo\Desktop\flowbot\flow_system\research\results\v71_v7_sizing_1x.png
