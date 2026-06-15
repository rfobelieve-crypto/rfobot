# Verify: v7.1 + Kernel (方式 C) vs Pure v7.1 Baseline

Generated: 2026-05-16 12:08 UTC

## Step 1 — Kernel implementation & validation

- Kernel formula: w[i] = (1 + i^2 / (h^2*2*r))^(-r), i=0..26; h=8 r=8 x=25
- Recomputed yhat1 on 4978 Binance 1h bars (2025-10-17 → 2026-05-12)
- Cross-check vs deployed LDC kernel (ldc_signals_full.parquet), 4921 overlapping bars:
    max abs diff = 0.000e+00   mean abs diff = 0.000e+00
    kernel-state agreement = 99.98% (4920/4921)
- Numeric tolerance (<0.001 vs deployed kernel): PASS
- NOTE: a manual TradingView cross-check (5 reference bars) is still required to certify against the live Pine Script; this script validates the Python port is reproducible and identical to the kernel already running in production LDC swing.

## Step 2 — Kernel state on v7.1 OOS timeframe

- v7.1 OOS span: 2025-11-27 04:00 → 2026-04-30 03:00  (3696 bars)
- bull:     1823  (49.3%)
- bear:     1873  (50.7%)
- neutral:     0  (0.0%)

## Step 3 — v7.1 signal decode & kernel confluence filter

- Total v7.1 signals (Strong+Moderate, UP+DOWN): 1112
- Kept (kernel confluence):   489  (44.0%)
- Filtered out:               623  (56.0%)

  By tier:
    Strong   :  417 signals, 257 filtered (61.6%), 160 kept
    Moderate :  695 signals, 366 filtered (52.7%), 329 kept
  By direction:
    UP       :  589 signals, 348 filtered (59.1%), 241 kept
    DOWN     :  523 signals, 275 filtered (52.6%), 248 kept

## Step 4-5 — 方式 C backtest & metrics

**方式 C (kernel-confluence entry + kernel/stop/72h exit):**
  n=489  WR=28.4%  net=+15.5bps  gross=+23.5bps  |ret|=108.9bps  hold=11.1h  cum=+10.43%  Sharpe=0.85  MDD=14.41%

  Exit-reason breakdown:
    hard_stop  : n=104  WR=0.0%  net=-106.0bps  gross=-98.0bps  |ret|=106.0bps  hold=5.2h  cum=-42.48%  Sharpe=-10.12  MDD=42.82%
    kernel_flip: n=385  WR=36.1%  net=+48.3bps  gross=+56.3bps  |ret|=109.6bps  hold=12.7h  cum=+37.64%  Sharpe=2.42  MDD=10.94%

  By v7.1 tier:
    Strong   : n=160  WR=28.1%  net=-4.8bps  gross=+3.2bps  |ret|=86.3bps  hold=10.3h  cum=+2.21%  Sharpe=0.33  MDD=9.26%
    Moderate : n=329  WR=28.6%  net=+25.3bps  gross=+33.3bps  |ret|=119.8bps  hold=11.5h  cum=+9.65%  Sharpe=0.81  MDD=14.84%
  By direction:
    UP       : n=241  WR=22.4%  net=-18.6bps  gross=-10.6bps  |ret|=92.0bps  hold=9.6h  cum=-3.98%  Sharpe=-0.45  MDD=16.41%
    DOWN     : n=248  WR=34.3%  net=+48.6bps  gross=+56.6bps  |ret|=125.2bps  hold=12.6h  cum=+13.01%  Sharpe=1.17  MDD=9.44%

## Step 6 — Pure v7.1 baseline (4h fixed exit, same period)

**v7.1 baseline (all Strong+Moderate signals, 4h fixed hold):**
  n=1112  WR=60.9%  net=+30.0bps  gross=+38.0bps  |ret|=81.7bps  hold=4.0h  cum=+99.47%  Sharpe=5.36  MDD=7.07%

  By v7.1 tier:
    Strong   : n=417  WR=65.7%  net=+48.4bps  gross=+56.4bps  |ret|=89.0bps  hold=4.0h  cum=+89.34%  Sharpe=5.84  MDD=3.94%
    Moderate : n=695  WR=58.0%  net=+18.9bps  gross=+26.9bps  |ret|=77.4bps  hold=4.0h  cum=+33.47%  Sharpe=2.61  MDD=13.38%
  By direction:
    UP       : n=589  WR=58.2%  net=+20.9bps  gross=+28.9bps  |ret|=86.8bps  hold=4.0h  cum=+7.08%  Sharpe=0.72  MDD=19.16%
    DOWN     : n=523  WR=63.9%  net=+40.2bps  gross=+48.2bps  |ret|=76.0bps  hold=4.0h  cum=+67.33%  Sharpe=6.38  MDD=3.34%

### Side-by-side comparison

| Metric | v7.1 baseline | 方式 C | Delta |
|---|---|---|---|
| Total trades | 1112 | 489 | -623 |
| Win rate % | 0.609 | 0.284 | -0.325 |
| Avg net /trade (bps) | 30.0 | 15.5 | -14.5 |
| Avg gross /trade (bps) | 38.0 | 23.5 | -14.5 |
| Sharpe (annualised) | 5.36 | 0.85 | -4.51 |
| Max drawdown % | 7.07 | 14.41 | +7.35 |
| Cum net % | 99.47 | 10.43 | -89.04 |
| Avg holding (h) | 4.0 | 11.1 | +7.1 |

## Step 7 — Decision

- Sharpe: 5.36 → 0.85  (-84.2%)
- Win rate: 60.9% → 28.4%  (-32.5pp)
- Net/trade: +30.0 → +15.5 bps

**DECISION: C — 方式 C worse than baseline. Verification failed; document failure mode, stop here.**

### Failure-mode decomposition

Three trade sets, identical accounting:
- [1] baseline ALL signals (4h exit):   n=1112  WR=60.9%  net=+30.0bps  gross=+38.0bps  |ret|=81.7bps  hold=4.0h  cum=+99.47%  Sharpe=5.36  MDD=7.07%
- [2] baseline KEPT signals (4h exit):  n=489  WR=54.6%  net=+23.3bps  gross=+31.3bps  |ret|=85.5bps  hold=4.0h  cum=+25.04%  Sharpe=2.14  MDD=12.93%
- [3] 方式 C   KEPT signals (kernel exit): n=489  WR=28.4%  net=+15.5bps  gross=+23.5bps  |ret|=108.9bps  hold=11.1h  cum=+10.43%  Sharpe=0.85  MDD=14.41%

- **Kernel FILTER effect** ([2] vs [1], same 4h exit): net +30.0 → +23.3 bps, WR 60.9% → 54.6%, Sharpe 5.36 → 2.14
- **Kernel EXIT effect** ([3] vs [2], same kept signals): net +23.3 → +15.5 bps, WR 54.6% → 28.4%, Sharpe 2.14 → 0.85

