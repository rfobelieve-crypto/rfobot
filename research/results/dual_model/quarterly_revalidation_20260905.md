# Quarterly Re-Validation — 2026-09-05 01:00 UTC

Data: 3999 bars  2026-03-22 → 2026-09-05

## 1. AUC / IC ceiling
- sign_AUC = 0.5866  (baseline 0.59, band [0.55,0.62]) → OK
- Spearman IC = +0.1395  (baseline +0.16)
- n_oos = 3696

## 1b. Signal-to-noise — UNAVAILABLE: 'y'
## 2. Recent IC decay (concept-drift check)
  2026-04: IC=+0.150  (n=658)
  2026-05: IC=+0.147  (n=744)
  2026-06: IC=+0.172  (n=720)
  2026-07: IC=+0.197  (n=744)
  2026-08: IC=+0.084  (n=744)
  2026-09: IC=+0.383  (n=86)
- recent 60d IC = +0.151  vs older = +0.145  → OK

## 2b. Production output-level drift (rank metrics are blind to this)
- live 30d pred mean = +0.00026  (floor ±0.0008)  tails beyond floor: dn 22.0% / up 36.9%  → OK

## 3. Tier edge (sign-accuracy by decoded tier)
- Strong   : sign-acc=66.1%  (n=372)
- Moderate : sign-acc=64.1%  (n=719)

## 4. Regime breakdown (where the edge lives)
- BULL   : IC=+0.099  sign-acc=56.4%  (n=573)
- BEAR   : IC=+0.246  sign-acc=57.9%  (n=454)
- CHOPPY : IC=+0.135  sign-acc=55.7%  (n=2669)

## 5. Orthogonal-data scan (manual — the only path to a NEW edge)
Re-evaluate each channel's availability / cost / expected lift. v7 is
saturated on OHLCV+CG+Deribit+order-flow; a breakthrough needs a NEW
source, not more same-source features.
- [ ] Options gamma exposure (GEX) — Deribit/Glassnode paid; most-cited untested
- [ ] On-chain whale wallet flow — Glassnode/BGeometrics
- [ ] Bitcoin ETF AUM/flow — already wired (cg_etf_flow); re-check IC at daily res
- [ ] Cross-asset (SPX/DXY/Gold/US10Y) — SPX_return_1d already strongest cross feat
- [ ] Funding/basis term structure across venues
- [ ] Social/sentiment (Twitter/Reddit) DIY scraper

## VERDICT
**PASS** — edge is where it was; no structural drift detected. Keep running; re-check next quarter.