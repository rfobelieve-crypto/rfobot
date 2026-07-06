# Quarterly Re-Validation — 2026-07-05 15:47 UTC

Data: 3999 bars  2026-01-20 → 2026-07-05

## 1. AUC / IC ceiling
- sign_AUC = 0.5988  (baseline 0.59, band [0.55,0.62]) → OK
- Spearman IC = +0.1769  (baseline +0.16)
- n_oos = 3696

## 2. Recent IC decay (concept-drift check)
  2026-02: IC=+0.185  (n=668)
  2026-03: IC=+0.156  (n=744)
  2026-04: IC=+0.243  (n=720)
  2026-05: IC=+0.100  (n=744)
  2026-06: IC=+0.178  (n=720)
  2026-07: IC=+0.204  (n=100)
- recent 60d IC = +0.150  vs older = +0.204  → OK

## 3. Tier edge (sign-accuracy by decoded tier)
- Strong   : sign-acc=72.3%  (n=361)
- Moderate : sign-acc=61.7%  (n=700)

## 4. Regime breakdown (where the edge lives)
- BULL   : IC=+0.119  sign-acc=53.6%  (n=472)
- BEAR   : IC=+0.196  sign-acc=58.2%  (n=433)
- CHOPPY : IC=+0.187  sign-acc=56.7%  (n=2791)

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