# Quarterly Re-Validation — 2026-06-09 09:44 UTC

Data: 3999 bars  2025-12-24 → 2026-06-09

## 1. AUC / IC ceiling
- sign_AUC = 0.5868  (baseline 0.59, band [0.55,0.62]) → OK
- Spearman IC = +0.1639  (baseline +0.16)
- n_oos = 3696

## 2. Recent IC decay (concept-drift check)
  2026-01: IC=+0.209  (n=626)
  2026-02: IC=+0.212  (n=672)
  2026-03: IC=+0.114  (n=744)
  2026-04: IC=+0.201  (n=720)
  2026-05: IC=+0.173  (n=744)
  2026-06: IC=+0.091  (n=190)
- recent 60d IC = +0.171  vs older = +0.158  → OK

## 3. Tier edge (sign-accuracy by decoded tier)
- Strong   : sign-acc=71.1%  (n=477)
- Moderate : sign-acc=61.3%  (n=727)

## 4. Regime breakdown (where the edge lives)
- BULL   : IC=+0.120  sign-acc=56.7%  (n=668)
- BEAR   : IC=+0.204  sign-acc=57.1%  (n=737)
- CHOPPY : IC=+0.161  sign-acc=55.8%  (n=2291)

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