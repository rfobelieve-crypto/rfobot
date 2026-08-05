# Quarterly Re-Validation — 2026-08-05 01:00 UTC

Data: 3999 bars  2026-02-19 → 2026-08-05

## 1. AUC / IC ceiling
- sign_AUC = 0.5848  (baseline 0.59, band [0.55,0.62]) → OK
- Spearman IC = +0.1377  (baseline +0.16)
- n_oos = 3696

## 2. Recent IC decay (concept-drift check)
  2026-03: IC=+0.092  (n=682)
  2026-04: IC=+0.189  (n=720)
  2026-05: IC=+0.166  (n=744)
  2026-06: IC=+0.112  (n=720)
  2026-07: IC=+0.196  (n=744)
  2026-08: IC=+0.148  (n=86)
- recent 60d IC = +0.153  vs older = +0.148  → OK

## 3. Tier edge (sign-accuracy by decoded tier)
- Strong   : sign-acc=70.1%  (n=472)
- Moderate : sign-acc=58.5%  (n=702)

## 4. Regime breakdown (where the edge lives)
- BULL   : IC=+0.082  sign-acc=52.4%  (n=477)
- BEAR   : IC=+0.060  sign-acc=50.3%  (n=314)
- CHOPPY : IC=+0.161  sign-acc=56.1%  (n=2905)

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