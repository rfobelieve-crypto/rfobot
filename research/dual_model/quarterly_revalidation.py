"""Quarterly re-validation ritual — "is the world still the same?"

Run this every ~3 months. It answers whether the v7 edge is still where it was,
whether it is decaying recently, and reminds you to scan for newly-available
orthogonal data. The system's longevity comes from this ritual, not from any
model being eternal (see CLAUDE.md / memory on edge decay).

It does NOT change anything. It prints a PASS / DRIFT verdict + writes a dated
report. Reuses the production training + decode so numbers are canonical.

Checks:
  1. AUC/IC ceiling      — current walk-forward sign_AUC + Spearman IC vs the
                            documented baseline (AUC ~0.59, IC ~0.16, ceiling
                            band 0.54-0.62). Below band = decay; above = new
                            signal or leak — both warrant investigation.
  2. Recent IC decay     — monthly IC trend; recent 60d vs older. Recent << old
                            = concept drift (the Feb/Mar Mag-halving pattern).
  3. Tier edge intact    — Strong vs Moderate sign-accuracy still separated and
                            Strong still > ~60%.
  4. Regime breakdown    — where the edge is alive/dead (bull/bear/chop).
  5. Orthogonal-data scan — manual checklist (the only path to a NEW edge once
                            the current data sources saturate).

Usage:  python -m research.dual_model.quarterly_revalidation
"""
from __future__ import annotations

import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "research"))

from research.dual_model.shared_data import load_and_cache_data, RESULTS_DIR
from research.dual_model.direction_features_v2 import FULL_DIRECTION
from research.dual_model.train_direction_reg_4h import train_direction_reg_walk_forward
from verify_kernel_method_c import decode_tiers   # rolling-percentile = live decode

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Documented baselines (update here if the edge legitimately re-baselines) ──
BASELINE_AUC = 0.59          # canonical V7 OOS sign_AUC
BASELINE_IC = 0.16           # canonical V7 OOS Spearman IC
AUC_FLOOR = 0.55             # below → edge decaying, investigate
AUC_CEIL = 0.62              # above → new signal OR leak, investigate
STRONG_WR_FLOOR = 0.60       # Strong tier sign-acc should stay above this
RECENT_DAYS = 60             # "recent" window for the decay check

# Orthogonal-data scan: the only breakthrough path once OHLCV+CG+Deribit+order
# flow saturates (mistake.md 2026-06-02). Re-evaluate availability/cost each run.
ORTHO_CHANNELS = [
    "Options gamma exposure (GEX) — Deribit/Glassnode paid; most-cited untested",
    "On-chain whale wallet flow — Glassnode/BGeometrics",
    "Bitcoin ETF AUM/flow — already wired (cg_etf_flow); re-check IC at daily res",
    "Cross-asset (SPX/DXY/Gold/US10Y) — SPX_return_1d already strongest cross feat",
    "Funding/basis term structure across venues",
    "Social/sentiment (Twitter/Reddit) DIY scraper",
]


def _sign_acc(pred: np.ndarray, y: np.ndarray) -> float:
    m = (np.isfinite(pred) & np.isfinite(y) & (y != 0))
    if m.sum() < 5:
        return float("nan")
    return float((np.sign(pred[m]) == np.sign(y[m])).mean())


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []

    def log(s: str = ""):
        lines.append(s)
        print(s)

    log(f"# Quarterly Re-Validation — {stamp}\n")
    df = load_and_cache_data(limit=4000)
    log(f"Data: {len(df)} bars  {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}\n")

    # ── 1. AUC/IC ceiling ────────────────────────────────────────────────
    logger.info("Running canonical walk-forward (FULL_DIRECTION)…")
    oos, metrics, _ = train_direction_reg_walk_forward(
        df, FULL_DIRECTION, objective="mse")
    auc, ic = metrics["auc_sign"], metrics["spearman_ic"]
    flags = []
    auc_state = ("OK" if AUC_FLOOR <= auc <= AUC_CEIL
                 else ("DECAY (below floor)" if auc < AUC_FLOOR
                       else "ABOVE CEIL (new signal or leak — investigate)"))
    if auc_state != "OK":
        flags.append(f"AUC {auc:.3f} → {auc_state}")
    log("## 1. AUC / IC ceiling")
    log(f"- sign_AUC = {auc:.4f}  (baseline {BASELINE_AUC}, band [{AUC_FLOOR},{AUC_CEIL}]) → {auc_state}")
    log(f"- Spearman IC = {ic:+.4f}  (baseline {BASELINE_IC:+.2f})")
    log(f"- n_oos = {len(oos)}\n")

    # ── 2. Recent IC decay ───────────────────────────────────────────────
    log("## 2. Recent IC decay (concept-drift check)")
    by_month = []
    for mlabel, g in oos.groupby(oos.index.to_period("M")):
        if len(g) >= 30:
            r = spearmanr(g["pred_ret"], g["y_path_ret_4h"]).correlation
            by_month.append((str(mlabel), float(r), len(g)))
    for ml, r, ng in by_month:
        log(f"  {ml}: IC={r:+.3f}  (n={ng})")
    cutoff = oos.index.max() - pd.Timedelta(days=RECENT_DAYS)
    recent, older = oos[oos.index >= cutoff], oos[oos.index < cutoff]
    ic_recent = (spearmanr(recent["pred_ret"], recent["y_path_ret_4h"]).correlation
                 if len(recent) > 30 else float("nan"))
    ic_older = (spearmanr(older["pred_ret"], older["y_path_ret_4h"]).correlation
                if len(older) > 30 else float("nan"))
    decay = (np.isfinite(ic_recent) and np.isfinite(ic_older)
             and ic_recent < 0.5 * ic_older)
    if decay:
        flags.append(f"recent IC {ic_recent:+.3f} < half of older {ic_older:+.3f}")
    log(f"- recent {RECENT_DAYS}d IC = {ic_recent:+.3f}  vs older = {ic_older:+.3f}  "
        f"→ {'DRIFT — recent edge halved' if decay else 'OK'}\n")

    # ── 3. Tier edge intact ──────────────────────────────────────────────
    log("## 3. Tier edge (sign-accuracy by decoded tier)")
    dec = decode_tiers(oos["pred_ret"])
    dec = dec.join(oos["y_path_ret_4h"], how="inner")
    tier_wr = {}
    for tier in ("Strong", "Moderate"):
        sub = dec[dec["tier"] == tier]
        if len(sub):
            # directional: pred sign vs realized sign
            wr = _sign_acc(sub["pred_ret"].values, sub["y_path_ret_4h"].values)
            tier_wr[tier] = (wr, len(sub))
            log(f"- {tier:9s}: sign-acc={wr*100:.1f}%  (n={len(sub)})")
    sw = tier_wr.get("Strong", (float("nan"), 0))
    if np.isfinite(sw[0]) and sw[0] < STRONG_WR_FLOOR:
        flags.append(f"Strong sign-acc {sw[0]*100:.1f}% < floor {STRONG_WR_FLOOR*100:.0f}%")
    log("")

    # ── 4. Regime breakdown ──────────────────────────────────────────────
    log("## 4. Regime breakdown (where the edge lives)")
    if {"is_trending_bull", "is_trending_bear"}.issubset(df.columns):
        reg = pd.Series("CHOPPY", index=df.index)
        reg[df["is_trending_bull"] == 1] = "BULL"
        reg[df["is_trending_bear"] == 1] = "BEAR"
        j = oos.join(reg.rename("regime"), how="left")
        for rname in ("BULL", "BEAR", "CHOPPY"):
            sub = j[j["regime"] == rname]
            if len(sub) >= 20:
                r = spearmanr(sub["pred_ret"], sub["y_path_ret_4h"]).correlation
                wr = _sign_acc(sub["pred_ret"].values, sub["y_path_ret_4h"].values)
                log(f"- {rname:7s}: IC={r:+.3f}  sign-acc={wr*100:.1f}%  (n={len(sub)})")
    else:
        log("- (regime columns not in feature frame — skipped)")
    log("")

    # ── 5. Orthogonal-data scan (manual) ─────────────────────────────────
    log("## 5. Orthogonal-data scan (manual — the only path to a NEW edge)")
    log("Re-evaluate each channel's availability / cost / expected lift. v7 is")
    log("saturated on OHLCV+CG+Deribit+order-flow; a breakthrough needs a NEW")
    log("source, not more same-source features.")
    for c in ORTHO_CHANNELS:
        log(f"- [ ] {c}")
    log("")

    # ── Verdict ──────────────────────────────────────────────────────────
    log("## VERDICT")
    if not flags:
        log("**PASS** — edge is where it was; no structural drift detected. "
            "Keep running; re-check next quarter.")
    else:
        log("**DRIFT / INVESTIGATE** — one or more checks flagged:")
        for f in flags:
            log(f"  - {f}")
        log("\nResponse protocol (do NOT improvise):")
        log("  1. Confirm it's drift not a data bug (check mistake.md, recent backfills).")
        log("  2. If real decay: per-fold-sane retrain on recent window; if that")
        log("     doesn't recover, de-stage / tighten until re-validated.")
        log("  3. Do NOT add leverage to compensate a decaying edge.")
        log("  4. If ceiling exceeded (AUC>ceil): suspect leak before celebrating.")

    out = RESULTS_DIR / f"quarterly_revalidation_{datetime.now(timezone.utc):%Y%m%d}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nReport → {out}")

    # Best-effort Telegram push so a scheduled run actually surfaces.
    # Plain text (no Markdown specials) to avoid parse errors.
    try:
        import os
        from indicator.okx.alerter import send_critical
        chat = (os.environ.get("TG_ALERT_CHAT_ID")
                or os.environ.get("TG_CRITICAL_CHAT_ID") or "")
        verdict = "PASS" if not flags else "DRIFT/INVESTIGATE"
        msg = (f"Monthly re-validation {stamp}\n"
               f"Verdict: {verdict}\n"
               f"AUC {auc:.3f} (band 0.55-0.62) | IC {ic:+.3f}\n"
               f"recent60d IC {ic_recent:+.3f} vs older {ic_older:+.3f}\n"
               f"Strong sign-acc {sw[0]*100:.0f}pct")
        if flags:
            msg += "\nflags: " + " ; ".join(flags)
        if chat:
            send_critical(chat, msg)
    except Exception:
        logger.exception("revalidation_telegram_push_failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
