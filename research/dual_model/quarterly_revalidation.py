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
STALE_HOURS = 48             # feature data older than this → verdict untrustworthy

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

    # Fail-loud staleness guard. 2026-07-05: DNS was down at the scheduled
    # 09:00 run — the auto-backfill failure was logged "non-critical" and the
    # ritual silently graded a 16-day-old cache as PASS. A verdict computed on
    # stale data must never present itself as a fresh PASS/DRIFT.
    data_end = df.index[-1]
    data_age_h = (pd.Timestamp.now(tz=data_end.tz) - data_end).total_seconds() / 3600.0
    stale = data_age_h > STALE_HOURS
    if stale:
        log(f"**DATA STALE** — last bar {data_end:%Y-%m-%d %H:%M} is {data_age_h:.0f}h old "
            f"(threshold {STALE_HOURS}h). Backfill failed (network?); metrics below run on "
            f"an old cache and the verdict is NOT trustworthy.\n")

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

    # ── 1b. Signal-to-noise ratio, WITH its null floor ───────────────────
    # SNR = Var(mu)/Var(eps) = R²/(1-R²), computed straight from the
    # pred-vs-realised correlation (no latent-variable estimation needed
    # when you have the predictions). Reported ONLY alongside a shuffle
    # null: an SNR without its floor is worse than no number — the rolling
    # mean estimator commonly recommended for this returns ~1/(k+1) on pure
    # noise (4.8% at k=20), which is larger than this edge. Registered
    # 2026-09-02; see research/snr_monitor.py for the full rationale.
    # This is a TIMING diagnostic (its numerator is the variance of the
    # conditional expectation), never a sizing input.
    try:
        from research.snr_monitor import snr_from_corr, shuffle_null
        _p = oos["pred_ret"].to_numpy(float)
        _y = oos["y"].to_numpy(float)
        _snr_p = snr_from_corr(float(np.corrcoef(_p, _y)[0, 1]))
        _snr_s = snr_from_corr(float(pd.Series(_p).corr(pd.Series(_y),
                                                        method="spearman")))
        _null = shuffle_null(_p, _y, n=200)
        log("## 1b. Signal-to-noise (with shuffle null)")
        log(f"- SNR(Pearson)  = {_snr_p*100:.3f}%")
        log(f"- SNR(Spearman) = {_snr_s*100:.3f}%")
        log(f"- shuffle null: mean {_null['mean']*100:.3f}% / "
            f"p95 {_null['p95']*100:.3f}% / p99 {_null['p99']*100:.3f}%")
        if max(_snr_p, _snr_s) <= _null["p99"]:
            log("- → BELOW the shuffle null's p99 — this reading is not "
                "evidence on its own")
            flags.append("SNR at/below shuffle null p99")
        else:
            log("- → above the null, edge present but thin (expected: this "
                "is a 4h direction edge, not a mispricing)")
        log("")
    except Exception as _e:
        log("## 1b. Signal-to-noise — UNAVAILABLE: " + str(_e)[:100])

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

    # ── 2b. PRODUCTION output-level drift (rank-blind failure mode) ─────
    # 2026-08-08: the 05-01 model's live pred MEAN drifted +0.0024 in four
    # months while every IC check above stayed green — Spearman is
    # shift-invariant, and the WF harness refits per fold so its preds
    # re-centre themselves. Only the FROZEN production stream shows the
    # drift. The two-tail rolling-percentile decode turned that level
    # shift into direction skew (July fired 14 UP : 1 DOWN Strong) and the
    # executor traded almost only its weak side (live LONG -27 bps vs
    # SHORT +38 bps). So this check reads indicator_history, not WF OOS.
    log("## 2b. Production output-level drift (rank metrics are blind to this)")
    _LVL_FLOOR = 0.0008
    try:
        from shared.db import get_db_conn as _gdc
        _c = _gdc()
        try:
            with _c.cursor() as _cur:
                _cur.execute(
                    "SELECT pred_return_4h p FROM indicator_history "
                    "WHERE dt >= DATE_SUB(NOW(), INTERVAL 30 DAY) "
                    "AND pred_return_4h IS NOT NULL")
                _lp = [float(r["p"]) for r in _cur.fetchall()]
        finally:
            _c.close()
        if len(_lp) >= 240:
            _a = np.asarray(_lp)
            _m = float(_a.mean())
            _dn = float((_a <= -_LVL_FLOOR).mean())
            _up = float((_a >= _LVL_FLOOR).mean())
            _drift = abs(_m) > _LVL_FLOOR or min(_dn, _up) < 0.02
            if _drift:
                flags.append(f"PRODUCTION level drift: 30d pred mean {_m:+.5f}, "
                             f"tails dn {_dn:.1%} / up {_up:.1%}")
            _verdict = "LEVEL DRIFT — two-tail decode will skew" if _drift else "OK"
            log(f"- live 30d pred mean = {_m:+.5f}  (floor ±{_LVL_FLOOR})  "
                f"tails beyond floor: dn {_dn:.1%} / up {_up:.1%}  → {_verdict}\n")
        else:
            # 07-05 lesson: an unavailable check must FLAG, never silently pass
            flags.append(f"level-drift check UNAVAILABLE (only {len(_lp)} live preds)")
            log(f"- only {len(_lp)} live preds in 30d — UNAVAILABLE (flagged)\n")
    except Exception as _e:  # noqa: BLE001
        flags.append("level-drift check UNAVAILABLE: " + str(_e)[:80])
        log(f"- level-drift check failed: {_e} (flagged, not silently passed)\n")

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
    if stale:
        log(f"**STALE-DATA — RE-RUN REQUIRED.** Feature data ends {data_end:%Y-%m-%d %H:%M} "
            f"({data_age_h/24:.1f} days old); the most recent window is missing entirely, "
            f"so neither PASS nor DRIFT can be concluded. Fix backfill/network, then re-run "
            f"this script before acting on anything above.")
        if flags:
            log("\n(flags raised on the stale window, for reference only:)")
            for f in flags:
                log(f"  - {f}")
    elif not flags:
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
        log("  5. LEVEL-DRIFT flag (2b): rank-invisible model aging, NOT edge decay —")
        log("     run the maintenance refresh: retrain → research/validate_direction_refresh.py")
        log("     (G1-G4 pre-registered gates) → deploy. Precedent: 2026-08-08, deployed")
        log("     model sat 99 days, pred mean drifted +0.0024, July fired 14 UP : 1 DOWN.")

    out = RESULTS_DIR / f"quarterly_revalidation_{datetime.now(timezone.utc):%Y%m%d}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nReport → {out}")

    # Telegram push so a scheduled run actually surfaces. Plain text (no
    # Markdown specials) to avoid parse errors. Must survive a transient
    # network outage: 2026-07-05 DNS was down exactly at the 09:00 run, the
    # push died once and nothing retried — the operator never saw the verdict.
    pushed = False
    try:
        import os
        import time as _time
        from indicator.okx.alerter import send_critical
        chat = (os.environ.get("TG_ALERT_CHAT_ID")
                or os.environ.get("TG_CRITICAL_CHAT_ID") or "")
        verdict = ("STALE-DATA (re-run required)" if stale
                   else ("PASS" if not flags else "DRIFT/INVESTIGATE"))
        msg = (f"Monthly re-validation {stamp}\n"
               f"Verdict: {verdict}\n"
               f"Data ends {data_end:%Y-%m-%d %H:%M} ({data_age_h:.0f}h old)\n"
               f"AUC {auc:.3f} (band 0.55-0.62) | IC {ic:+.3f}\n"
               f"recent60d IC {ic_recent:+.3f} vs older {ic_older:+.3f}\n"
               f"Strong sign-acc {sw[0]*100:.0f}pct")
        if flags:
            msg += "\nflags: " + " ; ".join(flags)
        if chat:
            for attempt in range(1, 7):        # 6 tries / ~5 min of outage cover
                pushed = send_critical(chat, msg)
                if pushed:
                    break
                logger.warning("revalidation_telegram_push_retry attempt=%d/6", attempt)
                if attempt < 6:
                    _time.sleep(60)
    except Exception:
        logger.exception("revalidation_telegram_push_failed")
    if not pushed:
        # Stamp the failure into the report itself so a later reader knows the
        # verdict never reached the operator.
        logger.error("revalidation_telegram_push_gave_up")
        with out.open("a", encoding="utf-8") as fh:
            fh.write("\n> TELEGRAM PUSH FAILED — this verdict never reached the "
                     "operator. Check network, then re-run or push manually.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
