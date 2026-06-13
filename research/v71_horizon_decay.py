"""
V7.1 Horizon Decay Curve
========================
Task spec (user, 2026-05-16). Measure the time structure of v7.1's alpha.
Output is DATA, not architecture decisions (per task constraint).

Two views, because the cached label files do not measure pure horizon return:

  BARRIER view  — straight from labels_winrate_TP50_SL30_H*.parquet.
      Fixed TP=50bps / SL=30bps, only the max-hold window H varies.
      WARNING: barrier WR vs H is mechanically confounded — a longer window
      gives more time to touch *either* barrier, and SL (30) sits closer than
      TP (50), so WR drifts down with H for geometric reasons, not alpha decay.
      Reported for completeness (it is the file the task names).

  HORIZON view  — pure close-to-close return at horizon H (the clean measure).
      r_H(t) = close[t+H]/close[t] - 1, signed by v7.1 direction.
      IC(H)  = Spearman(pred_ret, r_H) over the full OOS sample.
      This IS the alpha decay curve.

v7.1 signals + decoding identical to verify_kernel_method_c.py.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KLINES = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"
V71_OOS = PROJECT_ROOT / "research" / "results" / "dual_model" / "direction_reg_oos_mse.parquet"
CACHE = PROJECT_ROOT / "research" / "dual_model" / ".cache"
REPORT = PROJECT_ROOT / "research" / "results" / "v71_horizon_decay.md"
PLOT = PROJECT_ROOT / "research" / "results" / "v71_horizon_decay.png"

HORIZONS = [4, 6, 8, 12, 24]
PCT_WINDOW, WARMUP = 500, 100
STRONG_FRAC, MOD_FRAC = 0.05, 0.15
TP_BPS, SL_BPS = 50.0, -30.0

log_lines: list[str] = []


def log(msg: str = ""):
    print(msg)
    log_lines.append(msg)


def _strip_tz(idx):
    return idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx


def decode_tiers(pred: pd.Series) -> pd.DataFrame:
    """Rolling-percentile decode of signed pred_ret into tier+direction."""
    q = lambda f: pred.rolling(PCT_WINDOW, min_periods=WARMUP).quantile(f).shift(1)
    strong_hi, strong_lo = q(1 - STRONG_FRAC), q(STRONG_FRAC)
    mod_hi, mod_lo = q(1 - MOD_FRAC), q(MOD_FRAC)
    direction = pd.Series("NEUTRAL", index=pred.index, dtype=object)
    tier = pd.Series("None", index=pred.index, dtype=object)
    up_mod, dn_mod = pred >= mod_hi, pred <= mod_lo
    up_strong, dn_strong = pred >= strong_hi, pred <= strong_lo
    direction[up_mod] = "UP"
    direction[dn_mod] = "DOWN"
    tier[up_mod | dn_mod] = "Moderate"
    tier[up_strong | dn_strong] = "Strong"
    return pd.DataFrame({"pred_ret": pred, "direction": direction, "tier": tier})


def main():
    log("# V7.1 Horizon Decay Curve\n")
    log(f"Generated: {pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC\n")

    # ---- load v7.1 OOS predictions + decode signals -----------------------
    v = pd.read_parquet(V71_OOS)
    v.index = _strip_tz(v.index)
    v = v[~v.index.duplicated(keep="last")].sort_index()
    dec = decode_tiers(v["pred_ret"])
    sigs = dec[dec["direction"] != "NEUTRAL"].copy()
    sig_sign = sigs["direction"].map({"UP": 1.0, "DOWN": -1.0})

    # ---- continuous close for pure horizon returns ------------------------
    k = pd.read_parquet(KLINES)[["close"]].dropna()
    k.index = _strip_tz(k.index)
    k = k[~k.index.duplicated(keep="last")].sort_index()
    close = k["close"]

    log(f"v7.1 OOS span: {v.index[0]:%Y-%m-%d} -> {v.index[-1]:%Y-%m-%d}  "
        f"({len(v)} bars)")
    log(f"v7.1 signals (Strong+Moderate, UP+DOWN): {len(sigs)}  "
        f"(Strong {int((sigs.tier=='Strong').sum())}, "
        f"Moderate {int((sigs.tier=='Moderate').sum())})\n")

    rows = []
    for H in HORIZONS:
        # pure horizon return on the continuous close series
        fwd = close.shift(-H) / close - 1.0           # r_H at every bar

        # ---- HORIZON view (signal set) ------------------------------------
        sig_fwd = fwd.reindex(sigs.index)
        signed = sig_fwd * sig_sign                   # direction-adjusted return
        valid = signed.notna()
        n_sig = int(valid.sum())
        h_wr = float((signed[valid] > 0).mean())
        h_ret_bps = float(signed[valid].mean() * 1e4)
        # IC: full OOS sample, signed pred vs signed forward return
        full_fwd = fwd.reindex(v.index)
        m = v["pred_ret"].notna() & full_fwd.notna()
        ic_full, p_full = spearmanr(v["pred_ret"][m], full_fwd[m])
        # IC: signal set, conviction (|pred|) vs signed realized return
        conv = v["pred_ret"].reindex(sigs.index).abs()
        ms = valid & conv.notna()
        ic_sig, _ = spearmanr(conv[ms], signed[ms])

        # ---- BARRIER view (cached TP50/SL30 label file) -------------------
        lbl = pd.read_parquet(CACHE / f"labels_winrate_TP50_SL30_H{H}.parquet")
        lbl.index = _strip_tz(lbl.index)
        lbl = lbl[~lbl.index.duplicated(keep="last")].sort_index()
        lj = lbl.reindex(sigs.index)
        is_up = sigs["direction"] == "UP"
        win = np.where(is_up, lj["y_long_win"], lj["y_short_win"])
        ltype = np.where(is_up, lj["long_label_type"], lj["short_label_type"])
        bvalid = ~pd.isna(win)
        b_wr = float(np.nanmean(win))
        # per-trade barrier return: tp -> +50, sl/ambig_sl -> -30,
        # timeout -> realized signed close-to-close return
        bret = np.full(len(sigs), np.nan)
        bret[ltype == "tp"] = TP_BPS / 1e4
        bret[(ltype == "sl") | (ltype == "ambig_sl")] = SL_BPS / 1e4
        to_mask = pd.Series(ltype, index=sigs.index).str.startswith("timeout").fillna(False).values
        bret[to_mask] = signed.values[to_mask]
        b_ret_bps = float(np.nanmean(bret) * 1e4)

        rows.append(dict(
            H=H, n_sig=n_sig,
            h_wr=h_wr, h_ret_bps=h_ret_bps, ic_full=ic_full, p_full=p_full,
            ic_sig=ic_sig, b_wr=b_wr, b_ret_bps=b_ret_bps,
        ))

    res = pd.DataFrame(rows)

    # ---- tables -----------------------------------------------------------
    log("## HORIZON view — pure close-to-close return at H (clean alpha decay)\n")
    log("| H (h) | n signals | WR | avg ret/trade (bps) | IC full-sample | p-value |")
    log("|---|---|---|---|---|---|")
    for _, r in res.iterrows():
        log(f"| {int(r.H)} | {int(r.n_sig)} | {r.h_wr*100:.1f}% | "
            f"{r.h_ret_bps:+.1f} | {r.ic_full:.4f} | {r.p_full:.2e} |")
    log("")
    log("IC = Spearman(pred_ret, signed forward H-bar return) over the full "
        "3696-bar OOS sample. avg ret is GROSS (round-trip fee ~8bps for scale).\n")

    log("## BARRIER view — cached TP50/SL30 labels (confounded, see header)\n")
    log("| H (h) | barrier WR | barrier avg ret/trade (bps) |")
    log("|---|---|---|")
    for _, r in res.iterrows():
        log(f"| {int(r.H)} | {r.b_wr*100:.1f}% | {r.b_ret_bps:+.1f} |")
    log("")
    log("Confounded: fixed TP50/SL30 means a longer window only changes how "
        "often a barrier is reached; with SL(30) closer than TP(50), WR drifts "
        "down with H for geometric reasons. NOT a clean alpha-decay signal.\n")

    log("## Signal-level conviction IC\n")
    log("| H (h) | IC(|pred|, signed realized return) |")
    log("|---|---|")
    for _, r in res.iterrows():
        log(f"| {int(r.H)} | {r.ic_sig:.4f} |")
    log("")

    # ---- peak / half-life on the clean IC curve ---------------------------
    log("## Alpha time-structure\n")
    ic = res.set_index("H")["ic_full"]
    peak_H = int(ic.abs().idxmax())
    peak_ic = float(ic.loc[peak_H])
    log(f"- Peak horizon (max |IC|): **H={peak_H}**, IC={peak_ic:.4f}")
    ic4 = float(ic.loc[4])
    log(f"- IC at 4h (current fixed exit): {ic4:.4f}")
    # half-life: first H past the peak where |IC| <= 0.5*|peak|
    half = 0.5 * abs(peak_ic)
    hl = None
    past = ic[ic.index > peak_H]
    for H, val in past.items():
        if abs(val) <= half:
            prevH = peak_H if past.index[0] == H else past.index[past.index.get_loc(H) - 1]
            prevV = peak_ic if prevH == peak_H else float(ic.loc[prevH])
            if abs(prevV) != abs(val):
                frac = (abs(prevV) - half) / (abs(prevV) - abs(val))
                hl = prevH + frac * (H - prevH)
            else:
                hl = H
            break
    if hl is not None:
        log(f"- Half-life (|IC| -> 50% of peak): ~**{hl:.1f}h**")
    else:
        log(f"- Half-life: |IC| never drops to 50% of peak within "
            f"H<=24 (min |IC| past peak = {past.abs().min():.4f})")
    # persistence past 4h
    past4 = ic[ic.index > 4]
    grew = (past4.abs() > abs(ic4)).any()
    ratio_max = float(past4.abs().max() / abs(ic4)) if abs(ic4) > 0 else float("nan")
    log(f"- Alpha past 4h: max |IC| for H>4 is {past4.abs().max():.4f} "
        f"= {ratio_max*100:.0f}% of the 4h IC "
        f"({'persists/grows' if grew else 'decays'} past 4h)")
    log("")

    # ---- plot -------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(res.H, res.h_wr * 100, "o-", color="#1f77b4")
    ax[0].axhline(50, ls="--", c="grey", lw=0.8)
    ax[0].set_title("Horizon WR vs H")
    ax[0].set_xlabel("Horizon H (hours)")
    ax[0].set_ylabel("Win rate %")
    ax[1].plot(res.H, res.h_ret_bps, "o-", color="#2ca02c")
    ax[1].axhline(0, ls="--", c="grey", lw=0.8)
    ax[1].axhline(8, ls=":", c="red", lw=0.8, label="~fee 8bps")
    ax[1].set_title("Horizon avg return/trade vs H")
    ax[1].set_xlabel("Horizon H (hours)")
    ax[1].set_ylabel("Avg signed return (bps, gross)")
    ax[1].legend(fontsize=8)
    ax[2].plot(res.H, res.ic_full, "o-", color="#d62728", label="IC full-sample")
    ax[2].plot(res.H, res.ic_sig, "s--", color="#ff7f0e", lw=1,
               label="IC conviction (signals)")
    ax[2].axhline(0, ls="--", c="grey", lw=0.8)
    ax[2].set_title("Information Coefficient vs H")
    ax[2].set_xlabel("Horizon H (hours)")
    ax[2].set_ylabel("Spearman IC")
    ax[2].legend(fontsize=8)
    fig.suptitle("v7.1 Horizon Decay — pred_ret trained on 4h TWAP target",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT, dpi=110)
    log(f"Plot saved -> {PLOT}\n")

    res.to_csv(PROJECT_ROOT / "research" / "results" / "v71_horizon_decay.csv",
               index=False)
    REPORT.write_text("\n".join(log_lines), encoding="utf-8")
    log(f"Report saved -> {REPORT}")


if __name__ == "__main__":
    main()
