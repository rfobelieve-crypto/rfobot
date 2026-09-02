# -*- coding: utf-8 -*-
"""Signal-to-noise ratio of the direction edge — with its null floor.

Registered 2026-09-02 after an article on SNR for strategy evaluation. The
concept is sound; two things in its practical advice are not, and both are
handled here explicitly.

WHAT SNR IS
    r_t = mu_t + eps_t          SNR = Var(mu) / Var(eps)
Because Var(mu)/Var(r) is just R², this is computable in closed form from
the correlation between prediction and realised return:

    SNR = R² / (1 - R²)

so it needs no latent-variable estimation at all when you HAVE the
predictions — which this project does. Estimating mu with a rolling mean of
realised returns (the article's suggestion) is only necessary when the
predictions are unavailable, and it carries a large bias (below).

WHY THE NULL FLOOR IS REPORTED EVERY TIME
The article's rolling-mean estimator applied to PURE NOISE returns
SNR ≈ 1/(k+1): with k=20 that is 4.8%, which is LARGER than this system's
real SNR (0.2–1.9%). An SNR number without its null floor is therefore
worse than no number. Every reading here ships with:
  * shuffle null  — the same computation on shuffled labels (destroys the
    pairing, keeps both marginals), repeated N times
  * the article's rolling-mean estimator on pure noise, for the same k, so
    the bias is visible rather than inherited

WHAT IT IS NOT
SNR's numerator is the variance of the CONDITIONAL expected return, i.e.
how much the edge varies and is predicted. A strategy with a constant
positive mu scores SNR = 0 while having an excellent Sharpe. So this is a
TIMING diagnostic, not a profitability one — it belongs next to IC/AUC, and
it must never drive position size (this project has already rejected
sizing on estimated quantities three times: empirical Kelly on 18 trades,
confidence-weighted, calibrated-edge; the 2x notional cap is ruin maths,
not an optimisation target).

Usage:
    python research/snr_monitor.py                  # clean WF OOS parquet
    python research/snr_monitor.py --parquet PATH   # any pred/y table
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEFAULT_PARQUET = ROOT / "research/results/gate_a_revalidate_clean_oos.parquet"
OUT = ROOT / "research/results/snr_monitor.json"
N_SHUFFLE = 500
ROLLING_K = 20            # the k the article's example uses

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def snr_from_corr(r: float) -> float:
    r2 = float(r) ** 2
    return r2 / (1.0 - r2) if r2 < 1 else float("inf")


def shuffle_null(pred: np.ndarray, y: np.ndarray, n: int = N_SHUFFLE,
                 seed: int = 42) -> dict:
    """SNR of the SAME data with the pairing destroyed.

    This is the floor a reading has to clear. Shuffling keeps both marginal
    distributions (so fat tails, autocorrelation of |y|, etc. are preserved
    in the marginals) and removes only the association — which is exactly
    the thing SNR claims to measure.
    """
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        r = np.corrcoef(pred, rng.permutation(y))[0, 1]
        vals.append(snr_from_corr(r))
    v = np.array(vals)
    return {"mean": float(v.mean()), "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)), "n": n}


def rolling_mean_estimator_bias(k: int = ROLLING_K, n_obs: int = 3696,
                                reps: int = 200, seed: int = 0) -> dict:
    """The article's estimator, run on data with a KNOWN answer (zero).

    mu_hat = rolling mean of past returns; eps_hat = r - mu_hat. On pure
    noise the true SNR is 0, but this returns ~1/(k+1) because the rolling
    mean is itself an average of noise (Var = sigma^2/k).
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        noise = rng.normal(0, 1, n_obs)
        mu = pd.Series(noise).rolling(k).mean().shift(1)
        eps = noise - mu
        m = ~mu.isna()
        out.append(float(np.var(mu[m]) / np.var(eps[m])))
    v = np.array(out)
    return {"k": k, "mean": float(v.mean()), "std": float(v.std()),
            "theory_1_over_k_plus_1": 1.0 / (k + 1)}


def measure(df: pd.DataFrame, pred_col="pred_ret", y_col="y") -> dict:
    d = df.dropna(subset=[pred_col, y_col])
    p, y = d[pred_col].to_numpy(float), d[y_col].to_numpy(float)
    pear = float(np.corrcoef(p, y)[0, 1])
    spear = float(pd.Series(p).corr(pd.Series(y), method="spearman"))
    return {"n": int(len(d)),
            "pearson": pear, "spearman": spear,
            "snr_pearson": snr_from_corr(pear),
            "snr_spearman": snr_from_corr(spear),
            "null": shuffle_null(p, y)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    ap.add_argument("--pred-col", default="pred_ret")
    ap.add_argument("--y-col", default="y")
    a = ap.parse_args()

    path = Path(a.parquet)
    if not path.exists():
        print(f"snr_monitor: {path} 不存在")
        return 1
    df = pd.read_parquet(path)
    res = {"source": path.name}
    res["all"] = measure(df, a.pred_col, a.y_col)
    if "strong" in df.columns:
        s = df[df["strong"].astype(str) != "none"]
        if len(s) >= 100:
            res["strong"] = measure(s, a.pred_col, a.y_col)
    res["article_estimator_on_pure_noise"] = rolling_mean_estimator_bias()

    a_ = res["all"]
    print(f"SNR 監測 — {path.name}（n={a_['n']}）")
    print(f"  Pearson  {a_['pearson']:+.4f} → SNR {a_['snr_pearson']*100:.3f}%")
    print(f"  Spearman {a_['spearman']:+.4f} → SNR {a_['snr_spearman']*100:.3f}%")
    nl = a_["null"]
    print(f"  洗牌 null：均值 {nl['mean']*100:.3f}%、p95 {nl['p95']*100:.3f}%、"
          f"p99 {nl['p99']*100:.3f}%（n={nl['n']}）")
    verdict = ("高於 null p99 — 訊號存在但極薄"
               if a_["snr_pearson"] > nl["p99"] else
               "**未超過洗牌 null 的 p99——這個讀數本身不構成證據**")
    print(f"  → {verdict}")
    if "strong" in res:
        s_ = res["strong"]
        print(f"  Strong 子集 n={s_['n']}: Pearson {s_['pearson']:+.4f} → "
              f"SNR {s_['snr_pearson']*100:.3f}%（null p99 "
              f"{s_['null']['p99']*100:.3f}%）")
    b = res["article_estimator_on_pure_noise"]
    print(f"\n  對照：滾動均值估計式（k={b['k']}）在**純噪聲**上給出 "
          f"{b['mean']*100:.2f}% ± {b['std']*100:.2f}%"
          f"（理論 1/(k+1) = {b['theory_1_over_k_plus_1']*100:.2f}%）")
    print("  → 那個估計式的雜訊底比本系統的真實 SNR 還大，故不採用；"
          "本檔一律用 R²/(1−R²) 直接算。")
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
