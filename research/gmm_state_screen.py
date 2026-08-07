"""GMM state posteriors as V7 input features — the one untested variant.

Source idea: the GMM regime article (X, 2026-08-06). Every other use it
proposes has a verdict in this repo already:
  per-state models   -> regime submodels 2026-04-13, BEAR AUC 0.378, NO-GO
  favorable-state gate -> bear-UP gate 2026-06-05, would have hurt, NO-GO
  regime overlay     -> HMM overlay 2026-06-06 NO-GO (de-leaked)
  state-prob sizing  -> stage-4 sizing gauntlet, only tier-scaling survived
The one shape NOT tested: feed the state POSTERIOR to the model as a
feature and let XGB decide. Prior is weak (same-source, fourth attempt at
saturated data) but the screen is cheap and the pipeline for it is frozen
discipline: conditional IC vs V7 residual FIRST, ensemble A/B only if that
passes (2026-06-01 lesson).

PRE-REGISTERED CRITERIA (written before running):
  pass_screen =
        |conditional IC vs V7 residual| >= 0.03
    AND bootstrap 95% CI clear of 0
    AND monthly sign consistency >= 60%
    AND |corr(posterior, rv24)| < 0.8      # else it's repackaged vol,
                                           # which V7 already has 12 ways
  Anything less -> NO-GO, recorded, no A/B run.

CAUSALITY: GMM refit weekly on a trailing 90d window, posteriors predicted
one week forward with the frozen fit. Features standardised with trailing
stats. Full-sample fit would be look-ahead — the article never mentions
this, which is exactly the trap.

Components are label-switched between refits; states are re-ordered by
their RV mean each refit (state 0 = calmest) so the posterior series has a
stable meaning.

Usage: python research/gmm_state_screen.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")
from scipy.stats import spearmanr  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

RAW = ROOT / "market_data" / "raw_data"
DM = ROOT / "research" / "results" / "dual_model"
RNG = np.random.default_rng(17)

FIT_WIN = 2160          # 90d trailing window
REFIT_EVERY = 168       # weekly
KS = (2, 3)


def build_features() -> pd.DataFrame:
    k = pd.read_parquet(RAW / "binance_klines_1h.parquet")
    if not isinstance(k.index, pd.DatetimeIndex):
        tcol = "time" if "time" in k.columns else k.columns[0]
        ts = pd.to_numeric(k[tcol], errors="coerce")
        k = k.set_index(pd.to_datetime(ts, unit="s" if ts.max() < 1e12 else "ms"))
    if k.index.tz is not None:
        k.index = k.index.tz_localize(None)
    k = k[~k.index.duplicated(keep="last")].sort_index()
    px, hi, lo, vol = (pd.to_numeric(k[c], errors="coerce")
                       for c in ("close", "high", "low", "volume"))
    f = pd.DataFrame(index=k.index)
    f["r1"] = np.log(px).diff()                             # 報酬率
    f["rv24"] = f["r1"].rolling(24).std()                   # 實現波動率
    f["range"] = (hi - lo) / px                             # 價差/微結構代理
    f["vol_anom"] = vol / vol.rolling(720, min_periods=240).median()  # 成交量異常度
    # trailing 標準化（GMM 對尺度敏感；用全樣本統計就是前視）
    for c in f.columns:
        m = f[c].rolling(720, min_periods=240).mean()
        s = f[c].rolling(720, min_periods=240).std()
        f[c] = ((f[c] - m) / s.replace(0, np.nan)).clip(-6, 6)
    return f.dropna()


def walkforward_posteriors(f: pd.DataFrame, k: int) -> pd.DataFrame:
    X = f.to_numpy()
    n = len(f)
    post = np.full((n, k), np.nan)
    bics = []
    for start in range(FIT_WIN, n, REFIT_EVERY):
        Xw = X[start - FIT_WIN:start]
        try:
            g = GaussianMixture(n_components=k, covariance_type="full",
                                random_state=7, n_init=2).fit(Xw)
        except Exception:
            continue
        # label switching：按各 component 的 rv24 均值排序（state0 = 最平靜）
        order = np.argsort(g.means_[:, list(f.columns).index("rv24")])
        end = min(start + REFIT_EVERY, n)
        p = g.predict_proba(X[start:end])
        post[start:end] = p[:, order]
        bics.append(g.bic(Xw))
    out = pd.DataFrame(post, index=f.index,
                       columns=[f"gmm{k}_p{i}" for i in range(k)])
    out.attrs["bic"] = float(np.mean(bics)) if bics else np.nan
    return out


def main() -> int:
    f = build_features()
    print(f"特徵 {list(f.columns)}  n={len(f)}  "
          f"{f.index[0]:%Y-%m-%d} → {f.index[-1]:%Y-%m-%d}")

    v7 = pd.read_parquet(DM / "direction_concept_drift_oos.parquet")
    v7["ts"] = pd.to_datetime(v7["ts"], utc=True).dt.tz_localize(None)
    v7 = v7.set_index("ts").sort_index()

    # V7 殘差（秩空間）：y 裡 V7 沒抓到的部分
    ry = v7["y"].rank().to_numpy()
    rx = v7["pred"].rank().to_numpy()
    X1 = np.column_stack([np.ones(len(rx)), rx])
    v7["resid"] = ry - X1 @ np.linalg.lstsq(X1, ry, rcond=None)[0]

    print(f"V7 OOS n={len(v7)}  {v7.index[0]:%Y-%m-%d} → {v7.index[-1]:%Y-%m-%d}")
    results = []
    for k in KS:
        post = walkforward_posteriors(f, k)
        print(f"\n── K={k}（平均 BIC {post.attrs['bic']:,.0f}）──")
        j = post.join(v7[["resid", "y"]], how="inner").dropna()
        print(f"   與 V7 OOS 對齊 n={len(j)}")
        months = j.index.to_period("M")
        for col in post.columns[:-1]:      # K 個後驗和為 1，最後一個是線性冗餘
            x = j[col].to_numpy()
            res = j["resid"].to_numpy()
            ic = spearmanr(x, res).correlation
            boot = []
            for _ in range(2000):
                i = RNG.integers(0, len(j), len(j))
                boot.append(spearmanr(x[i], res[i]).correlation)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            # 月份符號一致性
            mm = []
            for m in months.unique():
                sel = months == m
                if sel.sum() >= 100:
                    mm.append(spearmanr(x[sel], res[sel]).correlation)
            cons = (np.mean([np.sign(v) == np.sign(ic) for v in mm])
                    if mm else np.nan)
            # 冗餘檢查：跟 rv24 本身的相關
            rvc = spearmanr(x, j.join(f[["rv24"]], how="left")["rv24"]).correlation
            ok = (abs(ic) >= 0.03 and lo * hi > 0 and
                  (cons if np.isfinite(cons) else 0) >= 0.60 and abs(rvc) < 0.8)
            results.append(ok)
            print(f"   {col}: cond-IC {ic:+.4f}  CI [{lo:+.4f},{hi:+.4f}]  "
                  f"月一致 {cons:.0%}  corr(rv24) {rvc:+.2f}  "
                  f"→ {'PASS' if ok else 'no-go'}")
    print("\n" + "=" * 60)
    if any(results):
        print("有後驗通過篩選 → 依既有紀律，下一步才是 ensemble A/B（四條 sanity）。")
    else:
        print("VERDICT: NO-GO — 無任何 GMM 狀態後驗對 V7 殘差帶邊際資訊。")
        print("同源資料第四次確認飽和；GMM 是模型端的變換，不是新資訊源。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
