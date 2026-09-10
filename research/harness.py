# -*- coding: utf-8 -*-
"""研究線的共用鷹架（2026-09-10）

**為什麼會有這個檔案**：2026-09-10 一天內寫了三條研究線（§1.04 相關矩陣、
§1.05 止損地圖、§1.06 擁擠->幅度），每支 250~330 行，其中約七成是同一件事
的第三次實作：as-of 對齊、日聚類 bootstrap、尾部計數、逐幣一致性、置換。
而那七成正是最容易出錯的地方 —— 同一天就錯了四次（時間戳單位、
median vs nanmedian、補丁半套、heredoc 引號），**沒有一次是研究錯的**。

**這個檔案本身是「第二份實作」風險的放大器**：一個 bug 會同時污染所有結論，
而且不會有人發現，因為每條線都會得到看起來合理的數字。所以：

    每一個函式的算術都是從既有腳本**逐行搬過來**的，不是重新推導的。
    驗收條件是 `research/tests/test_research_regression.py` 那 7 個測試
    一個數字都不許動（那些是 2026-09-10 三份判決的已知答案）。

要改這裡任何一行算術，先問：那三份判決要不要跟著重跑？答案是要，
那就不是重構，是新的研究。
"""
from __future__ import annotations

import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

DAY_MS = 86_400_000
TZ_MS = 8 * 3600 * 1000        # UTC+8，全專案的日界（TODO §1.03m）


# ─────────────────────────── 時間 ───────────────────────────

def to_ms(v):
    """把時間欄位正規化成毫秒。**永遠不要寫死單位**——同一個 provider 的
    不同端點就會不同（mistake.md 2026-04-12，2026-09-10 又中一次）。"""
    v = np.asarray(v)
    if np.issubdtype(v.dtype, np.datetime64):
        return v.astype("datetime64[ms]").astype(np.int64)
    v = v.astype(np.int64)
    return v * 1000 if v.max() < 1e12 else v


def day8(ms, tz_ms=TZ_MS):
    """UTC+8 的日編號。"""
    return (np.asarray(ms, np.int64) + tz_ms) // DAY_MS


def asof(event_ms, series_ms):
    """對每個事件取**嚴格早於**它的最後一筆序列索引。

    回傳 (idx, ok)：`ok=False` 代表事件早於序列起點，呼叫端必須自己處理
    （通常是填 NaN），**不可以默默用 idx=0 那一筆**。

    「兩個時點拿到一模一樣的東西」永遠是切點失效的徵兆
    （mistake.md 2026-09-07），所以這裡不做任何 fallback。
    """
    sm = to_ms(series_ms)
    em = np.asarray(event_ms, np.int64)
    j = np.searchsorted(sm, em, side="left") - 1
    return np.clip(j, 0, len(sm) - 1), (j >= 0)


# ─────────────────────── 不確定性 ───────────────────────

def boot_days(days, vals, n=4000, seed=20260910):
    """日聚類 bootstrap。回傳 (均值, SE, CI 下緣, P(均值>0))。

    **四個一起回傳是刻意的**：CI 下緣上升有兩種相反的意思（變異數下降 vs
    均值上升），只報其中一個會誤述（使用者 2026-09-10 訂立）。而
    「CI 不跨零」是宣稱發現的門檻，不是下注的門檻，所以 P(>0) 一定要在。
    """
    rng = np.random.default_rng(seed)
    by = {}
    for d, v in zip(np.asarray(days), np.asarray(vals, float)):
        by.setdefault(int(d), []).append(v)
    ks = list(by)
    if len(ks) < 5:
        return float(np.mean(vals)), np.nan, np.nan, np.nan
    arr = [np.array(by[k]) for k in ks]
    idx = rng.integers(0, len(ks), size=(n, len(ks)))
    o = np.array([np.concatenate([arr[j] for j in idx[i]]).mean()
                  for i in range(n)])
    return (float(np.mean(vals)), float(o.std(ddof=1)),
            float(np.percentile(o, 2.5)), float((o > 0).mean()))


def boot_corr(x, y, n=2000, seed=20260910, method="pearson"):
    """相關係數的 bootstrap 區間。逐行搬自 strategy_corr（§1.04 判決值）。"""
    rng = np.random.default_rng(seed)
    xs, ys = np.asarray(x, float), np.asarray(y, float)
    m = len(xs)
    if m < 10:
        return float("nan"), float("nan")
    idx = rng.integers(0, m, size=(n, m))
    out = np.empty(n)
    for i in range(n):
        a, b = xs[idx[i]], ys[idx[i]]
        if method == "spearman":
            a = pd.Series(a).rank().to_numpy()
            b = pd.Series(b).rank().to_numpy()
        out[i] = 0.0 if a.std() == 0 or b.std() == 0 else np.corrcoef(a, b)[0, 1]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ─────────────────────── 尾部計數 ───────────────────────

def binom_upper(k, n, p0):
    """P(X >= k)，單尾。"""
    if n == 0:
        return 1.0
    return float(sum(comb(n, i) * p0 ** i * (1 - p0) ** (n - i)
                     for i in range(k, n + 1)))


def tail_test(sub, var, tail, strata=None, ycol="amp", sym_col="sym"):
    """尾部事件是否集中在 `var` 的高半邊。逐行搬自 sdv_amp（§1.06 判決值）。

    **為什麼是計數不是均值**：厚尾分布上，均值的雜訊比任何候選效應都大
    （SDV 實測 MDE 0.746 ATR vs 最大效應 0.287），而數東西便宜得多。

    `strata` 給定時在每一層內各自切中位 —— 那是「超出某個已知混淆之後
    還剩多少」的做法（§1.06 的 R2）。
    """
    v = sub[var].to_numpy(float)
    ok = np.isfinite(v)
    hi = np.zeros(len(sub), bool)
    syms = sub[sym_col].to_numpy()
    keys = (syms if strata is None
            else np.char.add(syms.astype(str), np.asarray(strata).astype(str)))
    for kk in np.unique(keys):
        m = (keys == kk) & ok
        if m.sum() >= 20:
            hi |= m & (v > np.nanmedian(v[m]))
    t = np.asarray(tail) & ok
    n, k = int(t.sum()), int((t & hi).sum())
    if n < 20:
        return None
    p0 = float((hi & ok).sum() / max(ok.sum(), 1))
    per = {}
    for s in np.unique(syms):
        m = (syms == s) & t
        if m.sum() >= 3:
            per[s] = float((m & hi).sum() / m.sum() - p0)
    return dict(n_tail=n, k_hi=k, share=k / n, p0=p0, p=binom_upper(k, n, p0),
                n_pos=sum(1 for x in per.values() if x > 0), n_sym=len(per),
                per_sym=per,
                amp_hi=float(np.nanmean(sub[ycol].to_numpy()[hi & ok])),
                amp_lo=float(np.nanmean(sub[ycol].to_numpy()[(~hi) & ok])))


def tail_mask(sub, col, q=0.95, sym_col="sym"):
    """逐標的的尾部旗標（跨標的尺度不同，不可以用全體分位）。"""
    th = sub.groupby(sym_col)[col].transform(lambda x: x.quantile(q))
    return (sub[col] > th).to_numpy()


def permute_within(df, cols, sym_col="sym", rng=None):
    """在**標的內**重排指定欄位：保留每個標的的分布與樣本數，只打斷它與
    結果的對應。回傳新的 DataFrame（不就地修改）。"""
    rng = rng or np.random.default_rng(20260910)
    w = df.copy()
    for c in cols:
        x = w[c].to_numpy(float).copy()
        for _s, idx in w.groupby(sym_col).indices.items():
            x[idx] = rng.permutation(x[idx])
        w[c] = x
    return w


def permute_best(df, cols, score, n=400, seed=20260910, sym_col="sym"):
    """置換檢定，**多重比較內建**：每一輪隨機也享有同樣的挑選機會。

    `score(w)` 要回傳「那一輪所有格子裡最好的那個統計量」。
    回傳 (p, 隨機值陣列)。p = (隨機 >= 真實 + 1) / (n + 1)。
    """
    real = score(df)
    rng = np.random.default_rng(seed)
    vals = np.array([score(permute_within(df, cols, sym_col, rng))
                     for _ in range(n)])
    return float((int((vals >= real).sum()) + 1) / (n + 1)), vals, real


# ─────────────────────── 凍結與輸出 ───────────────────────

def spec_hash(**spec):
    """判準的指紋。把它跟結果寫在一起 —— 判準改了、結果就對不上，自動現形。

    這是把「事前寫死、事後不放寬」從紀律變成機制的最小做法。
    """
    s = json.dumps(spec, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def write_report(path, res, spec=None):
    """全格輸出。`spec` 給定時一併寫入指紋。

    **不挑格**：呼叫端要把所有格子都放進 `res`，包含不好看的那些
    （驗證儀式：全格報告不挑格）。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = dict(res)
    if spec is not None:
        out["_spec"] = spec
        out["_spec_hash"] = spec_hash(**spec)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float),
                 encoding="utf-8")
    return p
