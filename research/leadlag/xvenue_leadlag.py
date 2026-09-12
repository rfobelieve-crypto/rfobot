# -*- coding: utf-8 -*-
"""跨場館 lead-lag 的**註冊計分器**（TODO §1.37，預註冊 2026-09-13，判決 09-20）

來源：`small-trader-alpha-6` Part 3b「Volume & Lead-Lag」/ 3c「Latency &
Message Ordering」/ 3e「Global Lead-Lag」，加上 `strategy-discussion-lead-lag`
的「Simplest Method」（原文：「there's nothing fancy here, just standard
linear regression」）。

===========================================================================
**這支唯一算數的時鐘是 `rx_ms`，不是交易所的 `ts`。**
===========================================================================
2026-09-13 實測：兩邊的 `ts` 都是交易所自己戳的（Lighter `timestamp` /
HL `time`），**沒有共同時鐘**。用兩台獨立儀器估那個偏移：

    中價錄製器 stale_ms（p1）   Lighter   18 ms    HL  389 ms   -> +371 ms
    逐筆帶 rx_ms（p1）          Lighter   -5 ms    HL  163 ms   -> +168 ms

而量到的效應本身只有 100-270 ms。**人造量大於效應量，方向不可判定。**
最乾淨的證據是 `rx_ms` 的 Lighter **min = -7 ms**：交易所時戳比我們收到的
時刻還晚，物理上只有「兩個時鐘不可互比」一種解釋。

而照 3c，`rx_ms` 不只是「沒有污染的」時鐘，它是**決策該用的**那個：
「the message that is received first is the most accurate price, and you tend
to get incompletes on the leg that the message for the price was received
last」。我們要判斷的是「此刻我看到的兩個價格」，不是交易所各自的內部時戳。

所以本支**預設 strict**：只用有 `rx_ms` 的列。`--allow-exchange-ts` 會跑，
但會印一整段警告，而且方向那一欄不可引用。

===========================================================================
判準（預註冊，事後不放寬）
===========================================================================
L1  在 rx_ms 上仍有非零 lead-lag：峰值相關 CI 下緣 > 0 且峰值 lag != 0，
    大標的至少 6/9 同號。**L1 不過 -> 整條線停，不做 L2-L5**
    （Gate 0 排在資訊層之前，核心原則 11）。

L3 可行性關  每個成交額三分位裡「兩邊都有成交」的桶要 >= 70%。
    不到就判 **INSUFFICIENT（儀器不足），不判 FAIL** —— 250 ms 下成交帶有
    三分之二的桶至少一邊沒成交，三分位會退化成 nan，而那是儀器的事不是
    主張的事（mistake.md 2026-08-02：有桶為 0 一律先當儀器壞掉）。

自曝關（每次都跑，紅了就不准解讀下面任何數字）：
  S1  兩個場館都要有 rx_ms 的列，否則印「錄製器還沒重啟過」並停
  S2  逐場館印 rx_ms - ts 的 min / p1 / 中位 —— 那就是污染量本身
  S3  逐三分位印桶覆蓋率，即 L3 可行性關的原始數字
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

LIGHTER = "D:/flowbot_data/lighter/trades"
HL = "D:/flowbot_data/hl/trades"
MIN_BUCKETS = 600          # 兩邊都有成交的桶下限
COVERAGE_MIN = 0.70        # L3 可行性關
BOOT = 400                 # bootstrap 次數
BLOCK = 40                 # block bootstrap 的 block 長度（桶）


def _load(root: str, dates, cols):
    """讀逐筆帶。重啟之前的檔沒有 `rx_ms`，補 NaN 而不是讓它炸。"""
    fs = []
    for d in dates:
        fs += sorted(glob.glob("%s/%s/*.parquet" % (root, d)))
    if not fs:
        return None
    out = []
    for f in fs:
        d0 = pd.read_parquet(f)
        for c in cols:
            if c not in d0.columns:
                d0[c] = np.nan
        out.append(d0[cols])
    return pd.concat(out, ignore_index=True)


def _ms(v):
    """毫秒化，單位自動判斷（mistake.md 2026-04-12：不要假設 unit）。"""
    v = np.asarray(v, dtype="float64")
    m = np.nanmax(v)
    if m > 1e17:
        return v / 1e6
    if m > 1e14:
        return v / 1e3
    if m > 1e11:
        return v
    return v * 1e3


def prepare(dates, strict=True):
    lt = _load(LIGHTER, dates, ["ts", "rx_ms", "coin", "px", "usd"])
    ht = _load(HL, dates, ["ts", "rx_ms", "coin", "px", "sz"])
    if lt is None or ht is None:
        raise RuntimeError("找不到逐筆帶檔案 —— 先查 D 槽的目錄連結"
                           "（python research/ops/data_manifest.py）")
    ht["usd"] = ht.px * ht.sz
    rep = {}
    for d, lab in ((lt, "Lighter"), (ht, "HL")):
        d["ts_ms"] = _ms(d.ts.values)
        has = d.rx_ms.notna()
        r = {"rows": len(d), "with_rx": int(has.sum())}
        if has.any():
            off = d.rx_ms[has].astype("float64").values - d.ts_ms[has].values
            r["off_min"] = float(np.min(off))
            r["off_p1"] = float(np.percentile(off, 1))
            r["off_med"] = float(np.median(off))
        rep[lab] = r
        rx = d.rx_ms.astype("float64")
        d["t"] = (rx if strict else rx.fillna(d.ts_ms)) / 1000.0
        d.dropna(subset=["t"], inplace=True)
    return lt, ht, rep


def xcorr_ci(x, y, lag, nboot=BOOT, seed=0):
    """相關 + block bootstrap CI。

    用 block（預設 40 桶）而不是逐點重抽：相鄰桶的報酬有買賣價跳動造成的
    負自相關，逐點重抽會把 CI 算得太窄。
    """
    if lag >= 0:
        a, b = x[:len(x) - lag], y[lag:]
    else:
        a, b = x[-lag:], y[:len(y) + lag]
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 200 or a.std() == 0 or b.std() == 0:
        return np.nan, np.nan, np.nan, len(a)
    c = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(seed)
    n = len(a)
    nblk = max(1, n // BLOCK)
    vals = []
    for _ in range(nboot):
        st = rng.integers(0, max(1, n - BLOCK), size=nblk)
        idx = (st[:, None] + np.arange(BLOCK)[None, :]).ravel()
        idx = idx[idx < n]
        aa, bb = a[idx], b[idx]
        if aa.std() > 0 and bb.std() > 0:
            vals.append(float(np.corrcoef(aa, bb)[0, 1]))
    if not vals:
        return c, np.nan, np.nan, len(a)
    return (c, float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)), len(a))


def run(dates, bucket_s=0.25, max_lag=12, strict=True):
    lt, ht, rep = prepare(dates, strict=strict)
    clock = ("**rx_ms（我們自己的時鐘）**" if strict
             else "exchange ts（**已污染，方向不可引用**）")
    print("=" * 100)
    print("跨場館 lead-lag 計分器｜時鐘 = %s" % clock)
    print("桶 %0.0f ms｜lag 掃 +-%d 桶｜日期 %s"
          % (bucket_s * 1000, max_lag, ",".join(dates)))
    print("=" * 100)
    print("S1 / S2 自曝關：")
    for lab in ("Lighter", "HL"):
        r = rep[lab]
        print("  %-8s %10s 列｜有 rx_ms %9s｜rx-ts  min %+8.0f  p1 %+8.0f  "
              "中位 %+9.0f ms"
              % (lab, format(r["rows"], ","), format(r["with_rx"], ","),
                 r.get("off_min", float("nan")), r.get("off_p1", float("nan")),
                 r.get("off_med", float("nan"))))
    if strict and min(rep[l]["with_rx"] for l in rep) == 0:
        print("\n**有場館一列 rx_ms 都沒有 —— 錄製器還沒重啟過。"
              "不准解讀下面任何數字。**")
        return None
    print("  （S2 那兩個 p1 之差就是污染量本身；它們不為 0 正是本支不用 ts 的理由）")

    lo = max(lt.t.min(), ht.t.min())
    hi = min(lt.t.max(), ht.t.max())
    lt = lt[(lt.t >= lo) & (lt.t <= hi)]
    ht = ht[(ht.t >= lo) & (ht.t <= hi)]
    nb = int((hi - lo) // bucket_s) + 1
    print("\n共同時窗 %.3f 小時｜桶數 %s" % ((hi - lo) / 3600, format(nb, ",")))

    def ser(d, c):
        x = d[d.coin == c]
        k = ((x.t.values - lo) // bucket_s).astype(np.int64)
        g = pd.DataFrame({"k": k, "px": x.px.values, "usd": x.usd.values})
        a = (g.groupby("k").agg(px=("px", "last"), usd=("usd", "sum"))
             .reindex(range(nb)))
        return a.px.ffill().values, a.usd.fillna(0.0).values, a.px.notna().values

    shared = sorted(set(lt.coin) & set(ht.coin))
    print("共同標的 %d 個" % len(shared))
    rows = []
    for c in shared:
        pa, va, ha = ser(lt, c)
        pb, vb, hb = ser(ht, c)
        both = (ha & hb)[1:]
        if both.sum() < MIN_BUCKETS:
            continue
        ra, rb = np.diff(np.log(pa)), np.diff(np.log(pb))
        best = (-9.0, 0, float("nan"), float("nan"), 0)
        for L in range(-max_lag, max_lag + 1):
            c0, cl, ch, n = xcorr_ci(ra, rb, L)
            if np.isfinite(c0) and c0 > best[0]:
                best = (c0, L, cl, ch, n)
        pc, pl, cl, ch, n = best
        tot = va[1:] + vb[1:]
        q = np.quantile(tot, [1 / 3, 2 / 3])
        sel = (tot <= q[0], (tot > q[0]) & (tot <= q[1]), tot > q[1])
        cov = [float(both[s].mean()) if s.sum() else float("nan") for s in sel]
        rows.append(dict(coin=c, n=n, both_buckets=int(both.sum()),
                         peak_lag_ms=pl * bucket_s * 1000, peak_c=pc,
                         ci_lo=cl, ci_hi=ch,
                         cov_lo=cov[0], cov_mid=cov[1], cov_hi=cov[2]))

    r = pd.DataFrame(rows)
    if r.empty:
        print("\n**沒有任何標的湊到 %d 個兩邊都有成交的桶 —— "
              "樣本還不夠，這不是判決。**" % MIN_BUCKETS)
        return r
    r = r.sort_values("peak_c", ascending=False)
    pd.set_option("display.width", 200)
    fmt = lambda v: "%9.4f" % v

    print("\nL1｜峰值相關與它的 block-bootstrap CI（正 lag = Lighter 領先）")
    print(r[["coin", "both_buckets", "peak_lag_ms", "peak_c", "ci_lo", "ci_hi"]]
          .to_string(index=False, float_format=fmt))
    ok = r[(r.ci_lo > 0) & (r.peak_lag_ms != 0)]
    print("\nL1：CI 下緣 > 0 且 lag != 0 的標的 **%d / %d**" % (len(ok), len(r)))
    if len(ok):
        pos = int((ok.peak_lag_ms > 0).sum())
        print("     其中 Lighter 領先 %d、HL 領先 %d%s"
              % (pos, len(ok) - pos,
                 "" if strict else "  <- **污染，不可引用**"))

    print("\nS3 / L3 可行性關｜三分位的「兩邊都有成交」覆蓋率（要 >= %.0f%%）"
          % (COVERAGE_MIN * 100))
    print(r[["coin", "cov_lo", "cov_mid", "cov_hi"]]
          .to_string(index=False, float_format=fmt))
    feas = r[(r.cov_lo >= COVERAGE_MIN) & (r.cov_mid >= COVERAGE_MIN)
             & (r.cov_hi >= COVERAGE_MIN)]
    print("\nL3：三個三分位都過覆蓋率的標的 **%d / %d** -> %s"
          % (len(feas), len(r), "可判" if len(feas) >= 6
             else "**INSUFFICIENT（儀器不足，不判 FAIL）**"))

    out = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "results", "xvenue_leadlag.csv"))
    r.to_csv(out, index=False)
    print("\n寫出 %s" % out)
    return r


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", required=True,
                    help="UTC 日期資料夾名，例如 20260913")
    ap.add_argument("--bucket-ms", type=float, default=250.0)
    ap.add_argument("--max-lag", type=int, default=12)
    ap.add_argument("--allow-exchange-ts", action="store_true",
                    help="退回交易所時戳。**已污染**，只為了看重啟前的舊資料。")
    a = ap.parse_args()
    if a.allow_exchange_ts:
        print("!" * 78)
        print("!! 用交易所時戳跑。兩邊時鐘偏移實測 +168 ~ +371 ms，而效應本身")
        print("!! 只有 100-270 ms —— **方向完全不可報**。量級還可以看。")
        print("!" * 78)
    run(a.dates, a.bucket_ms / 1000.0, a.max_lag, strict=not a.allow_exchange_ts)
