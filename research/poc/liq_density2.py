# -*- coding: utf-8 -*-
"""E7 重跑 —— 從 OI 推導清算位分布，修掉第一版兩個儀器錯（**另行註冊**）

使用者 2026-09-09：「我之前給的 OI 推導清算位指標，在 swing 高低點如果有
密集流動性就會像強迫流一樣延續行情，這個有驗證過嗎」。

驗過（`liq_density.py`，2026-09-07），而且**四關過三關**：

    G1 效應      +0.1328  CI [+0.0645, +0.2360]   位置有資訊
    G3 方向同號  PASS
    G4 半衰期    3/7/14 天 = +0.158 / +0.133 / +0.112（單調，未調參）
    G2 安慰劑    FAIL  <- 判決卡在這裡

**而作廢理由是兩個錯都在我的儀器上**，原檔白紙黑字寫著、也寫了修法，
但那件事沒做。本檔就是去做它。

===========================================================================
修掉的兩個錯
===========================================================================
**錯 1：安慰劑沒在測它宣稱要測的東西**
    原檔頭寫「總質量與時間結構不變，只打散位置」，程式碼卻是
    `X = RNG.permutation(X)` —— 那是把 X 的**時間序列**整個打亂，
    位置與時間結構**一起**毀掉。於是它測的不是「位置有沒有資訊」，
    而是「X 與 Y 有沒有任何同期關聯」。
    **本檔的修法**：在**投影**那一步，把每個事件算出來的清算位 bin
    隨機平移一個與價格無關的位移（每個事件獨立抽）。
    -> 什麼時候有質量被建立、建立多少、衰減如何 **全部不變**；
       只有「落在哪個價位」被打散。這才是位置安慰劑。

**錯 2：判準對 n=240 萬是錯的設計**
    「安慰劑係數的 CI 必須含零」在這種樣本數下幾乎注定不可能通過 ——
    任何殘留的微小相關都會「顯著」。這是 mistake.md 2026-08-26 那族：
    寫下判準之後沒有代進去問「它有沒有可能通過」。
    **本檔的修法**：改用**置換 p 值** —— 跑 N=200 次不同隨機位移的安慰劑，
    p = (安慰劑係數 ≥ 真值係數的次數 + 1) / (N + 1)。
    這個判準**與樣本數無關**，而且是標準做法，不是看完數字才發明的比值。

===========================================================================
判準（跑之前寫死，事後不放寬）
===========================================================================
    H1  真值係數的日聚類 bootstrap CI **下緣 > 0**
        （沿用原 G1，那一關本來就過了）
    H2  **置換 p < 0.05**（N=200）。這取代原本不可能過的 G2。
    H3  方向性：多倉位／空倉位分開跑，兩邊係數同號（沿用原 G3）
    H4  半衰期 3/7/14 天並列報告，**不挑**（沿用原 G4，是報告不是判準）
    H5  **交易假設（使用者問的那一個，本檔第一次測）**：
        H1~H3 都過才解讀。對每個交會事件，取被掃價位附近的模型密度
        （用事件**之前**的帳本狀態，因果），分高／低密度兩組比較
        事後 480 分鐘的順勢報酬。
        判準：高密度組減低密度組的日聚類 CI **下緣 > 0** 且逐幣 ≥6/9。
        -> 「樞紐 × 密集清算流動性 -> 延續」成立，可成為第三個成分。

**先驗仍然不利，照抄原檔不打折**：`oi_drop_pct` 對掃單事件的 AUC = 0.4996
（BRIDGE.md），`liq_usd` 只覆蓋 3.8% 的事件。本檔問的是不同的問題
（不問「事件當下 OI 變多少」，問「掃過的位置能不能預測 OI 掉多少」），
但先驗不因為換了問法就變好。
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import event_census as ec  # noqa: E402
import liq_density as L1  # noqa: E402   只借 day_stats / boot_slope_from_days

BARS = HERE / "data" / "bars"
OI = HERE / "data" / "oi"
OUT = HERE / "data" / "results"
BIN_BPS = 10.0
LEV = (5, 10, 25, 50, 100)
HALF_LIVES_D = (3.0, 7.0, 14.0)
HL_MAIN = 7.0
STEP_MS = 5 * 60_000
N_PLACEBO = 200
SHIFT_BINS = 300          # 安慰劑位移範圍（±300 bin ≈ ±3%，與清算距離同量級）
RNG = np.random.default_rng(20260909)


def build(sym, half_life_d, placebo_rng=None, side=None, probes=None):
    """回傳 (X, Y, V, days) 與 probe 密度。

    placebo_rng 不是 None -> **位置安慰劑**：投影落點隨機平移，
                             時間結構與質量完全不變。
    probes: [(ts_ms, level_px)] -> 額外回傳每個 probe 當下、被掃價位附近的
            模型密度（用該時刻之前的帳本，因果）。
    """
    b = pd.read_parquet(BARS / f"{sym}.parquet",
                        columns=["ts", "high", "low", "close", "volume", "delta"])
    mts = b["ts"].to_numpy(np.int64)
    mhi = b["high"].to_numpy(float)
    mlo = b["low"].to_numpy(float)
    mvol = np.nan_to_num(b["volume"].to_numpy(float), nan=0.0)
    mdel = np.nan_to_num(b["delta"].to_numpy(float), nan=0.0)

    o = pd.read_parquet(OI / f"{sym}.parquet",
                        columns=["create_time", "sum_open_interest"])
    ots = (pd.to_datetime(o["create_time"], utc=True).astype("int64") // 10 ** 6).to_numpy()
    oiv = o["sum_open_interest"].to_numpy(float)
    g = np.isfinite(oiv) & (oiv > 0)
    ots, oiv = ots[g], oiv[g]
    if len(ots) < 100:
        return None

    step = np.log1p(BIN_BPS / 1e4)
    ref = float(np.nanmedian(b["close"].to_numpy(float)))

    def to_bin(px):
        return np.floor(np.log(np.maximum(px, 1e-12) / ref) / step).astype(np.int64)

    lo_b = int(to_bin(np.nanmin(mlo)) - 5)
    hi_b = int(to_bin(np.nanmax(mhi)) + 5)
    nb = hi_b - lo_b + 1
    if nb <= 0 or nb > 4_000_000:
        return None
    mass = np.zeros(nb)
    last_t = np.zeros(nb, dtype=np.int64)
    lam = np.log(2.0) / (half_life_d * 86_400_000.0)

    pr_ts = np.array([p[0] for p in probes], np.int64) if probes else np.zeros(0, np.int64)
    pr_px = np.array([p[1] for p in probes], float) if probes else np.zeros(0)
    pr_out = np.full(len(pr_ts), np.nan)
    pk = 0
    order = np.argsort(pr_ts) if len(pr_ts) else np.zeros(0, int)
    pr_ts, pr_px = (pr_ts[order], pr_px[order]) if len(pr_ts) else (pr_ts, pr_px)

    xs, ys, vs, ds = [], [], [], []
    j_lo = np.searchsorted(mts, ots - STEP_MS, side="left")
    j_hi = np.searchsorted(mts, ots, side="right")

    for i in range(1, len(ots)):
        t = int(ots[i])
        # --- probe：在推進到 t 之前，先用**當下的帳本**回答 probe -------
        while pk < len(pr_ts) and pr_ts[pk] <= t:
            pb = int(to_bin(np.array([pr_px[pk]]))[0]) - lo_b
            if 0 <= pb < nb:
                a0, a1 = max(0, pb - 10), min(nb, pb + 11)     # ±0.1% 鄰域
                dec = np.exp(-lam * np.maximum(t - last_t[a0:a1], 0))
                pr_out[pk] = float(np.sum(mass[a0:a1] * dec))
            pk += 1
        a, z = int(j_lo[i]), int(j_hi[i])
        if z <= a:
            continue
        seg_hi = float(np.nanmax(mhi[a:z]))
        seg_lo = float(np.nanmin(mlo[a:z]))
        if not (np.isfinite(seg_hi) and np.isfinite(seg_lo)) or seg_hi <= 0:
            continue
        bh = int(to_bin(np.array([seg_hi]))[0]) - lo_b
        bl = int(to_bin(np.array([seg_lo]))[0]) - lo_b
        bl, bh = max(0, min(bl, bh)), min(nb - 1, max(bl, bh))

        w = mass[bl:bh + 1]
        if len(w):
            dec = np.exp(-lam * np.maximum(t - last_t[bl:bh + 1], 0))
            swept = float(np.sum(w * dec))
            mass[bl:bh + 1] = 0.0
            last_t[bl:bh + 1] = t
        else:
            swept = 0.0

        d_oi = float(oiv[i] - oiv[i - 1])
        vol = float(np.nansum(mvol[a:z]))
        xs.append(swept)
        ys.append(max(0.0, -d_oi))
        vs.append(vol)
        ds.append(t // 86_400_000)

        if d_oi <= 0:
            continue
        dsum = float(np.nansum(mdel[a:z]))
        if dsum == 0:
            continue
        is_long = dsum > 0
        if side is not None and is_long != (side == "long"):
            continue
        span = np.arange(bl, bh + 1)
        if len(span) == 0:
            continue
        per_entry = d_oi / len(span) / len(LEV)
        entry_px = ref * np.exp((span + lo_b) * step)
        # **位置安慰劑在這裡**：每個事件抽一個與價格無關的位移。
        # 質量、時間、衰減全部不變，只有落點被打散。
        shift = int(placebo_rng.integers(-SHIFT_BINS, SHIFT_BINS + 1)) \
            if placebo_rng is not None else 0
        for Lv in LEV:
            liq_px = entry_px * (1 - 1 / Lv) if is_long else entry_px * (1 + 1 / Lv)
            tb = to_bin(liq_px) - lo_b + shift
            ok = (tb >= 0) & (tb < nb)
            tb = tb[ok]
            if not len(tb):
                continue
            dec = np.exp(-lam * np.maximum(t - last_t[tb], 0))
            mass[tb] = mass[tb] * dec + per_entry
            last_t[tb] = t

    if len(xs) < 200:
        return None
    res = (np.array(xs), np.array(ys), np.array(vs),
           np.array(ds, dtype=np.int64))
    if probes:
        back = np.empty(len(pr_out))
        back[order] = pr_out
        return res + (back,)
    return res


def slope(X, Y, V, days, b=800):
    G, c, _ = L1.day_stats(X, Y, V, days)
    return L1.boot_slope_from_days(G, c, b=b)


def pooled(half_life=HL_MAIN, placebo_rng=None, side=None, b=800):
    Xs, Ys, Vs, Ds = [], [], [], []
    for sym in ec.CORE9:
        r = build(sym, half_life, placebo_rng=placebo_rng, side=side)
        if r is None:
            continue
        X, Y, V, D = r
        # 逐幣標準化，避免大幣的絕對量支配池化係數
        sx, sy = np.std(X) or 1.0, np.std(Y) or 1.0
        Xs.append(X / sx); Ys.append(Y / sy); Vs.append(V / (np.std(V) or 1.0)); Ds.append(D)
    if not Xs:
        return None
    return slope(np.concatenate(Xs), np.concatenate(Ys),
                 np.concatenate(Vs), np.concatenate(Ds), b=b)


def main():
    res = {}
    print("=== H1 真值係數（位置有沒有資訊）===")
    r = pooled(HL_MAIN)
    if r is None:
        print("  資料不足"); return 1
    m, lo, hi = (r if len(r) == 3 else (r[0], r[1], r[2]))
    print(f"  係數 {m:+.4f}   日聚類 CI [{lo:+.4f}, {hi:+.4f}]  -> "
          + ("**PASS**" if lo > 0 else "**FAIL（CI 含零）**"))
    res["H1"] = dict(coef=m, ci=[lo, hi], passed=bool(lo > 0))
    print()
    print(f"=== H2 位置安慰劑置換檢定（N={N_PLACEBO}，只打散落點）===")
    print("    每次抽一個與價格無關的位移;質量、時間、衰減完全不變。")
    ph = []
    for k in range(N_PLACEBO):
        rp = pooled(HL_MAIN, placebo_rng=np.random.default_rng(90000 + k), b=1)
        if rp is not None:
            ph.append(rp[0])
        if (k + 1) % 40 == 0:
            print(f"    ...{k+1}/{N_PLACEBO}  目前安慰劑中位 {np.median(ph):+.4f}")
    ph = np.array(ph)
    p = (np.sum(ph >= m) + 1) / (len(ph) + 1)
    print(f"  安慰劑係數:中位 {np.median(ph):+.4f}  p95 {np.percentile(ph,95):+.4f}"
          f"  最大 {ph.max():+.4f}")
    print(f"  真值 {m:+.4f}   **置換 p = {p:.4f}** -> "
          + ("**PASS（< 0.05）**" if p < 0.05 else "**FAIL**"))
    res["H2"] = dict(placebo_median=float(np.median(ph)), p=float(p),
                     passed=bool(p < 0.05), n=len(ph))
    print()
    print("=== H3 方向性（多倉位／空倉位分開）===")
    for sd in ("long", "short"):
        rs = pooled(HL_MAIN, side=sd)
        if rs is None:
            print(f"  {sd}: 資料不足"); continue
        print(f"  {sd:>5s}  係數 {rs[0]:+.4f}  CI [{rs[1]:+.4f}, {rs[2]:+.4f}]")
        res.setdefault("H3", {})[sd] = dict(coef=rs[0], ci=[rs[1], rs[2]])
    if "H3" in res and len(res["H3"]) == 2:
        same = np.sign(res["H3"]["long"]["coef"]) == np.sign(res["H3"]["short"]["coef"])
        print(f"  -> 同號 {'PASS' if same else '**FAIL**'}")
        res["H3"]["same_sign"] = bool(same)
    print()
    print("=== H4 半衰期敏感度（報告，不是判準）===")
    for hl in HALF_LIVES_D:
        rh = pooled(hl, b=400)
        if rh:
            print(f"  {hl:.0f} 天  係數 {rh[0]:+.4f}  CI [{rh[1]:+.4f}, {rh[2]:+.4f}]")
            res.setdefault("H4", {})[str(hl)] = dict(coef=rh[0], ci=[rh[1], rh[2]])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "liq_density2.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print()
    ok = res["H1"]["passed"] and res["H2"]["passed"]
    print("=== 構造驗證判定 ===")
    print("  -> " + ("**H1+H2 通過 —— 密度的位置有資訊，可以進 H5 交易假設**"
                     if ok else "**未通過，H5 不解讀**"))
    print()
    print("written ->", OUT / "liq_density2.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
