# -*- coding: utf-8 -*-
"""
========================================================================
2026-09-07 **判決：INCONCLUSIVE-BY-DESIGN。G2 FAIL，G1 不得解讀。**
========================================================================
實測  G1 +0.1328 CI [+0.0645,+0.2360]   G2 安慰劑 +0.0014 CI [+0.0000,+0.0027]
      G3 同號 PASS   G4 半衰期 3/7/14 = +0.158/+0.133/+0.112（單調，非調參）

依事前寫死的判準，G2 的 CI 不含零 -> FAIL -> G1 不解讀。**判準不事後放寬。**
但要記下判準錯在哪，因為兩個錯都在我的儀器上：

1) **門檻對 n=2,418,378 是錯的設計**。置換式安慰劑在這種樣本數下，任何殘留
   的微小相關都會變「顯著」，「CI 必須含零」幾乎注定不可能通過。這與
   mistake.md 2026-08-26 同族：寫下判準之後沒有把現有數字代進去問「它有沒有
   可能通過」。有資訊的量是**比值**（真值是安慰劑的 95 倍），但不得拿一個
   看完數字才發明的比值判準來宣告 PASS。

2) **安慰劑沒在測它宣稱要測的東西**。檔頭寫「總質量與時間結構不變，只打散
   位置」，程式碼卻是 `RNG.permutation(X)` —— 那是把 X 的**時間序列**整個
   打亂，位置與時間結構一起毀掉。正確的位置安慰劑要打散**密度的 bin 位置**
   （保留「何時有質量被創造」），例如在投影時把 liq bin 隨機平移一個與價格
   無關的位移。同族：mistake.md 2026-08-11（用對現象免疫的量去檢查）。

重跑之前必須先改這兩件事，並**另行註冊**（判準改成比值＋正確的位置安慰劑）。
本次結果保留可稽核，標記作廢。
E7 — 從 OI 增量推導未觸發的清算位分布，並做構造驗證。

來源
    使用者 2026-09-07 提供的 TradingView Pine（"Liquidation Levels on OI
    w/ profile"）。那支做對了兩件事，本檔照抄：
      · **只在 OI_delta > 0 時投影** —— 沒有新部位建立就沒有未來的清算位
      · **價格穿過就停止延伸** —— 位被消耗掉就清空
    三個缺陷本檔修掉（使用者的 E7 規格自己也點出了前兩個）：
      · 進場價用 (O+H+L+C)/4 單點 -> 改成 **uniform 撒到 [low, high]**
      · 槓桿檔位硬編且沒有衰減 -> 5/10/25/50/100x **等權、不事前優化**，
        加**時間衰減半衰期 7 天**（3/14 天做敏感度）
      · 方向純看 OI 增量 -> 用**同期 taker delta 的符號**分多空
        （OI↑ ∧ delta>0 -> 開多 -> 清算位在下；OI↑ ∧ delta<0 -> 在上；
         OI↓ 不投影）

===========================================================================
構造驗證的設計（**這一節是本檔的重點，判準寫在跑之前**）
===========================================================================
使用者的 E7 規格把「崩後那些位置的密度是否清空」列為最關鍵。
**照字面做會是套套邏輯**：清空是本檔自己的消耗規則做的（價格穿過就移除），
所以它必然清空——那是在驗自己的程式碼，不是驗市場。同族見
`absorb_matched.py`（λ 的分子含價格移動）與 `resting_fill.py`
（分層變數就是結果，Spearman +0.69）。

非套套的版本是拿**獨立資料**對：

    價格穿過模型算出的密度團塊時，**真實 OI 有沒有等比例地掉下去？**

    Y = 該 5 分鐘區間的真實 OI 減少量（來自 Binance metrics，與模型無關）
    X = 該區間價格掃過的**模型密度質量**（用區間**之前**的帳本狀態算，因果）
    控制 V = 該區間成交量（大行情本來就會有大 OI 變化）

判準（跑之前寫死，寫在 CI 上不寫在點估計上）
    G1 位置有沒有資訊（主閘門）
        Y ~ X + V 的 X 係數，日聚類 bootstrap CI **下緣 > 0**
        -> 模型密度的**位置**能預測真實 OI 銷毀，構造成立
        CI 含零 -> **INCONCLUSIVE**，E7 不往下走
    G2 位置安慰劑（**必須**過，否則 G1 不解讀）
        把每個 bin 的質量在價格軸上隨機重排（**總質量與時間結構不變，
        只打散位置**），重跑 G1。安慰劑的 X 係數 CI **必須含零**。
        若安慰劑也顯著，代表 X 量到的只是「價格動很大」，位置無資訊。
    G3 方向性
        分「多倉清算位」與「空倉清算位」各跑一次，兩邊係數同號。
        不同號 -> 方向映射有問題，停手查儀器。
    G4 衰減敏感度（報告，不是判準）
        半衰期 3 / 7 / 14 天三個版本的 G1 係數並列。若只有某一個成立，
        那是調參不是發現。

**先驗是不利的，明寫**（`.claude/rules/factor-research.md` 總則 2）
    · `oi_drop_pct` 對掃單事件的 AUC = **0.4996**（BRIDGE.md）——淨 OI
      在事件判別上等於丟硬幣。本檔用的是同一個 OI 序列，只是問法不同
      （不問「事件當下 OI 變多少」，問「掃過的位置能不能預測 OI 掉多少」）。
    · `liq_usd` 只覆蓋 3.8% 的事件。多數事件不伴隨可測清算。
    · 5 分鐘 OI 的一列**不是 create_time 當下的快照**（event_census 檔頭
      記載的前視），所以本檔一律只用 `create_time <= t` 且以區間差分使用。

**本檔只做構造驗證，不碰標籤。** 主變數 `dist_to_next_cluster`（低循環）
與次要的 `liq_density_at_sweep`（有循環風險）都要等 G1/G2 過了才註冊。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import event_census as ec  # noqa: E402

BARS = HERE / "data" / "bars"
OI = HERE / "data" / "oi"
OUT = HERE / "data" / "results"
BIN_BPS = 10.0                      # 價格網格：10 bps 等比（尺度無關）
LEV = (5, 10, 25, 50, 100)          # 等權，不事前優化
HALF_LIVES_D = (3.0, 7.0, 14.0)
HL_MAIN = 7.0
STEP_MS = 5 * 60_000
RNG = np.random.default_rng(20260907)


def build(sym, half_life_d, placebo=False, side=None):
    """回傳每個 5 分鐘區間的 (X 掃過的密度質量, Y 真實 OI 減少, V 量, day)。"""
    b = pd.read_parquet(BARS / f"{sym}.parquet",
                        columns=["ts", "high", "low", "close", "volume", "delta"])
    mts = b["ts"].to_numpy(np.int64)
    mhi = b["high"].to_numpy(float)
    mlo = b["low"].to_numpy(float)
    mvol = np.nan_to_num(b["volume"].to_numpy(float), nan=0.0)
    mdel = np.nan_to_num(b["delta"].to_numpy(float), nan=0.0)

    o = pd.read_parquet(OI / f"{sym}.parquet",
                        columns=["create_time", "sum_open_interest"])
    ots = (pd.to_datetime(o["create_time"], utc=True).astype("int64") // 10**6).to_numpy()
    oiv = o["sum_open_interest"].to_numpy(float)
    g = np.isfinite(oiv) & (oiv > 0)
    ots, oiv = ots[g], oiv[g]
    if len(ots) < 100:
        return None

    # 價格 -> bin（等比網格）
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

    xs, ys, vs, ds = [], [], [], []
    j_lo = np.searchsorted(mts, ots - STEP_MS, side="left")
    j_hi = np.searchsorted(mts, ots, side="right")

    for i in range(1, len(ots)):
        t = int(ots[i])
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

        # --- X：這個區間掃過的密度質量（用**進入區間前**的帳本，因果）----
        w = mass[bl:bh + 1]
        if len(w):
            dec = np.exp(-lam * np.maximum(t - last_t[bl:bh + 1], 0))
            swept = float(np.sum(w * dec))
            mass[bl:bh + 1] = 0.0          # 被價格穿過 -> 消耗掉
            last_t[bl:bh + 1] = t
        else:
            swept = 0.0

        # --- Y：真實 OI 減少量（獨立資料）--------------------------------
        d_oi = float(oiv[i] - oiv[i - 1])
        vol = float(np.nansum(mvol[a:z]))
        xs.append(swept)
        ys.append(max(0.0, -d_oi))         # 只看減少
        vs.append(vol)
        ds.append(t // 86_400_000)

        # --- 投影新的清算位（只在 OI 增加時）------------------------------
        if d_oi <= 0:
            continue
        dsum = float(np.nansum(mdel[a:z]))
        if dsum == 0:
            continue
        is_long = dsum > 0                 # 主動買推升 OI -> 開多 -> 位在下
        if side is not None and is_long != (side == "long"):
            continue
        span = np.arange(bl, bh + 1)
        if len(span) == 0:
            continue
        per_entry = d_oi / len(span) / len(LEV)     # uniform 撒到 [low, high]
        entry_px = ref * np.exp((span + lo_b) * step)
        for L in LEV:
            liq_px = entry_px * (1 - 1 / L) if is_long else entry_px * (1 + 1 / L)
            tb = to_bin(liq_px) - lo_b
            ok = (tb >= 0) & (tb < nb)
            tb = tb[ok]
            if not len(tb):
                continue
            dec = np.exp(-lam * np.maximum(t - last_t[tb], 0))
            mass[tb] = mass[tb] * dec + per_entry
            last_t[tb] = t

    if len(xs) < 200:
        return None
    X = np.array(xs)
    if placebo:
        # 位置安慰劑：總質量與時間結構不變，只把「掃到多少」重新洗牌
        X = RNG.permutation(X)
    return X, np.array(ys), np.array(vs), np.array(ds, dtype=np.int64)


def day_stats(X, Y, V, days):
    """把每一天壓成 OLS 的充分統計量：G_d = sum a a^T（3x3）、c_d = sum a y。

    為什麼不用逐列重抽：第一版每次 bootstrap 都對 240 萬列做 fancy indexing
    建新矩陣，1000 次 x 七個呼叫 —— 記憶體被吃爆（工作被系統殺掉）。
    OLS 的正規方程是**可加的**，所以按日預先累積之後，每次重抽只是把幾百個
    3x3 小矩陣加起來：記憶體 O(日數)，每次重抽 O(日數) 而不是 O(列數)。
    數值上完全等價。
    """
    uq, inv = np.unique(days, return_inverse=True)
    nd = len(uq)
    G = np.zeros((nd, 3, 3))
    c = np.zeros((nd, 3))
    A = np.empty((len(X), 3))
    A[:, 0] = X
    A[:, 1] = V
    A[:, 2] = 1.0
    for k in range(3):
        for j in range(3):
            G[:, k, j] = np.bincount(inv, weights=A[:, k] * A[:, j], minlength=nd)
        c[:, k] = np.bincount(inv, weights=A[:, k] * Y, minlength=nd)
    return G, c, uq


def boot_slope_from_days(G, c, b=1000):
    """從日層級充分統計量做日聚類 bootstrap，取 X 的係數。"""
    def solve(g, v):
        try:
            return float(np.linalg.solve(g, v)[0])
        except np.linalg.LinAlgError:
            return np.nan

    nd = len(G)
    point = solve(G.sum(axis=0), c.sum(axis=0))
    reps = np.empty(b)
    for i in range(b):
        p = RNG.integers(0, nd, nd)
        reps[i] = solve(G[p].sum(axis=0), c[p].sum(axis=0))
    reps = reps[np.isfinite(reps)]
    if len(reps) < b // 2:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def pooled(half_life, placebo=False, side=None):
    """逐幣建、逐幣標準化、立刻壓成日層級統計量後丟掉原始列。

    標準化是必要的：OI 與成交量的單位跨幣差幾個數量級，不標準化整個池化
    迴歸就是被 BTC 主導（等於只測了一個幣）。
    """
    # 依**日曆日**累積，跨幣合併到同一天：市場級衝擊會同時打九個幣，
    # 若九個幣的同一天被當成九個獨立的群重抽，CI 會被系統性低估
    # （sweep_forward 也是為了同一件事從 iid CI 改成日聚類，VIF 2.95）。
    acc: dict[int, list] = {}
    n_rows = 0
    for sym in ec.CORE9:
        r = build(sym, half_life, placebo=placebo, side=side)
        if r is None:
            continue
        x, y, v, d = r
        sx = float(np.std(x)) or 1.0
        sy = float(np.std(y)) or 1.0
        sv = float(np.std(v)) or 1.0
        G, c, uq = day_stats(x / sx, y / sy, v / sv, d)
        for i, dd in enumerate(uq):
            e = acc.get(int(dd))
            if e is None:
                acc[int(dd)] = [G[i].copy(), c[i].copy()]
            else:
                e[0] += G[i]
                e[1] += c[i]
        n_rows += len(x)
        del x, y, v, d, r, G, c
    if not acc:
        return None
    keys = sorted(acc)
    return (np.array([acc[k][0] for k in keys]),
            np.array([acc[k][1] for k in keys]), n_rows)


def main():
    res = {}
    print("=== G1 位置有沒有資訊（主閘門）===")
    print("   Y = 真實 OI 減少量  X = 該區間掃過的模型密度  V = 成交量（控制）")
    print()
    p = pooled(HL_MAIN)
    if p is None:
        sys.exit("資料不足")
    G1, c1, n1 = p
    b1, lo1, hi1 = boot_slope_from_days(G1, c1)
    v1 = "PASS" if lo1 > 0 else ("REJECT" if hi1 < 0 else "INCONCLUSIVE")
    print(f"   n={n1:,} 個 5 分鐘區間、{len(G1):,} 個 UTC 日   "
          f"X 係數 {b1:+.4f}  日聚類 CI [{lo1:+.4f}, {hi1:+.4f}]  -> {v1}")
    res["G1"] = dict(slope=b1, ci=[lo1, hi1], n=int(n1), days=int(len(G1)),
                     verdict=v1)

    print()
    print("=== G2 位置安慰劑（必須含零，否則 G1 不解讀）===")
    pp = pooled(HL_MAIN, placebo=True)
    G2, c2, _ = pp
    b2, lo2, hi2 = boot_slope_from_days(G2, c2)
    v2 = "PASS（含零）" if (lo2 <= 0 <= hi2) else "**FAIL — 位置無資訊，G1 不解讀**"
    print(f"   X 係數 {b2:+.4f}  CI [{lo2:+.4f}, {hi2:+.4f}]  -> {v2}")
    res["G2"] = dict(slope=b2, ci=[lo2, hi2], verdict=v2)

    print()
    print("=== G3 方向性（多倉位 vs 空倉位，需同號）===")
    dirs = {}
    for s in ("long", "short"):
        ps = pooled(HL_MAIN, side=s)
        if ps is None:
            print(f"   {s}: 資料不足")
            continue
        bs, los, his = boot_slope_from_days(ps[0], ps[1])
        dirs[s] = dict(slope=bs, ci=[los, his])
        print(f"   {s:5s}  係數 {bs:+.4f}  CI [{los:+.4f}, {his:+.4f}]")
    if len(dirs) == 2:
        same = dirs["long"]["slope"] * dirs["short"]["slope"] > 0
        print(f"   同號 -> {'PASS' if same else '**FAIL — 方向映射有問題，停手**'}")
        res["G3"] = dict(same_sign=bool(same), **dirs)

    print()
    print("=== G4 衰減敏感度（報告，不是判準）===")
    sens = {}
    for hl in HALF_LIVES_D:
        ph = pooled(hl)
        if ph is None:
            continue
        bh, loh, hih = boot_slope_from_days(ph[0], ph[1])
        sens[str(hl)] = dict(slope=bh, ci=[loh, hih])
        print(f"   半衰期 {hl:4.0f} 天  係數 {bh:+.4f}  CI [{loh:+.4f}, {hih:+.4f}]")
    res["G4"] = sens
    print("   （若只有某一個半衰期成立，那是調參不是發現。）")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "liq_density.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print()
    print("written ->", OUT / "liq_density.json")
    print()
    print("**本檔只做構造驗證，不碰標籤。** G1/G2 過了才註冊 dist_to_next_cluster。")


if __name__ == "__main__":
    main()
