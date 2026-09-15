# -*- coding: utf-8 -*-
"""多場館單標的 MRP：把權重解出來，而不是假設 (+1, −1)（2026-09-13，TODO §1.39）

===========================================================================
先更正一個我自己在 §1.38 結尾講錯的預期
===========================================================================
我說「同一標的跨場館的 PC1 佔比預期遠低於 67-81%，因為市場因子被建構移掉」。
**那是錯的。** 同一個標的在 N 個場館的報酬幾乎一模一樣，所以 PC1 會是
**99%+ 而不是更低**；殘差（那不到 1%）才是套利訊號。

所以「PC1 低」不是這裡的判準 —— 它降格成一道**自曝檢查**（C1）：PC1 要是
沒有很高，代表我把兩個不同的東西當成同一個標的了。

**真正的判準是 Gate 0（核心原則 11）：可實現的邊際 ÷ 該場館的來回成本。**

===========================================================================
MRP 在這裡的增量只有一件，但很具體
===========================================================================
    §0.75 現況     2 個場館，權重**寫死** (+1, −1)
    MRP 能給的     同一標的 **N 個場館**，權重**解出來**

所以這支只回答一個問題：**解出來的 N 腿權重，扣掉費用之後的淨值，有沒有比
最好的那個 (+1,−1) 兩腿配對高？** 有，這條線就有東西；沒有，§0.75 的寫死
權重就是對的，而 MRP 這一族在我們手上結案。

**N 腿的費用不是兩腿的一倍而是按權重加權**：持有 w_i 單位名目的第 i 腿，
一趟來回付 2 × Σ|w_i| × fee_i（bps）。這與 `arblib.fees.round_trip_bps` 在
(+1,−1) 上**完全相同**（Σ|w| = 2 時退化成 2(fee_a + fee_b)），所以兩臂可比。
而「振幅 ÷ 成本」對權重的整體縮放免疫 —— 那個比值才是判準。

===========================================================================
三個容易出錯的地方，都是這個專案已經付過學費的
===========================================================================
1. **ticker 要從 `pair` 取，不是 `sym_a`。** `sym_a` 是場館自己的代號
   （BTC / BTC-USDT-SWAP / BTCUSDT 是同一個標的），用它 group 會把同一個
   ticker 拆成三組，每組的場館數都變少 —— 而那個錯不會報錯，只會讓
   「出現在 >= 3 個場館的 ticker」莫名其妙地少。
2. **兩臂都只能用前半挑。** 第一版我用全樣本挑最好的兩腿配對、只讓 MRP
   用前半 —— 那是讓基準偷看答案（mistake.md 2026-09-09：「報告前半選到
   什麼，而不只是我選的那個在後半如何」）。現在兩臂同一個程序：
   **前半挑 → 後半算**。
3. **低 portmanteau 不等於可交易。** 純白噪音的 portmanteau ≈ 0，讀起來
   「最會均值回歸」，但那是「180 秒取樣之下沒有任何結構」—— §1.37 已經量到
   跨場館的對齊住在 1 秒以下，所以這裡的殘差有一部分依定義就是取樣雜訊。
   **分辨的方法是振幅對成本，不是 portmanteau。**

===========================================================================
**掃描器的腿不是同時抓的 —— 這一點決定了預設要開 --sync-only（2026-09-13）**
===========================================================================
`scanner.py` 的 `scan_once` 把 `ts = int(time.time())` 在**週期開頭取一次**，
然後 `quote_all` 分**三段序列**抓，而整個週期的每一列共用那一個 ts：

    1. kind=="hl" 的腿      ThreadPoolExecutor(4)    HL / xyz / para / mkts / io
    2. CEX 的腿             ThreadPoolExecutor(4)    okx / bitget / binance
    3. **lighter 家族       普通 for 迴圈，每筆 time.sleep(0.06)**

從資料量到的落後相關（BTC，以 binance 報酬為基準）完全對上這個結構：
okx ρ=0.996（≈1 秒，同一池）、HL ρ=0.957（≈8 秒，前一段）、
**lighter / lighter-rh ρ=0.704（≈53 秒，最後一段）**。

後果是**假價差**：偏移 δ 在格寬 T 之下會注入 σ√(δ/T) 的價差離散度。
BTC 每 180 秒報酬 σ=7.9 bps -> 53 秒偏移注入 **4.33 bps 預測 / 4.64 實測**。
逐項對上（預測/實測）：okx 0.48/0.71、bitget 0.54/0.75、HL 1.64/1.92 ——
殘差一致的 ~0.25 bps 就是真正的基差雜訊，依 √(真²+假²) 疊加。
跨儀器對照（錄製器走兩條常駐 WS、同一瞬間讀兩本簿口）：
BTC 價差 std **0.978 vs 掃描器 4.829 = 4.9 倍**。
完整量測在 `scanner_leg_skew.py`（改過 scanner 的抓取順序要回來重跑那一支）。

**這個假價差依建構就是「一格內完成的均值回歸」**，也就是 MRP 最想找的東西
——所以在污染過的面板上解權重，解出來的是儀器不是市場
（mistake.md 2026-09-11：「這個偏誤會確認你的假說，那是最危險的方向」）。

**所以 `--sync-only`（預設開）只取同一個抓取段內的場館**：CEX 那組剛好是
三個（binance / bitget / okx，彼此 ≈1 秒），HL 家族也給得出三個以上。
`--no-sync-only` 保留污染版當對照 —— 兩個都印，因為「污染版的結論跟乾淨版
一樣」本身就是 MRP 判決的一部分。

===========================================================================
自曝檢查（全部在輸出裡，不挑格）
===========================================================================
C1  PC1 佔比要高（同一標的跨場館）。低的那些先當儀器壞掉不解讀。
C2  MRP 必須打敗**隨機的零和權重**，不是隨機權重 —— 任何零和權重都給出
    定態價差，所以拿任意權重當對照是一道恆真的關（第一版就是這個錯）。
C3  **BTC 當雜訊底**（factor-research #9 的控制配對）：兩個深簿、最緊的
    價差，帶應該最小。任何 ticker 的振幅落在 BTC 的 2 倍以內，就是
    「與儀器的雜訊分不開」，沿用 `scan_rank.CONTROL_MULT` 的既有慣例。
C4  cost=0 對照（factor-research #10）：含成本的淨值不得 >= 零成本的淨值。
C5  實測腿間偏移（見上）：段內必須是個位數秒。
C6  **Σw ≈ 0（市場中性）**。這是最關鍵的一道，理由見下。
C7  半衰期要落在解析度內（<= 2.5 x lags 根 = 75 分，§1.38 量過的上限）。

===========================================================================
C6 為什麼是最關鍵的一道
===========================================================================
portmanteau GEP 最小化的是**可預測性**，它**沒有約束 Σw = 0**。
而跨場館價差要能交易，必須市場中性 —— 否則「價差」裡混著裸的方向曝險，
它的 band 就只是標的自己的波動，不是基差。

第一版的表上看到的症狀是：MRP 的 band 反而**比最好兩腿大**
（ONE 123 vs 109、FLNC 38 vs 12），而我的假設是「用更多裸方向曝險買 band」。

**量完之後這個假設被否證了**：Σw/Σ|w| 中位 **−0.000**，352 列裡只有 **1 列**
的淨曝險超過 0.10 —— GEP 的解本來就幾乎是零和的（白化矩陣 rho_inv_sqrt
會把共同因子那個方向壓掉，所以零和是它的自然解，不需要外加約束）。

真正的解釋是：GEP 最小化的是**正規化後**的自相關（白化已經把變異數除掉），
所以它可以挑一個變異數更大、但正規化可預測性更小的組合。band 大不是作弊，
是目標函數本來就不管 band。**所以兩者可比，而零和投影幾乎是 no-op。**
投影版仍然算並且當判決用（`z_*`），因為「幾乎」不等於「是」。
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "research", "results", "mrp_crossvenue.json")

from mrp_portmanteau import halflife, portmanteau_gep, portmanteau_stat  # noqa: E402

# 180 秒一筆 -> lags=10 是 30 分鐘的窗，可偵測的半衰期上限約 2.5 x 10 = 25 根
# = 75 分鐘（§1.38 量到的解析度限制）。跨場館價差的回歸遠快於此。
BAR_SEC = 180
LAGS = 10
MIN_OBS = 400          # 每個 ticker x 場館至少這麼多共同時點
CONTROL_MULT = 2.0     # 與 arblib/scan_rank.py 同值

# 抓取段 —— **直接對應 `scanner.quote_all` 的三個區塊**，不是我分的類。
# hl 段與 cex 段的腿是同一個 ThreadPoolExecutor 同時發出去的（實測相互偏移
# 1-8 秒），跨段之間是序列的（實測 HL->CEX 8 秒、CEX->lighter 53 秒）。
# **lighter 段是個例外：它是普通 for 迴圈，所以段內也不同時**
# （實測 lighter-lighter-rh 價差 std 2.29 bps 不是 0）。把它列成一段是
# 近似不是事實 —— 乾淨面板實際用的是 cex 那組，它是真的執行緒池。
# 這張表要是跟 `quote_all` 漂開，這支的 --sync-only 就失去意義 ——
# 改 scanner 的抓取順序時要回來改這裡。
FETCH_GROUPS = {
    "hl": ["HL", "xyz", "para", "mkts", "io", "ENTROPY", "hyna"],
    "cex": ["okx", "bitget", "binance"],
    "lighter": ["lighter", "lighter-rh"],
}
VENUE_GROUP = {v: g for g, vs in FETCH_GROUPS.items() for v in vs}


def _load():
    """呼叫 arb 的 `scan_rank.load()` —— **同一顆，不是調成差不多的第二份**。

    它帶兩道已凍結的儀器修正（丟同場館配對、丟修正前的 Bitget 列），
    自己重寫一次就是 mistake.md 2026-09-07 那個病。
    """
    # 2026-09-15: arb location comes from research/arb_home.py (ARB_HOME overrides), not a hardcoded path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import arb_home  # noqa: E402
    arb_home.add_to_path()
    from arblib import scan_rank
    d = scan_rank.load()
    parts = []
    for pre in ("a", "b"):
        x = d[["ts", "pair", "leg_%s" % pre,
               "%s_bid" % pre, "%s_ask" % pre]].copy()
        x.columns = ["ts", "pair", "venue", "bid", "ask"]
        parts.append(x)
    p = pd.concat(parts, ignore_index=True)
    del parts, d
    # ticker = pair 的 @ 之前（canonical），不是 sym_a（場館自己的代號）
    p["sym"] = p["pair"].str.split("@").str[0]
    p = p.drop(columns=["pair"])
    p = p[(p.bid > 0) & (p.ask > 0)]
    p["mid"] = (p.bid + p.ask) / 2.0
    p["t"] = (p.ts // BAR_SEC) * BAR_SEC
    return p.groupby(["sym", "venue", "t"], as_index=False)["mid"].last()


def _fees(venues):
    # 2026-09-15: arb location comes from research/arb_home.py (ARB_HOME overrides), not a hardcoded path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import arb_home  # noqa: E402
    arb_home.add_to_path()
    from arblib import fees
    return {v: fees.fee_bps(v, maker=False, rebate=True) for v in venues}


def _cost(venues, w, fee):
    """一趟來回的費用（bps）= 2 x Σ|w_i| x fee_i。(+1,−1) 下退化成既有公式。"""
    return 2.0 * float(sum(abs(wi) * fee[v] for v, wi in zip(venues, w)))


def _band_events(s, span_days):
    """band = p90(|偏離|)；事件 = |偏離| >= band 的每一段極大連續區間，
    而且**要求收斂**（之後回到 band/2 以內）才算一次。

    次數數的是**事件**不是分鐘 —— mistake.md 2026-09-03 的那個 325 倍高估
    就是把 p90 造成的「恰好 10% 的分鐘」當成了交易次數。
    """
    d = np.abs(s - np.mean(s))
    band = float(np.percentile(d, 90))
    if band <= 0 or not np.isfinite(band):
        return 0.0, 0.0
    hot = d >= band
    n, i, m = 0, 0, len(d)
    while i < m:
        if not hot[i]:
            i += 1
            continue
        j = i
        while j + 1 < m and hot[j + 1]:
            j += 1
        k = j + 1
        while k < m and d[k] > band / 2.0:       # 等收斂
            k += 1
        if k < m:
            n += 1
        i = max(j + 1, k)
    return band * 1e4, n / max(span_days, 1e-9)   # band 以 bps 回報


def _score(s, venues, w, fee, span_days, zero_cost=False):
    band, ev = _band_events(s, span_days)
    c = 0.0 if zero_cost else _cost(venues, w, fee)
    return dict(band_bps=band, events_day=ev,
                cost_bps=c, net_bps=band / 2.0 - c,
                bps_day=ev * max(band / 2.0 - c, 0.0),
                pmt=portmanteau_stat(s, LAGS),
                hl_min=halflife(s) * BAR_SEC / 60.0)


def _sync_subset(w, min_venues):
    """只留**同一個抓取段**的場館，取列數最多的那一段。

    段內偏移 1-8 秒、跨段 53 秒，而 53 秒在 180 秒格上注入約 0.54σ 的假價差
    —— 混段的面板上，MRP 解出來的是儀器的偏移不是市場的價差。
    """
    best = None
    for g, vs in FETCH_GROUPS.items():
        c = [v for v in w.columns if v in vs]
        if len(c) < min_venues:
            continue
        k = w[c].dropna()
        if len(k) >= MIN_OBS and (best is None or len(k) > len(best[1])):
            best = (g, k)
    return best


def analyse(p, min_venues=3, sync_only=True):
    fee = None
    rows = []
    for sym, g in p.groupby("sym"):
        w = g.pivot(index="t", columns="venue", values="mid")
        w = w.dropna(axis=1, thresh=MIN_OBS)
        grp = "混段"
        if sync_only:
            sel = _sync_subset(w, min_venues)
            if sel is None:
                continue
            grp, w = sel
        else:
            w = w.dropna()
        if w.shape[1] < min_venues or len(w) < MIN_OBS:
            continue
        if fee is None:
            fee = _fees(sorted(p.venue.unique()))
        venues = list(w.columns)
        lp = np.log(w)
        if not np.isfinite(lp.values).all():
            continue
        span = (w.index[-1] - w.index[0]) / 86400.0
        # C1：PC1 佔比（報酬的相關矩陣）
        r = np.diff(lp.values, axis=0)
        C = np.corrcoef(r, rowvar=False)
        if not np.isfinite(C).all():
            continue
        evals = np.sort(np.linalg.eigvalsh(C))[::-1]
        pc1 = float(evals[0] / evals.sum())

        n = len(lp)
        tr, te = lp.iloc[:n // 2], lp.iloc[n // 2:]
        span_te = (te.index[-1] - te.index[0]) / 86400.0
        if len(tr) < MIN_OBS // 2 or len(te) < MIN_OBS // 2:
            continue

        # ── 臂 1：最好的 (+1,−1) 兩腿 —— **只用前半挑** ──────────────
        pick, best = None, np.inf
        for a, b in itertools.combinations(venues, 2):
            q = portmanteau_stat((tr[a] - tr[b]).values, LAGS)
            if np.isfinite(q) and q < best:
                best, pick = q, (a, b)
        if pick is None:
            continue
        wp = [0.0] * len(venues)
        wp[venues.index(pick[0])] = 1.0
        wp[venues.index(pick[1])] = -1.0
        s_pair = (te[pick[0]] - te[pick[1]]).values
        pair_sc = _score(s_pair, venues, wp, fee, span_te)

        # ── 臂 2：N 腿權重解出來 —— 同樣只用前半 ──────────────────
        try:
            _, W = portmanteau_gep(tr, LAGS)
            wm = np.real(W[:, 0])
        except Exception:                                 # noqa: BLE001
            continue
        if not np.isfinite(wm).all() or np.abs(wm).sum() == 0:
            continue
        raw_sumw = float(wm.sum() / np.abs(wm).sum())    # 正規化後的淨曝險
        wm = 2.0 * wm / np.abs(wm).sum()      # 正規化成 Σ|w| = 2，與 (+1,−1) 同毛曝險
        s_mrp = te.values @ wm
        mrp_sc = _score(s_mrp, venues, wm, fee, span_te)
        # C4 cost=0 對照
        mrp_free = _score(s_mrp, venues, wm, fee, span_te, zero_cost=True)

        # C6 投影到零和（市場中性）再重算 —— **判決看這一版**
        w0 = wm - wm.mean()
        if np.abs(w0).sum() == 0:
            continue
        w0 = 2.0 * w0 / np.abs(w0).sum()
        s_z = te.values @ w0
        z_sc = _score(s_z, venues, w0, fee, span_te)

        # C2：隨機**零和**權重（不是隨機權重 —— 那是恪真的關）
        rng = np.random.default_rng(0)
        rnd = []
        for _ in range(200):
            v = rng.normal(size=len(venues))
            v -= v.mean()                                  # 零和
            if np.abs(v).sum() == 0:
                continue
            v = 2.0 * v / np.abs(v).sum()
            rnd.append(portmanteau_stat(te.values @ v, LAGS))
        rnd_med = float(np.nanmedian(rnd)) if rnd else np.nan

        rows.append(dict(
            sym=sym, grp=grp, venues=len(venues), n=n,
            span_d=round(span, 2), pc1=pc1, pair="%s/%s" % pick,
            pair_band=pair_sc["band_bps"], pair_net=pair_sc["net_bps"],
            pair_ev=pair_sc["events_day"], pair_cost=pair_sc["cost_bps"],
            pair_bpsd=pair_sc["bps_day"], pair_pmt=pair_sc["pmt"],
            pair_hl=pair_sc["hl_min"],
            mrp_band=mrp_sc["band_bps"], mrp_net=mrp_sc["net_bps"],
            mrp_ev=mrp_sc["events_day"], mrp_cost=mrp_sc["cost_bps"],
            mrp_bpsd=mrp_sc["bps_day"], mrp_pmt=mrp_sc["pmt"],
            mrp_hl=mrp_sc["hl_min"], mrp_free_net=mrp_free["net_bps"],
            legs=int((np.abs(wm) > 0.10).sum()), rnd_pmt=rnd_med,
            pmt_is=portmanteau_stat(tr.values @ wm, LAGS),
            sumw=raw_sumw,
            z_band=z_sc["band_bps"], z_cost=z_sc["cost_bps"],
            z_net=z_sc["net_bps"], z_ev=z_sc["events_day"],
            z_bpsd=z_sc["bps_day"], z_pmt=z_sc["pmt"], z_hl=z_sc["hl_min"],
            z_legs=int((np.abs(w0) > 0.10).sum()),
            wts="|".join("%s:%+.2f" % (v, x) for v, x in zip(venues, w0)
                         if abs(x) > 0.10)))
    return pd.DataFrame(rows)


def offsets(p, base="binance", sym="BTC"):
    """C5：實測腿間偏移。段內應該是個位數秒，跨段是幾十秒。"""
    w = p[p.sym == sym].pivot(index="t", columns="venue", values="mid")
    w = w.dropna(axis=1, thresh=MIN_OBS)
    if base not in w.columns:
        return {}
    out = {}
    for c in w.columns:
        if c == base:
            continue
        k = w[[base, c]].dropna()
        if len(k) < MIN_OBS:
            continue
        d = np.diff(np.log(k.values), axis=0)
        rho = float(np.corrcoef(d[:, 0], d[:, 1])[0, 1])
        out[c] = (rho, max(0.0, (1.0 - rho)) * BAR_SEC)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-venues", type=int, default=3)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--no-sync-only", action="store_true",
                    help="不限制同一抓取段（污染版對照）")
    a = ap.parse_args()
    sync = not a.no_sync_only

    p = _load()
    print("面板：%s 列｜%d 個 ticker｜%d 個場館｜%.1f 天｜模式 **%s**"
          % (format(len(p), ","), p.sym.nunique(), p.venue.nunique(),
             (p.t.max() - p.t.min()) / 86400.0,
             "同一抓取段（sync-only）" if sync else "混段（污染版對照）"))

    print("\nC5 實測腿間偏移（BTC，以 binance 報酬為基準；段內應為個位數秒）")
    for v, (rho, dt) in sorted(offsets(p).items(), key=lambda x: x[1][1]):
        print("  %-11s 段=%-8s ρ=%.3f -> 偏移約 **%4.0f 秒**"
              % (v, VENUE_GROUP.get(v, "?"), rho, dt))

    r = analyse(p, a.min_venues, sync_only=sync)
    if r.empty:
        print("沒有 ticker 同時出現在 >= %d 個場館且樣本足夠" % a.min_venues)
        return 1
    pd.set_option("display.width", 250)
    fmt = lambda v: "%8.3f" % v                              # noqa: E731

    print("\n%d 個 ticker 出現在 >= %d 個場館（>= %d 個共同時點）｜逐段：%s"
          % (len(r), a.min_venues, MIN_OBS,
             ", ".join("%s=%d" % (k, v)
                       for k, v in r.grp.value_counts().items())))

    # ── 自曝檢查 ────────────────────────────────────────────────
    print("\n自曝檢查")
    print("  C1 PC1 佔比中位 **%.4f**（同一標的跨場館，應該接近 1）" % r.pc1.median())
    lo = r[r.pc1 < 0.90]
    if len(lo):
        print("     **%d 個 PC1 < 0.90 —— 不解讀，先當我把不同東西當成同一個**：%s"
              % (len(lo), ", ".join(lo.sym.head(8))))
        r = r[r.pc1 >= 0.90]
    c2 = int((r.mrp_pmt < r.rnd_pmt).sum())
    print("  C2 MRP 打敗隨機**零和**權重（中位）：**%d / %d**" % (c2, len(r)))
    btc = r[r.sym == "BTC"]
    floor = float(btc.pair_band.iloc[0]) if len(btc) else np.nan
    if np.isfinite(floor):
        print("  C3 BTC 雜訊底（控制配對）：兩腿 band **%.3f bps**"
              "（%s）-> 任何 band < %.3f 視為與儀器分不開"
              % (floor, btc.pair.iloc[0], CONTROL_MULT * floor))
    else:
        print("  C3 **BTC 不在表上 —— 沒有控制配對，下面的 band 沒有雜訊底可比**")
    c4 = int((r.mrp_net >= r.mrp_free_net).sum())
    print("  C4 cost=0 對照：含成本淨值 >= 零成本淨值的列 = **%d**（必須是 0）" % c4)
    print("  C6 **Σw / Σ|w|（淨曝險，0 = 市場中性）：中位 %+.3f｜"
          "|淨曝險| > 0.10 的有 %d / %d**"
          % (r.sumw.median(), int((r.sumw.abs() > 0.10).sum()), len(r)))
    hl_cap = 2.5 * LAGS * BAR_SEC / 60.0
    oor = int(((r.z_hl.abs() > hl_cap) | ~np.isfinite(r.z_hl)).sum())
    print("  C7 半衰期超出解析度（> %.0f 分 = 2.5 x lags）的列 = **%d / %d**"
          % (hl_cap, oor, len(r)))

    # ── 主表：樣本外 ───────────────────────────────────────────
    r = r.sort_values("z_net", ascending=False)
    print("\n" + "=" * 118)
    print("樣本外（後半）：**零和（市場中性）的 N 腿** vs 前半挑出的最好 (+1,−1) 兩腿")
    print("  band = p90 偏離 x2（bps）｜net = band/2 − 來回費用｜hl = 半衰期（分）")
    print("  `mrp_*` = 未投影的原始解，只當對照 —— 它的 band 含方向曝險，不可比")
    print("=" * 118)
    cols = ["sym", "venues", "pc1", "pair", "pair_band", "pair_net",
            "z_band", "z_cost", "z_net", "z_legs", "z_hl", "z_bpsd",
            "sumw", "mrp_band", "mrp_net"]
    print(r[cols].head(a.top).to_string(index=False, float_format=fmt))

    print("\n判決（全部看零和版）")
    wn = int((r.z_net > r.pair_net).sum())
    print("  **扣費後 MRP（零和）贏過最好兩腿：%d / %d（%.0f%%）**"
          % (wn, len(r), 100.0 * wn / len(r)))
    wn2 = int((r.z_pmt < r.pair_pmt).sum())
    print("  純看均值回歸品質（portmanteau，不扣費）贏：**%d / %d（%.0f%%）**"
          % (wn2, len(r), 100.0 * wn2 / len(r)))
    pos_p = int((r.pair_net > 0).sum())
    pos_z = int((r.z_net > 0).sum())
    print("  扣費後淨值為正：兩腿 **%d / %d**｜MRP 零和 **%d / %d**"
          % (pos_p, len(r), pos_z, len(r)))
    if np.isfinite(floor):
        ok = r[(r.z_net > 0) & (r.z_band > CONTROL_MULT * floor)
               & (r.z_hl.abs() <= hl_cap)]
        print("  **三關全過（淨值>0 ∧ band > 雜訊底 2 倍 ∧ 半衰期在解析度內）"
              "：%d 個**%s"
              % (len(ok), ("：" + ", ".join(ok.sym.head(10))) if len(ok) else ""))
    print("  零和解的有效腿數中位 %.1f（兩腿基準 2）｜半衰期中位 %.1f 分"
          % (r.z_legs.median(), r.z_hl.replace([np.inf, -np.inf], np.nan).median()))
    print("  （樣本內 portmanteau 中位 %.4f vs 樣本外 %.4f —— 比值 %.2f 是過擬合讀數）"
          % (r.pmt_is.median(), r.z_pmt.median(),
             r.z_pmt.median() / max(r.pmt_is.median(), 1e-9)))

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(r.to_dict("records"), fh, ensure_ascii=False,
                  indent=1, default=str)
    print("\n寫出 %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
