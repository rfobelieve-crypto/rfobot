# -*- coding: utf-8 -*-
"""R1：**條件在成交上**的毛利（2026-09-12）

===========================================================================
為什麼這一支比「拿得到多少 maker 成交」更根本
===========================================================================
原本登記的 R1 是「每小時再平衡拿得到多少比例的 maker 成交」。讀完
〈Advanced Market Making〉之後那個問法太窄了，他的原話是：

> 「Most people think about the forecasting problem as optimizing for
>  R2/MSE/IC of their model fit when in reality **most of the moves you are
>  forecasting are ones you will never get filled on** so it doesn't really
>  matter if your forecast is good or not. **You want to be forecasting
>  conditional on you getting filled.**」

我們的 §1.23 毛利（樣本外 +0.29 ~ +0.93 bps/小時）是這樣算的：

    毛利 = 權重 × (mid[t+1] / mid[t] − 1)

那句式子**假設我們在 t 這一刻用 mid 拿到了想要的部位**。但掛單的人不決定
自己成不成交 —— 我們想買的時候，只有在**價格往下走到我們的掛價**時才成交。
也就是說：**成交本身跟報酬是相關的，而且是往壞的方向相關。**

所以這一支不是「加一個成交率折扣」，是**換一個估計量**。

===========================================================================
分工：簿口只判定成交，不准碰報酬的尺（2026-09-12 改寫的原因）
===========================================================================
第一版讓模擬**自己從簿口重建一條 mid 序列**去算持有期報酬，於是 D1 只能用
「約等於」去比 §1.23，而且比不過（−0.2456 vs −0.5052）。那個差額不是記帳
係數錯，是**我替同一個量造了第二把尺**——一把在 `(t0, t1]` 間找最近快照、
會隨覆蓋率漂移的尺。

改成：

    持有期報酬   用 §1.23 的 `fwd = mid.shift(-1)/mid − 1`，**一個字不改**
    簿口         只回答一件事：「這張掛單在這一小時內成交了嗎」

好處不是「讓 D1 過」，是**把自由度拿掉而不是去測它**。

**這裡不需要知道成交發生在第幾分鐘**，而我第一版誤以為需要：損益只由
**成交價**與**出場價**決定，而成交價就是我們的掛價（固定 = mid(t0)×(1∓δ)），
不是成交那一刻的 mid。所以新增的那一筆 Δw 賺的是 `mid(t1)/px − 1`
≈ `fwd + δ`，而這正好是本支記的 —— 第 5 分鐘成交還是第 55 分鐘成交，
同一個數字。

真正的偏誤在別處，列出來並標明偏哪一邊：

  **偏樂觀**
  · 全有全無，不做部分成交（真實上很可能只成交一部分）
  · 不管佇列位置：對手方穿到我們的價位就算成交。小額單大致成立
    （[[2026-09-07]]：規模決定哪一類簿口機制適用），但這是個假設
  · 一小時只掛一次、不改價（真實做市會 reprice，能拿到更多成交）

  **偏保守**
  · 簿口只有約每分鐘一筆，**分鐘內的插針看不到** -> 成交率是下限
  · 那一小時完全沒有快照時算「沒成交」（會計數印出來，不藏）

===========================================================================
模型（刻意保守，每一處都寫明偏哪一邊）
===========================================================================
每個再平衡時刻 t：

  1. 目標權重 w(t) 照凍結的構造算（`XS.weights_from`，一個字不改）
  2. 要交易的量是 **Δw = w(t) − w_實際(t−1)** —— 不是 w 本身。
     沒成交就**維持舊部位**，不是變成零（那是另一種假設，而且更樂觀）
  3. 掛單價 = mid(t) × (1 ∓ δ)，δ 是我們讓出的距離（bps）
  4. **成交判定：對手方要穿到我們的價位。**
     買單成交 ⟺ (t, t+1h) 內某一筆快照的 `ask_l1_price` ≤ 我們的買價
     賣單成交 ⟺ 同窗內某一筆快照的 `bid_l1_price` ≥ 我們的賣價
     用對手方的價格（不是 mid、不是同側），因為要有人**跨過來**才成交。
     這比「mid 碰到就算成交」保守 —— 後者會把一半的未成交算成成交。
  5. 成交價 = 我們的掛價。所以 δ 同時是**成交機會的代價**與**成交價的
     優勢**，兩邊都算進去。全有全無，不做部分成交（簡化，偏樂觀）
  6. 報酬 = 持有期報酬（部位 × §1.23 的 fwd）＋ 執行優勢（一次性）

δ=0 是「掛在 mid」—— 那是一個**不存在的掛法**（mid 在價差中間），
列出來只是為了跟現行的無條件毛利對照。真實可掛的最小值是半個價差，
所以本支會把**實測的半價差**印出來，讓人知道哪幾列的 δ 構得到。

**只在 §1.23 的估計量有定義的格子上動作**（w 與 fwd 都是有限值）。
那不是挑樣本，是「兩邊在同一個樣本上比」——否則差額裡會混進
「誰多算了幾格」這種跟問題無關的東西。

===========================================================================
自曝檢查（跑之前寫死）
===========================================================================
D1  `--assume-fill` + δ=0 必須**逐格重現** §1.23 的無條件毛利。
    改寫後這是構造上的恆等式，所以它測的不再是「兩把尺合不合」，而是
    **部位記帳（held / Δw / 沒成交就留著）在全成交時會不會塌回 w**。
D1b **有牙齒的那一道**：`assume_fill` 下 δ 的毛利減掉 δ=0 的毛利，
    必須精確等於 `δ × Σ|Δw|`（純算術，抓 `ex` 的符號與尺度）；
    而 `Σ|Δw|` 必須 ≈ **2 × §1.24 凍結的單邊換手率**（跨儀器對照）。
    恆真的守衛等於沒有守衛，所以 D1 旁邊一定要有這一道。
D2  成交率必須**隨 δ 單調下降**。不單調 = 成交判定寫錯了。
D3  **第一版寫錯了，連同錯誤一起留檔**（不是放寬，是它在量錯的東西）。
    原文是「δ=0 的成交率應接近擲硬幣（25~80%）：買單要價格跌到 mid 以下、
    賣單要漲上去，一小時內大致各半」。實測 **91.1%**，亮紅燈。

    紅燈是對的，錯的是門檻：我寫下的 50% 是「**一小時後**落在 mid 之下」
    的機率，而成交問的是「**一小時內曾經**碰到」—— 那是首次通過機率。
    一條 48 步的無偏漫步全程不越過起點的機率 ≈ 1/√(π·48) = **8.1%**，
    所以該期待的是 **≈92%**，而不是 50%。
    （同族：mistake.md 2026-08-26「判準寫在錯的那一端」。寫下判準之後
      要把數字代進去，我代了，但代的是另一個分布的數字。）

    改成**有對照組**的版本：同一套程式，把成交判定從「對手方的價格」
    換成 `(bid+ask)/2`（= 假裝零價差）。要有人跨過來必然比 mid 碰到更難，
    所以：**實測成交率必須嚴格低於零價差對照**，而且兩者的落差要跟
    半價差的量級相稱。若實測 ≥ 對照，就是 bid/ask 讀反或買賣方向寫反
    —— 而那個錯法在舊版的 25~80% 帶裡**照樣會過**（它會給出接近 100%）。

    python research/mft_fill_conditional.py --days 120
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import gate0_xs_turnover as XS                   # noqa: E402

OUT = ROOT / "research" / "results" / "mft_fill_conditional.json"
# 讓出的距離（bps）。0 = 掛在 mid（不可能，只當對照）。
DELTAS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)
RNG = np.random.default_rng(20260912)


def load_book(days: int) -> pd.DataFrame:
    """逐分鐘的最佳買賣價 —— 成交判定要用對手方的價格，不能只有 mid。"""
    from shared.db import get_db_conn
    conn = get_db_conn()
    q = """
        SELECT canonical_symbol AS sym, ts_ms,
               bid_l1_price AS bid, ask_l1_price AS ask
        FROM orderbook_snapshots_1m
        WHERE created_at >= DATE_SUB(UTC_TIMESTAMP(), INTERVAL %s DAY)
        ORDER BY ts_ms
    """
    d = pd.read_sql(q, conn, params=(days,))
    conn.close()
    d = d[(d.bid > 0) & (d.ask > 0) & (d.ask >= d.bid)].copy()
    d["minute"] = d.ts_ms // 60000
    return d


def prep_book(book: pd.DataFrame, zero_spread: bool = False):
    """標的 -> (minute, bid, ask) 的 numpy 陣列。**只給成交判定用。**

    `zero_spread=True` 把兩側都換成 `(bid+ask)/2` —— 那是 D3 的對照組：
    假裝沒有價差，成交只需要「mid 碰到我們的價位」。真實（要對手方跨過來）
    必然比它難，所以它是實測成交率的**上界**。
    """
    by = {}
    for s, g in book.groupby("sym"):
        g = g.sort_values("minute")
        b, a = g.bid.to_numpy(float), g.ask.to_numpy(float)
        if zero_spread:
            m = (b + a) / 2.0
            b = a = m
        by[s] = (g.minute.to_numpy(np.int64), b, a)
    return by


def day_boot(x, days, b=3000):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, days = x[ok], np.asarray(days)[ok]
    if len(x) < 20:
        return (np.nan,) * 4
    uq, inv = np.unique(days, return_inverse=True)
    ix = [np.where(inv == k)[0] for k in range(len(uq))]
    r = np.empty(b)
    for i in range(b):
        p = RNG.integers(0, len(uq), len(uq))
        r[i] = x[np.concatenate([ix[k] for k in p])].mean()
    return (float(x.mean()), float(r.std(ddof=1)),
            float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))


def simulate(w: pd.DataFrame, fwd: pd.DataFrame, mid0: pd.DataFrame,
             by: dict, delta_bps: float, assume_fill: bool = False):
    """回傳 dict：逐期組合報酬（Series）、成交率、逐筆執行優勢、Σ|Δw| 等。

    w / fwd / mid0 同索引（再平衡分鐘）同欄位（標的）：
        w     目標權重（§1.23 構造）
        fwd   這一期到下一期的 mid 報酬（§1.23 的尺，**報酬只能用它**）
        mid0  本期的 mid（只當掛單價的錨）
    by        成交判定器（簿口），見 `prep_book`
    """
    rb = np.asarray(w.index, np.int64)
    syms = list(w.columns)
    W, F, M = w.to_numpy(float), fwd.to_numpy(float), mid0.to_numpy(float)
    nk, ns = W.shape
    held = np.zeros(ns)
    port = np.full(nk, np.nan)
    fills = tries = nodata = 0
    edges, dwabs = [], np.zeros(nk)
    # 兩條換手，**刻意分開**：
    #   dwabs      實際部位的變動（缺格時部位會落後，所以偏大）-> 付的手續費
    #   dwabs_tgt  純目標對目標的變動 -> 才跟 §1.24 的凍結換手率同一個定義
    # 第一版只有前者，卻拿它去對凍結值，差 13~17%，而我用一條 15% 的寬容帶
    # 蓋過去。那是「比了兩個不同的量再放寬門檻」——守衛該精確的地方要精確。
    dwabs_tgt = np.full(nk, np.nan)
    # 機制關（CLAUDE.md 2026-09-09 E4 那一關）：**它沒成交的到底是不是贏家。**
    # 對每一個「想交易」的格子記下參考那側本來會記的貢獻 w×fwd，
    # 以及 |fwd|。成交與沒成交分兩堆，事後比。
    #   *_c  目標部位本來的貢獻 tgt×fwd（看「這一格本來是賺是賠」）
    #   *_d  **那一期真正的損失** Δw×fwd（沒成交 = 少了這一塊），攤到每期才對
    #   *_a  |fwd|（看沒成交那堆是不是行情更大的那些）
    # **注意 *_d 與 *_c 的符號對「沒成交」那一堆是恆正的，那是定義後果不是
    # 發現**：買單沒成交 ⟺ 價格從沒跌到掛價 ⟺ fwd > 0 ⟺ Δw×fwd > 0，
    # 賣單鏡像。有內容的是**量級**，不是符號。
    mech = {"fill_c": [], "miss_c": [], "fill_a": [], "miss_a": [],
            "fill_d": [], "miss_d": []}
    for k in range(nk):
        t0 = rb[k]
        t1 = rb[k + 1] if k + 1 < nk else t0 + 60
        acc, used = 0.0, False
        if k >= 1:
            prev, cur = W[k - 1], W[k]
            mm = np.isfinite(prev) & np.isfinite(cur)
            dwabs_tgt[k] = float(np.abs(cur[mm] - prev[mm]).sum())
        for j, s in enumerate(syms):
            tgt, fw, m0 = W[k, j], F[k, j], M[k, j]
            if not (m0 > 0):
                # 這個名字這一期不在資料裡 -> 不能交易也不能評分，部位留著
                continue
            if not (np.isfinite(tgt) and np.isfinite(fw)):
                # **不可評分的格子：把部位對齊到「估計量認定我們持有的東西」，
                # 但不下單、不計量。**
                #   · 目標有值但缺下一根 mid -> 對齊到目標
                #   · 目標是 NaN            -> 對齊到 **0**，因為 §1.23 的
                #     估計量就是這樣算的（`w.where(both)` 讓 NaN 貢獻 0 =
                #     沒有部位）。分子說空倉、分母說還抱著，會是兩個故事。
                # 為什麼不在這裡下單：下了就會被記進成交/未成交，而這一期
                # **不會被評分**，於是 D1c 的歸因等式少一塊（實測殘差 +0.562，
                # 那一關當場擋下來了）。只在估計量有定義的地方動手，才閉合。
                # 不對齊的後果也實測過：`held` 卡在舊值，下一個可評分期的 Δw
                # 被缺口撐大，分母比真實的目標換手大 15%。
                held[j] = tgt if np.isfinite(tgt) else 0.0
                continue
            used = True
            dw = tgt - held[j]
            if abs(dw) <= 1e-9:
                # 差到小數第 9 位才不送單，那就當它已經到位 —— 否則 held 會
                # 永遠停在差 1e-9 的地方，D1 的逐格門檻（1e-9）會假性失敗。
                held[j] = tgt
            else:
                tries += 1
                buy = dw > 0
                px = m0 * (1 - delta_bps / 1e4) if buy \
                    else m0 * (1 + delta_bps / 1e4)
                if assume_fill:
                    got = True
                else:
                    arr = by.get(s)
                    got = False
                    if arr is None:
                        nodata += 1
                    else:
                        mins, bid, ask = arr
                        # 掛單活在 (t0, t1)：t0 那一刻才下單，所以成交只能
                        # 發生在它**之後**；t1 就撤單換一輪。
                        a = int(np.searchsorted(mins, t0, side="right"))
                        b = int(np.searchsorted(mins, t1, side="left"))
                        if b <= a:
                            # 這一小時內沒有任何簿口快照 -> 判不了。
                            # 算成「沒成交」（保守），但要計數印出來：
                            # 缺資料與真的沒成交在數字上長得一樣。
                            nodata += 1
                        elif buy:
                            got = bool(np.any(ask[a:b] <= px))
                        else:
                            got = bool(np.any(bid[a:b] >= px))
                tag = "fill" if got else "miss"
                mech[tag + "_c"].append(tgt * fw * 1e4)
                mech[tag + "_d"].append(dw * fw * 1e4)
                mech[tag + "_a"].append(abs(fw) * 1e4)
                if got:
                    fills += 1
                    # **執行面的優劣勢是一筆一次性的損益**，不是持有期報酬的
                    # 一部分。買在 mid 以下 -> 賺；賣在 mid 以上 -> 也賺。
                    # 這個式子兩個方向都對（dw 與 (m0−px) 同號），而且
                    # 總和必然 = δ × Σ|Δw| —— D1b 就是在釘這一點。
                    ex = dw * (m0 - px) / m0 * 1e4
                    acc += ex
                    edges.append(ex)
                    dwabs[k] += abs(dw)
                    held[j] = tgt
            # 持有期報酬：**用 §1.23 的 fwd**，不自己從簿口重建 mid。
            if abs(held[j]) > 1e-12:
                acc += held[j] * fw * 1e4
        if used:
            port[k] = acc
    s = pd.Series(port, index=w.index).dropna()
    valid = ~np.isnan(port)
    # **一律用「總量對總量」，不要用「每期平均」。**（2026-09-12 更正）
    # 第一版把每期的毛利平均（977 期）減掉每期的手續費平均（1452 期攤），
    # 兩個不同的分母 —— 費被低估 1452/977 = 1.49 倍，而那個數字已經被我
    # 發佈出去了（見 TODO §1.27 的更正框）。
    # 「每單位成交量賺多少」對「每單位成交量付多少」才是沒有歧義的比較，
    # 因為費率本來就是按成交量收的，跟「有幾期能評分」無關。
    tot_g = float(np.nansum(port))
    tot_v = float(np.nansum(dwabs))
    return dict(port=s, fill_rate=(fills / tries if tries else np.nan),
                fills=fills, tries=tries, nodata=nodata,
                edges=np.asarray(edges, float),
                # **兩個分母都留著，因為它們回答不同的問題。**
                # dwabs_valid：只算毛利有定義的那些期 -> 跟 port.mean()
                #   同一個母體，D1b 的算術關必須用它（用另一個就會差
                #   1117/917 = 1.218 倍，而那個差看起來像符號錯）
                # dwabs_all：算全部期 -> 才跟 §1.24 的凍結換手率同母體
                dwabs_valid=float(dwabs[valid].mean()) if valid.any() else np.nan,
                dwabs_all=float(dwabs.mean()),
                total_gross=tot_g, total_volume=tot_v, n_scored=int(valid.sum()),
                # 每單位成交量的毛利 —— 本支的主指標，直接跟費率比
                per_vol=(tot_g / tot_v if tot_v > 0 else np.nan),
                # 目標對目標的換手，帶索引回傳 —— 要跟凍結那支在**同一組期數**
                # 上比才是同一個量（不然就是在比兩個不同的平均）
                dwabs_tgt=pd.Series(dwabs_tgt, index=w.index),
                mech={k: np.asarray(v, float) for k, v in mech.items()})


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--col", default="new5",
                    help="用哪一臂（預設 new5 = 誠實挑選挑到的那一個）")
    # 費率跟 mft_xs_alpha.py 共用 XS 的凍結表（每個數字都有出處），
    # 預設 bitget 是因為執行路徑是 jarvis -> Bitget。
    # **符號必須是參數，因為誠實挑選挑到的是「new5 配符號 −」**（§1.23b：
    # IC 是負的而權重做多高 z，所以預設符號那一臂不是任何人會下注的東西）。
    # 不給預設成 −1，是因為 D1 的參考式子要跟 §1.23 原樣對照。
    ap.add_argument("--sign", type=float, default=1.0,
                    help="權重符號；§1.23b 誠實挑選挑到 new5 的符號是 -1")
    ap.add_argument("--venue", default="bitget")
    ap.add_argument("--rebate", type=float, default=None,
                    help="返佣比例；不給就用 XS 表裡的預設（bitget 0.5）")
    a = ap.parse_args()

    print("=== R1：條件在成交上的毛利 ===")
    print("「You want to be forecasting conditional on you getting filled.」")
    print("  —— Advanced Market Making, 2025-08-02\n")

    print("抓資料（%d 天）…" % a.days)
    d = XS.load_pairs(a.days)
    f = XS.build(d)
    if f is None or not len(f):
        print("沒有再平衡點，停。")
        return 2
    book = load_book(a.days)
    hours = max(book.minute.nunique() / 60.0, 1)
    print("再平衡 %d 次、%d 標的；簿口 %d 列（每標的每小時 %.1f 筆）\n"
          % (f.minute.nunique(), f.sym.nunique(), len(book),
             len(book) / hours / max(book.sym.nunique(), 1)))

    # ── §1.23 的構造，一個字不改 ──────────────────────────────────
    feat = f.pivot_table(index="minute", columns="sym", values=a.col)
    midp = f.pivot_table(index="minute", columns="sym", values="mid")
    w = XS.weights_from(feat) * float(a.sign)
    midp = midp.reindex(index=w.index, columns=w.columns)
    fwd = midp.shift(-1) / midp - 1.0
    by = prep_book(book)
    days_all = pd.to_datetime(pd.Series(w.index) * 60000, unit="ms", utc=True) \
                 .dt.strftime("%Y-%m-%d")
    days_all.index = w.index

    res = {"asof": time.strftime("%Y-%m-%d %H:%M:%S"), "days": a.days,
           "col": a.col, "sign": float(a.sign), "venue": a.venue,
           "rebalances": int(len(w)), "deltas": {}}
    print("臂 %s、符號 %+.0f%s\n"
          % (a.col, a.sign,
             "（§1.23b 誠實挑選挑到的那一個）" if a.sign < 0 else
             "（**不是**誠實挑選挑到的符號，只當對照）"))

    # ---------- D1：**斷言，不是印出來** ----------
    # 第一版把這一關寫成 print，於是它印出 +387（正確值 ±0.5）之後
    # 程式照樣往下跑，把一整頁荒謬的數字印完。**一道不會擋的守衛
    # 跟沒有守衛一樣**（mistake.md 2026-09-03）。
    # 參考值在**同一份資料上**重算 §1.23 的式子，不是讀凍結檔 ——
    # 讀凍結檔會因為資料窗不同而必不符（mistake.md 2026-09-10）。
    print("=== D1 自曝：全成交 + δ=0 必須逐格重現 §1.23 的無條件毛利 ===")
    both = w.notna() & fwd.notna()
    ref_series = (w.where(both) * fwd.where(both)) \
        .sum(axis=1, min_count=1).dropna() * 1e4
    ref = float(ref_series.mean())
    u = simulate(w, fwd, midp, by, 0.0, assume_fill=True)
    got = float(u["port"].mean())
    # 構造上應該逐格相同，所以門檻釘在浮點誤差，不是「約等於」
    same_n = len(u["port"]) == len(ref_series)
    al = ref_series.align(u["port"], join="inner")
    worst = float((al[0] - al[1]).abs().max()) if len(al[0]) else np.inf
    ok = same_n and worst < 1e-9
    print("  合池：模擬 %+.4f   §1.23 式子 %+.4f" % (got, ref))
    print("  期數：模擬 %d   參考 %d   逐期最大差 %.3e  -> %s"
          % (len(u["port"]), len(ref_series), worst,
             "PASS" if ok else "**FAIL**"))
    res["unconditional"] = got
    res["reference"] = ref
    res["D1"] = bool(ok)
    res["D1_worst_cell"] = worst
    if not ok:
        print()
        print("**D1 未過 —— 部位記帳在全成交下沒有塌回 w，以下全部不解讀。**")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2,
                                  default=str), encoding="utf-8")
        return 2

    # ---------- D1b：有牙齒的那一道 ----------
    print()
    print("=== D1b 自曝：執行優勢的尺度與符號（純算術 + 跨儀器對照）===")
    dl_t = 1.0
    u1 = simulate(w, fwd, midp, by, dl_t, assume_fill=True)
    lift = float(u1["port"].mean()) - got
    want = dl_t * u1["dwabs_valid"]
    ar1 = abs(lift - want) < 1e-6
    print("  δ=%.1f 的毛利增量 %+.6f   δ×Σ|Δw|（同母體）%+.6f   -> %s"
          % (dl_t, lift, want, "PASS" if ar1 else "**FAIL**"))
    # 跨儀器：**在同一組期數上、用同一個定義（目標對目標）** 比，所以應該精確相等。
    _, dw_frozen = XS.turnover(f, a.col)
    tg = u1["dwabs_tgt"].dropna()
    idx = tg.index.intersection(dw_frozen.index)
    mine = float(tg.loc[idx].mean())
    theirs = 2 * float(dw_frozen.loc[idx].mean())
    ar2 = abs(mine - theirs) < 1e-9
    print("  目標對目標換手 %.6f   2 × §1.24 凍結單邊換手 %.6f（同 %d 期）-> %s"
          % (mine, theirs, len(idx), "PASS" if ar2 else "**FAIL**"))
    print("  實際成交量 Σ|Δw| = %.4f  > 上面那個目標對目標的值。"
          % u1["dwabs_all"])
    print("   **差額不是誤差，是約定**：NaN 權重 = 空倉，所以「退出」與")
    print("   「重新進場」各算一筆真的交易，而凍結那支的 `w.diff()` 在 NaN")
    print("   上得到 NaN、整個跳過那些進出。手續費要用這一個，它才是成交量。")
    res["D1b_arith"] = bool(ar1)
    res["D1b_vs_frozen_turnover"] = bool(ar2)
    res["sum_abs_dw_valid"] = u1["dwabs_valid"]
    res["sum_abs_dw_all"] = u1["dwabs_all"]
    res["turnover_target_x2"] = mine
    res["frozen_turnover_x2"] = theirs
    if not ar2:
        print()
        print("**D1b 跨儀器關未過 —— 與 §1.24 的凍結換手率對不上，以下不解讀。**")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2,
                                  default=str), encoding="utf-8")
        return 2
    if not ar1:
        print()
        print("**D1b 算術關未過 —— 執行優勢的符號或尺度寫錯，以下不解讀。**")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2,
                                  default=str), encoding="utf-8")
        return 2
    print()

    # ── 實測半價差：哪幾列的 δ 其實構得到 ──────────────────────────
    hs = ((book.ask - book.bid) / ((book.ask + book.bid) / 2) * 1e4 / 2) \
        .groupby(book.sym).median()
    print("=== 實測半價差（bps，中位）—— δ 小於它的列是掛不到的 ===")
    print("  " + "  ".join("%s %.2f" % (k.replace("-USD", ""), v)
                           for k, v in hs.sort_values().items()))
    print("  全體中位 %.2f bps" % float(hs.median()))
    res["half_spread_bps"] = {k: float(v) for k, v in hs.items()}
    print()

    print("=== 逐 δ：成交率與條件毛利（全格報告，不挑）===")
    print("%6s %8s %8s %10s %10s %10s %10s"
          % ("δ(bps)", "成交率", "判不了", "條件毛利", "SE", "CI下", "CI上"))
    for dl in DELTAS:
        r = simulate(w, fwd, midp, by, dl)
        p = r["port"]
        mu, se, lo, hi = day_boot(p.to_numpy(), days_all.loc[p.index].to_numpy())
        print("%6.1f %7.1f%% %7.1f%% %+10.4f %10.4f %+10.4f %+10.4f"
              % (dl, r["fill_rate"] * 100,
                 100.0 * r["nodata"] / max(r["tries"], 1),
                 mu, se, lo, hi))
        res["deltas"][str(dl)] = dict(fill_rate=r["fill_rate"], gross=mu,
                                      se=se, lo=lo, hi=hi,
                                      fills=r["fills"], tries=r["tries"],
                                      nodata=r["nodata"],
                                      per_vol=r["per_vol"],
                                      total_gross=r["total_gross"],
                                      total_volume=r["total_volume"],
                                      n_scored=r["n_scored"])
    print()

    # ---------- D2 / D3 ----------
    frs = [res["deltas"][str(x)]["fill_rate"] for x in DELTAS]
    mono = all(frs[i] >= frs[i + 1] - 1e-9 for i in range(len(frs) - 1))
    print("=== D2 自曝：成交率必須隨 δ 單調下降 ===")
    print("  " + ("PASS" if mono else
                  "**FAIL —— 成交判定寫錯了，以下不解讀**"))
    print("=== D3 自曝：對照組（零價差）必須成交得比真實更容易 ===")
    print("  第一版的判準（25~80%）寫錯了，見檔頭：50% 是『一小時後』的機率，")
    print("  成交問的是『一小時內曾經』—— 48 步漫步不越過起點只有 8.1%，")
    print("  所以該期待 ≈92%。改用對照組，它對『讀反 bid/ask』才有分辨力。")
    by0 = prep_book(book, zero_spread=True)
    d3 = {}
    for dl in (0.0, 1.0):
        rr = simulate(w, fwd, midp, by0, dl)
        d3[dl] = rr["fill_rate"]
    print("%6s %12s %12s %10s %8s" % ("δ", "真實(跨過來)", "對照(零價差)",
                                      "落差pp", "判定"))
    okd3 = True
    for dl in (0.0, 1.0):
        real, ctrl = res["deltas"][str(dl)]["fill_rate"], d3[dl]
        good = real < ctrl - 1e-12
        okd3 = okd3 and good
        print("%6.1f %11.1f%% %11.1f%% %+10.2f %8s"
              % (dl, real * 100, ctrl * 100, (real - ctrl) * 100,
                 "PASS" if good else "**FAIL**"))
    if not okd3:
        print("  **FAIL —— 真實成交率沒有低於零價差對照，bid/ask 或方向寫反了。**")
    # 無偏漫步的首次通過機率當第三個對照（純算術，不依賴資料）
    n_obs = 48.4
    fp = 1.0 - 1.0 / np.sqrt(np.pi * n_obs)
    print("  旁證：無偏漫步 %0.1f 步的首次通過機率 %.1f%%（實測 δ=0 真實 %.1f%%）"
          % (n_obs, fp * 100, frs[0] * 100))
    res["D2"] = bool(mono)
    res["D3"] = bool(okd3)
    res["D3_control"] = {str(k): float(v) for k, v in d3.items()}
    res["D3_v1_threshold_was_wrong"] = True
    print()

    # ---------- 機制關：沒成交的是不是贏家 ----------
    print("=== 機制：沒成交的那些，本來會賺還是會賠（δ=0，全格報告）===")
    r0 = simulate(w, fwd, midp, by, 0.0)
    mc = r0["mech"]
    nper = max(len(r0["port"]), 1)
    print("%8s %8s %14s %14s %12s"
          % ("", "筆數", "Δw×fwd(這期)", "tgt×fwd(全倉)", "|報酬|bps"))
    for lab, tag in (("成交", "fill"), ("沒成交", "miss")):
        dv, cv, av = mc[tag + "_d"], mc[tag + "_c"], mc[tag + "_a"]
        print("%8s %8d %+14.4f %+14.4f %12.2f"
              % (lab, len(dv), dv.mean() if len(dv) else np.nan,
                 cv.mean() if len(cv) else np.nan,
                 av.mean() if len(av) else np.nan))
    tot = len(mc["fill_d"]) + len(mc["miss_d"])
    lost = float(mc["miss_d"].sum()) / nper if len(mc["miss_d"]) else 0.0
    gap = res["deltas"]["0.0"]["gross"] - got
    print("  沒成交佔 %.1f%%；**那一期直接少掉的**攤到每期 = %+.4f bps/期"
          % (100.0 * len(mc["miss_d"]) / max(tot, 1), lost))
    # ---------- D1c：歸因必須完全閉合（會擋的一致性關） ----------
    # 這是構造上的等式：sim − ref = Σ_{沒成交}(held_舊 − tgt)×fwd
    #                            = −Σ_{沒成交} Δw×fwd
    # 所以殘差必須是 0。**「部位沒補上之後一直錯下去」不是另一項**——
    # 它已經包在 Δw 裡了（Δw 是對**實際部位**算的，上一期沒補，這一期的
    # Δw 就更大）。第一版把殘差講成第二個機制，那是錯的。
    # 殘差不是 0 就代表記帳漏了一塊，所以這一關有牙齒。
    closed = abs(gap + lost) < 1e-9
    print("  全部落差 %+.4f bps/期；歸因殘差 %+.3e -> %s"
          % (gap, gap + lost, "PASS（閉合）" if closed else "**FAIL（記帳漏了一塊）**"))
    res["D1c_attribution_closed"] = bool(closed)
    if not closed:
        print()
        print("**D1c 未過 —— 落差無法被『沒成交的 Δw×fwd』完全解釋，以下不解讀。**")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2,
                                  default=str), encoding="utf-8")
        return 2
    print()
    print("  **這一關不判過不過**，它回答「這個估計量是不是在做我以為的事」。")
    print("  而且要自己戳破一半：**「沒成交的本來會賺」是恆真的，不是發現**")
    print("  —— 買單沒成交 ⟺ 價格從沒跌到掛價 ⟺ fwd>0 ⟺ Δw×fwd>0，賣單鏡像。")
    print("  有內容的是量級：|報酬| %.1f vs %.1f bps（沒成交那堆是行情大的那些，"
          % (mc["miss_a"].mean(), mc["fill_a"].mean()))
    print("  因為「一小時內從沒回頭」的路徑本來就走得遠），以及上面那 %+.2f。"
          % lost)
    res["mech"] = {
        "fill_n": int(len(mc["fill_d"])), "miss_n": int(len(mc["miss_d"])),
        "fill_dw_contrib": float(mc["fill_d"].mean()) if len(mc["fill_d"]) else None,
        "miss_dw_contrib": float(mc["miss_d"].mean()) if len(mc["miss_d"]) else None,
        "fill_tgt_contrib": float(mc["fill_c"].mean()) if len(mc["fill_c"]) else None,
        "miss_tgt_contrib": float(mc["miss_c"].mean()) if len(mc["miss_c"]) else None,
        "fill_absret": float(mc["fill_a"].mean()) if len(mc["fill_a"]) else None,
        "miss_absret": float(mc["miss_a"].mean()) if len(mc["miss_a"]) else None,
        "miss_direct_bps_per_period": float(lost),
        "total_gap_bps_per_period": float(gap),
        "stale_residual_bps_per_period": float(gap + lost)}
    print()

    # ---------- 決策題：掛單省下的吃單費，抵不抵得過被動執行的代價 ----------
    tk, mk, src = XS.fees_for(a.venue, a.rebate)
    print("=== R1 要回答的那個決策：吃單 vs 掛單（一律換算成每單位成交量）===")
    print("  場館 %s（%s）：吃單 %.2f / 掛單 %.2f bps 每邊" % (a.venue, src, tk, mk))
    print("  **為什麼用每單位成交量**：手續費是按成交量收的，而能評分的期數")
    print("  （%d）少於總期數（%d）。拿「每期毛利」減「每期攤的費」會混到"
          % (u["n_scored"], len(w)))
    print("  兩個不同的分母 —— 第一版就是這樣把費低估了 %.2f 倍。"
          % (len(w) / max(u["n_scored"], 1)))
    print()
    print("%10s %12s %12s %12s %12s %9s"
          % ("執行方式", "毛/量", "費率", "淨/量", "淨 bps/h", "成交率"))
    tv_t = u["total_volume"]
    pv_t = u["per_vol"]
    net_t_vol = pv_t - tk
    # 換回 bps/小時只是為了可讀：乘以「每期平均成交量」
    vol_per_h = tv_t / max(u["n_scored"], 1)
    print("%10s %+12.4f %12.2f %+12.4f %+12.4f %9s"
          % ("吃單", pv_t, tk, net_t_vol, net_t_vol * vol_per_h, "100%"))
    best = None
    for dl in DELTAS:
        v = res["deltas"][str(dl)]
        pv = v["per_vol"]
        nv = pv - mk
        v["net_per_vol"] = nv
        v["net_bps_h"] = nv * (v["total_volume"] / max(v["n_scored"], 1))
        print("%10s %+12.4f %12.2f %+12.4f %+12.4f %8.1f%%"
              % ("掛單δ=%.1f" % dl, pv, mk, nv, v["net_bps_h"],
                 v["fill_rate"] * 100))
        if best is None or nv > best[1]:
            best = (dl, nv, v["net_bps_h"])
    res["net_taker_per_vol"] = net_t_vol
    res["net_taker_bps_h"] = net_t_vol * vol_per_h
    res["best_maker"] = {"delta": best[0], "net_per_vol": best[1],
                         "net_bps_h": best[2]}
    print()
    print("  吃單淨 %+.4f   最好的掛單淨 %+.4f（δ=%.1f）   差 %+.4f bps/單位量"
          % (net_t_vol, best[1], best[0], best[1] - net_t_vol))
    print()
    # **這一行才是可以搬到別條線去的東西**：它不再跟這個訊號的強弱有關，
    # 只跟「小時級被動執行」這個執行方式有關。
    pen_unit = pv_t - res["deltas"]["0.0"]["per_vol"]
    print("  **正規化之後的那個常數（可搬到任何小時級再平衡的設計）**：")
    print("    被動執行的代價 = %+.4f − (%+.4f) = **%.2f bps 每單位成交量**"
          % (pv_t, res["deltas"]["0.0"]["per_vol"], pen_unit))
    print("    吃單手續費     = %.2f bps 每單位成交量" % tk)
    print("    -> 當價格接受者便宜 %.1f 倍。這個比值與訊號好壞無關，"
          % (pen_unit / tk if tk else np.nan))
    print("       它是「一小時只掛一次、不改價」這個執行方式的性質。")
    res["passive_penalty_bps_per_unit_volume"] = float(pen_unit)
    res["taker_bps_per_unit_volume"] = float(tk)
    print()

    print("=== 讀法 ===")
    print("  δ=0 是掛在 mid，**那是一個不存在的掛法**（mid 在價差中間），")
    print("  只用來跟無條件毛利對照。真正可掛的最小值是半個價差（上表）。")
    print("  兩個數要一起看：**成交率**決定你做得到幾筆，")
    print("  **條件毛利**決定那幾筆值多少 —— 而後者才是他說的那個估計量。")
    print("  毛利隨 δ 變大有兩個相反的力：拿到的價格更好（+δ×換手），")
    print("  但成交變成更偏的樣本（只在行情走向不利時才成交）。淨值是哪邊")
    print("  贏，就是這張表要回答的事。")
    print()
    print("  **本支測的是什麼、沒測什麼 —— 不要把它讀成「做市不可行」：**")
    print("  測了：把**同一個每小時的目標部位**改用限價單被動執行（省掉吃單")
    print("        手續費）之後，毛利變成多少。")
    print("  沒測：真正的倉位型做市 —— **雙邊同時報價、用訊號偏移報價、靠來回")
    print("        賺價差**。那個損益結構不同（兩邊都收價差、庫存才是風險），")
    print("        本支的單邊一次性掛單不是它的近似。")
    print("  所以這張表能否證的是「用掛單執行現行的小時級目標」，")
    print("  不是「HFT/MFT 訊號當報價偏移」那條路（§1.19 講的那一條）。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print("\nwritten -> %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
