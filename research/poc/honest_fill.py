# -*- coding: utf-8 -*-
"""
========================================================================
2026-09-07 **H1 CONFIRMED、H2 修正後為負。這條線的歷史 edge 是成交假設。**
========================================================================
    凍結（現行）            7,044   **+0.0365**  [+0.0114, +0.0612]   9/9 幣
    乾淨臂（A 照凍結+B1）   7,044   **−0.0483**  [−0.0721, −0.0241]   **0/9 幣**
      其中 A 情境（42.1%，R 未動）  −0.0435  1/9
      其中 B 情境（57.9%）          −0.0518  1/9
    B2 掛單等回踩            6,176   −0.0528  成交率 87.7%  每事件 **−0.0463**

    H1 乾淨 − 凍結 = **−0.0848**  CI [−0.0905, −0.0795]  CONFIRMED-樂觀
    H2 **修正後為負** —— 逐幣 ADA −0.018 / AVAX −0.040 / BNB −0.027 /
       BTC −0.014 / DOGE −0.089 / ETH −0.057 / LINK −0.073 / SOL −0.047 /
       XRP −0.071，**九個幣全負**
    H3 PASS（A 情境逐筆 max|ΔR| = 0，沒動到不該動的）

分解出來的結構（把 A/B 拆開才看得到）
    A 情境 凍結 R  **−0.0435**（42.1%，完全沒動過的凍結數字）
    B 情境 凍結 R  **+0.0946**（57.9%）
    加權還原 +0.0365 ✓
    -> **凍結引擎的全部利潤都在 B 情境。**

B 情境是什麼、壞在哪（**用詞要精確**）
    買側被掃（價格向上穿過前高），而掃單 bar **自己收回到價位下方** ——
    就是這條策略命名的那個「掃單失敗」。**論文假設在這裡是成立的**：
    回踩（收回內側）確實發生了，只是發生在掃單 bar 那一根之內。

    壞掉的**只有進場價**：引擎在下一根記「成交在價位」，但市場早就在
    價位之外 **中位 42.6 bps**。對做空來說那是賣在市價之上 —— 拿不到。
    42.6 bps 是凍結成本模型（7~10 bps/腿）的 **4~6 倍**。

    換算：42.6 bps ÷ (3.5 × ATR) ≈ 0.085 R，**正好等於 H1 量到的落差**。
    獨立佐證：`engine_audit` 量到穿透深度中位 0.525 ATR，收盤回到另一側
    0.30 ATR 完全在同一量級。

    所以缺陷是**進場價**不是訊號邏輯。這個區別決定後續怎麼修：
    要救這條線，得換一個**真的拿得到的進場價**，不是改訊號。

不受本判決影響的東西（範圍要講清楚）
    · `TRIAGE.md` 的交會事件是量**價格移動**（收盤到收盤），不經過引擎的
      成交假設，所以不受影響；它的前瞻時鐘（`conj_clock.py`，0/300）照跑。
    · 受影響的：Gate F 的前瞻記帳（同一個引擎）、回測檢視器顯示的進場價、
      以及所有以凍結 R 為單位的既有結論。
誠實執行下的完整凍結規則 —— 把 `resting_limit.py` 留下的時點偏差補掉。

`resting_limit.py`（2026-09-07）判出 Q1 CONFIRMED-樂觀，但它的 MARKETABLE
臂在**掃單 bar 就成交**，而凍結規則是**等回踩才進場** —— 出場窗因此不同。
所以那支的「誠實執行 = −0.055」帶著一個時點差異，不是凍結規則的乾淨重算。
本檔補的就是這一格。

===========================================================================
「誠實」有兩種讀法，而引擎自己指定了一種
===========================================================================
`sweep_core` 的註解寫死：`entry = lvl + d*SLIP*A  # stop-entry slippage
against us`。**它是停損單（觸發後市價成交），不是限價單。** 照這個讀法，
兩種處境的誠實程度完全不同：

  A 情境　掃單 bar 收盤在價位的**外**側（穿越那一側）
          價格要從外面走回價位。賣停單掛在 lvl，價格跌到 lvl 才觸發，
          成交 ≈ lvl 再付滑價。**凍結的做法在這裡是誠實的。**
          （d=-1 做空：cl[j] > lvl；d=+1 做多：cl[j] < lvl）

  B 情境　掃單 bar 收盤已經在價位的**內**側（59.3% 的交易）
          賣停單**早就被觸發**了，成交在**當下的市價**，而市價已經比 lvl
          差。**凍結假設「成交在 lvl」在這裡拿不到。**

所以乾淨的臂 = A 照凍結、B 用真實可成交價。缺陷被隔離在 B，
不摻進 `resting_limit` 那個「提早進場」的時點差異。

B 情境的可成交價（兩種，都報）
  B1 下一分鐘收盤　掃單 bar 收盤後的第一根 1 分鐘 K 的收盤價。
                   這是「訊號一成立就送市價單」實際拿得到的價。
                   與 `entry_decomp` 的 B 臂同一個口徑。
  B2 掛單等回踩　　在 lvl 掛限價，只有價格回到 lvl 才成交，回踩窗內
                   沒回來就**不成交**（沒有部位）。＝ `resting_limit` 的
                   RESTING 模型，但這裡只套用在 B 情境。

出場一律是**完整凍結規則**：3.5 ATR 災難停損（從成交 bar 的下一根開始查）
＋ HOLD=8 小時時間出場。SLIP 一個字不動。

===========================================================================
判準（跑之前寫死，寫在 CI 上不寫在點估計上）
===========================================================================
    H1 缺陷有多大（主判準）
        乾淨臂（A 照凍結 + B1）與凍結的**配對差**，日聚類 CI **上緣 < 0**
        -> 凍結在 B 情境上的樂觀是實質的，差值就是修正量
        CI 含零 -> INCONCLUSIVE
    H2 修正後這條策略還活著嗎
        乾淨臂的 meanR 日聚類 CI **下緣 > 0** -> 仍為正
        CI 上緣 < 0 -> **修正後為負，這條線的歷史 edge 是成交假設造出來的**
        含零 -> INCONCLUSIVE（不得宣稱任一方向）
    H3 A 情境必須逐筆等於凍結（已知答案的對照）
        A 情境的 R 與凍結 R 的 max|Δ| < 1e-9。
        不等於 -> 我的重算動到了不該動的地方，H1/H2 一律不解讀。
    H4 B2 並報
        掛單版的 meanR 與成交率一起印，不挑對自己有利的那個報。
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
sys.path.insert(0, str(HERE.parents[0] / "sweep_failure"))
import sweep_core as sc  # noqa: E402

BARS = HERE / "data" / "bars"
CACHE = HERE.parents[0] / "sweep_failure" / ".cache"
OUT = HERE / "data" / "results"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
HOUR_MS = 3_600_000
RNG = np.random.default_rng(20260907)


def exit_from(b1_h, b1_l, b1_c, n, f_bar, d, entry, risk, A):
    """完整凍結出場：停損從 f_bar+1 開始查，否則 HOLD 根後收盤。"""
    stop = entry - d * risk
    for qq in range(f_bar + 1, min(f_bar + sc.HOLD + 1, n)):
        if (d == 1 and b1_l[qq] <= stop) or (d == -1 and b1_h[qq] >= stop):
            return -1.0 - sc.SLIP / sc.DIS
    exb = min(f_bar + sc.HOLD, n - 1)
    return d * (b1_c[exb] - d * sc.SLIP * A - entry) / risk


def main():
    rows = []
    for sym in CORE9:
        b1 = sc.load_csv(str(CACHE / f"{sym}USDT_1h.csv"))
        h = [x[sc.H] for x in b1]
        lo = [x[sc.L] for x in b1]
        cl = [x[sc.C] for x in b1]
        n = len(b1)

        m = pd.read_parquet(BARS / f"{sym}.parquet",
                            columns=["ts", "high", "low", "close"])
        mts = m["ts"].to_numpy(np.int64)
        mhi = np.nan_to_num(m["high"].to_numpy(float), nan=-np.inf)
        mlo = np.nan_to_num(m["low"].to_numpy(float), nan=np.inf)
        mcl = m["close"].to_numpy(float)
        nm = len(mts)

        for e in sc.backtest_symbol(b1, detail=True):
            j, lvl, A, d, risk = (e["j"], e["level"], e["atr"], e["d"], e["risk"])
            fill = e["fill"]
            t0 = int(b1[j][0]) * 1000 + HOUR_MS      # 掃單 bar 收盤時刻
            t1 = int(b1[min(j + sc.W, n - 1)][0]) * 1000 + HOUR_MS
            i0 = int(np.searchsorted(mts, t0, side="left"))
            i1 = int(np.searchsorted(mts, t1, side="left"))
            if i0 >= nm or i1 <= i0:
                continue

            # 情境：掃單 bar 收盤在價位的哪一側
            # d=-1 做空：收在 lvl 之上 = 外側(A)；收在之下 = 內側(B)
            inside = (cl[j] < lvl) if d == -1 else (cl[j] > lvl)

            r_frozen = float(e["R"])
            if not inside:                            # A 情境：照凍結
                r_clean = r_frozen
                r_b2 = r_frozen
                filled_b2 = 1
                entry_used = lvl + d * sc.SLIP * A
            else:                                     # B 情境
                # B1：掃單 bar 收盤後第一根 1 分鐘 K 的收盤 = 實際可成交價
                px = float(mcl[i0])
                entry_used = px + d * sc.SLIP * A
                # 成交發生在**掃單 bar 收盤後的第一分鐘**，那一分鐘屬於 j+1，
                # 所以出場排程要從 j+1 起算（停損從 j+2 查、時間出場 j+1+HOLD）。
                # 第一版誤用 j，讓停損早查一根、時間出場也早一根 —— 那正是本檔
                # 要修掉的那種時點偏差，不能自己再犯一次。
                f_b1 = min(j + 1, n - 1)
                r_clean = exit_from(h, lo, cl, n, f_b1, d, entry_used, risk, A)
                # B2：在 lvl 掛限價，等價格回到 lvl；回踩窗內沒回來就不成交
                if d == -1:
                    hit = np.flatnonzero(mhi[i0:i1] >= lvl)
                else:
                    hit = np.flatnonzero(mlo[i0:i1] <= lvl)
                if len(hit):
                    k = i0 + int(hit[0])
                    hts = np.array([int(x[0]) for x in b1], np.int64) * 1000
                    fb = int(np.searchsorted(hts, int(mts[k]), side="right")) - 1
                    if 0 <= fb and fb + 1 < n:
                        r_b2 = exit_from(h, lo, cl, n, fb, d,
                                         lvl + d * sc.SLIP * A, risk, A)
                        filled_b2 = 1
                    else:
                        r_b2, filled_b2 = np.nan, 0
                else:
                    r_b2, filled_b2 = np.nan, 0

            rows.append(dict(
                sym=sym, inside=bool(inside), R_frozen=r_frozen,
                R_clean=float(r_clean), R_b2=float(r_b2) if r_b2 == r_b2 else np.nan,
                filled_b2=int(filled_b2), entry=float(entry_used), lvl=float(lvl),
                day=pd.Timestamp(int(b1[j][0]) * 1000, unit="ms",
                                 tz="UTC").strftime("%Y-%m-%d")))

    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "honest_fill.parquet", index=False)

    def day_ci(x, days, b=2000):
        x = np.asarray(x, float)
        ok = np.isfinite(x)
        x, days = x[ok], np.asarray(days)[ok]
        if len(x) < 30:
            return (float("nan"),) * 4
        uq, inv = np.unique(days, return_inverse=True)
        ix = [np.where(inv == k)[0] for k in range(len(uq))]
        reps = np.empty(b)
        for i in range(b):
            p = RNG.integers(0, len(uq), len(uq))
            reps[i] = x[np.concatenate([ix[k] for k in p])].mean()
        return (float(x.mean()), float(np.percentile(reps, 2.5)),
                float(np.percentile(reps, 97.5)), float(np.std(reps, ddof=1)))

    days = d.day.to_numpy()
    print(f"凍結交易 {len(d):,} 筆   "
          f"A 情境（收在外側，照凍結）{(~d.inside).sum():,} "
          f"({(~d.inside).mean()*100:.1f}%)   "
          f"B 情境（收在內側，凍結價拿不到）{d.inside.sum():,} "
          f"({d.inside.mean()*100:.1f}%)")
    print()

    # H3 已知答案的對照：A 情境必須逐筆等於凍結
    a = d[~d.inside]
    worst = float(np.nanmax(np.abs(a.R_clean - a.R_frozen))) if len(a) else 0.0
    h3 = worst < 1e-9
    print(f"H3 A 情境逐筆 max|ΔR| = {worst:.2e}（需 <1e-9）-> "
          f"{'PASS' if h3 else '**FAIL — 重算動到不該動的地方，以下不解讀**'}")
    print()

    print(f"{'臂':26s} {'n':>7s} {'meanR':>9s} {'日聚類 CI95':>24s} {'逐幣為正':>9s}")
    res = {"n": int(len(d)), "share_inside": float(d.inside.mean()), "H3": bool(h3)}
    for name, col, sel in (("凍結（現行）", "R_frozen", np.ones(len(d), bool)),
                           ("乾淨臂（A 凍結 + B1）", "R_clean", np.ones(len(d), bool)),
                           ("  其中 A 情境", "R_clean", (~d.inside).to_numpy()),
                           ("  其中 B 情境", "R_clean", d.inside.to_numpy())):
        g = d[sel]
        mm, ll, hh, se = day_ci(g[col].to_numpy(), days[sel])
        pc = g.groupby("sym")[col].mean()
        res[name.strip()] = dict(n=int(len(g)), mean=mm, ci=[ll, hh], se=se,
                                 coins_pos=int((pc > 0).sum()))
        print(f"{name:26s} {len(g):7,d} {mm:+9.4f}  [{ll:+.4f}, {hh:+.4f}] "
              f"{int((pc>0).sum()):>7d}/9")

    # H4 B2 掛單版
    b2 = d[d.filled_b2 == 1]
    mm2, ll2, hh2, _ = day_ci(b2.R_b2.to_numpy(), b2.day.to_numpy())
    fr = float(d.filled_b2.mean())
    print(f"{'B2 掛單等回踩（並報）':26s} {len(b2):7,d} {mm2:+9.4f}  "
          f"[{ll2:+.4f}, {hh2:+.4f}]   成交率 {fr*100:.1f}%")
    res["B2"] = dict(n=int(len(b2)), mean=mm2, ci=[ll2, hh2], fill_rate=fr,
                     per_event=mm2 * fr)
    print(f"{'':26s} {'':>7s} 每事件期望 {mm2*fr:+.4f}")

    print()
    print("=== 預註冊判準 ===")
    print()
    diff = (d.R_clean - d.R_frozen).to_numpy(float)
    md, dlo, dhi, _ = day_ci(diff, days)
    v1 = ("CONFIRMED-樂觀" if dhi < 0 else
          "**反向 — 停手查儀器**" if dlo > 0 else "INCONCLUSIVE")
    print(f"H1 乾淨 − 凍結 = {md:+.4f}  CI [{dlo:+.4f}, {dhi:+.4f}]  -> {v1}")
    res["H1"] = dict(diff=md, ci=[dlo, dhi], verdict=v1)

    c = res["乾淨臂（A 凍結 + B1）"]
    v2 = ("PASS（修正後仍為正）" if c["ci"][0] > 0 else
          "**修正後為負 — 歷史 edge 是成交假設造出來的**" if c["ci"][1] < 0
          else "INCONCLUSIVE（修正後含零）")
    print(f"H2 乾淨臂 meanR {c['mean']:+.4f} CI "
          f"[{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]  -> {v2}")
    res["H2"] = dict(verdict=v2)

    (OUT / "honest_fill.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print()
    print("written ->", OUT / "honest_fill.json")


if __name__ == "__main__":
    main()
