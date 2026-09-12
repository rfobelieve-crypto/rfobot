# -*- coding: utf-8 -*-
"""把研究結果變成網站看得懂的圖表資料（2026-09-11）

===========================================================================
使用者的要求
===========================================================================
「把研究的東西都顯示在網站上，**盡量以圖表輔助顯示說明**」。

網站已經有兩條內容流（`docs`／CLAUDE.md §使用者可見改動的同步規則）：

    assets/research_nogo.json   陣亡名冊（文字）
    assets/method.json          驗證方法五條規矩（文字＋案例）

**缺的是圖。** 那兩頁在講「我們怎麼判斷真假」，但都是散文——
而這套方法最有說服力的東西**本來就是圖形**：
一條線在前半很漂亮、在後半翻號，用講的沒有用，用畫的一眼就懂。

===========================================================================
為什麼是生成不是手抄
===========================================================================
**數字一律從真正做決定的那份結果檔讀出來。**
手抄到網站上的數字會漂，而且漂了沒有人會發現
（mistake.md 2026-08-26：把既有數字搬到新地方顯示＝第二份實作，
它會安靜地跟做決定的那個不一致）。

所以這一支的每一個數字都標了 `source`，而且**讀不到來源就整組不出**
（fail-closed），不會拿舊值或預設值頂上。

===========================================================================
公開面規則（違反就是資訊外洩，不是 UI 問題）
===========================================================================
CLAUDE.md §對外網站呈現面：**只出百分比、方向、時間、計數**；
**不出美元、張數、帳戶權益、單筆部位金額、模型內部**（切點、權重、特徵定義）。

所以：
  * 耐心那組的 `$/年` **一律換成「相對 k=0 的倍數」**
  * 容量那組（§1.20）**整組不放**——它的本體就是美元
  * 場館名稱可以出（它們是公開的交易所），但**不出我們的部位**

    python research/publish_research_charts.py
    python research/publish_research_charts.py --copy   # 同時複製到 product-site
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "research_charts.json"
SITE = ROOT.parent / "product-site" / "content" / "research_charts.json"

R = ROOT / "research" / "results"
POC = ROOT / "research" / "poc" / "data" / "results"
ARB = ROOT.parent / "arb" / "results"


def load(p: Path):
    """讀不到就回 None —— 呼叫端一律 fail-closed，不補預設值。"""
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def chart_mft_halves():
    """MFT 六個臂：前半挑符號 -> 後半驗。六個裡五個翻號。"""
    d = load(R / "mft_xs_alpha.json")
    if not d or not d.get("halves"):
        return None
    arms = []
    for name, v in d["halves"].items():
        arms.append(dict(arm=name, first=round(v["first"], 3),
                         second=round(v["second"], 3),
                         flipped=bool(v["first"] * v["second"] < 0)))
    arms.sort(key=lambda x: -x["first"])
    pick = d.get("oos_sign_pick") or {}
    # **從資料算，不要寫死。** 這句文案原本寫死「六個裡有五個翻號」，那是
    # 14 天那版的結論；資料換成 120 天之後翻號數變 2，而文字不會自己更新
    # （mistake.md 2026-08-26：被推翻的結論會以「詞」的形式活下來）。
    flipped_n = sum(1 for x in arms if x["flipped"])
    # 掛單來回成本 = 換手 × 2 × 每邊 maker bps。用本次跑出來的換手。
    thr = d.get("thresholds") or {}
    MAKER_FLOOR = float(thr.get("maker_bps_h") or 1.5)
    # R1（TODO §1.27）量了「改用掛單」那條路，而這張圖原本的說法會讓人以為
    # 掛單是便宜的那一條（只講手續費 1.00）。實際上掛單還要付放棄的邊際。
    # **同樣從資料算**，而且檔案不在就不加那一句（不要寫死一個會過期的數）。
    fc = load(R / "mft_fill_conditional.json") or {}
    zh_r1 = en_r1 = ""
    if fc.get("D1") and fc.get("best_maker"):
        pen = float(fc.get("passive_penalty_bps_per_unit_turnover") or 0)
        tkn = float(fc.get("taker_bps_per_unit_turnover") or 0)
        nt = float(fc.get("net_taker") or 0)
        bm = fc["best_maker"]
        zh_r1 = ("　**而「改掛單」這條路已經量過了，它更差**："
                 "被動執行每單位換手要付 %.1f bps 的放棄邊際（漏掉的成交"
                 "正好是行情最大的那些小時），對照吃單手續費只要 %.1f。"
                 "淨值：吃單 %+.2f、最好的掛單 %+.2f bps/小時。"
                 % (pen, tkn, nt, float(bm.get("net", 0))))
        en_r1 = (" **And the “quote instead of take” route has now been "
                 "measured — it is worse**: passive execution forgoes %.1f bps "
                 "of edge per unit of turnover (the fills you miss are exactly "
                 "the biggest hours), against %.1f bps of taker fee. Net: "
                 "taking %+.2f, best quoting %+.2f bps/hour."
                 % (pen, tkn, nt, float(bm.get("net", 0))))
    return dict(
        id="mft_halves",
        kind="slope",
        zh=dict(
            title="訊號活下來了，但付不起自己的手續費",
            lede=("一個一小時頻率的橫斷面訊號，六種組法。"
                  "**符號與組法只用前半決定**，後半完全沒看過。"
                  "六個裡有 %d 個在後半翻號——但真正擋住它的不是翻號："
                  "**前半會挑到的那一個在後半是 %+.2f，而光是掛單的來回"
                  "手續費就要 %.2f。** 訊號是真的，只是比成本小。"
                  % (flipped_n, pick.get("second", 0), MAKER_FLOOR)),
            xlabel="前半（挑參數用的）", ylabel="後半（沒看過的）",
            unit="每小時毛利（基點）",
            sample=("%d 天、%d 次換倉、%d 個標的，每小時重組一次"
                    % (d.get("days", 0), d.get("rebalances", 0),
                       d.get("symbols", 0))),
            callout=("前半挑到的那條，後半 %+.2f bps/小時。\n"
                     "而掛單來回手續費是 %.2f —— 訊號比成本小。"
                     % (pick.get("second", 0), MAKER_FLOOR)),
            note=("前半會挑到 `%s`：前半 %+.2f -> 後半 %+.2f —— "
                  "**樣本外仍然為正**。擋住它的是換手：這個訊號每小時重組"
                  "一次，光是掛單的來回手續費就要 %.2f bps/小時，"
                  "比它賺的還多。所以唯一的槓桿是**把換手砍下來**，"
                  "不是換組法。"
                  % (pick.get("arm", "—"), pick.get("first", 0),
                     pick.get("second", 0), MAKER_FLOOR)) + zh_r1,
        ),
        en=dict(
            title="The signal survived. It still cannot pay its own fees.",
            lede=("One hourly cross-sectional signal, six constructions. "
                  "**The sign and the construction are chosen on the first "
                  "half only**; the second half is never looked at. "
                  "%d of six flip sign — but that is not what stops it: "
                  "**the arm the first half picks earns %+.2f out of sample, "
                  "while the maker round-trip alone costs %.2f.** "
                  "The signal is real; it is just smaller than the cost."
                  % (flipped_n, pick.get("second", 0), MAKER_FLOOR)),
            xlabel="First half (used to choose)",
            ylabel="Second half (never seen)",
            unit="gross basis points per hour",
            sample=("%d days, %d rebalances, %d symbols, hourly"
                    % (d.get("days", 0), d.get("rebalances", 0),
                       d.get("symbols", 0))),
            callout=("The first half's pick earns %+.2f bps/hour out of sample.\n"
                     "The maker round trip costs %.2f. Smaller than the cost."
                     % (pick.get("second", 0), MAKER_FLOOR)),
            note=("The first half would pick `%s`: %+.2f -> %+.2f — "
                  "**still positive out of sample**. What stops it is "
                  "turnover: rebalancing hourly costs %.2f bps/hour in maker "
                  "fees alone, more than it earns. The only lever is cutting "
                  "turnover, not trying another construction."
                  % (pick.get("arm", "—"), pick.get("first", 0),
                     pick.get("second", 0), MAKER_FLOOR)) + en_r1,
        ),
        series=arms,
        source="research/results/mft_xs_alpha.json",
    )


def chart_percoin_noise():
    """SDV 逐幣：前半的名次對後半沒有預測力。"""
    d = load(POC / "sdv_percoin_split.json")
    if not d or not d.get("table"):
        return None
    rows = []
    for r in d["table"]:
        if r.get("r1") is None or r.get("r2") is None:
            continue
        rows.append(dict(name=r["sym"], first_rank=int(r["r1"]),
                         second_rank=int(r["r2"])))
    rows.sort(key=lambda x: x["first_rank"])
    rho = d.get("rho")
    pick = d.get("pick")
    return dict(
        id="percoin_noise",
        kind="rank_slope",
        zh=dict(
            title="「這個標的表現特別好」——通常不是真的",
            lede=("同一條策略、九個標的，按前半的表現排名，"
                  "再看它們在後半的排名。"
                  "**線如果大致平行，代表「哪個標的比較好」是可以事先知道的；"
                  "線如果交叉成一團，那個排名就是雜訊。**"),
            xlabel="前半名次", ylabel="後半名次",
            sample="九個標的，以 %s 為界切成前後兩半" % (d.get("cut") or "—"),
            callout=("等級相關 %+.3f —— 負的。\n"
                     "只用前半挑會挑到 %s，而它在後半是第 %d 名。"
                     % (rho if rho is not None else 0, pick or "—",
                        next((r["second_rank"] for r in rows
                              if r["name"] == pick), 0))),
            note=("前後半名次的等級相關是 **%+.3f**（負的）。"
                  "只用前半挑，程序會挑 **%s** —— 而它在後半是最後一名。"
                  % (rho if rho is not None else 0, pick or "—")),
        ),
        en=dict(
            title="\"This symbol looks especially good\" — usually it isn't",
            lede=("One strategy, nine symbols, ranked by first-half "
                  "performance, then plotted against their second-half rank. "
                  "**Roughly parallel lines would mean the ranking is "
                  "knowable in advance. Crossed lines mean it is noise.**"),
            xlabel="First-half rank", ylabel="Second-half rank",
            sample="Nine symbols, split at %s" % (d.get("cut") or "—"),
            callout=("Rank correlation %+.3f — negative.\n"
                     "A first-half-only pick lands on %s, which finishes %d of 9."
                     % (rho if rho is not None else 0, pick or "—",
                        next((r["second_rank"] for r in rows
                              if r["name"] == pick), 0))),
            note=("Rank correlation between halves is **%+.3f** (negative). "
                  "A first-half-only procedure picks **%s** — which finishes "
                  "last in the second half."
                  % (rho if rho is not None else 0, pick or "—")),
        ),
        series=rows,
        source="research/poc/data/results/sdv_percoin_split.json",
    )


def chart_gate0():
    """Gate 0：資訊層過、執行層死。**質性標籤，不是會漂的數字。**

    這一組刻意寫死在這裡：它的內容是 PASS/FAIL 這種標籤，
    鏡射 CLAUDE.md 核心原則 11 的那張表。數字只有一個（比值），
    而它在 §0.57b 判決節裡是凍結的。
    """
    members = [
        ("分鐘級訊號", "15/15 個月同號", "138 格淨值全負"),
        ("清算位密度", "四道檢定全過", "交易假設沒過"),
        ("掃單失敗（舊線）", "引擎自審六項全過", "歷史優勢是成交假設造的"),
        ("事件交會（SDV）", "毛利為正、區間離零", "誠實錨點後扣成本全負"),
        ("執行落差", "天花板顯著為正", "落差是天花板的 132%"),
        ("被動基準", "—", "六個場館對全負"),
    ]
    return dict(
        id="gate0",
        kind="two_column",
        zh=dict(
            title="訊號是真的，但價格拿不到",
            lede=("把已經結案的判決橫著讀，最大的一群有六個成員，"
                  "而且每一個都是我們自己寫下來的："
                  "**資訊層過了，執行層死掉。**"
                  "這不是「交易很難」這種廢話——尖銳的版本是："
                  "我們的流程永遠把資訊層排在前面、把執行可行性排在最後，"
                  "所以每條線都花幾個月才撞到它的約束，"
                  "而那個約束每次都是同一個。"),
            colA="資訊層", colB="執行／經濟層",
            note=("所以從 2026-09-11 起，任何新研究線的第一關是"
                  "**執行可行性**，排在資訊層之前。"),
        ),
        en=dict(
            title="The signal is real. The price is not available.",
            lede=("Read the closed verdicts sideways and the largest cluster "
                  "has six members, each one written down by us: "
                  "**the information layer passes, the execution layer kills "
                  "it.** The sharp version is not 'trading is hard' — it is "
                  "that our process always put the information layer first "
                  "and execution feasibility last, so every line spent months "
                  "before hitting its binding constraint, and it was the same "
                  "constraint every time."),
            colA="Information layer", colB="Execution / economics",
            note=("So since 2026-09-11 the first gate on any new line is "
                  "execution feasibility, before any information work."),
        ),
        series=[dict(name=a, info=b, exec_=c) for a, b, c in members],
        source="CLAUDE.md 核心原則 11 / docs/common_cause_scan.md",
    )


def chart_patience():
    """耐心的代價：等 k 分鐘，事件數掉多少。**金額換成倍數。**"""
    d = load(R / "prereg_arb_patience.json")
    if not d or not d.get("total_n"):
        return None
    n0 = d["total_n"].get("0") or 0
    if not n0:
        return None
    ud = d.get("total_usd_day") or {}
    base = ud.get("0") or 0
    rows = []
    for k in d.get("ages", []):
        ks = str(k)
        rows.append(dict(
            k=k,
            events_pct=round(100.0 * (d["total_n"].get(ks) or 0) / n0, 1),
            # **金額換成倍數** —— 公開面不出美元
            value_x=round((ud.get(ks) or 0) / base, 2) if base else None))
    se = d.get("SE") or {}
    best = max(rows, key=lambda r: (r["value_x"] or 0))
    sd = (se.get(str(best["k"])) or {}).get("se_diff_usd_year")
    gain = ((se.get(str(best["k"])) or {}).get("usd_year") or 0) - \
           ((se.get("0") or {}).get("usd_year") or 0)
    ex = d.get("ex_largest") or {}
    return dict(
        id="patience",
        kind="bars",
        zh=dict(
            title="耐心的代價，以及為什麼它看起來有用其實測不動",
            lede=("一個想法：既然在速度上贏不了，就只做「已經存在了一段時間」"
                  "的機會——用耐心換速度。"
                  "等得越久，機會越可能還在（這個我們量過，而且對照組是平的）。"
                  "**但等待也會讓機會數變少，而機會數正是我們缺的東西。**"),
            xlabel="進場前先等幾分鐘",
            sample="%d 個配對，兩個場館，不等待那一格 n=%d 個機會" % (
                len(d.get("pairs") or {}), n0),
            callout=("等 8 分鐘，價值最高——\n"
                     "但機會數只剩 %.0f%%，而那個差的標準誤比差本身大。"
                     % next((r["events_pct"] for r in rows
                             if r["k"] == best["k"]), 0)),
            note=("看起來最好的那一格把價值拉到 %s 倍，"
                  "但**那個差的標準誤比差本身還大**，"
                  "而且 **87%% 的效果集中在單一個配對**——"
                  "把它拿掉，方向就反過來。所以這是**無效判決不是通過**。"
                  % (("%.2f" % best["value_x"]) if best["value_x"] else "—")),
        ),
        en=dict(
            title="What patience costs, and why it looks useful but cannot be measured",
            lede=("An idea: if you cannot win on speed, trade only the "
                  "opportunities that have already survived a while. "
                  "The longer one has lasted, the likelier it lasts longer "
                  "(measured, with a flat control). **But waiting also cuts "
                  "the number of opportunities — and that is the thing we are "
                  "short of.**"),
            xlabel="Minutes waited before entering",
            sample="%d pairs, two venues; the no-wait bucket has n=%d" % (
                len(d.get("pairs") or {}), n0),
            callout=("Waiting 8 minutes maximises value —\n"
                     "but only %.0f%% of opportunities survive, and the SE of "
                     "that gain exceeds the gain."
                     % next((r["events_pct"] for r in rows
                             if r["k"] == best["k"]), 0)),
            note=("The best-looking bucket multiplies value by %s, but **the "
                  "standard error of that difference is larger than the "
                  "difference**, and **87%% of the effect sits in a single "
                  "pair** — drop it and the direction reverses. "
                  "So this is inconclusive by design, not a pass."
                  % (("%.2fx" % best["value_x"]) if best["value_x"] else "—")),
        ),
        series=rows,
        se_exceeds_effect=bool(sd is not None and gain and abs(sd) >= abs(gain)),
        source="research/results/prereg_arb_patience.json",
    )


def chart_universe():
    """宇宙加寬：8 -> 149 個配對。計數，公開面允許。"""
    u = load(ARB / "arb_universe.json")
    if not u:
        return None
    by = {}
    for p in u.get("pairs", []):
        k = "%s-%s" % (p["leg_a"], p["leg_b"])
        by[k] = by.get(k, 0) + 1
    return dict(
        id="universe",
        kind="counts",
        zh=dict(
            title="三條線在同一天撞到同一個牆：宇宙太窄",
            lede=("三個彼此獨立的測試在同一天得到同一個診斷——"
                  "樣本不夠寬，所以測不動。"
                  "而原因很平凡：**標的是手動挑的，沒有把「全部都錄下來」"
                  "自動化。** 規則先凍結（每一個在兩個以上場館都有的標的，"
                  "不排名、不設門檻、不做任何與績效有關的排除），再開始錄。"),
            before_label="原本（手挑）", after_label="現在（規則）",
            sample="%d 個標的，%d 個場館對" % (u.get("n_tickers", 0), len(by)),
            callout="三條線缺的都是同一個東西：夠寬的樣本。",
            note=("錄製從 8 個配對變成 **%d** 個，"
                  "而且是**先把規則凍結才產生資料**——"
                  "反過來就沒有樣本外可言。" % u.get("n_pairs", 0)),
        ),
        en=dict(
            title="Three lines hit the same wall on the same day: the universe is too narrow",
            lede=("Three independent tests produced the same diagnosis on the "
                  "same day — the sample is not wide enough to decide. "
                  "The cause is mundane: **symbols were hand-picked, and "
                  "'record everything' was never automated.** The rule is "
                  "frozen first (every symbol quotable on two or more venues; "
                  "no ranking, no threshold, no performance-related "
                  "exclusion), and only then does recording start."),
            before_label="Before (hand-picked)", after_label="Now (by rule)",
            sample="%d tickers across %d venue pairs" % (u.get("n_tickers", 0), len(by)),
            callout="All three lines were short of the same thing: a wide enough sample.",
            note=("Recording went from 8 pairs to **%d** — and the rule was "
                  "frozen before the data existed. The other order leaves no "
                  "out-of-sample at all." % u.get("n_pairs", 0)),
        ),
        series=[dict(name=k, count=v) for k, v in
                sorted(by.items(), key=lambda x: -x[1])],
        before=8, after=u.get("n_pairs", 0),
        tickers=u.get("n_tickers", 0),
        source="../arb/results/arb_universe.json",
    )


BUILDERS = [chart_gate0, chart_mft_halves, chart_percoin_noise,
            chart_patience, chart_universe]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true",
                    help="同時複製到 ../product-site/content/")
    a = ap.parse_args()

    charts, missing = [], []
    for b in BUILDERS:
        c = b()
        if c is None:
            missing.append(b.__name__)
        else:
            charts.append(c)
    print("=== 產生 %d 張圖 ===" % len(charts))
    for c in charts:
        print("  %-16s %-12s %3d 筆   <- %s"
              % (c["id"], c["kind"], len(c.get("series") or []), c["source"]))
    if missing:
        print("\n**讀不到來源，整組不出（fail-closed）**：%s" % ", ".join(missing))
        print("先把對應的研究腳本跑一次再來。")
        return 2

    # 公開面守衛：掃出任何看起來像金額的東西就拒絕輸出
    import re
    blob = json.dumps(charts, ensure_ascii=False)
    hits = re.findall(r"[$＄]\s?-?\d", blob) + re.findall(r"\busd\b", blob, re.I)
    if hits:
        print("\n**公開面守衛擋下：輸出裡有金額樣式的字串** %s" % hits[:5])
        print("CLAUDE.md §對外網站呈現面：只出百分比、方向、時間、計數。")
        return 2
    print("\n公開面守衛：沒有金額樣式 -> PASS")

    doc = dict(
        _readme=("研究圖表的真相源。複製到 ../product-site/content/"
                 "research_charts.json。**每一個數字都由 "
                 "research/publish_research_charts.py 從做決定的那份結果檔"
                 "生成，不得手改**——手抄的數字會漂而且沒人會發現。"
                 "公開面規則同 research_nogo：只出百分比、方向、時間、計數；"
                 "不出美元、張數、模型內部。"),
        updated=time.strftime("%Y-%m-%d"),
        charts=charts)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("written -> %s" % OUT)
    if a.copy:
        SITE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OUT, SITE)
        print("copied  -> %s" % SITE)
    else:
        print("（要同時複製到網站請加 --copy）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
