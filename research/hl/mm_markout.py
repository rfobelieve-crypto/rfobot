# -*- coding: utf-8 -*-
"""§1.29 倉位型做市的 Gate 0：HL 上「當掛單方」一筆值多少（2026-09-12）

===========================================================================
為什麼做這一支
===========================================================================
§1.27 / §1.28 之後，MFT 只剩一條路，而那條路的門檻用的是**別人的數字**：

    吃單        3.00 bps/單位成交量   （Bitget 返佣後，實查）
    掛單        8.27                  （我們實測：被動代價 7.27 + 掛單費 1.00）
    倉位型做市  **≈ 0.5**             ← Advanced MM 的說法，**我們沒驗過**

核心原則 11 的 Gate 0 (b)：「那個價格在目標場館拿得到嗎 —— **要量測，
不是論證**」。這支就是把那個 0.5 換成我們自己量的。

===========================================================================
量什麼
===========================================================================
對每一筆成交（成交價 P、吃單方向 s），站在**掛單方**（被吃的那一方）：

    掛單方單筆損益 = s × (P − mid(t+h)) / P × 1e4   s=+1 吃單方是買、−1 是賣

**這一個式子就是完整的答案，不要再加半價差**（第一版加了，等於算兩遍；
見 `maker_edge` 的註解）。它可以拆成兩項，而拆解只是為了看得懂：

    掛單方損益 = (P − mid(t))        成交當下捕獲的半價差
               + (mid(t) − mid(t+h)) 之後行情往哪走（教科書講的 markout）

**若這個數是負的，這個場館的被動做市本身就是賠錢的** ——
那時候「用訊號偏移報價」連地基都沒有。

符號怎麼記：吃單方是**買**（s=+1）代表掛單方**賣**在 P。之後 mid 漲上去
（mid(t+h) > P）對賣方不利 -> `P − mid` 為負 -> markout 為負。對。

===========================================================================
判準（TODO §1.29 凍結，跑之前寫的）
===========================================================================
D1  **`side` 的語意用資料證明，不用猜**：吃單方是買的成交，成交價必須平均
    **高於**當下 mid；是賣的必須低於。`A`/`B` 猜反會讓 markout 整個變號，
    而輸出**看起來完全正常**。
D2  實測半價差要對得上 `hl_mid` 自己記的 `spread_bps`，而且前 30 名要落在
    CLAUDE.md 已量的頂檔 ~2.9 bps（半價差 ~1.45）。
D3  **安慰劑**：把 `side` 隨機打亂，損益必須塌回 ~0。
D4  **分解必須閉合**（跑完才補的一道，因為第一版就是死在這裡）：
    捕獲 + 漂移 必須精確等於單筆損益；而且在「mid 夠新鮮」的子樣本上，
    **捕獲要 ≈ 半價差** —— 對不上就代表 60 秒取樣太粗，這台儀器量不動做市。
MDE 先印，而且算在**被決定的那個量**（單筆毛利）上。

===========================================================================
誠實的限制（先寫，不是事後補）
===========================================================================
1. **這是上界。** 量的是**所有成交的平均** markout ＝ 假設我們拿到隨機的
   一片流量。**沒有排隊優先權的人拿不到隨機的一片** —— 逆選擇的定義就是
   壞的那些會優先成交到你。真實會比這裡差。
2. **秒級看不到**（mid 60 秒取樣）。1 分鐘 markout 若已經很負，結論成立；
   **若為正，需要更細的資料才能確認**。
3. **只有一天行情。** 筆數不是問題，regime 覆蓋才是。所以本支**不下判決**，
   判決與 §1.23b 的 HL 扳機同一天（09-18）換窗重跑，判準不改。

    python research/hl/mm_markout.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TAPE = Path("D:/flowbot_data/hl/trades")
MID = Path("D:/flowbot_data/hl/mid")
OUT = ROOT / "research" / "results" / "hl_mm_markout.json"
HORIZONS = (1, 5, 30)                 # 分鐘
RNG = np.random.default_rng(20260912)


def load_mid() -> pd.DataFrame:
    fs = sorted(MID.glob("*/*.parquet"))
    if not fs:
        raise SystemExit("沒有 hl_mid 資料")
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d = d[(d.bid > 0) & (d.ask > 0) & (d.ask >= d.bid) & (d.mid > 0)].copy()
    d["minute"] = d.ts // 60000
    # 同一分鐘可能有多筆（重啟時），留最後一筆
    d = d.sort_values("ts").drop_duplicates(["coin", "minute"], keep="last")
    return d


def load_tape(lo_ms: int, hi_ms: int) -> pd.DataFrame:
    """只讀跟 mid 重疊的那段。

    **注意不要按資料夾名篩**：`hl/trades` 有 39 個日期資料夾，其中 37 個是
    死掉的幣的最後幾筆成交（WS 訂閱時送的「最近成交」，殭屍市場的「最近」
    就是兩年前）。按 `ts` 篩才是對的，而且那些殭屍本來就會被 mid 篩掉。
    """
    fs = sorted(TAPE.glob("*/*.parquet"))
    out = []
    for f in fs:
        d = pd.read_parquet(f)
        d = d[(d.ts >= lo_ms) & (d.ts <= hi_ms)]
        if len(d):
            out.append(d)
    if not out:
        raise SystemExit("重疊窗內沒有成交")
    d = pd.concat(out, ignore_index=True)
    d["minute"] = d.ts // 60000
    return d


def attach(tr: pd.DataFrame, md: pd.DataFrame, h: int):
    """把成交接上「當下的 mid」與「h 分鐘後的 mid」。

    用 merge_asof（backward）取當下那一筆，**不是取同分鐘** —— 同分鐘那筆
    可能是成交之後才取樣的，那會把未來的價格塞進「當下」。
    """
    tr = tr.sort_values("ts")
    m0 = md[["coin", "ts", "mid", "bid", "ask", "spread_bps"]].sort_values("ts")
    a = pd.merge_asof(tr, m0, on="ts", by="coin", direction="backward",
                      tolerance=120_000, suffixes=("", "_m"))
    a = a.dropna(subset=["mid"])
    # h 分鐘後：forward 取第一筆 >= ts + h*60s
    fut = m0.rename(columns={"mid": "mid_f", "ts": "ts_f"})[["coin", "ts_f", "mid_f"]]
    a["ts_t"] = a.ts + h * 60_000
    a = a.sort_values("ts_t")
    a = pd.merge_asof(a, fut, left_on="ts_t", right_on="ts_f", by="coin",
                      direction="forward", tolerance=120_000)
    return a.dropna(subset=["mid_f"])


def maker_edge(a: pd.DataFrame, side_col: str = "side") -> np.ndarray:
    """掛單方的**完整**單筆損益（bps）。s=+1 吃單方是買（=掛單方賣）。

    ⚠ **這個式子已經含了捕獲的半價差，不可以再加一次。**（2026-09-12 修）
    第一版把它叫 `markout` 然後又加了 `half`，半價差被算了兩遍，
    前 30 名 h=1 從真正的 −0.66 變成 +0.32（**連正負號都反了**）。

    為什麼它本身就含：錨是**成交價 P**，不是成交當下的 mid。掛單方賣在 P，
    而 P 本來就高出當時的 mid 半個價差 —— 那半個價差已經在 (P − mid_f) 裡。

        掛單方損益 = P − mid(t+h)
                   = (P − mid(t))      捕獲的半價差
                   + (mid(t) − mid(t+h))  之後行情往哪走（教科書的 markout）

    下面 `decompose()` 把這兩項拆出來，而 D4 驗它們加起來等於本式。
    """
    s = np.where(a[side_col].values == "B", 1.0, -1.0)
    return s * (a.px.values - a.mid_f.values) / a.px.values * 1e4


def decompose(a: pd.DataFrame, side_col: str = "side"):
    """回傳 (捕獲, 漂移)：兩項相加必須等於 `maker_edge`。

    `捕獲` 用的是**成交之前最近一筆** mid，而那筆最多可能舊 60 秒
    （錄製器是牆鐘 60 秒取樣）。所以捕獲項帶著陳舊誤差 —— D4 就是在量
    那個誤差有多大，因為它決定這台儀器測不測得動做市的經濟性。
    """
    s = np.where(a[side_col].values == "B", 1.0, -1.0)
    cap = s * (a.px.values - a.mid.values) / a.px.values * 1e4
    dft = s * (a.mid.values - a.mid_f.values) / a.px.values * 1e4
    return cap, dft


def boot(x, keys, b=2000):
    """按幣聚類的 bootstrap —— 同一個幣的成交不是獨立的。"""
    x = np.asarray(x, float)
    keys = np.asarray(keys)
    ok = np.isfinite(x)
    x, keys = x[ok], keys[ok]
    if len(x) < 50:
        return (np.nan,) * 4
    uq, inv = np.unique(keys, return_inverse=True)
    ix = [np.where(inv == k)[0] for k in range(len(uq))]
    r = np.empty(b)
    for i in range(b):
        p = RNG.integers(0, len(uq), len(uq))
        r[i] = x[np.concatenate([ix[k] for k in p])].mean()
    return (float(x.mean()), float(r.std(ddof=1)),
            float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30, help="分層的切點（前 N 名）")
    a = ap.parse_args()

    print("=== §1.29 倉位型做市的 Gate 0：當掛單方一筆值多少 ===")
    print("把 Advanced MM 的『≈0.5 bps』換成我們自己量的數。\n")

    md = load_mid()
    lo, hi = int(md.ts.min()), int(md.ts.max())
    print("mid：%d 列、%d 幣、%s ~ %s"
          % (len(md), md.coin.nunique(),
             pd.to_datetime(lo, unit="ms", utc=True).strftime("%m-%d %H:%M"),
             pd.to_datetime(hi, unit="ms", utc=True).strftime("%m-%d %H:%M")))
    tr = load_tape(lo, hi)
    tr = tr[tr.coin.isin(set(md.coin.unique()))]
    print("tape：%d 筆（已篩到重疊窗與 mid 有的幣）、%d 幣\n"
          % (len(tr), tr.coin.nunique()))

    # 流動性分層：用重疊窗內的成交名目排名
    tr["notional"] = tr.px * tr.sz
    rank = tr.groupby("coin").notional.sum().sort_values(ascending=False)
    top = set(rank.index[:a.top])
    tr["tier"] = np.where(tr.coin.isin(top), "前%d" % a.top, "其餘")

    res = {"asof": time.strftime("%Y-%m-%d %H:%M:%S"),
           "window_utc": [str(pd.to_datetime(lo, unit="ms", utc=True)),
                          str(pd.to_datetime(hi, unit="ms", utc=True))],
           "n_trades": int(len(tr)), "n_coins": int(tr.coin.nunique()),
           "top_n": a.top, "horizons": list(HORIZONS)}

    # ---------- D1：side 的語意用資料證明 ----------
    print("=== D1 自曝：`side` 的語意用資料證明，不用猜 ===")
    base = attach(tr, md, HORIZONS[0])
    dev = (base.px.values - base.mid.values) / base.px.values * 1e4
    isb = base.side.values == "B"
    mb, ms = float(np.nanmean(dev[isb])), float(np.nanmean(dev[~isb]))
    ok1 = mb > 0 > ms
    print("  side='B' 的成交價相對當下 mid：%+.3f bps（n=%d）" % (mb, isb.sum()))
    print("  side='A' 的成交價相對當下 mid：%+.3f bps（n=%d）" % (ms, (~isb).sum()))
    print("  -> %s"
          % ("PASS：B = 吃單方買（掛單方賣）" if ok1 else
             "**FAIL：語意跟假設相反，markout 會整個變號，以下不解讀**"))
    res["D1"] = bool(ok1)
    res["dev_B"], res["dev_A"] = mb, ms
    if not ok1:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        return 2

    # ---------- D2：半價差對得上嗎 ----------
    print("\n=== D2 自曝：實測半價差 vs 錄製器自己記的 spread_bps ===")
    base["half_calc"] = (base.ask - base.bid) / base.mid * 1e4 / 2.0
    base["half_rec"] = base.spread_bps / 2.0
    d2 = float((base.half_calc - base.half_rec).abs().max())
    ok2 = d2 < 0.05
    print("  逐筆最大差 %.4f bps -> %s" % (d2, "PASS" if ok2 else "**FAIL**"))
    for t, g in base.groupby("tier"):
        print("  %-6s 半價差 中位 %.3f bps（CLAUDE.md 前 30 名頂檔 ~2.9 -> 半 ~1.45）"
              % (t, float(g.half_calc.median())))
    res["D2"] = bool(ok2)
    if not ok2:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        return 2

    # ---------- MDE 先印 ----------
    print("\n=== MDE 先講（算在被決定的那個量：單筆毛利）===")
    net0 = maker_edge(base)
    mu, se, lo_, hi_ = boot(net0, base.coin.values)
    print("  h=1min 單筆毛利 %+.4f bps（SE %.4f、CI [%+.4f, %+.4f]）"
          % (mu, se, lo_, hi_))
    print("  分辨得出「> 0」嗎：%s"
          % ("是（CI 不含零）" if lo_ > 0 or hi_ < 0 else "**否 —— CI 含零**"))
    res["mde"] = dict(mean=mu, se=se, lo=lo_, hi=hi_)

    # ---------- D3：安慰劑 ----------
    print("\n=== D3 自曝：打亂 `side` 之後 markout 必須塌回 0 ===")
    sh = base.copy()
    sh["side_shuf"] = RNG.permutation(sh.side.values)
    mks = maker_edge(sh, "side_shuf")
    mu_s, se_s, _, _ = boot(mks, sh.coin.values)
    mu_r, _, _, _ = boot(net0, base.coin.values)
    ok3 = abs(mu_s) < max(0.1 * abs(mu_r), 3 * se_s) if np.isfinite(mu_r) else False
    print("  真實損益 %+.4f  打亂後 %+.4f（SE %.4f）-> %s"
          % (mu_r, mu_s, se_s, "PASS" if ok3 else "**FAIL —— 量到的是漂移不是方向**"))
    res["D3"] = bool(ok3)
    res["markout_shuffled"] = mu_s

    # ---------- D4：分解必須閉合，而且捕獲要對得上半價差 ----------
    # **這一道是第一版死掉的地方才補的**，所以它有牙齒不是裝飾：
    # 第一版把「已經含半價差的損益」又加了一次半價差，前 30 名 h=1 從
    # 真正的 −0.66 變成 +0.32 —— 連正負號都反了，而三道守衛全都放行。
    print()
    print("=== D4 自曝：分解閉合 ＋ 捕獲 vs 半價差 ===")
    cap0, dft0 = decompose(base)
    resid = float(np.nanmax(np.abs(cap0 + dft0 - net0)))
    ok4a = resid < 1e-9
    print("  捕獲 + 漂移 − 單筆損益：最大殘差 %.3e -> %s"
          % (resid, "PASS" if ok4a else "**FAIL（記帳漏了一塊）**"))
    # mid 夠新鮮的子樣本：成交發生在 mid 取樣後 5 秒內
    fresh = (base.ts.values - base.ts_m.values) <= 5_000         if "ts_m" in base.columns else None
    if fresh is None or fresh.sum() < 500:
        # merge_asof 沒留下 mid 的時戳就退回用分鐘內位置近似
        fresh = (base.ts.values % 60_000) <= 5_000
    cf, hf = cap0[fresh], base.half_calc.values[fresh]
    print("  新鮮子樣本（成交落在取樣後 5 秒內）n=%s" % format(int(fresh.sum()), ","))
    print("    捕獲 %+.3f bps   半價差 %.3f bps   比值 %.2f"
          % (np.nanmean(cf), np.nanmean(hf), np.nanmean(cf) / np.nanmean(hf)))
    print("    全樣本捕獲 %+.3f bps（差距 = 60 秒取樣的陳舊誤差）"
          % float(np.nanmean(cap0)))
    ratio = float(np.nanmean(cf) / np.nanmean(hf))
    ok4b = 0.5 < ratio < 2.0
    print("  -> %s" % ("PASS：捕獲與半價差同量級，儀器量得動"
                       if ok4b else
                       "**可疑：捕獲跟半價差差太多，60 秒取樣可能量不動做市**"))
    res["D4_closed"], res["D4_capture_ratio"] = bool(ok4a), ratio
    if not ok4a:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        return 2

    # ---------- 主表 ----------
    print("\n=== 逐 horizon × 分層（全格報告，不挑）===")
    print("  單筆損益 = 掛單方 s×(成交價 − mid(t+h))，**已含捕獲的半價差**。")
    print("  『捕獲』那一欄只放著參考 —— D4 已經證明 60 秒取樣分不開它與漂移。")
    print("%6s %-8s %9s %10s %10s %10s %9s %10s %9s"
          % ("h(分)", "分層", "筆數", "半價差", "單筆損益", "CI下",
             "中位", "截尾均值", "捕獲*"))
    res["table"] = {}
    for h in HORIZONS:
        aa = attach(tr, md, h)
        aa["half"] = (aa.ask - aa.bid) / aa.mid * 1e4 / 2.0
        cap, dft = decompose(aa)
        aa["cap"], aa["dft"] = cap, dft
        aa["net"] = maker_edge(aa)
        for t, g in aa.groupby("tier"):
            m, s_, l_, h_ = boot(g.net.values, g.coin.values)
            # **截尾均值要跟均值並列。** dev 的分佈極寬（p5 −37、p95 +65 bps，
            # 而半價差中位只有 0.42），所以「這個負號是不是被尾巴撐出來的」
            # 必須當場回答，不能留給讀的人猜。實測截尾幾乎不動它。
            q1, q2 = g.net.quantile([0.005, 0.995])
            trim = float(g.net[(g.net >= q1) & (g.net <= q2)].mean())
            med = float(g.net.median())
            print("%6d %-8s %9s %10.3f %+10.3f %+10.3f %+9.3f %+10.3f %+9.3f"
                  % (h, t, format(len(g), ","), float(g.half.mean()),
                     m, l_, med, trim, float(g.cap.mean())))
            res["table"]["%d|%s" % (h, t)] = dict(
                n=int(len(g)), half=float(g.half.mean()),
                capture=float(g.cap.mean()), drift=float(g.dft.mean()),
                net=m, se=s_, lo=l_, hi=h_, median=med, trimmed=trim)

    print("\n=== 讀法 ===")
    print("  **單筆損益 = 捕獲 + 漂移**，而捕獲本身就是半價差那一項 ——")
    print("  所以不要再把半價差加上去（第一版加了，符號都反掉）。")
    print("  它要對的門檻是 0（不是 0.5）——")
    print("  0.5 是『用訊號偏移報價』要額外付的，而那個偏移只有在")
    print("  **被動做市本身先為正**的時候才有意義。")
    print("  這個數是**上界**：量的是所有成交的平均，等於假設我們拿到隨機的")
    print("  一片流量，而沒有排隊優先權的人拿不到隨機的一片。")
    print("  **本支不下判決** —— 只有一天行情，判決在 09-18 換窗重跑。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print("\nwritten -> %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
