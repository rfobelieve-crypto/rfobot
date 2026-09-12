# -*- coding: utf-8 -*-
"""鏈上資料的單位與完整性驗證（2026-09-11）

**為什麼這支存在**：使用者在開始累積歷史之前問「每筆的名目有沒有一樣」。
那正是 mistake.md 2026-09-03 的坑 —— Bitget 的 `sizeMultiplier` 被當成
合約面值，一顆 BTC 的頂檔掛單記成 **$14.57**（真值 $145,718，差一萬倍），
而那個 bug **精確地殺掉了加那個場館的唯一理由**（深度）。

規矩照那條教訓寫死：
  · 任何「數量 x 價格」的名目，先拿一筆真實資料回推量級
  · 跨場館平移程式碼時，**單位是最先要問的事**（張數 vs 幣、面值 vs 步長）
  · 一個新加的東西「表現不如預期」時，第一個假設是量它的儀器壞了

每一關都對上一個**獨立發布的數字**，不是自我一致性。自我一致只能證明
沒算錯，不能證明單位對。

跑法：python research/hl/hl_verify.py
輸出 results/hl_verify_last.json 給 freshness 的 json_flag 讀。
"""
from __future__ import annotations

import glob
import json
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
FLAG = ROOT / "research" / "results" / "hl_verify_last.json"
INFO = "https://api.hyperliquid.xyz/info"
import os
TAPE_DIR = Path(os.environ.get("HL_TAPE_DIR", r"D:\flowbot_data\hl\trades"))


def info(body, t=40):
    req = urllib.request.Request(INFO, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=t) as r:
        return json.loads(r.read().decode())


def main():
    import pandas as pd

    res, fails = {}, []

    def check(name, ok, detail):
        res[name] = dict(ok=bool(ok), detail=detail)
        print("%-6s %-52s %s" % (name, detail, "PASS" if ok else "**FAIL**"))
        if not ok:
            fails.append("%s: %s" % (name, detail))

    mc = info({"type": "metaAndAssetCtxs"})
    uni, ctxs = mc[0]["universe"], mc[1]
    meta = {m["name"]: dict(szd=m.get("szDecimals"), mark=float(c["markPx"]),
                            oi=float(c["openInterest"]),
                            vlm=float(c.get("dayNtlVlm") or 0))
            for m, c in zip(uni, ctxs)}

    # ── V1 逐部位：positionValue 是不是就等於 |szi| x markPx ──────────
    # 這一關直接回答「每筆的名目有沒有一樣」：如果 szi 是張數而不是幣，
    # 這裡會差一個合約面值；如果 positionValue 不是美元，量級會整個不對。
    pf = sorted(glob.glob(str(DATA / "positions" / "*.parquet")))
    if not pf:
        check("V1", False, "沒有 positions 檔可驗")
    else:
        d = pd.read_parquet(pf[-1])
        d = d[d.coin.isin(meta)].copy()
        # 有 mark_snap 的新檔用它當對照（`mark` 現在是同一瞬間反推的，
        # 拿它比會恆等於零，那就失去測量能力了 —— §1.05 的同族）。
        ref = "mark_snap" if "mark_snap" in d.columns else "mark"
        d["recomp"] = d.szi.abs() * d[ref]
        d["relerr"] = (d.value_usd - d.recomp).abs() / d.recomp.clip(lower=1e-9)
        med = float(d.relerr.median())
        p99 = float(d.relerr.quantile(0.99))
        bad = int((d.relerr > 0.02).sum())
        # **門檻測的是單位，不是時鐘**（2026-09-11 修）：`positionValue` 用
        # HL 自己那一刻的 mark，我存的 mark 來自幾秒前的另一次呼叫，所以
        # 必然有幾 bps 的漂移。原本寫 `med < 1e-6` 等於在測兩次 API 呼叫
        # 之間價格沒動 —— 那永遠不可能過（§1.05 那種「沒有測量能力的守衛」
        # 的反面：一個永遠不會過的門檻）。
        # 單位錯的量級是 10/100/10000 倍，所以判準是 1%，並加一關逐幣：
        # 某個幣系統性地偏掉才是單位問題，全體同幅小偏是時鐘。
        percoin = d.groupby("coin").relerr.median()
        worst_coin = float(percoin.max())
        check("V1", med < 0.01 and worst_coin < 0.05,
              "positionValue vs |szi|x" + ref + "：中位 %.1f bps、p99 %.1f bps、"
              "逐幣最差中位 %.1f bps、>2%% 的 %d/%d"
              % (med * 1e4, p99 * 1e4, worst_coin * 1e4, bad, len(d)))

        # ── V2 量級回推：一顆 BTC 的部位名目必須是五位數美元 ──────────
        b = d[d.coin == "BTC"]
        if len(b):
            per1 = float((b.value_usd / b.szi.abs()).median())
            check("V2", 1e4 < per1 < 1e6,
                  "每 1 BTC 的名目 = $%.0f（應等於現價量級 $%.0f）"
                  % (per1, meta["BTC"]["mark"]))
        else:
            check("V2", False, "快照裡沒有 BTC 部位")

        # ── V3 szDecimals：數量的小數位不得超過該資產允許的位數 ────────
        viol = 0
        for coin, g in d.groupby("coin"):
            szd = meta[coin]["szd"]
            if szd is None:
                continue
            q = 10 ** (-szd)
            r = (g.szi.abs() / q)
            viol += int(((r - r.round()).abs() > 1e-6).sum())
        check("V3", viol == 0, "szi 違反 szDecimals 的筆數 %d" % viol)

    # ── V4 成交帶的名目對上交易所公布的日成交額（獨立數字）─────────────
    tf = sorted(glob.glob(str(TAPE_DIR / "*" / "*.parquet")))
    if not tf:
        check("V4", False, "沒有 tape 檔可驗")
    else:
        # **用檔案裡最密集的連續 60 秒窗**（2026-09-11 第二次修）。
        # 第一版用檔案跨度正規化，但跨度 != 連續錄製時長：WS 訂閱時會先送
        # 每個幣的近期成交快照，所以一個 150 秒的錄製會產生跨度 2,182 秒的
        # 檔案 -> 預期量被高估 14 倍 -> 比值 0.07x 被誤判成失敗。
        # 60 秒窗自己就知道取樣長度，而且對單位錯仍然敏感（差 10 倍就差 10 倍）。
        t = pd.read_parquet(tf[-1])
        t = t[t.coin.isin(meta)].copy()
        t["ntl"] = t.px * t.sz
        t = t.sort_values("ts")
        ts = t.ts.to_numpy()
        W = 60_000
        best_lo, best_n = None, -1
        import numpy as np
        for k in range(0, len(ts), max(1, len(ts) // 400)):
            lo = ts[k]
            n = int(((ts >= lo) & (ts < lo + W)).sum())
            if n > best_n:
                best_n, best_lo = n, lo
        w = t[(t.ts >= best_lo) & (t.ts < best_lo + W)]
        expect = W / 1000.0 / 86400.0
        agg = w.groupby("coin").ntl.sum()
        rows = []
        for coin, v in agg.sort_values(ascending=False).head(6).items():
            day = meta[coin]["vlm"]
            frac = v / day if day else float("nan")
            rows.append((coin, v, day, frac,
                         frac / expect if expect else float("nan")))
        # ── 2026-09-12 第四次修：判準改成「跨幣總額、跨多個窗取中位」 ──
        #
        # 前三版都在**逐幣**的數字上設判準，而那個數字量不到它要量的東西：
        #
        #   · 第三版的「中位」寫成 `r5[len(r5)//2]`，偶數長度取的是**上半的
        #     中間值**不是中位數。6 個幣 3 高 3 低時它必然挑到高的那個 ——
        #     2026-09-12 實測 r5 = [0.019, 0.101, 0.136, 7.01, 7.25, 9.41]，
        #     它回 7.01（紅），真中位數是 3.57（綠）。一個幣的差別就跳 50 倍。
        #   · 更根本的：分子取**最密集**的窗、分母用**日平均**速率 ——
        #     依建構是「極大值 ÷ 平均值」，所以偏高是必然的，而門檻只給到 5。
        #     那天 ZEC 在噴（9.41x）而 ETH 安靜（0.019x），**跨幣差 500 倍**。
        #   · 而「跨幣一致性」也不能當判準：單位錯**不改變幣與幣的比值**，
        #     所以 100 倍的錯照樣是 500 倍的分散度 —— 這條修法過不了反向證明，
        #     所以沒有採用（先量再改，避免第二次修到錯的東西上）。
        #
        # 有分辨力的統計量是**跨幣總額**（籃子平均掉個別幣的爆發）配上
        # **多個滑動窗的中位**（不挑極大值）。2026-09-12 在兩個不同的 tape 檔
        # 上實測：**0.402x**（76 個窗）與 **0.293x**（136 個窗），
        # 窗內 p5~p95 只有 0.131~1.035 —— 8 倍區間，對比逐幣那個 500 倍。
        # 小於 1 是預期的：`dayNtlVlm` 與我們錄到的宇宙不完全相同。
        #
        # **帶 [0.05, 5]，以 0.29~0.40 的基線算，解析度是上方約 13~17 倍、
        # 下方約 6~8 倍。** 反向證明過（scratchpad/v4_reverse.py）：
        # 注入 `px x100` -> 中位 **29.269x** -> FAIL；還原 -> 0.293x -> PASS。
        # 注意注入後逐幣仍是 118x~2145x 的 18 倍分散度 ——
        # **分散度在單位錯之下照樣存在**，所以它不能當單位錯的指紋。
        #
        # 要更緊的單位檢查看 **V2**（每 1 BTC 的名目 vs 現價，實測差 0.06%）；
        # V4 的價值在於它用的是**交易所公布的成交額**這個獨立數字。
        import statistics as _st
        W_STEP = 10_000
        lo0, hi0 = int(ts[0]), int(ts[-1]) - W
        coins_seen = sorted(t.coin.unique())
        exp_tot = (sum(meta[c]["vlm"] for c in coins_seen)
                   * (W / 1000.0) / 86400.0)
        ratios = []
        if hi0 > lo0 and exp_tot > 0:
            ntl_all = t.ntl.to_numpy()
            for lo in range(lo0, hi0, W_STEP):
                m = (ts >= lo) & (ts < lo + W)
                if m.any():
                    ratios.append(float(ntl_all[m].sum()) / exp_tot)
        r5 = sorted(r[4] for r in rows if r[4] == r[4])
        if ratios:
            med_tot = _st.median(ratios)
            v4_ok = 0.05 <= med_tot <= 5.0
            det = ("跨幣總額/預期，%d 個滑動 60s 窗中位 %.3fx（帶 0.05~5）；"
                   "逐幣僅供參考（爆發性，不設判準）：" % (len(ratios), med_tot)
                   + "、".join("%s %.2fx" % (r[0], r[4]) for r in rows))
            res["V4_total_median"] = med_tot
            res["V4_windows"] = len(ratios)
        else:
            # 檔案不足一個窗：退回單一最密窗，只擋**離譜**的量級
            med_tot = _st.median(r5) if r5 else float("nan")
            v4_ok = bool(r5) and 0.02 <= med_tot <= 50.0
            det = ("檔案不足一個滑動窗，退回最密 60s 窗逐幣中位 %.2fx"
                   "（寬帶 0.02~50）：" % med_tot
                   + "、".join("%s %.2fx" % (r[0], r[4]) for r in rows))
        check("V4", v4_ok, det)
        res["V4_rows"] = [dict(coin=r[0], ntl=r[1], day_ntl=r[2],
                               frac=r[3], frac_over_expected=r[4])
                          for r in rows]
        res["V4_window_trades"] = int(best_n)

        # ── V5 時間戳單位：全部毫秒，且落在合理範圍 ───────────────────
        mn, mx = int(t.ts.min()), int(t.ts.max())
        check("V5", 1.5e12 < mn and mx < 2.2e12,
              "tape ts 範圍 %d ~ %d（毫秒應在 1.5e12~2.2e12）" % (mn, mx))

        # ── V6 tid 唯一（去重有效）────────────────────────────────────
        dup = int(len(t) - t.tid.nunique())
        check("V6", dup == 0, "同一檔內重複 tid %d 筆" % dup)

        # ── V7 永續與現貨不混：coin 不得出現 @index 形式 ───────────────
        atn = int(t.coin.astype(str).str.startswith("@").sum())
        check("V7", atn == 0, "tape 含現貨代號（@index）%d 筆" % atn)

        # ── V8 成交價與當時 mark 的偏離：>20% 代表抓錯市場或單位 ────────
        t["dev"] = (t.px / t.coin.map(lambda c: meta[c]["mark"]) - 1).abs()
        far = int((t.dev > 0.20).sum())
        check("V8", far / max(len(t), 1) < 0.05,
              "成交價偏離現價 >20%% 的 %d/%d（舊快照會有，比例須小）"
              % (far, len(t)))

    # ── V9 OI 名目的量級：用獨立的 24h 成交額當對照 ────────────────────
    oi_usd = {c: v["oi"] * v["mark"] for c, v in meta.items()}
    tot_oi = sum(oi_usd.values())
    tot_vlm = sum(v["vlm"] for v in meta.values())
    ratio = tot_oi / tot_vlm if tot_vlm else float("nan")
    check("V9", 0.2 < ratio < 20,
          "總 OI $%.2fB / 總 24h 量 $%.2fB = %.2f（周轉合理區間 0.2~20）"
          % (tot_oi / 1e9, tot_vlm / 1e9, ratio))

    # ── V10 觸發單：止損的 triggerPx 與方向必須一致 ─────────────────────
    # reduce-only 的止損：平多（side=A，賣出）的觸發價在現價**之下**，
    # 平空（side=B，買入）的在**之上**。反了就是我把 side 讀錯。
    of = sorted(glob.glob(str(DATA / "orders" / "*.json")))
    if not of:
        check("V10", False, "沒有 orders 檔可驗")
    else:
        trg = json.loads(Path(of[-1]).read_text(encoding="utf-8")).get("triggers", [])
        stops = [x for x in trg if x.get("reduce_only")
                 and "Stop" in str(x.get("order_type") or "")]
        consistent = sum(1 for x in stops
                         if (x["side"] == "A" and x["dist_pct"] < 0)
                         or (x["side"] == "B" and x["dist_pct"] > 0))
        check("V10", (not stops) or consistent / len(stops) > 0.80,
              "reduce-only 止損方向一致 %d/%d（平多在下、平空在上）"
              % (consistent, len(stops)))
        res["V10_n_triggers"] = len(trg)

    ok = not fails
    FLAG.parent.mkdir(parents=True, exist_ok=True)
    FLAG.write_text(json.dumps(dict(
        ok=ok, reason=("; ".join(fails)[:400] if fails
                       else "%d 關全過" % len(res)),
        checks=res, asof=time.strftime("%Y-%m-%d %H:%M:%S")),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("hl verify: %s  %s" % ("OK" if ok else "RED",
                                 "; ".join(fails) if fails else "全過"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
