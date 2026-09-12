# -*- coding: utf-8 -*-
"""TODO §1.30 C2：升格指標換成**比值**重排 3,124 個掃描配對（2026-09-12）

===========================================================================
為什麼這一題今天才有意義
===========================================================================
C2 原本登記的是「9 個錄製配對改用比值重排序」。但**同日的 §1.32 已經把
那 9 個逐個算過淨值了** —— 除了 ANTH 之外，八個配對**沒有任何下單金額
讓淨值為正**。重排序救不了一個已經窮舉算過的家族。

**所以 C2 真正的問題在上游一層**：掃描器看了 **3,124 個配對**，
凍結指標（2026-08-30）只升格其中幾個，而我們錄的就是那幾個。
**如果那個指標量錯了東西，我們錄錯了配對** —— 而那會解釋為什麼家族是 $0。

===========================================================================
凍結指標長什麼樣，以及它為什麼可能獎勵錯的東西
===========================================================================
    capturable_usd_per_day = fires_per_day x band_bps/1e4 x depth_usd
      band  = 正向可執行邊際的 **p90**
      fires = edge >= band 的樣本數 / 天

**`band` 是 p90，所以 `fires` 依定義就是「正邊際樣本數的 10%」。**
於是分數 ∝ **（有正邊際的樣本數）x 帶寬 x 深度**。

「有正邊際的樣本數」量的是**時間**，不是**機會**。一個持續 90 分鐘的偏移
會貢獻 90 個樣本，但它是**一筆**交易（而且可能一筆都不是）。
mistake.md 2026-09-03 已經抓過這個病並修了**下游的報表**
（`capturable_usd_per_day_tradeable`：事件數 x 半個帶 x 深度），
**但升格指標凍結在 09-03 之前，從來沒有被修**。

**假說（要量不要論證）**：凍結指標系統性偏好**持續偏移**，
而持續偏移正好是唯一不能套利的東西 —— §1.32 的 ANTH 就是那個形狀
（14.2% 的分鐘楔子開著、收斂事件平均每次開 93 分鐘、兩腿根本不是同一個合約）。

===========================================================================
三個排序（全格報告，不挑）
===========================================================================
    R0  凍結        fires x band/1e4 x depth         <- 現行，必須先重現
    R1  事件型      events/day x (band/2)/1e4 x depth <- 把 09-03 的修法搬到上游
    R2  比值型      sigma(edge) / 來回成本            <- SNVR 那一族的精神：
                    「價差波動率相對成本」，而不是水準

**R2 不叫 SNVR。** 原文的 SNVR 是「名目 ÷ 價差暴險、波動率調整」，
而且它是**最佳化的目標函數**不是排序分數；兩腿的 NVR ≈ 2 已經是下限
（§1.30）。這裡只借它的方向：**分子要的是波動率，不是水準**。
自己發明的東西就叫自己的名字（[[feedback_plain_language]]）。

===========================================================================
自曝關
===========================================================================
D1 **已知答案**：R0 的前 30 名必須重現 `results/arb_scan_rank.json` 的 `top`。
   對不上 -> 我載錯資料或算錯，**以下一律不解讀**（mistake.md 2026-07-29）。
D2 **取樣間隔要量不要假設**：`fires` 的單位取決於掃描間隔，
   而那個數字決定「分鐘 vs 事件」差幾倍。
D3 **持續性**：逐配對算「連續在帶上的平均分鐘數」。
   假說成立的判準是 **R0 前段的 run 長度明顯長於 R1 前段** ——
   若兩者相近，我的機制說法不成立，要撤回。
D4 **BTC 對照組**：BTC 兩個深簿、零帶，它在任何排序裡都該在後段。
   它跑到前段 = 那個排序壞了（§0.75 家族一直用它當雜訊底）。

跑法：
    python research/prereg_snvr_rerank.py
    python research/prereg_snvr_rerank.py --top 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "research"))

# **只有一個檔案知道 arb repo 在哪**（CLAUDE.md：`research/arb_home.py`，
# `ARB_HOME` 可覆寫）。不要自己拼相對路徑。
import arb_home                                                # noqa: E402

_miss = arb_home.missing()
if _miss:
    raise RuntimeError("arb repo 不可用：%s" % _miss)
arb_home.add_to_path()
ARB = arb_home.HOME
from arblib import fees as FEES                                 # noqa: E402
from arblib import scan_rank                                   # noqa: E402

OUT = ROOT / "research" / "results" / "prereg_snvr_rerank.json"
MIN_SAMPLES = 500          # 與 scan_rank 同（不是一輪定江山）


def round_trip_cost_bps(leg_a: str, leg_b: str) -> float:
    """來回成本 = 2 x (fee_a_eff + fee_b_eff)，與 arblib/fees 的決策規則同。

    用 taker 兩腿（最保守，也是 §0.75 家族原本的假設）。
    查不到的場館回 nan —— **不要用預設值填**，那會讓一個未知成本看起來像零。
    """
    tot = 0.0
    for leg in (leg_a, leg_b):
        v = FEES.VENUES.get(leg)
        if v is None:
            return float("nan")
        tot += v["taker_bps"] * (1.0 - v.get("rebate", 0.0))
    return 2.0 * tot


def per_side(g: pd.DataFrame, edge_col: str, depth_cols, cost_bps: float):
    """一側的三個分數 ＋ 持續性診斷。回 None = 樣本不足（與 scan_rank 同門檻）。"""
    pos = g[g[edge_col] > 0]
    span_days = max((g.ts.max() - g.ts.min()) / 86400, 1e-9)
    if len(pos) < 10:
        return None
    band = float(pos[edge_col].quantile(0.9))
    fat = g[g[edge_col] >= band]
    if not len(fat):
        return None
    depth = float(fat[list(depth_cols)].min(axis=1).median())
    fires_per_day = len(fat) / span_days

    # ── 事件數：連續在帶上的一段算一個事件（這就是 09-03 的修法）──────
    on = (g[edge_col] >= band).to_numpy()
    starts = int(np.sum(on[1:] & ~on[:-1])) + int(bool(on[0]))
    events_per_day = starts / span_days
    runs = []
    n = 0
    for v in on:
        if v:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    mean_run = float(np.mean(runs)) if runs else 0.0

    sd = float(g[edge_col].std(ddof=1)) if len(g) > 2 else float("nan")
    return {
        "band_bps": round(band, 3),
        "fires_per_day": round(fires_per_day, 1),
        "events_per_day": round(events_per_day, 2),
        "mean_run_samples": round(mean_run, 2),
        "depth_usd": round(depth, 0),
        "sigma_bps": round(sd, 3),
        "cost_bps": round(cost_bps, 3) if cost_bps == cost_bps else None,
        # R0 凍結：fires x band x depth
        "R0_frozen": round(fires_per_day * band / 1e4 * depth, 2),
        # R1 事件型：events x 半個帶 x depth
        "R1_event": round(events_per_day * (band / 2) / 1e4 * depth, 2),
        # R2 比值型：波動率 / 來回成本（無單位）
        "R2_ratio": (round(sd / cost_bps, 4)
                     if (cost_bps == cost_bps and cost_bps > 0
                         and sd == sd) else None),
    }


def main(a) -> int:
    df = scan_rank.load()
    if df.empty:
        raise RuntimeError("掃描資料是空的 —— 不要回空容器讓下游安靜地讀舊檔"
                           "（mistake.md 2026-09-11）")
    span = (df.ts.max() - df.ts.min()) / 86400
    print("=" * 92)
    print("  §1.30 C2：升格指標重排 —— 3 個排序、全格報告｜掃描跨度 %.2f 天｜%s 列"
          % (span, f"{len(df):,}"))
    print("=" * 92)

    # ── D2 取樣間隔（量不假設）────────────────────────────────────────
    d = df.sort_values(["pair", "ts"])
    gaps = d.groupby("pair").ts.diff().dropna()
    gaps = gaps[(gaps > 0) & (gaps < 3600)]
    med_gap = float(gaps.median()) if len(gaps) else float("nan")
    print("\nD2 掃描間隔中位 **%.0f 秒**（%.1f 分）—— `fires` 的單位是這個，不是分鐘"
          % (med_gap, med_gap / 60))

    rows = []
    for pair, g in df.groupby("pair"):
        if len(g) < MIN_SAMPLES:
            continue
        la, lb = g.leg_a.iloc[0], g.leg_b.iloc[0]
        cost = round_trip_cost_bps(la, lb)
        best = None
        for ec, dc in (("sell_edge_bps", ("a_bid_usd", "b_ask_usd")),
                       ("buy_edge_bps", ("a_ask_usd", "b_bid_usd"))):
            m = per_side(g, ec, dc, cost)
            if m is None:
                continue
            m["side"] = "sell" if ec.startswith("sell") else "buy"
            if best is None or m["R0_frozen"] > best["R0_frozen"]:
                best = m
        if best is None:
            continue
        best.update(pair=pair, leg_a=la, leg_b=lb, n=len(g),
                    sym=g.sym_a.iloc[0])
        rows.append(best)

    t = pd.DataFrame(rows)
    print("有 >= %d 筆樣本、算得出分數的配對：**%d**" % (MIN_SAMPLES, len(t)))

    # ── D1 已知答案：R0 前 30 必須重現 arb_scan_rank.json 的 top ──────
    pub = json.loads((ARB / "results" / "arb_scan_rank.json")
                     .read_text(encoding="utf-8"))
    pub_top = [x["pair"] if isinstance(x, dict) else x
               for x in (pub.get("top") or [])][:30]
    mine = t.sort_values("R0_frozen", ascending=False).pair.head(30).tolist()
    inter = len(set(pub_top) & set(mine))
    same_order = sum(1 for i, p in enumerate(mine[:len(pub_top)])
                     if i < len(pub_top) and p == pub_top[i])
    d1 = inter >= 27
    print("\nD1 已知答案對照（R0 前 30 vs 已發布的 top 30）")
    print("   交集 %d/30、同名次 %d -> %s"
          % (inter, same_order, "PASS" if d1 else "**FAIL**"))
    if not d1:
        print("   已發布前 10：%s" % ", ".join(pub_top[:10]))
        print("   本支  前 10：%s" % ", ".join(mine[:10]))
        print("\n**D1 未過 —— 我載錯資料或算錯，以下一律不解讀。**")
        return 1

    # ── 三個排序並列 ─────────────────────────────────────────────────
    def show(col, label):
        s = t.dropna(subset=[col]).sort_values(col, ascending=False)
        print("\n%s（前 %d）" % (label, a.top))
        print("  %-22s %10s %8s %8s %9s %10s %8s %9s"
              % ("配對", col, "帶bps", "σbps", "事件/天", "深度$", "run", "成本bps"))
        print("  " + "-" * 88)
        for _, r in s.head(a.top).iterrows():
            print("  %-22s %10.2f %8.2f %8.2f %9.2f %10.0f %8.1f %9s"
                  % (r.pair[:22], r[col], r.band_bps, r.sigma_bps,
                     r.events_per_day, r.depth_usd, r.mean_run_samples,
                     "—" if r.cost_bps is None else "%.2f" % r.cost_bps))
        return s.pair.head(a.top).tolist()

    top0 = show("R0_frozen", "R0 凍結：fires x band x depth（**現行升格指標**）")
    top1 = show("R1_event", "R1 事件型：events x 半帶 x depth（09-03 修法搬上游）")
    top2 = show("R2_ratio", "R2 比值型：sigma(edge) / 來回成本")

    # ── 換不換人 ─────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("  **排名換不換人**（§1.30 C2 要產出的就是這個）")
    print("=" * 92)
    for lab, tb in (("R1 事件型", top1), ("R2 比值型", top2)):
        ov = len(set(top0) & set(tb))
        print("  R0 vs %-8s 前 %d 名交集 **%d**（%.0f%%）—— %s"
              % (lab, a.top, ov, 100 * ov / max(1, a.top),
                 "指標換了東西" if ov < a.top * 0.7 else "沒有實質差異"))

    # ── D3 持續性：假說的判準 ────────────────────────────────────────
    r0run = t[t.pair.isin(top0)].mean_run_samples.median()
    r1run = t[t.pair.isin(top1)].mean_run_samples.median()
    allrun = t.mean_run_samples.median()
    print("\nD3 持續性（連續在帶上的平均樣本數，中位）")
    print("   R0 前段 **%.2f**｜R1 前段 %.2f｜全體 %.2f" % (r0run, r1run, allrun))
    hyp = r0run > r1run * 1.3
    print("   假說「凍結指標偏好持續偏移」：%s"
          % ("**成立**（R0 前段的 run 明顯更長）" if hyp
             else "**不成立 —— 我那個機制說法要撤回**"))

    # ── D4 BTC 對照組 ────────────────────────────────────────────────
    print("\nD4 BTC 對照組（兩個深簿、帶是雜訊底，任何排序裡都該在後段）")
    for lab, col in (("R0", "R0_frozen"), ("R1", "R1_event"), ("R2", "R2_ratio")):
        s = t.dropna(subset=[col]).sort_values(col, ascending=False)
        s = s.reset_index(drop=True)
        btc = s.index[s.sym.astype(str).str.upper().str.startswith("BTC")]
        pct = (100 * (btc.min() + 1) / len(s)) if len(btc) else float("nan")
        print("   %s：最高名次的 BTC 配對在前 %.1f%% -> %s"
              % (lab, pct, "OK" if pct > 20 else "**可疑：對照組跑到前段**"))

    res = dict(asof=pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M"),
               span_days=round(span, 2), rows=int(len(df)),
               pairs_scored=int(len(t)), sample_gap_sec=round(med_gap, 1),
               d1_pass=bool(d1), d1_overlap=int(inter),
               overlap_R0_R1=int(len(set(top0) & set(top1))),
               overlap_R0_R2=int(len(set(top0) & set(top2))),
               run_R0=float(r0run), run_R1=float(r1run), run_all=float(allrun),
               hypothesis_persistence=bool(hyp),
               top=dict(R0=top0, R1=top1, R2=top2))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("\n  -> %s" % OUT)
    print("\n  **本支不下判決**（§1.30 C2 的判準還沒凍結）。它要產出的是")
    print("  「排名換不換人」與「凍結指標是不是偏好持續偏移」兩個事實。")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=20, help="每個排序印前幾名")
    raise SystemExit(main(p.parse_args()))
