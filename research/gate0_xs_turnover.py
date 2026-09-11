# -*- coding: utf-8 -*-
"""Gate 0：1 小時橫斷面 alpha 的換手成本（2026-09-11）

===========================================================================
這支刻意**只算成本，不算損益**
===========================================================================
外部閱讀〈A Real HFT/MFT Alpha〉提出一個 1 小時橫斷面簿口 alpha：
按掛單「年齡」（用「上一分鐘這一檔有沒有超過 $100」代理）把深度拆成新/舊，
各自算 5/10 bps 帶內的失衡，合成 `new − old`，橫斷面 z 分數當權重，
**每小時再平衡**。他宣稱各 >2 Sharpe —— 但那是**原始訊號、未扣成本**。

CLAUDE.md 核心原則 11（Gate 0）說執行可行性排在資訊層之前。
套到這裡就是：**先算它每小時要付多少手續費，再決定要不要看它賺多少。**

**而換手率可以在完全不看報酬的情況下算出來** —— 它只跟特徵的時序穩定度有關。
所以這支的輸出是一個**門檻**：「原始訊號每小時至少要賺幾 bps 才不賠」。
算完之後才知道值不值得做下一步。這個順序是刻意的：
先看損益會讓人捨不得，先看成本不會。

===========================================================================
資料
===========================================================================
`orderbook_snapshots_1m` 的 `raw_levels`（完整 L20 JSON），Binance，11 個標的。
每個小時的**再平衡時刻 t** 需要兩筆：t 與 **t−1 分鐘**（新舊拆分要比對前一筆）。

**不是拿 HL 的資料算**：HL 的中價錄製 2026-09-11 才開始，只有幾分鐘。
用 Binance 的 120 天歷史算換手率，然後**套 HL 的費率**——
換手率是特徵的性質（跨場館差不多），費率才是場館的性質。
這個借用要明說，因為它是一個假設。

    python research/gate0_xs_turnover.py --days 30
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
OUT = ROOT / "research" / "results" / "gate0_xs_turnover.json"

OLD_USD = 100.0              # 抄自外部來源，未經我們驗證（見 hl_mid 的同名常數）
BANDS = (5, 10)              # 他用的兩個帶
HL_TAKER_BPS = 4.5           # gate0.py 實查
HL_MAKER_BPS = 1.5


def band_split(raw, mid, prev_map, side, band_bps):
    """回傳 (new_usd, old_usd)：帶內深度按「上一分鐘這一檔有沒有量」拆分。"""
    new = old = 0.0
    for px_s, sz_s in (raw or []):
        try:
            px, sz = float(px_s), float(sz_s)
        except Exception:
            continue
        if px <= 0 or mid <= 0:
            continue
        if abs(px - mid) / mid * 1e4 > band_bps:
            continue
        usd = px * sz
        if prev_map.get((side, px_s), 0.0) > OLD_USD:
            old += usd
        else:
            new += usd
    return new, old


def imb(a, b):
    t = a + b
    return (a - b) / t if t > 1e-9 else np.nan


def load_pairs(days):
    """抓每個整點的 (t, t−1min) 兩筆快照。"""
    from shared.db import get_db_conn
    conn = get_db_conn()
    q = """
        SELECT canonical_symbol, ts_ms, mid_price, raw_levels
        FROM orderbook_snapshots_1m
        WHERE created_at >= DATE_SUB(UTC_TIMESTAMP(), INTERVAL %s DAY)
          AND MINUTE(FROM_UNIXTIME(ts_ms/1000)) IN (0, 59)
        ORDER BY ts_ms
    """
    d = pd.read_sql(q, conn, params=(days,))
    conn.close()
    return d


def build(d):
    d = d.copy()
    d["minute"] = (d.ts_ms // 60000)
    d["is_reb"] = (pd.to_datetime(d.ts_ms, unit="ms").dt.minute == 0)
    rows = []
    for sym, g in d.groupby("canonical_symbol"):
        g = g.sort_values("minute")
        prev_by_min = {}
        for _, r in g.iterrows():
            try:
                lv = json.loads(r.raw_levels) if isinstance(r.raw_levels, str) else r.raw_levels
            except Exception:
                continue
            bids, asks = lv.get("bids") or [], lv.get("asks") or []
            pm = {}
            for px_s, sz_s in bids:
                try:
                    pm[("b", px_s)] = float(px_s) * float(sz_s)
                except Exception:
                    pass
            for px_s, sz_s in asks:
                try:
                    pm[("a", px_s)] = float(px_s) * float(sz_s)
                except Exception:
                    pass
            if r.is_reb:
                prev = prev_by_min.get(int(r.minute) - 1)
                if prev is not None:
                    rec = dict(sym=sym, minute=int(r.minute),
                               # 2026-09-11 加：mid 與 new/old 分開留。
                               # 原因：(a) 報酬目標要用 mid 不用成交價
                               # （mistake.md 2026-09-11）；(b) 他的核心機制
                               # 主張是「新舊兩個失衡符號相反」，只留合成的
                               # i_new − i_old **驗不了那一句**。
                               # **`combo%d` 的算式一個字沒動**，所以凍結的
                               # 換手數字不會移動（mft_xs_alpha.py 的 C1 會驗）。
                               mid=float(r.mid_price))
                    ok = True
                    for bnd in BANDS:
                        bn, bo = band_split(bids, r.mid_price, prev, "b", bnd)
                        an, ao = band_split(asks, r.mid_price, prev, "a", bnd)
                        i_new, i_old = imb(bn, an), imb(bo, ao)
                        if not (np.isfinite(i_new) and np.isfinite(i_old)):
                            ok = False
                        rec["new%d" % bnd] = i_new
                        rec["old%d" % bnd] = i_old
                        rec["combo%d" % bnd] = i_new - i_old
                    if ok:
                        rows.append(rec)
            prev_by_min[int(r.minute)] = pm
    return pd.DataFrame(rows)


def turnover(f, col):
    """橫斷面 z 分數 -> 縮放到總槓桿 1 -> 每次再平衡的單邊換手。"""
    p = f.pivot_table(index="minute", columns="sym", values=col)
    z = p.sub(p.mean(axis=1), axis=0).div(p.std(axis=1).replace(0, np.nan), axis=0)
    w = z.div(z.abs().sum(axis=1).replace(0, np.nan), axis=0)   # 總槓桿 = 1
    dw = w.diff().abs().sum(axis=1) / 2.0                        # 單邊換手
    return w, dw.dropna()


def weights_from(p, halflife=None):
    """特徵矩陣 -> 橫斷面 z -> 總槓桿 1 的權重。halflife 給了就先做 EMA。"""
    if halflife:
        p = p.ewm(halflife=halflife, min_periods=1).mean()
    z = p.sub(p.mean(axis=1), axis=0).div(p.std(axis=1).replace(0, np.nan), axis=0)
    return z.div(z.abs().sum(axis=1).replace(0, np.nan), axis=0)


def apply_band(w, band):
    """不交易帶：|Δw| 小於 band 就沿用上一期的權重。

    **逐列前推**，不能向量化——因為「上一期」是抑制後的值不是原始值，
    這個遞迴正是抑制的本體。寫錯成拿原始權重比會低估抑制效果。
    """
    if band <= 0:
        return w
    out = w.copy()
    prev = None
    for i in range(len(w)):
        cur = w.iloc[i]
        if prev is None:
            prev = cur.fillna(0.0)
            out.iloc[i] = prev
            continue
        keep = (cur - prev).abs() < band
        nw = cur.where(~keep, prev).fillna(prev)
        # 動完之後重新縮放回總槓桿 1（否則帶會讓槓桿漂走）
        s_ = nw.abs().sum()
        if s_ > 1e-12:
            nw = nw / s_
        out.iloc[i] = nw
        prev = nw
    return out


def sweep_damping(f, col):
    """回傳一張表：抑制強度 -> (換手, 成本, 與原始權重的相關)。"""
    p = f.pivot_table(index="minute", columns="sym", values=col)
    base = weights_from(p)
    base_flat = base.values.ravel()

    def row(name, w):
        dw = (w.diff().abs().sum(axis=1) / 2.0).dropna()
        m = float(dw.mean())
        ok = np.isfinite(base_flat) & np.isfinite(w.values.ravel())
        corr = (float(np.corrcoef(base_flat[ok], w.values.ravel()[ok])[0, 1])
                if ok.sum() > 10 else float("nan"))
        return dict(name=name, turnover=round(m, 4),
                    cost_taker_bps_h=round(m * 2 * HL_TAKER_BPS, 3),
                    cost_maker_bps_h=round(m * 2 * HL_MAKER_BPS, 3),
                    corr_with_base=round(corr, 4))

    out = [row("原始（無抑制）", base)]
    for band in (0.005, 0.01, 0.02, 0.05):
        out.append(row("不交易帶 %.3f" % band, apply_band(base, band)))
    for hl in (1, 2, 4, 8):
        out.append(row("EMA 半衰期 %dh" % hl, weights_from(p, halflife=hl)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--sweep", action="store_true",
                    help="掃換手抑制（一樣只看成本，不看損益）")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("撈 %d 天的整點與前一分鐘快照…" % a.days)
    raw = load_pairs(a.days)
    print("  %d 列、%d 標的" % (len(raw), raw.canonical_symbol.nunique()))
    f = build(raw)
    if f.empty:
        print("沒有配對得起來的快照"); return 1
    print("  配成 %d 個 (標的, 小時)、%d 個再平衡時點"
          % (len(f), f.minute.nunique()))

    res = dict(asof=time.strftime("%Y-%m-%d %H:%M:%S"), days=a.days,
               pairs=int(len(f)), rebalances=int(f.minute.nunique()),
               symbols=int(f.sym.nunique()), arms={})
    print("\n%-10s %10s %10s %10s %12s %12s"
          % ("帶", "換手中位", "換手均值", "p90", "taker bps/h", "maker bps/h"))
    for bnd in BANDS:
        col = "combo%d" % bnd
        if col not in f:
            continue
        _, dw = turnover(f, col)
        if not len(dw):
            continue
        m, med, p90 = float(dw.mean()), float(dw.median()), float(dw.quantile(.9))
        # 單邊換手 x 2（進出各一次）x 費率
        c_tk = m * 2 * HL_TAKER_BPS
        c_mk = m * 2 * HL_MAKER_BPS
        res["arms"]["%dbps" % bnd] = dict(
            turnover_median=round(med, 4), turnover_mean=round(m, 4),
            turnover_p90=round(p90, 4),
            cost_bps_per_hour_taker=round(c_tk, 3),
            cost_bps_per_hour_maker=round(c_mk, 3),
            cost_bps_per_day_taker=round(c_tk * 24, 2),
            n_rebalances=int(len(dw)))
        print("%-10s %10.3f %10.3f %10.3f %12.2f %12.2f"
              % ("%d bps" % bnd, med, m, p90, c_tk, c_mk))

    print("\n=== Gate 0 的門檻 ===")
    for k, v in res["arms"].items():
        print("  %-8s 原始訊號每小時至少要賺 **%.2f bps**（全 taker）"
              " / **%.2f bps**（全 maker）才不賠"
              % (k, v["cost_bps_per_hour_taker"], v["cost_bps_per_hour_maker"]))
        print("           換算每天 %.1f bps（taker）"
              % v["cost_bps_per_day_taker"])
    if a.sweep:
        print("\n" + "=" * 72)
        print("=== 換手抑制掃描（**一樣不算損益**）===")
        print("相關＝抑制後的權重向量 vs 原始權重向量。它量的是『部位變了多少』，")
        print("**不是**『賺多少』。相關撐住只是必要條件不是充分條件 ——")
        print("如果訊號的價值集中在它變化最快的時刻，抑制會砍掉最值錢的部分，")
        print("而權重相關仍然很高。")
        for bnd in BANDS:
            col = "combo%d" % bnd
            if col not in f:
                continue
            print("\n--- %d bps 帶 ---" % bnd)
            print("%-18s %10s %12s %12s %10s"
                  % ("抑制", "換手", "taker bps/h", "maker bps/h", "與原始相關"))
            rows = sweep_damping(f, col)
            res.setdefault("damping", {})["%dbps" % bnd] = rows
            for r in rows:
                print("%-18s %10.4f %12.2f %12.2f %10.3f"
                      % (r["name"], r["turnover"], r["cost_taker_bps_h"],
                         r["cost_maker_bps_h"], r["corr_with_base"]))

    print("\n**本支刻意沒有算任何損益。** 先看成本再決定要不要看損益 ——")
    print("反過來會捨不得（CLAUDE.md 核心原則 11 / common_cause_scan 假說 1）。")
    print("\n註：換手率用 Binance 的 %d 天歷史算（HL 中價才錄幾分鐘），"
          "費率用 HL 實查值。" % a.days)
    print("    這個借用是假設：**換手率是特徵的性質，費率才是場館的性質。**")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwritten -> " + str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
