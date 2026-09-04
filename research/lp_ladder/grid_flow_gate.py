# -*- coding: utf-8 -*-
"""網格 × 流動性波動閘門 —— 政策層驗證（預註冊 2026-09-05）

`research/flow_grid_gate.py` 在 10/10 幣上證明：撤單衝擊（`shock`）對未來 4h
波動**水平**有 trailing vol 之外的資訊（cond IC +0.09~+0.12，安慰劑歸零），
但對「波動加倍」這個尾巴事件沒有。所以閘門的職責是 sizing 不是生死：
**硬停損（§0.93 七/八已凍結）管尾巴，流閘門管部位。**

本檔是 `grid_adx_gate.py` 的克隆——同一套配對 MC（區塊 bootstrap 整根小時
bar，wick 保留）、同一個基準政策（每 90 天重錨＋硬停損）、同一個 `gate` 槽、
同一組漂移情境——**只換儀器**。§0.93 九用 ADX 進這個槽 NO-GO（對照組打平或
更好）；ADX 已被證明無前瞻力（ER/VR 雙證偽）。這裡問的是：換一個有前瞻力的
儀器，同一個槽會不會變成 GO。

**跟 ADX 版唯一的結構差異**：ADX 是價格的函數，可以在合成路徑上重算；
`shock` 不是。所以合成路徑重抽的是 **(bar, shock, trailvol) 三元組**——每根
被抽到的小時 bar 帶著它自己那一小時的撤單衝擊。區塊 48h 保留了「上一小時
的 shock → 這一小時的波動」的時序關係。代價：bar 宇宙只有 depth_deltas
覆蓋的 **57 天**（2026-07-09 → 09-04），不是 ADX 版的 2.55 年。

**在跑之前寫死的東西**

  訊號   shock_h(i) = 該小時 60 個分鐘 shock 的平均（shock 逐字沿用 watcher 凍結
         定義）。**因果**：bar i 的閘門用 shock_h(i−1) 決定（同 ADX 版的 shift）。
  門檻   shock_h 的 trailing 168 小時第 80 百分位（在合成路徑上算，無擬合）。
         前 168 根不設閘。
  對照一 TRAILVOL 閘門：同樣的構造，把 shock_h 換成 trailing 24h realized vol。
         **這是本檔的核心對照**——它就是 conditional IC 那個問題的政策版：
         流閘門若跟 trailing-vol 閘門打平，流沒有加任何東西。
  對照二 固定部位（ALWAYS-OFF, gate_scale=0.9）：半倉閘門大約 20% 時間半倉
         ≈ 平均 90% 部署；若固定 90% 做到一樣的事，閘門只是穿著指標外衣的
         去槓桿（§0.93 九的原話）。
  臂     無濾網基準 / FLOW 關閉 / FLOW 半倉 / TRAILVOL 關閉 / TRAILVOL 半倉 / 固定 90%
  情境   去漂移 / +30% / −30% / −60%（同 ADX 版，去漂移是隔離「收震盪租金」的那個）
  判準   **GO 必須同時滿足（寫死，不調）**：
         (1) 去漂移情境下 FLOW 半倉的年化中位 > TRAILVOL 半倉 **且** > 固定 90%
         (2) 逐路徑配對差（FLOW 半倉 − TRAILVOL 半倉）在去漂移情境的 bootstrap
             95% CI 不含零
         (3) 四個情境裡 (FLOW − TRAILVOL) 同號 ≥ 3
         (4) FLOW 半倉的 MDD p95 不比無濾網基準差超過 1 個百分點
         任一不過 = NO-GO。**不得事後換臂、換門檻、換情境。**
  成本   MAKER 2 bps 已在模擬器內；本檔另跑一次 MAKER=0 對照，含成本結果不得
         ≥ 零成本結果（mistake.md 2026-07-28：含成本比零成本還賺＝成本模型壞了）。
  功效   跑完報配對差的 SE 與「達 80% 功效需要的路徑數」；判準 (2) 本身是
         CI 型，構造上不會被雜訊矇過，但可能徒勞——徒勞就明寫。
  預測   FLOW 半倉在去漂移情境**小幅**優於 TRAILVOL 半倉（IC 0.11 不是大效應），
         MDD 改善有限（尾巴標籤 FAIL）。先驗：(1)(2) 五五開，(4) 大概率過。

Run: python research/lp_ladder/grid_flow_gate.py --paths 120
Out: research/results/lp_grid_flow_gate.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import grid_exec  # noqa: E402
from grid_exec import simulate  # noqa: E402
from grid_mc_policy import bar_stats  # noqa: E402
from shared.db import get_db_conn  # noqa: E402

OUT = ROOT / "research" / "results" / "lp_grid_flow_gate.json"
CSV = ROOT / "research" / "sweep_failure" / ".cache" / "BTCUSDT_1h.csv"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


# ------------------------------------------------------------------ data
def load_bars_with_flow():
    """Hourly BTC bars restricted to the depth_deltas overlap, each with its
    own hourly shock (mean of the frozen minute shock) and a causal
    trailing-24h realized vol."""
    import csv
    t, lo, hi, cl = [], [], [], []
    with open(CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            t.append(int(row["time"])); lo.append(float(row["low"]))
            hi.append(float(row["high"])); cl.append(float(row["close"]))
    t = np.array(t); lo = np.array(lo); hi = np.array(hi); cl = np.array(cl)

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT minute_start_ms m, bid_cancel_qty bc, ask_cancel_qty ac "
                "FROM depth_deltas_1m WHERE canonical_symbol='BTC-USD' "
                "AND exchange='binance_perp' ORDER BY minute_start_ms")
            rows = cur.fetchall()
    finally:
        conn.close()
    m = np.array([r["m"] for r in rows], np.int64)
    tot = np.array([float(r["bc"]) + float(r["ac"]) for r in rows])
    # frozen shock: tot / trailing-60m median (min 30)
    import pandas as pd
    s = pd.Series(tot)
    base = s.rolling(60, min_periods=30).median().replace(0, np.nan)
    shock = (s / base).values
    hour = (m // 3_600_000) * 3600            # epoch seconds, hour start
    df = pd.DataFrame({"h": hour, "shock": shock}).dropna()
    hs = df.groupby("h")["shock"].mean()

    # restrict bars to hours we have flow for
    keep = np.isin(t, hs.index.values)
    t, lo, hi, cl = t[keep], lo[keep], hi[keep], cl[keep]
    shock_h = hs.reindex(t).values
    # causal trailing-24h realized vol of hourly log returns
    r = np.diff(np.log(cl), prepend=np.log(cl[0]))
    tv = pd.Series(r).rolling(24, min_periods=12).std().values
    return t, lo, hi, cl, shock_h, tv


def synth_idx(n_src, n, block, rng):
    idx = np.empty(n, np.int64)
    i = 0
    while i < n:
        s = rng.integers(0, n_src - block)
        take = min(block, n - i)
        idx[i:i + take] = np.arange(s, s + take)
        i += take
    return idx


def synth(r, hi_r, lo_r, idx, drift_ann=None, demean=False):
    rr = r[idx]
    if demean or drift_ann is not None:
        rr = rr - rr.mean()
    if drift_ann is not None:
        rr = rr + math.log(1 + drift_ann) / (365 * 24)
    c = 100_000.0 * np.exp(np.cumsum(rr))
    return c * lo_r[idx], c * hi_r[idx], c


def causal_gate(feature, lb=168, q=0.80):
    """True = allowed to buy. Blocked when yesterday's feature exceeds its own
    trailing-lb 80th percentile. Decided from bar i-1, applied to bar i."""
    n = len(feature)
    hot = np.zeros(n, bool)
    for i in range(lb, n):
        w = feature[i - lb:i]
        w = w[np.isfinite(w)]
        if len(w) >= lb // 2 and np.isfinite(feature[i]):
            hot[i] = feature[i] > np.quantile(w, q)
    g = np.ones(n, bool)
    g[1:] = ~hot[:-1]
    return g


ARMS = [("無濾網（基準）", None, 0.0),
        ("FLOW 關閉買進", "FLOW", 0.0),
        ("FLOW 半倉", "FLOW", 0.5),
        ("TRAILVOL 關閉買進（對照）", "TV", 0.0),
        ("TRAILVOL 半倉（對照）", "TV", 0.5),
        ("固定 90% 倉位（對照）", "OFF", 0.9)]
SCEN = [("去漂移", dict(demean=True)),
        ("多頭 +30%/年", dict(drift_ann=0.30)),
        ("空頭 −30%/年", dict(drift_ann=-0.30)),
        ("空頭 −60%/年", dict(drift_ann=-0.60))]
BASE = dict(reanchor="time", stop="hard")


def run(paths, days, block, drop, bins, seed, maker=None):
    if maker is not None:
        grid_exec.MAKER = maker
    t, lo, hi, cl, shock_h, tv = load_bars_with_flow()
    r, hi_r, lo_r = bar_stats(lo, hi, cl)
    sh, tvv = shock_h[1:], tv[1:]          # align with bar_stats' diff
    n = days * 24
    rng = np.random.default_rng(seed)
    res = {}
    for sname, kw in SCEN:
        idxs = [synth_idx(len(r), n, block, rng) for _ in range(paths)]
        P = [synth(r, hi_r, lo_r, ix, **kw) for ix in idxs]
        bh = np.array([p[2][-1] / p[2][0] - 1 for p in P])
        gF = [causal_gate(sh[ix]) for ix in idxs]
        gT = [causal_gate(tvv[ix]) for ix in idxs]
        per = {}
        for aname, key, scale in ARMS:
            rets, mdds, beat, gof = [], [], [], []
            for k, (pl, ph, pc) in enumerate(P):
                g = (None if key is None else gF[k] if key == "FLOW"
                     else gT[k] if key == "TV" else np.zeros(n, bool))
                m_, _ = simulate(pl, ph, pc, drop=drop, N=bins, gate=g,
                                 gate_scale=scale, **BASE)
                rets.append(m_["cagr"]); mdds.append(m_["mdd"])
                beat.append((m_["final"] - 1) > bh[k]); gof.append(m_["gated_frac"])
            per[aname] = {"rets": np.array(rets), "mdds": np.array(mdds),
                          "beat": float(np.mean(beat)), "gated": float(np.mean(gof))}
        res[sname] = {"bh_med": float(np.median(bh)), "arms": per}
    return res, len(r)


def paired_ci(a, b, B=4000, seed=1):
    d = a - b
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(d, len(d)).mean() for _ in range(B)])
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(d.std(ddof=1) / math.sqrt(len(d)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=120)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--block", type=int, default=48)
    ap.add_argument("--drop", type=float, default=0.25)
    ap.add_argument("--bins", type=int, default=30)
    a = ap.parse_args()
    t0 = time.time()
    print("=" * 104)
    print(f"  流動性波動閘門 × 網格 — {a.paths} 條 {a.days} 天路徑/情境，"
          f"基準＝每90天重錨＋硬停損，區間 −{a.drop:.0%}；bar 宇宙＝depth_deltas 重疊期")
    print("=" * 104)
    res, nsrc = run(a.paths, a.days, a.block, a.drop, a.bins, 20260905)
    print(f"  bar 宇宙 {nsrc} 小時（{nsrc/24:.0f} 天）")
    out = {"params": vars(a), "n_src_bars": int(nsrc), "scen": {}}
    F, T, X = "FLOW 半倉", "TRAILVOL 半倉（對照）", "固定 90% 倉位（對照）"
    signs = []
    for sname, R in res.items():
        print(f"\n  [{sname}]  買進持有中位 {R['bh_med']:+.1%}")
        print(f"  {'臂':<26}{'年化中位':>10}{'年化平均':>10}{'p5':>9}{'虧損率':>8}"
              f"{'MDD中位':>9}{'MDD p95':>9}{'贏過持有':>9}{'擋掉時間':>9}")
        out["scen"][sname] = {}
        for aname, v in R["arms"].items():
            rr, mm = v["rets"], v["mdds"]
            row = {"med": float(np.median(rr)), "mean": float(rr.mean()),
                   "p5": float(np.percentile(rr, 5)), "loss": float((rr < 0).mean()),
                   "mdd_med": float(np.median(mm)), "mdd_p95": float(np.percentile(mm, 5)),
                   "beat_hold": v["beat"], "gated": v["gated"]}
            out["scen"][sname][aname] = row
            print(f"  {aname:<26}{row['med']:>+10.2%}{row['mean']:>+10.2%}{row['p5']:>+9.2%}"
                  f"{row['loss']:>8.0%}{row['mdd_med']:>+9.1%}{row['mdd_p95']:>+9.1%}"
                  f"{row['beat_hold']:>9.0%}{row['gated']:>9.0%}")
        d, lo_, hi_, se = paired_ci(R["arms"][F]["rets"], R["arms"][T]["rets"])
        d2, lo2, hi2, _ = paired_ci(R["arms"][F]["rets"], R["arms"][X]["rets"])
        out["scen"][sname]["paired_FLOW_minus_TV"] = {"mean": d, "ci": [lo_, hi_], "se": se}
        out["scen"][sname]["paired_FLOW_minus_FIX90"] = {"mean": d2, "ci": [lo2, hi2]}
        signs.append(np.sign(d))
        print(f"  配對 FLOW半倉 − TRAILVOL半倉: {d:+.3%} [{lo_:+.3%}, {hi_:+.3%}]  SE {se:.3%}"
              f"   | − 固定90%: {d2:+.3%} [{lo2:+.3%}, {hi2:+.3%}]")

    # ---------------- pre-registered verdict ----------------
    D = res["去漂移"]["arms"]
    c1 = (np.median(D[F]["rets"]) > np.median(D[T]["rets"])) and \
         (np.median(D[F]["rets"]) > np.median(D[X]["rets"]))
    dm, lo_, hi_, se = paired_ci(D[F]["rets"], D[T]["rets"])
    c2 = lo_ > 0 or hi_ < 0
    c3 = max(sum(1 for s in signs if s > 0), sum(1 for s in signs if s < 0)) >= 3
    mdd_base = np.percentile(D["無濾網（基準）"]["mdds"], 5)
    mdd_flow = np.percentile(D[F]["mdds"], 5)
    c4 = mdd_flow >= mdd_base - 0.01
    verdict = "GO" if (c1 and c2 and c3 and c4) else "NO-GO"
    need = (2.802 * se / abs(dm)) ** 2 * a.paths if dm else float("inf")
    print("\n" + "-" * 104)
    print(f"  判準 (1) 去漂移 FLOW半倉 中位 > 兩個對照: {'過' if c1 else '不過'}")
    print(f"  判準 (2) 配對差 CI 不含零: {'過' if c2 else '不過'}  ({dm:+.3%} [{lo_:+.3%}, {hi_:+.3%}])")
    print(f"  判準 (3) 四情境同號 ≥3: {'過' if c3 else '不過'}  (signs {[int(s) for s in signs]})")
    print(f"  判準 (4) MDD p95 不比基準差 >1pp: {'過' if c4 else '不過'}  (基準 {mdd_base:+.1%} vs FLOW {mdd_flow:+.1%})")
    print(f"  功效：配對差 SE {se:.3%}；以觀察到的差 {dm:+.3%} 達 80% 功效需 ≈ {need:.0f} 條路徑（現在 {a.paths}）")
    print(f"\n  ==> {verdict}")

    # ---------------- cost=0 control ----------------
    print("\n  成本對照（MAKER=0，只跑去漂移，兩臂）…")
    res0, _ = run(max(40, a.paths // 3), a.days, a.block, a.drop, a.bins, 20260905, maker=0.0)
    for arm in ("無濾網（基準）", F):
        with_c = float(np.median(res["去漂移"]["arms"][arm]["rets"]))
        no_c = float(np.median(res0["去漂移"]["arms"][arm]["rets"]))
        flag = "  << 含成本 ≥ 零成本，成本模型壞了" if with_c >= no_c else ""
        print(f"  {arm:<26} 含成本 {with_c:+.2%}  零成本 {no_c:+.2%}{flag}")
    out["verdict"] = {"go": verdict, "c1": bool(c1), "c2": bool(c2), "c3": bool(c3),
                      "c4": bool(c4), "paired": [dm, lo_, hi_, se], "paths_for_80pct_power": need}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n  {time.time()-t0:.0f}s -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
