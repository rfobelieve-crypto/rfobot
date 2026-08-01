# -*- coding: utf-8 -*-
"""Live V7 signals vs the research pipeline's signals — one picture.

Why this exists: terrain_fast_evidence ran the four confirmed dims on
BOTH populations and they disagreed (D1 and D5 flipped sign on the WF
OOS set). Before reading that as evidence about the dims, we have to know
whether the two populations are even comparable. They are produced by
different machinery:

  live      the production model, decoded by a 500-bar rolling percentile
            buffer, logged to tracked_signals as it fired
  research  per-fold walk-forward models, decoded inside the fold, saved
            to gate_a_revalidate_clean_oos.parquet

Known reason to distrust naive comparison: the 2026-04-19 warmup-buffer
fix changed the live decode (Strong volume fell from 114-175/month to
12-28/month), and fold models predict with a different spread than a
model trained on everything (mistake.md 2026-04-19: 3.5x std gap).

Panels:
  1 monthly signal counts per population — where each one even exists
  2 rolling accuracy — do they agree about how good the signals are
  3 same-bar agreement over the overlapping window — both / live-only /
    research-only, which is what decides whether the WF result carries
    information about the live book
  4 prediction spread — the failure mode that broke the buffer once

Run: python research/v7_signal_pipeline_compare.py
Out: research/results/v7_signal_pipeline_compare.png
"""
from __future__ import annotations

import sys
from collections import Counter
from datetime import timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from shared.db import get_db_conn  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/v7_signal_pipeline_compare.png"
OOS = ROOT / "research/results/gate_a_revalidate_clean_oos.parquet"
BG, FG, GRID = "#0e1116", "#d7dce3", "#1c222b"
C_STRONG, C_MOD, C_RES = "#00d1b2", "#7b6cff", "#f0b90b"

for k, v in {"figure.facecolor": BG, "axes.facecolor": BG,
             "savefig.facecolor": BG, "text.color": FG,
             "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
             "axes.edgecolor": GRID, "grid.color": GRID,
             "font.family": "Microsoft JhengHei"}.items():
    matplotlib.rcParams[k] = v


def load_live():
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT signal_time, strength, direction, correct, "
                "p_up, mag_pred FROM tracked_signals "
                "WHERE correct IS NOT NULL AND strength IN "
                "('Strong','Moderate') ORDER BY signal_time")
            return cur.fetchall()
    finally:
        conn.close()


def main() -> int:
    live = load_live()
    res = pd.read_parquet(OOS)
    res = res[res["strong"] != "none"].copy()
    res["correct"] = ((res["y"] > 0) == (res["pred_ret"] > 0)).astype(int)

    def mkey(ts):
        return f"{ts.year}-{ts.month:02d}"

    months = sorted({mkey(s["signal_time"]) for s in live}
                    | {mkey(d) for d in res.index})
    cnt = {k: Counter() for k in ("Strong", "Moderate", "research")}
    for s in live:
        cnt[s["strength"]][mkey(s["signal_time"])] += 1
    for d in res.index:
        cnt["research"][mkey(d)] += 1

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("V7 訊號：生產線 vs 研究管線（同一個模型家族，不同機器）",
                 color=FG, fontsize=14, y=0.98)

    # 1 monthly counts
    ax = axes[0][0]
    x = np.arange(len(months))
    w = 0.27
    ax.bar(x - w, [cnt["Strong"][m] for m in months], w, label="live Strong",
           color=C_STRONG)
    ax.bar(x, [cnt["Moderate"][m] for m in months], w, label="live Moderate",
           color=C_MOD)
    ax.bar(x + w, [cnt["research"][m] for m in months], w,
           label="研究 WF OOS", color=C_RES)
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax.axvline(months.index("2026-04") - 0.5 if "2026-04" in months else 0,
               color="#ff5c5c", ls="--", lw=1)
    ax.text(months.index("2026-04") if "2026-04" in months else 0,
            ax.get_ylim()[1] * 0.92, " 2026-04 warmup buffer 修復",
            color="#ff5c5c", fontsize=8)
    ax.set_title("① 每月訊號數：兩條管線根本不在同一段時間有量", fontsize=10)
    ax.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=FG)
    ax.grid(alpha=.25, axis="y")

    # 2 rolling accuracy (30-signal window)
    ax = axes[0][1]
    for label, xs, ys, col in (
            ("live Strong",
             [s["signal_time"] for s in live if s["strength"] == "Strong"],
             [s["correct"] for s in live if s["strength"] == "Strong"], C_STRONG),
            ("live Moderate",
             [s["signal_time"] for s in live if s["strength"] == "Moderate"],
             [s["correct"] for s in live if s["strength"] == "Moderate"], C_MOD),
            ("研究 WF OOS", list(res.index), list(res["correct"]), C_RES)):
        if len(ys) < 30:
            continue
        r = pd.Series(ys, index=pd.to_datetime(xs, utc=True)).rolling(30).mean()
        ax.plot(r.index, 100 * r.values, color=col, lw=1.4, label=label)
    ax.axhline(50, color="#ff5c5c", ls="--", lw=1)
    ax.set_ylabel("30 筆滾動方向準確率 %")
    ax.set_title("② 準確率：研究線只覆蓋 2026 上半，live 才有近期", fontsize=10)
    ax.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=FG)
    ax.grid(alpha=.25)

    # 3 same-bar agreement in the overlapping window
    ax = axes[1][0]
    lo, hi = res.index.min(), res.index.max()
    live_bars = {int(s["signal_time"].replace(tzinfo=timezone.utc).timestamp())
                 for s in live
                 if lo <= pd.Timestamp(s["signal_time"], tz="UTC") <= hi
                 and s["strength"] == "Strong"}
    res_bars = {int(d.timestamp()) for d in res.index}
    both = len(live_bars & res_bars)
    only_l = len(live_bars - res_bars)
    only_r = len(res_bars - live_bars)
    ax.bar(["兩邊都開火", "只有 live", "只有研究線"], [both, only_l, only_r],
           color=[C_STRONG, C_MOD, C_RES])
    for i, v in enumerate([both, only_l, only_r]):
        ax.text(i, v, f" {v}", ha="center", va="bottom", color=FG, fontsize=10)
    tot = both + only_l + only_r
    ax.set_title(f"③ 重疊窗口內同一根 K 是否都開火（重合率 "
                 f"{100*both/tot if tot else 0:.0f}%）", fontsize=10)
    ax.grid(alpha=.25, axis="y")

    # 4 on the bars where BOTH fire: same direction, and who was right?
    # (An earlier draft plotted live p_up against research pred_ret — a
    # probability against a return. Different units, unreadable, and it
    # dodged the question that actually matters.)
    ax = axes[1][1]
    live_dir = {int(s["signal_time"].replace(tzinfo=timezone.utc).timestamp()):
                (s["direction"], int(s["correct"]))
                for s in live if s["strength"] == "Strong"}
    res_dir = {int(d.timestamp()): ("UP" if r["pred_ret"] > 0 else "DOWN",
                                    int(r["correct"]))
               for d, r in res.iterrows()}
    shared = sorted(set(live_dir) & set(res_dir))
    agree = sum(1 for t in shared if live_dir[t][0] == res_dir[t][0])
    lc = 100 * np.mean([live_dir[t][1] for t in shared]) if shared else 0
    rc = 100 * np.mean([res_dir[t][1] for t in shared]) if shared else 0
    both_c = (100 * np.mean([live_dir[t][1] for t in shared
                             if live_dir[t][0] == res_dir[t][0]])
              if agree else 0)
    bars_ = [100 * agree / len(shared) if shared else 0, lc, rc, both_c]
    ax.bar(["方向一致", "live 對", "研究線對", "一致時\nlive 對"], bars_,
           color=[C_MOD, C_STRONG, C_RES, "#00ffa3"])
    for i, v in enumerate(bars_):
        ax.text(i, v, f" {v:.0f}%", ha="center", va="bottom", color=FG,
                fontsize=10)
    ax.axhline(50, color="#ff5c5c", ls="--", lw=1)
    ax.set_ylim(0, 105)
    ax.set_ylabel("%")
    ax.set_title(f"④ 同時開火的 {len(shared)} 根：方向一致嗎、誰對", fontsize=10)
    ax.grid(alpha=.25, axis="y")

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.text(0.5, 0.005,
             "研究管線＝逐折 walk-forward 模型（僅 2026 上半、fold 內解碼）；"
             "生產線＝正式模型＋500 根滾動百分位解碼。兩者訊號不可混用，"
             "門檻與預測尺度都不同。",
             ha="center", color="#8b93a1", fontsize=8)
    fig.savefig(OUT, dpi=140)
    print(f"  live Strong {sum(1 for s in live if s['strength']=='Strong')} · "
          f"Moderate {sum(1 for s in live if s['strength']=='Moderate')} · "
          f"研究 {len(res)}")
    print(f"  重疊窗口 {lo:%Y-%m-%d} ~ {hi:%Y-%m-%d}："
          f"兩邊都開火 {both} · 只有 live {only_l} · 只有研究 {only_r}")
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
