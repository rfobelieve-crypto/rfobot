# -*- coding: utf-8 -*-
"""把 walk-forward OOS 預測匯出成 jarvis 網格回測用的 flow_signals.json。

## 為什麼要有這支腳本

jarvis 那邊的 `backtest/data/flow_signals.json` 原本是**一次性手工產物** ——
兩個 repo 都沒有產生它的程式碼（2026-08-04 查證）。這代表每次要新資料都得重新
摸索一次：欄位名、時間語意（bar 標籤還是事件時刻）、是不是真的 OOS、有沒有混到
in-sample 的折，全靠記憶。而時間語意這件事在這個專案已經出過一次事
（mistake.md 2026-07-28：兩個 cohort 全部作廢）。

這支腳本把口徑固定下來。

## 輸出格式（與 jarvis 既有讀取端相容，不要改欄位名）

    {
      "v12": [{"ts": "2026-05-01 12:00:00", "p_no": .., "p_up": .., "p_dn": ..}, ...],
      "mag": [{"ts": "...", "y_pred": ..}, ...],
      "dir": [{"ts": "...", "pred_ret": ..}, ...]
    }

## 時間語意（重要，不要改）

`ts` 是 **bar 標籤**（該根 K 線的開盤時刻），不是訊號誕生時刻。訊號要到
`ts + 1h` 那根收盤後才存在。jarvis 端的所有回測腳本都用
`v12[j].t + H <= candles[i].time` 做嚴格 trailing 對齊，靠的就是這個約定。
**匯出端不要自作主張把 ts 平移**，否則 jarvis 那邊會平移兩次。

## 資料來源

    v12  research/results/dual_model/direction_v12_regime_T120_H8_oos.parquet
         （由 v12_train_regime.py 產生；跑之前要先跑 build_regime_labels.py）
    dir  research/results/dual_model/direction_reg_oos_mse.parquet
    mag  research/results/dual_model/magnitude_oos*.parquet（若存在）

用法：
    python research/dual_model/build_regime_labels.py --threshold 0.012 --horizon 8
    python research/dual_model/v12_train_regime.py    --threshold 0.012 --horizon 8
    python research/export_flow_signals.py [--out <路徑>]

預設輸出到 research/results/flow_signals.json；用 --out 指到 jarvis 的
backtest/data/flow_signals.json 就能直接被那邊的回測讀到。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DM = ROOT / "research" / "results" / "dual_model"

V12_FILE = DM / "direction_v12_regime_T120_H8_oos.parquet"
DIR_CANDIDATES = ["direction_reg_oos_mse.parquet", "direction_reg_oos_fresh.parquet"]
MAG_CANDIDATES = ["magnitude_oos.parquet", "magnitude_reg_oos.parquet"]

TS_FMT = "%Y-%m-%d %H:%M:%S"


def _ts_series(df: pd.DataFrame) -> pd.Series:
    """取出 bar 標籤欄位，統一成 tz-naive UTC 的字串。"""
    if "ts" in df.columns:
        s = pd.to_datetime(df["ts"], utc=True)
    else:
        idx = df.index
        s = pd.to_datetime(idx, utc=True).to_series(index=df.index)
    return s.dt.tz_convert("UTC").dt.tz_localize(None).dt.strftime(TS_FMT)


def load_v12() -> list[dict]:
    if not V12_FILE.exists():
        raise FileNotFoundError(
            f"{V12_FILE} 不存在。先跑 build_regime_labels.py 再跑 v12_train_regime.py")
    df = pd.read_parquet(V12_FILE)
    need = ["p_no_trend", "p_up_trend", "p_dn_trend"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"v12 parquet 缺欄位 {missing}，實際欄位 {list(df.columns)}")
    ts = _ts_series(df)
    out = [{"ts": t, "p_no": float(a), "p_up": float(b), "p_dn": float(c)}
           for t, a, b, c in zip(ts, df["p_no_trend"], df["p_up_trend"], df["p_dn_trend"])]
    return out


def load_optional(candidates: list[str], col_map: dict[str, str]) -> list[dict]:
    for name in candidates:
        f = DM / name
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        src = next((c for c in col_map if c in df.columns), None)
        if src is None:
            continue
        ts = _ts_series(df)
        key = col_map[src]
        return [{"ts": t, key: float(v)} for t, v in zip(ts, df[src])]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "research" / "results" / "flow_signals.json"))
    args = ap.parse_args()

    payload = {"v12": load_v12()}
    payload["dir"] = load_optional(DIR_CANDIDATES, {"pred_ret": "pred_ret", "pred": "pred_ret"})
    payload["mag"] = load_optional(MAG_CANDIDATES, {"y_pred": "y_pred", "pred": "y_pred"})

    for k, v in payload.items():
        if v:
            print(f"{k:5s} n={len(v):6d}  {v[0]['ts']} → {v[-1]['ts']}")
        else:
            print(f"{k:5s} （來源檔不存在，本次未匯出；jarvis 端若需要請沿用舊檔）")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"\n已存 {out}")
    print("提醒：ts 是 bar 標籤，jarvis 端會用 ts+1h 做 trailing 對齊，此處不要平移。")


if __name__ == "__main__":
    main()
