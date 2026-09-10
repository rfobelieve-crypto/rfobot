# -*- coding: utf-8 -*-
"""凍結研究回歸測試要用的資料（一次性，產物進 git）。

**為什麼**：要把共用鷹架抽出來之前，必須先有一組「答案已知」的樣本把現有
結果釘住，否則重寫之後沒有任何東西能告訴你算術變了
（`assets/method.json` 第二條的應用）。

三本帳各有一個不可重現的來源：

    V7    `tracked_signals` 在 MySQL —— 測試不該依賴資料庫，而且那張表
          每小時都在長。凍結。
    OLD   `sweep_failure/.cache` 是滾動 930 天窗（mistake.md 2026-09-10），
          重建會漂。凍結。
    SDV   來自 data/events + data/levels，那兩個是靜態檔，本來就穩定，
          但一起凍結比較省事，也讓測試不用跑整個 ledger。

跑法（只有要換基準時才跑）：
    python research/tests/make_fixtures.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

OUT = Path(__file__).resolve().parent / "fixtures"


def main():
    import strategy_corr as sc  # noqa: E402

    OUT.mkdir(parents=True, exist_ok=True)
    man = {}
    for name, fn in (("v7", sc.load_v7), ("sdv", sc.load_sdv),
                     ("old", sc.load_old)):
        d = fn().sort_values(["sym", "entry_ms"]).reset_index(drop=True)
        p = OUT / f"book_{name}.parquet"
        d.to_parquet(p, index=False)
        b = p.read_bytes()
        man[name] = dict(n=int(len(d)), first_ms=int(d.entry_ms.min()),
                         last_ms=int(d.entry_ms.max()),
                         r_sum=round(float(d.r.sum()), 10),
                         sha256=hashlib.sha256(b).hexdigest()[:32])
        print("%-4s %6d 筆  r 總和 %+12.6f  sha %s"
              % (name, len(d), d.r.sum(), man[name]["sha256"][:12]))

    (OUT / "manifest.json").write_text(json.dumps(dict(
        frozen_at="2026-09-10",
        why=("V7 來自 MySQL（每小時在長）、OLD 來自滾動 930 天窗，"
             "兩者都不可重現；凍結之後回歸測試才有固定的輸入。"),
        books=man), indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nwritten -> " + str(OUT))


if __name__ == "__main__":
    main()
