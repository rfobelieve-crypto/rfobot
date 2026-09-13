# -*- coding: utf-8 -*-
"""把檔案庫 92 篇裡的**具體 alpha** 掃出來，變成一張可以逐項打勾的清單。

===========================================================================
為什麼要這支
===========================================================================
使用者 2026-09-13：「Quant arb 之前有寫，還有很多 alpha 策略可以抓」。

我們到今天為止都是**一篇一篇讀**：撞到問題 -> 去找一篇 -> 讀 -> 轉譯。
結果是「讀過的那幾篇很熟、沒讀過的等於不存在」，而 92 篇裡**只有 5 篇**
被轉成過程式碼（§1.23 的 A Real HFT/MFT Alpha、§1.37 的 lead-lag、
今天的 HFT Alphas Pt.1、成本模型的 alpha-6、band 的 alpha-6）。

這支不解讀內容，**只做定位**：哪一篇有具體到可以實作的東西、在第幾行。
解讀還是要人去讀那幾行 —— 但至少不會再有「那篇原來有寫」這種事。

判斷「具體」的三個訊號（都很便宜，寧可多報不可漏報）：
  1. 有程式碼區塊（```）或 `def ` / `np.` / `.rolling(` 之類
  2. 有明確的特徵名（底線命名、或「we define ... as」）
  3. 有可驗證的數字（Sharpe / IC / bps / %），代表它量過不是空談
"""
from __future__ import annotations

import json
import os
import re
import sys

TXT = "D:/flowbot_data/quant_arb/_txt"
OUT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))),
    "research", "results", "alpha_inventory.json")

# 具體特徵名的樣子：snake_case 且不是常見英文字
FEAT = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+){1,3})\b")
STOP = {"read_full", "full_story", "the_quant", "quant_stack", "e_g",
        "i_e", "et_al", "read_more", "sign_up", "pdf_file"}
CODEY = re.compile(r"```|^\s*def \s|np\.|pd\.|\.rolling\(|\.ewm\(|lambda |"
                   r"z-?score|zscore", re.I | re.M)
NUMY = re.compile(r"\b\d+(?:\.\d+)?\s*(?:sharpe|sortino|bps|IC|%)\b|"
                  r"\b(?:sharpe|sortino|IC)\s*(?:of|=|:)?\s*\d", re.I)
DEFY = re.compile(r"we define|is defined as|our alpha|the alpha is|"
                  r"the signal is|the feature is|we compute|we calculate|"
                  r"formula", re.I)


def scan(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().split("\n")
    body = "\n".join(lines)
    feats = {}
    for i, ln in enumerate(lines):
        for m in FEAT.findall(ln):
            if m in STOP or len(m) < 5:
                continue
            feats.setdefault(m, i + 1)
    hits = {"code": [i + 1 for i, l in enumerate(lines) if CODEY.search(l)][:8],
            "numbers": [i + 1 for i, l in enumerate(lines) if NUMY.search(l)][:8],
            "definitions": [i + 1 for i, l in enumerate(lines)
                            if DEFY.search(l)][:8]}
    # 索引那一行（Substack 慣例：「1. Introduction 2. Index 3. ...」）
    idx = next((l.strip() for l in lines
                if re.match(r"^\s*1\.\s*Introduction", l)), "")
    score = (3 * len(hits["code"]) + 2 * len(hits["definitions"])
             + len(hits["numbers"]) + min(len(feats), 12))
    return {"file": os.path.basename(path), "lines": len(lines),
            "words": len(body.split()), "index": idx[:300],
            "feats": sorted(feats, key=lambda k: feats[k])[:14],
            "hits": hits, "score": score}


def main():
    fs = sorted(glob_all())
    rows = [scan(f) for f in fs]
    rows.sort(key=lambda r: -r["score"])
    print("掃了 %d 篇\n" % len(rows))
    print("%-52s %6s %5s %5s %5s %5s"
          % ("文章", "字數", "分數", "碼", "定義", "數字"))
    print("-" * 84)
    for r in rows[:28]:
        print("%-52s %6s %5d %5d %5d %5d"
              % (r["file"][:52], format(r["words"], ","), r["score"],
                 len(r["hits"]["code"]), len(r["hits"]["definitions"]),
                 len(r["hits"]["numbers"])))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    print("\n寫出 %s" % OUT)
    print("\n（分數只是**定位**用的粗排序，不是價值判斷 ——"
          "它高只代表那一篇有具體到可以實作的東西）")


def glob_all():
    import glob
    return [f for f in glob.glob(TXT + "/*")
            if f.endswith((".txt", ".md")) and "INDEX" not in f]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
