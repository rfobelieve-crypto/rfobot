# -*- coding: utf-8 -*-
"""把 ep9 的稿子送進網站的 /writeups（2026-09-12）。

**為什麼寫成腳本而不是手改 JSON**：手改的東西下次沒人重現得出來，而且
`writeups.json` 是 CRLF + indent 2 的檔案，手改很容易整檔重排，
讓一筆新增的 diff 變成幾百行（今天在 `research_nogo.json` 已經踩過一次）。

**內文的真相源是 `make_ep9_bps_on_volume.py` 的 `BODY`**，本支只做格式轉換
（純文字 -> blocks）。這樣改稿只要改一個地方，docx 與網站不會漂開。

> **文件與現實不符，順手記一筆**：CLAUDE.md §使用者可見改動的同步規則寫的
> 管線是 `assets/extract_for_site.py` -> `assets/site_writeups.json` ->
> 複製到 product-site。**那兩個檔案都已經不存在**，現在 `writeups.json`
> 只活在 product-site 一側。ep8 是怎麼進去的沒有留下腳本。
"""
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "assets"))
SITE = ROOT.parent / "product-site" / "content" / "writeups.json"

from make_ep9_bps_on_volume import BODY                      # noqa: E402

SLUG = "bps-on-volume"
DATE = "2026-09-12"
TAGS = ["複製研究", "執行成本", "Gate 0"]

# 小標一律列舉，不用啟發式判斷 —— 猜錯一行，網站上就會有一段變成標題，
# 而那種錯不會有任何東西變紅。
HEADINGS = {
    "先講好消息：機制複製出來了",
    "然後我開始算錢",
    "分歧不在數字，在於它要跟什麼比",
    "「那就掛限價單啊」",
    "他沒有誤導任何人",
    "那這個訊號是廢的嗎",
    "一句話收尾",
}


def to_blocks(body: str):
    lines = [x.strip() for x in body.strip().split("\n")]
    lines = [x for x in lines if x]
    title, rest = lines[0], lines[1:]
    blocks = [{"type": "h" if x in HEADINGS else "p", "text": x} for x in rest]
    hit = {b["text"] for b in blocks if b["type"] == "h"}
    missing = HEADINGS - hit
    if missing:
        raise SystemExit("這些小標在內文裡找不到（改稿時漏了同步？）：%s" % missing)
    return title, blocks


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    title, blocks = to_blocks(BODY)

    raw = io.open(SITE, "rb").read().decode("utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"      # 沿用原檔行尾，別讓整檔重排
    data = json.loads(raw)
    if any(x.get("slug") == SLUG for x in data):
        print("已經有 %s，不重複加入。" % SLUG)
        return 0

    # 這份清單是舊 -> 新（silent-failures 最舊排第一），所以往後面接
    data.append({"slug": SLUG, "date": DATE, "tags": TAGS,
                 "zh": {"title": title, "subtitle": "", "blocks": blocks}})

    txt = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    io.open(SITE, "w", encoding="utf-8", newline=nl).write(txt)

    nh = sum(1 for b in blocks if b["type"] == "h")
    print("寫入 -> %s" % SITE)
    print("  slug %s、日期 %s、標題「%s」" % (SLUG, DATE, title))
    print("  段落 %d（小標 %d、內文 %d）、共 %d 篇"
          % (len(blocks), nh, len(blocks) - nh, len(data)))

    # 公開面硬規則的機械檢查（CLAUDE.md §對外網站呈現面）：
    # 只出百分比/bps/方向/時間/計數，不出美元金額、張數、帳戶權益。
    body_txt = title + " " + " ".join(b["text"] for b in blocks)
    bad = [t for t in ("$", "USD", "美元", "張合約", "帳戶權益", "餘額")
           if t in body_txt]
    if bad:
        raise SystemExit("公開面違規：%s" % bad)
    print("  公開面檢查：只有 bps 與百分比 -> PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
