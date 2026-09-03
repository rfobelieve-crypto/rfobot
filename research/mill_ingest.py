# -*- coding: utf-8 -*-
"""磨坊逐筆資料攝取＋首輪體檢（研究端裁示_磨坊regime提案 §三 的研究端半邊）。

產品端 V1.51.4 開了匯出端點（token 保護），V1.62.0 搬到
/api/research/export/fills（header-only token、只收 id）。這支：
  1. 拉逐筆成交 → research/results/mill_fills.jsonl（本地留檔，可重跑）
  2. FIFO 配對成回合（同幣：買開→賣平），算每回合淨損益（含手續費）
  3. 印首輪體檢：回合數、每回合淨益、勝率、費用佔比 —— 這是「磨坊
     賺的是本事還是行情」判決的原料，不是判決本身

連線設定放環境變數（.env，不進 git）：
  MILL_EXPORT_URL   例 https://<jarvis-railway>/api/research/export/fills
                    （舊的 /api/u/export/... 會被本檔自動改寫，不必手動改）
  MILL_EXPORT_TOKEN 與 Railway 上 RESEARCH_EXPORT_TOKEN 同值（只走 header）
  MILL_EXPORT_UID   使用者 **id**（16 位 hex）——V1.62.0 起不收帳戶名

誠實註記：
  - src 歸屬 2026-08-24 前不可靠（cid_known=false），統計只用可靠列，
    不可靠列另計不混入
  - FIFO 配對是研究端的重建，不是產品端的帳——量級對不上時先查配對
    邏輯再查產品端
  - 這裡算的是磨坊「已平回合」的實現損益；在途庫存的浮動不在內
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "results" / "mill_fills.jsonl"
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _env(name):
    v = os.environ.get(name, "").strip()
    if not v:
        env_fp = ROOT / ".env"
        if env_fp.exists():
            for line in env_fp.read_text(encoding="utf-8").splitlines():
                if line.startswith(name + "="):
                    v = line.split("=", 1)[1].strip().strip('"')
                    break
    return v


def _export_url(base: str, leaf: str) -> str:
    """Normalise the export URL to the V1.62.0 location.

    2026-09-02 handover (產品端請求_研究端匯出搬家): the endpoint moved from
    /api/u/export/* to /api/research/export/*, the token is header-only
    (a token in the query string now returns 400 — it leaks into access logs
    and referers), and uid must be the 16-hex id, not the account name.
    Old URLs still in .env are rewritten here so the migration is one edit,
    not a scavenger hunt across machines.
    """
    base = base.replace("/api/u/export/", "/api/research/export/")
    if base.rstrip("/").endswith("/export"):
        base = base.rstrip("/") + "/" + leaf
    for other in ("fills", "v7"):
        if base.endswith("/export/" + other) and other != leaf:
            base = base[: -len(other)] + leaf
    return base


def _check_uid(uid: str) -> bool:
    """The product side stopped accepting account names (they are guessable)."""
    ok = len(uid) == 16 and all(c in "0123456789abcdefABCDEF" for c in uid)
    if not ok:
        print(f"MILL_EXPORT_UID='{uid}' 不是 16 位 hex 的使用者 id。"
              "產品端 V1.62.0 起只收 id（名字回 404）——請把 .env 裡的名字"
              "換成 id 後重跑。")
    return ok


def fetch():
    base, tok, uid = _env("MILL_EXPORT_URL"), _env("MILL_EXPORT_TOKEN"), _env("MILL_EXPORT_UID")
    if not (base and tok and uid):
        print("尚未設定 MILL_EXPORT_URL / MILL_EXPORT_TOKEN / MILL_EXPORT_UID（.env）")
        print("Railway 那端也要設同值的 RESEARCH_EXPORT_TOKEN。設好後重跑本腳本。")
        return None
    if not _check_uid(uid):
        return None
    url = _export_url(base, "fills") + "?" + urllib.parse.urlencode({"uid": uid})
    req = urllib.request.Request(url, headers={"x-export-token": tok})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    rows = d.get("rows", [])
    OUT.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows),
                   encoding="utf-8")
    print(f"拉到 {len(rows)} 筆成交 → {OUT.name}")
    return rows


def fifo_rounds(fills):
    """同幣 FIFO 配對：買單進佇列，賣單依序沖銷。回傳已平回合列表。"""
    rounds = []
    for mkt, fs in _by_market(fills).items():
        buys = deque()          # (t, price, size, fee_per_base)
        for f in fs:
            px, sz = float(f["price"]), abs(float(f["sizeBase"]))
            fee = abs(float(f["fee"] or 0.0))
            fee_pb = fee / sz if sz else 0.0
            if f["side"] == "buy":
                buys.append([f["t"], px, sz, fee_pb])
                continue
            # sell：沖銷最早的買
            remain = sz
            while remain > 1e-12 and buys:
                b = buys[0]
                take = min(remain, b[2])
                pnl = (px - b[1]) * take - (fee_pb + b[3]) * take
                rounds.append({"mkt": mkt, "open_t": b[0], "close_t": f["t"],
                               "size": take, "entry": b[1], "exit": px,
                               "pnl": pnl,
                               "fees": (fee_pb + b[3]) * take,
                               "gross": (px - b[1]) * take})
                b[2] -= take
                remain -= take
                if b[2] <= 1e-12:
                    buys.popleft()
    return rounds


def _by_market(fills):
    d = defaultdict(list)
    for f in fills:
        d[f["marketId"]].append(f)
    for v in d.values():
        v.sort(key=lambda x: x["t"])
    return d


def main() -> int:
    rows = fetch()
    if rows is None:
        return 1
    grid = [f for f in rows if f.get("src") == "grid" and f.get("mode") == "live"]
    reliable = [f for f in grid if f.get("cid_known")]
    print(f"live 磨坊成交 {len(grid)} 筆（歸屬可靠 {len(reliable)}、"
          f"08-24 前不可靠 {len(grid) - len(reliable)}——統計只用可靠列）")
    rounds = fifo_rounds(reliable)
    if not rounds:
        print("尚無已平回合——資料續累積")
        return 0
    import statistics as st
    pnls = [r["pnl"] for r in rounds]
    fees = sum(r["fees"] for r in rounds)
    gross = sum(r["gross"] for r in rounds)
    win = sum(1 for p in pnls if p > 0)
    print(f"\n磨坊首輪體檢（已平回合，含費）")
    print(f"  回合 {len(rounds)}｜淨益合計 {sum(pnls):+.2f}U｜"
          f"每回合 {st.mean(pnls):+.4f}U｜勝率 {win}/{len(rounds)}")
    print(f"  毛利 {gross:+.2f}U｜費用 {fees:.2f}U"
          f"（吃掉毛利的 {fees / abs(gross) * 100 if gross else 0:.0f}%）")
    (ROOT / "research" / "results" / "mill_rounds.json").write_text(
        json.dumps(rounds, ensure_ascii=False, indent=1), encoding="utf-8")
    print("回合明細 → mill_rounds.json（配 regime 標籤與 Kelly 是下一步）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
