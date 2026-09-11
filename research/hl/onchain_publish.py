# -*- coding: utf-8 -*-
"""把鏈上錄製的狀態變成網站的內容檔（2026-09-11）

**為什麼是產生而不是手寫**：網站要顯示「錄了幾小時、覆蓋多少、驗證過幾關」。
那些數字已經有真相源（錄製器與驗證器自己寫的旗標），手抄一份到網站就是
第二份實作 —— 它會安靜地跟真正的那個不一致（mistake.md 2026-08-26，
prereg 看板那次就是這樣把 1127 寫成 346）。

所以這支**只重新整形**，不計算任何判決數字。

**公開面規則**（CLAUDE.md）：只出百分比、方向、時間與計數。
不出美元金額 —— 連市場層級的未平倉名目也不出，理由跟套利線的
`arb_publish` 一樣：一旦開始出美元，下一個人就會順手把部位金額也出上去。
保留「質化的規模說明」那一欄，因為「市場很多但我們只覆蓋一小部分」
是這條線最重要的誠實註記，它必須在剝掉金額之後還活著。

輸出 assets/onchain_status.json（真相源），再複製到
../product-site/content/onchain_status.json。
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "assets" / "onchain_status.json"
RES = ROOT / "research" / "results"
DATA = HERE / "data"


def jread(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return default


def main():
    fuel = jread(RES / "hl_fuel_last.json", {}) or {}
    ver = jread(RES / "hl_verify_last.json", {}) or {}
    tape = jread(RES / "hl_tape_last.json", {}) or {}
    addr = jread(DATA / "addresses.json", {}) or {}
    addr_t = jread(DATA / "addresses_tape.json", {}) or {}

    snaps = len(glob.glob(str(DATA / "snapshots" / "*.json")))
    mkt = jread(sorted(glob.glob(str(DATA / "market" / "*.json")))[-1]
                if glob.glob(str(DATA / "market" / "*.json")) else None, {}) or {}
    rows = mkt.get("rows") or []
    n_markets = len(rows)
    n_venues = len({r.get("dex") for r in rows}) if rows else 0

    checks = ver.get("checks") or {}
    n_pass = sum(1 for v in checks.values()
                 if isinstance(v, dict) and v.get("ok"))
    n_checks = sum(1 for v in checks.values() if isinstance(v, dict))

    out = dict(
        _readme=("鏈上量化的狀態卡。**由 research/hl/onchain_publish.py 產生**，"
                 "不要手改 —— 數字的真相源是錄製器與驗證器自己寫的旗標。"
                 "公開面規則：只出百分比、方向、時間與計數，不出美元金額。"),
        updated=time.strftime("%Y-%m-%d"),
        asof_utc=fuel.get("asof") or time.strftime("%Y-%m-%d %H:%M:%S"),
        stage="recording",          # recording -> prereg -> verdict
        venues=n_venues,
        markets=n_markets,
        snapshots=snaps,
        addresses=int(addr.get("n") or 0) + int(addr_t.get("n") or 0),
        # 只有完整掃描的覆蓋率才發布。部分掃描（--max-addr 截短）的數字必然
        # 偏低，把它貼上網站就是用一個人為造成的低估去描述系統。
        coverage_pct=(None if fuel.get("coverage_partial")
                      else round(100 * float(fuel.get("coverage_frac") or 0), 2)),
        coverage_partial=bool(fuel.get("coverage_partial")),
        addrs_polled=int(fuel.get("addrs_polled") or 0),
        checks_passed=n_pass,
        checks_total=n_checks,
        recorder_ok=bool(fuel.get("ok")),
        verify_ok=bool(ver.get("ok")),
        tape_ok=bool(tape.get("ok")),
        zh=dict(
            name="鏈上量化",
            desc=("永續 DEX 的逐地址部位、清算價、真實止損單與全市場成交帶。"
                  "中心化交易所不公開這些，只能用代理推估 —— 而那在這套系統上"
                  "連續失敗了三次。這條線的前提是：不推估，直接讀。"),
            stage="錄製中（判準還沒寫，要先有真實分布才知道門檻該放哪）",
            note=("市場數很多，但我們每小時只覆蓋其中一部分的未平倉量；"
                  "覆蓋率跟著資料一起存，否則事後無法判斷一張「前方沒有燃料」"
                  "的圖是真的沒有，還是只是沒看到。"),
        ),
        en=dict(
            name="On-chain quant",
            desc=("Per-address positions, liquidation prices, real stop orders "
                  "and the full trade tape on perpetual DEXs. Centralised "
                  "venues publish none of this, so it has to be proxied — and "
                  "proxying it failed three times in a row on this system. "
                  "The premise here is to read it instead of estimating it."),
            stage=("Recording. No criteria frozen yet: thresholds are only "
                   "meaningful once the real noise level is known."),
            note=("There are many markets, but each hourly pass covers only "
                  "part of the open interest. Coverage is stored alongside the "
                  "data — otherwise an empty fuel map cannot be told apart "
                  "from one we simply did not see."),
        ),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    # 公開面自曝：不得出現金額。**只掃網站真正會渲染的欄位** ——
    # 第一版掃整份，結果抓到 `_readme` 裡「不出美元金額」那句規則本身
    # （守衛掃到自己的說明書）。掃錯範圍的守衛會製造假警報，而假警報會
    # 訓練人忽略它。
    blob = json.dumps({k: out[k] for k in ("zh", "en")}, ensure_ascii=False)
    import re
    bad = re.findall(r"\$|美元|USD|[0-9]+(?:\.[0-9]+)?\s*[BM]\b", blob)
    print("公開面檢查（不得出現金額）:", set(bad) or "clean")
    assert not bad, "內容含金額，違反公開面規則"
    print("venues=%d markets=%d snapshots=%d addresses=%d coverage=%s "
          "checks=%d/%d" % (n_venues, n_markets, snaps, out["addresses"],
                            ("部分掃描，不發布"
                             if out["coverage_pct"] is None
                             else "%.2f%%" % out["coverage_pct"]),
                            n_pass, n_checks))
    print("written -> " + str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
