# -*- coding: utf-8 -*-
"""研究資料的血統與「頭部有沒有被吃掉」（2026-09-10）

**這條守衛擋的是 2026-09-10 當天現形的那個病**：
`research/sweep_failure/.cache/*.csv` 是 `fetch_klines.py` 抓的，起點寫成
`now - days*86400` —— 那是一個**滾動 930 天窗**，每次刷新都從頭部丟掉舊 bar。
沒有任何檔案寫著這件事，也沒有下游知道。後果：

    · 釘在它上面的 sha 基準（parity 測試）在下次刷新就必紅，
      而且紅的原因跟被保護的東西完全無關
    · 同一份自曝檢查（「九幣應為 7,083 筆」）會無故失敗
    · 兩次跑同一條研究線會得到**兩個不同的母體**，而沒有東西會說

同族還有 2026-08-29（輪替 CSV，下游計數器從零重數）與 2026-08-01
（同一份資料兩份拷貝，只有一份有人更新）。共同形狀是
**資料的形狀變了，而消費者不知道**。

修法照 mistake.md 2026-09-01 的方向：把它翻譯成**某個數字不對**。

登記簿裡每一份資料要宣告它的 `kind`：

    rolling      起點錨在 now，頭部**必然**會前移。任何釘在絕對筆數或
                 整份 sha 上的東西都不可以指向它
    append       只在尾端長。**頭部前移 = 紅**（有人把它截斷了）
    frozen       完全不動。**任何變動 = 紅**（sha 比對）
    derived      由上游重新產生，內容會整份換掉；只記錄不判紅

輸出 results/data_manifest.json（含上一輪的比對結果）給 freshness 的
json_flag 讀。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "results" / "data_manifest.json"

# (名稱, glob, kind, 誰寫它 / 為什麼是這個 kind)
REGISTRY = [
    ("sweep_failure/.cache 1h bars", "research/sweep_failure/.cache/*USDT_1h.csv",
     "rolling",
     "fetch_klines.py --days 930，起點 = now - days*86400。**頭部必然前移**；"
     "任何 sha / 絕對筆數的基準都不可以指向這裡（2026-09-10 已為此把 parity "
     "測試的基準搬到 crowd_stops/frozen）"),
    ("crowd_stops/frozen 1h 切片", "research/crowd_stops/frozen/*USDT_1h.csv",
     "frozen",
     "stop_map 的母體，2026-09-10 從 .cache 複製後凍結。任何變動 = 紅"),
    ("tests/fixtures 三本帳", "research/tests/fixtures/book_*.parquet",
     "frozen",
     "研究回歸測試的輸入（V7 在 MySQL、OLD 來自滾動窗，都不可重現）。"
     "任何變動 = 紅，除非同時重跑 make_fixtures.py 並更新判決值"),
    ("poc/data 分鐘 bar", "research/poc/data/bars/*.parquet", "append",
     "fetch_bars.py。只在尾端長；頭部前移 = 有人截斷了它"),
    ("poc/data 未平倉量", "research/poc/data/oi/*.parquet", "append",
     "fetch_oi.py，5 分鐘級"),
    ("poc/data 清算", "research/poc/data/liq/*.parquet", "append",
     "小時級，2026-03-11 起"),
    ("poc/data 掃單快照", "research/poc/data/sweep_snapshot.parquet", "derived",
     "sweep_snapshot.py 整份重生；筆數變動是正常的，只記錄"),
]
TS_COLS = ("ts", "time", "create_time", "entry_ms")


def _csv_span(p: Path):
    first = last = None
    n = 0
    with open(p, newline="", encoding="utf-8-sig") as f:
        next(f, None)
        for line in f:
            if not line.strip():
                continue
            n += 1
            v = line.split(",", 1)[0]
            if first is None:
                first = v
            last = v
    def num(x):
        try:
            return int(float(x))
        except (TypeError, ValueError):
            return None
    return n, num(first), num(last)


def _pq_span(p: Path):
    import pyarrow.parquet as pq
    f = pq.ParquetFile(p)
    n = f.metadata.num_rows
    col = next((c for c in TS_COLS if c in f.schema_arrow.names), None)
    if col is None:
        return n, None, None
    import pandas as pd
    s = pq.read_table(p, columns=[col]).column(col).to_pandas()
    if hasattr(s.dtype, "tz") or str(s.dtype).startswith("datetime"):
        s = pd.to_datetime(s).astype("int64") // 10 ** 6
    return n, int(s.min()), int(s.max())


def scan():
    out = {}
    for name, pat, kind, note in REGISTRY:
        files = sorted(ROOT.glob(pat))
        if not files:
            out[name] = dict(kind=kind, note=note, files=0, missing=True)
            continue
        agg = dict(kind=kind, note=note, files=len(files), missing=False,
                   rows=0, first_ts=None, last_ts=None, bytes=0, sha=None)
        h = hashlib.sha256()
        for p in files:
            n, a, z = (_csv_span(p) if p.suffix == ".csv" else _pq_span(p))
            agg["rows"] += n
            agg["bytes"] += p.stat().st_size
            if a is not None:
                agg["first_ts"] = a if agg["first_ts"] is None else min(agg["first_ts"], a)
                agg["last_ts"] = z if agg["last_ts"] is None else max(agg["last_ts"], z)
            if kind == "frozen":
                h.update(p.read_bytes())
        if kind == "frozen":
            agg["sha"] = h.hexdigest()[:32]
        out[name] = agg
    return out


def main():
    prev = {}
    if OUT.exists():
        try:
            prev = json.loads(OUT.read_text(encoding="utf-8")).get("data", {})
        except json.JSONDecodeError:
            prev = {}
    cur = scan()

    bad, notes = [], []
    for name, a in cur.items():
        b = prev.get(name)
        if a.get("missing"):
            bad.append("%s -> 檔案不見了" % name)
            continue
        if not b or b.get("missing"):
            notes.append("%s -> 首次登記" % name)
            continue
        if a["kind"] == "frozen" and a["sha"] != b.get("sha"):
            bad.append("%s -> **凍結資料變了** sha %s != %s"
                       % (name, a["sha"][:8], str(b.get("sha"))[:8]))
        if a["kind"] == "append" and a["first_ts"] and b.get("first_ts") \
                and a["first_ts"] > b["first_ts"]:
            bad.append("%s -> **頭部被吃掉** first_ts %s -> %s"
                       % (name, b["first_ts"], a["first_ts"]))
        if a["kind"] == "rolling" and a["first_ts"] and b.get("first_ts") \
                and a["first_ts"] > b["first_ts"]:
            notes.append("%s -> 頭部前移（滾動窗，預期行為）" % name)

    ok = not bad
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        ok=ok, reason=("; ".join(bad)[:400] if bad else
                       "%d 份資料，%d 則備註" % (len(cur), len(notes))),
        notes=notes, asof=time.strftime("%Y-%m-%d %H:%M:%S"), data=cur),
        indent=2, ensure_ascii=False), encoding="utf-8")

    print("%-26s %-8s %9s %14s %14s" % ("資料", "kind", "列數", "first", "last"))
    for name, a in cur.items():
        if a.get("missing"):
            print("%-26s %-8s  **檔案不見了**" % (name[:25], a["kind"]))
            continue
        print("%-26s %-8s %9s %14s %14s"
              % (name[:25], a["kind"], format(a["rows"], ","),
                 a["first_ts"] or "-", a["last_ts"] or "-"))
    for n in notes:
        print("  note: " + n)
    print()
    print("manifest: %s  %s" % ("OK" if ok else "RED",
                                "; ".join(bad) if bad else "無變動"))
    print("written -> " + str(OUT))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
