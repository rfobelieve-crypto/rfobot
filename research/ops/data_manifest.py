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
    ("hl 清算價直方圖", "research/hl/data/snapshots/*.json", "append",
     "hl_fuel_recorder.py 每小時。**沒有歷史端點**，停了就永久缺那一小時"),
    ("hl 逐部位明細", "research/hl/data/positions/*.parquet", "append",
     "**真相源**。清算事件在公開端點沒有旗標，唯一判定是「部位在下一個快照"
     "消失且期間價格穿過它的清算價」—— 那需要逐部位明細。直方圖由它推導"),
    ("hl 市場狀態(OI/funding)", "research/hl/data/market/*.json", "append",
     "同上。OI 在 Hyperliquid 沒有歷史查詢，這是唯一來源"),
    ("hl L2 簿口", "research/hl/data/book/*.json", "append", "同上"),
    ("hl 掛單與觸發單", "research/hl/data/orders/*.json", "append",
     "同上。觸發單 = 真實止損，§1.05 只能用指標代理的那個量"),
    # 2026-09-11：HL 的歷史 K 線。**這是 HL 上唯一可以回填的歷史** ——
    # 部位/簿口/成交帶四樣都沒有歷史端點。但它也會從頭部掉資料：
    # candleSnapshot 只保留最近 **5000 根**（實測），所以 1h 只有 208 天、
    # 1m 只有 3.5 天。保留是以「根數」算的，週期越細歷史越短。
    # 2026-09-11：分鐘級中價與佇列（D 槽，不在 repo 裡）。
    # **這是唯一一份用 mid 而不是成交價的 HL 價格序列** —— 成交價有買賣價
    # 跳動，薄的標的會假性反轉。簿口沒有歷史端點，所以它跟 tape 一樣
    # 不可回填。`bid_n/ask_n` 是每檔掛單筆數，CEX 公開簿口沒有這一欄。
    ("hl 中價與佇列", "D:/flowbot_data/hl/mid/*/*.parquet", "append",
     "hl_mid.py 每 60 秒取樣主場量能前 40 名；不可回填"),
    ("§1.25 宇宙級分鐘簿口", "../arb/engine/logs/universe/*/minutes.csv", "append",
     "record_universe.py：126 個 ticker、150 個配對的逐分鐘頂檔。"
     "**不可回填**（WS 串流）。欄位 schema 與 §0.75 的 minutes.csv 逐欄相同，"
     "但 fund_* 三欄是空的（已知缺口，見 TODO §1.25）"),
    # 2026-09-11：這兩個現在是**指向 D 槽的目錄連結**（見下面的 junction 守衛）。
    ("market_data 原始資料", "market_data/raw_data/*.parquet", "append",
     "33.6 GB，2026-09-11 搬到 D:\flowbot_data\raw_data 並在原位留目錄連結；"
     "路徑對 77 個引用它的檔案完全沒變"),
    ("hl 歷史 K 線", "research/hl/data/candles/*.parquet", "append",
     "hl_candles.py 下載；端點保留上限 5000 根/週期，1h ~208 天。"
     "所以它既是可回填的、也是會從頭部腐蝕的 —— 兩者同時成立"),
    ("hl 地址宇宙", "research/hl/data/addresses.json", "append",
     "只增不減；覆蓋率隨它成長（實測 242 -> 633 個地址時覆蓋 7% -> 17%）"),
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


def _json_span(p: Path):
    """JSON 產物（鏈上錄製的每小時檔）：列數取 rows/resting/addresses 的長度，
    時點取頂層 ts。**不載入巨大檔案的全部欄位**——這些檔 <1MB，直接 load。"""
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return 0, None, None
    n = 0
    for k in ("rows", "resting", "triggers", "addresses"):
        v = d.get(k)
        if isinstance(v, list):
            n += len(v)
    ts = d.get("ts")
    if isinstance(ts, (int, float)):
        ms = int(ts) * 1000 if ts < 1e12 else int(ts)
        return n, ms, ms
    return n, None, None


def scan():
    out = {}
    for name, pat, kind, note in REGISTRY:
        # 2026-09-11：pathlib 的 glob **不吃絕對路徑、也不吃 `..`**
        # （會拋 NotImplementedError: Non-relative patterns are unsupported）。
        # 而註冊表現在同時有三種：repo 相對、`../arb/...`、`D:/flowbot_data/...`。
        # 這一行原本只處理第一種，加進後兩種之後**整支就在 scan() 炸掉**，
        # 而旗標停在上一次成功的綠 —— 又一個「看板說它活著」。
        # 改用 glob 模組（它三種都吃）。
        import glob as _g
        pat_s = str(pat)
        if Path(pat_s).is_absolute() or pat_s.startswith(".."):
            files = sorted(Path(x) for x in _g.glob(pat_s, recursive=True))
        else:
            files = sorted(ROOT.glob(pat_s))
        if not files:
            out[name] = dict(kind=kind, note=note, files=0, missing=True)
            continue
        agg = dict(kind=kind, note=note, files=len(files), missing=False,
                   rows=0, first_ts=None, last_ts=None, bytes=0, sha=None)
        h = hashlib.sha256()
        for p in files:
            # **讀不開要分兩種**：剛好在被寫（暫態）vs 真的壞了（要有人看）。
            # 用 mtime 分辨 —— 2026-09-11 DailyCollect 就是撞到成交帶的
            # 5 分鐘落盤，整條班車因此回 rc=1，而那個非零從此沒有分辨力。
            try:
                if p.suffix == ".csv":
                    n, a, z = _csv_span(p)
                elif p.suffix == ".json":
                    n, a, z = _json_span(p)
                else:
                    n, a, z = _pq_span(p)
            except Exception as e:
                try:
                    fresh = (time.time() - p.stat().st_mtime) < 120
                except Exception:
                    fresh = False
                bucket = "writing" if fresh else "corrupt"
                agg.setdefault(bucket, []).append(p.name)
                print("[%s] %s 讀不開（%s）: %s"
                      % ("SKIP" if fresh else "BAD", name, p.name,
                         str(e).split(chr(10))[0][:90]))
                continue
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


# ── junction 守衛（2026-09-11）────────────────────────────────────────────
# 兩個大資料夾搬到 D 槽、原位留目錄連結。**連結的失效方式是安靜的**：
# D 槽沒掛載時連結變成一個空目錄，於是每一支讀它的程式都讀到「零列」，
# 而零列在很多地方是合法狀態（mistake.md 2026-09-03：合法的空狀態與故障
# 長得一模一樣）。所以要有一條專門問「連結還通嗎」的檢查。
JUNCTIONS = [
    ("market_data/raw_data", "D:/flowbot_data/raw_data"),
    ("research/poc/data", "D:/flowbot_data/poc_data"),
]


def check_junctions(root: Path) -> list:
    """回傳問題清單；空 = 都通。**空目錄也算問題**，那正是失效的樣子。"""
    bad = []
    for rel, target in JUNCTIONS:
        p = root / rel
        if not p.exists():
            bad.append("%s 不存在（連結斷了？D 槽沒掛載？）" % rel)
            continue
        t = Path(target)
        if not t.exists():
            bad.append("%s 的目標 %s 不存在" % (rel, target))
            continue
        n = sum(1 for _ in p.rglob("*") if _.is_file())
        if n == 0:
            bad.append("%s 透過連結看到 **0 個檔** —— 連結在但目標是空的" % rel)
    return bad


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
        # 讀不開的檔：**corrupt 要變紅，writing 不算錯**。
        # writing 是常態（成交帶每 5 分鐘落盤，班車會撞上），
        # 把常態記成錯會訓練人忽略這個頻道。
        if a.get("corrupt"):
            bad.append("%s -> **檔案讀不開且不是正在寫**: %s"
                       % (name, ", ".join(a["corrupt"][:3])))
        if a.get("writing"):
            notes.append("%s -> %d 個檔正在被寫入，本輪跳過（預期行為）"
                         % (name, len(a["writing"])))
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

    # junction 守衛：**接在這裡才算存在**。2026-09-11 這一整個 session 的
    # 主旋律就是「守衛寫了但沒接線」（成交帶沒進 freshness、levels/events
    # 沒進更新器、看門狗讀不懂旗標）——所以寫完當場接上並反向證明過。
    jbad = check_junctions(ROOT)
    if jbad:
        bad.extend("junction: " + x for x in jbad)

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
