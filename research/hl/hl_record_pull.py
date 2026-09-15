# -*- coding: utf-8 -*-
"""把 Railway 上 hl-record 服務錄的檔拉回本機（本機發起，出站 only）。

    python research/hl/hl_record_pull.py            # 拉一次（排程 FlowBot_HLRecord 跑這個）
    python research/hl/hl_record_pull.py --status   # 只看遠端狀態
    python research/hl/hl_record_pull.py --seed     # 一次性：把本機地址宇宙種到 volume

設定：HL_RECORD_URL、HL_RECORD_TOKEN（env 優先，flow_system/.env 回退）。

===========================================================================
為什麼有這支
===========================================================================
錄製器 2026-09-15 搬上 Railway（理由在 hl_record_service.py 檔頭：它跟 HMM 的
HL 腿搶本機 IP 的限流額度）。**下游一行都不改**：`prereg_fuel_mechanism`、
`onchain_publish`、`hl_verify`、`data_manifest` 全部照舊讀
`research/hl/data/`，這支負責讓那個目錄跟搬家前長得一樣。

===========================================================================
三個決定
===========================================================================
**一、保留遠端的 mtime。** 新鮮度看板的 `hl_fuel_last.json` 那一列用檔案
mtime 判斷年齡。如果拉取時把 mtime 改成「現在」，雲端錄製器死掉而拉取照跑
的時候，那一列會永遠是綠的 —— 一個量不到「錄製器死了」的守衛
（mistake.md 2026-09-03）。所以寫完用 `os.utime` 蓋回遠端的時間，而且
**只有遠端的檔變了才重抓**。

**二、本機永不刪。** 雲端只留 30 天，本機是長期真相源；這支只新增與覆寫。

**三、拉取斷掉必須看得見。** 自己的旗標 `hl_record_pull_last.json` 每次成功
才更新 mtime，失敗寫 ok=False。拉取斷了而雲端活著，本機資料會停在昨天，
而下游一個都不會報錯（arb scan_pull.py 檔頭同一條）。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
RESULTS = ROOT / "research" / "results"
STATE = DATA / ".pull_state.json"
FLAG = RESULTS / "hl_record_pull_last.json"
# results/ 底下只拉這兩個，其他同名檔一律不碰 —— 本機 results/ 還有幾十個
# 別的旗標，遠端不可以有機會覆寫它們。
RESULT_WHITELIST = {"hl_fuel_last.json", "hl_record_service.json"}
TIMEOUT = 120
# **切換點（UTC 小時，檔名格式）。** 早於它的每小時檔是本機錄製器寫的，永不覆寫。
# 理由：雲端上線當天的驗收試跑寫的是 20260915_09，而本機 17:05（09 UTC）那一輪
# 已經寫了同名檔；本機 18:05（10 UTC）那一輪也照跑。拉回來會把一個小時的真實
# 快照換成另一個時點的，而那兩份檔名一模一樣 —— 事後分不出來。
CUTOVER_HOUR = "20260915_11"
HOURLY_DIRS = ("snapshots", "positions", "market", "book", "orders")


def _dotenv() -> dict:
    out = {}
    p = ROOT / ".env"
    if p.exists():
        for ln in io.open(p, encoding="utf-8", errors="replace"):
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, v = ln.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def cfg(key: str) -> str:
    return os.environ.get(key) or _dotenv().get(key, "")


def _req(path: str, data: bytes | None = None, method: str = "GET"):
    url = cfg("HL_RECORD_URL").rstrip("/") + path
    r = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": "Bearer " + cfg("HL_RECORD_TOKEN"),
        "Content-Type": "application/json", "User-Agent": "flowbot-hl-pull/1.0"})
    with urllib.request.urlopen(r, timeout=TIMEOUT) as resp:
        return resp.read()


def local_path(remote: str):
    if remote.startswith("hl/"):
        return DATA / remote[3:]
    if remote.startswith("results/"):
        name = remote[len("results/"):]
        return RESULTS / name if name in RESULT_WHITELIST else None
    return None


def _merge_addresses(local_file: Path, remote_body: bytes) -> bytes:
    remote = json.loads(remote_body.decode("utf-8"))
    local = json.loads(local_file.read_text(encoding="utf-8"))
    union = sorted(set(remote.get("addresses", [])) | set(local.get("addresses", [])))
    remote["addresses"] = union
    remote["n"] = len(union)
    return json.dumps(remote, ensure_ascii=False).encode("utf-8")


def _flag(ok: bool, reason: str, **kw) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FLAG.write_text(json.dumps(dict(ok=ok, reason=reason,
                                    asof=time.strftime("%Y-%m-%d %H:%M:%S"), **kw),
                               ensure_ascii=False, indent=2), encoding="utf-8")


def pull() -> int:
    if not (cfg("HL_RECORD_URL") and cfg("HL_RECORD_TOKEN")):
        _flag(False, "HL_RECORD_URL / HL_RECORD_TOKEN 沒設定")
        print("**HL_RECORD_URL / HL_RECORD_TOKEN 沒設定**")
        return 2
    try:
        listing = json.loads(_req("/files").decode("utf-8"))
    except Exception as e:                                   # noqa: BLE001
        _flag(False, "列不到遠端檔案：%s" % type(e).__name__)
        print("列不到遠端檔案：%s" % e)
        return 1
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        state = {}

    got, nbytes, errs, skipped_pre = 0, 0, [], 0
    for f in listing.get("files", []):
        dst = local_path(f["path"])
        if dst is None:
            continue
        parts = f["path"].split("/")
        if (len(parts) == 3 and parts[1] in HOURLY_DIRS
                and dst.stem[:11] < CUTOVER_HOUR):
            skipped_pre += 1
            continue
        sig = [f["size"], f["mtime"]]
        if state.get(f["path"]) == sig and dst.exists():
            continue
        try:
            body = _req("/file?path=" + urllib.parse.quote(f["path"]))
            if len(body) != f["size"]:
                raise IOError("size %d != listed %d" % (len(body), f["size"]))
            if f["path"] == "hl/addresses.json" and dst.exists():
                # **地址宇宙取聯集，不覆寫。** 它的定義是只增不減；切換那一小時
                # 本機錄製器還在跑、還在擴充本機這份，直接覆寫會丟掉那些地址。
                body = _merge_addresses(dst, body)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            tmp.write_bytes(body)
            os.replace(tmp, dst)
            os.utime(dst, (f["mtime"], f["mtime"]))          # 決定一
            state[f["path"]] = sig
            got += 1
            nbytes += len(body)
        except Exception as e:                               # noqa: BLE001
            errs.append("%s: %s" % (f["path"], e))
    DATA.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=0), encoding="utf-8")

    last = (listing.get("status") or {}).get("last") or {}
    ok = not errs
    reason = ("拉回 %d 個檔 %.1f MB；遠端上一輪 rc=%s %s"
              % (got, nbytes / 1e6, last.get("rc"), last.get("asof", "")))
    if errs:
        reason = "**%d 個檔拉取失敗**：%s" % (len(errs), errs[0][:120])
    _flag(ok, reason, pulled=got, bytes=nbytes, errors=errs[:10],
          skipped_pre_cutover=skipped_pre,
          remote_last=last, remote_files=len(listing.get("files", [])))
    print(reason)
    return 0 if ok else 1


def status() -> int:
    d = json.loads(_req("/files").decode("utf-8"))
    print(json.dumps(d.get("status"), ensure_ascii=False, indent=1))
    print("遠端檔案 %d 個" % len(d.get("files", [])))
    return 0


def seed() -> int:
    p = DATA / "addresses.json"
    raw = p.read_bytes()
    n = len(json.loads(raw.decode("utf-8"))["addresses"])
    try:
        print(_req("/seed", data=raw, method="POST").decode("utf-8"))
        print("已上傳 %d 個地址" % n)
        return 0
    except urllib.error.HTTPError as e:
        print("HTTP %d %s" % (e.code, e.read().decode("utf-8", "replace")[:200]))
        return 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--seed", action="store_true")
    a = ap.parse_args()
    sys.exit(seed() if a.seed else status() if a.status else pull())
