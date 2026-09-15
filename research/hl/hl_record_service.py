# -*- coding: utf-8 -*-
"""HL 燃料錄製器的 Railway 服務：每小時跑一次 hl_fuel_recorder.py ＋ 唯讀檔案端點。

    python research/hl/hl_record_service.py          # Railway 的 CMD

===========================================================================
為什麼搬上雲（2026-09-15，使用者選 A）
===========================================================================
`FlowBot_HLRecord` 在本機每小時 :05 啟動、跑約 28 分鐘、約 400 req/分，而
HMM 引擎的 HL 腿從**同一台機器、同一個 IP** 出去，HL 的限流是按 IP 算的：

    MON runner.log 的 HL 側 429   274 行
    落在每小時 05–33 分之間        274 行（均勻的話該是 48%）
    15:13:58 撤單拿到 429 -> 撤單未確認 -> 21 分鐘後成交 -> 過度對沖 -> HALT

錯開時段沒用（HMM 24 小時在跑），降速或砍地址會改變 §1.10 的資料定義，
而這份資料**不可回填**。所以把請求移到別的 IP。這是 arb 的掃描器
2026-09-13 搬上 Railway 的同一個理由（arb docs/DEPLOY.md §6）。

===========================================================================
設計（照 arb 的 scan_service / scan_pull 那一對）
===========================================================================
**只蒐集，不判斷。** 驗證（hl_verify.py）要讀本機的 WS 成交帶，留在本機；
下游研究一律讀本機拉回來的檔。

**本機發起拉取，出站 only。** 端點全部唯讀，除了一個 `/seed`：
地址宇宙（`addresses.json`，只增不減）是這條錄製線的連續性，而 rfobot 是
public repo，所以它不進 git、不進映像檔，改由本機上傳一次。`/seed` 只在
volume 上**還沒有** addresses.json 時接受，之後永遠回 409 ——
它不能被拿來覆寫一個正在累積的宇宙。

**token**：`HL_RECORD_TOKEN`（Bearer）。外洩的後果是別人讀得到公開鏈上資料的
整理版，或在 volume 還空的那幾分鐘塞一份假地址表，不是任何交易能力。
`/health` 不帶 token（Railway 健康檢查用），只回存活與上一輪的狀態碼。

**排程是牆鐘的 :05**，跟本機舊排程同一個分鐘，所以每小時檔名
（`YYYYMMDD_HH`，UTC）的語意不變。一輪沒跑完不會疊下一輪。

**保留 KEEP_DAYS 天**：本機才是長期真相源，volume 只是緩衝。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
ROOT_DIR = Path(os.environ.get("HL_ROOT", "/data"))
DATA = ROOT_DIR / "hl"                         # 對應本機 research/hl/data
RESULTS = ROOT_DIR / "results"                 # 對應本機 research/results
STATUS = RESULTS / "hl_record_service.json"
TOKEN = os.environ.get("HL_RECORD_TOKEN", "")
MAX_ADDR = int(os.environ.get("HL_MAX_ADDR", "2500"))   # 本機 hl_record.bat 的值
RUN_MINUTE = int(os.environ.get("HL_RUN_MINUTE", "5"))
KEEP_DAYS = float(os.environ.get("HL_KEEP_DAYS", "30"))
RUN_TIMEOUT = 55 * 60                          # 一輪超過 55 分鐘就收掉，不疊下一輪
HOURLY_DIRS = ("snapshots", "positions", "market", "book", "orders")

_lock = threading.Lock()
_state = {"started": time.time(), "runs": 0, "last": None, "running": False}


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def _write_status() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(_state, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, STATUS)


def run_once() -> None:
    """跑一輪錄製器。stdout 進 Railway log；結果記進狀態檔。"""
    with _lock:
        if _state["running"]:
            log("上一輪還在跑，這一輪跳過")
            return
        _state["running"] = True
    t0 = time.time()
    env = dict(os.environ, HL_DATA_DIR=str(DATA),
               HL_FLAG_PATH=str(RESULTS / "hl_fuel_last.json"),
               PYTHONIOENCODING="utf-8")
    rc, why = None, ""
    if not (DATA / "addresses.json").exists():
        # 沒有種子就跑 = 從零開始累積地址宇宙，覆蓋率會從 22% 掉到個位數，
        # 而那一段資料看起來完全正常。所以寧可不跑、把原因寫出來。
        why = "volume 上沒有 addresses.json —— 等本機 hl_record_pull.py --seed"
        log("跳過：" + why)
    else:
        try:
            p = subprocess.run(
                [sys.executable, str(HERE / "hl_fuel_recorder.py"),
                 "--max-addr", str(MAX_ADDR)],
                env=env, timeout=RUN_TIMEOUT, capture_output=True, text=True,
                encoding="utf-8", errors="replace")
            rc = p.returncode
            tail = (p.stdout or "")[-3000:] + (p.stderr or "")[-2000:]
            for ln in tail.splitlines():
                log("  | " + ln)
        except subprocess.TimeoutExpired:
            why = "超過 %d 分鐘被收掉" % (RUN_TIMEOUT // 60)
            log(why)
        except Exception as e:                           # noqa: BLE001
            why = "%s: %s" % (type(e).__name__, e)
            log("錄製器啟動失敗：" + why)
    _prune()
    with _lock:
        _state["running"] = False
        _state["runs"] += 1
        _state["last"] = {"start": t0, "end": time.time(), "rc": rc, "why": why,
                          "asof": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        _write_status()


def _prune() -> None:
    cut = time.time() - KEEP_DAYS * 86400
    n = 0
    for d in HOURLY_DIRS:
        for f in (DATA / d).glob("*"):
            try:
                if f.is_file() and f.stat().st_mtime < cut:
                    f.unlink()
                    n += 1
            except OSError:
                pass
    if n:
        log("清掉 %d 個超過 %.0f 天的檔（本機已經拉回去了）" % (n, KEEP_DAYS))


def scheduler() -> None:
    last_hour = None
    while True:
        now = time.gmtime()
        key = (now.tm_yday, now.tm_hour)
        if now.tm_min >= RUN_MINUTE and key != last_hour:
            last_hour = key
            threading.Thread(target=run_once, daemon=True).start()
        time.sleep(20)


def _files() -> list:
    out = []
    for base, prefix in ((DATA, "hl/"), (RESULTS, "results/")):
        if not base.exists():
            continue
        for f in base.rglob("*"):
            if f.is_file() and not f.name.endswith(".tmp"):
                st = f.stat()
                out.append({"path": prefix + f.relative_to(base).as_posix(),
                            "size": st.st_size, "mtime": st.st_mtime})
    return out


def _resolve(rel: str):
    """只允許 hl/ 與 results/ 底下的檔。任何跳出 ROOT_DIR 的路徑一律拒絕。"""
    if not (rel.startswith("hl/") or rel.startswith("results/")):
        return None
    p = (ROOT_DIR / rel).resolve()
    try:
        p.relative_to(ROOT_DIR.resolve())
    except ValueError:
        return None
    return p if p.is_file() else None


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):                           # 不要每個請求一行
        pass

    def _send(self, code: int, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self) -> bool:
        # 沒設 token = 全部拒絕（fail-closed），不是全部放行。
        return bool(TOKEN) and self.headers.get("Authorization") == "Bearer " + TOKEN

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/health":
            with _lock:
                last = _state["last"]
            return self._send(200, {"ok": True, "runs": _state["runs"],
                                    "last_rc": last and last["rc"],
                                    "seeded": (DATA / "addresses.json").exists()})
        if not self._authed():
            return self._send(401, {"error": "unauthorized"})
        if u.path == "/files":
            return self._send(200, {"files": _files(), "status": _state})
        if u.path == "/file":
            rel = (parse_qs(u.query).get("path") or [""])[0]
            p = _resolve(rel)
            if p is None:
                return self._send(404, {"error": "not found"})
            return self._send(200, p.read_bytes(), "application/octet-stream")
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        if not self._authed():
            return self._send(401, {"error": "unauthorized"})
        if u.path == "/seed":
            target = DATA / "addresses.json"
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 50 * 1024 * 1024:
                return self._send(400, {"error": "bad size"})
            # 先把 body 讀完再回 —— 沒讀完就回 409，客戶端看到的是連線被重置
            # 而不是 409（本機測試實測：curl 拿到 000）。
            raw = self.rfile.read(n)
            if target.exists():
                return self._send(409, {"error": "already seeded; the universe only grows here"})
            try:
                d = json.loads(raw.decode("utf-8"))
                assert isinstance(d.get("addresses"), list) and d["addresses"]
            except Exception:                            # noqa: BLE001
                return self._send(400, {"error": "not an addresses.json"})
            DATA.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(raw)
            os.replace(tmp, target)
            log("種子已寫入：%d 個地址" % len(d["addresses"]))
            return self._send(200, {"ok": True, "addresses": len(d["addresses"])})
        if u.path == "/run":
            # 手動觸發一輪（部署後驗收用）。跟排程共用同一個「不疊」的鎖。
            threading.Thread(target=run_once, daemon=True).start()
            return self._send(202, {"ok": True})
        return self._send(404, {"error": "not found"})


def main() -> int:
    for d in (DATA, RESULTS):
        d.mkdir(parents=True, exist_ok=True)
    if not TOKEN:
        log("**HL_RECORD_TOKEN 沒設** —— 檔案端點全部回 401，本機拉不到任何東西")
    threading.Thread(target=scheduler, daemon=True).start()
    port = int(os.environ.get("PORT", "8080"))
    log("hl record service :%d  root=%s  max_addr=%d  run at :%02d UTC  keep %.0fd"
        % (port, ROOT_DIR, MAX_ADDR, RUN_MINUTE, KEEP_DAYS))
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
