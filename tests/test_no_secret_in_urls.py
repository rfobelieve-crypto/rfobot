# -*- coding: utf-8 -*-
"""結構性守衛：長期祕密不准出現在網址字串裡（2026-09-05）

**為什麼**：Telegram 帳號被盜當天查曝險面，發現 `ADMIN_HEAL_TOKEN` 被嵌在送去
聊天室的按鈕網址裡（`/dashboard?token=...`）。**每一則舊訊息都是一把長期鑰匙**，
開得了 `/okx-admin/heal`、`/admin/flow-bars-export` 與全部 `/research/*`。

網址會留在瀏覽器歷史、referrer、代理 log、聊天記錄。這條測試把「不要再這樣做」
從記憶變成機器檢查——比照 `test_okx_client.py` 的 facade 守衛與
`test_public_payload_shape.py` 的欄位守衛。

**規則**：任何 f-string 或字串串接，若同時含有 `http` 或以 `/` 開頭的路徑、
且插入了名字像祕密的變數（TOKEN/SECRET/KEY/PASSWORD/PASSPHRASE），即失敗。
要放連結給人點，用 `shared.signed_link.make_query()`（短效、綁路徑）。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SECRETY = re.compile(r"(TOKEN|SECRET|KEY|PASSWORD|PASSPHRASE)", re.I)
URLISH = re.compile(r"(https?://|\?token=|&token=|/admin|/research/|/okx-admin)")
SKIP_DIRS = {"node_modules", ".claude", "tests", "promo_video", "__pycache__"}
# 具名豁免：必須寫理由，不可以只加名字
EXEMPT = {
    # signed_link 自己要組 query，但它組的是簽章不是長期祕密
    "shared/signed_link.py",
}


def _py_files():
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT).as_posix()
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if rel in EXEMPT:
            continue
        yield rel, p


# 廠商 API 的端點本身就含 token（Telegram Bot API 的設計），那是伺服器對外的
# 請求 URL、不是交給人點的連結，也無法改成簽章。列為已知並接受的風險，
# 記在 docs/SECURITY.md，不由這條測試管。
VENDOR_HOSTS = ("api.telegram.org",)


def _offending_lines(text: str):
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if not URLISH.search(line):
            continue
        if any(h in line for h in VENDOR_HOSTS):
            continue
        # 只看有插值的行：f-string 的 {VAR} 或 " + VAR"
        for m in re.finditer(r"\{([^{}]+)\}|\+\s*([A-Za-z_][\w.]*)", line):
            name = (m.group(1) or m.group(2) or "")
            if SECRETY.search(name):
                out.append((i, line.strip()[:110]))
                break
    return out


def test_no_long_lived_secret_in_url_strings():
    bad = []
    for rel, p in _py_files():
        try:
            txt = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for ln, src in _offending_lines(txt):
            bad.append(f"{rel}:{ln}  {src}")
    assert not bad, (
        "長期祕密被放進網址字串（用 shared.signed_link.make_query 取代）:\n  "
        + "\n  ".join(bad))


def test_guard_detects_a_planted_violation():
    """反向證明：故意造一行違規，守衛必須抓到。沒有這條，上面那條綠燈
    可能只是因為規則寫錯而永遠不會紅（mistake.md 2026-09-03）。"""
    planted = 'url = f"{base}/dashboard?token={ADMIN_HEAL_TOKEN}"'
    assert _offending_lines(planted), "守衛抓不到明顯的違規 —— 規則本身壞了"


def test_signed_link_roundtrip_and_expiry(monkeypatch):
    import time
    import sys
    sys.path.insert(0, str(ROOT))
    from shared import signed_link as sl

    monkeypatch.setenv("ADMIN_HEAL_TOKEN", "unit-test-key")
    q = sl.make_query("/dashboard", ttl=60)
    parts = dict(kv.split("=", 1) for kv in q.split("&"))
    assert sl.verify("/dashboard", parts["exp"], parts["sig"])
    # 綁路徑：同一個簽章不能用在別的路徑
    assert not sl.verify("/okx-admin/heal", parts["exp"], parts["sig"])
    # 會過期
    old = int(time.time()) - 3600
    assert not sl.verify("/dashboard", old, sl.sign("/dashboard", old))
    # 沒有金鑰時不發簽章（行為與原本「沒 token 就不附加」一致）
    monkeypatch.setenv("ADMIN_HEAL_TOKEN", "")
    assert sl.make_query("/dashboard") == ""
