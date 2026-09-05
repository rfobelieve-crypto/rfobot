# -*- coding: utf-8 -*-
"""短效簽章連結——讓「可分享的網址」不再等於「長期祕密」（2026-09-05）

**為什麼有這個檔**：2026-09-05 操作者的 Telegram 帳號被盜。查曝險面時發現
`ADMIN_HEAL_TOKEN` 被直接嵌在送去 Telegram 的按鈕網址裡
（`/dashboard?token=...`、`/research/shadow-review?...&token=...`）——
**任何一則舊訊息都等於一把長期鑰匙**，而那把鑰匙開得了 `/okx-admin/heal`、
`/admin/flow-bars-export`、以及全部 `/research/*` 的策略輸出。

網址會出現在瀏覽器歷史、referrer、反向代理 log、聊天記錄裡。**長期祕密不該
放在網址裡**，但操作者又需要「點一下就能開」的連結。折衷是簽章：

    路徑 + 到期時間 → HMAC-SHA256（金鑰＝ADMIN_HEAL_TOKEN）→ ?exp=…&sig=…

- 簽章**綁定路徑**：`/dashboard` 的連結開不了 `/okx-admin/heal`
- 簽章**會過期**（預設 24 小時）：三個月前的聊天記錄裡那條連結是死的
- 原本的 `X-Admin-Token` 標頭與 `?token=` 照舊可用（服務之間的呼叫、CLI），
  **這個模組是新增一條路，不是取代**——避免把運維工具一起弄壞

**它擋不了什麼（要誠實）**：如果 `ADMIN_HEAL_TOKEN` 本身外洩，簽章一樣可以
自己生。它降低的是「連結外洩」的傷害，不是「金鑰外洩」的傷害。金鑰外洩的
對策是輪替，見 `docs/SECURITY.md`。
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time

DEFAULT_TTL = 24 * 3600
_SKEW = 60          # 容許的時鐘偏差（秒）


def _key() -> bytes:
    return os.environ.get("ADMIN_HEAL_TOKEN", "").encode("utf-8", "replace")


def sign(path: str, exp: int) -> str:
    msg = f"{path}|{exp}".encode("utf-8", "replace")
    return hmac.new(_key(), msg, hashlib.sha256).hexdigest()[:32]


def make_query(path: str, ttl: int = DEFAULT_TTL) -> str:
    """回傳 `exp=…&sig=…`（不含問號）。沒有設金鑰時回空字串——呼叫端據此
    決定要不要附加，行為與原本「沒有 token 就不附加」一致。"""
    if not _key():
        return ""
    exp = int(time.time()) + int(ttl)
    return f"exp={exp}&sig={sign(path, exp)}"


def verify(path: str, exp: str | int, sig: str) -> bool:
    """路徑必須完全相同、未過期、簽章相符。任何一項不符一律 False。"""
    if not _key() or not sig:
        return False
    try:
        exp_i = int(exp)
    except (TypeError, ValueError):
        return False
    if exp_i + _SKEW < int(time.time()):
        return False
    return hmac.compare_digest(sig.strip(), sign(path, exp_i))
