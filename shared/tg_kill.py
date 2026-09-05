# -*- coding: utf-8 -*-
"""Telegram 全面斷流閘門（2026-09-05，帳號遭盜用）

**為什麼存在**：操作者的 Telegram 帳號在 2026-09-05 被盜且已無法登入。系統的
所有告警（訊號、進出場、部位張數、名目美元、權益變化）都是推進那個帳號的聊天室
——**每一次告警都是在餵資料給控制那個帳號的人**。所以送出端必須先停，不是等
token 換完再說。

**設計是 fail-closed 的**：預設封鎖，要恢復必須把環境變數設成一個**特定字串**
（不是 1、不是 true），這樣任何「順手把某個旗標打開」都不會意外復通。

    TELEGRAM_REENABLE=I_ROTATED_ALL_TOKENS

恢復的前提（缺一不可，這是給未來的自己看的檢查表）：
  1. Telegram 帳號已取回，且已終止所有其他工作階段、開啟兩步驟驗證
  2. 四支 bot 的 token 全部在 BotFather 重新產生（TELEGRAM_BOT_TOKEN、
     CANCEL_TG_BOT_TOKEN、INDICATOR_BOT_TOKEN、AGENT_BOT_TOKEN）
  3. Railway 各服務的變數已更新，且已重設 webhook
  4. webhook 已改用 secret_token（見 TODO）——現在的設計把 token 同時當認證
     與路徑，一個外洩就兩個都破

進來的方向（webhook）另外由各自的路由擋，這個模組只管**送出去**。
"""
from __future__ import annotations

import os
import threading

_REENABLE_PHRASE = "I_ROTATED_ALL_TOKENS"
_warned = set()
_lock = threading.Lock()


def tg_blocked() -> bool:
    """True = 不准送。預設 True。"""
    return os.getenv("TELEGRAM_REENABLE", "") != _REENABLE_PHRASE


def note_blocked(where: str) -> None:
    """每個呼叫點只印一次，避免洗版；但一定要留痕——靜默的封鎖跟靜默的故障
    在畫面上長得一樣（mistake.md 2026-08-01）。"""
    with _lock:
        if where in _warned:
            return
        _warned.add(where)
    print(f"[TG-CUTOVER] blocked outbound telegram from {where} "
          f"(帳號遭盜用 2026-09-05；設 TELEGRAM_REENABLE={_REENABLE_PHRASE} 才恢復)",
          flush=True)


def guard(where: str) -> bool:
    """呼叫端一行搞定：`if guard("send_message"): return`"""
    if tg_blocked():
        note_blocked(where)
        return True
    return False
