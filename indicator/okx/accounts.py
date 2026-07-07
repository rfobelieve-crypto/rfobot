"""Multi-account registry for V7 OKX follow-trading (Phase 0).

Friend accounts are registered by the operator via Telegram admin
commands (handled here, dispatched from BTC_perp_data.py webhook):

    /okx_addacct <label> <api_key> <api_secret> <passphrase> <capital_usd>
    /okx_accounts
    /okx_delacct <label>
    /okx_pauseacct <label>  /okx_resumeacct <label>

Security invariants (non-negotiable):
  1. API key MUST NOT have withdraw permission — registration rejects it.
  2. Credentials are Fernet-encrypted at rest (OKX_CRED_MASTER_KEY env).
  3. The Telegram message containing raw credentials is deleted
     immediately after parsing (best effort).
  4. Only TG_ADMIN_CHAT_ID (fallback TG_CRITICAL_CHAT_ID) may run these.
  5. Risk caps may be tighter than main's Stage-3 caps, never looser.

Phase 1 (multi-account executor stacks) reads ACTIVE rows from here.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests as _requests

from shared.db import get_db_conn

logger = logging.getLogger(__name__)

# Stage-3 hard ceilings — per-account caps may be tighter, never looser.
MAX_CAPITAL_USD = 1000.0
MAX_LEV_MULT = 2.0
DAILY_CAP_FLOOR = -20.0   # e.g. -25 is looser → rejected
TOTAL_CAP_FLOOR = -30.0


# ── Encryption ─────────────────────────────────────────────────────────

def _get_fernet():
    """Build Fernet from OKX_CRED_MASTER_KEY. Raises if unset/invalid."""
    from cryptography.fernet import Fernet
    key = os.environ.get("OKX_CRED_MASTER_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OKX_CRED_MASTER_KEY not set — refusing to store credentials. "
            "Generate one with: python -c \"from cryptography.fernet import "
            "Fernet; print(Fernet.generate_key().decode())\""
        )
    return Fernet(key.encode())


def encrypt(plaintext: str) -> bytes:
    return _get_fernet().encrypt(plaintext.encode())


def decrypt(ciphertext: bytes) -> str:
    return _get_fernet().decrypt(bytes(ciphertext)).decode()


# ── Validation against OKX ─────────────────────────────────────────────

@dataclass
class ValidationResult:
    ok: bool
    reason: str = ""
    perm: str = ""
    equity_usd: Optional[float] = None


def validate_credentials(api_key: str, api_secret: str,
                         passphrase: str) -> ValidationResult:
    """Read-only probe: balance + account config + permission check.

    Rejects keys carrying withdraw permission (Safety belt #5 applied at
    the registration gate instead of executor startup).
    """
    from indicator.okx.config import OkxConfig
    from indicator.okx.rest import OkxRestClient

    cfg = OkxConfig(
        is_simulated=0 if os.environ.get("STAGE", "") == "live" else 1,
        stage_label="acct_validation",
        api_key=api_key, api_secret=api_secret, passphrase=passphrase,
    )
    try:
        rest = OkxRestClient(cfg)
        acct_cfg = rest.get_account_config()
        rows = (acct_cfg or {}).get("data") or []
        if not rows:
            return ValidationResult(False, "OKX 拒絕連線（key/secret/passphrase 錯誤，或 IP 白名單不符）")
        perm = str(rows[0].get("perm", ""))
        if "withdraw" in perm.lower():
            return ValidationResult(
                False,
                f"API key 帶有提幣權限（perm={perm}）— 拒絕註冊。"
                "請重建一組只勾『讀取+交易』的 key。",
                perm=perm,
            )
        if "trade" not in perm.lower():
            return ValidationResult(
                False, f"API key 沒有交易權限（perm={perm}）— 無法跟單。",
                perm=perm)
        bal = rest.get_balance()
        equity = float(bal.total_eq_usd) if bal else None
        return ValidationResult(True, perm=perm, equity_usd=equity)
    except Exception as e:
        logger.exception("acct_validation_error")
        return ValidationResult(False, f"驗證過程異常: {type(e).__name__}")


# ── CRUD ───────────────────────────────────────────────────────────────

def add_account(label: str, api_key: str, api_secret: str, passphrase: str,
                capital_usd: float, owner_chat_id: str = "") -> str:
    """Validate + encrypt + insert. Returns human-readable outcome."""
    label = label.strip().lower()
    if not label.isidentifier() or len(label) > 32:
        return "❌ label 只能是英數/底線，32 字內"
    if label == "main":
        return "❌ 'main' 保留給主帳戶（env vars）"
    if not (0 < capital_usd <= MAX_CAPITAL_USD):
        return f"❌ capital 必須在 (0, {MAX_CAPITAL_USD:.0f}] USD"

    v = validate_credentials(api_key, api_secret, passphrase)
    if not v.ok:
        return f"❌ 驗證失敗: {v.reason}"

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO okx_accounts
                    (label, owner_chat_id, api_key_enc, api_secret_enc,
                     passphrase_enc, initial_capital_usd, status,
                     validated_at, perm_snapshot, equity_at_reg)
                VALUES (%s,%s,%s,%s,%s,%s,'PENDING',%s,%s,%s)
                """,
                (label, owner_chat_id or None,
                 encrypt(api_key), encrypt(api_secret), encrypt(passphrase),
                 capital_usd, datetime.utcnow(), v.perm, v.equity_usd),
            )
        conn.commit()
    except Exception as e:
        if "Duplicate" in str(e):
            return f"❌ label '{label}' 已存在"
        logger.exception("add_account_insert_failed")
        return f"❌ 寫入失敗: {type(e).__name__}"
    finally:
        conn.close()

    eq = f"${v.equity_usd:,.2f}" if v.equity_usd is not None else "?"
    return (
        f"✅ 帳戶 <b>{label}</b> 註冊成功\n"
        f"權限: {v.perm}（無提幣 ✓）\n"
        f"帳戶餘額: {eq}\n"
        f"跟單資金上限: ${capital_usd:,.0f}\n"
        f"狀態: PENDING（尚未跟單）\n\n"
        f"Phase 1 多帳戶執行器上線後，用 /okx_resumeacct {label} 啟動跟單。"
    )


def list_accounts() -> str:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT label, status, initial_capital_usd, owner_chat_id, "
                "       perm_snapshot, created_at "
                "FROM okx_accounts ORDER BY id"
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return "（尚無跟單帳戶）\n新增: /okx_addacct label key secret pass capital"
    lines = ["<b>跟單帳戶</b>"]
    for r in rows:
        lines.append(
            f"· <b>{r['label']}</b> [{r['status']}] "
            f"cap ${float(r['initial_capital_usd']):,.0f} "
            f"perm={r['perm_snapshot'] or '?'}"
        )
    return "\n".join(lines)


def set_account_status(label: str, status: str) -> str:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            n = cur.execute(
                "UPDATE okx_accounts SET status=%s WHERE label=%s",
                (status, label.strip().lower()),
            )
        conn.commit()
    finally:
        conn.close()
    if not n:
        return f"❌ 找不到帳戶 '{label}'"
    return f"✅ {label} → {status}"


def delete_account(label: str) -> str:
    """Hard-delete a registration (credentials removed from DB)."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            n = cur.execute(
                "DELETE FROM okx_accounts WHERE label=%s",
                (label.strip().lower(),),
            )
        conn.commit()
    finally:
        conn.close()
    if not n:
        return f"❌ 找不到帳戶 '{label}'"
    return f"✅ 已刪除 '{label}'（credentials 已從 DB 移除）"


# ── Telegram plumbing ──────────────────────────────────────────────────

def _tg_token() -> str:
    return os.environ.get("TELEGRAM_BOT_TOKEN", "")


def _tg_send(chat_id: str, text: str) -> None:
    token = _tg_token()
    if not token:
        return
    try:
        _requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        logger.exception("acct_tg_send_failed")


def _tg_delete(chat_id: str, message_id) -> None:
    """Best-effort delete of the message that carried raw credentials."""
    token = _tg_token()
    if not token or not message_id:
        return
    try:
        _requests.post(
            f"https://api.telegram.org/bot{token}/deleteMessage",
            json={"chat_id": chat_id, "message_id": message_id},
            timeout=10,
        )
    except Exception:
        logger.exception("acct_tg_delete_failed")


def _is_admin(chat_id: str) -> bool:
    admin = (os.environ.get("TG_ADMIN_CHAT_ID", "")
             or os.environ.get("TG_CRITICAL_CHAT_ID", "")).strip()
    return bool(admin) and str(chat_id).strip() == admin


# ── Command handlers (run in worker threads from the webhook) ──────────

def handle_addacct(chat_id: str, raw_text: str, message_id=None) -> None:
    # Delete the credentials message FIRST — even if parsing fails, the
    # secrets should not linger in chat history.
    _tg_delete(chat_id, message_id)
    if not _is_admin(chat_id):
        _tg_send(chat_id, "❌ 僅限 admin 操作")
        return
    parts = raw_text.split()
    if len(parts) != 6:
        _tg_send(chat_id,
                 "格式: /okx_addacct label api_key api_secret passphrase capital\n"
                 "例: /okx_addacct friend_a xxxx yyyy zzzz 100\n"
                 "（原訊息已自動刪除，credentials 不留聊天室）")
        return
    _, label, api_key, api_secret, passphrase, cap = parts
    try:
        capital = float(cap)
    except ValueError:
        _tg_send(chat_id, "❌ capital 必須是數字（USD）")
        return
    _tg_send(chat_id, f"⏳ 驗證 {label} 的 API key（讀 balance + 權限檢查）…")
    result = add_account(label, api_key, api_secret, passphrase, capital)
    _tg_send(chat_id, result)


def handle_accounts_list(chat_id: str) -> None:
    if not _is_admin(chat_id):
        _tg_send(chat_id, "❌ 僅限 admin 操作")
        return
    _tg_send(chat_id, list_accounts())


def handle_acct_status(chat_id: str, raw_text: str, status: str) -> None:
    """Shared handler for /okx_pauseacct /okx_resumeacct /okx_delacct."""
    if not _is_admin(chat_id):
        _tg_send(chat_id, "❌ 僅限 admin 操作")
        return
    parts = raw_text.split()
    if len(parts) != 2:
        _tg_send(chat_id, "格式: 指令 + label，例: /okx_pauseacct friend_a")
        return
    label = parts[1]
    if status == "DELETE":
        _tg_send(chat_id, delete_account(label))
    else:
        _tg_send(chat_id, set_account_status(label, status))
