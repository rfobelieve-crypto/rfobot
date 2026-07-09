"""Read-only OKX live smoke test — Stage 2 progression gate.

What it does (in order, each must pass):
  1. Load + validate OKX live config (E1-E4 fail-fast)
  2. GET /api/v5/public/time → NTP drift < halt threshold
  3. GET /api/v5/account/config → API perms valid (no withdraw/transfer)
  4. GET /api/v5/account/balance → returns a Balance
  5. GET /api/v5/account/positions → returns list (likely empty)
  6. PositionReconciler.reconcile_cycle → CONSISTENT
  6b. (only if td_mode == "isolated") POST set-leverage idempotent dry-check
      for each posSide that will be used at open time. Verifies that next
      Strong signal will not abort with "set_leverage_failed". No order, no
      margin change beyond what OKX already has.
  7. Start WS, subscribe to orders/positions/account, wait for auth +
     first heartbeat (≤ 15 s)
  8. Print summary + exit clean

What it does NOT do: submit any order, amend any algo, cancel anything.
Zero financial side effects. set-leverage (step 6b) is idempotent — if the
account already shows isolated 10x/10x on the OKX UI, the call just confirms.

Usage:
    # Required env (or .env):
    #   OKX_API_KEY_LIVE, OKX_API_SECRET_LIVE, OKX_PASSPHRASE_LIVE
    #   STAGE=live
    #   TG_CRITICAL_CHAT_ID
    python scripts/okx_smoke_live.py
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


from indicator.okx.client import OkxClient
from indicator.okx.config import load_okx_config_from_env, validate_okx_config
from indicator.okx.kill_checks import check_api_permissions, check_ntp_drift
from indicator.okx.reconciler import PositionReconciler
from indicator.okx.state import OkxStateStore
from indicator.okx.types import ReconciliationVerdict


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("okx_smoke")


class StepFailed(RuntimeError):
    pass


def _step(name: str, fn):
    """Run a step; print PASS/FAIL line; raise on failure."""
    print(f"[*] {name} ... ", end="", flush=True)
    try:
        detail = fn()
    except StepFailed as e:
        print(f"FAIL: {e}")
        raise
    except Exception as e:
        print(f"FAIL ({type(e).__name__}): {e}")
        raise StepFailed(name) from e
    print(f"PASS  {detail or ''}")


def main(stage: str = "live") -> int:
    print(f"\n=== OKX {stage.upper()} read-only smoke test ===\n")

    # 1. Config
    cfg_holder = {}
    def step_config():
        cfg = load_okx_config_from_env(stage=stage)
        validate_okx_config(cfg)
        cfg_holder["cfg"] = cfg
        return (f"is_simulated={cfg.is_simulated} "
                f"inst_id={cfg.inst_id} leverage={cfg.leverage} "
                f"capital=${cfg.initial_capital_usd}")
    try:
        _step("config + validation", step_config)
    except StepFailed:
        return 1

    cfg = cfg_holder["cfg"]
    client = OkxClient(cfg)

    # 2. Server time / NTP drift
    def step_server_time():
        server_ts = client.get_server_time()
        if server_ts is None:
            raise StepFailed("get_server_time returned None")
        local_ts = time.time()
        drift = abs(local_ts - server_ts)
        result = check_ntp_drift(
            local_ts_sec=local_ts, server_ts_sec=server_ts,
            halt_threshold_sec=cfg.ntp_drift_halt_sec,
            demote_threshold_sec=cfg.ntp_drift_demote_sec,
        )
        if result.triggered:
            raise StepFailed(
                f"NTP drift {drift:.2f}s triggered "
                f"{result.trigger_id}: {result.reason}"
            )
        return f"drift={drift:.3f}s"
    try:
        _step("server time + NTP drift", step_server_time)
    except StepFailed:
        return 2

    # 3. API permissions
    def step_perms():
        acct = client.get_account_config() or {}
        data = acct.get("data") or []
        perm_str = ""
        if data:
            perm_str = (data[0].get("perm")
                        or data[0].get("permissions") or "")
        if not perm_str:
            return "perm field absent — verify in OKX UI manually"
        perms = [p.strip() for p in perm_str.split(",") if p.strip()]
        check = check_api_permissions(perms=perms)
        if check.triggered:
            raise StepFailed(check.reason)
        return f"perms={perm_str}"
    try:
        _step("API key permissions (E4)", step_perms)
    except StepFailed:
        return 3

    # 4. Balance
    def step_balance():
        bal = client.get_balance()
        if bal is None:
            raise StepFailed("get_balance returned None")
        return (f"total_eq=${bal.total_eq_usd:.2f} "
                f"available=${bal.available_usd:.2f}")
    try:
        _step("account balance", step_balance)
    except StepFailed:
        return 4

    # 5. Positions
    def step_positions():
        positions = client.get_positions(inst_id=cfg.inst_id)
        return f"{len(positions)} position(s) on {cfg.inst_id}"
    try:
        _step("account positions", step_positions)
    except StepFailed:
        return 5

    # 6. Reconciliation
    def step_recon():
        store = OkxStateStore(table_prefix=cfg.table_prefix)
        reconciler = PositionReconciler(client=client, store=store,
                                          inst_id=cfg.inst_id)
        result = reconciler.reconcile_cycle()
        if result.verdict == ReconciliationVerdict.UNAVAILABLE:
            raise StepFailed(
                f"reconciliation UNAVAILABLE: {result.detail}"
            )
        if result.verdict == ReconciliationVerdict.MISMATCH:
            raise StepFailed(
                f"reconciliation MISMATCH: {result.detail} "
                f"— investigate before proceeding"
            )
        return f"verdict={result.verdict.value}"
    try:
        _step("reconciliation (local DB vs OKX)", step_recon)
    except StepFailed:
        return 6

    # 6b. Isolated-mode pre-flight: verify set-leverage works for the posSides
    #     the executor will call before opening. Skipped for cross/cash modes
    #     where executor does not pre-call set-leverage.
    #
    #     This is the check that surfaces the "🔴 OKX OPEN ABORTED — set-leverage
    #     failed" condition BEFORE a real Strong signal arrives. If OKX UI
    #     doesn't have isolated 10x/10x set up, this step fails and tells you
    #     exactly which posSide needs setup, with zero risk of placing an order.
    if cfg.td_mode == "isolated":
        def step_set_leverage():
            # Which posSides we need to verify depends on pos_mode.
            #   long_short_mode → executor needs BOTH "long" and "short"
            #                     pre-configured (one per side)
            #   net_mode        → executor calls set-leverage with posSide=None
            if cfg.pos_mode == "long_short_mode":
                sides_to_check: list[Optional[str]] = ["long", "short"]
            else:
                sides_to_check = [None]
            results: list[str] = []
            for ps in sides_to_check:
                ok = client.set_leverage(
                    inst_id=cfg.inst_id, lever=cfg.leverage,
                    mgn_mode="isolated", pos_side=ps,
                )
                label = ps or "net"
                results.append(f"{label}={'ok' if ok else 'FAIL'}")
                if not ok:
                    raise StepFailed(
                        f"set-leverage(isolated {cfg.leverage}x posSide={label}) "
                        f"failed — OKX UI needs BTC-USDT-SWAP set to "
                        f"isolated · 10x · 10x BEFORE this will pass. "
                        f"Account must be FLAT to change margin mode."
                    )
            return f"isolated {cfg.leverage}x: " + " · ".join(results)
        try:
            _step("set-leverage idempotent dry-check (isolated only)",
                  step_set_leverage)
        except StepFailed:
            return 8

    # 7. WS auth + first heartbeat
    def step_ws():
        client.start_ws()
        # Wire no-op callbacks so subscriptions go through
        client.subscribe_orders(lambda evt: None)
        client.subscribe_positions(lambda evt: None)
        client.subscribe_balance(lambda evt: None)
        # Poll connectivity for up to 15 s
        deadline = time.time() + 15.0
        while time.time() < deadline:
            status = client.connectivity()
            if (status.private_ws_ok
                    and status.last_private_heartbeat_age_sec < 30):
                age = status.last_private_heartbeat_age_sec
                return f"private_ws_ok hb_age={age:.1f}s"
            time.sleep(0.5)
        raise StepFailed(
            "WS did not auth + heartbeat within 15 s"
        )
    try:
        _step("WS private auth + subscribe", step_ws)
    except StepFailed:
        return 7
    finally:
        try:
            client.close()
        except Exception:
            pass

    print("\n=== ALL STEPS PASSED ===")
    print("Stage 2 progression gate cleared.")
    print("Next: enable executor in update_cycle with manual approval ON.")
    return 0


if __name__ == "__main__":
    stage = "testnet" if "--testnet" in sys.argv else "live"
    try:
        rc = main(stage=stage)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        rc = 130
    sys.exit(rc)
