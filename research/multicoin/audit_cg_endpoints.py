"""
Multi-coin data audit — Step 1 of V7 multi-coin feasibility study.

Tests every Coinglass v4 endpoint used by the production feature pipeline
(indicator/data_fetcher.py) with ETH and SOL params (BTC as control), plus
Deribit DVOL. Reports availability, row counts, and timestamp unit per coin.

Pure research script: no production imports, no DB writes.
Reads COINGLASS_API_KEY from flow_system/.env.

Usage:
    python research/multicoin/audit_cg_endpoints.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
CG_BASE = "https://open-api-v4.coinglass.com/api"
DERIBIT_BASE = "https://www.deribit.com/api/v2/public"
SLEEP = 0.8  # spacing between calls, stay well under plan rate limit

COINS = ["BTC", "ETH", "SOL"]


def load_api_key() -> str:
    env_path = ROOT / ".env"
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("COINGLASS_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("COINGLASS_API_KEY not found in .env")


API_KEY = load_api_key()
HEADERS = {"accept": "application/json", "CG-API-KEY": API_KEY}

# Mirror of CG_ENDPOINTS in indicator/data_fetcher.py.
# symbol_style: "pair" -> BTCUSDT, "coin" -> BTC, None -> no symbol param (BTC-only endpoint)
TIMESERIES = {
    "oi":               {"path": "/futures/open-interest/history", "exchange": "Binance", "symbol_style": "pair"},
    "oi_agg":           {"path": "/futures/open-interest/aggregated-history", "symbol_style": "coin"},
    "liquidation":      {"path": "/futures/liquidation/history", "exchange": "Binance", "symbol_style": "pair"},
    "long_short":       {"path": "/futures/top-long-short-account-ratio/history", "exchange": "Binance", "symbol_style": "pair"},
    "global_ls":        {"path": "/futures/global-long-short-account-ratio/history", "exchange": "Binance", "symbol_style": "pair"},
    "funding":          {"path": "/futures/funding-rate/history", "exchange": "Binance", "symbol_style": "pair"},
    "taker":            {"path": "/futures/taker-buy-sell-volume/history", "exchange": "Binance", "symbol_style": "pair"},
    "coinbase_premium": {"path": "/coinbase-premium-index", "symbol_style": None},
    "bitfinex_margin":  {"path": "/bitfinex-margin-long-short", "symbol_style": "coin"},
    "top_ls_position":  {"path": "/futures/top-long-short-position-ratio/history", "exchange": "Binance", "symbol_style": "pair"},
    "futures_cvd_agg":  {"path": "/futures/aggregated-cvd/history", "symbol_style": "coin", "extra": {"exchange_list": "Binance"}},
    "spot_cvd_agg":     {"path": "/spot/aggregated-cvd/history", "symbol_style": "coin", "extra": {"exchange_list": "Binance"}},
    "liq_agg":          {"path": "/futures/liquidation/aggregated-history", "symbol_style": "coin", "extra": {"exchange_list": "Binance"}},
    "oi_coin_margin":   {"path": "/futures/open-interest/aggregated-coin-margin-history", "symbol_style": "coin", "extra": {"exchange_list": "Binance"}},
}


def cg_get(path: str, params: dict) -> tuple[str, object]:
    """Return (status, payload). status: OK / API:<code> / HTTP:<code> / ERR."""
    try:
        resp = requests.get(CG_BASE + path, params=params, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return f"HTTP:{resp.status_code}", None
        body = resp.json()
        code = str(body.get("code", "?"))
        if code != "0":
            return f"API:{code} {str(body.get('msg', ''))[:40]}", None
        return "OK", body.get("data")
    except Exception as e:  # noqa: BLE001
        return f"ERR:{type(e).__name__}", None


def ts_unit(sample_ts) -> str:
    """Detect timestamp unit (mistake.md 2026-04-12: never assume ms)."""
    try:
        v = float(sample_ts)
    except (TypeError, ValueError):
        return "?"
    return "ms" if v > 1e12 else "s"


def describe_rows(data) -> str:
    if data is None:
        return "-"
    rows = data
    if isinstance(data, dict):
        for key in ("data_list", "list"):
            if key in data and isinstance(data[key], list):
                rows = data[key]
                break
        else:
            return f"dict({len(data)} keys)"
    if isinstance(rows, list):
        n = len(rows)
        if n == 0:
            return "EMPTY"
        first = rows[0]
        if isinstance(first, dict):
            tkey = next((k for k in ("time", "t", "ts", "createTime") if k in first), None)
            unit = ts_unit(first.get(tkey)) if tkey else "?"
            return f"{n} rows, ts={unit}"
        return f"{n} rows"
    return type(rows).__name__


def audit_timeseries(results: list):
    for name, cfg in TIMESERIES.items():
        for coin in COINS:
            style = cfg["symbol_style"]
            params = {"interval": "1h", "limit": 10}
            params.update(cfg.get("extra", {}))
            if cfg.get("exchange"):
                params["exchange"] = cfg["exchange"]
            if style == "pair":
                params["symbol"] = f"{coin}USDT"
            elif style == "coin":
                params["symbol"] = coin
            elif coin != "BTC":
                results.append((name, coin, "N/A (no symbol param — BTC-only endpoint)"))
                continue
            status, data = cg_get(cfg["path"], params)
            detail = describe_rows(data) if status == "OK" else status
            results.append((name, coin, detail))
            time.sleep(SLEEP)


def audit_snapshots(results: list):
    # Options (Deribit via CG)
    for coin in COINS:
        status, data = cg_get("/option/max-pain", {"symbol": coin, "exchange": "Deribit"})
        results.append(("opt_max_pain", coin, describe_rows(data) if status == "OK" else status))
        time.sleep(SLEEP)
        status, data = cg_get("/option/info", {"symbol": coin})
        results.append(("opt_info", coin, describe_rows(data) if status == "OK" else status))
        time.sleep(SLEEP)
        status, data = cg_get("/index/option-vs-futures-oi-ratio", {"symbol": coin})
        results.append(("opt_fut_ratio", coin, describe_rows(data) if status == "OK" else status))
        time.sleep(SLEEP)

    # ETF flow/AUM — per-coin dedicated paths on CG
    etf_paths = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}
    for coin, slug in etf_paths.items():
        status, data = cg_get(f"/etf/{slug}/flow-history", {})
        results.append(("etf_flow", coin, describe_rows(data) if status == "OK" else status))
        time.sleep(SLEEP)
        status, data = cg_get(f"/etf/{slug}/aum", {})
        results.append(("etf_aum", coin, describe_rows(data) if status == "OK" else status))
        time.sleep(SLEEP)

    # Netflow lists — single call returns all symbols; check row presence
    for path, label in [("/futures/netflow-list", "futures_netflow"),
                        ("/spot/netflow-list", "spot_netflow")]:
        status, data = cg_get(path, {"symbol": "BTC"})
        if status == "OK" and isinstance(data, list):
            symbols = {str(r.get("symbol", "")).upper() for r in data}
            for coin in COINS:
                results.append((label, coin, "row present" if coin in symbols
                                else f"row MISSING (list has {len(symbols)} symbols)"))
        else:
            for coin in COINS:
                results.append((label, coin, status))
        time.sleep(SLEEP)

    # Hyperliquid whale positions — list of positions with symbol field
    status, data = cg_get("/hyperliquid/whale-position", {"limit": 200})
    if status == "OK" and isinstance(data, list):
        counts = {}
        for r in data:
            s = str(r.get("symbol", "")).upper()
            counts[s] = counts.get(s, 0) + 1
        for coin in COINS:
            results.append(("hl_whale", coin, f"{counts.get(coin, 0)} positions in top-200"))
    else:
        for coin in COINS:
            results.append(("hl_whale", coin, status))
    time.sleep(SLEEP)

    # Fear & Greed — market-wide, shared across coins
    status, data = cg_get("/index/fear-greed-history", {"limit": 2})
    fg = describe_rows(data) if status == "OK" else status
    for coin in COINS:
        results.append(("fear_greed", coin, f"SHARED ({fg})"))


def audit_deribit_dvol(results: list):
    now_ms = int(time.time() * 1000)
    for coin in COINS:
        try:
            resp = requests.get(DERIBIT_BASE + "/get_volatility_index_data", params={
                "currency": coin, "resolution": "3600",
                "start_timestamp": now_ms - 7200_000, "end_timestamp": now_ms,
            }, timeout=30)
            body = resp.json()
            if "result" in body:
                n = len(body["result"].get("data", []))
                results.append(("deribit_dvol", coin, f"{n} candles" if n else "EMPTY"))
            else:
                err = body.get("error", {}).get("message", "unknown error")
                results.append(("deribit_dvol", coin, f"API: {err[:50]}"))
        except Exception as e:  # noqa: BLE001
            results.append(("deribit_dvol", coin, f"ERR:{type(e).__name__}"))
        time.sleep(0.3)


def main():
    results: list[tuple[str, str, str]] = []
    print("Auditing Coinglass timeseries endpoints (14 x 3 coins)...")
    audit_timeseries(results)
    print("Auditing snapshot endpoints...")
    audit_snapshots(results)
    print("Auditing Deribit DVOL...")
    audit_deribit_dvol(results)

    # Pivot into endpoint x coin table
    endpoints = list(dict.fromkeys(name for name, _, _ in results))
    table = {name: {} for name in endpoints}
    for name, coin, detail in results:
        table[name][coin] = detail

    lines = ["| endpoint | BTC | ETH | SOL |", "|---|---|---|---|"]
    for name in endpoints:
        row = table[name]
        lines.append(f"| {name} | {row.get('BTC', '-')} | {row.get('ETH', '-')} | {row.get('SOL', '-')} |")
    report = "\n".join(lines)
    print("\n" + report)

    out = Path(__file__).parent / "audit_results.md"
    out.write_text(
        f"# Coinglass multi-coin endpoint audit\n\nRun: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}\n\n"
        + report + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved -> {out}")

    raw_out = Path(__file__).parent / "audit_results_raw.json"
    raw_out.write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
