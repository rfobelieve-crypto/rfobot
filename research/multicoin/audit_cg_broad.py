"""
Multi-coin data audit — broadened shortlist (2026-07-23).

Extends Step 1's audit_cg_endpoints.py pattern from {BTC, ETH, SOL} to a
liquidity-ranked shortlist of real-crypto Binance USDT-margined perpetuals
that Coinglass tracks (filters out Binance's newer TRADIFI_PERPETUAL
tokenized-stock/commodity products, which are NOT crypto and would pollute
a naive "top volume" ranking).

Shortlist construction:
  1. Coinglass /futures/supported-exchange-pairs -> Binance USDT-margined
     base assets Coinglass actually tracks.
  2. Binance /fapi/v1/exchangeInfo -> filter contractType == "PERPETUAL"
     (excludes contractType == "TRADIFI_PERPETUAL", the tokenized
     equity/commodity products Binance recently listed on the same venue).
  3. Binance /fapi/v1/ticker/24hr -> rank by quoteVolume, top 20 excl.
     BTC/ETH/SOL (already audited in Step 1).

Pure research script: no production imports, no DB writes.
Reads COINGLASS_API_KEY from flow_system/.env.

Usage:
    python research/multicoin/audit_cg_broad.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
CG_BASE = "https://open-api-v4.coinglass.com/api"
SLEEP = 0.8

N_CANDIDATES = 20


def load_api_key() -> str:
    env_path = ROOT / ".env"
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("COINGLASS_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("COINGLASS_API_KEY not found in .env")


API_KEY = load_api_key()
HEADERS = {"accept": "application/json", "CG-API-KEY": API_KEY}


def build_shortlist(n: int = N_CANDIDATES) -> list[str]:
    r = requests.get(CG_BASE + "/futures/supported-exchange-pairs",
                     headers=HEADERS, timeout=15)
    binance = r.json().get("data", {}).get("Binance", [])
    cg_usdt_bases = {p["base_asset"] for p in binance
                     if p.get("quote_asset") == "USDT"}

    info = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=15).json()
    real_crypto_syms = {s["symbol"] for s in info.get("symbols", [])
                        if s.get("contractType") == "PERPETUAL"
                        and s.get("quoteAsset") == "USDT"}

    resp = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=15).json()
    rows = []
    for t in resp:
        sym = t.get("symbol", "")
        if sym in real_crypto_syms:
            base_asset = sym[:-4]
            if base_asset in cg_usdt_bases:
                try:
                    qv = float(t.get("quoteVolume", 0))
                except (TypeError, ValueError):
                    qv = 0.0
                rows.append((base_asset, qv))
    rows.sort(key=lambda x: -x[1])
    excl = {"BTC", "ETH", "SOL", "USDC", "USDT", "FDUSD", "BUSD"}
    return [b for b, _ in rows if b not in excl][:n]


# Mirror of production CG_ENDPOINTS (indicator/data_fetcher.py) minus
# coinbase_premium (Step 1 proved it's symbol-ignoring / BTC-only).
TIMESERIES = {
    "oi":               {"path": "/futures/open-interest/history", "exchange": "Binance", "symbol_style": "pair"},
    "oi_agg":           {"path": "/futures/open-interest/aggregated-history", "symbol_style": "coin"},
    "liquidation":      {"path": "/futures/liquidation/history", "exchange": "Binance", "symbol_style": "pair"},
    "long_short":       {"path": "/futures/top-long-short-account-ratio/history", "exchange": "Binance", "symbol_style": "pair"},
    "global_ls":        {"path": "/futures/global-long-short-account-ratio/history", "exchange": "Binance", "symbol_style": "pair"},
    "funding":          {"path": "/futures/funding-rate/history", "exchange": "Binance", "symbol_style": "pair"},
    "taker":            {"path": "/futures/taker-buy-sell-volume/history", "exchange": "Binance", "symbol_style": "pair"},
    "bitfinex_margin":  {"path": "/bitfinex-margin-long-short", "symbol_style": "coin"},
    "top_ls_position":  {"path": "/futures/top-long-short-position-ratio/history", "exchange": "Binance", "symbol_style": "pair"},
    "futures_cvd_agg":  {"path": "/futures/aggregated-cvd/history", "symbol_style": "coin", "extra": {"exchange_list": "Binance"}},
    "spot_cvd_agg":     {"path": "/spot/aggregated-cvd/history", "symbol_style": "coin", "extra": {"exchange_list": "Binance"}},
    "liq_agg":          {"path": "/futures/liquidation/aggregated-history", "symbol_style": "coin", "extra": {"exchange_list": "Binance"}},
    "oi_coin_margin":   {"path": "/futures/open-interest/aggregated-coin-margin-history", "symbol_style": "coin", "extra": {"exchange_list": "Binance"}},
}


def cg_get(path: str, params: dict) -> tuple[str, object]:
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


def describe_rows(data) -> tuple[str, int]:
    """Returns (short_desc, n_rows)."""
    if data is None:
        return "-", 0
    rows = data
    if isinstance(rows, list):
        n = len(rows)
        if n == 0:
            return "EMPTY", 0
        return f"{n} rows", n
    return type(rows).__name__, 0


def audit_coin(coin: str) -> dict:
    """Returns {endpoint: (status, n_rows)} for one coin."""
    out = {}
    for name, cfg in TIMESERIES.items():
        style = cfg["symbol_style"]
        params = {"interval": "1h", "limit": 500}
        params.update(cfg.get("extra", {}))
        if cfg.get("exchange"):
            params["exchange"] = cfg["exchange"]
        if style == "pair":
            params["symbol"] = f"{coin}USDT"
        elif style == "coin":
            params["symbol"] = coin
        status, data = cg_get(cfg["path"], params)
        desc, n = describe_rows(data) if status == "OK" else (status, 0)
        out[name] = (desc, n)
        time.sleep(SLEEP)
    return out


def main():
    print("Building liquidity-ranked shortlist...")
    shortlist = build_shortlist()
    print(f"Shortlist ({len(shortlist)}): {shortlist}")

    results = {}
    for i, coin in enumerate(shortlist):
        print(f"\n[{i+1}/{len(shortlist)}] Auditing {coin}...")
        results[coin] = audit_coin(coin)
        n_ok = sum(1 for _, n in results[coin].values() if n > 0)
        print(f"  {coin}: {n_ok}/{len(TIMESERIES)} endpoints have data")

    # Coverage summary table
    lines = ["# Coinglass 廣泛小幣資料覆蓋率稽核（2026-07-23）\n",
             f"Shortlist（Coinglass 追蹤 + Binance 真實 PERPETUAL + 24h 量排序，"
             f"排除 BTC/ETH/SOL 及 TRADIFI_PERPETUAL 代幣化股票/商品）：\n",
             f"{', '.join(shortlist)}\n",
             "| coin | 覆蓋端點數/總數 | 缺失端點 |",
             "|---|---|---|"]
    coverage_pct = {}
    for coin in shortlist:
        r = results[coin]
        ok_endpoints = [name for name, (desc, n) in r.items() if n > 0]
        missing = [name for name, (desc, n) in r.items() if n == 0]
        pct = len(ok_endpoints) / len(TIMESERIES)
        coverage_pct[coin] = pct
        lines.append(f"| {coin} | {len(ok_endpoints)}/{len(TIMESERIES)} ({pct*100:.0f}%) | {', '.join(missing) if missing else '-'} |")

    report = "\n".join(lines)
    print("\n" + report)

    out_dir = Path(__file__).parent
    (out_dir / "audit_broad_results.md").write_text(report, encoding="utf-8")
    (out_dir / "audit_broad_raw.json").write_text(
        json.dumps({c: {k: v for k, v in r.items()} for c, r in results.items()}, indent=1),
        encoding="utf-8")
    print(f"\nSaved -> {out_dir / 'audit_broad_results.md'}")

    ranked = sorted(coverage_pct.items(), key=lambda x: -x[1])
    print("\n覆蓋率排序（高到低）：")
    for coin, pct in ranked:
        print(f"  {coin}: {pct*100:.0f}%")


if __name__ == "__main__":
    main()
