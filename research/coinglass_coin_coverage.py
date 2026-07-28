"""Which coins can actually run a V7-style pipeline?

Before copying the BTC pipeline onto 9 alts, establish what data exists for
them. V7's deployed feature set is 136 columns, and the 2026-07-28 audit put
91 of them (64.6% of gain importance) on Coinglass. Our backfill config only
ever named BTC and ETH, but that is OUR limit — whether Coinglass itself
serves SUI/UNI/AAVE on the same endpoints is a separate question, and the
only honest way to answer it is to ask the API.

This probes each coin-scoped endpoint V7 consumes against every candidate
symbol and reports what comes back. Endpoints known to be BTC-only by
construction (Coinbase premium, Bitfinex margin, the ETF series) are probed
anyway rather than assumed — assumptions about vendor coverage are exactly
what this script exists to replace.

Output is a coverage matrix: for each coin, how many of V7's Coinglass
channels are available, and therefore how much of the 64.6% is reachable.

Run: python research/coinglass_coin_coverage.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CG_BASE = "https://open-api-v4.coinglass.com/api"
OUT = ROOT / "research/results/coinglass_coin_coverage.json"

COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "BNB",
         "LINK", "SUI", "UNI", "AAVE"]

# (name, path, uses_exchange, extra) — mirrors indicator/data_fetcher.py's
# CG_ENDPOINTS so coverage is measured against what V7 actually consumes.
ENDPOINTS = [
    ("oi",              "/futures/open-interest/history", True, None),
    ("oi_agg",          "/futures/open-interest/aggregated-history", False, None),
    ("oi_coin_margin",  "/futures/open-interest/aggregated-coin-margin-history",
                        False, {"exchange_list": "Binance"}),
    ("funding",         "/futures/funding-rate/history", True, None),
    ("liquidation",     "/futures/liquidation/history", True, None),
    ("liq_agg",         "/futures/liquidation/aggregated-history", False,
                        {"exchange_list": "Binance"}),
    ("top_ls_account",  "/futures/top-long-short-account-ratio/history", True, None),
    ("top_ls_position", "/futures/top-long-short-position-ratio/history", True, None),
    ("global_ls",       "/futures/global-long-short-account-ratio/history", True, None),
    ("taker",           "/futures/taker-buy-sell-volume/history", True, None),
    ("futures_cvd_agg", "/futures/aggregated-cvd/history", False,
                        {"exchange_list": "Binance"}),
    ("spot_cvd_agg",    "/spot/aggregated-cvd/history", False,
                        {"exchange_list": "Binance"}),
    ("bitfinex_margin", "/bitfinex-margin-long-short", False, None),
    ("coinbase_premium", "/coinbase-premium-index", False, None),
    ("opt_vs_fut_oi",   "/index/option-vs-futures-oi-ratio", False, None),
]

# Rough share of V7 gain importance each channel family carries, from the
# 2026-07-28 audit. Used to turn "n endpoints available" into "how much of
# the model is reachable" — 12 cheap endpoints are not worth 3 expensive ones.
WEIGHT = {
    "oi": 5.1, "oi_agg": 5.1, "oi_coin_margin": 5.0,
    "funding": 7.0,
    "liquidation": 3.2, "liq_agg": 3.2,
    "top_ls_account": 3.6, "top_ls_position": 3.6, "global_ls": 3.7,
    "taker": 2.0,
    "futures_cvd_agg": 4.6, "spot_cvd_agg": 4.5,
    "bitfinex_margin": 7.3,
    "coinbase_premium": 2.5, "opt_vs_fut_oi": 1.2,
}


def api_key() -> str:
    k = os.environ.get("COINGLASS_API_KEY", "")
    if k:
        return k
    env = ROOT / ".env"
    if env.exists():
        for line in io.open(env, encoding="utf-8", errors="replace").read().splitlines():
            if line.startswith("COINGLASS_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("COINGLASS_API_KEY not set")


def probe(key: str, coin: str, path: str, use_ex: bool, extra,
          tries: int = 5) -> tuple[bool, str]:
    """One endpoint/coin. Retries on rate limiting with backoff.

    The first version of this script used a flat 0.35s gap and read the
    resulting "Too Many Requests" as "endpoint unavailable for this coin" —
    which silently turned a throttle into a false coverage verdict. BTC, which
    demonstrably has every channel in production, came back 10/15. Treat any
    run where the BTC control is not 15/15 as invalid rather than as data.
    """
    params = {"interval": "1h", "limit": 5, "symbol": coin}
    if use_ex:
        params["exchange"] = "Binance"
        params["symbol"] = f"{coin}USDT"
    if extra:
        params.update(extra)
    delay = 2.0
    for attempt in range(tries):
        try:
            r = requests.get(CG_BASE + path, params=params,
                             headers={"CG-API-KEY": key}, timeout=25)
        except Exception as exc:
            return False, f"net:{type(exc).__name__}"
        if r.status_code == 429:
            time.sleep(delay); delay *= 2
            continue
        if r.status_code != 200:
            return False, f"http{r.status_code}"
        try:
            d = r.json()
        except Exception:
            return False, "badjson"
        msg = str(d.get("msg") or "")
        if "Too Many" in msg or str(d.get("code")) == "429":
            time.sleep(delay); delay *= 2
            continue
        if str(d.get("code")) != "0":
            return False, msg[:22] or str(d.get("code"))
        rows = d.get("data")
        if not isinstance(rows, list) or not rows:
            return False, "empty"
        return True, f"{len(rows)}rows"
    return False, "RATE-LIMITED"     # never counts as "unavailable"


def main() -> int:
    key = api_key()
    total_w = sum(WEIGHT.values())
    print(f"probing {len(ENDPOINTS)} endpoints x {len(COINS)} coins "
          f"(weights cover {total_w:.1f}% of V7 gain)\n")

    grid, out = {}, {}
    hdr = "endpoint".ljust(18) + "".join(c.rjust(6) for c in COINS)
    print(hdr)
    print("-" * len(hdr))
    for name, path, use_ex, extra in ENDPOINTS:
        row = {}
        cells = ""
        for coin in COINS:
            ok, why = probe(key, coin, path, use_ex, extra)
            row[coin] = dict(ok=ok, detail=why)
            cells += ("  ok  " if ok else ("  RL  " if why == "RATE-LIMITED"
                                           else "   ·  ")).rjust(6)
            time.sleep(1.2)           # base spacing; probe() backs off on 429
        grid[name] = row
        print(name.ljust(18) + cells, flush=True)

    # Control check: BTC provably has every channel in production. If it does
    # not come back clean, the run measured our own throttling, not coverage.
    btc_ok = sum(1 for n in grid if grid[n]["BTC"]["ok"])
    rl = sum(1 for n in grid for c in COINS
             if grid[n][c]["detail"] == "RATE-LIMITED")
    print(f"\nBTC 對照組 {btc_ok}/{len(ENDPOINTS)}；仍被限流的格子 {rl}")
    if btc_ok < len(ENDPOINTS) or rl:
        print("!! 對照組未滿分或仍有限流 —— 本次結果不可作為覆蓋率判斷")

    print("\n" + "=" * 62)
    print("每個幣可取得的 V7 Coinglass 通道（依重要性加權）")
    print("=" * 62)
    print(f"{'coin':<7}{'端點':>8}{'加權覆蓋':>12}   缺少的通道")
    for coin in COINS:
        have = [n for n in grid if grid[n][coin]["ok"]]
        w = sum(WEIGHT[n] for n in have)
        miss = [n for n in grid if not grid[n][coin]["ok"]]
        out[coin] = dict(n_ok=len(have), weighted_pct=round(w, 1),
                         have=have, missing=miss)
        print(f"{coin:<7}{len(have):>3}/{len(ENDPOINTS):<4}{w:>10.1f}%   "
              f"{','.join(miss[:5])}{'…' if len(miss) > 5 else ''}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(grid=grid, summary=out), indent=2,
                              ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {OUT}")
    print("\n判讀：加權覆蓋 = 該幣拿得到的 Coinglass 通道佔 V7 gain 的比例。"
          "\nBTC 是滿分基準；某幣若只有 30%，複製管線得到的是殘缺模型，"
          "\n不是「V7 的該幣版本」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
