"""
Symbol mapping: exchange raw_symbol -> canonical_symbol.

Phase 1: BTC-USD, ETH-USD only. Hard-coded.
"""
from __future__ import annotations

# 2026-07-28: 8 alts added on binance only. They already stream into
# depth_deltas_1m, but every cancel-flow playbook gates on vshock (flow_bars)
# and most need ret_1m (orderbook mid), so without trades + depth snapshots
# they cannot be scanned. okx/bybit stay BTC+ETH — a third venue multiplies
# WS load and DB growth for no benefit to the single-venue playbooks.
SYMBOL_MAP = {
    "binance": {
        "BTCUSDT": "BTC-USD",
        "ETHUSDT": "ETH-USD",
        "SOLUSDT": "SOL-USD",
        "XRPUSDT": "XRP-USD",
        "DOGEUSDT": "DOGE-USD",
        "ADAUSDT": "ADA-USD",
        "BNBUSDT": "BNB-USD",
        "LINKUSDT": "LINK-USD",
        "SUIUSDT": "SUI-USD",
        "UNIUSDT": "UNI-USD",
        "AAVEUSDT": "AAVE-USD",
    },
    "okx": {
        "BTC-USDT-SWAP": "BTC-USD",
        "ETH-USDT-SWAP": "ETH-USD",
    },
    "bybit": {
        "BTCUSDT": "BTC-USD",
        "ETHUSDT": "ETH-USD",
    },
}

# Contract sizes per (exchange, raw_symbol).
# Binance perp trades are already in base units (size_unit="base"), so contract_size=1.
# OKX SWAP trades are in contracts: BTC 0.01 BTC/contract, ETH 0.1 ETH/contract.
# Bybit USDT linear perp trades: size is in base units (BTC for BTCUSDT).
CONTRACT_INFO = {
    "binance": {
        # All Binance USDT-M perp trades report size in base units, so
        # contract_size is 1.0 regardless of the coin's price.
        "BTCUSDT": {"contract_size": 1.0, "size_unit": "base"},
        "ETHUSDT": {"contract_size": 1.0, "size_unit": "base"},
        "SOLUSDT": {"contract_size": 1.0, "size_unit": "base"},
        "XRPUSDT": {"contract_size": 1.0, "size_unit": "base"},
        "DOGEUSDT": {"contract_size": 1.0, "size_unit": "base"},
        "ADAUSDT": {"contract_size": 1.0, "size_unit": "base"},
        "BNBUSDT": {"contract_size": 1.0, "size_unit": "base"},
        "LINKUSDT": {"contract_size": 1.0, "size_unit": "base"},
        "SUIUSDT": {"contract_size": 1.0, "size_unit": "base"},
        "UNIUSDT": {"contract_size": 1.0, "size_unit": "base"},
        "AAVEUSDT": {"contract_size": 1.0, "size_unit": "base"},
    },
    "okx": {
        "BTC-USDT-SWAP": {"contract_size": 0.01, "size_unit": "contract"},
        "ETH-USDT-SWAP": {"contract_size": 0.1, "size_unit": "contract"},
    },
    "bybit": {
        "BTCUSDT": {"contract_size": 1.0, "size_unit": "base"},
        "ETHUSDT": {"contract_size": 1.0, "size_unit": "base"},
    },
}


def to_canonical(exchange: str, raw_symbol: str) -> str | None:
    """Map raw_symbol to canonical_symbol. Returns None if not tracked."""
    return SYMBOL_MAP.get(exchange, {}).get(raw_symbol)


def get_contract_info(exchange: str, raw_symbol: str) -> dict:
    """Return contract_size and size_unit for a given instrument."""
    return CONTRACT_INFO.get(exchange, {}).get(raw_symbol, {
        "contract_size": 1.0,
        "size_unit": "base",
    })


def tracked_symbols(exchange: str) -> list[str]:
    """Return list of raw_symbols we track for a given exchange."""
    return list(SYMBOL_MAP.get(exchange, {}).keys())
