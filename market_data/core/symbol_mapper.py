"""
Symbol mapping: exchange raw_symbol -> canonical_symbol.

Phase 1: BTC-USD, ETH-USD only. Hard-coded.
"""

SYMBOL_MAP = {
    "binance": {
        "BTCUSDT": "BTC-USD",
        "ETHUSDT": "ETH-USD",
    },
    "okx": {
        "BTC-USDT-SWAP": "BTC-USD",
        "ETH-USDT-SWAP": "ETH-USD",
    },
}

# Contract sizes per (exchange, raw_symbol).
# Binance perp trades are already in base units (size_unit="base"), so contract_size=1.
# OKX SWAP trades are in contracts: BTC 0.01 BTC/contract, ETH 0.1 ETH/contract.
CONTRACT_INFO = {
    "binance": {
        "BTCUSDT": {"contract_size": 1.0, "size_unit": "base"},
        "ETHUSDT": {"contract_size": 1.0, "size_unit": "base"},
    },
    "okx": {
        "BTC-USDT-SWAP": {"contract_size": 0.01, "size_unit": "contract"},
        "ETH-USDT-SWAP": {"contract_size": 0.1, "size_unit": "contract"},
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
