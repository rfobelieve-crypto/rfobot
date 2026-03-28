"""
Normalize raw trades from any exchange adapter into the unified trade schema.
"""

import time
from market_data.core.symbol_mapper import to_canonical, get_contract_info


def normalize(raw: dict) -> dict | None:
    """
    Convert a raw trade dict (from an adapter) to the normalized schema.

    Required raw fields:
        exchange, raw_symbol, price, size, taker_side, trade_id, ts_exchange

    Optional raw fields:
        is_aggregated_trade (default False)

    Returns None if the symbol is not tracked.
    """
    exchange = raw["exchange"]
    raw_symbol = raw["raw_symbol"]

    canonical = to_canonical(exchange, raw_symbol)
    if canonical is None:
        return None

    info = get_contract_info(exchange, raw_symbol)
    contract_size = info["contract_size"]
    size_unit = info["size_unit"]

    price = float(raw["price"])
    size = float(raw["size"])

    # notional = price * size * contract_size
    # For base-unit exchanges (Binance), contract_size=1, so notional = price * size.
    # For contract-unit exchanges (OKX), size is in contracts.
    notional_usd = price * size * contract_size

    return {
        "exchange": exchange,
        "raw_symbol": raw_symbol,
        "canonical_symbol": canonical,
        "instrument_type": "perp",
        "price": price,
        "size": size,
        "size_unit": size_unit,
        "taker_side": raw["taker_side"],
        "notional_usd": notional_usd,
        "trade_id": str(raw.get("trade_id", "")),
        "ts_exchange": int(raw["ts_exchange"]),
        "ts_received": int(raw.get("ts_received", time.time() * 1000)),
        "is_aggregated_trade": bool(raw.get("is_aggregated_trade", False)),
    }
