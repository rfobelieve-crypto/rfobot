import csv
import uuid
import random
from datetime import datetime, timedelta, timezone


OUTPUT_FILE = "liquidity_events_simulated.csv"
NUM_ROWS = 300
DAYS_BACK = 45


def simulate_one_event(row_id: int, base_time: datetime):
    symbol = random.choice(["BTCUSDT", "ETHUSDT"])
    liquidity_side = random.choice(["buy", "sell"])
    event_type = "liquidity_sweep"

    if symbol == "BTCUSDT":
        trigger_price = random.uniform(85000, 105000)
    else:
        trigger_price = random.uniform(2200, 5200)

    observation_seconds = random.choice([60, 300, 900, 3600])

    upper_target = trigger_price * (1 + random.uniform(0.001, 0.006))
    lower_target = trigger_price * (1 - random.uniform(0.001, 0.006))

    flow_until_hit_buy = random.uniform(50000, 500000)
    flow_until_hit_sell = random.uniform(50000, 500000)
    flow_until_hit_delta = flow_until_hit_buy - flow_until_hit_sell
    flow_until_hit_trades = random.randint(100, 3000)

    flow_1h_buy = random.uniform(100000, 1000000)
    flow_1h_sell = random.uniform(100000, 1000000)
    flow_1h_delta = flow_1h_buy - flow_1h_sell
    flow_1h_trades = random.randint(500, 10000)

    base_reversal_prob = 0.55

    if liquidity_side == "buy":
        if flow_until_hit_delta < -50000:
            reversal_prob = 0.75
        elif flow_until_hit_delta > 80000:
            reversal_prob = 0.35
        else:
            reversal_prob = base_reversal_prob

        is_reversal = random.random() < reversal_prob
        if is_reversal:
            first_hit_side = "down"
            outcome_label = "reversal"
            hit_price = lower_target
        else:
            first_hit_side = "up"
            outcome_label = "continuation"
            hit_price = upper_target

    else:
        if flow_until_hit_delta > 50000:
            reversal_prob = 0.75
        elif flow_until_hit_delta < -80000:
            reversal_prob = 0.30
        else:
            reversal_prob = base_reversal_prob

        is_reversal = random.random() < reversal_prob
        if is_reversal:
            first_hit_side = "up"
            outcome_label = "reversal"
            hit_price = upper_target
        else:
            first_hit_side = "down"
            outcome_label = "continuation"
            hit_price = lower_target

    hit_latency_seconds = random.randint(10, observation_seconds)
    hit_ts = int(base_time.timestamp()) + hit_latency_seconds

    return {
        "id": row_id,
        "event_uuid": str(uuid.uuid4()),
        "event_type": event_type,
        "liquidity_side": liquidity_side,
        "symbol": symbol,
        "trigger_price": round(trigger_price, 8),
        "tv_time": base_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trigger_ts": int(base_time.timestamp()),
        "trigger_time_text": base_time.strftime("%Y-%m-%d %H:%M:%S"),
        "observation_seconds": observation_seconds,
        "upper_target": round(upper_target, 8),
        "lower_target": round(lower_target, 8),
        "first_hit_side": first_hit_side,
        "outcome_label": outcome_label,
        "hit_price": round(hit_price, 8),
        "hit_ts": hit_ts,
        "hit_latency_seconds": hit_latency_seconds,
        "flow_until_hit_buy": round(flow_until_hit_buy, 4),
        "flow_until_hit_sell": round(flow_until_hit_sell, 4),
        "flow_until_hit_delta": round(flow_until_hit_delta, 4),
        "flow_until_hit_trades": flow_until_hit_trades,
        "flow_1h_buy": round(flow_1h_buy, 4),
        "flow_1h_sell": round(flow_1h_sell, 4),
        "flow_1h_delta": round(flow_1h_delta, 4),
        "flow_1h_trades": flow_1h_trades,
        "created_at": base_time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def generate_csv():
    fieldnames = [
        "id",
        "event_uuid",
        "event_type",
        "liquidity_side",
        "symbol",
        "trigger_price",
        "tv_time",
        "trigger_ts",
        "trigger_time_text",
        "observation_seconds",
        "upper_target",
        "lower_target",
        "first_hit_side",
        "outcome_label",
        "hit_price",
        "hit_ts",
        "hit_latency_seconds",
        "flow_until_hit_buy",
        "flow_until_hit_sell",
        "flow_until_hit_delta",
        "flow_until_hit_trades",
        "flow_1h_buy",
        "flow_1h_sell",
        "flow_1h_delta",
        "flow_1h_trades",
        "created_at",
    ]

    now = datetime.now(timezone.utc)
    start_time = now - timedelta(days=DAYS_BACK)

    rows = []
    for i in range(1, NUM_ROWS + 1):
        event_time = start_time + timedelta(
            seconds=random.randint(0, DAYS_BACK * 24 * 3600)
        )
        rows.append(simulate_one_event(i, event_time))

    rows.sort(key=lambda x: x["trigger_ts"])

    for i, row in enumerate(rows, start=1):
        row["id"] = i

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ 已產生模擬資料：{OUTPUT_FILE}")
    print(f"✅ 筆數：{len(rows)}")


if __name__ == "__main__":
    generate_csv()