"""Second-resolution depth event collector — BTC only, price-banded, with
cancel→re-add matching.  The F7 pre-registered bet (TODO.md §0.46).

Why this exists (2026-08-13): the 1m collector subscribes to @depth@100ms and
then keeps FOUR scalars per minute — a 600:1 compression that destroys the
two structures most likely to carry information:

  1. PRICE.  A cancel at the best bid and a cancel 2% away are the same
     number in depth_deltas_1m.  "Which layer was pulled" is the entire
     content of "is the wall real".
  2. CANCEL→RE-ADD NETTING.  A spoofed wall's defining signature is
     cancel-then-replace at the same price; inside a one-minute bucket that
     adds +X to the cancel column and +X to the add column and vanishes.

Every directional cancel hypothesis (5 registered families, all FAIL) was
tested on data with those structures ground off.  The surviving volatility
cell also says the signal lives at the SHORTEST horizon measured (activity-
controlled partial IC h=5min +0.040 [+0.019,+0.060], the only cell clear of
zero) — and 5 minutes is 5 data points at 1m resolution.  This collector
raises the floor to 1 second and keeps what the 1m table threw away.

FROZEN DEFINITIONS (2026-08-13, before any 1s data exists — do not tune):
  bands      distance of the touched price from pre-update mid, in bps:
             NEAR < 5, MID 5–25, FAR > 25
  re-add     an add at price p on side s within 5000 ms of a cancel at the
             same (s, p); matched FIFO, qty = min(add, pending cancel);
             counted in the second the RE-ADD lands (also still counted in
             its add band — research subtracts, the collector never nets)
  mid        (best_bid + best_ask)/2 of the maintained book BEFORE applying
             the update being classified — the state the actor acted on
  warm-up    no counting until both sides hold >= 20 levels and the book is
             uncrossed; a crossed book resets state (stale after gaps)

Storage is MySQL, not parquet: the container filesystem is wiped 2-10x/day
by deploys (mistake.md 2026-08-11 — rolling state in the container is how
the decoder stayed broken for three months).  BTC spot+perp only:
~173k rows/day; the §0.46 checkpoint is 30 days, retention decision at 90.
"""
from __future__ import annotations

import json
import logging
import threading
import time

import websocket

from shared.db import get_db_conn

logger = logging.getLogger(__name__)

SPOT_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@depth@100ms"
PERP_WS_URL = "wss://fstream.binance.com/ws/btcusdt@depth@100ms"
CANONICAL = "BTC-USD"

# ── frozen 2026-08-13 (pre-registration §0.46) ──────────────────────────
NEAR_BPS = 5.0
MID_BPS = 25.0
REATTACH_MS = 5_000
MIN_BOOK_LEVELS = 20
# ────────────────────────────────────────────────────────────────────────

FLUSH_SEC = 10
BANDS = ("near", "mid", "far")


def _ensure_schema() -> None:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS depth_events_1s (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                exchange VARCHAR(20) NOT NULL,
                canonical_symbol VARCHAR(20) NOT NULL,
                sec_start_ms BIGINT NOT NULL,
                mid_px DOUBLE NULL,
                bid_add_near DOUBLE NOT NULL DEFAULT 0,
                bid_add_mid DOUBLE NOT NULL DEFAULT 0,
                bid_add_far DOUBLE NOT NULL DEFAULT 0,
                bid_cancel_near DOUBLE NOT NULL DEFAULT 0,
                bid_cancel_mid DOUBLE NOT NULL DEFAULT 0,
                bid_cancel_far DOUBLE NOT NULL DEFAULT 0,
                ask_add_near DOUBLE NOT NULL DEFAULT 0,
                ask_add_mid DOUBLE NOT NULL DEFAULT 0,
                ask_add_far DOUBLE NOT NULL DEFAULT 0,
                ask_cancel_near DOUBLE NOT NULL DEFAULT 0,
                ask_cancel_mid DOUBLE NOT NULL DEFAULT 0,
                ask_cancel_far DOUBLE NOT NULL DEFAULT 0,
                bid_readd_qty DOUBLE NOT NULL DEFAULT 0,
                ask_readd_qty DOUBLE NOT NULL DEFAULT 0,
                update_count INT NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uq_sec (exchange, canonical_symbol, sec_start_ms),
                INDEX idx_sec (sec_start_ms)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
        conn.commit()
        logger.info("depth_events_1s table ready")
    finally:
        conn.close()


def band_of(px: float, mid: float) -> str:
    """Distance band of a touched price from the pre-update mid."""
    dist_bps = abs(px - mid) / mid * 1e4
    if dist_bps < NEAR_BPS:
        return "near"
    if dist_bps <= MID_BPS:
        return "mid"
    return "far"


def _empty_bucket() -> dict[str, float]:
    b: dict[str, float] = {"n": 0, "mid": 0.0}
    for side in ("bid", "ask"):
        for act in ("add", "cancel"):
            for bd in BANDS:
                b[f"{side}_{act}_{bd}"] = 0.0
        b[f"{side}_readd"] = 0.0
    return b


class DepthEvents1sCollector:
    """1-second buckets of banded add/cancel qty + cancel→re-add matching.

    Deliberately additive (architecture.md): a THIRD consumer of the same
    stream, never a modification of DepthDeltaCollector — the 1m BTC series
    is frozen under the 07-10/07-15 registrations and must stay unbroken.
    """

    def __init__(self, ws_url: str = SPOT_WS_URL, exchange: str = "binance",
                 canonical_symbol: str = CANONICAL) -> None:
        self._ws_url = ws_url
        self._exchange = exchange
        self._canonical = canonical_symbol
        self._book: dict[str, dict[float, float]] = {"bid": {}, "ask": {}}
        # pending cancels awaiting a re-add: side -> price -> [(expire_ms, qty)]
        self._pending: dict[str, dict[float, list[tuple[int, float]]]] = {
            "bid": {}, "ask": {}}
        self._buckets: dict[int, dict[str, float]] = {}
        self._warmup_skipped = 0
        self._lock = threading.Lock()
        self._ws: websocket.WebSocketApp | None = None

    # ── book state ─────────────────────────────────────────────────────
    def _mid(self) -> float | None:
        bids, asks = self._book["bid"], self._book["ask"]
        if len(bids) < MIN_BOOK_LEVELS or len(asks) < MIN_BOOK_LEVELS:
            return None
        bb, ba = max(bids), min(asks)
        if bb >= ba:          # crossed = stale after a gap → reset, re-warm
            self._book = {"bid": {}, "ask": {}}
            self._pending = {"bid": {}, "ask": {}}
            return None
        return (bb + ba) / 2.0

    def _match_readd(self, side: str, px: float, qty: float,
                     ts_ms: int) -> float:
        """FIFO-match an add against pending cancels at the same price."""
        lst = self._pending[side].get(px)
        if not lst:
            return 0.0
        matched = 0.0
        keep: list[tuple[int, float]] = []
        for expire, pqty in lst:
            if expire < ts_ms:
                continue                       # expired — true withdrawal
            take = min(qty - matched, pqty)
            if take > 0:
                matched += take
                pqty -= take
            if pqty > 1e-12:
                keep.append((expire, pqty))
        if keep:
            self._pending[side][px] = keep
        else:
            self._pending[side].pop(px, None)
        return matched

    # ── message processing (WS-free for tests) ─────────────────────────
    def process(self, msg: dict, ts_ms: int | None = None) -> None:
        ts = int(ts_ms if ts_ms is not None else msg.get("E", time.time() * 1e3))
        sec = (ts // 1_000) * 1_000
        with self._lock:
            mid = self._mid()      # pre-update state — what the actor saw
            b = self._buckets.setdefault(sec, _empty_bucket())
            b["n"] += 1
            for side, key in (("bid", "b"), ("ask", "a")):
                levels = self._book[side]
                for px_s, qty_s in msg.get(key, []):
                    px, qty = float(px_s), float(qty_s)
                    prev = levels.get(px, 0.0)
                    d = qty - prev
                    if mid is not None:
                        bd = band_of(px, mid)
                        if d > 0:
                            b[f"{side}_add_{bd}"] += d
                            b[f"{side}_readd"] += self._match_readd(
                                side, px, d, ts)
                        elif d < 0:
                            b[f"{side}_cancel_{bd}"] += -d
                            self._pending[side].setdefault(px, []).append(
                                (ts + REATTACH_MS, -d))
                    else:
                        self._warmup_skipped += 1
                    if qty <= 0:
                        levels.pop(px, None)
                    else:
                        levels[px] = qty
            if mid is not None:
                b["mid"] = mid

    # ── WS plumbing ────────────────────────────────────────────────────
    def _on_message(self, _ws, raw: str) -> None:
        try:
            self.process(json.loads(raw))
        except Exception:
            logger.exception("depth_events_parse_error")

    def _on_error(self, _ws, err) -> None:
        logger.warning("depth_events_ws_error: %s", err)

    def _prune_pending(self, now_ms: int) -> None:
        for side in ("bid", "ask"):
            dead = []
            for px, lst in self._pending[side].items():
                lst[:] = [(e, q) for e, q in lst if e >= now_ms]
                if not lst:
                    dead.append(px)
            for px in dead:
                self._pending[side].pop(px, None)

    def _flush_loop(self) -> None:
        while True:
            time.sleep(FLUSH_SEC)
            try:
                self._flush()
            except Exception:
                logger.exception("depth_events_flush_failed")

    def _flush(self) -> None:
        now_ms = int(time.time() * 1_000)
        now_sec = (now_ms // 1_000) * 1_000
        with self._lock:
            done = {s: v for s, v in self._buckets.items() if s < now_sec}
            for s in done:
                self._buckets.pop(s, None)
            self._prune_pending(now_ms)
            skipped, self._warmup_skipped = self._warmup_skipped, 0
        if skipped:
            logger.info("depth_events_warmup_skipped exchange=%s n=%d",
                        self._exchange, skipped)
        if not done:
            return
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                for s, v in sorted(done.items()):
                    cur.execute(
                        """
                        INSERT INTO depth_events_1s
                            (exchange, canonical_symbol, sec_start_ms, mid_px,
                             bid_add_near, bid_add_mid, bid_add_far,
                             bid_cancel_near, bid_cancel_mid, bid_cancel_far,
                             ask_add_near, ask_add_mid, ask_add_far,
                             ask_cancel_near, ask_cancel_mid, ask_cancel_far,
                             bid_readd_qty, ask_readd_qty, update_count)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON DUPLICATE KEY UPDATE
                            mid_px = COALESCE(VALUES(mid_px), mid_px),
                            bid_add_near = bid_add_near + VALUES(bid_add_near),
                            bid_add_mid = bid_add_mid + VALUES(bid_add_mid),
                            bid_add_far = bid_add_far + VALUES(bid_add_far),
                            bid_cancel_near = bid_cancel_near + VALUES(bid_cancel_near),
                            bid_cancel_mid = bid_cancel_mid + VALUES(bid_cancel_mid),
                            bid_cancel_far = bid_cancel_far + VALUES(bid_cancel_far),
                            ask_add_near = ask_add_near + VALUES(ask_add_near),
                            ask_add_mid = ask_add_mid + VALUES(ask_add_mid),
                            ask_add_far = ask_add_far + VALUES(ask_add_far),
                            ask_cancel_near = ask_cancel_near + VALUES(ask_cancel_near),
                            ask_cancel_mid = ask_cancel_mid + VALUES(ask_cancel_mid),
                            ask_cancel_far = ask_cancel_far + VALUES(ask_cancel_far),
                            bid_readd_qty = bid_readd_qty + VALUES(bid_readd_qty),
                            ask_readd_qty = ask_readd_qty + VALUES(ask_readd_qty),
                            update_count = update_count + VALUES(update_count)
                        """,
                        (self._exchange, self._canonical, s,
                         v["mid"] or None,
                         v["bid_add_near"], v["bid_add_mid"], v["bid_add_far"],
                         v["bid_cancel_near"], v["bid_cancel_mid"], v["bid_cancel_far"],
                         v["ask_add_near"], v["ask_add_mid"], v["ask_add_far"],
                         v["ask_cancel_near"], v["ask_cancel_mid"], v["ask_cancel_far"],
                         v["bid_readd"], v["ask_readd"], v["n"]))
            conn.commit()
            logger.info("depth_events_flush exchange=%s seconds=%d",
                        self._exchange, len(done))
        finally:
            conn.close()

    def start(self) -> None:
        _ensure_schema()
        threading.Thread(target=self._flush_loop, daemon=True,
                         name="depth-events-flush").start()
        while True:   # reconnect loop — same pattern as the 1m collector
            try:
                self._ws = websocket.WebSocketApp(
                    self._ws_url, on_message=self._on_message,
                    on_error=self._on_error)
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:
                logger.exception("depth_events_ws_crashed")
            with self._lock:   # stale book after a gap — reset & re-warm
                self._book = {"bid": {}, "ask": {}}
                self._pending = {"bid": {}, "ask": {}}
            logger.warning("depth_events_ws_reconnect_in_5s")
            time.sleep(5)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["spot", "perp"], default="spot")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.market == "perp":
        DepthEvents1sCollector(ws_url=PERP_WS_URL,
                               exchange="binance_perp").start()
    else:
        DepthEvents1sCollector().start()


if __name__ == "__main__":
    main()
