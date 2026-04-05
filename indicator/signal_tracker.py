"""
Strong signal performance tracker.

Records Strong direction signals and backfills 4h outcomes automatically.
Provides cumulative win rate, average return, and signal history.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _get_db_conn():
    from shared.db import get_db_conn
    return get_db_conn()


def _ensure_table():
    """Create strong_signals table if not exists."""
    conn = _get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS `strong_signals` (
                    `id` INT AUTO_INCREMENT PRIMARY KEY,
                    `signal_time` DATETIME NOT NULL UNIQUE,
                    `direction` VARCHAR(10) NOT NULL,
                    `p_up` DOUBLE NOT NULL,
                    `mag_pred` DOUBLE NOT NULL,
                    `confidence` DOUBLE NOT NULL,
                    `entry_price` DOUBLE NOT NULL,
                    `regime` VARCHAR(30) DEFAULT '',
                    `exit_price` DOUBLE DEFAULT NULL,
                    `actual_return_4h` DOUBLE DEFAULT NULL,
                    `correct` TINYINT DEFAULT NULL,
                    `filled` TINYINT DEFAULT 0,
                    `shap_top` TEXT DEFAULT NULL,
                    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
        conn.commit()
    finally:
        conn.close()


def record_strong_signal(signal_time: datetime, direction: str, p_up: float,
                         mag_pred: float, confidence: float, entry_price: float,
                         regime: str = "", shap_json: str = ""):
    """Record a new Strong signal. Called when Strong signal fires."""
    _ensure_table()

    # Auto-add shap_top column if missing (existing tables won't have it)
    conn = _get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT shap_top FROM strong_signals LIMIT 0")
            except Exception:
                cur.execute("ALTER TABLE strong_signals ADD COLUMN `shap_top` TEXT DEFAULT NULL")
                logger.info("Added shap_top column to strong_signals")
                conn.commit()
    finally:
        conn.close()

    conn = _get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO `strong_signals`
                    (signal_time, direction, p_up, mag_pred, confidence, entry_price, regime, shap_top)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    direction=VALUES(direction), p_up=VALUES(p_up),
                    mag_pred=VALUES(mag_pred), confidence=VALUES(confidence),
                    entry_price=VALUES(entry_price), regime=VALUES(regime),
                    shap_top=VALUES(shap_top)
            """, (
                signal_time.strftime("%Y-%m-%d %H:%M:%S"),
                direction, float(p_up), float(mag_pred),
                float(confidence), float(entry_price), regime, shap_json or None,
            ))
        conn.commit()
        logger.info("Strong signal recorded: %s %s @ $%.0f", direction, signal_time, entry_price)
    finally:
        conn.close()


def backfill_outcomes(current_prices: dict[str, float] | None = None):
    """
    Backfill 4h outcomes for unfilled signals.

    Called every update cycle. Checks if 4h has passed since signal,
    then fills exit_price, actual_return_4h, correct.

    current_prices: optional dict with signal_time_str → close price at +4h
    """
    _ensure_table()
    conn = _get_db_conn()
    try:
        with conn.cursor() as cur:
            # Find unfilled signals older than 4h
            cur.execute("""
                SELECT id, signal_time, direction, entry_price
                FROM `strong_signals`
                WHERE filled = 0
                  AND signal_time <= DATE_SUB(NOW(), INTERVAL 4 HOUR)
            """)
            unfilled = cur.fetchall()

            if not unfilled:
                return

            for row in unfilled:
                sig_id = row["id"]
                sig_time = row["signal_time"]
                direction = row["direction"]
                entry = row["entry_price"]

                # Find the close price 4h after signal from indicator_history
                cur.execute("""
                    SELECT `close` FROM `indicator_history`
                    WHERE dt >= DATE_ADD(%s, INTERVAL 4 HOUR)
                    ORDER BY dt ASC
                    LIMIT 1
                """, (sig_time,))
                exit_row = cur.fetchone()

                if not exit_row:
                    continue  # 4h bar not yet available

                exit_price = float(exit_row["close"])
                actual_ret = (exit_price - entry) / entry
                correct = 1 if (
                    (direction == "UP" and actual_ret > 0) or
                    (direction == "DOWN" and actual_ret < 0)
                ) else 0

                cur.execute("""
                    UPDATE `strong_signals`
                    SET exit_price = %s, actual_return_4h = %s, correct = %s, filled = 1
                    WHERE id = %s
                """, (exit_price, actual_ret, correct, sig_id))

                logger.info("Strong signal outcome: %s %s entry=$%.0f exit=$%.0f ret=%.2f%% %s",
                            direction, sig_time, entry, exit_price, actual_ret * 100,
                            "CORRECT" if correct else "WRONG")

        conn.commit()
    finally:
        conn.close()


def get_performance_report() -> str:
    """Generate Strong signal performance report for Telegram."""
    _ensure_table()
    conn = _get_db_conn()
    try:
        with conn.cursor() as cur:
            # Overall stats
            cur.execute("""
                SELECT COUNT(*) as total,
                       SUM(filled) as filled,
                       SUM(correct) as wins,
                       AVG(CASE WHEN filled=1 THEN actual_return_4h END) as avg_ret,
                       MAX(CASE WHEN filled=1 THEN actual_return_4h END) as best_ret,
                       MIN(CASE WHEN filled=1 THEN actual_return_4h END) as worst_ret
                FROM `strong_signals`
            """)
            stats = cur.fetchone()

            total = int(stats["total"] or 0)
            filled = int(stats["filled"] or 0)
            wins = int(stats["wins"] or 0)
            pending = total - filled

            if total == 0:
                return "<b>📊 Strong Signal Performance</b>\n\n尚無 Strong 信號紀錄"

            lines = ["<b>📊 Strong Signal Performance</b>\n"]

            # Win rate
            if filled > 0:
                win_rate = wins / filled * 100
                avg_ret = float(stats["avg_ret"] or 0)
                best_ret = float(stats["best_ret"] or 0)
                worst_ret = float(stats["worst_ret"] or 0)

                lines.append(f"<b>累積績效</b> ({filled} 筆已結算)")
                lines.append(f"  勝率: {win_rate:.1f}% ({wins}W / {filled - wins}L)")
                lines.append(f"  平均報酬: {avg_ret * 100:+.2f}%")
                lines.append(f"  最佳: {best_ret * 100:+.2f}%")
                lines.append(f"  最差: {worst_ret * 100:+.2f}%")
            else:
                lines.append("<b>累積績效</b>")
                lines.append("  尚無已結算信號")

            if pending > 0:
                lines.append(f"\n  等待結算: {pending} 筆")

            # Per-direction breakdown
            cur.execute("""
                SELECT direction,
                       COUNT(*) as cnt,
                       SUM(correct) as wins,
                       AVG(actual_return_4h) as avg_ret
                FROM `strong_signals`
                WHERE filled = 1
                GROUP BY direction
            """)
            dir_rows = cur.fetchall()
            if dir_rows:
                lines.append(f"\n<b>方向拆解</b>")
                for dr in dir_rows:
                    d = dr["direction"]
                    cnt = int(dr["cnt"])
                    w = int(dr["wins"] or 0)
                    ar = float(dr["avg_ret"] or 0)
                    icon = "🟢" if d == "UP" else "🔴"
                    lines.append(f"  {icon} {d}: {w}/{cnt} ({w/cnt*100:.0f}%) avg {ar*100:+.2f}%")

            # Recent signals (last 10)
            cur.execute("""
                SELECT signal_time, direction, p_up, entry_price,
                       exit_price, actual_return_4h, correct, filled
                FROM `strong_signals`
                ORDER BY signal_time DESC
                LIMIT 10
            """)
            recent = cur.fetchall()
            if recent:
                lines.append(f"\n<b>最近信號</b>")
                for r in recent:
                    t = str(r["signal_time"])[5:16]  # MM-DD HH:MM
                    d = r["direction"]
                    icon = "🟢▲" if d == "UP" else "🔴▼"
                    entry = float(r["entry_price"])

                    if r["filled"]:
                        ret = float(r["actual_return_4h"])
                        result = "✅" if r["correct"] else "❌"
                        lines.append(f"  {icon} {t} ${entry:,.0f} → {ret*100:+.2f}% {result}")
                    else:
                        lines.append(f"  {icon} {t} ${entry:,.0f} → ⏳ 等待中")

            return "\n".join(lines)
    finally:
        conn.close()
