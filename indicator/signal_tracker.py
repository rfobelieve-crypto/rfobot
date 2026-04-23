"""
Signal performance tracker — tracks Strong + Moderate directional signals.

Records signals with confidence >= 65 (Moderate+) and auto-backfills
4h outcomes from indicator_history. Provides win rate reports by tier.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

TZ_TPE = timezone(timedelta(hours=8))

TABLE = "tracked_signals"


def _get_db_conn():
    from shared.db import get_db_conn
    return get_db_conn()


def _ensure_table():
    """Create tracked_signals table if not exists. Migrate from strong_signals if needed."""
    conn = _get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS `{TABLE}` (
                    `id` INT AUTO_INCREMENT PRIMARY KEY,
                    `signal_time` DATETIME NOT NULL,
                    `direction` VARCHAR(10) NOT NULL,
                    `strength` VARCHAR(10) NOT NULL DEFAULT 'Strong',
                    `p_up` DOUBLE NOT NULL,
                    `mag_pred` DOUBLE NOT NULL,
                    `confidence` DOUBLE NOT NULL,
                    `entry_price` DOUBLE NOT NULL,
                    `regime` VARCHAR(30) DEFAULT '',
                    `mag_pct_200` DOUBLE DEFAULT NULL,
                    `exit_price` DOUBLE DEFAULT NULL,
                    `actual_return_4h` DOUBLE DEFAULT NULL,
                    `correct` TINYINT DEFAULT NULL,
                    `filled` TINYINT DEFAULT 0,
                    `shap_top` TEXT DEFAULT NULL,
                    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY `uq_time_strength` (`signal_time`, `strength`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            # Additive migration: add mag_pct_200 column if pre-existing
            # table was created before this feature. Safe to run repeatedly.
            cur.execute("""
                SELECT COUNT(*) AS n FROM information_schema.columns
                WHERE table_schema = DATABASE() AND table_name = %s
                  AND column_name = 'mag_pct_200'
            """, (TABLE,))
            if int(cur.fetchone()["n"]) == 0:
                cur.execute(f"ALTER TABLE `{TABLE}` ADD COLUMN `mag_pct_200` DOUBLE DEFAULT NULL")
                logger.info("Added mag_pct_200 column to %s", TABLE)
            # Migrate old strong_signals data if exists and tracked_signals is empty
            cur.execute(f"SELECT COUNT(*) as cnt FROM `{TABLE}`")
            if cur.fetchone()["cnt"] == 0:
                try:
                    cur.execute("SELECT COUNT(*) as cnt FROM `strong_signals`")
                    old_cnt = cur.fetchone()["cnt"]
                    if old_cnt > 0:
                        cur.execute(f"""
                            INSERT IGNORE INTO `{TABLE}`
                                (signal_time, direction, strength, p_up, mag_pred,
                                 confidence, entry_price, regime, exit_price,
                                 actual_return_4h, correct, filled, shap_top, created_at)
                            SELECT signal_time, direction, 'Strong', p_up, mag_pred,
                                   confidence, entry_price, regime, exit_price,
                                   actual_return_4h, correct, filled, shap_top, created_at
                            FROM `strong_signals`
                        """)
                        logger.info("Migrated %d rows from strong_signals to %s", old_cnt, TABLE)
                except Exception:
                    pass  # strong_signals doesn't exist, that's fine
        conn.commit()
    finally:
        conn.close()


def record_signal(signal_time: datetime, direction: str, strength: str,
                  p_up: float, mag_pred: float, confidence: float,
                  entry_price: float, regime: str = "", shap_json: str = "",
                  mag_pct_200: float = None):
    """Record a Strong or Moderate directional signal.

    mag_pct_200: rolling 200-bar percentile of |mag_pred| at signal time.
    Used downstream to evaluate dual-gate (direction+mag) performance
    tiers. NULL for pre-feature historical signals.
    """
    if direction not in ("UP", "DOWN"):
        return
    if strength not in ("Strong", "Moderate"):
        return

    _ensure_table()
    conn = _get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO `{TABLE}`
                    (signal_time, direction, strength, p_up, mag_pred,
                     confidence, entry_price, regime, shap_top, mag_pct_200)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    direction=VALUES(direction), p_up=VALUES(p_up),
                    mag_pred=VALUES(mag_pred), confidence=VALUES(confidence),
                    entry_price=VALUES(entry_price), regime=VALUES(regime),
                    shap_top=VALUES(shap_top), mag_pct_200=VALUES(mag_pct_200)
            """, (
                signal_time.strftime("%Y-%m-%d %H:%M:%S"),
                direction, strength, float(p_up), float(mag_pred),
                float(confidence), float(entry_price), regime, shap_json or None,
                float(mag_pct_200) if mag_pct_200 is not None else None,
            ))
        conn.commit()
        logger.info("%s signal recorded: %s %s @ $%.0f conf=%.0f mag_p=%s",
                    strength, direction, signal_time, entry_price, confidence,
                    f"{mag_pct_200:.0f}" if mag_pct_200 is not None else "?")
    finally:
        conn.close()


# Backward compat alias
def record_strong_signal(signal_time, direction, p_up, mag_pred,
                         confidence, entry_price, regime="", shap_json="",
                         mag_pct_200=None):
    record_signal(signal_time, direction, "Strong", p_up, mag_pred,
                  confidence, entry_price, regime, shap_json, mag_pct_200)


def backfill_outcomes():
    """
    Backfill 4h TWAP outcomes for unfilled signals.

    Called every update cycle. For each unfilled signal older than 4h,
    looks up close prices at +1h, +2h, +3h, +4h from indicator_history,
    computes TWAP return = mean(close[t+1..t+4]) / entry - 1.
    This matches the training target (y_path_ret_4h).
    """
    _ensure_table()
    conn = _get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, signal_time, direction, entry_price, strength
                FROM `{TABLE}`
                WHERE filled = 0
                  AND signal_time <= DATE_SUB(NOW(), INTERVAL 4 HOUR)
            """)
            unfilled = cur.fetchall()

            if not unfilled:
                return

            filled_count = 0
            for row in unfilled:
                sig_id = row["id"]
                sig_time = row["signal_time"]
                direction = row["direction"]
                entry = row["entry_price"]

                # Fetch close prices at +1h, +2h, +3h, +4h for TWAP
                cur.execute("""
                    SELECT `close` FROM `indicator_history`
                    WHERE dt > %s AND dt <= DATE_ADD(%s, INTERVAL 4 HOUR)
                    ORDER BY dt ASC
                """, (sig_time, sig_time))
                path_rows = cur.fetchall()

                if len(path_rows) < 4:
                    # Not enough bars yet — need all 4
                    from datetime import datetime as dt_cls
                    age_h = (dt_cls.now(timezone.utc) - sig_time.replace(tzinfo=timezone.utc)).total_seconds() / 3600
                    if age_h > 8:
                        logger.warning("Signal %s at %s: only %d/4 bars after %.0fh, skipping",
                                       row["strength"], sig_time, len(path_rows), age_h)
                    continue

                # TWAP: mean of the 4 close prices
                path_closes = [float(r["close"]) for r in path_rows[:4]]
                twap = sum(path_closes) / len(path_closes)
                actual_ret = (twap - entry) / entry
                exit_price = path_closes[-1]  # store last close for reference
                correct = 1 if (
                    (direction == "UP" and actual_ret > 0) or
                    (direction == "DOWN" and actual_ret < 0)
                ) else 0

                cur.execute(f"""
                    UPDATE `{TABLE}`
                    SET exit_price = %s, actual_return_4h = %s, correct = %s, filled = 1
                    WHERE id = %s
                """, (exit_price, actual_ret, correct, sig_id))

                logger.info("%s outcome: %s %s entry=$%.0f twap=$%.0f ret=%.2f%% %s",
                            row["strength"], direction, sig_time,
                            entry, twap, actual_ret * 100,
                            "CORRECT" if correct else "WRONG")
                filled_count += 1

        conn.commit()
        if filled_count:
            logger.info("Backfilled %d signal outcomes (TWAP)", filled_count)
    finally:
        conn.close()


def get_performance_report() -> str:
    """Generate signal performance report for Telegram (Strong only)."""
    _ensure_table()
    conn = _get_db_conn()
    try:
        with conn.cursor() as cur:
            lines = ["<b>📊 Signal Performance</b>\n"]

            # Per-tier stats
            for tier in ["Strong", "Moderate"]:
                tier_icon = "🔥" if tier == "Strong" else "📊"
                cur.execute(f"""
                    SELECT COUNT(*) as total,
                           SUM(filled) as filled,
                           SUM(correct) as wins,
                           AVG(CASE WHEN filled=1 THEN actual_return_4h END) as avg_ret
                    FROM `{TABLE}` WHERE strength = %s
                """, (tier,))
                s = cur.fetchone()
                total = int(s["total"] or 0)
                filled = int(s["filled"] or 0)
                wins = int(s["wins"] or 0)
                if filled > 0:
                    wr = wins / filled * 100
                    avg_r = float(s["avg_ret"] or 0) * 100
                    lines.append(f"{tier_icon} <b>{tier}</b> ({filled} 結算 / {total} 總計)")
                    lines.append(f"  勝率: {wr:.1f}% ({wins}W/{filled-wins}L) avg={avg_r:+.2f}%")
                elif total > 0:
                    lines.append(f"{tier_icon} <b>{tier}</b> ({total} 筆，尚無結算)")
                else:
                    lines.append(f"{tier_icon} <b>{tier}</b>: 尚無信號")

                # Per-direction for this tier
                cur.execute(f"""
                    SELECT direction,
                           COUNT(*) as cnt, SUM(correct) as wins,
                           AVG(actual_return_4h) as avg_ret
                    FROM `{TABLE}` WHERE filled = 1 AND strength = %s
                    GROUP BY direction ORDER BY direction
                """, (tier,))
                dir_rows = cur.fetchall()
                for dr in dir_rows:
                    d = dr["direction"]
                    cnt, w = int(dr["cnt"]), int(dr["wins"] or 0)
                    ar = float(dr["avg_ret"] or 0)
                    icon = "🟢" if d == "UP" else "🔴"
                    lines.append(f"  {icon} {d}: {w}/{cnt} ({w/cnt*100:.0f}%) avg={ar*100:+.2f}%")
                lines.append("")  # blank line between tiers

            # Recent signals
            cur.execute(f"""
                SELECT signal_time, direction, strength, confidence, entry_price,
                       actual_return_4h, correct, filled
                FROM `{TABLE}`
                WHERE signal_time >= DATE_SUB(NOW(), INTERVAL 3 DAY)
                  AND strength IN ('Strong', 'Moderate')
                ORDER BY signal_time DESC LIMIT 10
            """)
            recent = cur.fetchall()
            if recent:
                lines.append("<b>最近信號</b>")
                for r in recent:
                    sig_utc = r["signal_time"]
                    if hasattr(sig_utc, 'replace'):
                        sig_utc = sig_utc.replace(tzinfo=timezone.utc)
                    sig_local = sig_utc.astimezone(TZ_TPE)
                    t = sig_local.strftime("%m-%d %H:%M")
                    d = r["direction"]
                    tier_tag = "S" if r["strength"] == "Strong" else "M"
                    icon = "🟢▲" if d == "UP" else "🔴▼"
                    entry = float(r["entry_price"])
                    conf = float(r["confidence"])
                    if r["filled"]:
                        ret = float(r["actual_return_4h"])
                        mark = "✅" if r["correct"] else "❌"
                        lines.append(f"  {icon}[{tier_tag}] {t} ${entry:,.0f} c={conf:.0f} → {ret*100:+.2f}% {mark}")
                    else:
                        lines.append(f"  {icon}[{tier_tag}] {t} ${entry:,.0f} c={conf:.0f} → ⏳")

            return "\n".join(lines)
    finally:
        conn.close()
