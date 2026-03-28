"""
MySQL connection management for market data layer.

Delegates to shared.db for connection config (env / .env / config.json).
"""

import logging
from shared.db import get_db_conn

logger = logging.getLogger(__name__)


def get_conn():
    """Get a new MySQL connection (same as shared.db.get_db_conn)."""
    return get_db_conn()


def run_migration(sql_path: str):
    """Execute a .sql migration file. Each statement runs independently."""
    with open(sql_path, "r", encoding="utf-8") as f:
        sql = f.read()

    # Strip comment-only lines before splitting
    lines = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        lines.append(line)
    cleaned = "\n".join(lines)

    conn = get_conn()
    try:
        with conn.cursor() as cursor:
            for statement in cleaned.split(";"):
                # Remove leading/trailing whitespace
                statement = statement.strip()
                if not statement:
                    continue
                try:
                    cursor.execute(statement)
                except Exception as e:
                    err_msg = str(e).lower()
                    if "duplicate" in err_msg or "already exists" in err_msg:
                        logger.debug("Skipping (already applied): %s", e)
                    else:
                        raise
        logger.info("Migration applied: %s", sql_path)
    finally:
        conn.close()
