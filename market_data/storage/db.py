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
    """Execute a .sql migration file."""
    with open(sql_path, "r", encoding="utf-8") as f:
        sql = f.read()

    conn = get_conn()
    try:
        with conn.cursor() as cursor:
            for statement in sql.split(";"):
                statement = statement.strip()
                if statement:
                    cursor.execute(statement)
        logger.info("Migration applied: %s", sql_path)
    finally:
        conn.close()
