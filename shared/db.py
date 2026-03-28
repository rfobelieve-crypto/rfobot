"""
Shared MySQL connection helper with connection pooling.

Resolution order:
1. Environment variables (Railway / cloud deploy)
2. .env file in project root (local development)
3. config.json in project root (legacy fallback)

Used by both BTC_perp_data.py (main bot) and market_data layer.
"""

import os
import json
import logging
import threading
import queue

import pymysql

logger = logging.getLogger(__name__)

# Project root = directory containing this file's parent
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_config_loaded = False
_db_config: dict = {}


def _load_dotenv():
    """Parse .env file (KEY=VALUE lines) without requiring python-dotenv."""
    env_path = os.path.join(_PROJECT_ROOT, ".env")
    if not os.path.exists(env_path):
        return {}

    result = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            result[key] = value
    return result


def _load_config_json():
    """Read MySQL settings from config.json if present."""
    config_path = os.path.join(_PROJECT_ROOT, "config.json")
    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "MYSQL_HOST": data.get("mysql_host", data.get("MYSQL_HOST", "")),
        "MYSQL_PORT": str(data.get("mysql_port", data.get("MYSQL_PORT", "3306"))),
        "MYSQL_USER": data.get("mysql_user", data.get("MYSQL_USER", "")),
        "MYSQL_PASSWORD": data.get("mysql_password", data.get("MYSQL_PASSWORD", "")),
        "MYSQL_DB": data.get("mysql_db", data.get("MYSQL_DB", "")),
    }


def _resolve_config():
    """Resolve MySQL config: env vars > .env > config.json."""
    global _config_loaded, _db_config

    if _config_loaded:
        return

    # Layer 1: environment variables (always highest priority)
    env_host = os.getenv("MYSQL_HOST", "")

    if env_host:
        _db_config = {
            "host": env_host,
            "port": int(os.getenv("MYSQL_PORT", "3306")),
            "user": os.getenv("MYSQL_USER", ""),
            "password": os.getenv("MYSQL_PASSWORD", ""),
            "database": os.getenv("MYSQL_DB", ""),
            "source": "environment",
        }
        _config_loaded = True
        return

    # Layer 2: .env file
    dotenv = _load_dotenv()
    if dotenv.get("MYSQL_HOST"):
        _db_config = {
            "host": dotenv["MYSQL_HOST"],
            "port": int(dotenv.get("MYSQL_PORT", "3306")),
            "user": dotenv.get("MYSQL_USER", ""),
            "password": dotenv.get("MYSQL_PASSWORD", ""),
            "database": dotenv.get("MYSQL_DB", ""),
            "source": ".env",
        }
        # Also inject into os.environ so other modules that read env vars directly
        # (e.g. legacy code) will pick them up too.
        for key in ("MYSQL_HOST", "MYSQL_PORT", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DB"):
            if key in dotenv and not os.getenv(key):
                os.environ[key] = dotenv[key]
        _config_loaded = True
        return

    # Layer 3: config.json
    cfg = _load_config_json()
    if cfg.get("MYSQL_HOST"):
        _db_config = {
            "host": cfg["MYSQL_HOST"],
            "port": int(cfg.get("MYSQL_PORT", "3306")),
            "user": cfg.get("MYSQL_USER", ""),
            "password": cfg.get("MYSQL_PASSWORD", ""),
            "database": cfg.get("MYSQL_DB", ""),
            "source": "config.json",
        }
        for key in ("MYSQL_HOST", "MYSQL_PORT", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DB"):
            if cfg.get(key) and not os.getenv(key):
                os.environ[key] = cfg[key]
        _config_loaded = True
        return

    _config_loaded = True  # mark loaded even if empty, to avoid re-loading


def _check_config():
    """Validate required fields and raise a clear error if missing."""
    _resolve_config()

    missing = []
    if not _db_config.get("host"):
        missing.append("MYSQL_HOST")
    if not _db_config.get("user"):
        missing.append("MYSQL_USER")
    if not _db_config.get("database"):
        missing.append("MYSQL_DB")

    if missing:
        source_hint = (
            "Set them as environment variables (Railway), "
            "or create a .env file in the project root, "
            "or add them to config.json."
        )
        raise ValueError(
            f"MySQL configuration incomplete — missing: {', '.join(missing)}. "
            f"{source_hint}"
        )


def _create_conn():
    """Create a fresh pymysql connection."""
    return pymysql.connect(
        host=_db_config["host"],
        port=_db_config["port"],
        user=_db_config["user"],
        password=_db_config["password"],
        database=_db_config["database"],
        charset="utf8mb4",
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
        ssl={"ssl": {}},
    )


# ── Connection Pool ──────────────────────────────────────
_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
_pool: queue.Queue | None = None
_pool_lock = threading.Lock()


def _get_pool() -> queue.Queue:
    """Lazy-init the connection pool."""
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is not None:
            return _pool
        _check_config()
        _pool = queue.Queue(maxsize=_POOL_SIZE)
        # Pre-fill pool
        for _ in range(_POOL_SIZE):
            try:
                _pool.put_nowait(_create_conn())
            except Exception:
                logger.debug("Pre-fill connection failed, will create on demand")
        logger.info("DB pool initialized (size=%d, filled=%d)",
                     _POOL_SIZE, _pool.qsize())
        return _pool


class PooledConnection:
    """Wrapper that returns the connection to the pool on close()."""

    def __init__(self, conn, pool: queue.Queue):
        self._conn = conn
        self._pool = pool

    def cursor(self, *args, **kwargs):
        return self._conn.cursor(*args, **kwargs)

    def close(self):
        """Return connection to pool instead of closing it."""
        try:
            self._conn.ping(reconnect=True)
            self._pool.put_nowait(self._conn)
        except (queue.Full, Exception):
            # Pool full or connection broken — actually close
            try:
                self._conn.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


def get_db_conn():
    """
    Get a pooled MySQL connection.

    Returns a PooledConnection with autocommit=True and DictCursor.
    Call .close() to return it to the pool (don't forget finally blocks).
    """
    pool = _get_pool()

    # Try to get from pool (non-blocking)
    try:
        conn = pool.get_nowait()
        # Verify the connection is alive
        try:
            conn.ping(reconnect=True)
            return PooledConnection(conn, pool)
        except Exception:
            # Dead connection, create new
            try:
                conn.close()
            except Exception:
                pass
    except queue.Empty:
        pass

    # Pool empty or connection dead — create new
    _check_config()
    conn = _create_conn()
    return PooledConnection(conn, pool)


def get_db_info() -> dict:
    """Return non-sensitive DB connection info for status reports."""
    _resolve_config()
    return {
        "host": _db_config.get("host", "NOT SET"),
        "port": _db_config.get("port", 3306),
        "database": _db_config.get("database", "NOT SET"),
        "user": _db_config.get("user", "NOT SET"),
        "source": _db_config.get("source", "none"),
    }
