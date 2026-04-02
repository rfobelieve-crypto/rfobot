"""
Data Collector — 收集所有回測所需資料並備份到 D 槽。

儲存結構:
  D:/flowbot_data/
  ├── raw/
  │   ├── aggtrades/binance/          ← aggTrade 日 CSV (最大宗)
  │   ├── aggtrades/okx/             ← OKX aggTrade
  │   ├── klines/binance/            ← 15m K 線 CSV
  │   ├── coinglass/                 ← Coinglass API parquet
  │   ├── metrics/                   ← Binance futures metrics
  │   └── funding_rate/              ← Funding rate CSV
  ├── processed/
  │   ├── BTC_USD_1h_enhanced.parquet
  │   ├── BTC_USD_indicator_4h.parquet
  │   └── BTC_USD_signals_*.parquet
  └── manifest.json                  ← 收集記錄 (時間、檔案數、大小)

模式:
  --full          完整備份 (含 33GB aggTrades)
  --light         輕量備份 (跳過 aggTrades 原始檔，只保留處理後資料)
  --update        增量更新 (只複製新增或修改的檔案)
  --coinglass     只更新 Coinglass 資料 (每日排程用)

Usage:
    python -m market_data.backfill.data_collector --light
    python -m market_data.backfill.data_collector --full
    python -m market_data.backfill.data_collector --update
    python -m market_data.backfill.data_collector --coinglass
"""
from __future__ import annotations

import os
import json
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[2]
SRC_RAW  = ROOT / "market_data" / "raw_data"
SRC_ML   = ROOT / "research" / "ml_data"
DEST_ROOT = Path(os.environ.get("FLOWBOT_BACKUP_DIR", "D:/flowbot_data"))


def _ensure_dirs():
    """Create destination directory structure."""
    for sub in ["raw/aggtrades/binance", "raw/aggtrades/okx",
                "raw/klines/binance", "raw/coinglass",
                "raw/metrics", "raw/funding_rate", "processed"]:
        (DEST_ROOT / sub).mkdir(parents=True, exist_ok=True)


def _copy_if_newer(src: Path, dst: Path) -> bool:
    """Copy file only if source is newer or destination doesn't exist."""
    if not src.exists():
        return False
    if dst.exists():
        if src.stat().st_mtime <= dst.stat().st_mtime:
            return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _copy_dir_incremental(src_dir: Path, dst_dir: Path,
                           pattern: str = "*") -> tuple[int, int, int]:
    """Copy directory contents incrementally. Returns (copied, skipped, total_bytes)."""
    copied = skipped = total_bytes = 0
    if not src_dir.exists():
        return copied, skipped, total_bytes

    for f in sorted(src_dir.rglob(pattern)):
        if not f.is_file():
            continue
        rel = f.relative_to(src_dir)
        dst = dst_dir / rel
        if _copy_if_newer(f, dst):
            copied += 1
            total_bytes += f.stat().st_size
        else:
            skipped += 1
    return copied, skipped, total_bytes


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.2f} GB"


def collect_coinglass() -> dict:
    """Collect Coinglass API data (parquet files)."""
    logger.info("=== Coinglass data ===")
    src = SRC_RAW / "coinglass"
    dst = DEST_ROOT / "raw" / "coinglass"
    c, s, b = _copy_dir_incremental(src, dst, "*.parquet")
    logger.info("  Coinglass: %d copied, %d skipped, %s", c, s, _format_size(b))
    return {"copied": c, "skipped": s, "bytes": b}


def collect_metrics() -> dict:
    """Collect Binance futures metrics."""
    logger.info("=== Binance metrics ===")
    src = SRC_RAW / "metrics"
    dst = DEST_ROOT / "raw" / "metrics"
    c, s, b = _copy_dir_incremental(src, dst, "*.parquet")
    logger.info("  Metrics: %d copied, %d skipped, %s", c, s, _format_size(b))
    return {"copied": c, "skipped": s, "bytes": b}


def collect_funding() -> dict:
    """Collect funding rate data."""
    logger.info("=== Funding rate ===")
    src = SRC_RAW / "funding rate"
    dst = DEST_ROOT / "raw" / "funding_rate"
    c, s, b = _copy_dir_incremental(src, dst)
    logger.info("  Funding: %d copied, %d skipped, %s", c, s, _format_size(b))
    return {"copied": c, "skipped": s, "bytes": b}


def collect_klines() -> dict:
    """Collect 15m kline CSVs."""
    logger.info("=== Klines ===")
    src = SRC_RAW / "klines" / "binance"
    dst = DEST_ROOT / "raw" / "klines" / "binance"
    c, s, b = _copy_dir_incremental(src, dst, "*.csv")
    logger.info("  Klines: %d copied, %d skipped, %s", c, s, _format_size(b))
    return {"copied": c, "skipped": s, "bytes": b}


def collect_aggtrades() -> dict:
    """Collect aggTrade CSVs (large! ~33GB)."""
    logger.info("=== aggTrades (this may take a while) ===")
    stats = {"copied": 0, "skipped": 0, "bytes": 0}

    for exchange in ["binance", "okx"]:
        src = SRC_RAW / "aggtrades" / exchange
        dst = DEST_ROOT / "raw" / "aggtrades" / exchange
        if not src.exists():
            continue
        c, s, b = _copy_dir_incremental(src, dst, "*.csv")
        logger.info("  aggTrades/%s: %d copied, %d skipped, %s",
                     exchange, c, s, _format_size(b))
        stats["copied"] += c
        stats["skipped"] += s
        stats["bytes"] += b

    return stats


def collect_processed() -> dict:
    """Collect processed ML parquet files."""
    logger.info("=== Processed ML data ===")
    dst_dir = DEST_ROOT / "processed"
    c = b = s = 0

    for f in sorted(SRC_ML.glob("*.parquet")):
        dst = dst_dir / f.name
        if _copy_if_newer(f, dst):
            c += 1
            b += f.stat().st_size
        else:
            s += 1

    logger.info("  Processed: %d copied, %d skipped, %s", c, s, _format_size(b))
    return {"copied": c, "skipped": s, "bytes": b}


def write_manifest(results: dict):
    """Write collection manifest with metadata."""
    manifest = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "source": str(ROOT),
        "destination": str(DEST_ROOT),
        "results": {},
    }

    total_copied = total_bytes = 0
    for name, stats in results.items():
        manifest["results"][name] = stats
        total_copied += stats.get("copied", 0)
        total_bytes += stats.get("bytes", 0)

    manifest["total_files_copied"] = total_copied
    manifest["total_bytes_copied"] = total_bytes
    manifest["total_size"] = _format_size(total_bytes)

    # Calculate destination total size
    dest_size = sum(f.stat().st_size for f in DEST_ROOT.rglob("*") if f.is_file())
    manifest["destination_total_size"] = _format_size(dest_size)

    manifest_path = DEST_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info("\nManifest saved: %s", manifest_path)
    logger.info("Total copied: %d files, %s", total_copied, _format_size(total_bytes))
    logger.info("Destination total: %s", manifest["destination_total_size"])


def run(mode: str = "light"):
    _ensure_dirs()
    results = {}

    if mode == "coinglass":
        results["coinglass"] = collect_coinglass()
    elif mode == "light":
        results["coinglass"] = collect_coinglass()
        results["metrics"] = collect_metrics()
        results["funding"] = collect_funding()
        results["klines"] = collect_klines()
        results["processed"] = collect_processed()
    elif mode in ("full", "update"):
        results["coinglass"] = collect_coinglass()
        results["metrics"] = collect_metrics()
        results["funding"] = collect_funding()
        results["klines"] = collect_klines()
        results["aggtrades"] = collect_aggtrades()
        results["processed"] = collect_processed()

    write_manifest(results)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="收集回測資料到 D 槽")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_const", dest="mode",
                       const="full", help="完整備份 (含 aggTrades ~33GB)")
    group.add_argument("--light", action="store_const", dest="mode",
                       const="light", help="輕量備份 (跳過 aggTrades)")
    group.add_argument("--update", action="store_const", dest="mode",
                       const="update", help="增量更新所有資料")
    group.add_argument("--coinglass", action="store_const", dest="mode",
                       const="coinglass", help="只更新 Coinglass")
    ap.set_defaults(mode="light")
    args = ap.parse_args()

    logger.info("Data Collector — mode: %s", args.mode)
    logger.info("Source: %s", ROOT)
    logger.info("Destination: %s", DEST_ROOT)
    run(mode=args.mode)


if __name__ == "__main__":
    main()
