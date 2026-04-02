"""
Pipeline 健康檢查腳本 — 下載完資料、跑完 pipeline 後手動執行。

Usage:
    python check_pipeline.py           # 完整檢查
    python check_pipeline.py --raw     # 只看原始檔案
    python check_pipeline.py --db      # 只看 DB
    python check_pipeline.py --ml      # 只看 Parquet 輸出
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OK   = "✓"
WARN = "⚠"
FAIL = "✗"

def _c(sym, msg): print(f"  {sym}  {msg}")
def _h(title): print(f"\n{'─'*50}\n  {title}\n{'─'*50}")


# ─── 1. 原始檔案檢查 ──────────────────────────────────────────────────────────

def check_raw_files():
    _h("RAW FILES")

    klines_dir = ROOT / "market_data" / "raw_data" / "klines"
    agg_binance = ROOT / "market_data" / "raw_data" / "aggtrades" / "binance"
    agg_okx     = ROOT / "market_data" / "raw_data" / "aggtrades" / "okx"

    # klines
    kfiles = sorted(glob.glob(str(klines_dir / "BTCUSDT-1m-*.csv")))
    if kfiles:
        sizes = [os.path.getsize(f) // 1024 for f in kfiles]
        _c(OK, f"klines: {len(kfiles)} 個檔  ({sum(sizes)//1024} MB total)")
        _c("  ", f"  最早: {Path(kfiles[0]).stem.split('-1m-')[1]}")
        _c("  ", f"  最新: {Path(kfiles[-1]).stem.split('-1m-')[1]}")
    else:
        _c(FAIL, "klines: 找不到任何 BTCUSDT-1m-*.csv")

    # binance aggtrades
    bagg = sorted(glob.glob(str(agg_binance / "**" / "*.csv"), recursive=True))
    if bagg:
        total_mb = sum(os.path.getsize(f) for f in bagg) // 1024 // 1024
        _c(OK, f"Binance aggtrades: {len(bagg)} 個檔  ({total_mb} MB)")
        # 快速看第一筆確認 USDT-M
        import csv
        with open(bagg[0]) as fp:
            row = next(csv.DictReader(fp))
        qty = float(row.get("quantity", 0))
        if qty < 10:
            _c(OK, f"  格式: USDT-M 確認 (quantity={qty} BTC)")
        else:
            _c(WARN, f"  格式警告: quantity={qty}，可能是 Coin-M contract 數量")
    else:
        _c(WARN, "Binance aggtrades: 無本地檔案（pipeline 會自動下載）")

    # okx aggtrades
    oakg = sorted(glob.glob(str(agg_okx / "**" / "*.csv"), recursive=True))
    if oakg:
        total_mb = sum(os.path.getsize(f) for f in oakg) // 1024 // 1024
        _c(OK, f"OKX aggtrades: {len(oakg)} 個檔  ({total_mb} MB)")
    else:
        _c(WARN, "OKX aggtrades: 無本地檔案")


# ─── 2. DB 檢查 ───────────────────────────────────────────────────────────────

def check_db():
    _h("DATABASE")
    try:
        from shared.db import get_db_conn
    except Exception as e:
        _c(FAIL, f"無法載入 shared.db: {e}")
        return

    try:
        conn = get_db_conn()
    except Exception as e:
        _c(FAIL, f"DB 連線失敗: {e}")
        return

    _c(OK, "DB 連線成功")

    with conn.cursor() as cur:

        # 各表 row count
        tables = [
            "ohlcv_1m",
            "flow_bars_1m_ml",
            "features_5m",
            "features_15m",
            "features_1h",
            "oi_snapshots",
            "funding_rates",
        ]
        for t in tables:
            try:
                cur.execute(f"SELECT COUNT(*) AS n FROM `{t}`")
                n = cur.fetchone()["n"]
                sym = OK if n > 0 else WARN
                _c(sym, f"{t:<25} {n:>10,} rows")
            except Exception:
                _c(FAIL, f"{t:<25} 表不存在")

        # ohlcv_1m 日期範圍
        try:
            cur.execute("""
                SELECT MIN(ts_open) AS mn, MAX(ts_open) AS mx,
                       COUNT(DISTINCT DATE(FROM_UNIXTIME(ts_open/1000))) AS days
                FROM ohlcv_1m WHERE symbol='BTC-USD'
            """)
            r = cur.fetchone()
            if r and r["mn"]:
                dt_min = datetime.fromtimestamp(r["mn"]/1000, tz=timezone.utc).strftime("%Y-%m-%d")
                dt_max = datetime.fromtimestamp(r["mx"]/1000, tz=timezone.utc).strftime("%Y-%m-%d")
                _c(OK, f"ohlcv_1m 日期範圍: {dt_min} ~ {dt_max}  ({r['days']} 天)")
        except Exception as e:
            _c(WARN, f"ohlcv_1m 日期查詢失敗: {e}")

        # flow_bars_1m_ml 日期範圍
        try:
            cur.execute("""
                SELECT MIN(ts_open) AS mn, MAX(ts_open) AS mx,
                       COUNT(DISTINCT DATE(FROM_UNIXTIME(ts_open/1000))) AS days
                FROM flow_bars_1m_ml WHERE symbol='BTC-USD'
            """)
            r = cur.fetchone()
            if r and r["mn"]:
                dt_min = datetime.fromtimestamp(r["mn"]/1000, tz=timezone.utc).strftime("%Y-%m-%d")
                dt_max = datetime.fromtimestamp(r["mx"]/1000, tz=timezone.utc).strftime("%Y-%m-%d")
                _c(OK, f"flow_bars_1m_ml 日期範圍: {dt_min} ~ {dt_max}  ({r['days']} 天)")
        except Exception as e:
            _c(WARN, f"flow_bars_1m_ml 日期查詢失敗: {e}")

        # features_15m NaN 比例抽查
        try:
            cur.execute("""
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN oi IS NULL THEN 1 ELSE 0 END)       AS oi_null,
                       SUM(CASE WHEN delta IS NULL THEN 1 ELSE 0 END)    AS delta_null,
                       SUM(CASE WHEN funding_rate IS NULL THEN 1 ELSE 0 END) AS fr_null
                FROM features_15m WHERE symbol='BTC-USD'
            """)
            r = cur.fetchone()
            if r and r["total"]:
                total = r["total"]
                oi_pct = r["oi_null"] / total * 100
                sym = WARN if oi_pct > 90 else OK
                _c(sym, f"features_15m NaN率: OI={oi_pct:.0f}%  delta={r['delta_null']/total*100:.0f}%  funding={r['fr_null']/total*100:.0f}%")
        except Exception:
            pass

        # funding_rates 最新時間
        try:
            cur.execute("SELECT MAX(ts_exchange) AS mx FROM funding_rates WHERE symbol='BTC-USD'")
            r = cur.fetchone()
            if r and r["mx"]:
                dt = datetime.fromtimestamp(r["mx"]/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                _c(OK, f"funding_rates 最新: {dt} UTC")
        except Exception:
            pass

    conn.close()


# ─── 3. Parquet 輸出檢查 ──────────────────────────────────────────────────────

def check_ml_output():
    _h("ML OUTPUT (Parquet)")
    ml_dir = ROOT / "research" / "ml_data"
    if not ml_dir.exists():
        _c(WARN, "research/ml_data/ 不存在，尚未執行 export")
        return

    files = sorted(ml_dir.glob("*.parquet"))
    if not files:
        _c(WARN, "ml_data/ 內無 .parquet 檔")
        return

    try:
        import pandas as pd
    except ImportError:
        for f in files:
            _c(OK, f"{f.name}  ({f.stat().st_size//1024//1024} MB)")
        return

    for f in files:
        try:
            df = pd.read_parquet(f)
            rows, cols = df.shape
            nan_pct = df.isnull().mean().mean() * 100
            sym = WARN if nan_pct > 20 else OK
            _c(sym, f"{f.name:<45} {rows:>8,} rows × {cols} cols  NaN={nan_pct:.1f}%")

            # 日期範圍
            ts_col = next((c for c in ["bucket_15m","ts_open","timestamp"] if c in df.columns), None)
            if ts_col:
                t0 = datetime.fromtimestamp(df[ts_col].min()/1000, tz=timezone.utc).strftime("%Y-%m-%d")
                t1 = datetime.fromtimestamp(df[ts_col].max()/1000, tz=timezone.utc).strftime("%Y-%m-%d")
                _c("  ", f"  日期: {t0} ~ {t1}")
        except Exception as e:
            _c(FAIL, f"{f.name}: 讀取失敗 — {e}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--db",  action="store_true")
    ap.add_argument("--ml",  action="store_true")
    args = ap.parse_args()

    run_all = not (args.raw or args.db or args.ml)

    if run_all or args.raw:
        check_raw_files()
    if run_all or args.db:
        check_db()
    if run_all or args.ml:
        check_ml_output()

    print()


if __name__ == "__main__":
    main()
