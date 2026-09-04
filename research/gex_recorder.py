# -*- coding: utf-8 -*-
"""Dealer gamma (GEX) recorder — §0.88d（2026-09-05 上線）

**為什麼錄**：V7 唯一可判定的弱點是「追」47% vs「接」69%（§0.88d）——它分不出
「趨勢裡的回檔」與「過度延伸」。要回答「這個 24h 走勢會延續還是回頭」，
有機制的候選是做市商的 gamma 部位：正 gamma → 對沖把價格壓回（回頭）；
負 gamma → 追漲殺跌（延續）。它不是描述 regime，是造成 regime 的那隻手。

**為什麼要自己錄**：Deribit 公開 API 一次呼叫給全部合約的 OI／mark IV／標的價，
但**沒有歷史**；Coinglass 只有聚合 OI。跟 depth_deltas 一樣：先錄再說。
每小時 1 次呼叫／幣，零成本。

**存什麼**：每個快照一列——幾個聚合數（naive GEX、call/put 分開、±5%/±10%
內的 GEX、gamma flip 價位、總 OI、spot）＋ **逐履約價的原始表**壓成 JSON 存在
同一列（strike / expiry / type / OI / IV / gamma），所以未來任何 GEX 定義
（不同 dealer 方向假設、不同到期權重）都能從原始表重算，不必回頭抓。

**gamma 怎麼算**：Black–Scholes，r=0，IV 用 Deribit 的 mark_iv，T 到該合約
08:00 UTC 到期。GEX_i = gamma_i × OI_i × S² × 1%（每 1% 價格變動的 dealer
對沖量，USD）。**naive 慣例**：call 為正、put 為負（假設 dealer 多 call 空 put）。
慣例是假設不是事實——所以 call_gex / put_gex 分開存，符號可以事後翻。

**不做的事**：不在這裡下任何判斷。這是錄製器，判決另有預註冊（bar 級 MR 強度
依 gamma 符號分桶，資料到 2–3 個月才跑）。

Run: python research/gex_recorder.py            # one snapshot (hourly train)
Out: MySQL gex_snapshots ; research/results/gex_last.json (freshness flag)
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.db import get_db_conn  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

API = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
FLAG = ROOT / "research" / "results" / "gex_last.json"
CURRENCIES = ("BTC", "ETH")
MONTHS = {m: i for i, m in enumerate(
    ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), 1)}

DDL = """
CREATE TABLE IF NOT EXISTS gex_snapshots (
  id            BIGINT AUTO_INCREMENT PRIMARY KEY,
  created_at    DATETIME NOT NULL,
  currency      VARCHAR(8) NOT NULL,
  spot          DOUBLE NOT NULL,
  n_instr       INT NOT NULL,
  total_oi      DOUBLE NOT NULL,
  call_gex_usd  DOUBLE NOT NULL,
  put_gex_usd   DOUBLE NOT NULL,
  net_gex_usd   DOUBLE NOT NULL,
  net_gex_5pct  DOUBLE NOT NULL,
  net_gex_10pct DOUBLE NOT NULL,
  flip_strike   DOUBLE NULL,
  max_gex_strike DOUBLE NULL,
  strikes_json  MEDIUMTEXT NOT NULL,
  UNIQUE KEY uq_cur_ts (currency, created_at),
  KEY k_cur_ts (currency, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def parse_name(name: str):
    # BTC-25SEP26-70000-C  →  expiry datetime (08:00 UTC), strike, type
    _, exp, k, t = name.split("-")
    d, mon, yy = int(exp[:-5]), exp[-5:-2], int(exp[-2:])
    expiry = dt.datetime(2000 + yy, MONTHS[mon], d, 8, tzinfo=dt.timezone.utc)
    return expiry, float(k), t


def bs_gamma(S: float, K: float, T: float, iv: float) -> float:
    if T <= 0 or iv <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + 0.5 * iv * iv * T) / (iv * math.sqrt(T))
    return math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) / (S * iv * math.sqrt(T))


def snapshot(cur: str, now: dt.datetime) -> dict:
    r = requests.get(API, params={"currency": cur, "kind": "option"}, timeout=25)
    r.raise_for_status()
    rows = r.json()["result"]
    spot = float(np.median([x["underlying_price"] for x in rows if x.get("underlying_price")]))
    per, cg, pg, n5, n10, tot = [], 0.0, 0.0, 0.0, 0.0, 0.0
    for x in rows:
        oi = float(x.get("open_interest") or 0.0)
        iv = float(x.get("mark_iv") or 0.0) / 100.0
        if oi <= 0 or iv <= 0:
            continue
        expiry, K, typ = parse_name(x["instrument_name"])
        T = (expiry - now).total_seconds() / (365.0 * 86400.0)
        if T <= 0:
            continue
        g = bs_gamma(spot, K, T, iv)
        gex = g * oi * spot * spot * 0.01          # USD per 1% move, one side
        signed = gex if typ == "C" else -gex        # naive dealer convention
        tot += oi
        if typ == "C":
            cg += gex
        else:
            pg += gex
        if abs(K / spot - 1) <= 0.05:
            n5 += signed
        if abs(K / spot - 1) <= 0.10:
            n10 += signed
        per.append([round(K, 1), expiry.strftime("%Y-%m-%d"), typ, round(oi, 2),
                    round(iv, 4), float(f"{g:.3e}"), round(signed, 0)])
    # per-strike net (all expiries), for flip / max
    by_k: dict = {}
    for K, _, _, _, _, _, sg in per:
        by_k[K] = by_k.get(K, 0.0) + sg
    ks = sorted(by_k)
    flip = None
    for a, b in zip(ks, ks[1:]):
        if by_k[a] * by_k[b] < 0 and (a <= spot <= b or abs((a + b) / 2 / spot - 1) < 0.15):
            flip = float(a + (b - a) * abs(by_k[a]) / (abs(by_k[a]) + abs(by_k[b])))
            if a <= spot <= b:
                break
    max_k = max(ks, key=lambda k: abs(by_k[k])) if ks else None
    return {"currency": cur, "spot": spot, "n_instr": len(per), "total_oi": tot,
            "call_gex_usd": cg, "put_gex_usd": pg, "net_gex_usd": cg - pg,
            "net_gex_5pct": n5, "net_gex_10pct": n10, "flip_strike": flip,
            "max_gex_strike": float(max_k) if max_k is not None else None,
            "strikes_json": json.dumps(per, separators=(",", ":"))}


def main() -> int:
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    flag = {"ok": False, "reason": "", "asof": now.isoformat()}
    try:
        conn = get_db_conn()
        try:
            with conn.cursor() as c:
                c.execute(DDL)
                lines = []
                for cur in CURRENCIES:
                    s = snapshot(cur, now)
                    c.execute(
                        "INSERT IGNORE INTO gex_snapshots (created_at,currency,spot,n_instr,total_oi,"
                        "call_gex_usd,put_gex_usd,net_gex_usd,net_gex_5pct,net_gex_10pct,flip_strike,"
                        "max_gex_strike,strikes_json) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (now.replace(tzinfo=None), cur, s["spot"], s["n_instr"], s["total_oi"],
                         s["call_gex_usd"], s["put_gex_usd"], s["net_gex_usd"], s["net_gex_5pct"],
                         s["net_gex_10pct"], s["flip_strike"], s["max_gex_strike"], s["strikes_json"]))
                    sign = "+" if s["net_gex_5pct"] > 0 else "−"
                    flip_txt = f"{s['flip_strike']:,.0f}" if s["flip_strike"] else "n/a"
                    lines.append(f"{cur} spot {s['spot']:,.0f} net GEX(±5%) {sign}${abs(s['net_gex_5pct'])/1e6:,.1f}M "
                                 f"flip {flip_txt} n={s['n_instr']}")
            conn.commit()
        finally:
            conn.close()
        flag["ok"] = True
        flag["reason"] = " | ".join(lines)
        print(f"gex_recorder: {now:%Y-%m-%d %H:%M}Z  " + " | ".join(lines))
        rc = 0
    except Exception as e:  # noqa: BLE001
        flag["reason"] = f"{type(e).__name__}: {e}"[:300]
        print(f"gex_recorder FAILED: {flag['reason']}")
        rc = 1
    FLAG.write_text(json.dumps(flag, ensure_ascii=False), encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
