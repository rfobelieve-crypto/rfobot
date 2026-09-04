# -*- coding: utf-8 -*-
"""Pre-registration board publisher — every open hypothesis in one place.

Why (2026-08-26): the pre-registrations live in TODO.md sections and their
progress is only visible by running a script per clock. The operator asked
"是這樣我才知道進度怎麼樣" — the discipline is invisible from outside, so
from outside it looks like nothing is happening while five clocks run.

Design rule, and the reason this file adds almost no arithmetic: THE BOARD
SHOWS PROGRESS, NOT VERDICTS. Each verdict keeps exactly one owner (the
scorer that froze its criteria); duplicating a number here would create a
second implementation that can silently disagree — the failure mode of
mistake.md 2026-08-01 (two copies of the same data, one rots quietly).

So each entry sources its progress one of three ways, in order of
preference:
  json    read the clock file its owner already writes
  count   count rows in the frozen ledger (unambiguous, no scoring)
  date    days elapsed since the frozen registration date — for gates whose
          binding constraint IS wall time, this is exact from the date alone

Off-cloud-recorder pattern (fourth instance after raid-signals,
weather_station, v7_veto_clock): THIS machine has the caches and rides the
hourly train, so it computes and UPSERTs; the agent only SELECTs.

Public-surface rule (CLAUDE.md): percentages, R multiples, counts and dates
only. No contract sizes, no USD, no balances. Nothing here touches those.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "research" / "results"
LOG = RES / "sweep_shadow_log.csv"
CORE9 = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"}
HOME = {"RANGING", "TREND_DOWN"}                      # §0.59b
SEP_59 = datetime(2026, 8, 26, tzinfo=timezone.utc)


def _days_since(d: datetime) -> float:
    return (datetime.now(timezone.utc) - d).total_seconds() / 86400


def _json(name: str):
    try:
        return json.loads((RES / name).read_text(encoding="utf-8"))
    except Exception:
        return None


def _shadow_rows():
    try:
        with open(LOG, newline="", encoding="utf-8-sig") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def _e_clock():
    """Variant E progress — read from its OWNING scorer, never recounted here.

    shadow_engine.e_clock() holds the four frozen conditions (§0.474b); this
    board only displays what it returns. If the two ever disagree, this file
    is wrong. Returns None when the engine cannot be imported so the board
    degrades to "no row" instead of inventing one.
    """
    try:
        sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))
        import shadow_engine as SE
        return SE.e_clock(SE.read_log(), "E")
    except Exception:
        return None


def _n_gate_f(rows):
    """Gate F progress — MUST reproduce shadow_engine's "B (pierce) closed=".

    Deliberately NOT filtered to core9. The first version of this function
    added that filter and produced 346 against the published 1127 — the
    exact "second implementation quietly disagrees" failure this file's
    docstring warns about, committed inside the file that warns about it.
    If this number ever stops matching the hourly log's B-row, THIS is
    wrong, not the log.

    2026-09-02: variant B was judged FAIL at n=1428 and moved to the settled
    list, so this count no longer drives an open row. Kept as the log-parity
    reference; do not resurrect the open row from it.
    """
    return sum(1 for r in rows if r.get("variant_b") == "1"
               and r.get("status") == "CLOSED")


def _n_059(rows):
    """Fresh home-regime fills since the §0.59 floor.

    core9 HERE IS CORRECT and must not be "fixed" to match _n_gate_f: the
    §0.58/§0.59 evidence base is core9 throughout, so the clock counts the
    same population the verdict will score. Two different filters on the
    same CSV, each right for its own question.

    Counting only — no meanR here. The verdict is sweep_forward's job.
    """
    n = 0
    for r in rows:
        if (r.get("variant_b") != "1" or r.get("status") != "CLOSED"
                or r.get("universe") != "core9"):
            continue
        if r.get("regime_cell") not in HOME:
            continue
        try:
            ts = int(float(r["fill_ts"]))
        except (ValueError, TypeError, KeyError):
            continue
        if ts >= SEP_59.timestamp():
            n += 1
    return n


def build():
    rows = _shadow_rows()
    q2 = _json("v7_regime_q2_clock.json") or {}
    veto = _json("v7_veto_clock.json") or {}
    gf = _json("sweep_forward_gate.json") or {}
    _ec = _e_clock()   # variant A, owned by sweep_forward.py
    open_items = [
        {
            "id": "0.59", "line": "流動性獵取", "title": "regime 進場濾網",
            "hypothesis": "只在主場（震盪 ∪ 空頭趨勢）開火，其餘一律不進場",
            "why": "主場在兩批獨立樣本間穩定，崩的全在非主場（§0.58）",
            "registered": "2026-08-26", "source": "count",
            "n": _n_059(rows), "gate_n": 150,
            "days": round(_days_since(SEP_59), 1), "gate_days": 30,
            "note": "提出這條規則用掉的樣本已作廢，只採 08-26 之後的新成交",
        },
        {
            "id": "0.60 Q2", "line": "V7", "title": "上漲趨勢只收看空訊號",
            "hypothesis": "TREND_UP 格擋掉 UP 訊號，只接受 DOWN 訊號",
            "why": "該格做多 48.1%（八格最差），做空 65.7%（第二好）",
            "registered": "2026-08-26", "source": "json",
            "n": q2.get("n", 0), "gate_n": q2.get("gate_n", 60),
            "days": q2.get("days", 0), "gate_days": q2.get("gate_days", 30),
            "note": "已有一批 11 筆的反例（55%），寫進註冊書不得遺忘",
        },
        {
            "id": "地形扳機", "line": "V7", "title": "地形濾網上線扳機",
            "hypothesis": "保留 vs 否決的勝率差 ≥8pp 且新訊號 ≥60 筆",
            "why": "地形四維過了三關但仍是 display-only，扳機決定能否進場規則",
            "registered": "2026-08-02", "source": "json",
            "n": veto.get("strong_since_trigger", 0),
            "gate_n": veto.get("trigger_target", 60),
            "days": round(_days_since(datetime(2026, 8, 2, tzinfo=timezone.utc)), 1),
            "gate_days": None,
            "note": "看板主數字是已結算的；剛開火的訊號要等約 4 小時才會動",
        },
        {
            # 2026-09-02: variant B judged FAIL (see settled). The formal track
            # (variant A, no filter, core9, frozen 2026-07-28) is what is still
            # running. Its number comes from the sweep_forward.py artifact --
            # that scorer owns it, this file only displays it; it refreshes when
            # the scorer runs (monthly on the 5th or by hand), not hourly.
            "id": "Gate F·A", "line": "流動性獵取",
            "title": "正式軌道前瞻驗證（無濾網·core9）",
            "hypothesis": "掃單失敗的 edge 在前瞻樣本上仍為正（規則凍結 2026-07-28）",
            "why": "變體 B（＋淺穿越濾網）09-02 判 FAIL，濾網路線作廢；只剩無濾網的正式軌道",
            "registered": "2026-07-28", "source": "json",
            "n": gf.get("n", 0), "gate_n": gf.get("gate_n", 1400),
            "days": round(_days_since(datetime(2026, 7, 28, tzinfo=timezone.utc)), 1),
            "gate_days": None,
            "ci_low": gf.get("ci_low"), "mean_r": gf.get("mean_r"),
            "pos": gf.get("pos"), "pos_of": gf.get("pos_of"),
            "scored_at": gf.get("asof_utc"),
            "note": "數字由判決程式每月 5 號（或手動）計分時產出，不是每小時更新；"
                    "約 8 筆/天，1400 筆預計 2027-01 中到期",
        },
        *([{
            "id": "0.474b E", "line": "流動性獵取",
            "title": "獵取當下的衍生品讀法（BTC）",
            "hypothesis": "獵取當下 OI 下降＋清算爆量＝止損沖洗，反轉勝率高於一般獵取",
            "why": "B 的濾網路線作廢後，這是唯一有機會在近期拿到判決的線；"
                   "判準 2026-09-03 才凍結，在那之前它只有定義沒有門檻",
            "registered": "2026-08-02", "source": "engine",
            "n": _ec.get("n", 0), "gate_n": _ec.get("floor", 60),
            "days": round(_days_since(datetime(2026, 8, 2, tzinfo=timezone.utc)), 1),
            "gate_days": None,
            "ci_low": _ec.get("ci_low"), "mean_r": _ec.get("mean_r"),
            "pos": None, "pos_of": None,
            "note": "四條全要：n≥60、日聚類 CI 下緣>0、均netR 高於同期非本組 BTC "
                    f"獵取≥+0.08R（現差 {_ec.get('gap')}）、前後兩半皆正"
                    f"（現 {_ec.get('halves')}）。約 0.8 筆/天",
        }] if _ec else []),
        {
            "id": "0.52", "line": "縮帆（風控）", "title": "ADX 趨勢環境減碼",
            "hypothesis": "偵測到趨勢時把部位砍半，降低單邊行情的受傷幅度",
            "why": "不預測風暴，只收帆——趨勢環境是獵取與網格的逆風",
            "registered": "2026-08-17", "source": "date",
            "n": None, "gate_n": None,
            "days": round(_days_since(datetime(2026, 8, 17, tzinfo=timezone.utc)), 1),
            "gate_days": 30,
            "note": "數字見市場天氣站卡片；此處只顯示時間閘門的進度",
        },
        {
            "id": "0.53", "line": "縮帆（風控）", "title": "波動目標倉位",
            "hypothesis": "依近期波動連續縮放部位（波動大→下小注）",
            "why": "§0.52 的連續版；儀表排名裡 trailing vol 遠強於 ADX",
            "registered": "2026-08-20", "source": "date",
            "n": None, "gate_n": None,
            "days": round(_days_since(datetime(2026, 8, 20, tzinfo=timezone.utc)), 1),
            "gate_days": 30,
            "note": "成交在時間上高度集中，短窗等於一個行情段落不是一批樣本",
        },
    ]
    # §0.75 arb clock: minutes recorded, from the arb repo's engine CSV.
    # Count BOTH files for the primary pair: the 2026-08-28 instrument
    # upgrade rotated the first 257 minutes to minutes.csv.old — same
    # window, same clock.
    # 2026-09-01: the line became a FAMILY (5 more pairs registered 08-30,
    # TODO §0.75) but this counter still showed only SNDK, so from outside
    # the site the extra recorders did not exist. The gate stays SNDK's
    # (its clock started first and its verdict day is 09-04); the family
    # count rides along in the note so the board stops understating the
    # line — same shape as §0.86: work that exists but never surfaces.
    def _count_csv(*parts) -> int:
        try:
            with open(ROOT.parent.joinpath("arb", "engine", "logs", *parts),
                      encoding="utf-8") as fh:
                return max(0, sum(1 for _ in fh) - 1)
        except Exception:
            return 0

    def _count_pair(*dirs) -> int:
        """Live file + EVERY rotation.

        2026-09-01 the recorder's rotation became TIMESTAMPED
        (minutes.csv.<ts>.old) so a second schema change could not overwrite
        the first one's data — and this counter, which only knew the literal
        ".old" name, promptly dropped from ~6,000 to 914. That is
        mistake.md 2026-08-29 verbatim (rotate a file, a downstream counter
        silently restarts from a smaller number), committed in the same
        session that fixed the rotation. Glob, never a literal name.
        """
        import glob as _g
        base = ROOT.parent.joinpath("arb", "engine", "logs", *dirs, "minutes.csv")
        n = _count_csv(*dirs, "minutes.csv")
        for f in _g.glob(str(base) + "*.old"):
            try:
                with open(f, encoding="utf-8") as fh:
                    n += max(0, sum(1 for _ in fh) - 1)
            except Exception:
                pass
        return n

    arb_min = _count_pair()
    _fam = {p: _count_pair(p)
            for p in ("NBIS", "ANTH", "BTC", "ZEC", "NEAR", "HYPE")}
    _fam_live = {k: v for k, v in _fam.items() if v > 0}
    # §0.91 in-venue basis (registered 2026-09-02, product-side request).
    # Counting FORWARD rows only: the 180-day backfill is prior context and
    # must never move this number, or the clock would start life "full".
    _basis_n = _basis_gate = 0
    try:
        from shared.db import get_db_conn as _gdb
        _c = _gdb()
        try:
            with _c.cursor() as _cur:
                _cur.execute("SELECT COUNT(*) n FROM basis_obs "
                             "WHERE ts_received >= %s AND is_verdict=1",
                             (int(datetime(2026, 9, 2,
                                           tzinfo=timezone.utc).timestamp() * 1000),))
                _basis_n = int((_cur.fetchone() or {}).get("n") or 0)
        finally:
            _c.close()
        # 2 symbols x 6 per hour x 24 x 28 days
        _basis_gate = 2 * 6 * 24 * 28
    except Exception:
        pass
    if _basis_gate:
        open_items.append({
            "id": "0.91", "line": "套利（第四線）",
            "title": "站內資金費收租分佈",
            "hypothesis": "Bitget 現貨多＋永續空，BTC/ETH 結算資金費年化"
                          "中位 ≥8% 且翻負時段 <25%",
            "why": "永續與現貨是同一標的的兩個價，資金費就是那個價差的定價"
                   "——不需要預測任何東西",
            "registered": "2026-09-02", "source": "count",
            "n": _basis_n, "gate_n": _basis_gate,
            "days": round(_days_since(datetime(2026, 9, 2,
                                               tzinfo=timezone.utc)), 1),
            "gate_days": 28,
            "note": "判準凍結後才拉 180 天回填：回填只作背景不判決（它的中位"
                    "已低於門檻，但那不是這個窗口的答案）；只錄不交易",
        })
    open_items.append({
        "id": "0.75", "line": "套利（第四線）", "title": "兩場館溢價錄製",
        "hypothesis": "SNDK 在 Entropy 與 Robinhood 鏈之間的溢價，扣費後有可交易的肉",
        "why": "新場館=定價未磨平+零費率補貼期；Entropy 全是股票/私募永續（含 OpenAI/Anthropic）",
        "registered": "2026-08-28", "source": "count",
        "n": arb_min, "gate_n": 7 * 1440,
        "days": round(_days_since(datetime(2026, 8, 28, 10, 28,
                                           tzinfo=timezone.utc)), 1),
        "gate_days": 7,
        "note": ("判準已凍結（≥1bps 帶、日均≥10 次、兩半皆成立）；只錄不交易，"
                 "下單路徑未經審計前不碰錢"
                 + (f"｜同判準的家族還有 {len(_fam_live)} 個配對在錄"
                    f"（{'、'.join(_fam_live)}，各約 "
                    f"{min(_fam_live.values())//60} 小時），"
                    f"含 BTC 對照組——全部報告不挑" if _fam_live else "")),
    })
    settled = [
        # 2026-08-27 的一整輪:十二個候選、兩份 TradingView 指標、
        # 依預註冊零個過關。全部列出——只顯示存活者的看板是在對「過程」說謊。
        {"id": "0.71", "line": "流動性獵取", "title": "流動性是線,不是區間",
         "verdict": "負結果", "tone": "warn",
         "text": "穿越深度對報酬單調遞減:最淺那桶(0.00–0.03 ATR)最好,"
                 "+0.1176/61.6%。而且它的停損率與其他桶相同(10.6% vs 10.3%)"
                 "——不是「淺=安全」,是「淺=勝率真的高」。參考指標的 ±0.1% "
                 "區間在 3.5 ATR 的停損尺度下是雜訊。"},
        {"id": "0.73", "line": "流動性獵取", "title": "粗糙本身就是濾網",
         "verdict": "已定案", "tone": "ok",
         "text": "pivot 粒度 4/5/8/10/14 全測:meanR 單調遞增到 10、14 掉下來。"
                 "大擺盪點承載的流動性比小擺盪點多,細化不是多拿樣本是稀釋"
                 "好樣本。現行的 10 就在峰值上。"},
        {"id": "0.74", "line": "流動性獵取", "title": "區域成交量權重無分離",
         "verdict": "負結果", "tone": "warn",
         "text": "給每個池一個「自形成以來累積了多少成交量」的權重,成立桶只差"
                 "+0.0214R。但勝率單調上升 3pp 而報酬不動——重的區域比較常贏"
                 "但贏得小。這解釋了為什麼這類指標在圖上看起來有用。"},
        {"id": "0.71c", "line": "流動性獵取", "title": "堆疊:四家族上未達標",
         "verdict": "未達標", "tone": "warn",
         "text": "同一價位疊了幾種池,三個可測家族方向一致(疊得少較好),"
                 "但合併只差 +0.0178R(需 >0.03)、家族 2/4(需 ≥3)。"
                 "不因方向一致升級——那正是預註冊要擋的。稀釋來源是佔母體 49% "
                 "的 session 池,而它是最「時鐘化」的一種。"},
        {"id": "0.72", "line": "方法", "title": "驗證圖濾掉了 82%",
         "verdict": "已修正", "tone": "ok",
         "text": "七天流動性圖用「離現價 ±4%」當邊界,72 個 pivot 只畫出 13 個。"
                 "操作者圈出三個「沒記錄到」的點,兩個是這個濾網造成的。"
                 "驗證用的圖不可以過濾它要驗證的東西。已改用視窗自己的價格範圍。"},
        {"id": "0.59b", "line": "流動性獵取", "title": "主場定義修正",
         "verdict": "已定案", "tone": "ok",
         "text": "主場從「只有震盪」改為「震盪 ∪ 空頭趨勢」。原定義是從一個"
                 "不分方向的舊框架帶來的殘留；判準誠實套用會選出兩格。"},
        {"id": "0.58", "line": "流動性獵取", "title": "前瞻落差的來源",
         "verdict": "已定案", "tone": "ok",
         "text": "回測與前瞻的落差有 75% 是格內衰退、只有 7% 是行情組成。"
                 "但主場兩期一致（+0.098 → +0.093）——不是 edge 死了，"
                 "是非主場崩了。"},
        {"id": "0.60 Q1", "line": "V7", "title": "方向軸不成立",
         "verdict": "負結果", "tone": "warn",
         "text": "獵取那條線上「趨勢要分上下」成立，V7 這裡不成立"
                 "（上漲 58.0% vs 下跌 58.5%，差 0.6pp）。"
                 "一條線有的效應不能假設另一條也有。"},
        {"id": "0.60 Q4", "line": "V7", "title": "格內衰退成立",
         "verdict": "已定案", "tone": "warn",
         "text": "近期訊號準度下滑，拆解後組成效應僅 −0.1pp、格內效應 −6.3pp。"
                 "主場不但沒退還變好，崩的是非主場——與獵取同構。"},
        {"id": "Gate F", "line": "流動性獵取", "title": "變體 B 前瞻驗證 FAIL",
         "verdict": "已陣亡", "tone": "dead",
         "text": "2026-09-02 判決（n=1428/1400）：日聚類 CI 下緣 −0.094、"
                 "meanR −0.010、勝率 55.8%、15/29 幣正 → FAIL。"
                 "C（767/400）、D（371/400）建立在 B 之上，連坐作廢。"
                 "不做 B′、不挑子集重判；正式軌道 A 繼續累積。"},
        {"id": "撤單流", "line": "撤單流", "title": "方向性判決 FAIL",
         "verdict": "已陣亡", "tone": "dead",
         "text": "三個方向性檢定全滅，bar 級的方向領先主張到此為止。"
                 "唯一存活的是波動預測，但系統沒有旋鈕接得住。"},
    ]
    _flag_verdict_due(open_items)
    return {
        "asof_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "open": open_items, "settled": settled,
        "principle": "預註冊 = 先寫死規則與判準，再累積全新樣本。"
                     "提出假設用掉的樣本一律作廢，不得重複計入判決。",
        "disclaimer": "Forward validation in progress — research status "
                      "only, not a live strategy, not financial advice.",
    }


def _flag_verdict_due(items):
    """2026-08-31: a full clock must announce itself.  Variant C (§0.44) sat
    at n>=400 for ~9 days and the Moderate clock (§0.491) for 4 days with no
    verdict recorded — the board showed progress but nothing said "the gate
    is met, judge it".  An open clock whose sample gate (and time gate, when
    one exists) is met gets verdict_due=True; the site can badge it and the
    hourly log line prints ⚠VERDICT-DUE."""
    for c in items:
        try:
            n_ok = (c.get("gate_n") is not None and c.get("n") is not None
                    and c["n"] >= c["gate_n"])
            d_ok = c.get("gate_days") is None or (
                c.get("days") is not None and c["days"] >= c["gate_days"])
            if n_ok and d_ok:
                c["verdict_due"] = True
        except Exception:
            pass
    return items


def main() -> int:
    payload = build()
    from shared.db import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS prereg_clocks (
                    id TINYINT PRIMARY KEY,
                    payload MEDIUMTEXT NOT NULL,
                    updated_at DATETIME NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
            cur.execute(
                "INSERT INTO prereg_clocks (id, payload) VALUES (1, %s) "
                "ON DUPLICATE KEY UPDATE payload = VALUES(payload)",
                (json.dumps(payload, ensure_ascii=False),))
        conn.commit()
    finally:
        conn.close()
    o = payload["open"]
    print(f"prereg_clocks published: {len(o)} open, "
          f"{len(payload['settled'])} settled | "
          + " | ".join(f"{x['id']}:{x['n']}/{x['gate_n']}"
                         + (" ⚠VERDICT-DUE" if x.get("verdict_due") else "")
                       for x in o if x["gate_n"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
