# -*- coding: utf-8 -*-
"""實測 Lighter WS 的訂閱上限（2026-09-13，TODO §1.40）

===========================================================================
為什麼要實測而不是讀文件
===========================================================================
`lighter_tape.py` 要把宇宙從前 80 名放寬到涵蓋長尾（約 227 個 active 市場），
而它在 `connected` 時是**一個迴圈把 N 個 subscribe 一次全送出去、沒有間隔**。
80 個能過不代表 227 個能過，而**被限流時的失敗方式是安靜少列**
（CLAUDE.md 已記：Lighter REST 被限流時回 text/html）——
那比不放寬更糟，因為看板會是綠的。

判準是**產物**：訂了 N 個，要收到 N 個 `subscribed/trade` 的 ack。
收不齊就是有上限，而缺的那些**不會報錯**。

===========================================================================
這支不碰常駐錄製器
===========================================================================
* 只讀、只數，**不寫任何 parquet**（mistake.md 2026-09-11：第二個實例
  對同一批 parquet 做 read-modify-write，後果事後無法量化）。
* 自己的 WS 連線，跟 `lighter_tape` 那條無關。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import threading
import time

import requests
import websocket

BASE = os.environ.get("LIGHTER_REST", "https://mainnet.zklighter.elliot.ai")
WS = os.environ.get("LIGHTER_WS", "wss://mainnet.zklighter.elliot.ai/stream")


def _mid(ch):
    """`trade:231` / `trade/231` -> 231；取不出來回 -1（不會誤算成 ack）。"""
    s = str(ch or "")
    for sep in (":", "/"):
        if sep in s:
            try:
                return int(s.rsplit(sep, 1)[-1])
            except ValueError:
                return -1
    return -1


def markets(tries=5):
    """**帶退避**：被限流時這個端點回 text/html 不回 JSON（CLAUDE.md 已記），
    而 `.json()` 會拋 JSONDecodeError。第一版在一行 print 裡重複呼叫了一次
    就把自己打限流了 —— 呼叫端現在只准呼叫一次。"""
    r = None
    for i in range(tries):
        try:
            resp = requests.get(BASE + "/api/v1/orderBookDetails", timeout=20)
            r = resp.json()
            break
        except Exception as e:                              # noqa: BLE001
            ct = getattr(locals().get("resp", None), "headers", {}) \
                .get("content-type", "?")
            print("  orderBookDetails 第 %d 次失敗（content-type=%s）：%r"
                  % (i + 1, ct, e))
            time.sleep(5 * (i + 1))
    if r is None:
        raise RuntimeError("orderBookDetails 連 %d 次都拿不到 JSON（限流？）"
                           % tries)
    out = []
    for ob in r.get("order_book_details") or r.get("order_books") or []:
        if ob.get("status") != "active" or "/" in ob.get("symbol", ""):
            continue
        out.append((int(ob["market_id"]), ob["symbol"],
                    float(ob.get("daily_quote_token_volume") or 0.0)))
    out.sort(key=lambda x: -x[2])            # 按日成交額排序，與 tape 同
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0,
                    help="訂閱前 N 個（0 = 全部 active）")
    ap.add_argument("--seconds", type=int, default=75)
    ap.add_argument("--spacing", type=float, default=0.0,
                    help="每個 subscribe 之間的間隔秒數（測節流有沒有幫助）")
    ap.add_argument("--channel", default="trade",
                    help="trade（成交帶）或 order_book（簿口，回整本快照、重得多）")
    ap.add_argument("--batch", type=int, default=0,
                    help="每批幾個（0 = 不分批）；配 --pause 用")
    ap.add_argument("--pause", type=float, default=1.0,
                    help="批與批之間的秒數")
    a = ap.parse_args()

    mk = markets()              # **只准呼叫一次** —— 多一次就打到限流
    n_all = len(mk)
    if a.n:
        mk = mk[:a.n]
    want = {m for m, _, _ in mk}
    sym = {m: s for m, s, _ in mk}
    print("active 市場 %d 個｜本次訂閱 **%d 個**｜間隔 %.3fs｜跑 %d 秒"
          % (n_all, len(want), a.spacing, a.seconds))
    print("  前 5 名：%s" % ", ".join("%s" % s for _, s, _ in mk[:5]))
    print("  末 5 名：%s" % ", ".join("%s" % s for _, s, _ in mk[-5:]))

    acked, trades, errs = set(), collections.Counter(), []
    sent = [0]
    ack_total = [0]
    unparsed = []
    t0 = time.time()
    done = threading.Event()

    def on_open(ws):
        pass

    def on_msg(ws, m):
        try:
            d = json.loads(m)
        except Exception:
            errs.append(("non-json", m[:120]))
            return
        t = d.get("type")
        if t == "connected":
            def sub():
                lst = list(want)
                step = a.batch or len(lst)
                for i in range(0, len(lst), step):
                    for mid in lst[i:i + step]:
                        try:
                            ws.send(json.dumps({"type": "subscribe",
                                                "channel": "%s/%d" % (a.channel, mid)}))
                            sent[0] += 1
                        except Exception as e:              # noqa: BLE001
                            errs.append(("send", "%r @ #%d" % (e, sent[0])))
                            return
                        if a.spacing:
                            time.sleep(a.spacing)
                    if a.batch:
                        time.sleep(a.pause)
            threading.Thread(target=sub, daemon=True).start()
            return
        if t == "ping":
            ws.send(json.dumps({"type": "pong"}))
            return
        if t == "subscribed/%s" % a.channel:
            # **訂閱時送 `trade/N`，回來的 channel 是 `trade:N`** —— 分隔符
            # 不同。第一版用 "/" 切，217 個訂閱拿到 0 個 ack，而同時有 39 個
            # 頻道在收成交 —— 那個矛盾就是「我的儀器壞了不是場館有上限」的
            # signature（mistake.md：自己剛寫的儀器比別人的舊儀器更危險）。
            #
            # 第二版還是不乾淨：`_mid` 解析失敗回 -1，而 `set.add(-1)` 會把
            # **每一個解析失敗的 ack 疊成同一個元素** -> `len(acked)` 低估，
            # 看起來像「場館少 ack 了幾個」。所以**總數與可歸屬分開數**：
            # ack_total 是真的收到幾個，acked 是歸屬得出市場的那些。
            ack_total[0] += 1
            mid = _mid(d.get("channel"))
            if mid >= 0:
                acked.add(mid)
            else:
                unparsed.append(str(d.get("channel"))[:40])
            return
        if t == "update/%s" % a.channel:
            n = len(d.get("trades") or []) + len(d.get("liquidation_trades") or [])
            trades[_mid(d.get("channel"))] += n
            return
        if t and "error" in str(t).lower():
            errs.append(("type", json.dumps(d)[:200]))

    def on_err(ws, e):
        errs.append(("ws", repr(e)[:200]))

    def on_close(ws, code, msg):
        errs.append(("close", "code=%s msg=%s" % (code, str(msg)[:120])))
        done.set()

    ws = websocket.WebSocketApp(WS, on_open=on_open, on_message=on_msg,
                                on_error=on_err, on_close=on_close)
    th = threading.Thread(target=lambda: ws.run_forever(ping_interval=0),
                          daemon=True)
    th.start()
    while time.time() - t0 < a.seconds and not done.is_set():
        time.sleep(1)
    try:
        ws.close()
    except Exception:                                       # noqa: BLE001
        pass
    time.sleep(1)

    print("\n結果（跑了 %.0f 秒）" % (time.time() - t0))
    print("  送出 subscribe **%d**｜**收到 ack 訊息 %d 筆**｜"
          "其中歸屬得出市場的 %d 個｜解析不出 channel 的 %d 筆"
          % (sent[0], ack_total[0], len(acked), len(unparsed)))
    if unparsed:
        print("  解析不出的 channel 樣本：%s" % unparsed[:3])
    print("  **沒收到 ack 的市場數 = %d**" % (len(want) - len(acked)))
    miss = [sym[m] for m in want if m not in acked]
    if miss:
        print("  **沒有 ack 的（前 20）**：%s%s"
              % (", ".join(miss[:20]), " …" if len(miss) > 20 else ""))
    print("  有成交進來的頻道 **%d** 個｜總成交筆數 %d"
          % (len(trades), sum(trades.values())))
    if errs:
        print("  **錯誤/關閉事件 %d 筆**：" % len(errs))
        for k, v in errs[:8]:
            print("    [%s] %s" % (k, v))
    else:
        print("  沒有錯誤、沒有被關連線")
    ok = (sent[0] == len(want)) and (ack_total[0] >= len(want))
    print("\n判讀：**%s**"
          % ("訂閱 %d 個沒有上限問題（送出=ack=目標、零錯誤）" % len(want) if ok
             else "**有問題** —— 見上面的缺口與錯誤，不要直接放寬常駐錄製器"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
