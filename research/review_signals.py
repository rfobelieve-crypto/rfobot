"""Review recent signal performance."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pymysql
from datetime import datetime, timezone, timedelta

TZ = timezone(timedelta(hours=8))
import os
conn = pymysql.connect(
    host=os.getenv("MYSQLHOST", os.getenv("MYSQL_HOST", "")),
    port=int(os.getenv("MYSQLPORT", os.getenv("MYSQL_PORT", "3306"))),
    user=os.getenv("MYSQLUSER", os.getenv("MYSQL_USER", "root")),
    password=os.getenv("MYSQLPASSWORD", os.getenv("MYSQL_PASSWORD", "")),
    database=os.getenv("MYSQLDATABASE", os.getenv("MYSQL_DB", "railway")),
    cursorclass=pymysql.cursors.DictCursor)
cur = conn.cursor()

cur.execute("""
    SELECT signal_time, direction, strength, confidence, p_up,
           entry_price, exit_price, actual_return_4h, correct, filled, regime
    FROM tracked_signals
    WHERE signal_time >= DATE_SUB(NOW(), INTERVAL 3 DAY)
    ORDER BY signal_time ASC
""")
rows = cur.fetchall()

print("=== 近 3 天信號復盤 ===\n")

wins, losses, pending = 0, 0, 0
wrong_signals = []

for r in rows:
    t_utc = r['signal_time']
    t_local = t_utc.replace(tzinfo=timezone.utc).astimezone(TZ)
    t_str = t_local.strftime('%m/%d %H:%M')

    d = r['direction']
    s = r['strength'][0]
    conf = r['confidence']
    p_up = r['p_up']
    entry = r['entry_price']
    regime = r['regime'] or '?'

    if r['filled']:
        ret = r['actual_return_4h'] * 100
        ok = r['correct']
        if ok:
            wins += 1
            icon = '+'
        else:
            losses += 1
            icon = 'X'
            wrong_signals.append(r)

        print(f"  {t_str} [{s}] {d:>4} c={conf:5.0f} P(UP)={p_up:.2f} ${entry:>8,.0f} -> {ret:+.2f}% {'OK' if ok else 'WRONG'} {icon}  [{regime}]")
    else:
        pending += 1
        print(f"  {t_str} [{s}] {d:>4} c={conf:5.0f} P(UP)={p_up:.2f} ${entry:>8,.0f} -> pending  [{regime}]")

total = wins + losses
print()
print(f"=== Summary ===")
print(f"  Filled: {total}, Pending: {pending}")
if total > 0:
    print(f"  Win Rate: {wins}/{total} ({wins/total*100:.0f}%)")

# Analyze wrong signals
if wrong_signals:
    print(f"\n=== 判斷錯誤分析 ({len(wrong_signals)} 筆) ===\n")
    for r in wrong_signals:
        t_local = r['signal_time'].replace(tzinfo=timezone.utc).astimezone(TZ)
        t_str = t_local.strftime('%m/%d %H:%M')
        ret = r['actual_return_4h'] * 100
        d = r['direction']
        p_up = r['p_up']

        print(f"  {t_str} | 預測 {d} (P(UP)={p_up:.2f}, c={r['confidence']:.0f}) | 實際 {ret:+.2f}%")

        # Root cause analysis
        if abs(ret) < 0.1:
            print(f"    -> 微幅波動 ({ret:+.2f}%), 方向邊界判定，非真正錯誤")
        elif d == 'UP' and ret < -0.5:
            print(f"    -> 預測看多但下跌 {ret:.2f}%, P(UP)={p_up:.2f}")
            if p_up < 0.65:
                print(f"    -> 原因: P(UP) 僅 {p_up:.2f}, 信念不夠強但仍觸發信號")
            if r['regime'] == 'CHOPPY':
                print(f"    -> 原因: CHOPPY regime 下方向判斷本來就不穩定")
        elif d == 'DOWN' and ret > 0.5:
            print(f"    -> 預測看空但上漲 {ret:+.2f}%, P(UP)={p_up:.2f}")
            if p_up > 0.35:
                print(f"    -> 原因: P(UP)={p_up:.2f} 離 0.4 閾值太近, 信念不強")
            if r['regime'] == 'CHOPPY':
                print(f"    -> 原因: CHOPPY regime 下方向判斷本來就不穩定")
        print()

# Price context
cur.execute("SELECT dt, close FROM indicator_history WHERE dt >= DATE_SUB(NOW(), INTERVAL 3 DAY) ORDER BY dt ASC LIMIT 1")
start = cur.fetchone()
cur.execute("SELECT dt, close FROM indicator_history WHERE dt >= DATE_SUB(NOW(), INTERVAL 3 DAY) ORDER BY dt DESC LIMIT 1")
end = cur.fetchone()
cur.execute("SELECT MIN(close) as lo, MAX(close) as hi FROM indicator_history WHERE dt >= DATE_SUB(NOW(), INTERVAL 3 DAY)")
rng = cur.fetchone()

if start and end:
    s_p, e_p = float(start['close']), float(end['close'])
    ch = (e_p / s_p - 1) * 100
    print(f"=== BTC 走勢 (近 3 天) ===")
    print(f"  {float(start['close']):>10,.0f} -> {float(end['close']):>10,.0f} ({ch:+.1f}%)")
    if rng:
        print(f"  Low: ${float(rng['lo']):,.0f}  High: ${float(rng['hi']):,.0f}  Range: {(float(rng['hi'])-float(rng['lo']))/float(rng['lo'])*100:.1f}%")

conn.close()
