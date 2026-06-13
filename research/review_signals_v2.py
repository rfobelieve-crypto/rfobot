"""Review signals from indicator_history (matches chart triangles)."""
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

# Get ALL indicator_history
cur.execute("""
    SELECT dt, close, pred_direction_code, strength_code,
           confidence_score, dir_prob_up, pred_return_4h, regime_code
    FROM indicator_history
    ORDER BY dt ASC
""")
rows = cur.fetchall()
conn.close()

dir_map = {1: 'UP', -1: 'DOWN', 0: 'NEUTRAL'}
str_map = {3: 'Strong', 2: 'Moderate', 1: 'Weak'}
regime_map = {2: 'BULL', -2: 'BEAR', 0: 'CHOPPY', -99: 'WARMUP'}

# Build list with 4h outcomes
signals = []
close_by_dt = {r['dt']: float(r['close']) for r in rows}
dt_list = sorted(close_by_dt.keys())

for i, r in enumerate(rows):
    d = dir_map.get(int(r['pred_direction_code'] or 0), 'N')
    s = str_map.get(int(r['strength_code'] or 1), 'Weak')
    if d == 'NEUTRAL' or s == 'Weak':
        continue

    dt = r['dt']
    entry = float(r['close'])
    target_dt = dt + timedelta(hours=4)

    # Find close 4h later
    exit_price = None
    for later_r in rows[i+1:]:
        if later_r['dt'] >= target_dt:
            exit_price = float(later_r['close'])
            break

    if exit_price:
        ret = (exit_price - entry) / entry * 100
        correct = (d == 'UP' and ret > 0) or (d == 'DOWN' and ret < 0)
    else:
        ret = None
        correct = None

    signals.append({
        'dt': dt, 'direction': d, 'strength': s,
        'confidence': float(r['confidence_score'] or 0),
        'p_up': float(r['dir_prob_up'] or 0.5),
        'entry': entry, 'ret': ret, 'correct': correct,
        'regime': regime_map.get(int(r['regime_code'] or 0), '?'),
    })

# Filter last 3 days
cutoff = datetime.now(timezone.utc) - timedelta(days=3)
recent = [s for s in signals if s['dt'].replace(tzinfo=timezone.utc) >= cutoff]

print(f"=== 近 3 天圖表信號復盤 ({len(recent)} 筆) ===\n")

wins, losses, pending = 0, 0, 0
wrong = []

for s in recent:
    t_local = s['dt'].replace(tzinfo=timezone.utc).astimezone(TZ)
    t_str = t_local.strftime('%m/%d %H:%M')
    st = s['strength'][0]

    if s['ret'] is not None:
        mark = 'OK +' if s['correct'] else 'WRONG X'
        if s['correct']:
            wins += 1
        else:
            losses += 1
            wrong.append(s)
        print(f"  {t_str} [{st}] {s['direction']:>4} c={s['confidence']:5.0f} P(UP)={s['p_up']:.2f} ${s['entry']:>8,.0f} -> {s['ret']:+.2f}% {mark}  [{s['regime']}]")
    else:
        pending += 1
        print(f"  {t_str} [{st}] {s['direction']:>4} c={s['confidence']:5.0f} P(UP)={s['p_up']:.2f} ${s['entry']:>8,.0f} -> pending  [{s['regime']}]")

total = wins + losses
print()
print(f"=== Summary ===")
print(f"  Filled: {total}, Pending: {pending}")
if total > 0:
    print(f"  Win Rate: {wins}/{total} ({wins/total*100:.0f}%)")

if wrong:
    print(f"\n=== 判斷錯誤分析 ({len(wrong)} 筆) ===\n")
    for s in wrong:
        t_local = s['dt'].replace(tzinfo=timezone.utc).astimezone(TZ)
        print(f"  {t_local.strftime('%m/%d %H:%M')} | {s['direction']} (P(UP)={s['p_up']:.2f} c={s['confidence']:.0f}) | 4h ret={s['ret']:+.2f}%")
        if abs(s['ret']) < 0.15:
            print(f"    -> borderline ({s['ret']:+.2f}%), direction call ambiguous")
        elif s['direction'] == 'DOWN' and s['ret'] > 0:
            if s['regime'] == 'CHOPPY':
                print(f"    -> CHOPPY regime, direction unreliable")
            if s['p_up'] > 0.35:
                print(f"    -> P(UP)={s['p_up']:.2f} close to 0.4 threshold, weak conviction")
        elif s['direction'] == 'UP' and s['ret'] < 0:
            if s['regime'] == 'CHOPPY':
                print(f"    -> CHOPPY regime, direction unreliable")
            if s['p_up'] < 0.65:
                print(f"    -> P(UP)={s['p_up']:.2f} close to 0.6 threshold, weak conviction")
        print()
