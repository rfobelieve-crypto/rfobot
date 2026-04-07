import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

df = pd.read_parquet('research/dual_model/.cache/features_all.parquet')

DUAL_DIR = Path('indicator/model_artifacts/dual_model')
dir_model = xgb.XGBClassifier()
dir_model.load_model(str(DUAL_DIR / 'direction_xgb.json'))
with open(DUAL_DIR / 'direction_feature_cols.json') as f:
    dir_features = json.load(f)
mag_model = xgb.XGBRegressor()
mag_model.load_model(str(DUAL_DIR / 'magnitude_xgb.json'))
with open(DUAL_DIR / 'magnitude_feature_cols.json') as f:
    mag_features = json.load(f)

X_dir = df.reindex(columns=dir_features, fill_value=0).fillna(0)
dir_prob = dir_model.predict_proba(X_dir)[:, 1]
X_mag = df.reindex(columns=mag_features, fill_value=0).fillna(0)
mag_pred = np.maximum(mag_model.predict(X_mag), 0)

mag_history = []
confidences = []
for i in range(len(mag_pred)):
    if len(mag_history) >= 30:
        pct = np.searchsorted(np.sort(mag_history), mag_pred[i]) / len(mag_history) * 100
    else:
        pct = 50.0
    dir_conviction = abs(dir_prob[i] - 0.5) * 2
    conf = np.clip(pct * (0.7 + 0.3 * dir_conviction), 0, 100)
    confidences.append(conf)
    mag_history.append(mag_pred[i])

directions = ['UP' if p > 0.60 else 'DOWN' if p < 0.40 else 'NEUTRAL' for p in dir_prob]
df['actual_4h'] = df['close'].shift(-4) / df['close'] - 1

print('=' * 60)
print('STRONG vs MODERATE SIGNAL QUALITY')
print('=' * 60)

for tier_name, lo, hi in [('Strong', 80, 999), ('Moderate', 65, 80)]:
    signals = []
    for i in range(len(df)):
        c = confidences[i]
        d = directions[i]
        if lo <= c < hi and d != 'NEUTRAL':
            actual = df.iloc[i].get('actual_4h', np.nan)
            if pd.notna(actual):
                correct = (d == 'UP' and actual > 0.001) or (d == 'DOWN' and actual < -0.001)
                signals.append({'correct': correct, 'actual': actual, 'dir': d})

    if not signals:
        print(f'\n{tier_name}: no signals')
        continue

    sig_df = pd.DataFrame(signals)
    total = len(sig_df)
    wins = int(sig_df['correct'].sum())
    wr = wins / total * 100
    avg_abs = sig_df['actual'].abs().mean() * 100

    up_sigs = sig_df[sig_df['dir'] == 'UP']
    dn_sigs = sig_df[sig_df['dir'] == 'DOWN']
    up_wr = up_sigs['correct'].mean() * 100 if len(up_sigs) > 0 else 0
    dn_wr = dn_sigs['correct'].mean() * 100 if len(dn_sigs) > 0 else 0

    print(f'\n--- {tier_name} (conf {lo}~{hi}) ---')
    print(f'  Signals: {total} ({total/len(df)*100:.1f}% of bars)')
    print(f'  Win rate: {wr:.1f}% ({wins}W / {total-wins}L)')
    print(f'  Avg |return|: {avg_abs:.3f}%')
    print(f'  UP: {len(up_sigs)} signals, {up_wr:.1f}% win rate')
    print(f'  DOWN: {len(dn_sigs)} signals, {dn_wr:.1f}% win rate')

print()
print('=' * 60)
print('LAST 30 DAYS FREQUENCY')
print('=' * 60)
n = len(df)
start = max(0, n - 720)
strong_30 = sum(1 for i in range(start, n) if confidences[i] >= 80 and directions[i] != 'NEUTRAL')
moderate_30 = sum(1 for i in range(start, n) if 65 <= confidences[i] < 80 and directions[i] != 'NEUTRAL')
days = (n - start) / 24
print(f'  Strong:   {strong_30} ({strong_30/days:.1f}/day)')
print(f'  Moderate: {moderate_30} ({moderate_30/days:.1f}/day)')
print(f'  Combined: {strong_30+moderate_30} ({(strong_30+moderate_30)/days:.1f}/day)')
