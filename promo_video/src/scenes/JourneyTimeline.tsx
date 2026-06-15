import React from 'react';
import {AbsoluteFill, spring, useCurrentFrame, useVideoConfig,
        interpolate} from 'remotion';
import {GradientBG} from '../components/Primitives';

const FONT = '"Noto Sans TC", "Microsoft JhengHei", sans-serif';

const PHASES = [
  {date: '2026.03', title: '訂單流基礎設施', color: '#58A6FF',
   items: ['多交易所 WS → flow bars', 'Sweep outcome 追蹤', 'MySQL 多服務部署']},
  {date: '2026.04', title: 'ML 預測指標 v7', color: '#BC8CFF',
   items: ['Dual XGBoost 雙模型', '136 + 76 特徵', 'Rolling percentile 解碼']},
  {date: '2026.05', title: '交易系統化', color: '#FF9F43',
   items: ['Staged framework', 'Kelly / vol drag 數學', '10 道 kill switch']},
  {date: '2026.06', title: 'Live 實盤運行', color: '#3FB950',
   items: ['真錢 Stage 3 上線', '分數合約 sizing 2x', '事故 → 機制化修補']},
];

export const JourneyTimeline: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  const titleO = spring({frame, fps, config: {damping: 200}});
  const lineProgress = interpolate(frame, [40, 70 + 3 * 90], [0, 1],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  return (
    <GradientBG>
      <AbsoluteFill style={{fontFamily: FONT, padding: 80}}>
        <div style={{opacity: titleO,
                     transform: `translateY(${(1 - titleO) * 30}px)`}}>
          <div style={{color: '#56D4E0', fontSize: 28, fontWeight: 700,
                       letterSpacing: '0.15em'}}>
            PROJECT JOURNEY
          </div>
          <h1 style={{color: '#fff', fontSize: 72, margin: '12px 0 0 0',
                      fontWeight: 800}}>
            從指標到實盤自動交易
          </h1>
          <p style={{color: '#9aa5c4', fontSize: 30, marginTop: 12}}>
            每一步升級都由數據與驗證驅動 — 三個月、四個階段
          </p>
        </div>

        {/* timeline line */}
        <div style={{position: 'absolute', left: 100, right: 100, top: 430,
                     height: 5, background: '#21262D', borderRadius: 3}}>
          <div style={{width: `${lineProgress * 100}%`, height: '100%',
                       background: 'linear-gradient(90deg,#58A6FF,#3FB950)',
                       borderRadius: 3}} />
        </div>

        {PHASES.map((p, i) => {
          const d = 40 + i * 90;
          const o = spring({frame: Math.max(0, frame - d), fps,
                            config: {damping: 16, stiffness: 70}});
          const left = 100 + i * 430;
          return (
            <div key={p.date}>
              <div style={{position: 'absolute', left: left + 175, top: 416,
                           width: 32, height: 32, borderRadius: 16,
                           background: p.color, opacity: o,
                           transform: `scale(${o})`,
                           boxShadow: `0 0 ${24 * o}px ${p.color}`}} />
              <div style={{position: 'absolute', left, top: 490, width: 385,
                           opacity: o,
                           transform: `translateY(${(1 - o) * 40}px)`,
                           background: '#161B22', borderRadius: 18,
                           padding: '26px 30px',
                           border: `1px solid ${p.color}33`}}>
                <div style={{color: p.color, fontSize: 26, fontWeight: 800}}>
                  {p.date}
                </div>
                <div style={{color: '#fff', fontSize: 32, fontWeight: 700,
                             margin: '6px 0 14px 0'}}>
                  {p.title}
                </div>
                {p.items.map((it) => (
                  <div key={it} style={{color: '#B0B8C4', fontSize: 21,
                                        lineHeight: 1.55}}>
                    • {it}
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </AbsoluteFill>
    </GradientBG>
  );
};
