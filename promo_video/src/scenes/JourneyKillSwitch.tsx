import React from 'react';
import {AbsoluteFill, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {GradientBG} from '../components/Primitives';

const FONT = '"Noto Sans TC", "Microsoft JhengHei", sans-serif';

const SWITCHES = [
  {id: 'CAP-2', name: '資金上限', desc: 'equity > 1.5× 本金 → HALT', c: '#FF9F43'},
  {id: 'CAP-3', name: '單日虧損', desc: '日內 −20% → HALT', c: '#FF6B6B'},
  {id: 'CAP-4', name: '總虧損', desc: '累計 −30% → DEMOTE', c: '#FF6B6B'},
  {id: 'E4', name: 'API 權限', desc: '可提幣 → 拒絕啟動', c: '#BC8CFF'},
  {id: 'A4', name: '對帳不一致', desc: '本地 ≠ OKX → HALT', c: '#56D4E0'},
  {id: 'B4', name: 'Stop 延遲', desc: '5 秒沒掛上 → 緊急平倉', c: '#FF6B6B'},
  {id: 'C5/C6', name: '時鐘漂移', desc: '>5s HALT / >30s DEMOTE', c: '#58A6FF'},
  {id: 'A1–A3', name: '連線狀態', desc: '斷線 / 心跳逾時', c: '#58A6FF'},
  {id: 'MAX-POS', name: '倉位計數', desc: '>1 倉 = 自身 bug → DEMOTE', c: '#FF9F43'},
  {id: 'PRESUBMIT', name: '下單前防線', desc: '槓桿 >3x → 拒發單', c: '#3FB950'},
];

export const JourneyKillSwitch: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const titleO = spring({frame, fps, config: {damping: 200}});
  const footO = spring({frame: Math.max(0, frame - 380), fps,
                        config: {damping: 200}});

  return (
    <GradientBG from="#0a0e1a" to="#241a1a">
      <AbsoluteFill style={{fontFamily: FONT, padding: 80}}>
        <div style={{opacity: titleO,
                     transform: `translateY(${(1 - titleO) * 30}px)`}}>
          <div style={{color: '#FF6B6B', fontSize: 28, fontWeight: 700,
                       letterSpacing: '0.15em'}}>
            RISK ENGINE
          </div>
          <h1 style={{color: '#fff', fontSize: 68, margin: '12px 0 0 0',
                      fontWeight: 800}}>
            10 道 Hard Kill Switch — 不靠紀律
          </h1>
          <p style={{color: '#9aa5c4', fontSize: 28, marginTop: 10}}>
            全部寫進 production code，每一道都被故意觸發測試過
          </p>
        </div>

        {SWITCHES.map((s, i) => {
          const col = i % 2;
          const row = Math.floor(i / 2);
          const d = 55 + i * 28;
          const o = spring({frame: Math.max(0, frame - d), fps,
                            config: {damping: 15, stiffness: 90}});
          return (
            <div key={s.id}
                 style={{position: 'absolute',
                         left: 80 + col * 890, top: 330 + row * 108,
                         width: 850, height: 92,
                         opacity: o,
                         transform: `translateX(${(1 - o) * (col ? 60 : -60)}px)`,
                         background: '#161B22', borderRadius: 14,
                         display: 'flex', alignItems: 'center',
                         padding: '0 28px',
                         border: `1px solid ${s.c}30`}}>
              <div style={{background: s.c, color: '#0D1117',
                           fontWeight: 800, fontSize: 22,
                           borderRadius: 999, padding: '6px 20px',
                           minWidth: 150, textAlign: 'center'}}>
                {s.id}
              </div>
              <div style={{color: '#fff', fontSize: 27, fontWeight: 700,
                           marginLeft: 26, width: 220}}>
                {s.name}
              </div>
              <div style={{color: '#B0B8C4', fontSize: 23, marginLeft: 14}}>
                {s.desc}
              </div>
            </div>
          );
        })}

        <div style={{position: 'absolute', left: 80, right: 80, bottom: 56,
                     opacity: footO, textAlign: 'center'}}>
          <span style={{color: '#FFD93D', fontSize: 30, fontWeight: 700}}>
            嚴重度分級：HALT（暫停・自動恢復）&lt; DEMOTE（平倉・人工重啟）&lt; HARD FREEZE
          </span>
        </div>
      </AbsoluteFill>
    </GradientBG>
  );
};
