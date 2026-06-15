import React from 'react';
import {AbsoluteFill, useCurrentFrame, useVideoConfig, spring,
        interpolate} from 'remotion';
import {GradientBG} from '../components/Primitives';

const PROBLEMS = [
  {title: '訊號黑盒', detail: '不知道為什麼說多空'},
  {title: '上線即實盤', detail: '沒有 staged rollout'},
  {title: '無 Kill Switch', detail: '紀律永遠輸給情緒'},
];

const Card: React.FC<{p: typeof PROBLEMS[0]; index: number}> = ({p, index}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  // Stagger entrance
  const delay = 30 + index * 25;
  const o = spring({frame: Math.max(0, frame - delay), fps,
                    config: {damping: 200}});
  const y = interpolate(o, [0, 1], [50, 0]);
  return (
    <div style={{
      opacity: o,
      transform: `translateY(${y}px)`,
      background: 'rgba(255,255,255,0.05)',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: 16,
      padding: '48px 40px',
      width: 380,
      textAlign: 'center',
      fontFamily: '"Noto Sans TC", sans-serif',
    }}>
      <div style={{
        fontSize: 60, color: '#ef4444', marginBottom: 16,
        fontWeight: 900,
      }}>✗</div>
      <div style={{
        fontSize: 40, color: '#fff', fontWeight: 700,
        marginBottom: 12,
      }}>{p.title}</div>
      <div style={{fontSize: 24, color: '#94a3b8'}}>{p.detail}</div>
    </div>
  );
};

export const Scene03Problem: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const headerO = spring({frame, fps, config: {damping: 200}});

  return (
    <GradientBG from="#0a0e1a" to="#1f1330">
      <AbsoluteFill style={{
        justifyContent: 'center',
        alignItems: 'center',
        flexDirection: 'column',
        gap: 80,
      }}>
        <h2 style={{
          opacity: headerO,
          fontSize: 64, color: '#fff', margin: 0,
          fontFamily: '"Noto Sans TC", sans-serif',
          fontWeight: 700,
        }}>
          市售量化 <span style={{color: '#ef4444'}}>3 個共通問題</span>
        </h2>
        <div style={{display: 'flex', gap: 40}}>
          {PROBLEMS.map((p, i) => (
            <Card key={i} p={p} index={i} />
          ))}
        </div>
      </AbsoluteFill>
    </GradientBG>
  );
};
