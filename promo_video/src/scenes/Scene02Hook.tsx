import React from 'react';
import {AbsoluteFill, useCurrentFrame, useVideoConfig, spring,
        interpolate} from 'remotion';
import {GradientBG} from '../components/Primitives';

export const Scene02Hook: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  const titleO = spring({frame, fps, config: {damping: 200}});
  const titleY = interpolate(titleO, [0, 1], [60, 0]);

  const subtitleO = spring({frame: Math.max(0, frame - 20), fps,
                            config: {damping: 200}});

  const lineW = interpolate(spring({frame: Math.max(0, frame - 35), fps,
                                     config: {damping: 12}}),
                             [0, 1], [0, 480]);

  return (
    <GradientBG from="#0a0e1a" to="#172649">
      <AbsoluteFill style={{justifyContent: 'center',
                            alignItems: 'center',
                            flexDirection: 'column'}}>
        <div style={{
          opacity: titleO,
          transform: `translateY(${titleY}px)`,
          textAlign: 'center',
          fontFamily: '"Noto Sans TC", "Microsoft JhengHei", sans-serif',
        }}>
          <h1 style={{
            fontSize: 168, color: '#fff', margin: 0,
            fontWeight: 900, letterSpacing: '0.04em',
            background: 'linear-gradient(135deg, #fff 0%, #93c5fd 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>V7</h1>
          <p style={{
            fontSize: 64, color: '#e2e8f0', margin: '8px 0 0',
            fontWeight: 700,
          }}>BTC 量化交易系統</p>
        </div>

        <div style={{
          width: lineW, height: 3,
          background: 'linear-gradient(90deg, transparent, #60a5fa, transparent)',
          marginTop: 36,
        }} />

        <p style={{
          opacity: subtitleO,
          fontSize: 32, color: '#94a3b8',
          marginTop: 36, fontFamily: '"Noto Sans TC", sans-serif',
          letterSpacing: '0.08em',
        }}>從訊號到自動下單的完整工程實踐</p>
      </AbsoluteFill>
    </GradientBG>
  );
};
