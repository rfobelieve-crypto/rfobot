import React from 'react';
import {AbsoluteFill, useCurrentFrame, useVideoConfig, spring,
        interpolate} from 'remotion';
import {GradientBG} from '../components/Primitives';

/**
 * Architecture: two services bracketing MySQL, with TG bot above.
 * Boxes pop in sequentially, then animated arrows show data flow.
 */

const Box: React.FC<{
  x: number; y: number; w: number; h: number;
  delay: number;
  title: string; sub?: string; color?: string;
}> = ({x, y, w, h, delay, title, sub, color = '#1e293b'}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const o = spring({frame: Math.max(0, frame - delay), fps,
                    config: {damping: 200}});
  const scale = interpolate(o, [0, 1], [0.85, 1]);
  return (
    <div style={{
      position: 'absolute', left: x, top: y, width: w, height: h,
      opacity: o, transform: `scale(${scale})`,
      transformOrigin: 'center',
      background: color, borderRadius: 14,
      border: '2px solid rgba(96,165,250,0.4)',
      display: 'flex', flexDirection: 'column',
      justifyContent: 'center', alignItems: 'center',
      fontFamily: '"Noto Sans TC", sans-serif',
      color: '#fff',
      boxShadow: '0 8px 30px rgba(0,0,0,0.4)',
    }}>
      <div style={{fontSize: 32, fontWeight: 700}}>{title}</div>
      {sub && (
        <div style={{fontSize: 18, color: '#93c5fd', marginTop: 6}}>{sub}</div>
      )}
    </div>
  );
};

const Arrow: React.FC<{
  x1: number; y1: number; x2: number; y2: number;
  delay: number; color?: string;
}> = ({x1, y1, x2, y2, delay, color = '#60a5fa'}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const o = spring({frame: Math.max(0, frame - delay), fps,
                    config: {damping: 12, stiffness: 80}});
  const dx = x2 - x1;
  const dy = y2 - y1;
  return (
    <svg style={{position: 'absolute', left: 0, top: 0,
                 width: '100%', height: '100%', pointerEvents: 'none'}}>
      <defs>
        <marker id={`ar-${delay}`} viewBox="0 0 10 10" refX="8" refY="5"
                markerWidth="6" markerHeight="6" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
        </marker>
      </defs>
      <line
        x1={x1} y1={y1}
        x2={x1 + dx * o} y2={y1 + dy * o}
        stroke={color} strokeWidth={3}
        markerEnd={o > 0.95 ? `url(#ar-${delay})` : undefined}
        opacity={0.85}
      />
    </svg>
  );
};

export const Scene04Architecture: React.FC = () => {
  return (
    <GradientBG from="#0a0e1a" to="#0f1a30">
      <AbsoluteFill style={{padding: 80}}>
        {/* Header */}
        <h2 style={{
          fontSize: 56, color: '#fff', margin: 0, textAlign: 'center',
          fontFamily: '"Noto Sans TC", sans-serif',
        }}>系統架構</h2>

        {/* TG bot (top center) */}
        <Box x={1920 / 2 - 200} y={220} w={400} h={120} delay={20}
             title="Telegram Bot" sub="webhook + alert" color="#1e3a5f" />

        {/* Two services */}
        <Box x={200} y={500} w={560} h={200} delay={50}
             title="market_data" sub="WS: Binance + OKX trades" />
        <Box x={1160} y={500} w={560} h={200} delay={80}
             title="indicator" sub="V7 Dual XGBoost + executor" />

        {/* MySQL (bottom center) */}
        <Box x={1920 / 2 - 240} y={800} w={480} h={150} delay={110}
             title="MySQL (Railway)" sub="共用狀態 / trades / signals" color="#1f1f3a" />

        {/* Arrows */}
        {/* market_data -> MySQL */}
        <Arrow x1={480} y1={700} x2={830} y2={870} delay={140} />
        {/* indicator -> MySQL */}
        <Arrow x1={1440} y1={700} x2={1090} y2={870} delay={150} />
        {/* indicator -> TG bot */}
        <Arrow x1={1440} y1={500} x2={1120} y2={340} delay={170}
               color="#a78bfa" />
        {/* TG bot -> indicator (user commands) */}
        <Arrow x1={800} y1={340} x2={1160} y2={500} delay={185}
               color="#fbbf24" />
      </AbsoluteFill>
    </GradientBG>
  );
};
