import React from 'react';
import {AbsoluteFill, spring, useCurrentFrame, useVideoConfig,
        interpolate} from 'remotion';

/** Section title that fades + slides up on enter. */
export const SceneTitle: React.FC<{children: React.ReactNode;
                                    subtitle?: string}> = ({children, subtitle}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const o = spring({frame, fps, config: {damping: 200}});
  const y = interpolate(o, [0, 1], [40, 0]);
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
      <div
        style={{
          opacity: o,
          transform: `translateY(${y}px)`,
          textAlign: 'center',
          fontFamily: '"Noto Sans TC", "Microsoft JhengHei", sans-serif',
        }}
      >
        <h1 style={{
          fontSize: 112, color: '#fff', margin: 0,
          fontWeight: 800, letterSpacing: '0.02em',
        }}>{children}</h1>
        {subtitle && (
          <p style={{
            fontSize: 36, color: '#9aa5c4', marginTop: 24,
          }}>{subtitle}</p>
        )}
      </div>
    </AbsoluteFill>
  );
};

/** Big single number that counts up. */
export const CountUp: React.FC<{
  target: number;
  prefix?: string;
  suffix?: string;
  decimals?: number;
  durationFrames?: number;
}> = ({target, prefix = '', suffix = '', decimals = 0,
       durationFrames = 60}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const o = spring({frame, fps, config: {damping: 18, stiffness: 80}});
  const v = interpolate(o, [0, 1], [0, target], {extrapolateRight: 'clamp'});
  return (
    <span>{prefix}{v.toFixed(decimals)}{suffix}</span>
  );
};

/** Background gradient panel. */
export const GradientBG: React.FC<{from?: string; to?: string;
                                    children?: React.ReactNode}> = ({
  from = '#0a0e1a', to = '#1a2540', children,
}) => (
  <AbsoluteFill style={{
    background: `linear-gradient(135deg, ${from} 0%, ${to} 100%)`,
  }}>
    {children}
  </AbsoluteFill>
);

/** Thin animated horizontal accent line. */
export const AccentLine: React.FC<{delay?: number;
                                    color?: string}> = ({delay = 0,
                                                          color = '#4f8cff'}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const o = spring({frame: Math.max(0, frame - delay), fps,
                    config: {damping: 12, stiffness: 60}});
  const w = interpolate(o, [0, 1], [0, 320]);
  return (
    <div style={{
      width: w, height: 4, background: color, marginTop: 24,
      borderRadius: 2,
    }} />
  );
};
