import React from 'react';
import {AbsoluteFill, useCurrentFrame, useVideoConfig, interpolate,
        spring} from 'remotion';
import {GradientBG} from '../components/Primitives';

/**
 * Cold open: dark night-trader vibe.
 * Visual layout:
 *  - Deep gradient (almost black) background
 *  - A faint BTC candlestick silhouette sliding right-to-left
 *  - A red downward spike + percent flash at the apex of the spike
 *  - Top-right clock counting up to 03:00 AM
 *  - All subtitle text handled by global <Subtitles> layer
 */
export const Scene01ColdOpen: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();

  // Clock — fades to "03:00 AM"
  const clockOpacity = spring({frame, fps, config: {damping: 200}});

  // Spike trigger around 5s (frame 150 @ 30fps)
  const spikeProgress = interpolate(
    frame, [120, 165], [0, 1], {extrapolateLeft: 'clamp',
                                 extrapolateRight: 'clamp'},
  );
  const spikeOpacity = interpolate(frame, [120, 140, 220, 280],
                                    [0, 1, 1, 0],
                                    {extrapolateRight: 'clamp'});

  // Candlestick scroll (simulated)
  const candleOffset = (frame * 3) % width;

  return (
    <GradientBG from="#02050c" to="#0c1326">
      {/* Faint candlestick band */}
      <AbsoluteFill style={{opacity: 0.18}}>
        <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`}>
          <g transform={`translate(${-candleOffset}, ${height * 0.55})`}>
            {Array.from({length: 60}).map((_, i) => {
              const x = i * 40;
              const h = 30 + (Math.sin(i * 0.7) * 20);
              const up = (i % 3) === 0;
              return (
                <rect
                  key={i}
                  x={x} y={-h / 2}
                  width={16} height={h}
                  fill={up ? '#4ade80' : '#f87171'}
                />
              );
            })}
          </g>
        </svg>
      </AbsoluteFill>

      {/* Clock top-right */}
      <div style={{
        position: 'absolute', top: 60, right: 80,
        fontFamily: '"JetBrains Mono", monospace',
        fontSize: 56, color: '#7d8aa8',
        opacity: clockOpacity,
        letterSpacing: '0.06em',
      }}>03:00 AM</div>

      {/* The dive spike (centered) */}
      <AbsoluteFill style={{
        justifyContent: 'center', alignItems: 'center',
        opacity: spikeOpacity,
      }}>
        <svg width={800} height={480}>
          <defs>
            <linearGradient id="spikeGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#ef4444" stopOpacity="0.0" />
              <stop offset="60%" stopColor="#ef4444" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#ef4444" stopOpacity="0.9" />
            </linearGradient>
          </defs>
          <path
            d={`M 100 80 L 200 120 L 280 200 L 380 ${100 + spikeProgress * 280}
                 L 500 ${260 + spikeProgress * 180} L 600 320 L 720 360`}
            stroke="#ef4444" strokeWidth={6} fill="none"
            strokeLinecap="round"
          />
        </svg>

        <div style={{
          marginTop: 24,
          fontSize: 96, fontWeight: 900,
          color: '#ef4444',
          opacity: spikeOpacity,
          fontFamily: '"JetBrains Mono", monospace',
          textShadow: '0 4px 20px rgba(239,68,68,0.6)',
        }}>
          -5.0%
        </div>
      </AbsoluteFill>
    </GradientBG>
  );
};
