import React from 'react';
import {useCurrentFrame, AbsoluteFill} from 'remotion';
import {SUBTITLES} from '../data/subtitles';

/**
 * Burns Chinese subtitles into the bottom 18% of the frame.
 * Cue resolution by linear scan (N=80 cues; trivial cost).
 */
export const Subtitles: React.FC = () => {
  const frame = useCurrentFrame();
  const active = SUBTITLES.find(
    (c) => frame >= c.start && frame < c.end && c.text.length > 0,
  );
  if (!active) return null;
  return (
    <AbsoluteFill
      style={{
        justifyContent: 'flex-end',
        alignItems: 'center',
        paddingBottom: 100,
        pointerEvents: 'none',
      }}
    >
      <div
        style={{
          maxWidth: 1400,
          padding: '18px 36px',
          background: 'rgba(0,0,0,0.62)',
          borderRadius: 12,
          color: 'white',
          fontSize: 44,
          fontWeight: 600,
          lineHeight: 1.35,
          textAlign: 'center',
          fontFamily:
            '"Noto Sans TC", "Microsoft JhengHei", "PingFang TC", sans-serif',
          letterSpacing: '0.04em',
          textShadow: '0 2px 8px rgba(0,0,0,0.8)',
        }}
      >
        {active.text}
      </div>
    </AbsoluteFill>
  );
};
