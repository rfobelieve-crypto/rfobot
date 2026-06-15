import React from 'react';
import {AbsoluteFill, Sequence, spring, useCurrentFrame,
        useVideoConfig} from 'remotion';
import {JourneyTimeline} from './scenes/JourneyTimeline';
import {JourneyKillSwitch} from './scenes/JourneyKillSwitch';
import {JourneyIncidents} from './scenes/JourneyIncidents';

const FONT = '"Noto Sans TC", "Microsoft JhengHei", sans-serif';
const FPS = 30;

// Segment lengths (frames @ 30fps)
const T_TIMELINE = 24 * FPS;   // 0:00–0:24
const T_KILL = 22 * FPS;       // 0:24–0:46
const T_INCIDENT = 24 * FPS;   // 0:46–1:10
const T_OUTRO = 8 * FPS;       // 1:10–1:18
export const JOURNEY_FRAMES = T_TIMELINE + T_KILL + T_INCIDENT + T_OUTRO;

const Outro: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const o = spring({frame, fps, config: {damping: 200}});
  return (
    <AbsoluteFill style={{background: '#0D1117', justifyContent: 'center',
                          alignItems: 'center', fontFamily: FONT}}>
      <div style={{opacity: o, transform: `translateY(${(1 - o) * 30}px)`,
                   textAlign: 'center', padding: '0 140px'}}>
        <div style={{color: '#56D4E0', fontSize: 26, fontWeight: 700,
                     letterSpacing: '0.2em', marginBottom: 30}}>
          TAKEAWAY
        </div>
        <h1 style={{color: '#fff', fontSize: 64, fontWeight: 800,
                    lineHeight: 1.4, margin: 0}}>
          做出模型只是起點 —<br />
          讓真錢系統在不確定的 edge 下活著，<br />
          才是量化的全部。
        </h1>
        <p style={{color: '#7A828E', fontSize: 26, marginTop: 44}}>
          BTC Quant Trading System｜2026.03 – present
        </p>
      </div>
    </AbsoluteFill>
  );
};

export const JourneyHighlights: React.FC = () => {
  return (
    <AbsoluteFill style={{backgroundColor: '#0a0e1a'}}>
      <Sequence from={0} durationInFrames={T_TIMELINE} name="Timeline">
        <JourneyTimeline />
      </Sequence>
      <Sequence from={T_TIMELINE} durationInFrames={T_KILL} name="KillSwitch">
        <JourneyKillSwitch />
      </Sequence>
      <Sequence from={T_TIMELINE + T_KILL} durationInFrames={T_INCIDENT}
                name="Incidents">
        <JourneyIncidents />
      </Sequence>
      <Sequence from={T_TIMELINE + T_KILL + T_INCIDENT}
                durationInFrames={T_OUTRO} name="Outro">
        <Outro />
      </Sequence>
    </AbsoluteFill>
  );
};
