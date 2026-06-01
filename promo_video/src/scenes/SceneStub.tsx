import React from 'react';
import {AbsoluteFill, useCurrentFrame, useVideoConfig} from 'remotion';
import {SCENES} from '../data/script';
import {GradientBG} from '../components/Primitives';

/**
 * Placeholder scene — shown for scenes 5–14 until they get a proper
 * component.  Displays scene metadata + voiceover so you can iterate
 * the spoken track in Remotion Studio without a final visual.
 *
 * To turn a stub into a real scene:
 *   1. Copy Scene01ColdOpen.tsx as the template
 *   2. Wire it into SCENE_COMPONENTS in Composition.tsx
 */
export const SceneStub: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps, width} = useVideoConfig();

  // Identify which scene we're inside by checking the global frame.
  // (Sequence in Composition gives us local frame, so we need parent.
  // Workaround: read SCENES and find the one whose duration matches.)
  // For preview clarity we just show "STUB SCENE".
  return (
    <GradientBG from="#1a1f2e" to="#0a0e1a">
      <AbsoluteFill style={{justifyContent: 'center',
                            alignItems: 'center',
                            flexDirection: 'column',
                            gap: 24,
                            fontFamily: '"Noto Sans TC", sans-serif'}}>
        <div style={{
          fontSize: 28, color: '#475569', letterSpacing: '0.2em',
        }}>STUB SCENE</div>
        <div style={{
          fontSize: 48, color: '#e2e8f0', fontWeight: 700,
          textAlign: 'center', maxWidth: 1400,
        }}>視覺待補</div>
        <div style={{
          fontSize: 22, color: '#64748b', marginTop: 12,
        }}>see data/script.ts SCENES[].visual_hint for direction</div>
      </AbsoluteFill>
    </GradientBG>
  );
};
