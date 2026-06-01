import React from 'react';
import {AbsoluteFill, Sequence, staticFile, Audio} from 'remotion';
import {SCENES} from './data/script';
import {Subtitles} from './components/Subtitles';
import {Scene01ColdOpen} from './scenes/Scene01ColdOpen';
import {Scene02Hook} from './scenes/Scene02Hook';
import {Scene03Problem} from './scenes/Scene03Problem';
import {Scene04Architecture} from './scenes/Scene04Architecture';
import {SceneStub} from './scenes/SceneStub';

const SCENE_COMPONENTS: Record<number, React.FC> = {
  1: Scene01ColdOpen,
  2: Scene02Hook,
  3: Scene03Problem,
  4: Scene04Architecture,
  // 5–14 fall through to SceneStub (you flesh them out using the same
  // pattern as Scene01-04; the script in data/script.ts already has
  // voiceover + visual_hint per scene).
};

/**
 * Audio resolution per scene.
 * Drop scene_NN.mp3 (NN zero-padded) into public/audio/ and the
 * voice track gets layered onto the timeline automatically.
 * If a file is absent, that scene runs silent (useful while iterating).
 */
const audioPath = (id: number) =>
  staticFile(`audio/scene_${String(id).padStart(2, '0')}.mp3`);

export const PromoVideo: React.FC = () => {
  return (
    <AbsoluteFill style={{backgroundColor: '#0a0e1a'}}>
      {SCENES.map((scene) => {
        const Comp = SCENE_COMPONENTS[scene.id] ?? SceneStub;
        return (
          <Sequence
            key={scene.id}
            from={scene.start}
            durationInFrames={scene.end - scene.start}
            name={`Scene ${scene.id}: ${scene.title}`}
          >
            <Comp />
            {/* Per-scene voiceover — file may not exist while iterating. */}
            <Audio
              src={audioPath(scene.id)}
              volume={1}
              // Remotion will warn if the file is missing; that's OK in dev.
            />
          </Sequence>
        );
      })}
      {/* Subtitles layer sits above all scenes */}
      <Subtitles />
    </AbsoluteFill>
  );
};
