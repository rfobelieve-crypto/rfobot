import React from 'react';
import {Composition} from 'remotion';
import {PromoVideo} from './Composition';
import {FPS, WIDTH, HEIGHT, TOTAL_FRAMES} from './data/script';

export const Root: React.FC = () => {
  return (
    <>
      <Composition
        id="PromoVideo"
        component={PromoVideo}
        durationInFrames={TOTAL_FRAMES}
        fps={FPS}
        width={WIDTH}
        height={HEIGHT}
      />
    </>
  );
};
