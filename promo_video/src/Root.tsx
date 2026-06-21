import React from 'react';
import {Composition} from 'remotion';
import {PromoVideo} from './Composition';
import {JourneyHighlights, JOURNEY_FRAMES} from './JourneyComposition';
import {SeriesTrailer, SERIES_TRAILER_FRAMES} from './SeriesTrailer';
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
      <Composition
        id="JourneyHighlights"
        component={JourneyHighlights}
        durationInFrames={JOURNEY_FRAMES}
        fps={FPS}
        width={WIDTH}
        height={HEIGHT}
      />
      <Composition
        id="SeriesTrailer"
        component={SeriesTrailer}
        durationInFrames={SERIES_TRAILER_FRAMES}
        fps={FPS}
        width={WIDTH}
        height={HEIGHT}
      />
    </>
  );
};
