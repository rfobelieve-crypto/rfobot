import React from 'react';
import {
  AbsoluteFill, Sequence, Img, staticFile, spring,
  useCurrentFrame, useVideoConfig, interpolate,
} from 'remotion';

const FONT = '"Noto Sans TC", "Microsoft JhengHei", sans-serif';
const FPS = 30;

// ---- timing (frames) ----
const T_TITLE = 75;     // 2.5s
const T_THESIS = 75;    // 2.5s
const T_CARD = 78;      // 2.6s each
const T_CLOSE = 96;     // 3.2s

type Card = {ep: string; img: string; hook: string; light: boolean};
const CARDS: Card[] = [
  {ep: 'EP01', img: 'series/ep1_origin-bot.png',       hook: '從一個只會數買賣量的機器人開始', light: false},
  {ep: 'EP05', img: 'series/ep5_insample-trap.png',    hook: '我差點用一個作弊的測試騙過自己', light: true},
  {ep: 'EP08', img: 'series/ep8_impossible-target.png', hook: '我追的目標，數學上根本不存在', light: true},
  {ep: 'EP11', img: 'series/ep11_isolation-lesson.png', hook: 'kill switch 救不了操作者本人', light: true},
  {ep: 'EP13', img: 'series/ep13_ast-guard.png',       hook: '同一個盲點犯三次，我寫了會自動抓它的測試', light: false},
  {ep: 'EP15', img: 'series/ep15_gate-a.png',          hook: '我量化了自己，到底作弊多少', light: true},
];

export const SERIES_TRAILER_FRAMES =
  T_TITLE + T_THESIS + CARDS.length * T_CARD + T_CLOSE;

const GRADIENT = 'linear-gradient(135deg, #0a0e1a 0%, #14223f 100%)';
const ACCENT = '#56D4E0';

const useFades = (dur: number, inF = 12, outF = 12) => {
  const f = useCurrentFrame();
  const fin = interpolate(f, [0, inF], [0, 1], {extrapolateRight: 'clamp'});
  const fout = interpolate(f, [dur - outF, dur], [1, 0], {extrapolateLeft: 'clamp'});
  return Math.min(fin, fout);
};

const TitleCard: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const o = spring({frame, fps, config: {damping: 200}});
  const op = useFades(T_TITLE);
  const w = interpolate(o, [0, 1], [0, 360]);
  return (
    <AbsoluteFill style={{background: GRADIENT, justifyContent: 'center',
                          alignItems: 'center', fontFamily: FONT}}>
      <div style={{opacity: op, transform: `translateY(${(1 - o) * 30}px)`,
                   textAlign: 'center'}}>
        <div style={{color: ACCENT, fontSize: 30, fontWeight: 700,
                     letterSpacing: '0.32em', marginBottom: 28}}>
          16 篇連載
        </div>
        <h1 style={{color: '#fff', fontSize: 104, fontWeight: 800,
                    lineHeight: 1.18, margin: 0, letterSpacing: '0.01em'}}>
          從訂單流機器人<br />到自動交易系統
        </h1>
        <div style={{height: 5, width: w, background: ACCENT, borderRadius: 3,
                     margin: '40px auto 0'}} />
        <p style={{color: '#9aa5c4', fontSize: 34, marginTop: 36}}>
          一個 BTC 量化系統的演化日誌
        </p>
      </div>
    </AbsoluteFill>
  );
};

const ThesisCard: React.FC = () => {
  const op = useFades(T_THESIS);
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const o = spring({frame, fps, config: {damping: 200}});
  return (
    <AbsoluteFill style={{background: GRADIENT, justifyContent: 'center',
                          alignItems: 'center', fontFamily: FONT, padding: '0 160px'}}>
      <h1 style={{opacity: op, transform: `translateY(${(1 - o) * 26}px)`,
                  color: '#fff', fontSize: 64, fontWeight: 800, lineHeight: 1.5,
                  textAlign: 'center', margin: 0}}>
        一個只想預測方向的指標，<br />
        怎麼變成一個<span style={{color: ACCENT}}>會自己關掉自己</span>的系統。
      </h1>
    </AbsoluteFill>
  );
};

const ImageCard: React.FC<{card: Card}> = ({card}) => {
  const f = useCurrentFrame();
  const op = useFades(T_CARD, 12, 10);
  const scale = interpolate(f, [0, T_CARD], [1.0, 1.06]);
  const hookO = interpolate(f, [10, 26], [0, 1], {extrapolateRight: 'clamp'});
  const hookY = interpolate(f, [10, 26], [24, 0], {extrapolateRight: 'clamp'});
  return (
    <AbsoluteFill style={{background: GRADIENT, fontFamily: FONT,
                          justifyContent: 'space-between', alignItems: 'center',
                          padding: '56px 0 64px'}}>
      <div style={{opacity: op, color: ACCENT, fontSize: 26, fontWeight: 700,
                   letterSpacing: '0.3em'}}>
        {card.ep}
      </div>
      <div style={{opacity: op, transform: `scale(${scale})`,
                   display: 'flex', justifyContent: 'center', alignItems: 'center',
                   width: '100%', flex: 1, padding: '12px 0'}}>
        <Img src={staticFile(card.img)}
             style={{maxWidth: '76%', maxHeight: '100%', objectFit: 'contain',
                     borderRadius: 16,
                     background: card.light ? '#fff' : 'transparent',
                     boxShadow: '0 24px 70px rgba(0,0,0,0.55)',
                     border: '1px solid rgba(255,255,255,0.08)'}} />
      </div>
      <div style={{opacity: hookO, transform: `translateY(${hookY}px)`,
                   color: '#fff', fontSize: 50, fontWeight: 800, lineHeight: 1.35,
                   textAlign: 'center', maxWidth: '82%'}}>
        {card.hook}
      </div>
    </AbsoluteFill>
  );
};

const ClosingCard: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const o = spring({frame, fps, config: {damping: 200}});
  const op = useFades(T_CLOSE, 14, 14);
  const subO = interpolate(frame, [22, 40], [0, 1], {extrapolateRight: 'clamp'});
  return (
    <AbsoluteFill style={{background: GRADIENT, justifyContent: 'center',
                          alignItems: 'center', fontFamily: FONT, padding: '0 150px'}}>
      <div style={{opacity: op, transform: `translateY(${(1 - o) * 26}px)`,
                   textAlign: 'center'}}>
        <h1 style={{color: '#fff', fontSize: 70, fontWeight: 800, lineHeight: 1.45,
                    margin: 0}}>
          每一個錯誤，<br />
          換成一條<span style={{color: ACCENT}}>寫進程式碼</span>的規則。
        </h1>
        <p style={{opacity: subO, color: '#9aa5c4', fontSize: 32, marginTop: 48,
                   letterSpacing: '0.06em'}}>
          16 篇連載　·　每週二、四更新
        </p>
      </div>
    </AbsoluteFill>
  );
};

export const SeriesTrailer: React.FC = () => {
  let at = 0;
  const seqs: React.ReactNode[] = [];
  seqs.push(
    <Sequence key="title" from={at} durationInFrames={T_TITLE} name="Title">
      <TitleCard />
    </Sequence>);
  at += T_TITLE;
  seqs.push(
    <Sequence key="thesis" from={at} durationInFrames={T_THESIS} name="Thesis">
      <ThesisCard />
    </Sequence>);
  at += T_THESIS;
  CARDS.forEach((card, i) => {
    seqs.push(
      <Sequence key={`card${i}`} from={at} durationInFrames={T_CARD}
                name={card.ep}>
        <ImageCard card={card} />
      </Sequence>);
    at += T_CARD;
  });
  seqs.push(
    <Sequence key="close" from={at} durationInFrames={T_CLOSE} name="Closing">
      <ClosingCard />
    </Sequence>);
  return <AbsoluteFill style={{backgroundColor: '#0a0e1a'}}>{seqs}</AbsoluteFill>;
};
