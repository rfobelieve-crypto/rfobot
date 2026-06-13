import React from 'react';
import {AbsoluteFill, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {GradientBG} from '../components/Primitives';

const FONT = '"Noto Sans TC", "Microsoft JhengHei", sans-serif';

const INCIDENTS = [
  {date: '2026-06-05', name: 'int() 假性槓桿', c: '#FF6B6B',
   symptom: '$89 帳戶被開出 ~7x 名目槓桿',
   root: '張數被無條件捨去成整數，小帳戶遭「湊整」反向放大',
   fix: '分數合約 sizing + presubmit guard（>3x 拒發單）'},
  {date: '2026-06-04 / 07', name: 'admin_heal 連環誤觸', c: '#FF9F43',
   symptom: '好倉的本地紀錄被歸零 ×2 次',
   root: '破壞性操作掛在 GET — link-preview 預取自動觸發',
   fix: 'POST-only + 執行前先查交易所實倉，有倉直接拒絕'},
  {date: '2026-05-31', name: 'Decorator 被插隊', c: '#BC8CFF',
   symptom: 'Telegram bot 全部指令靜默死亡 10 小時',
   root: '插入函式拆散 @app.route 綁定 — 測試全過、上線才炸',
   fix: 'url_map 綁定檢查納入部署流程'},
];

export const JourneyIncidents: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const titleO = spring({frame, fps, config: {damping: 200}});
  const footO = spring({frame: Math.max(0, frame - 430), fps,
                        config: {damping: 200}});

  return (
    <GradientBG from="#0a0e1a" to="#1f1410">
      <AbsoluteFill style={{fontFamily: FONT, padding: 80}}>
        <div style={{opacity: titleO,
                     transform: `translateY(${(1 - titleO) * 30}px)`}}>
          <div style={{color: '#FFD93D', fontSize: 28, fontWeight: 700,
                       letterSpacing: '0.15em'}}>
            POST-MORTEM
          </div>
          <h1 style={{color: '#fff', fontSize: 68, margin: '12px 0 0 0',
                      fontWeight: 800}}>
            三次真錢事故，三個機制化修補
          </h1>
        </div>

        {INCIDENTS.map((inc, i) => {
          const d = 50 + i * 120;
          const o = spring({frame: Math.max(0, frame - d), fps,
                            config: {damping: 17, stiffness: 65}});
          return (
            <div key={inc.name}
                 style={{position: 'absolute', left: 80 + i * 590, top: 300,
                         width: 555, height: 560,
                         opacity: o,
                         transform: `translateY(${(1 - o) * 60}px)`,
                         background: '#161B22', borderRadius: 20,
                         padding: '34px 36px',
                         border: `1px solid ${inc.c}40`}}>
              <div style={{color: '#7A828E', fontSize: 22, fontWeight: 700}}>
                {inc.date}
              </div>
              <div style={{color: inc.c, fontSize: 38, fontWeight: 800,
                           margin: '8px 0 22px 0'}}>
                {inc.name}
              </div>
              <div style={{color: '#fff', fontSize: 25, lineHeight: 1.5,
                           marginBottom: 20}}>
                {inc.symptom}
              </div>
              <div style={{color: '#B0B8C4', fontSize: 23, lineHeight: 1.55,
                           marginBottom: 20}}>
                根因：{inc.root}
              </div>
              <div style={{borderTop: '1px solid #21262D', paddingTop: 18,
                           color: '#3FB950', fontSize: 23, lineHeight: 1.5,
                           fontWeight: 700}}>
                修補：{inc.fix}
              </div>
            </div>
          );
        })}

        <div style={{position: 'absolute', left: 80, right: 80, bottom: 50,
                     opacity: footO, textAlign: 'center'}}>
          <span style={{color: '#3FB950', fontSize: 32, fontWeight: 800}}>
            三次事故的發現者，都是對帳系統 — 不是人
          </span>
        </div>
      </AbsoluteFill>
    </GradientBG>
  );
};
