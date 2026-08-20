# Mistake Log

Record logic errors and bad decisions to avoid repeating them.

---

# Product-Site（前端網站建置）

跟下面量化系統本體的錯誤是不同類別——這裡記的是 `product-site`（Next.js
行銷/產品網站，獨立 repo，Vercel 部署）建置過程踩的坑。2026-07-22 這天
從「幫圖表接上網站」一路做到「首頁標題特效」，密度很高，獨立開一類。

## 2026-08-01: 累積 netR 曲線整組靜默消失——補丁引用了還沒定義的變數，一個空的 `except: pass` 把它吞了

**What happened:**
給 shadow-review 的曲線加「每條線都從凍結日的 0 開始、平延到現在」的補丁時，
在 `cum_series()` 內部引用了 `now_ts`，但那個變數是在函式**後面**才定義的。
Python 丟 `NameError`，而外層包著一個 `try: ... except: pass`（原意是「圖畫
不出來也不要害整頁掛掉」）——於是**兩個曲線面板整組不見，頁面其他部分完全
正常**，沒有任何錯誤訊息。我 push 了，過了一段時間使用者才回報「曲線怎麼都
不見了我要有曲線統計啊」。

**Root cause:**
兩件事疊起來才致命：(a) 補丁寫在函式內、引用了定義在下面的名字——這在
Python 只有執行到那一行才炸；(b) **裸 except + pass**，把「這段功能整個死掉」
偽裝成「這段功能沒有內容」。頁面其餘部分照常渲染，所以看起來像是資料不夠，
不像是壞掉。這是 [[mistake 2026-04-22 silent failure]] 的前端版：進程活著、
頁面 200、功能死亡。

**Correct approach（已修）:**
改用 `bars[-1][0]`（現成的、必然已定義的最後一根 K 時間）當右端點；並把
`except: pass` 改成 `except Exception as e: print("[WARN] equity curves failed:", e)`
——渲染仍然不會整頁掛掉，但失敗會在產生頁面的 log 留下痕跡。

**Rule:** 渲染/輸出層的 `try/except` **永遠不可以是靜默的**——至少要印一行
WARN，否則「功能消失」和「沒有資料」在畫面上長得一模一樣。往既有函式裡插
補丁時，先確認引用的每個名字在**插入點**已經定義（不是在檔案裡存在就好）。
自己 push 的視覺改動，push 前要真的看一眼產出的頁面：這次有一版壞掉的圖表
在線上掛了好幾個小時，只因為我相信「程式沒報錯就是好的」。

---

## 2026-08-01: i18n 短錨點命中了更深層縮排的同名字串，整個命名空間被塞進別人肚子裡

**What happened:**
用 Python 腳本把新的 `ledger` 命名空間插進 `messages/zh.json`／`en.json`，
錨點寫成 `'  "signals": {'`（兩格縮排）。這個字串同時也是**更深層**某個區塊
裡某一行的**子字串**，`str.index()` 命中了那個位置——`ledger` 就被插進
`explore` 裡面變成巢狀。網站上整片顯示 `LEDGER.CLOCKS.…` 這種原始 key，
使用者截圖問「這是什麼」。

**Root cause:**
用字串比對做 JSON 手術時，把「縮排」當成了唯一性保證。縮排只在**同一層**
唯一；子字串比對不管層級。而且插入後我只確認了「檔案還是合法 JSON」，沒有
確認「這個 key 落在第幾層」——合法 ≠ 正確。

**Correct approach（已修）:**
用 `json.JSONDecoder().raw_decode` 找到頂層物件的實際邊界，把 `ledger` 搬回
第一層；插入後**斷言 `json.load(f)["ledger"]["clocks"]` 取得到值**（不只是
`json.load` 不拋例外）。

**Rule:** 對結構化檔案（JSON/YAML/TOML）做程式化編輯，**不要用字串錨點**——
用真正的 parser 讀進來、改物件、寫回去。萬不得已要用字串錨點，錨點必須**長
到全檔唯一**，而且改完要斷言**目標路徑**取得到值，不是只斷言檔案還能 parse。
同一個坑的近親：[[2026-05-31 Edit 把函式插進 decorator 和 def 中間]]——都是
「文字上看起來對，結構上放錯層」。

---

## 2026-07-22: Vercel 部署失敗——盲猜帳號登入 3 輪，真正原因是方案限制 + 舊接錯的 Git 整合

**What happened:**
幫 product-site 第一次連 GitHub + 部署到 Vercel，選了私有 repo。部署一直
失敗，錯誤只寫「not a member of the team」。我第一直覺是 CLI 登入身分不對
（`rfobelieve-crypto` vs `rfobelieve2@gmail.com`），來回試了 `vercel logout`
/`vercel login --github`、重推 3 次，每次都失敗、每次都送一封失敗信。
使用者截圖轉發 Vercel 網頁後台的實際錯誤畫面後才看到真正訊息：
「Deployment Blocked — commit author 對這個 project 沒有 contributing
access。The Hobby Plan does not support collaboration for private
repositories.」——這個 project 掛在 team scope 底下，team 是免費 Hobby
方案，Hobby 方案的規則就是**team scope + 私有 repo = 一律擋**，跟登入
哪個帳號完全無關。

同一個 session 稍後，flow_system（另一個完全不相關的 repo）一次
正常 push 後，使用者的 Gmail 收到一封「product-site Build Failed，
Command "npm run vercel-build" exited with 1」的信。查證後發現：這個
Vercel project 的 Git 整合早就（在今天連 product-site 的 GitHub repo
**之前**）連到了 `rfobelieve-crypto/rfobot`（flow_system 的 repo）
main 分支——大概是 Vercel project 剛建立時 product-site 的 GitHub repo
根本還不存在，順手接到了當時唯一能選的 repo。這條接錯的線一直沒人
發現，直到今天 push flow_system 才第一次真正觸發。

**Root cause:**
1. 看到「not a member of the team」這種帳號權限風味的錯誤訊息，直覺
   跳去查登入身分，但**沒有先去 Vercel 部署頁本身看完整錯誤內容**——
   那頁一開始就寫著真正原因（方案限制），CLI 這邊的錯誤訊息是被
   截斷/簡化過的版本。
2. 從沒有主動查過這個 Vercel project 的 Git 整合實際指向哪個 repo——
   一直假設「我剛連的那個就是唯一一個」，沒想過專案建立當下可能
   已經連了別的東西。

**Correct approach（已修）:**
- 私有 repo + team scope 撞牆：查證這個帳號類型**沒有個人 scope 可切換**
  （`personal_scope_not_allowed`）後，改把 GitHub repo 從私有改公開
  （`gh repo edit --visibility public`）解決，不用花錢升級方案。
- 接錯的 Git 整合：直接用 Vercel API（`GET /v9/projects/{id}`）查
  `link` 欄位確認真正連到哪個 repo，找到後 `vercel git disconnect`
  斷開，之後 flow_system push 不會再誤觸發假失敗信。

**Rule:** 看到任何「權限/帳號」風味的部署錯誤，**第一步是去看該平台
自己的部署詳情頁**（Vercel/Netlify/Railway 等都有），不是先猜身分
重登入——CLI 回傳的錯誤訊息經常被截斷，網頁後台通常有完整原因跟
解法連結。連結一個新 GitHub repo 到既有的付費平台 project 前，**先用
API 或後台查證這個 project 現有的 Git 整合實際指向哪裡**，不要假設
它是乾淨的——項目可能在很久以前就被接到別的地方，且这类误接不会
主动报错，只在下次不相关的 push 命中时才现形。

---

## 2026-07-22: `next build` 跟 `next dev` 共用 `.next` 目錄互相打架，同一個坑踩了三次

**What happened:**
同一個 session 裡，我習慣在改完程式碼後跑 `npm run build` 做最終確認，
但背景常常還留著一個更早開的 `npm run dev`（同一個專案目錄，共用
`.next/`）。至少三次，`build` 跑完後，那個還活著的 `dev` server 就開始
報 `Cannot find module './vendor-chunks/xxx.js'` 或
`TypeError: __webpack_modules__[moduleId] is not a function`，
截圖驗證因此撞見 Next.js 錯誤畫面而不是真正的頁面——一度誤以為是
自己剛寫的程式碼壞了，浪費時間排查了根本沒问题的 component。

另一個疊加的小狀況：每次 `rm -rf .next && npm run dev` 重開，只要
port 3000 被前一輪沒清乾淨的孤兒 node 進程佔用，Next.js 會**靜默**
改用 3001/3002/3003…，而我的 Playwright 測試腳本還在打舊 port，
得到的 404/500 一開始也被誤讀成「程式碼壞了」。

**Root cause:**
`next dev` 的 webpack 模組編號跟 `next build` 產生的正式建置輸出
不是同一套 module id 對照表；兩者同時寫入同一個 `.next/` 目錄，
dev server 之後的熱重載會讀到跟自己內部狀態對不上的模組登記表。
Port 假設錯誤則是純粹沒有去讀 `npm run dev` 自己印出來的實際 port
就直接寫死在測試腳本裡。

**Correct approach（已固定成習慣）:**
- 需要跑 `next build` 驗證時，先確認/停掉背景的 `dev` server 再跑；
  build 完之後如果還要繼續用瀏覽器驗證，`rm -rf .next` 乾淨重開一個
  新的 `dev`。
- 每次重開 dev server 後，**先讀它自己印出的 port**（可能因為前一個
  沒清乾淨而跳號），再更新截圖腳本裡的 URL，不要沿用上一輪的 port。
- 順手清掉確認是自己這個 session 產生的孤兒進程（`Stop-Process`），
  避免下一輪又跳號。

**Rule:** 同一個專案目錄下，`next dev` 跟 `next build` 不能同時活著
共用 `.next/`——要嘛先關 dev 再 build，要嘛 build 完清快取重開 dev。
看到 `__webpack_modules__ is not a function` 或
`Cannot find module './vendor-chunks/...'` 這類錯誤，**先假設是
dev/build 快取衝突**，重開乾淨的 dev server 再重新判斷，不要立刻
懷疑自己剛寫的程式碼。每次起新 dev server 都要讀它實際印出的 port，
不要沿用假設。

---

## 2026-07-22/23: 兩個移植的 Canvas 文字特效都踩到同一個坑——`ctx.font` 不吃 CSS variable

**What happened:**
從 21st.dev/originkit 移植了兩個外部 Canvas/WebGL 文字特效組件
（PixelDrift 粒子字、後來換成的 MeshTextHover 網格扭曲字），兩次
都把 `fontFamily="var(--font-display)"` 這樣的 CSS 變數字串直接傳給
`ctx.font = "700 100px " + fontFamily`。第一次（PixelDrift）的症狀是
中文標題渲染出來只剩零星幾個點、幾乎看不見，一度以為是取樣密度或
中文筆劃太細的問題，花了一輪排查（包括另外寫一支腳本單獨畫「原始
字形」出來看）才確認：**字型從頭到尾沒有真的套用**，Canvas 默默
退回極小的預設字級，取樣自然抓不到什麼。修完 PixelDrift 之後，
換成 MeshTextHover 時，同一行程式碼模式又原封不動地移植過去，
同一個坑等於預先埋好等著再踩一次（這次在寫的當下就直接照搬第一次
的修法補上了，但事後看，**如果一開始就把這個當成「移植任何 canvas
文字效果都要檢查的固定步驟」，第二次根本不用重新推導一次）。

**Root cause:**
`CanvasRenderingContext2D.font` 是純字串屬性，瀏覽器用 CSS 字型
簡寫語法解析它，但**不會**解析 CSS 自訂屬性（`var(--x)`）——不是
報錯，是整段賦值被判定成不合法語法後**靜默不生效**，context 保留
先前的字型設定（通常是極小的預設值）。這是 Canvas 2D API 眾所皆知
但容易忘記的限制：CSS 的東西進 Canvas 之前，變數/計算式一律要先
被瀏覽器解析成字面值。

**Correct approach（兩次都用同一招，已驗證有效）:**
把 `fontFamily` 同時當成一個真實 DOM 元素（外層 wrapper div）的
inline style 設定上去，讓瀏覽器透過正常 CSS cascade 解析 `var()`；
再用 `window.getComputedStyle(wrapper).fontFamily` 讀出解析後的
**字面值字串**，這個才是真正安全餵給 `ctx.font` 的版本。

**Rule:** 任何要把 CSS 變數（字型、顏色、間距皆同理）交給 Canvas 2D
或 WebGL 的地方，**一律先透過一個真實掛載的 DOM 元素 + `getComputedStyle`
解析成字面值**，絕不要把 `var(--x)` 字串直接傳給 canvas API。第一次
修完這類坑之後，要馬上把它記成「移植任何 canvas 文字/繪圖組件」的
標準檢查項，不要指望「這次應該不會忘記」——這次確實沒忘記，是因為
剛好前一個組件才修過、記憶猶新，換一個更久之後的情境完全可能重演。

---

## 2026-07-22: 首頁標題特效——追著同一個技術路線調參數三輪，真正解法是換掉整個技術路線

**What happened:**
使用者要求把首頁標題換成 originkit 的「Pixel Drift」粒子字特效
（英文 Latin 字體的展示 demo）。移植後中文標題渲染出來是一團幾乎
看不出字形的雜訊點——追查發現原組件把取樣密度硬性上限鎖在
`particleCount<=50`（換算最細只有 3px 取樣間距），這個上限是針對
**英文粗體**設計的，對中文細密筆劃完全不夠。拿掉上限、把中文取樣
密度開到最高後，桌面版清楚了，但手機寬度下（8 個中文字擠進
~340px）密度開到最高還是不夠——這是取樣間距的物理極限撞到「字太小」
的天花板，不是能再調參數解決的問題。第一個解法是手機版整個退回
靜態文字（放棄特效），使用者拿起手機看到「怎麼沒動靜」才發現這個
取捨沒有講清楚。討論後使用者自己點出真正解法：**改標題文案本身**，
把長句拆成天然更短的行（後來又換成「會犯錯的AI交易」這個新標語），
手機也能上特效。做完這輪，使用者又回報「畫質還是偏低」「顆粒感
還是太重」——依序修了 dpr 上限（2→3）、粒子相對字體的比例
（particleSize 依螢幕層級調整）、方形粒子改圓形——每次都有感改善，
但每次都不是「解決問題」，只是「把同一個根本限制往後推一點」。
最後使用者直接指名要換成 MeshTextHover（連續材質貼圖扭曲，不是
離散粒子取樣）——這個技術路線從根本上沒有取樣密度/顆粒感這個
問題，前面三輪的所有補丁（手機斷行、particleSize 補償、dpr、
方圓形）全部一次作廢，因為問題的載體被整個換掉了。

**Root cause:**
特效效果分兩層：Latin demo 沒驗證過中文（正常，demo 作者根本
不會想到這件事）；但更深一層是我自己**把每一輪「還是不夠好」的
使用者回饋都當成「這個參數還沒調對」去處理**，而不是在第二、
第三輪類似性質的回饋出現時停下來問「是不是這整個技術做法本身就有
天花板」。離散粒子取樣文字，物理上必然有「取樣密度 vs 字體大小」
的比值下限，低於這個下限不管怎麼調都會顆粒化——這是這個技術類別
（sampling-based）內建的限制，不是這次實作特有的 bug，多花一輪
去意識到這件事，就能提早問使用者「要不要考慮換一種完全不同的做法」
而不是自己悶頭再調一輪參數。

**Correct approach:**
换到 MeshTextHover 之後（真實 Canvas2D 文字渲染貼到 WebGL 網格，
只做幾何扭曲不做像素取樣），中文在任何螢幕寬度下都跟桌面版等
清晰度——因為根本沒有「取樣密度不夠」这个概念存在。手機不用再
特別拆短句，直接跟桌面共用同一份文案，只是 fontSize 依斷點調整。
所有 PixelDrift 專屬的補丁程式碼（含元件本身）直接刪除，不留
「以防萬一」的殘骸。

**Rule:** 同一個視覺/渲染方向連續 2-3 輪收到「還是不夠好」的回饋時，
**先問這是不是這個技術類別本身的天花板**，不要預設只是參數沒調對
就急著調下一輪。移植任何為英文/Latin 內容設計的展示型 demo 元件
（字體特效、動畫模板）時，**第一步就用實際要上線的語言內容
（這裡是中文）測過一輪**，不要等使用者在真機上看到才發現——別的
語言的視覺密度/筆劃複雜度可能跟示範用的完全不是一個量級。技術
路線之間的取捨（離散取樣 vs 連續紋理）要主動列出來給使用者選，
不要只沿著第一個選中的路線一路打補丁到底。

---

## 2026-08-11: 解碼把 DOWN 側「算術上」鎖死三個月——buffer 存在容器裡，每次部署都被重置回訓練集內的預測

**What happened:**
08-08 重訓後使用者說「多頭訊號還是很多沒有改善」。查 tracked_signals：部署後
**9 UP : 0 DOWN**，最近 6 筆真實部位全是 LONG。而市場那三天實際漲的 bar 只有
46%，模型卻 74% 預測往上——**重訓前後「過度偏多」完全沒變，都是 +28pp**。

不是機率問題，是**算術問題**：
```
上線後模型輸出範圍          [-0.001480, +0.002425]
DOWN 側最寬鬆的門檻          -0.001786   ← 比模型史上最低值還低
任何 tier 的 DOWN 可觸發 bar: 0/74
```
解碼是「拿今天的預測跟 500 根 buffer 比排名，落在 2.5%/7.5% 尾巴就開火」。
buffer 裝的是模型在**自己訓練集上**的預測（mean −0.000005），而模型上線後
實際落在 mean +0.000716——**偏了 +0.60 個 buffer 標準差**，於是 DOWN 的切點
整個掉到模型構不到的地方。

**為什麼是「三個月」而不是「重訓後幾天」——這是最關鍵的一層：**
buffer 的持久化路徑是**容器內**的 `training_stats.json`。Railway 檔案系統是
暫存的，而這個 repo 每天部署 2-10 次——**每次 push，buffer 都被重置回 git 裡
那 500 筆 in-sample 種子**。它從來沒有真的長成 live 過。我一開始判斷成「重訓
後的暫時現象，21 天會自己輪替掉」，那是錯的：它是持續性的。

**08-08 我自己寫的四道關卡全部漏掉：**
- G4 檢查的是 buffer 的**離散度比值**（0.78，落在 [0.5,2.0] → 過關），
  **比值對位置偏移完全免疫**——跟 rank 指標對水平漂移免疫是同一種盲。
- G2/G3 是在**訓練窗**上重放的，而訓練窗的對中是最小平方法的**建構保證**。
  使用者當天就說了「但是是 in sample 的資料」，我沒把它當成擋關條件。
- **沒有任何一關問「切點在不在模型輸出得到的範圍內」**——那是唯一一個
  一問就會發現的問題，而且答案是確定的、不是機率的。

**Root cause:**
跟 [[2026-08-08 水平漂移]] 同一個根：**解碼的參考分佈不是它要判斷的那個
分佈**。只是換了一種錯法——那次是 buffer 對、模型漂走；這次是模型對、
buffer 從來沒對過。任何「跟自己的歷史比排名」的解碼都有這個結構性弱點，
而它的失效**不會拋例外、不會讓數字看起來畸形**，只會安靜地變成單邊。

**Correct approach（已修）:**
1. **buffer 改從 DB 重建**（`app.py:_rehydrate_pred_buffer`）：開機時讀
   `indicator_history` 最近 N 筆、**限定當前 model_version** 的 pred。
   容器重啟不再洗掉它，而且 buffer 依定義就是 live 分佈——**兩側可達性
   變成建構上的保證**，不再需要祈禱。
2. **export 不再 seed**（改寫空陣列），暖機期**不開火**而不是退回 WF 固定
   門檻（重放顯示暖機期正是偏斜所在：五個 reset 點各是 53:17、64:4、
   54:20、65:2、43:5）。代價是重訓後約 100 根（~4 天）靜默。
3. **窗口 500 → 200**：重放全程 UP 佔比 60% → 51%，方向命中 50.0% → 52.1%。
4. **G4 反轉 + 新增 G5 可達性**：G4 現在要求 committed buffer 必須是**空的**；
   G5 直接算「兩側各有幾根 bar 構得到切點」，Strong 兩側各 ≥1、Moderate
   兩側各 ≥3。這一關是純算術，過不了就是確定會出事，不是風險。
5. **`OKX_ENTRY_PAUSED`**（新旗標）：只擋開新倉，出場/對帳/kill 照跑。
   當時有一筆未平倉部位，`OKX_EXECUTOR_ENABLED=0` 會讓它的移動停損變孤兒。

**儀器紀律（這次做對的一件事）：** 第一版重放 harness 說 8 月是 14:18（平衡），
與實際 9:0 矛盾。**我沒有解讀它，先查了產生它的程式碼**——漏了「部署會用 seed
**取代** buffer」這個動作，重放讓 seed 早就被輪替掉了。修好後從 08-08 reset
重放得 11:0，重現了病灶，harness 才拿來排序修法。這條現在寫死在檔案裡：
**重現不了已知病灶的 harness，不能用來排序修法**。

**Rule:** 任何「跟自己的歷史比排名」的解碼，**必須有一關直接問「切點落在
被判斷對象的輸出範圍內嗎」**——比值、離散度、rank 指標全都測不到這件事，
而它是唯一一個失效時是**確定**而非機率的環節。滾動狀態（buffer、warmup、
校準）**不可以存在容器檔案系統**：每次部署都會重置，而重置回的是**寫進
git 的那份**，於是「暫時的偏差」變成「每天重演的常態」。判斷任何滾動狀態
是否健康，看它**實際跑起來的分佈**，不看它被 seed 成什麼樣子。**在訓練窗
上做的對中檢查沒有證明力**——最小平方法保證它會對中。

---

## 2026-08-08: 模型輸出水平漂移 3 個月無人發現——rank 指標對平移全盲，兩尾解碼把偏置轉成方向偏斜

**What happened:**
使用者反映「V7 最近太差了」。追查發現：生產模型（2026-05-01 訓練後凍結）的
**輸出整體水平**在四個月裡從 −0.0014 漂到 +0.0012（漂了 +0.0024 = 絕對 floor
的 3 倍）。兩尾 rolling percentile 解碼是「跟自己過去 500 根比排名」——分佈
持續上漂時，今天的 pred 幾乎永遠高於 trailing 窗口（那裡塞滿舊的低值）→
**UP 訊號狂發、DOWN 訊號餓死**（7 月 Strong 開火 14 UP:1 DOWN、8 月 20:1）。
而系統的 edge 在空側（live SHORT +38bps vs LONG −27bps）——解碼偏斜把
executor 鎖在自己的弱側三個月。連帶 opp_signal 出場也被餓死（持倉中等反向
Strong 的機率隨頻率崩掉，17 筆平倉 12 筆淪為 trail_stop 收尾）。

**為什麼三個月沒人發現——儀器對這種病結構性全盲：**
1. **月度復驗用 Spearman IC / sign_AUC，全是 rank 指標，對整體平移完全免疫**
   （全班每人加 10 分，名次不變）。復驗月月 PASS，病在體檢看不到的刻度上。
2. **復驗的 walk-forward 逐折重訓，每折的 pred 自我對中**——只有凍結的生產
   流才顯示漂移，而復驗從不看生產流。
3. 沒有任何規則要求重訓：模型呆了 99 天，「復驗 PASS」被誤讀成「不用動」。

**Root cause:**
凍結模型 + 漂移的特徵分佈 = 輸出零點歪掉（像三個月沒歸零的體重計）。
rolling percentile 解碼假設 pred 分佈大致平穩；在持續趨勢下它退化成
「偵測模型自己輸出的動量」，把水平偏差轉譯成方向訊號。

**Correct approach（已修，commit 9d319e9 / c98a819）：**
1. **maintenance refresh**：同特徵、同超參數、同 tier 定義，只用新資料重訓
   （不是研究，是校準）。四關預先寫死才部署：G1 WF AUC in band、
   **G2 對中 |mean|<0.5×floor 且兩尾各 ≥2% 越過 floor**、G3 解碼重放
   DOWN 佔比 30-70%、G4 buffer std 比 [0.5,2]。結果：mean +0.00118→+0.00011，
   DOWN 佔比 7%→59%。
2. **三層防護**：復驗加 §2b 直接量**生產 pred 流**的水平（不是 WF OOS）；
   portfolio_clocks 加週檢「21d 開火方向 ≥85% 單邊即告警」（症狀層，天級）；
   CLAUDE.md 寫死 **60 天重訓上限**。第 2 層上線當天就對舊模型的 50:3 正確
   開火——如果它三個月前就在，不需要等使用者「感覺到爛」。

**Rule:** 任何「跟自己的歷史比排名」的解碼/正規化（rolling percentile、
z-score、分位數門檻），都隱含「分佈平穩」的假設——**上游模型凍結而市場
在漂時，這個假設必然壞掉，而且 rank 指標測不到**。監控必須同時看兩層：
排序（IC/AUC）**和刻度**（生產輸出的 mean/尾部佔比）。凍結的生產模型要有
重訓節律上限，「復驗 PASS」只代表排序還在，不代表刻度沒歪。方向訊號系統
的**開火方向比例**本身就是最便宜的健康指標——它偏斜到 85:15 時不需要任何
統計檢定就知道有事。

---

## 2026-08-02: 同一個地形戰役裡兩個儀器 bug——一個被「內部矛盾」抓到，一個被「桶子憑空歸零」抓到

**What happened:**
(a) 最終統計關的腳本裡，三個宣稱各自帶一組 `(壞桶, 好桶)` 的 predicate，A2
（背靠支撐）那組**槽位寫反了**。第一次跑出 −12.1pp、3 個逆風季——跟先前每
一個切片（+12~14pp、兩半穩定）**完全相反**。我沒有直接相信輸出，是因為它跟
既有證據互相矛盾；查了才發現是 tuple 順序。修正後 +12.1pp、p=0.0005。
(b) D4「牆的等級」第一版，用「最近的那個池」決定牆的等級，同距離時的
tie-break 落在確認時間上——結果 **swing 桶 n=0、上週桶 n=0**，全部被 session
吃光。原因是**同一個價位常常同時是好幾種池**（一個 swing high 幾乎必然也是
某個時段的高點），誰先確認純屬記帳細節。修正成「取堆疊標籤中最高等級」後
四桶都有樣本（上週 4 / 昨日 74 / 波段 27 / 時段 154）。

**Root cause:**
兩個都不是分析錯，是**儀器錯**，而且兩個都不會拋例外、不會讓數字看起來畸形
——(a) 給出一個「乾淨的、可發表的」反向結論；(b) 給出一張「桶子分佈很極端」
的合理表格。差別只在於我**碰巧**有先驗可以對照：(a) 跟三輪切片矛盾、(b) 出現
了物理上不可能的 n=0（BTC 一年不可能沒有任何一個 swing high 當過最近的牆）。
沒有先驗的維度，同類 bug 會直接變成結論。

**Correct approach（已做）:**
兩個都在下結論前修掉並重跑。這一輪之後固定兩個習慣：分桶結果先看
**n 的分佈合不合物理**（有桶為 0 或佔比 >90% 一律先當儀器壞掉，D8 就是靠
這條被判「註冊桶退化」而不是硬解讀）；任何「和既有結論反向」的新數字，
**第一動作是查產生它的程式碼，不是解讀它**。

**Rule:** 這是 [[2026-07-29 自己寫的診斷有 bug]] 的同族第 2、3 次——**自己
剛寫的儀器比別人的舊儀器更危險**。對照組要在寫的當下就想好：這次是「連續
市場跳空率必為 0」的地形版——**每個分桶都必須有物理上說得通的樣本量**。
還有一條：**跟先驗矛盾的漂亮結論，和符合先驗的漂亮結論，都要查儀器**——
前者我查了（救回 A2），後者是 2026-07-28/29 連踩兩次的那個坑。

---

## 2026-08-01: 研究線的 Coinglass parquet 停更 5 天，而每日排程全綠——同一個供應商有兩套平行儲存

**What happened:**
準備跑地形研究時發現 `cg_*_1h` parquet 停在 07-28。但 DailyCollect 排程
（07-06 才剛修好路徑、`lastResult=0`）每天都成功。原因：這個 repo 對 Coinglass
有**兩套平行的落地**——生產線寫 `market_data/raw_data/cg_*.parquet`（排程有
跑），研究線讀 `research/` 底下由 `backfill_all_parquet.py` 維護的另一組
（**沒有掛進任何排程**）。07-06 修排程時我修的是前者，後者從來就不在排程裡，
只是先前有人手動跑過所以看起來是新的。

**Root cause:**
把「Coinglass 資料有沒有在更新」當成一個單一問題來回答，實際上它有兩個答案。
排程健康 ≠ 我要讀的那份檔案健康——這跟 [[2026-04-12 is_stale 只檢查 klines]]
是同一個形狀：**用最可靠的那個組件代表全體**。

**Correct approach（已修）:**
`daily_collect.bat` 加 step 1.5 跑 `research/backfill_all_parquet.py`，研究線
納入同一個排程。手動補跑一次把缺口補齊後才開始跑研究。

**Rule:** 開始任何研究之前，先看**你要讀的那個檔案**的 mtime／最後一根資料
時間，不要看排程面板、不要看別的檔案。同一份資料在 repo 裡有兩份拷貝時，
**兩份都要有人負責更新**，否則沒被排程的那份會安靜地腐爛——而且它腐爛時，
排程還是綠的。

---

## 2026-07-29: 我自己寫的診斷有 bug，拿它宣告了一個策略死亡——連續市場出現「跳空」就該立刻停手

**What happened:**
使用者對「掃單失敗在 15/16 個非 crypto 市場重現」表示合理懷疑（「大宗商品
怎麼可能那麼容易找出量化策略」）。我去查 Yahoo 期貨資料的已知陷阱，寫了
`xasset_diagnostics.py`，其中跳空進場的判定寫成「**進場棒的開盤是否已越過
價位**」。結果顯示 53-62% 的進場是「跳空」，清理後合池從 +0.0405 (t+5.10)
翻成 −0.0508 (t−3.67)。**我據此向使用者宣告「這個結果是假的」。**

**那個宣告是錯的。** 正確的跳空判定必須是「**前一根整根在價位另一側** 且
這根開盤越過」——因為「掃單棒自己反轉、收在價位下方，所以下一根自然開在
下面」是完全合法且可成交的情境，我的 v1 測試把它全部誤標。修正後：真實
跳空比例 crypto 0.5-1.3%、期貨 0.9-8.0%，**去掉它們兩邊結果都幾乎不變**
（crypto t 7.67→7.80、跨資產 5.10→4.92）。

**最刺眼的是我當時就看到了反證卻沒想通**：crypto 是 24/7 連續交易、
`gap_sweep` = 0%，卻報出 60% 的 `gap_fill`。**一個在連續市場和分時段市場
給出相同答案的診斷，量的是它自己，不是資料。**

**Root cause:**
同一天我才在兩處抓到「先審儀器再信輸出」（shadow harness 的時間錨、
sweep_core 的成本符號），然後**自己寫了一個沒審過的儀器，並用它下了殺死
策略的判決**。差別在於：審別人的程式碼時我會找反證，寫自己的程式碼時我
直接相信輸出。而且這個錯誤方向特別危險——它「證實」了使用者的懷疑，
**符合預期的答案讓我停止檢查**（與 2026-07-28 那條同源，第三次）。

**Correct approach（已修）:**
1. 跳空判定改 v2（prev bar 整根在另一側 + 本根開盤越過），註解裡寫死
   「連續市場與分時段市場給出同樣答案 = 診斷壞了」這個 sanity。
2. 診斷腳本本身必須有**已知答案的對照組**：crypto 24/7 的真實跳空率必須
   趨近 0，任何診斷跑出高值就是它自己壞了。這比事後 debug 便宜太多。

**Rule:** **新寫的診斷在用來推翻任何結論之前，先在一個「答案已知」的資料
上跑一次**——這裡就是 24/7 的 crypto（跳空率必為 ~0）。診斷工具跟被診斷
的對象一樣需要驗證，而且**自己剛寫的工具比別人的舊工具更危險**，因為
沒有「這是別人寫的、可能有坑」的警覺。特別是當診斷結果**恰好證實了某人
（包括使用者）的懷疑**時——那正是最該多查一輪的時刻，不是最該宣布的時刻。

---

## 2026-07-29: 未釘版的 mcp 在上游發 2.0 當天炸掉 agent-mcp——跟 commit 內容無關的部署崩潰

**What happened:**
推了三個純研究 commit（沒碰 indicator/agent/ 一行），agent-mcp 服務卻
「Deploy Crashed」。root cause：`requirements.indicator.txt` 的 `mcp`
**沒釘版本**，而 `http_server.py` import 了 `mcp.server.transport_security`
這種深路徑模組。PyPI 前一晚剛發 **mcp 2.0.0 大版本**——每次 push 都是
全新 build、`pip install mcp` 抓當下最新 → 2.0.0 的破壞性改動讓容器
啟動即死。昨天以前的 build 抓到 1.x 所以一直沒事。**同一份 requirements
21 個套件全部未釘版**，mcp 只是第一顆爆的（xgboost/numpy 若哪天跳大版，
炸的就是模型服務）。

**Fix:** `mcp==1.28.1`（本機 l30d 驗證過的版本）。全面釘版另列 hygiene
任務——不能盲釘：production 各服務現跑的版本未知，拿本機版本亂釘反而
可能改變 production 行為（xgboost 版本與模型 artifacts 的相容性尤其）。

**Rule:** 服務在「沒動到它的 push」之後崩潰，第一嫌疑是**全新 build 拉到
上游新版**——先查 requirements 有沒有未釘版的套件 + PyPI 最近有沒有
大版本發布（`pip index versions X` 十秒）。凡是 import **深路徑**
（`pkg.sub.module`）的相依套件必須釘版，深路徑在 minor 版就常被移動。
Railway 信裡的「Restart Deployment」按鈕只會重啟**同一個壞掉的 image**，
對 build 期錯誤無效——修法永遠是改 requirements 後重新 push。

---

## 2026-07-28: sweep-failure 回測的滑價符號寫反——「含成本 t=8.27」實質是零成本；修正後 t=3.35

**What happened:**
啟動策略 #3（sweep-failure）forward 驗證前審計引擎，發現 `sweep_core.py`
進場滑價 `entry = lvl - d*SLIP*A` 是**對我們有利**的方向（註解卻寫著
"slippage against us"），與出場的不利滑價在時間出場路徑**精確抵銷**。
證明方法：SLIP=0.05 跑分（pool +0.0616R）**優於** SLIP=0（+0.0603R）——
「含成本」比零成本還賺 = 成本模型必壞。README 頭條（PF 1.29 / t=+8.27 /
9/9 幣正、宣稱含 0.05 ATR/邊）實質是零成本數字。修正符號 + 改用逐幣
真實 bps 費用重算：目標執行情境 pool +0.0255R / t=+3.35 / PF 1.11，
全 taker 情境 t=+2.29；逐幣 t 多在 0.2-1.2，前半段 5/9 幣為負。edge
從「肥」變「薄而依賴執行」——仍是最佳候選（n 大），但天花板完全不同。

**Root cause:**
一個符號 typo 讓兩條成本腿互相抵銷，而「跑出來有扣成本的樣子」（SLIP
參數在、README 寫了成本）讓所有人（含遷入時的審查）以為成本已計。
統一 ATR 單位的滑價還會**奉承低波動幣**（BTC taker 5bps = 0.079 ATR，
是 DOGE 的近 2 倍），進一步高估籃子可行性。

**Rule:** 任何**宣稱含成本**的回測，收工前必跑一次 **cost=0 對照**：
含成本結果 ≥ 零成本結果 = 成本模型壞了（符號/抵銷/漏腿），這個檢查
5 秒鐘，能擋掉整類錯誤。費用一律用**逐標的真實 bps** 換算，不用統一
ATR 單位（低波動標的的相對成本會被統一單位低估一半以上）。同日第二次
應驗「分析輸出之前先審儀器」——一天內 shadow harness（時間錨）與
sweep-failure（成本符號）兩個獨立 harness 都在輸出層看起來完全正常。

---

## 2026-07-28: shadow 執行研究把窗口錨在 bar 標籤——重播的是訊號誕生前的一小時，兩個 cohort（n=52+14）全部作廢

**What happened:**
`research/shadow_exec_window.py`（07-10 預註冊的執行層 A/B：R1 流擇時 vs
R2 maker 掛單）用 `entry_time`/`signal_time` 當重播窗口起點。但這兩個欄位
是 **bar 開盤標籤**：訊號實際在 label+1h 的 bar 收盤後才誕生、executor 在
label+1h+2.5min 成交（5 筆 live 全部 `created_at` = entry_time+1h02m28s，
成交價與 created_at 時刻的 1m 價格吻合、與標籤時刻差 60-140 bps）。於是
harness 重播的「60 分鐘執行窗口」= **訊號自己的形成 bar，訊號還不存在的
那一小時** = 純 look-ahead。兩個 cohort 的所有數字都是假象：live-R1 的
+16.7 bps =「動能 bar 裡任何早於收盤的分鐘都贏收盤價」；signals-R2 的
3 筆 −76 bps「未成交」= bar 收在極值上的形狀，不是逆選擇。我在發現前
已經拿這些數字對使用者下過兩輪結論（先 R2 FAIL、再據此修正 maker 評估）
——**兩輪都要撤回**。

連帶的第二個坑：我先用 parquet 裡的 s3912 價格（60105.6，開盤標籤價）
對比 live 成交（60799.9），得出「live 進場平均劣化 +25~40 bps」的重大
發現——**也是假的**，那是訊號 bar 自己的漲跌，不是執行成本。查 DB 才
發現 tracked_signals 的 entry_price 本來就記的是開火時刻價格，跟 live
成交價一致到 0.1；是 harness 自己丟掉它、用窗口第一筆快照重算了 baseline。

**Root cause:**
專案裡兩種時間慣例並存：研究層 bar 以**開盤時刻為標籤**（label T 的 bar
在 T+1h 才完整，intrabar_volume_ic.py 有做 400/400 對齊驗證），DB 事件列
（tracked_signals/v7_okx_positions 的 created_at）是**牆鐘時間**。任何把
label 索引的列 join 到牆鐘資料（1m 快照、成交）的程式，必須先把 label
轉成開火時刻——harness 沒轉，而且它的 docstring 寫著「window from actual
entry」，作者（先前 session）以為 entry_time 就是成交時刻。放大因素：我
**兩次分析它的輸出之後才審它的時間對齊**——「先看結果再查測試設計」
（2026-04-13 calibration 教訓）的重播版。

**Correct approach（已修，同日）:**
1. 窗口錨點改 `created_at`（帶守門：不在 label+[55,75]min 內就 fallback
   到 label+65min，防 backfill 列的 created_at 失真）。
2. signals cohort 的 baseline 改用 tracked_signals.entry_price（開火價，
   已對 live 成交驗證）。
3. 每列輸出 `anchor_gap_bps`（baseline vs 錨點後第一筆 mid）——spread 量級
   = 錨對了，bar 量級 = 錨錯了。這個 sanity 讓同類錯誤下次自己現形。
4. 預註冊的 R1/R2 規則與切換門檻（n≥30 + CI 低緣>0）不動；證據從 0 重新累積。

**Rule:** 任何把 bar 標籤列 join 到牆鐘資料的研究，動手前先做**價格對齊
證明**：拿列上記錄的事件價 vs 宣稱錨點時刻的 1m 價，差超過 ~5 bps = 錨錯
了。`v7_okx_positions.entry_time` 和 `tracked_signals.signal_time` 是 bar
標籤不是事件時刻；事件牆鐘時間一律看 `created_at`。**分析任何 harness 的
輸出之前，先審它的時間對齊**——輸出層面的「發現」（不管多驚人）在對齊
未證明前一律當 artifact 處理。發現一個「重大成本/重大 edge」時，第一步
是找一筆能獨立交叉驗證的紀錄（這次是 DB 的 entry_price），不是往下推論。

---

## 2026-07-28: meta-labeling 出場模型 NO-GO——出場決策在這份資料上「學不到」，不是「學得到但不賺」

**What happened:**
使用者提出一個概念上正確的批評：「用做空訊號平多單很奇怪，為什麼不在
持有多單時訓練一個出場模型、持有空單時訓練另一個」。進場模型回答的是
「**從空手開始**未來會漲跌」，持倉時要回答的是「**已經在倉裡**，現在平
比續抱好嗎」——後者有成本基礎、浮盈浮虧、MFE/MAE、持有時間，前者全部
看不到。這正是 2026-07-24 我列為 (a)「風險最低、推薦」但沒被選的選項
（當時選了 (c) RL，四折全輸）。

建了 `research/meta_exit_model.py`：重用 RL 那次的反事實列舉，標籤定義為
「此刻平倉的淨報酬 > baseline（3xATR trail / opp_signal）最終交付的淨
報酬」，所以模型**只能新增提早出場、不能取消 baseline 出場**（沒學到
東西就退化成 baseline，不會退化成某個沒驗證過的新東西）。`side` 當特徵
不切分（mistake.md 2026-04-13）。120,396 個持倉狀態、3,689 個進場 bar。

**PnL 四關全 FAIL**：per-fold mean −6.3 bps、中位 −2.2、正折 3/6、
bootstrap CI [−26.3, +12.5]。但每折只有 2–6 筆交易，這個測試本身低統計力。

**真正的答案來自 AUC 診斷**（用 5,600–11,600 筆狀態，樣本紮實）：
**OOS AUC 平均 0.458 — 低於 0.5**，而且 0.161 到 0.616 劇烈擺盪。前幾折
（訓練少）是 0.16–0.37，模型學到的是**系統性錯誤的規則**。

**Root cause:**
不是「有預測力但換不成 PnL」（那是 2026-07-28 棒內量分布那次的形狀），
是**訊號根本不存在**。AUC 的不穩定（0.16→0.62）像非平穩而非樣本不足：
「現在平倉比較好」這件事本身隨 regime 翻轉，沒有跨期穩定的結構可學。
這也回頭解釋 RL 為何失敗——不只是 Q function 在 9 維上太複雜，是底層
就沒東西。同形狀見 mistake.md 2026-04-13（BEAR 子模型 AUC 0.378）。

**諷刺但重要的觀察：** 使用者的批評成立（用反向訊號平倉概念上是錯的），
但那個「錯」的機制實證上是現有出場裡最好的——`opp_signal` 回測 85.7% WR
（live n=4，+2.35%/75%，樣本太小僅供參考），`trail_stop` 回測 37.0%
（live n=10，−0.68%/40%）。可能的解釋：`opp_signal` 有效不是因為它答對了
出場的問題，而是**進場模型的方向翻轉本身就是強訊號**，剛好也適合當出場
條件；用更複雜的模型取代它反而丟掉這個乾淨訊號。

**Rule:** 「概念上正確的批評」不等於「那個修法會work」——兩者要分開驗，
而且要**先跑 AUC/IC 診斷再跑 PnL**。PnL 測試在小樣本下（這裡每折 2-6 筆）
分不出「沒訊號」和「有訊號但沒經濟價值」，而這兩者的後續行動完全不同：
前者該收手，後者該找更好的執行方式或特徵。學習型出場在這份資料上已經
連續失敗兩次（RL 值函數、監督式 meta-labeling），第三次要再提之前，先
問「新增了什麼這兩次沒有的資訊源」——不是換演算法，換演算法已經證明
無效。

---

## 2026-07-28: 收緊 config guard 卻沒查各服務現有的 env 值——把 executor 弄停 50 分鐘

**What happened:**
把 Stage 3 資金基準從 $1218.44 改成 $274 時，順手把 `config.py` 的 live
capital 上限從 1500 收回 500（理由正當：上限該待在現行基準之上一級，不該
停在一筆已不存在的存款的高水位）。改完、408 個測試綠、push。

**executor 從此停了 50 分鐘。** `validate_okx_config` 對 1218.44 拋
RuntimeError → `runner.get_executor()` 的 except 設 `_INIT_FAILED=True`
→ 永久回傳 None → WS 不連 → 權益快照從 00:25:30 全斷、01:02 的
update_cycle 沒有任何 OKX 動作。

而且我一開始找錯地方：這個專案有 5 個 Railway 服務，我假設 OKX executor
在 `rfobot`（名字最像主服務），在那裡查到 `OKX_INITIAL_CAPITAL_USD=89`
就當成「找到根源」。實際上 `Dockerfile.indicator` 跑
`gunicorn indicator.wsgi:app`，交易 cycle 在 `indicator/app.py:694`
呼叫 `get_executor()`——**executor 在「輸出圖表」服務**。rfobot 的 89 只是
Telegram admin 指令用的陳舊副本，對交易路徑毫無影響。「輸出圖表」的值是
1218.44，正是被我新 guard 擋下的那個。

**Root cause:**
收緊一個 guard = 讓一組先前合法的值變成非法。**改 guard 之前沒有列舉「所有
會載入這份 config 的部署環境現在實際帶什麼值」**，等於在製造啟動失敗。本機
測試完全測不到這件事：測試餵的是測試自己構造的 cfg，不是 Railway 的 env。

放大因素跟同日上一條同源：**一個符合預期的答案讓我停止搜尋**。看到 rfobot
的 89（「果然是陳舊值！」）就沒有再問「這個服務真的是跑 executor 的那個
嗎」。同一天寫下這條教訓，同一天又犯。

**Correct approach:**
1. 任何**收緊**型的 config 變更（上限調低、增加必填、縮小允許集合），push
   前先把**每個**會載入它的服務的相關 env 值列出來對照。Railway 的
   Raw Editor 一次看得到整組，比逐列點快也不易看漏。
2. 不確定哪個服務跑哪支程式時，**看 Dockerfile 的 CMD**，不要用服務名字猜。
   `grep -rn "get_executor()" --include=*.py .` 三秒就能定位真正的呼叫端。
3. fail-closed 的 init（`_INIT_FAILED=True` 後永不重試）配上「Railway 顯示
   Online」= 典型 silent failure。**判斷 executor 活著的訊號是
   `v7_okx_balance_snapshots` 的新鮮度**（WS 活著每 ~5 秒一筆），不是服務
   的綠燈，也不是 `v7_okx_executor_status`（那張表在行程內 DEMOTE 時不會
   更新）。

**Rule:** 收緊任何 guard 前，先枚舉所有部署環境的現值——本機測試對
production env 是瞎的。服務職責一律以 Dockerfile CMD 與 callgraph 為準，
不以服務名稱推測；這個 repo 的 5 個服務共用同一份程式碼，同名 env 在不同
服務可以完全不同值（89 vs 1218.44 就這樣共存了兩週）。判斷交易系統死活，
永遠看**資料的新鮮度**，不看平台的健康燈。

---

## 2026-07-28: 用「有沒有對應的 trade」判斷資金階躍成因——這個檢查對孤兒倉是瞎的

**What happened:**
使用者說「資金確定了就用現在的 274 開始跑」。基準是 $1218.44，掉到 $274
是 −77%，遠超 total cap −30%，所以我必須先判斷這是 operator 提領（→ 基準
重置，標準程序）還是策略虧損（→ hit kill trigger，必須降階重驗）。

我照 [[2026-07-13 資金調度]] 那條記下來的判別法做：查 `v7_okx_positions`
找有沒有對應的已平倉交易。結果是「最後一筆是 07-16 的 id=20，之後零筆」，
kill log 也只有 CAP-2 over-funding、沒有任何虧損型 trigger。我據此向使用者
斷言「**這是提領，不是虧損**，走基準重置」。

**這個結論是錯的。** 繼續查 `v7_okx_reconciliation_log` 才看到 07-27 13:02
有一筆 `orphan_exchange`：**37.11 張 LONG @ 65050**（≈$24,140 名目，對當時
$1218 權益約 20x；executor 開的倉一向 0.31-0.61 張）。再拉權益的逐小時軌跡
更確定：07-27 12:00 起開始劇烈擺盪，22:00 從 1124 崩到 379，23:00 觸底
**$16.62**，之後入金回到 $274。是**手動爆倉**，不是提領。

**Root cause:**
`orphan_exchange` 的定義就是「**OKX 有、DB 沒有**」。用「DB 裡有沒有對應
trade」去判斷成因，對這類部位**在定義上必然看不見**——我用了一個對目標
現象結構性失明的檢查，還把「查無結果」當成陽性證據。

更糟的是 [[2026-07-13]] 那條把這個檢查寫成了未來的修法建議（「CAP-4 觸發
前檢查 equity 階躍是否有對應的已平倉 trade」）。**那條建議如果照做會實作出
一個同樣瞎的 guard**：手動爆倉（最需要擋的情境）永遠不會有對應 trade，
guard 會一律判成「operator 資金調度」然後放行。

放大因素：查到「零筆交易」時我停止了搜尋。零筆本身就該是紅旗——12 天沒
交易的帳戶憑什麼少掉 77%？我讓一個符合預期的答案終止了調查。

**Correct approach（這次實際用的）:**
判斷資金階躍成因，**權益的時間軌跡才是主證據**，不是 trade 紀錄：
- **提領/入金** = 單一時點的階躍，前後平坦
- **持倉虧損** = 連續變動，有中間值、有來回擺盪

`v7_okx_balance_snapshots` 每 5 秒一筆，resample 成逐小時的
first/min/max/last 一眼就分得出來。這次的軌跡（1218 平坦 → 整天 908~1435
擺盪 → 崩到 16.6 → 跳回 274）不需要任何其他證據就能定性。

配套要查的表：`v7_okx_reconciliation_log`（孤兒倉）+ `v7_okx_kill_log`。
只查 `v7_okx_positions` 等於只看 executor 自己知道的世界。

**Rule:** 判斷「錢為什麼變少」，**先看 `v7_okx_balance_snapshots` 的時間
軌跡**（階躍 vs 連續），再用 `v7_okx_reconciliation_log` 找 executor 不知道
的部位，最後才是 `v7_okx_positions`。永遠不要用「DB 裡沒有對應紀錄」去證明
「沒有發生過交易」——DB 只記錄 executor 自己下的單，手動單對它是不存在的。
任何「查無 X 所以沒發生 X」的推論，先問「這個查詢在 X 真的發生時看得到嗎」。
**符合預期的答案最危險，因為它會讓你停止搜尋**；金額對不上時，在找到能
定量解釋整個缺口的機制之前，調查都還沒結束。

---

## 2026-07-24: 聯合進出場 offline RL——資料量級撐不起模型複雜度，4 折全輸 baseline

**What happened:**
使用者問「當初為什麼沒有把進場出場一起訓練」，我解釋這是刻意簡化（避免
在小樣本上疊加自由度），並提出三個可能路線給使用者選：(a) 獨立訓練一個
meta-labeling 風格的出場模型（風險最低，推薦）、(b) 進出場參數聯合網格
搜索、(c) RL policy 把進場出場當一個完整決策序列一起學（技術上最貼近
「聯合訓練」但這個專案的資料量級遠低於 RL 通常需要的量，overfit 風險
最高）。使用者明確選了 (c)。

依照使用者選擇建置：offline FQI（Fitted Q-Iteration，Q function 用
XGBoost 近似，跟專案既有建模方式一致）、逐 bar MDP（不是逐筆交易，
樣本數才夠）、state 用現有模型輸出（pred_ret/vol_regime/atr_pct）+
持倉狀態（bars_held/unrealized/MFE/MAE/decay_streak）、exact
counterfactual transition enumeration（利用「自己的交易不影響
市場」這個性質，對每個 bar 精確算出所有動作的真實後續，不需要
importance sampling）。套用專案既有的 walk_forward_splits
（purge+embargo）+ 4 條驗證關卡（aggregate lift / per-fold mean /
frac positive folds / bootstrap CI）。

4 折結果：**per-fold mean lift −112.17 bps、0/4 折為正、bootstrap
95% CI [−156.95, −70.42]（整段在零以下）**。RL policy 在每一折都
輸給現有 baseline（trail_stop/opp_signal），而且輸得不小——不是
邊緣性的「差一點沒過」，是方向一致的全面落敗。

**Root cause:**
每折訓練資料只有 1200-1900 根左右的 in_oos bar（3-4 個月），對一個
9 維 state 空間的 Q function 來說太稀疏。更根本的技術原因：這版 FQI
沒有加 offline RL 常見的保守正則化（例如 CQL 那類，用來壓制 Q
function 對訓練資料沒怎麼見過的 state-action pair 產生過度樂觀的
外推值）——這是 offline RL 的已知通病，樣本一少就特別明顯，會讓
greedy policy 被 Q function 的雜訊帶偏。這不是實作 bug（4 折都乾淨
跑完、沒有 crash、逐項核對過 state/action/reward 定義都正確），是
方法本身的複雜度跟資料量級不匹配。

**Correct approach:**
選路線前已經用 AskUserQuestion 明確標註「RL 的資料量需求遠高於這個
專案能提供的量級」這個風險，使用者知情選擇。結果驗證了這個顧慮不是
保守，是準的。要繼續這條路線，需要先解決資料量問題（例如擴大到
bar-level 的跨資產 pooling、或加 CQL 級別的正則化），但這些改動的
預期報酬相對工程成本，現階段不划算——這個結論本身已經是研究產出。

**Rule:** 資料量級和模型複雜度要先粗估匹配度再動工，不要因為「這是
教科書上更完整/更先進的方法」就假設它一定更好——RL 需要的有效樣本數
通常是幾萬到幾百萬個 transition，這個專案能提供的（幾千根 bar）差了
1-2 個數量級，這個差距不會因為换更好的演算法或調參數而消失。使用者
明確選擇某條高風險路線時，照做但要誠實跑完整套既有驗證紀律（per-fold
+ bootstrap CI，不只看 aggregate），讓資料自己說話而不是提前放棄或
硬撐；4 折一致的方向性結果（不是邊緣性差異）代表這不是運氣，可以放心
記為 NO-GO 收尾，不需要再多折驗證。

---

## 2026-07-13: 資金調度觸發 CAP-4 DEMOTE——kill switch 分不出 operator transfer 和 strategy loss

**What happened:**
使用者臨時需要資金，把 OKX 交易帳戶的錢轉出。CAP-4（total loss cap −30%）看到
equity 對 initial capital 掉了超過 30%，判定「策略累積虧損超限」→ DEMOTE（終態，
需人工介入）。資金轉回（$197.55）後又因為超過舊基準（$89）的 1.5x 觸發 CAP-2
over-funding HALT。整個過程**沒有任何一筆策略虧損**，純粹是 operator 資金調度，
但系統經歷了 DEMOTE + HALT 兩次停機，恢復耗掉一整個 session。

**恢復路徑（記下來，下次照做）：**
1. DEMOTED 只活在 process 記憶體（`executor.py` cycle guard 不回讀 DB）→
   **重啟 service 就會重新 init**（空 commit push main 觸發 Railway redeploy 即可）
2. 重啟後 `start()` 重跑全部檢查，kill check 用當前 equity 重算——資金回來了
   CAP-4 就不再觸發
3. CAP-2/CAP-3 的 HALT 是可自動恢復的：trigger 條件消失後下一個 cycle 自動回 ACTIVE
4. 若 equity 和 `OKX_INITIAL_CAPITAL_USD` 基準對不上，改 env（Railway 會自動
   redeploy）；報表基準另在 `report.py` EXECUTOR_RESTART_CAPITAL_USD

**Root cause:**
Kill switch 的輸入只有 equity 數字，沒有「錢為什麼變少」的資訊。策略虧損是
一筆一筆漸變（每筆 trade 有 v7_okx_positions 紀錄對得上），operator transfer
是無對應 trade 的瞬間階躍——這個特徵完全可以機器判別，但目前沒有做。

**Correct approach（未來修法，擇一）：**
1. 加 `/okx-admin/pause` endpoint（POST + confirm）：operator 資金調度前先合法
   暫停 executor，調完 resume——kill switch 不會看到「假虧損」
2. CAP-4 觸發前檢查「equity 階躍是否有對應的已平倉 trade」：無 trade 對應的
   大階躍 → 改推「偵測到資金轉出，請確認」告警而不是直接 DEMOTE

**Rule:** 動帳戶資金（轉入/轉出）之前，先想 kill switch 會看到什麼。目前系統下，
轉出 >30% 必觸發 CAP-4 DEMOTE、轉入超過基準 1.5x 必觸發 CAP-2 HALT——這不是 bug，
是 cap 的設計本意（防 ruin / 防意外注資），但 operator 要把「資金調度 → 先 pause
或事後照上面恢復路徑走」當成標準流程。kill trigger 的告警文字如果跟實際操作
（自己轉錢）對得上，不要當成策略故障去 debug。

---

## 2026-07-05: 月度復驗三重靜默失敗——舊資料上的 PASS、沒人收到的推送、假裝在跑的排程

**What happened:**
每月 5 號 09:00 的月度復驗排程準時執行、verdict PASS、報告落地——看起來一切正常。全面體檢才發現三層疊加的靜默失敗：
1. **驗證跑在 16 天舊的資料上**：執行當下本機 DNS 剛好斷線，auto-backfill 失敗被 log 成 "non-critical" 後 fallback 到 06-18 的快取特徵。報告裡「6 月 IC +0.016 貼零（n=416）」被解讀成「概念漂移的第一聲」，實際上是**資料截尾 artifact**——網路恢復後補滿資料重跑，6 月 IC = **+0.178（n=720）**、7 月頭 100 根 +0.204，完全正常。差點基於斷檔資料做出「edge 開始漂移」的判斷（第三篇 LinkedIn 貼文就要拿這根柱子當主視覺發出去了）。
2. **PASS 推送沒人收到**：同一波 DNS 斷線讓 Telegram 推送也失敗（`telegram_critical_exception`），且**無重試**——排程「向人回報」這一環死了，operator 根本不知道這期跑過。復驗儀式的存在意義是「偵測快」，但它自己的失敗沒人偵測。
3. **DailyCollect 排程指向已刪除的舊路徑約 96 天**：repo 資料夾更名後，Windows 排程的 action 還指向舊 CJK 路徑 → 每天 04:00 exit code 1，Coinglass parquet 備援線 3 月底起停更。排程面板顯示「就緒、有在跑」= 看起來活著。

**Root cause:**
與 [[mistake 2026-04-22 / silent failure]] 同族但發生在**自動化排程層**：(a) 資料新鮮度失敗被降級為 non-critical 後，verdict 沒有攜帶「本次基於舊資料」的標記——fail-open 的結果看起來跟 fail-safe 一樣漂亮；(b) 告警送出層單次失敗即放棄（跟 2026-06-19 出場告警同款，只是這次死因是網路不是格式）；(c) 排程的 lastResult=1 沒有任何監控。三個都是「執行成功的外觀」掩蓋「核心功能已死」。更深一層：**單月 n 不足的統計量（IC 貼零）在下結論前沒先問「資料完整嗎」**——又一次「先看結果再查測試設計」（2026-04-13 calibration 教訓的排程版）。

**Correct approach（已修，2026-07-06）:**
1. `quarterly_revalidation.py` 加 **STALE-DATA guard**：特徵尾端 > 48h 舊 → verdict 強制標 `STALE-DATA — RE-RUN REQUIRED`（不給 PASS/DRIFT），Telegram 訊息帶資料截止日。
2. Telegram 推送加 **6 次 × 60s 重試**；最終失敗把 `TELEGRAM PUSH FAILED` 戳進報告檔本身。
3. DailyCollect 排程 action 修正指向 `flow_system\market_data\backfill\daily_collect.bat`，手動觸發驗證 lastResult=0。
4. 網路恢復後重跑復驗：資料補滿到 07-05、PASS（AUC 0.5988 / IC +0.177）、推送成功收到。

**Rule:** 任何**基於資料的自動化 verdict**，必須把「資料新鮮度」當成 verdict 的一部分——資料過期時寧可輸出「無法判定」也不能輸出一個漂亮的 PASS/FAIL。任何「向人回報」的排程（告警、報告推送），送出層必須有重試 + 最終失敗要在某個人會看到的地方留痕。看到單月統計量異常（IC 驟降、WR 崩），**第一個檢查是該月的 n 和資料截止日**，不是開始解讀市場含義——n=416 vs n=720 就是「漂移的第一聲」和「什麼事都沒有」的差別。排程改路徑/搬 repo 後，`schtasks` 的 action 是不會自己跟著搬的。

---

## 2026-06-19: 出場 Telegram 告警「整個 live 史上」靜默失敗——exit reason 的 '_' 破壞 Markdown → Telegram 400 → 被吞

**What happened:**
一筆 live SHORT（id=8）由 opp_signal 正常獲利平倉（+2.15% net，DB 正確 CLOSED），但**沒有任何 Telegram 出場通知**。查 DB 確認 `_close_position` 跑完了（net_pct/equity_after 都算了），告警卻沒出去。root cause：`send_critical` 用 `parse_mode="Markdown"`，而 `format_exit_alert` 把 exit reason 直接塞進訊息 `*OKX LIVE EXIT* (opp_signal)`——**`opp_signal` 的單一 `_` 在 legacy Markdown 是未閉合的斜體標記 → Telegram 回 400「can't parse entities」→ send_critical 只 log（`telegram_critical_failed`）不重試 → 告警靜默消失**。所有 exit reason 都帶 `_`（`opp_signal`/`trail_stop`/`time_cap`/`manual_close_trail_bug`），所以**每一筆出場告警從上線以來從沒成功過**。entry 告警沒事（無 `_`）；OPEN ABORTED / kill 告警沒事（訊息剛好無 `_`），所以一直沒人發現出場那條壞了。

**Root cause:**
把**動態字串塞進 Markdown 訊息卻沒跳脫/包 code span**，加上**送失敗只 log 不 fallback**——兩個疊起來＝典型 silent failure。最隱蔽的是：同一個 `send_critical` 對「無特殊字元」的訊息（OPEN ABORTED）正常，對「帶 `_`」的訊息（出場）必失敗，所以「告警系統看起來能用」掩蓋了「某一類告警全死」。跟 [[mistake 2026-04-22 / silent failure]] 同調：Railway 綠、進程活、其他告警會動 = 看起來健康，但某條路徑靜默死亡。

**Correct approach（已修，commit 待 push）:**
1. `format_exit_alert`：reason 改用反引號包（`` (`{reason}`) ``）——code span 內 `_` 是字面量，Markdown 不再被破壞。
2. `send_critical`：**400 時去掉 parse_mode 用 plain text 重發一次**——critical 告警絕不可因格式 bug 而丟失（429/5xx 不重發，去 parse_mode 也救不了）。
3. 回歸測試 `tests/test_okx_alerter.py`：(a) 400 → plain-text fallback 且第二次不帶 parse_mode；(b) 429 不雙發；(c) 每個 exit reason 在訊息裡都被反引號包。324 okx 測試綠。

**Rule:** 任何要送進 Telegram（Markdown/HTML parse_mode）的訊息，**動態插值的欄位（reason / id / 任何含 `_ * [ ] ( ) ` 的字串）必須跳脫或包 code span**，否則一個特殊字元就讓整條訊息被 400 拒收。更重要：**critical 告警的送出層必須有「parse 失敗 → 降級 plain text 重送」的 fallback**——告警是用來在出事時通知人的，絕不能因為一個格式字元而靜默消失。看到「DB/狀態正確但通知沒來」第一個查**送出層的 parse_mode + 該訊息有沒有未跳脫的特殊字元**。

---

## 2026-06-17: facade-skip bug 第三次——isolated 切換漏了 OkxClient.set_leverage，live 永遠開不了倉

**What happened:**
啟用 `OKX_TD_MODE=isolated` 後第一個 Strong 信號，live 推 `🔴 OKX OPEN ABORTED: set-leverage(isolated 10x posSide=long) failed — no order sent`。executor 在 isolated 開倉路徑呼叫 `self._client.set_leverage(...)`（executor.py:1010），但 live 下 `self._client` 是 **OkxClient facade（client.py）**，而 facade **沒有 set_leverage passthrough**（只存在於 rest.py:240 與 mock_client.py:198）→ 拋 `AttributeError` → 被 executor 的 bare `except Exception`（executor.py:1014）吞掉 → `lev_ok=False` → 中止開倉。net effect：isolated 模式下 executor **永遠開不了任何倉**。告警裡的 `posSide=long` 是 config 拼進字串的值，**不是 OKX 真的拒了**——AttributeError 在到達 OKX 之前就拋了。

**Root cause:**
**這是 facade-skip bug 第三次復發**（第一次 amend_algo_stop 漏 inst_id 2026-06-10，第二次同 bug 修不完全 2026-06-14，見 [[project_trailing_stop_amend_bug]]）。完全相同的盲點：一個跨層 feature（這次是 isolated margin 的 set_leverage）**只加了 executor 呼叫端 + rest 底層 + mock 測試替身，唯獨漏了中間的 OkxClient facade**。測試全綠是因為 trading 測試注入 `MagicMock` 當 client，MagicMock **auto-vivify 任何屬性**（`client.set_leverage(...)` 自動回傳 truthy mock），所以測試以為方法存在；唯一用 faithful MockOkxClient 的測試又只跑 `td_mode="cross"`，根本不進 isolated 分支。**facade↔executor 這條縫零覆蓋——client.py 本來連一個測試檔都沒有**。2026-06-14 的 LESSON 明明已寫「修 call site + 底層卻跳過 facade = 沒修完，要驗整條 call chain」，但下一個跨層 feature 還是踩同一個坑——因為當時只手動修了那一個方法，**沒有建立結構性防護**。

**Correct approach（已修，commit 5a41ad7 pushed）:**
1. client.py 補 set_leverage passthrough，簽名與 rest.set_leverage 鎖死（keyword-only inst_id/lever/mgn_mode/pos_side）。
2. **新增 tests/test_okx_client.py（facade 本來零覆蓋）= 結構性防護**：用 AST 自省抓出 executor 在 `self._client.<name>` 上呼叫的**每一個**方法，斷言 facade 都有定義（`test_facade_exposes_every_method_executor_calls`）+ 簽名 superset 檢查（facade 不可 drop rest 接受的參數，專抓 amend_algo_stop 那種「方法在但簽名漂移」`test_facade_signatures_match_rest_for_shared_methods`）。**反向證明過**：刪掉 set_leverage → 測試立刻變紅、精確指出缺失方法。這對測試能擋整類 facade-skip（amend + set_leverage 都會被抓），不用手維護方法清單。
3. pos_mode 查證：帳戶 long_short_mode（commit 4c982c4 live smoke + 先前 SHORT 0.29 帶 posSide=short 對帳 CONSISTENT 實證），config 與帳戶一致 → 補完 facade 後真實 OKX 呼叫成功、無 51000 風險。318 okx 測試綠。

**Rule:** 任何**跨層**的新方法/簽名變更（call site → facade → rest/ws → mock）**必須同時改 facade，而且要有結構性測試保證 facade 暴露 executor 呼叫的每個方法**——不能靠「記得改 facade」（已證明記不住，三次了）。test double（MockOkxClient）的簽名要**嚴格**，optional 參數正好遮這類 bug；MagicMock 當 client 永遠測不出 facade 缺方法。新增任何 facade↔executor 之間的方法，跑 `tests/test_okx_client.py`。看到「測試全綠但 live 報 AttributeError / TypeError / OPEN ABORTED」第一個查的就是 **facade 是不是漏了 / 簽名漂移了**。

---

## 2026-06-20: 一個 case 的 FOMO 差點變成 threshold-sweep overfit（避免成功、紀律守住）

**What happened:**
2026-06-19 01:02 (TPE) 一根 **Moderate** BULLISH 訊號（BTC $62,664、Confidence 85、Mag p98、Driver = cg_bfx_margin_delta +90,359、Regime TRENDING_BEAR）我按紀律沒進場（current rule: Strong-only），但 46h 後 BTC 回到 $63,930（+2.02%）—— 訊號方向 + 時間都對。

我自然想問：「**Strong threshold 2.5% 是不是太緊？**」「**改 5% 是不是更好？**」

差點就跑 threshold sweep（top 2.5% / 3% / 4% / 5% / 6% / 7% 各算 WR、avg bps、cum），找「歷史最 CP 值的 threshold」並改 production。

幸好 user 自己 spot 到：「**可是這會 overfit 不是嗎**」——當場叫停。

**Root cause（為什麼這是經典陷阱）:**

1. **單一 case bias**: 6/19 那筆是 1 個 sample。整個歷史證據 stack（5.5mo backtest, 1980 tracked_signals, live cohort）都顯示 Strong > Moderate。改 rule 需要等比例的證據，不是 1 case + 直覺。

2. **Threshold sweep ≠ 中性研究**: 即使跑 walk-forward，掃連續 threshold = multiple comparisons + selection bias。1980 樣本切 5 個 threshold 桶 = 每桶 ~400 有效樣本，期望 25% false positive rate (5 tests × p=0.05)。「最好那個」80% 是運氣不是 edge。

3. **Crypto regime non-stationary**: 即使歷史 optimal 在 5%，未來 regime 變了不一定還在。Past optimal threshold 是 in-sample fit、forward 沒保證。

4. **Optimization 對象錯了**: 想擠 0.5% threshold-tuning alpha vs 加新的真實 alpha source（cross-asset / 異源資料 / compound trigger），時間 ROI 完全不成比例。Threshold tuning 是 low-yield high-overfit 的研究路線。

5. **FOMO 偽裝成 research**: 「我漏掉 6/19」的情緒 → 「該改 rule」的合理化 → threshold sweep 看似嚴謹但本質是 chase。

**Correct approach（守住的紀律）:**

- **Strong-only rule 不動**。已有 5.5mo backtest + 1980 tracked_signals + live cohort 三層證據撐住，改變需要相同等級的證據。
- **不 sweep threshold**——撤回原本要 commit 的 `scripts/threshold_sweep.py`。
- 若未來確實想驗證「premium Moderate 有沒有 edge」，正確路徑 = **categorical compound trigger watcher**（不 sweep、不改 entry）：
    ```
    TIER_B candidate = Moderate tier
                     + Mag p95+ (categorical flag, 不是 tuned threshold)
                     + Driver class in [whale_margin, short_squeeze_setup]
                     + Regime in [TRENDING_BEAR, CHOPPY]
    ```
    fire 時純 Telegram alert（不 auto take）→ 累積 6 個月 → 30+ case + WR 統計顯著才考慮 carve out 成新 tier。
- **不是 data-driven search**（容易 overfit）→ 是 **hypothesis-driven testing**（基於 domain knowledge）。
- 即使這個方案也只「先收集 evidence」，不直接改 entry rule。

**Rule（給未來的 self 跟 Claude）:**

1. **單一 case ≠ rule change**。改 entry rule 的舉證責任 = 對應原始 rule 的證據強度。Strong-only 用 5.5mo backtest + 1980 signal + live 驗過 → 推翻它需要等比例 evidence、不是 1 個亮眼 case。

2. **Threshold sweep 永遠是 last resort、不是 first response**。每次想 sweep 之前先問：(a) 是不是因為最近某個 case 觸發？(b) 連續 search space 是不是會放大 selection bias？(c) sample 夠不夠每個桶 > 200？任一答「是」就停。

3. **FOMO-driven research = bad research**。「我漏掉那筆」的情緒永遠不該是 research direction 的觸發點。情緒當下記筆記、24h 後冷靜再評估，多數時候會發現原本紀律是對的。

4. **改 rule = categorical > continuous**。Compound trigger（多個 categorical flag 共振）比 threshold tuning（連續參數 search）overfit 風險低 5x。前者基於 hypothesis，後者基於 data mining。

5. **「漏掉好訊號」是紀律的成本、不是 bug**。期望值 EV 高的 rule 必然會錯過部分好機會（type I error 換 type II error 的取捨）。機構紀律：寧願漏掉 10 個 6/19，不要為不漏改 rule 然後吃 30 個爛單。

**這是「avoided mistake」的紀錄，不是「committed mistake」**——這類紀錄比真實踩雷更 valuable：它證明紀律在情緒衝擊下守住了，下次同類情境（必然會再來）能更快識別。

---

## 2026-06-16: facade signature drift（第 2 次重演）—— OkxClient 缺 set_leverage proxy，每次 isolated 開倉 AttributeError 被吞成「set-leverage failed」

**What happened:**
2026-06-16 23:03 (TPE) 收到 `🔴 OKX OPEN ABORTED set-leverage(isolated 10x posSide=long) failed — no order sent`。我（跟 user）一開始全程往「OKX UI 設定不對」方向 debug：檢查持倉模式、margin mode、雙向 leverage、帳戶子模式。耗了多次來回 user 都回「OKX 都有設好」。

最後 grep `set_leverage` 才發現真相：`indicator/okx/executor.py:1010` 呼叫 `self._client.set_leverage(...)`，`self._client` 是 `OkxClient` facade，**而 `OkxClient` 根本沒有 `set_leverage` 這個 method**。OkxClient 只 proxy 了 `submit_market_order / submit_algo_stop / amend_algo_stop / cancel_algo_stop / get_positions / get_balance / get_account_config / get_server_time` 8 個 method，2f04e4d 加 isolated path 時忘了補 set_leverage。

執行流程實際上是：
```
1. cfg.td_mode == "isolated" → 進入 set_leverage 路徑
2. self._client.set_leverage(...)  → Python AttributeError（method 不存在）
3. executor.py:1014 的 except Exception 接住 → lev_ok = False
4. 觸發 abort + Telegram alert「set-leverage failed」
```

**完全沒打到 OKX 一次 request**。user 怎麼調 OKX UI 都沒救——這個 abort 是 Python 物件層級錯誤，不是 API 拒絕。但 Telegram alert 的文字寫「set-leverage failed」，誤導 user（跟我）以為是 OKX 那邊的問題。

時間軸：2f04e4d（2026-06-XX）加 isolated dormant capability，沒同步 OkxClient → 期間 cross 模式不會觸發、bug 潛伏。user 啟用 `OKX_TD_MODE=isolated` 那一刻起，每個 Strong signal 都會 abort。第一次 abort 才暴露。

**Root cause:**

**這是 [[mistake 2026-06-07 trail bug 三輪修]] 的 P0-2「facade 對齊」教訓的第 2 次重演**——只是換成「facade 整個缺 method」而不是「facade signature 缺參數」。同 root pattern：

> **新功能加在 REST adapter 層，忘了在 facade 層加 proxy。**

trail bug：`OkxRestClient.amend_algo_stop` 加了 `inst_id` 參數、`OkxClient.amend_algo_stop` 沒加 → TypeError 被吞。
這次：`OkxRestClient.set_leverage` 存在、`OkxClient.set_leverage` 整個沒有 → AttributeError 被吞。

兩個都被 executor 的 generic `except Exception` 吞掉，alert 文字只寫「failed」、不寫真實 exception type，**讓使用者（含未來的我）誤以為是 exchange 那邊的問題**——這是更深層的設計缺陷：fail-safe 設計把錯誤捕獲了但**沒把錯誤分類傳遞**給操作員。

放大因素：
1. **Telegram alert 文字過度泛化**（「set-leverage failed」對「AttributeError」跟「OKX 50014」一視同仁）→ 誤導診斷方向
2. **executor.py:1015 用 `logger.exception("set_leverage_exception")`** 確實會 log 完整 traceback，但 Railway logs 不在 alert 流程裡，user 手機看不到、跟我之前對話也沒查
3. **07eadff 修 trail 時加的 signature-parity 測試只覆蓋 amend_algo_stop**，沒擴展到「整個 OkxClient 對 executor 用的所有 method」

**Correct approach（已修，commit 914870c）:**

1. **OkxClient 補 set_leverage + set_leverage_detail proxy**（純 passthrough 給 `self._rest`）。
2. **rest.set_leverage_detail 新方法**回傳 `{ok, code, msg, raw}` 完整 OKX response，給診斷介面（未來的 `/okx-admin/isolated-check` endpoint）surface 真實錯誤碼用。
3. **rest.set_leverage 失敗時 `logger.error` 帶 code + msg**——bool wrapper 不再吃掉錯誤資訊，Railway logs 能看到真實 OKX 5xxxx。
4. **加 AST-based signature-parity 測試** `test_executor_called_methods_exist_on_facade`：scan `executor.py` 找所有 `self._client.<X>` 呼叫，assert 全部存在於 OkxClient。這是 trail bug 修法（07eadff signature-parity test）的**自動化升級版**——不靠人列出哪些 method 要驗，AST 直接抓 callgraph。

**Rule:**

新功能加 OkxRestClient method 時，**必須同步加 OkxClient proxy**——這條沒人會記得，所以靠 test 強制。AST signature-parity test 已部署，未來任何 facade drift 都會在 pre-commit / CI fail。

**更根本的 rule（給未來自己跟未來 AI 協作）**：

**fail-safe except 必須分類錯誤後再決定 user-facing 文字**。寫 generic `except Exception: alert("X failed")` 是把所有問題壓成同一條訊息、誤導 downstream debug。正確做法：
- `except AttributeError` → alert 「internal facade error, missing method X」+ raise to top
- `except OkxAPIError` → alert 「OKX rejected: code=Y msg=Z」
- `except (ConnectionError, TimeoutError)` → alert 「network unreachable, will retry」

每一種 user 採取的下一步動作完全不同。把它們合併成「failed」= 強迫操作員猜根因 = 浪費時間 + 誤判風險。

**Symptom-to-search 規則**：看到「某 exchange method failed」alert，第一個 grep 不是 OKX docs，是 `grep -n "<method>" indicator/okx/*.py` 看 facade chain 是不是斷的。**Trail bug 兩次、set_leverage 一次，這個方向應該排第 1。**

---

## 2026-06-07: admin_heal 第二次造孤兒倉——破壞性操作掛在無認證 GET + 只改 DB 不平 OKX

**What happened:**
`/okx-admin/heal` 這個 endpoint 在 06-07 08:26 自動把一筆 live SHORT 0.29（id=6，executor 02:00 正常開的）在 DB 裡標成 `status=CLOSED`（exit_reason=admin_heal），但**完全沒去 OKX 平倉**。09:02 對帳發現「OKX 有、DB active 沒有」→ `orphan_exchange` → executor DEMOTE，推了一條「MANUAL INTERFERENCE DETECTED」假警報（其實不是手動，是 endpoint 被自動觸發）。

時間線鐵證：08:02 對帳還 CONSISTENT（兩邊都有 SHORT）→ 08:26 admin_heal 抹 DB → 09:02 orphan_exchange。OKX 唯讀查證實 SHORT 還活著、強平價 $97K（離現價 +55%、碰不到）、stop algo 還 live @ 63148——所以**根本沒有風險事件，只有人造的狀態不一致**。

這是 **6/4 admin_heal 事件的第二次重演**（6/4 是 orphan_local，CLAUDE.md 記過）。第一次只處理了當次孤兒倉，沒根治 endpoint 本身，於是換個方向（orphan_exchange）又炸一次。

**Root cause:**
兩個疊加的設計缺陷：
1. **破壞性操作掛在無認證 GET**：`@app.route("/okx-admin/heal", methods=["GET"])`，要 `?confirm=YES` 才執行。問題是一個存過的 `.../heal?confirm=YES` 完整鏈接（Telegram 訊息/文檔/監控配置裡），會被 **link-preview bot / 瀏覽器預取 / uptime probe 連 query string 一起 GET**，自動帶 confirm=YES 觸發歸零。GET 依設計應幂等只讀，把「歸零 live 倉位」放 GET = 等著被預取誤觸。
2. **heal 只 UPDATE DB、不碰 OKX**：函數體全是 SQL（close DB rows + reset executor + resolve kill_log），注釋 L947 聲稱「Re-fetch positions from OKX via REST」但**代碼根本沒這段**。所以一旦 OKX 實際有倉，歸零 DB 必然製造 orphan_exchange。

**Correct approach（已修，commit 待 push）:**
1. **破壞性路徑改 POST-only**：`methods=["GET","POST"]`，但 `execute = request.method=="POST" and confirm=="YES"`。GET 永遠 dry-run，link-preview/預取（都是 GET）再也觸發不了歸零。
2. **歸零前先查 OKX，有倉則拒絕**：execute 前 `OkxRestClient.get_positions("BTC-USDT-SWAP")`，只要 OKX 還有非 FLAT 倉位就回 409 拒絕，附 OKX 倉位明細 + 「先平 OKX 再 heal」提示。heal 從此只能清真正的 orphan_local（DB 有 rows、OKX flat）。
3. 驗證：py_compile + `app.url_map` 確認 `/okx-admin/heal → okx_admin_heal_api [GET,POST]` 綁定沒脫鉤。

**Rule:** 任何會改變**真實交易所狀態 / 真錢**的 endpoint，**絕不可掛在 GET**——GET 必須幂等只讀，否則 link-preview / 預取 / 重載會在你不知情時觸發。破壞性 admin 操作 = POST + token + **執行前先核對真實外部狀態**（never zero local state that the exchange still holds）。修一個 ops bug 時，要修**機制本身**不是只清當次的髒數據——6/4 只清了孤兒倉沒修 endpoint，6/7 就用另一個方向重演。對帳出現 orphan 時，第一個問「是不是某個 heal/reset 工具只動了單邊（DB 或 exchange）」。

---

## 2026-06-02: aggregate AUC lift 被 2 個 outlier folds 撐起來，per-fold mean 是負的

**What happened:**
為了突破 V7 0.54 AUC ceiling，我跑 WorldQuant 101 alphas adapted for single-asset（rank → ts_rank），跑 conditional IC 找出 6 個強候選（alpha008/047/005/020/024/084，cond_IC > 0.03 + frac_pos > 65%）。然後用 production trainer (`train_direction_reg_walk_forward`) 跑 ensemble A/B：V7 baseline 136 features vs V7 + 6 alphas (142 features)。

**Aggregate 結果看起來 GO**：
- sign_AUC: 0.59755 → 0.60473 = **+0.00718**（剛過 +0.005 部署門檻）
- Strong thr=0.008 WR: 83% → 100%（6 笔全勝）
- Strong thr=0.010：新門檻達成 1 trade 100% WR

我寫了 verdict 文字「DEPLOY: WQ101 candidates bring measurable lift」，準備推 user 走 2 週 paper validation。差一步就 commit。

幸好部署前最後一個 sanity check：**per-fold AUC lift 分布**——只花 5 分鐘，但翻盤：
- Mean lift: **-0.00442**（負的！）
- Median lift: -0.00529（負的）
- Positive lift folds: **37/77 = 48.1%**（不到一半）
- Std: 0.091（極不穩）
- Worst fold: -0.318，Best fold: +0.279
- Capped mean (clip ±0.05): +0.00023（等同 0）
- Bootstrap 95% CI: [-0.026, +0.016]（含 0）
- Bootstrap p(lift ≤ 0): **0.666**（66% 機率根本沒 lift）

aggregate +0.0072 是被 1-2 個極端 fold（max +0.28）撐起來的。**Median 是 -0.0053**。

**Root cause:**
**Aggregate AUC 跟 per-fold mean AUC 是不同 metric**。
- **Aggregate**：把所有 fold 的 OOS predictions pool 起來再算一次 AUC
- **Per-fold mean**：每個 fold 各算 AUC 後平均

當有 1-2 個 fold 有極端 improvement（例如某段 quiet market 剛好 alpha008 抓到 momentum），會把 aggregate 拉高，但 per-fold 平均不變。Pooled metric 對 outlier 敏感，per-fold 才是真實 generalization 訊號。

更深問題：**conditional IC 顯著 ≠ ensemble A/B 過**。
- Conditional IC 量「**alpha 跟 V7 線性 residual** 的相關」
- Ensemble A/B 量「XGB 加 alpha 後**非線性 ensemble** 預測是否改善」
- 兩者可以背離：XGB 已透過 tree splits 非線性捕捉類似 pattern → 加 raw alpha 變成 noise

也就是說 conditional IC 顯著只證明「**alpha 帶 V7 沒有的線性訊息**」，但 XGB ensemble 可能透過 conditional split 隱式抓到了 → 加進去**反而 hurt**（看到 best fold +0.28 但 worst fold -0.32 = high-variance signal）。

放大因素：6/2 之前的 [[mistake 2026-06-01]] 已經建立了「conditional IC > raw IC 篩選」紀律，但**還缺一步 per-fold sanity**。我以為 aggregate 過了就 deploy，差點重蹈覆轍。

**Correct approach:**
任何 ensemble A/B 的 verdict 必須**同時看**：
1. **Aggregate lift > +0.005**
2. **Per-fold mean lift > +0.001**
3. **Frac_positive folds > 55%**
4. **Bootstrap 95% CI 不含 0**

4 條都過才算「真實 lift」，缺一就**疑似 outlier 撐起來的假 lift**。

具體實作：寫進 `wq101_ab.py` 之類的 A/B script 末段——

```python
fold_lifts = [auc(new_fold) - auc(base_fold) for fold in folds]
n_pos = sum(1 for x in fold_lifts if x > 0)
boot_p = bootstrap_p_value(fold_lifts, hypothesis="lift > 0", n=2000)

if (aggregate_lift > 0.005
    and np.mean(fold_lifts) > 0.001
    and n_pos / len(fold_lifts) > 0.55
    and boot_p < 0.05):
    verdict = "DEPLOY"
else:
    verdict = "NO-GO (aggregate may be outlier-driven)"
```

**Rule:** Ensemble A/B 看到 aggregate AUC lift 過門檻時，**強制再算 per-fold mean + frac_positive + bootstrap CI 4 條 sanity**。光看 aggregate 等於 [[mistake 2026-06-01]] 在升級版重演——只是這次「univariate IC 過」變成「aggregate AUC 過」，本質都是「**outlier 撐起 averaged metric 但 generalization 不行**」。Conditional IC 過只是「值得試 A/B」的 trigger，不是「值得 deploy」的證據；ensemble A/B aggregate 過也只是「值得 per-fold sanity」的 trigger，不是 deploy 證據。**驗證鏈條每加深一層都要重新 sanity check**。

更實務的紀律：**如果 5 分鐘的額外 check 能省下 2 週 paper validation，永遠先做這個 check**。今天這個 sanity 省下了：(a) 中斷現有 V7 paper cohort (b) 訓練 new model 等 1 小時 (c) 2 週 wait 然後發現沒差 (d) 浪費 14 天 paper baseline 比較性。**Validation discipline 的 ROI 是「上游 5 分鐘擋下下游 2 週的浪費」**。

**Update**: 證實 V7 對「OHLCV + Coinglass + Deribit + Binance order flow」這幾個 data source 已飽和。突破方向必須是**異源 channel**：(1) options gamma exposure (paid Deribit/Glassnode), (2) whale on-chain wallet flow (Glassnode), (3) Bitcoin ETF AUM/flow (CoinGecko 開放), (4) Twitter/Reddit sentiment (DIY scraper)。優先順序按「取得成本 vs 預期 lift」評估。

---

## 2026-06-01: walk-forward univariate IC 漂亮但加進 ensemble 沒 lift（feature redundancy）

**What happened:**
為驗證使用者「market moves to least resistance」的訂單流原則，我跑了一輪 walk-forward IC sweep（`research/liquidity_proxy_features.py`）。8 大類 21 個 microstructure proxy 特徵，做了 30d-train / 7d-OOS / 4-fold rolling 走勢驗證。結果非常漂亮：

- 12 個 feature 通過 |mean_IC| > 0.05 + 4/4 fold 同向
- 最強 `A_swing_high_dist_168h` mean_IC **+0.207**（V7 既有最強 feature ~0.07，看起來是 3x lift）
- 7 個獨立特徵（greedy de-dup |corr|<0.5）全部 4/4 同向

看起來非常有信心。於是寫了 A/B retrain script（`research/dual_model/train_with_liq_features.py`），用相同 XGB 超參數 + 77-fold WF split 比較「V7 baseline (136 features) vs V7 + 7 liq features (143 features)」。**結果：sign_AUC 從 0.5208 掉到 0.5178（-0.0030），IC 兩者都 ≈0**。

也就是說：univariate WF IC 看起來強的 feature，加進 ensemble **完全沒有 marginal information value**。

**Root cause:**
**Feature redundancy 在 XGBoost ensemble 裡是常見現象**。V7 的 136 個既有 feature（CVD divergence、OI delta、vol_kurtosis、impact_asymmetry、各種 z-score、return lag）已經透過 tree split 重組出類似 swing distance、sweep magnitude 的訊息。新加的 raw 特徵雖然 univariately 有訊號，但 **conditional on 既有 features 的訊號=零**。

更深的問題是我**只看 univariate IC 就下結論「這是 V7 強 3 倍的新 alpha」**。正確比較應該是「marginal IC given V7 model」— 也就是 V7 預測 residual 跟新 feature 的 IC。如果 V7 residual 跟新 feature 不相關，新 feature 對 V7 才有 lift。我這次直接用 raw IC 比較 V7 整體 IC，是 apples-to-oranges：raw IC 量「跟 target 相關」，但 V7 IC 量「ensemble 預測誤差」。一個 feature 可以很 univariately 相關但對 ensemble 全無 lift。

放大因素：walk-forward N=4 folds 太少。frac_positive=4/4 看起來很穩，但隨機 4/4 同向機率 = 1/16 = 6.25%。7 個獨立特徵全 4/4 同向是不太可能（聯合機率極低），但**每個獨立 feature 的 IC 估計值本身仍有大量 noise**。可能我看到的 +0.207 在更多 folds 之後會收斂到 +0.05 或更低 — 還是有 signal，但沒「3x V7」這麼誇張。

**Correct approach:**
1. **加新 feature 前永遠跑 ensemble A/B**，不是只看 univariate IC。Univariate IC 量的是「跟 target 的 raw 相關性」，ensemble 已經透過 tree split 吸了大半。要看 lift 必須是「加進去 ensemble 後 OOS AUC 是否提升 +0.005 以上」。
2. **若一定要用 univariate metric 做篩選**，用 **conditional IC**：先用 V7 baseline 預測，算 residual = y - pred，然後算新 feature vs residual 的 IC。Conditional IC 顯著 > 0 才值得進 ensemble。原始 IC 顯著只證明「跟 target 有關」，沒證明「V7 沒抓到」。
3. **WF fold 數 N < 10 時的「全 fold 同向」結論要打折**。N=4 同向看起來 4/4，實際統計強度約等於 binomial p=0.5 下 4 trials 全成功，p-value = 1/16 = 0.0625（剛過 5% 邊界）。要 N≥10 同向結論才篤定。
4. **負面結果一樣要記下來**，未來別人不會（或自己不會）重複跑同樣的 univariate IC sweep 結果 hyped。`research/orderbook_liq_features.py` 跟 `research/liquidity_proxy_features.py` 一起留作「univariate IC 高但 ensemble 沒 lift」的案例。

**Rule:** 任何「新 feature 加進 V7 / V8 ensemble」的決定必須基於 **ensemble A/B retrain 的 sign_AUC 或 IC lift**，不是 univariate WF IC。Univariate IC 高表示「跟 target 有 raw correlation」，但 conditional on ensemble 的剩餘 signal 才是真正的 marginal alpha。看到 univariate IC 比 V7 既有 feature 高 2-3 倍時 — **特別**要警覺，這往往是已經被 V7 吸收的訊息以另一種包裝出現。下次先跑「conditional IC vs V7 residual」 → 若顯著再 ensemble A/B → 都過才整合。

**Update 2026-06-02:** 重跑 A/B 用 production training function（`research/dual_model/rerun_liq_ab_with_prod_trainer.py`，import `train_direction_reg_walk_forward` 直接）驗證上面結論：BASELINE V7 sign_AUC 0.6030 / IC 0.180（跟 canonical OOS 0.593/0.170 對齊），NEW V7 + 9 liq features sign_AUC 0.6036 / IC 0.186 — **+0.0006 AUC、+0.006 IC**，仍遠低於 +0.005 部署門檻。原始結論「不要部署」**仍然成立**，但要注意：上次第一版 A/B baseline 訓練設定有差（custom eval_set 早停太凶導致預測退化），所以兩個 broken model 之間「無 lift」的觀察方向對，但**比較的絕對值都是錯的**。下次 A/B 要**直接 import 生產訓練函式**避免 hyperparam drift。

---

## 2026-05-31: Edit 把新函式塞進 `@app.route` 跟 `def webhook()` 之間，decorator 被靜默搶走，Telegram bot 全死

**What happened:**
commit c758336 加 `_handle_okx_perf` 函式時，我用 Edit 工具改 BTC_perp_data.py 的 old_string = `def _handle_okx_approval_response(...):`，new_string = `def _handle_okx_perf(...): ...\n\n\ndef _handle_okx_approval_response(...):`。結果新函式被插入到 `_handle_okx_approval_response` 之前。

`_handle_okx_approval_response` 本來就是我之前（commit e531b2c）用同樣手法插在 `def webhook():` 之前的——那一次也是把 webhook 上方的 `@app.route(f"/{TOKEN}", methods=["POST"])` decorator 跟 `def webhook()` 拆開了，但因為 `_handle_okx_approval_response` 的 signature 是 `(chat_id, raw_cmd)`，Flask 路由把 POST 進來時 Werkzeug 報 "TypeError: missing argument" → 變成 500 給 Telegram。**那次沒爆只是因為 Telegram 平常不會故意打 webhook 來驗證，Flask app 也沒在啟動時報錯**。直到我這次再插一個 `_handle_okx_perf` 在更前面，decorator 又被搶過去——這一次完全相同的問題終於在用戶按 V7 Stats 按鈕時暴露。

症狀很迷惑：bot service 的 `/` 主頁回 200 「OKX BTC Liquidity Outcome Bot is running」(因為 `/` 的 decorator 跟 def 是黏在一起的，沒被動到)，但 Telegram getWebhookInfo 顯示 `last_error_message: "Wrong response from the webhook: 500 Internal Server Error"`，每個指令、每個按鈕都死，**包括完全沒碰過的 /help**。用戶看起來就是「bot 沒反應」，沒有任何錯誤訊息能讓他自己 diagnose。

Python 語法檢查、import 檢查、unit test 全都過——因為這個 bug 是「decorator 綁錯函式」，不是任何 lint 工具會抓的。要等到 HTTP request 真的進來、Flask 帶錯誤參數 call 那個函式，才會炸。

**Root cause:**
我把 `def webhook():` 之前的某行（裡面有獨立函式 `_handle_okx_approval_response` 或 `_handle_okx_perf`）當成 anchor 點插入新函式，沒注意到該函式緊鄰 `@app.route(...)` decorator，而 Python decorator 是 syntactically 綁到「decorator 下面那一個 def」上的——我的 Edit 把新 def 塞在中間，等於把 decorator 從 `def webhook` 拔走、轉嫁到我新插的 def 上。

更深層原因：Flask 的 `@app.route` 沒有「綁定檢查」——decorator 隨便綁到哪個函式它都不會 raise，只是綁的那個函式變成 endpoint。Runtime 在 Telegram 打過來、Flask 用沒給 chat_id 參數的方式 call 它時才出 TypeError。再加上 Flask app 對 unhandled exception 的預設行為是回 500 給 client，沒任何 startup-time 警報。

放大因素：我寫 commit message 的時候 grep 路徑下的 `@app.route` 看其它路由還在，但沒去看每個 decorator 是不是綁到「原本應該的 def」上。 sanity check 是「routes 都存在」而不是「routes 都綁對函式」。

**Correct approach:**
1. 任何 Edit 動到「Flask / Django route 檔案中、靠近 `@app.route` 或 `@app.get` 之類 decorator 的位置」，必須**讀 Edit 後的整段** 至少 ±10 行，確認 decorator 跟原本的 def 仍然黏著。
2. 加新 helper 函式時，**找一個遠離 route handler 的安全位置**插入。例如：放到檔尾、或一個獨立的 `# === Helpers ===` 區塊。不要見縫插針地塞在現有 route 旁邊。
3. push 前如果改了 Flask app 檔案，用 `python -c "from BTC_perp_data import app; print([str(r) for r in app.url_map.iter_rules()])"` 檢查所有 route 跟 view function 的對應關係。`url_map` 印出來會看到 `<Rule '/<token>' (POST) -> webhook>`，如果看到 `-> _handle_okx_perf` 就是綁錯了。
4. Bot service 應該有一個 startup smoke——例如「啟動後對自己 webhook POST 一個空 JSON，預期回 200 ok」，啟動失敗的話 Railway 部署應該直接 fail，而不是部署成功讓 silent failure 跑半天。

**Rule:** 用 Edit 工具改 Flask / Django / FastAPI route 檔案時，**絕對不要把 anchor 點選在 `@app.route` decorator 下方那行**。如果一定要插入，要連同 decorator 一起包進 old_string，或選擇遠離任何 decorator 的位置（例如 helper section、檔尾）。Edit 後務必目視確認每個 decorator 還黏在原本的 view function 上。Python decorator 跟 def 沒有任何語法保護，綁錯了 lint/syntax/import 都不會炸，只在 runtime 有人打 endpoint 時才以「500 + missing positional argument」現身。**Symptom 是「Flask 主頁活著但某個 route 全死」、「Telegram 顯示 500 但 code 看起來沒問題」**——下次看到這種模式，第一個查 `app.url_map`。

---

## 2026-04-22: 新特徵邏輯貼進錯誤的 helper 函數，signature 不符導致 NameError 整夜停機

**What happened:**
b604afc commit 新增 SPX / DXY / US10Y / FNG 等 cross-market 特徵時，把一整段使用 `cross_market` 和 `fear_greed` 變數的程式碼**除了貼進 `build_live_features()`（正確位置，signature 有這兩個參數），又同時重複貼進 `_inject_coinglass(df, cg_data)` 這個 helper 函數裡**。後者的 signature 根本沒有這兩個參數。

commit 後不會在 import / 語法檢查階段報錯——因為 `_inject_coinglass` 只在 runtime 被呼叫，而且只有執行到 `if cross_market:` 那行才丟 `NameError: name 'cross_market' is not defined`。Railway 部署成功（build 綠），Process 啟動成功，但每次 `update_cycle()` 跑到該行都 crash。外層 `try/except` 把錯誤吞成 `_state["status"] = "error"`，整個下游（predict、render chart、Telegram 推送、寫 MySQL）全部靜默停擺。

結果：23:00 push 生效 → 到隔天 09:00 用戶打 `/indicator-status` 看到 `error: name 'cross_market' is not defined` 才發現，整整 10 小時圖表沒有新 bar。

**Root cause:**
加新特徵時沒有確認「我把這段貼進的函數，它的 signature 有沒有我要用的變數」。編輯器的 copy-paste / multi-insert 很容易一次改多處，中間的某一處 paste 點如果剛好在錯誤的 helper 函數裡，本機的靜態檢查（`python -c "import module"`）抓不到——只有 runtime 執行那條分支才會炸。push 前沒跑任何會觸發 `update_cycle()` 的本地測試。

另一個放大因素：`update_cycle()` 外層 `try: ... except Exception as e: _state["status"] = "error"` 把錯誤吞太深，只在 `/indicator-status` 才看得到；沒有任何 alert 機制通知「Railway 進程活著但內部邏輯壞了」這種 **silent failure**。Railway build 綠 + process 活著 = 用戶預期功能正常，但實際上功能靜默死亡。

**Correct approach:**
1. 任何 commit 如果碰到 `indicator/feature_builder_live.py`、`indicator/app.py` 的 `update_cycle()`、`indicator/inference.py` 等 hot-path 檔案，push 前必須至少跑一次 `python -c "from indicator.app import update_cycle; update_cycle()"` 或 `/force-update?sync=1` 的本地版。import 成功不代表 runtime 成功。
2. 加新變數到某段邏輯時，先 grep 確認自己貼進的那個 `def` 的 signature 到底有沒有這個變數。如果沒有，**不是** `def` 漏了參數（先問自己「這段邏輯是不是根本不該在這個函數裡」），而是貼錯函數了。
3. Silent failure 的監控：`update_cycle()` 外層 except 應該把 last error 暴露到一個更顯眼的健康指標（不是只有 `/indicator-status`），並且觸發 Telegram 告警——「Railway 活著但 update_cycle 連續 N 次 error」應該被當成 critical。這個後續要補。

**Rule:** 碰 hot-path 檔案（`feature_builder_live.py` / `app.py` `update_cycle` / `inference.py`）的 commit，push 前必須跑一次真實的 `update_cycle()` 或 `/force-update?sync=1`。**import OK 不代表程式碼會跑**——Python 的 NameError 只在執行那條分支時才炸。加新 feature code 時，grep 貼入處的 `def ...():` signature，確認引用的所有外部變數都在 signature 裡或在該 scope 可見。Silent failure（process 活著但邏輯死了）是最危險的故障模式，因為 Railway dashboard 看起來全綠。

---

## 2026-04-14: 把 Strong 勝率目標寫成 95%（從策略系統沿用未更新）

**What happened:**
CLAUDE.md 長期寫「Strong 信號勝率目標 > 95%」，花了一整天嘗試各種方法提升 Direction model 都碰天花板，才回頭檢查這個目標本身是否合理。跑 `research/topk_precision_sweep.py` 用 2726 筆 walk-forward OOS 預測做 bidirectional top-k：

| k | precision | CI | signals/月 |
|---|---|---|---|
| top 1% | 59.3% | [40.7, 75.5] | 5 |
| top 2% | 63.6% | [50.4, 75.1] | 11 |
| **top 5%** | **67.6%** | **[59.4, 74.9]** | **27** |
| top 10% | 65.6% | [59.8, 71.0] | 53 |
| top 20% | 60.2% | [56.0, 64.2] | 106 |

峰值 67.6%。AUC 0.57 的理論 top-5% precision 上限 68-72%，代表**已經貼著數學天花板**。95% 在這個 AUC 結構下永遠達不到，那是 AUC 0.80+ 的模型才談的數字。

**Root cause:**
95% 是從早期策略系統（有 TP/SL、能過濾掉不利情境）沿用到指標系統，沒人重新校準。而指標系統的訊號是「原始預測」，不過濾，所以上限直接由模型 AUC 決定。把策略目標搬到指標系統等於給自己設一個數學上不存在的目標。

**Correct approach:**
precision 目標必須從**模型 AUC 反推**，不是拍腦袋定。公式約等於：
- 給定 AUC，top-k precision 上限 ≈ 0.5 + (AUC - 0.5) × kernel(k)
- AUC 0.57 + k=5% → 理論 ~0.70，實測 0.676（非常接近）
- 如果要求 0.95，需要 AUC ≥ 0.85

現在 CLAUDE.md 已改為「point estimate ≥ 65%，stretch 70%」。未來任何討論 Strong 勝率時，第一句話要問「當前模型 AUC 是多少，這個目標在結構上可達嗎」，不是「為什麼還沒達到」。

**Rule:** 設定任何 precision/recall 目標前，先用當前模型的 AUC/IC 反推理論天花板。如果目標高於天花板就是錯的目標，改目標而不是追目標。絕對不要從不同系統（策略 vs 指標）沿用績效目標——運作機制不同，天花板也不同。

---

## 2026-04-13: 用 in-sample 月份 IC 判斷訊號健康度（高估 0.5 AUC 級別）

**What happened:**
為了診斷 Magnitude IC 衰退，寫 `diagnose_mag_decay.py` 用**當前生產模型**去預測過去每個月的 `|ret_4h|`，得到 Nov 0.60 / Dec 0.51 / Jan 0.57 / Feb 0.53 / Mar **0.60** / Apr 0.47。看起來訊號完全沒衰退、近月甚至還很強，幾乎下結論「Mag 訊號穩定，問題不在這」。

後來跑乾淨 walk-forward（`mag_level_feat_swap.py`，每個測試窗只用之前的資料訓練）得到真實 OOS IC：Nov 0.31 / Dec 0.36 / Jan 0.24 / Feb 0.20 / Mar **0.10** / Apr 0.12。**Mar 差距 0.50 IC，Apr 差距 0.35 IC**。真實情況是 Mag 從 Feb/Mar 交界發生 concept drift，IC 腰斬。

也就是說，我的第一版診斷**用了訓練集預測訓練集**——生產模型訓練時吃了全部 4000 bars，對任何歷史月份做預測都是 in-sample，結果無法反映 model 是否能從歷史學到新規律。

**Root cause:**
沒區分「model fit」和「model generalization」。生產模型的 IC 是在**全部資料訓練完**才算的，拿它去預測過去月份天生是作弊。這跟 Kaggle 新手用 `cross_val_score` 之後又用全資料重訓再看 train loss 是同一個錯誤——只是換了個包裝。更糟的是，月份切片讓我以為這是「time-slicing 驗證」，實際上完全沒做 time-based split。

**Correct approach:**
任何「模型是否仍然 work」的評估都必須是**嚴格 walk-forward**：每個測試點的模型只能看到該點之前的資料。生產模型的 in-sample 預測**永遠不能拿來回答「訊號是否衰退」「特徵是否還有效」「regime 是否改變」這類問題。能用 in-sample 回答的問題只有：「訓練收斂了沒」「在完整資料上 model 的 upper bound 在哪」。

**Rule:** 診斷 IC/AUC 衰退時，第一句 assert 必須是「這個預測是 walk-forward 還是 in-sample」。in-sample 的結果在「診斷衰退」這個 task 下**零資訊量**，不管數字多漂亮都等於沒測。如果檢查清單裡的測試方法是「用生產模型預測過去月份」，那就是錯的測試方法，換掉。

---

## 2026-04-13: Regime-specific 子模型在小樣本下退化成比隨機還差

**What happened:**
為了試圖突破 Direction model 天花板，訓練 bull/bear/chop 三個 regime-specific 子模型（`regime_specific_direction.py`）。假設是「每個 regime 的特徵→方向關係不同，獨立訓練應該贏過全局模型」。

結果全局模型在三個 regime 上的 AUC 分別是 CHOP 0.548 / BULL 0.500 / BEAR 0.497，regime 子模型是 CHOP 0.550 / BULL 0.440 / **BEAR 0.378**。BEAR 子模型 AUC 顯著低於 0.5，意味著它**系統性預測反方向**。

原因：BEAR 整個 4000 bar 資料集只有 724 筆，扣掉 walk-forward test + NaN，每個 split 的訓練集只剩 50-100 筆。XGB 在這個樣本數下嚴重 overfit 訓練集的雜訊方向，預測出來的機率跟實際 label 反相關。BULL 也有 16 個 split 因為訓練樣本不足 < 50 直接跳過，等於是選擇性覆蓋。

**Root cause:**
沒評估「資料切片後每個 regime 的有效樣本數是否夠訓練」。少於 ~500 的小樣本訓練 gradient boosting 會 overfit 到雜訊，而且資料越少 overfit 越嚴重，甚至可能學到完全相反的方向。把這當成「子模型比較弱」來解讀是錯的——這些子模型根本沒進入「能學東西」的 regime。

**Correct approach:**
切片訓練前先算 min(regime_sample_count) 是否 > 500（gradient boosting 的大略安全線）。如果不夠：
1. **退一步用 regime dummies 當 feature** 讓全局模型自己學 conditional split（這是 XGB 設計本來就能處理的）
2. 或用 `sample_weight` 在全局訓練時加權少數 regime，**不要**切開訓練
3. 如果真要切，只切樣本充足的 regime（這個資料集只有 CHOP 有 2000+ 筆，結論是：沒得切）

**Rule:** 分群訓練前，每群的有效訓練樣本數必須 > 500（至少要 > 300），否則不如用單一模型 + 分群特徵。小樣本下 gradient boosting 不會變「局部專家」，會變「雜訊放大器」。如果樣本數不夠，把分群改成 feature 而不是改成 partition。

---

## 2026-04-13: 用混合模型版本的數據下 calibration 結論

**What happened:**
跑 `calibration_check.py` 看到 Brier skill -0.098、ECE 0.16、over-confident +0.197，bootstrap CI 全部顯著（[-0.184, -0.014] 整條在零下，conf_gap [+0.115, +0.285] 整條在零上），就據此推論「模型 miscalibration 是真的」並開始討論 Platt scaling / isotonic / rolling percentile threshold 等解法。

然後往下挖才發現 244 個 valid 樣本全部來自 2026-04-02 → 2026-04-12 這 10 天，這個窗期：
  - 2026-04-03 部署 dual v7 初版（88 特徵）
  - 2026-04-09 切換成 pruned 29 特徵 + regime weighting
  - 2026-04-12 又重訓一次
  - 5.5 天 `cg_bfx_margin_ratio`（第 4 重要特徵）灌壞數據（2026-04-12 backfill bug）

也就是說：calibration 測試基於 **三個不同模型的混合預測 + 重大特徵被污染一半時間**。bin-level 極端區的怪象（p≥0.70 actual=0.50）很可能只是模型切換那幾小時產的 transient，不代表任何一個模型的穩態。前面提出的所有解法都建立在錯誤的前提上。

**Root cause:**
看到統計顯著的壞結果就急著找解法，沒先問「這個測試數據對應的是哪個模型？數據本身是乾淨的嗎？」最基本的 data sanity check 被跳過了。更糟的是 bootstrap CI 讓我更有信心下結論——但 CI 只能量**抽樣不確定性**，量不到**數據污染**或**模型版本混合**這種系統性偏差。統計顯著 ≠ 結論可信。

**Correct approach:**
評估模型前必須確認：
  1. **樣本範圍對應單一模型版本**：git log 查最新模型 deploy 時間，樣本必須在那之後。
  2. **樣本範圍不含已知數據污染窗**：查 mistake log 看近期有沒有數據 bug。
  3. **樣本數夠**：即使資料乾淨，n<100 的 calibration 點估計不穩定；n<500 做 isotonic 會 overfit。
  4. **先看時間切片**：分月/分週跑同一個測試，如果每段結論都不同，整體測試就沒意義。

已在 `calibration_check.py` 的 roadmap 加上 `--since` flag 和 model version guard（讀取最新 model mtime，樣本必須 >= 該時間），還沒實作。

**Rule:** 評估任何模型的統計量前，第一件事是「確認這份評估樣本是從同一個模型 + 同一份乾淨數據產生的」。這個 sanity check 要**在看結果之前**做，不是看到壞結果才回去查。Bootstrap / permutation / 顯著性檢定全部都只能處理抽樣誤差，不能處理「你在量錯的東西」這種問題。看到「顯著的壞結果」第一反應應該是懷疑測試設計，不是懷疑模型。

---

## 2026-03-28: price_change fallback over-engineering

**What happened:**
Item 9 (fix `_get_price_change` dependency on normalized_trades) was implemented with 3 chained queries:
1. Query flow_bars_1m to find nearest bar
2. Query normalized_trades within that bar's time window
3. Fallback to delta/volume estimation

Step 1→2 was pointless — querying normalized_trades scoped to a flow_bar window is the same as querying it directly. This tripled the DB load per snapshot for no benefit.

**Root cause:**
Jumped to a "clever" solution without thinking about whether the intermediate step added value. flow_bars_1m doesn't store price, so using it as an index to find normalized_trades was a round-trip to nowhere.

**Correct approach:**
1. Try normalized_trades first (works for events < 3 days, same as original)
2. Only if no data, fallback to delta/volume ratio from flow_bars_1m

**Rule:** When adding a fallback path, ask: "Does this intermediate step give me information I don't already have?" If not, skip it. Prefer the simplest query chain that solves the problem. Don't add queries that increase Railway DB usage without clear value.

---

## 2026-03-29: delta/volume ratio ≠ price change

**What happened:**
`_get_price_change()` fallback used `total_delta / total_vol * 100` when normalized_trades had no data. This produced values like +4.84% that looked like real price moves but were actually the **taker imbalance ratio** (what % of volume was net buy).

**Root cause:**
Confused two different metrics. delta/volume ratio measures buy-sell pressure, not price movement. The two are correlated but not interchangeable — especially on short windows where slippage is minimal.

**Correct approach:**
Return None when normalized_trades has no data. Don't fabricate price estimates from flow data.

---

## 2026-04-01: 把 MACD / EMA 放進訂單流研究的特徵集

**What happened:**
`feature_builder_v2.py` 計算了 `ema_9`, `ema_21`, `macd`, `macd_signal` 並寫入 features 表。這些欄位出現在 feature validation 結果中，ICIR 看起來很高（-0.85~-0.91），但這根本不該存在。

**Root cause:**
誤把傳統技術指標混入訂單流研究。這個專案的研究範疇是純訂單流（CVD、delta_ratio、aggTrade flow、funding rate、OI），不包含 price-derived 的技術指標如 MACD / EMA / RSI 等。

**Correct approach:**
feature_builder_v2.py 只能包含以下來源的特徵：
- aggTrade flow（CVD、delta_ratio、buy/sell vol、large order）
- Funding rate（rate、deviation、zscore）
- OI（未來補充）
- Cross-exchange divergence
- 純統計衍生（realized vol、return lags）— 可接受，因為是 price behavior 而非 pattern indicator

MACD / EMA / Bollinger 等技術指標一律不加。

**Rule:** 每次加新特徵前先問：「這是訂單流資料還是技術指標？」技術指標一律排除。

---

## 2026-04-02: 加 log 行導致 webhook 500 crash

**What happened:**
在 `indicator/app.py` 的 `/webhook` handler 中加了一行 `logger.info("Webhook command: %s", cmd, chat_id)`，但放在 `cmd = text.split()[0]...` 定義**之前**。導致每次收到 Telegram 指令都觸發 `NameError`，回傳 500，用戶的 `/chart` 指令完全無反應。

**Root cause:**
加 debug log 時沒注意變數的定義順序。修改生產環境的 request handler 後沒有做基本的 code review（變數是否已定義）。

**Correct approach:**
新增的 log 行必須放在所有引用變數的定義之後。修改 webhook/route handler 這類每個請求都會跑的代碼時，要特別小心：一個 crash 會影響所有用戶。

**Rule:** 在生產 handler 中加 log 或任何代碼後，立刻檢查：所有引用的變數是否已定義？是否在 try/except 內？不要假設「只是加一行 log」就不會出錯。

---

## 2026-04-12: backfill 時間戳 unit 硬編碼導致 5.5 天數據缺口

**What happened:**
`research/backfill_all_parquet.py` 的 `to_1h_df()` 用 `pd.to_datetime(df[time_col], unit="ms")` 硬編碼毫秒。但 Coinglass API 的 `coinbase_premium` 和 `bitfinex_margin` 端點回傳的 `time` 欄位是 **10 位秒級時間戳**（如 `1775998800`），不是 13 位毫秒。秒級時間戳被當毫秒解析後變成 1970 年日期，merge_parquet 沒報錯（index dedup 保留了壞行），最終 4131 行壞數據 + 數據停在 2026-04-07。

`cg_bfx_margin_ratio` 是剪枝模型第 4 重要的特徵，如果下次訓練前沒發現這個缺口，模型會在該特徵上訓練出偏差。

**Root cause:**
假設所有 Coinglass 端點的時間戳格式一致。實際上大部分端點用 13 位 ms，但 `coinbase_premium` 和 `bitfinex_margin` 用 10 位 s。生產端的 `data_fetcher.py` 早就有 `if ts.max() > 1e12` 的自動偵測，但 backfill 腳本是另外寫的，沒抄這段邏輯。

**Correct approach:**
時間戳解析永遠用自動偵測：`unit = "s" if sample_ts < 1e12 else "ms"`。已修復。

**Rule:** 凡是解析時間戳的地方，永遠不要假設 unit 固定。寫新的數據處理腳本時，先看生產代碼怎麼處理同一個 API 的格式。同一個 API provider 的不同端點可以有不同的時間戳格式。

---

## 2026-04-12: is_stale() 只檢查 klines 導致端點級故障無聲

**What happened:**
`backfill_all_parquet.py` 的 `is_stale()` 只讀 `binance_klines_1h.parquet` 來判斷是否需要回填。klines 永遠是最新的（Binance 公開 API 不需要 key），所以即使 CG 端點已經停滯 5.5 天，`ensure_fresh()` 也不會觸發回填。訓練管線 `shared_data.py` 調用 `ensure_fresh()` 時以為數據是新的，實際上 coinbase_premium / bitfinex_margin 缺了 132 小時。

**Root cause:**
用最穩定的數據源（Binance klines）代表所有數據源的新鮮度。這是一種「以偏概全」的監控盲區 — 最不可能故障的組件被選為健康指標。

**Correct approach:**
`is_stale()` 改為遍歷所有 parquet 文件，任何一個超過 max_age_hours 就回傳 True。已修復。

**Rule:** 新鮮度 / 健康檢查必須覆蓋最脆弱的組件，不是最穩定的。如果系統有 N 個數據源，健康檢查要查 N 個，不是只查最可靠的那一個。

---

## 2026-04-12: 用錯 IndicatorEngine 屬性名（dir_model vs dual_dir_model）

**What happened:**
watchdog 新增的 `_check_dual_model()` 檢查 `engine.dir_model` 和 `engine.mag_model` 是否為 None。但 dual mode 下的屬性名是 `dual_dir_model` 和 `dual_mag_model`。`dir_model` 是舊 regime mode 的屬性，dual mode 下根本不存在，導致 `AttributeError`。

**Root cause:**
沒有先 grep 確認屬性名就寫代碼。`IndicatorEngine` 有三種 mode（dual/regime/legacy），每種 mode 的屬性名不同。

**Correct approach:**
寫監控代碼前先 `grep self\.dual_dir` 確認屬性名。已修正為 `dual_dir_model` / `dual_mag_model` + `hasattr` 防禦。

**Rule:** 引用物件屬性前先 grep 確認。特別是有多種初始化路徑的類別（如 IndicatorEngine 的 dual/regime/legacy），不同路徑設定的屬性名不同。不要憑記憶猜。

---

## 2026-04-13: 用 sparse indicator 做 feature interaction 是退化操作

**What happened:**
為了解決 Direction Model 的 regime 適應性問題，原本想加 9 個 regime interaction 特徵：
```python
oi_agg_close_x_bear = cg_oi_agg_close * is_bear
bfx_margin_x_bull   = cg_bfx_margin_ratio * is_bull
ls_ratio_x_bear     = cg_ls_ratio * is_bear
# ... 等等
```
寫法看起來完全合理，是 ML 教科書經典 interaction term 寫法。

跑 IC 驗證後發現怪事：4 個本質完全不同的金融特徵在 ×is_bear 之後互相相關 0.96-0.98：
```
bfx_margin_x_bear ↔ oi_agg_close_x_bear     corr = +0.984
bfx_margin_x_bear ↔ ls_ratio_x_bear         corr = +0.957
oi_agg_close_x_bear ↔ ls_ratio_x_bear       corr = +0.968
```
而且 IC 全部從 base 的 -0.05~-0.07 掉到 +0.01，p-value 變不顯著，train/test FLIP。

**Root cause:**
BEAR 只佔 18% 樣本（724/4000）。`feature × is_bear` 等於：
- 非 BEAR 時 = 0（佔 82%）
- BEAR 時 = feature 原值（佔 18%）

問題在於**「在哪些 timestamp 是 0」這個 sparsity pattern 在所有 ×is_bear 特徵裡完全一樣**。所有特徵共享同一組 18% 的非零 mask。剩下 82% 的零值貢獻了大部分變異數。

結果 spearman correlation 主要在量「這個 sample 是不是在 BEAR 期間」，而不是「這個 feature 在 BEAR 的時候值是多少」。三個 base 完全不同的特徵看起來幾乎一模一樣，因為它們的 0/非0 pattern 完全重疊——indicator 的 sparsity 訊號壓過了被乘的特徵本身。

**Correct approach:**
1. **乘以 `(1 - is_X)` 才有意義**：保留 80%+ 樣本，只把死掉的 regime 設 0。例如 `vol_kurt_non_bear = vol_kurtosis * (1 - is_trending_bear)`，IC validated +0.054 stable, `oi_8h_non_bull = cg_oi_close_pctchg_8h * (1 - is_trending_bull)` IC validated -0.071（比 base -0.062 強 15%）。
2. **regime indicator 本身要直接當 feature 加進去**（is_trending_bull / is_trending_bear），讓 XGB 自己用 tree split 決定 conditional rule。手動寫 `feat × is_X` 是把訊號塞進更窄的 channel。
3. **inter-feature correlation matrix 必須當成標準驗證步驟**，跟 train/test split、rolling IC 同等重要。如果一群本應獨立的特徵互相相關 > 0.9，那不是訊號，是 indicator pattern leakage。

**Rule:** 設計 interaction feature 時，**永遠不要寫 `feat × sparse_indicator`**——當 indicator 的非零比例 < 30%，乘出來的特徵會被 sparsity pattern 主導，跟其他用同一 indicator 乘的特徵高度相關，IC 也會 collapse。如果要做 regime conditioning：(a) 把 indicator 直接當 feature，讓 XGB 自己學 split；(b) 只在「base feature 在某 regime 完全死掉」的情況下用 `feat × (1 - is_dead_regime)` 形式屏蔽噪音。設計完任何 interaction 都要先跑 inter-feature correlation matrix。

---

## 2026-04-19: 多個腳本覆寫同一個 JSON 導致 warmup buffer 被清空

**What happened:**
系統連續產出大量 DOWN 信號，比例明顯不合理。排查後發現 `training_stats.json` 裡的 `dir_pred_history`（Direction model 的 500 筆 warmup 預測）是空的。沒有 warmup buffer，系統永遠用固定 fallback 閾值解碼方向——這些閾值是歷史均值，無法適應當前 bearish regime，結果只要模型預測稍微偏負就觸發 DOWN。

事件鏈：
1. 4/15 `export_direction_reg_model.py` 正確寫入 500 筆 `dir_pred_history` ✅
2. 4/16 `deploy_new_models.py` 重訓 Magnitude model，用 `json.dump` **整個覆寫** `training_stats.json`，只寫了 `pred_history`（mag 的 warmup），`dir_pred_history` 被洗掉 ❌
3. 之後每次 Railway 重啟（git push 觸發），buffer 歸零，永遠不到 100 根 warmup 門檻
4. 系統永遠用 fallback 閾值 → bearish 市場下 DOWN 信號爆量

**Root cause:**
三個腳本寫同一個檔案，但寫法不一致：
- `export_direction_reg_model.py`：先讀再寫（read-then-update）✅
- `deploy_new_models.py`：直接 `json.dump` 覆寫 ❌
- `export_production_models.py`：直接 `json.dump` 覆寫 ❌

後面兩個腳本沒有意識到這個檔案是**共用的**，裡面有別的腳本存的資料。這是最基本的共用資源協調問題。

**Correct approach:**
寫入已存在的 JSON/config 檔案時，永遠用 read-then-update 模式：
```python
if stats_path.exists():
    with open(stats_path) as f:
        stats = json.load(f)
else:
    stats = {}
stats["my_key"] = my_value  # 只更新自己負責的 key
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)
```

額外加了兩層防護：
1. `app.py` 每次 update cycle 結束後持久化 `dir_pred_history`，這樣 Railway 重啟不會失去已累積的 warmup
2. 修復了 `deploy_new_models.py` 和 `export_production_models.py` 的寫法

**Rule:** 寫入任何共用檔案前，第一步是 `grep` 看還有誰也在寫這個檔案。如果有多個寫入者，必須用 read-then-update 模式，只動自己負責的 key。直接 `json.dump` 覆寫整個檔案等於對其他寫入者說「你存的東西我不在乎」——這在單一寫入者時沒問題，多個寫入者時是資料刪除。

---

## 2026-04-19: 用 WF OOS fold 模型的預測初始化 rolling percentile buffer（分佈差 3.5 倍）

**What happened:**
修復 buffer 被覆寫的問題後，重新 seed `dir_pred_history` 時，從 walk-forward OOS parquet 的 `pred_ret` 欄位取了 500 筆預測作為 warmup buffer。部署後圖表上幾乎**所有 bar 都是紅色 DOWN 三角形**。

排查發現：WF OOS fold 模型的預測 std=0.0008，但生產模型（用全部資料訓練）的預測 std=0.003，**差了 3.5 倍**。用小範圍的 buffer 去校準大範圍的預測，rolling percentile 的 DOWN 門檻大約在 -0.0006，而生產模型的正常預測值動輒 -0.002~-0.005，幾乎所有 bar 都超過 DOWN 門檻 → 全部是 DOWN 信號。

**Root cause:**
WF OOS 的 `pred_ret` 是每個 fold 的子模型產生的，每個 fold 只用部分資料訓練。子模型因為訓練資料少，學到的 pattern 弱，預測值集中在零附近，variance 小。生產模型用全部資料訓練，學到更多 pattern，預測值的 variance 明顯更大。

這是 walk-forward 驗證的根本特性：fold 模型和生產模型的預測分佈**不在同一個尺度**。拿 fold 模型的輸出去校準生產模型的閾值，等於用錯誤的尺去量。

事件鏈：
1. 第一次修 buffer：從 WF OOS parquet 取 500 筆 `pred_ret`，buffer std=0.0008
2. 部署後生產模型預測 std=0.003，幾乎所有預測都落在 buffer 的極端尾部
3. Rolling percentile 把正常預測解碼成 Strong DOWN
4. 圖表全紅，Telegram 每根 bar 都推 DOWN 信號

**Correct approach:**
每次重訓方向模型後，用**生產模型本身**在訓練資料上跑一次 predict，取最後 500 筆作為 `dir_pred_history`。同時用全部預測更新 `direction_reg_config.json` 的 fallback thresholds（2.5%/7.5% 分位數）。

驗證方式：比較 buffer std 和生產模型最近 200 筆的 std，ratio 應在 0.5~2.0 之間。最終修復後 ratio = 0.74x，信號分佈回到 5.5% UP / 88% NEUTRAL / 6.5% DOWN，符合 ~10%/80%/10% 的設計目標。

**Rule:** Rolling percentile buffer 的初始化**只能用生產模型的預測**，不能用 WF OOS fold 模型的預測。WF OOS 的預測只能拿來評估模型泛化能力（IC、AUC），不能拿來校準生產閾值——它們的分佈不在同一個尺度。每次 seed buffer 後，必須比較 buffer std vs 生產模型 std，ratio 偏離 0.5~2.0 就是 red flag。

---

## 2026-08-20: bash 雙引號把 .bat 附加行裡的 \v \r 吃成控制字元——「改完立刻執行一次」規矩當場抓到（avoided）

**What happened:**
往 `shadow_engine.bat` 附加 v7_veto_publish 那行時，為了避開 08-19 的
CRLF 坑，特地用 python 以 bytes + `\r\n` 附加——但 python 程式碼是包在
**bash 雙引號的 `python -c "..."`** 裡传的。bash 在雙引號內先吃一層
反斜線：`\\\\v` → python 看到 `\\v` → bytes 裡是**垂直定位符 0x0B**，
`\\r` 同理變 CR。落到 .bat 的是 `python research␋7_veto_publish.py >>
research␍esults\...`——cmd 報「系統找不到指定的路徑」。

**為什麼沒釀災**：08-19 立的規矩「改任何 .bat 之後必須立刻整支執行
一次、看產物不看狀態燈」當場執行了——log 尾巴沒有預期的發布行，兩分鐘
內定位到壞行。壞的只有新附加的最後一行，既有班車不受影響。

**Rule:** 要把含反斜線路徑的內容寫進檔案，**不要經過 bash 雙引號的
`python -c`**——寫成 .py 檔再執行（零 shell 轉義層），或在 python 字串裡
用 forward slash / `chr()` 組合。`\v`、`\r`、`\n`、`\t`、`\f` 開頭的
Windows 路徑段（`\results`、`\v7_...`、`\tasks`）是高危組合。08-19 規矩
補充版：**「執行一次」的判準是產物內容（log 末行是不是新步驟的輸出），
不只是 exit code**——這次 bat 整體 exit 0，死的只有最後一行。

---

## 2026-08-19: 用 Edit 工具改 .bat 把 CRLF 換成 LF，每小時記帳排程靜默死了 29 小時

**What happened:**
08-18 為了把 `pf_dry_intents.py` 接進每小時班車，用 Edit 工具改
`shadow_engine.bat`。Edit 寫回時把整個檔案的行尾從 CRLF 換成 LF。
Windows 排程照常在跑、`State=Ready`、每小時觸發——但 `LastTaskResult=1`，
而且**連 log 都沒寫進去一行**（run log 的 mtime 停在 08-18 07:05）。
cmd.exe 對純 LF 的 .bat 解析不了，整個檔案等於沒有內容。

停機 29 小時的東西不只是記錄：shadow log 少記 54 筆訊號（復跑後
Variant B 從 732 → 786）、天氣站快照凍在 08-18、pf 帳本與乾跑 intent
流全部沒跑。而且它是**在我查 JARVIS 橋接時偶然發現的**——沒有任何
告警說「每小時的東西一天沒跑了」。

**Root cause:**
兩層。(a) 文字編輯工具對 Windows 批次檔的行尾沒有保護，而 `.bat` 是
少數「行尾錯了就整個失效」的檔案格式；(b) 改完排程檔之後**沒有實際
執行它一次**——測了 python 腳本本身、沒測呼叫它的那層。這正是
[[2026-07-05 DailyCollect 排程指向舊路徑 96 天]] 的同款：排程面板
顯示健康、實際工作早就死了，而且死在「呼叫層」不是「被呼叫層」。

**Correct approach（已修）:**
1. 行尾轉回 CRLF，直接 `cmd /c` 跑一次 bat 驗證 exit=0 且 CSV mtime 更新。
2. 從此改任何 `.bat`／`.cmd` 之後，**必須立刻執行它一次**，不是只跑
   它內部的 python。
3. 排程健康的判準跟交易系統同一條：**看它產物的新鮮度**
   （`sweep_shadow_log.csv` 的 mtime），不看 `State=Ready`。
   `LastTaskResult=1` 要當紅燈，不是雜訊。

**Rule:** 用文字編輯工具碰 Windows 批次檔之後，**先驗行尾再驗執行**
（`python -c "print(open(p,'rb').read().count(b'\r\n'))"`，再 `cmd /c` 跑
一次）。任何「每小時／每天」的排程，判斷它活著一律看**產物 mtime**，
不看排程面板的狀態燈——面板顯示的是「有沒有被觸發」，不是「有沒有
做完事」。
