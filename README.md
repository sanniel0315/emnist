# 手寫英文字母辨識系統

這是一個基於深度學習的手寫英文字母辨識系統，使用 Streamlit 搭建網頁界面，允許用戶在畫布上繪製英文字母並進行辨識。

## 功能

* 提供互動式畫布供用戶繪製英文大寫字母
* 即時辨識繪製的字母
* 顯示辨識結果和置信度
* 顯示處理後的圖像
* 顯示各字母的預測機率

## 環境需求

* Python 3.7 或更高版本
* 所需套件列於 `requirements.txt`

## 安裝步驟

1. 克隆或下載本專案
2. 安裝必要的依賴套件：
   <pre><div class="relative group/copy rounded-lg"><div class="sticky opacity-0 group-hover/copy:opacity-100 top-2 py-2 h-12 w-0 float-right"><div class="absolute right-0 h-8 px-2 items-center inline-flex"><button class="inline-flex
     items-center
     justify-center
     relative
     shrink-0
     can-focus
     select-none
     disabled:pointer-events-none
     disabled:opacity-50
     disabled:shadow-none
     disabled:drop-shadow-none text-text-300
             border-transparent
             transition
             font-styrene
             duration-300
             ease-[cubic-bezier(0.165,0.85,0.45,1)]
             hover:bg-bg-400
             aria-pressed:bg-bg-400
             aria-checked:bg-bg-400
             aria-expanded:bg-bg-300
             hover:text-text-100
             aria-pressed:text-text-100
             aria-checked:text-text-100
             aria-expanded:text-text-100 h-8 w-8 rounded-md active:scale-95 backdrop-blur-md" type="button" aria-label="Copy to clipboard" data-state="closed"><div class="relative *:transition"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 256 256" class="scale-100"><path d="M200,32H163.74a47.92,47.92,0,0,0-71.48,0H56A16,16,0,0,0,40,48V216a16,16,0,0,0,16,16H200a16,16,0,0,0,16-16V48A16,16,0,0,0,200,32Zm-72,0a32,32,0,0,1,32,32H96A32,32,0,0,1,128,32Zm72,184H56V48H82.75A47.93,47.93,0,0,0,80,64v8a8,8,0,0,0,8,8h80a8,8,0,0,0,8-8V64a47.93,47.93,0,0,0-2.75-16H200Z"></path></svg><div class="absolute inset-0 flex items-center justify-center"><label class="select-none inline-flex gap-3 cursor-pointer text-left"><div class="relative"><input class="sr-only peer" type="checkbox" data-sharkid="__9"/><div class="w-4 h-4 overflow-hidden flex items-center justify-center border rounded transition-colors duration-100 ease-in-out peer-focus-visible:ring-1 ring-offset-2 ring-offset-bg-300 ring-accent-main-100 bg-bg-000 border-border-200 hover:border-border-100 cursor-pointer rounded-full scale-50 opacity-0"></div></div><span class="leading-none sr-only"></span></label></div></div></button></div></div><div class="text-text-500 text-xs p-3.5 pb-0">bash</div><div class=""><pre class="code-block__code !my-0 !rounded-lg !text-sm !leading-relaxed"><code class="language-bash"><span><span>pip </span><span class="token">install</span><span> -r requirements.txt</span></span></code></pre></div></div></pre>

## 使用方法

1. 首先訓練模型（首次使用時需要）：

   <pre><div class="relative group/copy rounded-lg"><div class="sticky opacity-0 group-hover/copy:opacity-100 top-2 py-2 h-12 w-0 float-right"><div class="absolute right-0 h-8 px-2 items-center inline-flex"><button class="inline-flex
     items-center
     justify-center
     relative
     shrink-0
     can-focus
     select-none
     disabled:pointer-events-none
     disabled:opacity-50
     disabled:shadow-none
     disabled:drop-shadow-none text-text-300
             border-transparent
             transition
             font-styrene
             duration-300
             ease-[cubic-bezier(0.165,0.85,0.45,1)]
             hover:bg-bg-400
             aria-pressed:bg-bg-400
             aria-checked:bg-bg-400
             aria-expanded:bg-bg-300
             hover:text-text-100
             aria-pressed:text-text-100
             aria-checked:text-text-100
             aria-expanded:text-text-100 h-8 w-8 rounded-md active:scale-95 backdrop-blur-md" type="button" aria-label="Copy to clipboard" data-state="closed"><div class="relative *:transition"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 256 256" class="scale-100"><path d="M200,32H163.74a47.92,47.92,0,0,0-71.48,0H56A16,16,0,0,0,40,48V216a16,16,0,0,0,16,16H200a16,16,0,0,0,16-16V48A16,16,0,0,0,200,32Zm-72,0a32,32,0,0,1,32,32H96A32,32,0,0,1,128,32Zm72,184H56V48H82.75A47.93,47.93,0,0,0,80,64v8a8,8,0,0,0,8,8h80a8,8,0,0,0,8-8V64a47.93,47.93,0,0,0-2.75-16H200Z"></path></svg><div class="absolute inset-0 flex items-center justify-center"><label class="select-none inline-flex gap-3 cursor-pointer text-left"><div class="relative"><input class="sr-only peer" type="checkbox" data-sharkid="__10"/><div class="w-4 h-4 overflow-hidden flex items-center justify-center border rounded transition-colors duration-100 ease-in-out peer-focus-visible:ring-1 ring-offset-2 ring-offset-bg-300 ring-accent-main-100 bg-bg-000 border-border-200 hover:border-border-100 cursor-pointer rounded-full scale-50 opacity-0"></div></div><span class="leading-none sr-only"></span></label></div></div></button></div></div><div class="text-text-500 text-xs p-3.5 pb-0">bash</div><div class=""><pre class="code-block__code !my-0 !rounded-lg !text-sm !leading-relaxed"><code class="language-bash"><span><span>python train_model.py</span></span></code></pre></div></div></pre>

   這會訓練一個辨識英文字母的模型並保存為 `emnist_model.h5`
2. 啟動 Streamlit 應用：

   <pre><div class="relative group/copy rounded-lg"><div class="sticky opacity-0 group-hover/copy:opacity-100 top-2 py-2 h-12 w-0 float-right"><div class="absolute right-0 h-8 px-2 items-center inline-flex"><button class="inline-flex
     items-center
     justify-center
     relative
     shrink-0
     can-focus
     select-none
     disabled:pointer-events-none
     disabled:opacity-50
     disabled:shadow-none
     disabled:drop-shadow-none text-text-300
             border-transparent
             transition
             font-styrene
             duration-300
             ease-[cubic-bezier(0.165,0.85,0.45,1)]
             hover:bg-bg-400
             aria-pressed:bg-bg-400
             aria-checked:bg-bg-400
             aria-expanded:bg-bg-300
             hover:text-text-100
             aria-pressed:text-text-100
             aria-checked:text-text-100
             aria-expanded:text-text-100 h-8 w-8 rounded-md active:scale-95 backdrop-blur-md" type="button" aria-label="Copy to clipboard" data-state="closed"><div class="relative *:transition"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 256 256" class="scale-100"><path d="M200,32H163.74a47.92,47.92,0,0,0-71.48,0H56A16,16,0,0,0,40,48V216a16,16,0,0,0,16,16H200a16,16,0,0,0,16-16V48A16,16,0,0,0,200,32Zm-72,0a32,32,0,0,1,32,32H96A32,32,0,0,1,128,32Zm72,184H56V48H82.75A47.93,47.93,0,0,0,80,64v8a8,8,0,0,0,8,8h80a8,8,0,0,0,8-8V64a47.93,47.93,0,0,0-2.75-16H200Z"></path></svg><div class="absolute inset-0 flex items-center justify-center"><label class="select-none inline-flex gap-3 cursor-pointer text-left"><div class="relative"><input class="sr-only peer" type="checkbox" data-sharkid="__11"/><div class="w-4 h-4 overflow-hidden flex items-center justify-center border rounded transition-colors duration-100 ease-in-out peer-focus-visible:ring-1 ring-offset-2 ring-offset-bg-300 ring-accent-main-100 bg-bg-000 border-border-200 hover:border-border-100 cursor-pointer rounded-full scale-50 opacity-0"></div></div><span class="leading-none sr-only"></span></label></div></div></button></div></div><div class="text-text-500 text-xs p-3.5 pb-0">bash</div><div class=""><pre class="code-block__code !my-0 !rounded-lg !text-sm !leading-relaxed"><code class="language-bash"><span><span>streamlit run app.py</span></span></code></pre></div></div></pre>
3. 在瀏覽器中打開顯示的網址（通常是 [http://localhost:8501](http://localhost:8501) ）
4. 在畫布上繪製一個英文大寫字母，然後點擊「辨識」按鈕

## 技術說明

本專案使用以下技術：

* **TensorFlow/Keras** ：建立和訓練深度學習模型
* **Streamlit** ：建立網頁界面
* **Streamlit Drawable Canvas** ：實現繪圖功能
* **Scikit-image** ：圖像處理
* **NumPy** ：數值運算
* **Matplotlib** ：資料視覺化

## 模型資訊

* 使用簡單的卷積神經網絡（CNN）
* 基於 MNIST 數據集訓練（將數字 0-25 映射為字母 A-Z）
* 輸入為 28x28 像素的灰度圖像

## 常見問題

**問：為什麼辨識結果不準確？**

答：模型是基於 MNIST 數據集訓練的，而不是專門的字母數據集。此外，手寫風格的差異也會影響辨識精度。

**問：如何提高辨識精度？**

答：可以考慮使用專門的 EMNIST 字母數據集重新訓練模型，或者增加更多的數據增強和使用更複雜的模型架構。

**問：可以辨識小寫字母或數字嗎？**

答：目前版本只支援辨識英文大寫字母（A-Z）。如需辨識小寫字母或數字，需要使用更全面的數據集重新訓練模型。

## 未來改進

* 使用 EMNIST 數據集提高辨識精度
* 增加對小寫字母和數字的辨識
* 添加更多的數據增強技術
* 優化模型架構
* 改進使用者界面
