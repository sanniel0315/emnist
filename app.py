import streamlit as st
from streamlit_drawable_canvas import st_canvas
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import resize
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.font_manager as fm
import matplotlib as mpl

# 設置支援中文的字體
mpl.rcParams['font.family'] = 'Noto Sans CJK TC'  # 使用 Noto Sans CJK TC 字體
# 設置頁面
st.set_page_config(page_title="手寫英文字母辨識", layout="wide")

# 設置標題
st.title('手寫英文字母辨識系統')
st.write('請在左側畫布上繪製一個英文大寫字母 (A-Z)，然後點擊辨識按鈕')

# 定義英文字母類別
class_names = [chr(ord('A')+i) for i in range(26)]

# 嘗試直接定義與訓練好的模型相同的結構
def create_model():
    # 這裡需要確保模型結構與訓練時完全一致
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(26, activation='softmax')
    ])
    
    # 編譯模型
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 檢查模型檔案是否存在
model_path = 'emnist_model.h5'
model_path_alternate = '../emnist_model.h5'  # 檢查上層目錄

# 載入或創建模型
@st.cache_resource
def load_or_create_model():
    try:
        # 嘗試載入現有模型
        if os.path.exists(model_path):
            st.info("嘗試載入模型...")
            try:
                return tf.keras.models.load_model(model_path)
            except Exception as e:
                st.warning(f"載入模型時發生錯誤: {e}，將使用新建模型")
        elif os.path.exists(model_path_alternate):
            st.info("嘗試從上層目錄載入模型...")
            try:
                return tf.keras.models.load_model(model_path_alternate)
            except Exception as e:
                st.warning(f"載入模型時發生錯誤: {e}，將使用新建模型")
        
        # 如果都失敗，創建新模型
        st.warning("找不到可用的模型檔案，將創建新模型")
        return create_model()
        
    except Exception as e:
        st.error(f"處理模型時發生錯誤: {e}")
        return create_model()

# 載入或創建模型
with st.spinner('準備模型中...'):
    model = load_or_create_model()
    st.success('模型準備完成！')

# 創建兩列布局
col1, col2 = st.columns(2)

# 第一列：繪圖畫布
with col1:
    st.subheader("繪製區域")
    st.write("在此區域繪製一個英文大寫字母")
    
    # 創建畫布
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",  # 填充顏色為透明
        stroke_width=15,  # 筆觸寬度
        stroke_color="#000000",  # 筆觸顏色為黑色
        background_color="#FFFFFF",  # 背景顏色為白色
        width=280,  # 畫布寬度
        height=280,  # 畫布高度
        drawing_mode="freedraw",  # 繪圖模式
        key="canvas",
    )
    
    # 清除按鈕
    if st.button('清除畫布'):
        st.experimental_rerun()

# 第二列：辨識結果
with col2:
    st.subheader("辨識結果")
    
    # 辨識按鈕
    if st.button('辨識'):
        if canvas_result.image_data is None:
            st.warning('請先在畫布上繪製一個字母')
        else:
            # 處理圖像
            with st.spinner('處理圖像中...'):
                # 轉換為灰度圖像
                image = rgb2gray(rgba2rgb(canvas_result.image_data))
                # 調整大小為 28x28 像素
                image_resized = resize(image, (28, 28), anti_aliasing=True)
                # 反轉顏色（因為 MNIST 數據集是黑底白字）
                image_processed = np.abs(1-image_resized)
                # 重塑為模型輸入格式
                X = image_processed.reshape(1, 28, 28, 1)
            
            # 預測
            with st.spinner('辨識中...'):
                predictions = model.predict(X)
                predicted_class = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class] * 100
            
            # 顯示結果
            st.markdown(f"### 辨識結果: {class_names[predicted_class]}")
            st.write(f"置信度: {confidence:.2f}%")
            
            # 顯示處理後的圖像
            st.subheader("處理後的圖像")
            st.image(image_processed, width=150, caption="調整大小至 28x28 像素")
            
            # 顯示所有字母的預測機率（條形圖）
            st.subheader("各字母的預測機率")
            
            # 創建數據框
            df = pd.DataFrame({
                '字母': class_names,
                '機率 (%)': predictions[0] * 100
            })
            
            # 按機率降序排序
            df = df.sort_values('機率 (%)', ascending=False).reset_index(drop=True)
            
            # 只顯示前 5 個最可能的字母
            chart = plt.figure(figsize=(10, 4))
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            plt.bar(df['字母'][:5], df['機率 (%)'][:5])
            plt.xlabel('Letter')
            plt.ylabel('Probability (%)')
            plt.title('Prediction Probabilities')
            plt.ylim(0, 100)
            
            # 顯示圖表
            st.pyplot(chart)

# 顯示使用說明
with st.expander("使用說明"):
    st.markdown("""
    ### 使用步驟:
    1. 在左側畫布上用滑鼠繪製一個英文大寫字母 (A-Z)
    2. 點擊「辨識」按鈕
    3. 查看右側的辨識結果
    4. 使用「清除畫布」按鈕重新開始
    
    ### 注意事項:
    - 請盡量清晰地繪製標準的英文大寫字母
    - 字母應填滿畫布的大部分區域以獲得更好的辨識效果
    - 系統是基於 MNIST 數據集訓練的，對某些手寫風格的辨識可能不夠準確
    """)

# 顯示關於頁面
with st.expander("關於本應用"):
    st.markdown("""
    ### 手寫英文字母辨識系統
    
    本應用使用深度學習技術辨識手寫英文大寫字母 (A-Z)。
    
    #### 技術細節:
    - 使用 TensorFlow/Keras 建立卷積神經網絡 (CNN) 模型
    - 基於 MNIST 數據集訓練
    - 使用 Streamlit 創建網頁界面
    - 使用 Streamlit Drawable Canvas 實現繪圖功能
    
    #### 如何改進:
    - 使用專門的 EMNIST 字母數據集可提高辨識精度
    - 增加數據增強和更複雜的模型架構可提升辨識能力
    - 加入對小寫字母和數字的辨識
    """)