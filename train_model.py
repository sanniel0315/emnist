import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import to_categorical

print("正在訓練模型，請稍候...")

# 載入 EMNIST 數據集 (letters)
# EMNIST letters 包含 26 個大寫英文字母 (A-Z)
def load_emnist():
    import tensorflow_datasets as tfds
    
    print("正在下載並準備 EMNIST 數據集...")
    
    # 載入 EMNIST letters 數據集
    ds_train, ds_test = tfds.load(
        'emnist/letters',
        split=['train', 'test'],
        as_supervised=True,
        with_info=False
    )
    
    def preprocess(image, label):
        # EMNIST 數據集中的標籤從 1 開始 (1-26 對應 A-Z)
        # 將標籤調整為從 0 開始
        label = tf.cast(label - 1, tf.int32)
        
        # 將圖像轉換為 float32 並標準化
        image = tf.cast(image, tf.float32) / 255.0
        
        # EMNIST 圖像需要轉置，因為它們是以不同的方向儲存的
        image = tf.transpose(image, perm=[1, 0, 2])
        
        return image, label
    
    # 應用預處理
    ds_train = ds_train.map(preprocess)
    ds_test = ds_test.map(preprocess)
    
    # 將數據集轉換為 NumPy 數組
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    for image, label in ds_train:
        train_images.append(image.numpy())
        train_labels.append(label.numpy())
    
    for image, label in ds_test:
        test_images.append(image.numpy())
        test_labels.append(label.numpy())
    
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    print(f"訓練數據集大小: {train_images.shape}")
    print(f"測試數據集大小: {test_images.shape}")
    
    return train_images, train_labels, test_images, test_labels

try:
    # 嘗試載入 EMNIST 數據集
    train_images, train_labels, test_images, test_labels = load_emnist()
except Exception as e:
    print(f"載入 EMNIST 數據集時發生錯誤: {e}")
    print("正在嘗試使用替代方法...")
    
    # 替代方法：使用手動下載的 EMNIST 數據集
    # 您需要事先下載 EMNIST 數據集並按照以下格式保存
    try:
        data = np.load('emnist_letters.npz')
        train_images = data['train_images']
        train_labels = data['train_labels']
        test_images = data['test_images']
        test_labels = data['test_labels']
        print("成功從本地文件載入 EMNIST 數據集")
    except Exception as e2:
        print(f"從本地文件載入 EMNIST 數據集時發生錯誤: {e2}")
        print("將使用 MNIST 數據集替代（辨識準確度將受限）")
        
        # 退回到使用 MNIST 數據集
        from tensorflow.keras.datasets import mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        
        # 只選擇標籤 0-25 的數據，對應於字母 A-Z
        mask_train = train_labels < 26
        train_images = train_images[mask_train]
        train_labels = train_labels[mask_train]
        
        mask_test = test_labels < 26
        test_images = test_images[mask_test]
        test_labels = test_labels[mask_test]
        
        print("使用 MNIST 數據集替代（映射數字 0-25 到字母 A-Z）")

# 數據預處理
train_images = train_images / 255.0 if np.max(train_images) > 1.0 else train_images
test_images = test_images / 255.0 if np.max(test_images) > 1.0 else test_images

# 數據集大小
print(f"最終訓練數據集大小: {train_images.shape}")
print(f"最終測試數據集大小: {test_images.shape}")

# 將標籤轉換為 one-hot 編碼
train_labels = to_categorical(train_labels, 26)
test_labels = to_categorical(test_labels, 26)

# 創建改進的 CNN 模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(26, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 重塑數據以匹配 CNN 輸入格式 (添加通道維度)
if len(train_images.shape) == 3:
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
if len(test_images.shape) == 3:
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

print("模型摘要:")
model.summary()

# 訓練模型
history = model.fit(
    train_images, train_labels, 
    epochs=10, 
    validation_data=(test_images, test_labels),
    batch_size=128
)

# 評估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'測試準確率: {test_acc:.4f}')

# 保存模型
model.save('emnist_model.h5')
print("模型已保存為 'emnist_model.h5'")

# 繪製訓練過程
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig('training_history.png')
print("訓練歷史已保存為 'training_history.png'")

# 顯示一些預測結果
predictions = model.predict(test_images[:10])
class_names = [chr(ord('A')+i) for i in range(26)]

plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    predicted_label = np.argmax(predictions[i])
    true_label = np.argmax(test_labels[i])
    plt.title(f"Predicted: {class_names[predicted_label]}\nActual: {class_names[true_label]}")
    plt.axis('off')

plt.savefig('predictions.png')
print("預測結果已保存為 'predictions.png'")
print("訓練完成！")