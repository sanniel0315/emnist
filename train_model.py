import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

print("正在訓練模型，請稍候...")

# 加載 MNIST 數據集
# 我們將使用 MNIST 數據集，然後映射數字 0-25 作為字母 A-Z
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 只選擇標籤 0-25 的數據，對應於字母 A-Z
mask_train = train_labels < 26
train_images = train_images[mask_train]
train_labels = train_labels[mask_train]

mask_test = test_labels < 26
test_images = test_images[mask_test]
test_labels = test_labels[mask_test]

# 數據預處理
train_images = train_images / 255.0
test_images = test_images / 255.0

# 數據集大小
print(f"訓練數據集大小: {train_images.shape}")
print(f"測試數據集大小: {test_images.shape}")

# 將標籤轉換為 one-hot 編碼
train_labels = to_categorical(train_labels, 26)
test_labels = to_categorical(test_labels, 26)

# 創建簡單的 CNN 模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(26, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 重塑數據以匹配 CNN 輸入格式 (添加通道維度)
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# 訓練模型
history = model.fit(train_images, train_labels, epochs=5,
                    validation_data=(test_images, test_labels))

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
print("Training history saved as 'training_history.png'")

# Show some prediction results
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
print("Predictions saved as 'predictions.png'")
print("Training completed!")