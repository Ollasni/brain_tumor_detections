import pickle
import cv2
import matplotlib.pyplot as plt
import os

# Загрузка модели из файла
with open('saved_model.pkl', 'rb') as file:
    sv = pickle.load(file)

# Загружаем изображение
image_path = 'image.jpg'
img = cv2.imread(image_path, 0)  # Чтение изображения в режиме grayscale

# Предобработка изображения (например, изменение размера и нормализация)
img_resized = cv2.resize(img, (200, 200))  # Изменение размера до 200x200
img_resized = img_resized.reshape(1, -1)   # Изменение формы для подачи в модель
img_resized = img_resized / 255.0          # Нормализация данных

# Предсказание модели
prediction = sv.predict(img_resized)

# Отображаем результат
dec = {0: "No Tumor", 1: "Tumor"}  # Предположим, что это ваша расшифровка меток
plt.figure(figsize=(6, 6))
plt.title(f"Prediction: {dec[prediction[0]]}")
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()
