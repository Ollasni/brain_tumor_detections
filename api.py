from fastapi import FastAPI, File, UploadFile
import pickle
import cv2
import numpy as np
from fastapi.responses import JSONResponse
import os
from io import BytesIO
from PIL import Image

# Инициализация FastAPI
app = FastAPI()

# Загрузка обученной модели
with open('saved_model.pkl', 'rb') as file:
    lg = pickle.load(file)

# Словарь с расшифровкой меток
dec = {0: "No Tumor", 1: "Tumor"}

# Функция для обработки изображения и предсказания
def predict_tumor(image):
    # Преобразуем изображение в градации серого
    img = np.array(image.convert("L"))  # Чтение изображения в grayscale

    # Изменение размера до 200x200
    img_resized = cv2.resize(img, (200, 200))
    
    # Преобразуем в одномерный массив и нормализуем
    img_resized = img_resized.reshape(1, -1) / 255.0
    
    # Предсказание модели
    prediction = lg.predict(img_resized)
    
    return dec[prediction[0]]

# Маршрут для предсказания по изображению
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Чтение файла изображения
        image = Image.open(BytesIO(await file.read()))
        
        # Получение предсказания
        prediction = predict_tumor(image)
        
        # Возвращаем результат предсказания в формате JSON
        return JSONResponse(content={"prediction": prediction})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

