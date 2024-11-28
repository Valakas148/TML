from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import joblib
import cv2
from typing import List


# Завантаження моделі та інших об'єктів
svm_model = joblib.load('svm_model.pkl')
kmeans_model = joblib.load('kmeans_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')  # Якщо використовували

# Ініціалізація детектора ознак (наприклад, SIFT)
sift = cv2.SIFT_create()

app = FastAPI()

def get_descriptors_from_image(image, detector):
    if image is None:
        return None
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return descriptors


def get_image_feature_vector_from_descriptors(descriptors, kmeans_model, num_clusters):
    if descriptors is not None:
        # Прогнозуємо кластери для дескрипторів зображення
        clusters = kmeans_model.predict(descriptors)
        # Створюємо гістограму
        hist, _ = np.histogram(clusters, bins=np.arange(num_clusters + 1))
        # Нормалізуємо гістограму
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        return hist
    else:
        return None



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Зчитуємо зображення з файлу
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return {"error": "Неможливо прочитати зображення"}

    # Отримуємо дескриптори
    descriptors = get_descriptors_from_image(img, sift)

    if descriptors is None:
        return {"error": "Не вдалося отримати дескриптори зображення"}

    # Отримуємо кількість кластерів з моделі KMeans
    num_clusters = kmeans_model.n_clusters

    # Отримуємо гістограму візуальних слів
    hist = get_image_feature_vector_from_descriptors(descriptors, kmeans_model, num_clusters)

    # Нормалізуємо ознаки (якщо використовували StandardScaler)
    hist_scaled = scaler.transform([hist])

    # Виконуємо прогноз
    prediction = svm_model.predict(hist_scaled)
    predicted_class = label_encoder.inverse_transform(prediction)[0]


    # Повертаємо результат
    return {"predicted_class": predicted_class}