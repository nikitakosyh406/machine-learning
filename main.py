from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Загрузка модели при запуске приложения
try:
    model = joblib.load("model.pkl")
except Exception as e:
    model = None
    raise RuntimeError(f"Не удалось загрузить модель: {e}")

# Тестовый эндпоинт
@app.get("/")
def read_root():
    return {"message": "ML API работает!"}

# Эндпоинт для предсказания
@app.post("/predict/")
def predict(data: dict):
    if model is None:
        return {"error": "Модель не загружена"}

    try:
        # Преобразуем входные данные в DataFrame
        input_data = pd.DataFrame([data])

        # Делаем предсказание
        prediction = model.predict(input_data)

        # Возвращаем результат
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}