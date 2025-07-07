import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#  Загрузка данных
df = pd.read_csv("SAP-4000.csv", names=[
    "Пол", "Часы_учёбы_в_неделю", "Репетиторство", "Регион",
    "Посещаемость", "Образование_родителей", "Балл_за_экзамен"
])

#  Преобразование численных признаков в float
for col in ["Часы_учёбы_в_неделю", "Посещаемость", "Балл_за_экзамен"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Удаление строк с пропусками
df.dropna(inplace=True)

#  Определяем типы признаков
X = df.drop(columns=["Балл_за_экзамен"])
y = df["Балл_за_экзамен"]

num_features = ["Часы_учёбы_в_неделю", "Посещаемость"]
cat_features = ["Пол", "Репетиторство", "Регион", "Образование_родителей"]

#  Создаем препроцессор для преобразования данных
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

#  Создаем пайплайн с моделью
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Обучение модели
model.fit(X_train, y_train)

#  Предсказание и оценка
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
print(f"Коэффициент детерминации R²: {r2:.2f}")

#  Пример использования модели
example = pd.DataFrame([{
    "Пол": "Female",
    "Часы_учёбы_в_неделю": 10,
    "Репетиторство": "Yes",
    "Регион": "Urban",
    "Посещаемость": 85,
    "Образование_родителей": "Tertiary"
}])
predicted_score = model.predict(example)[0]
print(f"\nПример предсказания: прогнозируемый балл за экзамен — {predicted_score:.1f}")