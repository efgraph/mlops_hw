import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Включаем логирование метрик (сохранение происходит в s3 через сервер MLFlow)
mlflow.sklearn.autolog()

# Создаем новый эксперимент, если еще не создан
experiment_name = "denis_spiridonov"
mlflow.set_experiment(experiment_name)

# Загружаем и разделяем данные такие же как в ДЗ_1
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Масштабируем признаки
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Запускаем parent run, имя ник в тг
with mlflow.start_run(run_name="johnhse", description="parent") as parent_run:

    # Берем модели как в ДЗ_1
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(),
        "DecisionTreeRegressor": DecisionTreeRegressor()
    }

    # Для каждой модели запускаем child run
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name, nested=True):

            # Обучаем модель
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Преобразуем данные в DataFrame для оценки
            eval_df = pd.DataFrame(X_test_scaled, columns=data.feature_names)
            eval_df["MedHouseVal"] = y_test
            eval_df["prediction"] = y_pred

            # Оцениваем модель и логируем метрики
            mlflow.evaluate(
                data=eval_df,
                targets="MedHouseVal",
                predictions="prediction",
                model_type="regressor",
                evaluators=["default"]
            )

    print(f"Experiment {experiment_name} finished")
