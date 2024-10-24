import os
import io
import json
import pickle
from datetime import datetime, timedelta

import pandas as pd
import mlflow
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

from airflow.providers.amazon.aws.hooks.s3 import S3Hook

BUCKET = Variable.get('S3_BUCKET')

DEFAULT_ARGS = {
    'owner': 'Denis Spiridonov',
    'email': 'example@gmail.com',
    'retry': 3,
    'retry_delay': timedelta(minutes=1),
}

model_names = ['random_forest', 'linear_regression', 'decision_tree']
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ])
)


def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)

    # Настраиваем адрес для отслеживания экспериментов в mlflow
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


def init(**kwargs):
    ti = kwargs['ti']
    timestamp = datetime.now().isoformat()
    configure_mlflow()
    experiment_name = "denis_spiridonov"
    mlflow.set_experiment(experiment_name)

    # Стартуем родительский run
    with mlflow.start_run(run_name="johnhse", description="parent") as run:
        experiment_id = run.info.experiment_id
        run_id = run.info.run_id

    # Сохраняем метрики
    metrics = {
        'timestamp': timestamp,
        'experiment_id': experiment_id,
        'run_id': run_id
    }

    # и передаем через XCom
    ti.xcom_push(key='metrics', value=metrics)
    print(f"Experiment {experiment_name} initialized at {timestamp}")


def get_data(**kwargs):
    ti = kwargs['ti']
    start_time = datetime.now().isoformat()

    # Загружаем данные для обучения
    data = fetch_california_housing(as_frame=True)
    df = pd.concat([data['data'], pd.DataFrame(data['target'], columns=['MedHouseVal'])], axis=1)

    s3_hook = S3Hook('s3_connection')
    filebuffer = io.BytesIO()
    df.to_pickle(filebuffer)
    filebuffer.seek(0)

    # Загружаем в s3
    s3_hook.load_file_obj(
        file_obj=filebuffer,
        bucket_name=BUCKET,
        key='DenisSpiridonov/datasets/data.pkl',
        replace=True
    )

    end_time = datetime.now().isoformat()
    dataset_size = df.shape

    # Сохраняем метрики в xcom
    metrics = {
        'start_time': start_time,
        'end_time': end_time,
        'dataset_size': dataset_size
    }

    ti.xcom_push(key='metrics', value=metrics)


def prepare_data(**kwargs):
    ti = kwargs['ti']
    start_time = datetime.now().isoformat()

    # Скачиваем из s3
    s3_hook = S3Hook('s3_connection')
    file_obj = s3_hook.get_key(
        key='DenisSpiridonov/datasets/data.pkl',
        bucket_name=BUCKET
    )
    filebuffer = io.BytesIO()
    file_obj.download_fileobj(filebuffer)
    filebuffer.seek(0)
    df = pd.read_pickle(filebuffer)

    # Разделяем признаки и целевую переменную
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Масштабируем признаки
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Сохраняем в s3
    for name, data_part in zip(
            ['X_train', 'X_test', 'y_train', 'y_test'],
            [X_train_scaled, X_test_scaled, y_train, y_test]
    ):
        filebuffer = io.BytesIO()
        pickle.dump(data_part, filebuffer)
        filebuffer.seek(0)
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            bucket_name=BUCKET,
            key=f'DenisSpiridonov/datasets/{name}.pkl',
            replace=True
        )

    end_time = datetime.now().isoformat()

    metrics = {
        'start_time': start_time,
        'end_time': end_time,
        'features': df.columns.tolist()
    }

    ti.xcom_push(key='metrics', value=metrics)


def train_model(m_name: str, **kwargs):
    ti = kwargs['ti']
    start_time = datetime.now().isoformat()

    s3_hook = S3Hook('s3_connection')
    data = {}
    for name in ['X_train', 'X_test', 'y_train', 'y_test']:
        file_obj = s3_hook.get_key(
            key=f'DenisSpiridonov/datasets/{name}.pkl',
            bucket_name=BUCKET
        )
        filebuffer = io.BytesIO()
        file_obj.download_fileobj(filebuffer)
        filebuffer.seek(0)
        data[name] = pickle.load(filebuffer)

    # Данные с предыдущих шагов
    init_metrics = ti.xcom_pull(task_ids='init', key='metrics')
    parent_run_id = init_metrics['run_id']
    experiment_id = init_metrics['experiment_id']
    prepare_data_metrics = ti.xcom_pull(task_ids='prepare_data', key='metrics')
    feature_names = prepare_data_metrics['features'][:-1]

    configure_mlflow()

    # Начинаем вложенный run для каждой модели
    with mlflow.start_run(
        run_name=m_name,
        experiment_id=experiment_id,
        nested=True,
        parent_run_id=parent_run_id,
        tags={'mlflow.parentRunId': parent_run_id}
    ) as run:
        mlflow.sklearn.autolog()

        model = models[m_name]
        model.fit(data['X_train'], data['y_train'])
        predictions = model.predict(data['X_test'])

        eval_df = pd.DataFrame(data['X_test'], columns=feature_names)
        eval_df["MedHouseVal"] = data['y_test'].reset_index(drop=True)
        eval_df["prediction"] = predictions

        # Оцениваем модель и логируем метрики
        mlflow.evaluate(
            data=eval_df,
            targets="MedHouseVal",
            predictions="prediction",
            model_type="regressor",
            evaluators=["default"]
        )

    end_time = datetime.now().isoformat()

    metrics = {
        'start_time': start_time,
        'end_time': end_time,
        'model_name': m_name,
        'run_id': run.info.run_id,
        'experiment_id': experiment_id
    }

    ti.xcom_push(key='metrics', value=metrics)


def save_results(**kwargs):
    ti = kwargs['ti']

    # Сбор метрик с предыдущих шагов
    init_metrics = ti.xcom_pull(task_ids='init', key='metrics')
    get_data_metrics = ti.xcom_pull(task_ids='get_data', key='metrics')
    prepare_data_metrics = ti.xcom_pull(task_ids='prepare_data', key='metrics')
    train_model_metrics_list = []
    for m_name in model_names:
        metrics = ti.xcom_pull(task_ids=f'train_model_{m_name}', key='metrics')
        train_model_metrics_list.append({m_name: metrics})

    # Объединяем метрики в один словарь
    metrics = {
        'init': init_metrics,
        'get_data': get_data_metrics,
        'prepare_data': prepare_data_metrics,
        'train_model': train_model_metrics_list
    }

    filebuffer = io.BytesIO()
    json_data = json.dumps(metrics).encode()
    filebuffer.write(json_data)
    filebuffer.seek(0)

    s3_hook = S3Hook('s3_connection')
    s3_hook.load_file_obj(
        file_obj=filebuffer,
        bucket_name=BUCKET,
        key='DenisSpiridonov/results/metrics.json',
        replace=True
    )
    print('Metrics saved to S3.')


with DAG(
        dag_id='denis_spiridonov_ml_pipeline',
        default_args=DEFAULT_ARGS,
        schedule_interval='0 1 * * *',
        start_date=days_ago(1),
        catchup=False,
        tags=['mlops']
) as dag:
    # Определяем задачи DAG
    task_init = PythonOperator(
        task_id='init',
        python_callable=init,
    )

    task_get_data = PythonOperator(
        task_id='get_data',
        python_callable=get_data,
    )

    task_prepare_data = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_data,
    )

    # Создаем задачи для каждой модели
    training_model_tasks = []
    for m_name in model_names:
        train_task = PythonOperator(
            task_id=f'train_model_{m_name}',
            python_callable=train_model,
            op_kwargs={'m_name': m_name},
        )
        training_model_tasks.append(train_task)

    task_save_results = PythonOperator(
        task_id='save_results',
        python_callable=save_results,
    )

    task_init >> task_get_data >> task_prepare_data >> training_model_tasks >> task_save_results
