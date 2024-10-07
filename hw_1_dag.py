import json
import io
from airflow.models import DAG, Variable
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from sklearn.datasets import fetch_california_housing
import pandas as pd
from datetime import datetime, timedelta
import pickle

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


def create_dag(dag_id: str, m_name: str):
    s3_hook = S3Hook('s3_connection')

    ####### DAG STEPS #######
    def init(**kwargs):
        ti = kwargs['ti']
        model_name = kwargs['model_name']
        timestamp = datetime.now().isoformat()

        metrics = {
            'timestamp': timestamp,
            'model_name': model_name
        }

        # Передаем метрики через xcom
        ti.xcom_push(key='metrics', value=metrics)
        print(f'Pipeline for model {model_name} initialized at {timestamp}')

    def get_data(s3_hook, **kwargs):
        ti = kwargs['ti']
        start_time = datetime.now().isoformat()

        # Загружаем датасет в память
        data = fetch_california_housing(as_frame=True)
        df = pd.concat([data['data'], pd.DataFrame(data['target'], columns=['MedHouseVal'])], axis=1)

        filebuffer = io.BytesIO()
        df.to_pickle(filebuffer)
        filebuffer.seek(0)

        # Загружаем датасет в S3
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            bucket_name=BUCKET,
            key=f'DenisSpiridonov/{m_name}/datasets/data.pkl',
            replace=True
        )

        end_time = datetime.now().isoformat()
        dataset_size = df.shape

        metrics = {
            'start_time': start_time,
            'end_time': end_time,
            'dataset_size': dataset_size
        }

        ti.xcom_push(key='metrics', value=metrics)

    def prepare_data(s3_hook, **kwargs):
        ti = kwargs['ti']
        start_time = datetime.now().isoformat()

        # Загрузка данных из S3
        file = s3_hook.download_file(bucket_name=BUCKET, key=f'DenisSpiridonov/{m_name}/datasets/data.pkl')
        df = pd.read_pickle(file)

        # Разделяем на фичи и таргет, train и test
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Масштабирование данных
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Сохраняем обработанные данные обратно в S3
        for name, data in zip(
                ['X_train', 'X_test', 'y_train', 'y_test'],
                [X_train_scaled, X_test_scaled, y_train, y_test]
        ):
            filebuffer = io.BytesIO()
            pickle.dump(data, filebuffer)
            filebuffer.seek(0)
            s3_hook.load_file_obj(
                file_obj=filebuffer,
                bucket_name=BUCKET,
                key=f'DenisSpiridonov/{m_name}/datasets/{name}.pkl',
                replace=True
            )

        end_time = datetime.now().isoformat()

        metrics = {
            'start_time': start_time,
            'end_time': end_time,
            'features': df.columns.tolist()
        }

        ti.xcom_push(key='metrics', value=metrics)

    def train_model(s3_hook, **kwargs):
        ti = kwargs['ti']
        start_time = datetime.now().isoformat()

        data = {}
        for name in ['X_train', 'X_test', 'y_train', 'y_test']:
            file = s3_hook.download_file(
                key=f'DenisSpiridonov/{m_name}/datasets/{name}.pkl',
                bucket_name=BUCKET
            )
            data[name] = pickle.load(open(file, 'rb'))

        # Обучение модели
        model = models[m_name]
        model.fit(data['X_train'], data['y_train'])
        predictions = model.predict(data['X_test'])

        # Метрики после обучения
        model_metrics = {
            'r2_score': r2_score(data['y_test'], predictions),
            'rmse': mean_squared_error(data['y_test'], predictions, squared=False),
            'mae': median_absolute_error(data['y_test'], predictions)
        }

        end_time = datetime.now().isoformat()

        metrics = {
            'start_time': start_time,
            'end_time': end_time,
            'model_metrics': model_metrics
        }

        ti.xcom_push(key='metrics', value=metrics)

    def save_results(s3_hook, **kwargs):
        ti = kwargs['ti']

        # Собираем и объединяем метрики со всех шагов
        init_metrics = ti.xcom_pull(key='metrics', task_ids='init')
        get_data_metrics = ti.xcom_pull(key='metrics', task_ids='get_data')
        prepare_data_metrics = ti.xcom_pull(key='metrics', task_ids='prepare_data')
        train_model_metrics = ti.xcom_pull(key='metrics', task_ids='train_model')

        metrics = {
            'init': init_metrics,
            'get_data': get_data_metrics,
            'prepare_data': prepare_data_metrics,
            'train_model': train_model_metrics
        }

        filebuffer = io.BytesIO()

        # Сохраняем метрики как json файл
        json_data = json.dumps(metrics).encode()
        filebuffer.write(json_data)
        filebuffer.seek(0)

        s3_hook.load_file_obj(
            file_obj=filebuffer,
            bucket_name=BUCKET,
            key=f'DenisSpiridonov/{m_name}/results/metrics.json',
            replace=True
        )
        print('Metrics saved to S3.')

    ####### INIT DAG #######
    dag = DAG(
        dag_id=dag_id,
        default_args=DEFAULT_ARGS,
        schedule_interval='0 1 * * *',  # Запуск каждый день в 1:00
        start_date=days_ago(1),
        catchup=False,
        tags=['mlops']
    )

    with dag:
        # Шаги пайплайна
        task_init = PythonOperator(task_id='init', python_callable=init, provide_context=True,
                                   op_kwargs={'model_name': m_name}, dag=dag)
        task_get_data = PythonOperator(task_id='get_data', python_callable=get_data, provide_context=True,
                                       op_kwargs={'s3_hook': s3_hook}, dag=dag)
        task_prepare_data = PythonOperator(task_id='prepare_data', python_callable=prepare_data, provide_context=True,
                                           op_kwargs={'s3_hook': s3_hook}, dag=dag)
        task_train_model = PythonOperator(task_id='train_model', python_callable=train_model, provide_context=True,
                                          op_kwargs={'s3_hook': s3_hook}, dag=dag)
        task_save_results = PythonOperator(task_id='save_results', python_callable=save_results, provide_context=True,
                                           op_kwargs={'s3_hook': s3_hook}, dag=dag)

        task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results

    return dag


for model_name in models.keys():
    dag_id = f'denis_spiridonov_{model_name}'
    globals()[dag_id] = create_dag(dag_id, model_name)
