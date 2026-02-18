"""
Модуль утилит общего назначения
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


@dataclass(slots=True)
class SaleDataSet:
    """ Класс DataSet продаж аптек

    """
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_data: lgb.Dataset
    valid_data: lgb.Dataset

def df_date_convert(df: pd.DataFrame):
    """ Конвертор столбцов 01_2023 --> 2023-01-01 """

    def upd(val: str):
        if not '_' in val:
            return val
        elif val.split('_')[0].isdigit():
            new_dt = pd.to_datetime(val, format='%m_%Y').date().strftime('%Y-%m-%d')
            return new_dt
        else:
            return val

    new_lolumns = [upd(column) for column in df.columns]
    df.columns = new_lolumns
    return df


def create_dataset(X, y):
    """ Создание dataset для модели градиентного бустинга задачи регрессии """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    return SaleDataSet(X_train=X_train,
                       X_test=X_test,
                       y_train=y_train,
                       y_test=y_test,
                       train_data=train_data,
                       valid_data=valid_data)


def evaluate_model(y_true, y_pred, model_name):
    """Функция для оценки модели регрессии"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} Результаты:")
    print(f"Среднеквадратичная ошибка (MSE): {mse:.4f}")
    print(f"Корень из MSE (RMSE): {rmse:.4f}")
    print(f"Средняя абсолютная ошибка (MAE): {mae:.4f}")
    print(f"Средняя абсолютная процентная ошибка (MAPE): {mape:.2f}%")
    print(f"Коэффициент детерминации (R²): {r2:.4f}")

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}