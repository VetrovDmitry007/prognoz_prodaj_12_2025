"""
Модуль утилит общего назначения
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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


def create_dataset(split_dataset):
    """ Создание dataset для модели градиентного бустинга задачи регрессии """
    train_data = lgb.Dataset(split_dataset.X_train, label=split_dataset.y_train)
    valid_data = lgb.Dataset(split_dataset.X_test, label=split_dataset.y_test, reference=train_data)
    print("Create dataset")
    return train_data, valid_data


def smape(y_true, y_pred, eps=1e-8):
    """ Symmetric MAPE -- устойчива даже если факт и прогноз 0

    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps))


def evaluate_model(y_true, y_pred, model_name):
    """Функция для оценки модели регрессии"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = smape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} Результаты:")
    print(f"Среднеквадратичная ошибка (MSE): {mse:.4f}")
    print(f"Корень из MSE (RMSE): {rmse:.4f}")
    print(f"Средняя абсолютная ошибка (MAE): {mae:.4f}")
    print(f"Средняя абсолютная процентная ошибка (MAPE): {mape:.2f}%")
    print(f"Коэффициент детерминации (R²): {r2:.4f}")

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def pred_proc(file_name: str):
    """ Предварительная обработка DataFrame
        1. Территория = Москва
        2. Тип учёта = "Продажа"
        3. (не использовать) В последнем учтённом месяце были продажи
        4. Конвертирование дат "01_2023" -> "2023-01-01"

    :return: DataFrame
    """
    df_csv = pd.read_csv(file_name)
    df_csv = df_csv[
        (df_csv['exit_type'] == 'Продажа') & (df_csv['Territory'].str.contains('moscow', case=False, na=False))]
    # df_csv = df_csv[df_csv['12_2025'].notna()]

    df_conv = df_date_convert(df_csv)
    return df_conv

