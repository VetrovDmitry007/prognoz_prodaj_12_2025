import numpy as np
import lightgbm as lgb
import pandas as pd
from tqdm import tqdm

from sales_dataset import SalesDataset
from utils import pred_proc, evaluate_model
from window_features import FeatureExtractor


def init_dataset():
    # Создание DataSet
    df_conv = pred_proc('etl/final_report_2026-01-27.csv')
    sale_ds = SalesDataset(df_conv)
    return sale_ds


def _get_data_to_predict(sale_ds, mdlp_id, debug: bool = False) -> np.ndarray:
    """ Получение X для прогноза
        X -- продажи последних 12 месяцев

    :param sale_ds: SalesDataset
    :param mdlp_id: ID аптеки
    :return: продажи последних 12 месяцев
    """
    # 1. Получение окна продаж
    res = sale_ds.get_last_window(mdlp_id)
    # 2. Получение фичей
    ft = FeatureExtractor()
    features = ft.compute_window(res.window_sale, num_month=res.month_predict)
    # 2. Прогноз продаж
    X = np.array(features.to_list(), dtype=float)
    X = X.reshape(1,-1)

    if debug:
        print(f'{res.month_predict=}, {res.window_sale=}\n')
        print(features)

    return X


def init_model(model_file):
    model = lgb.Booster(model_file=model_file)
    return model


def predict(sale_ds, model, mdlp_id):
    """ Прогноз продаж за last_mes + 1 месяц

    :param sale_ds: SalesDataset
    :param model: модель lightgbm
    :param mdlp_id: ID аптеки
    :return: Прогноз продаж за last_mes + 1 месяц
    """
    X = _get_data_to_predict(sale_ds, mdlp_id)
    y_pred = model.predict(X)
    result = max(0, round(y_pred[0]))
    return result


def export_result_predict(file_result: str):
    """ Прогноз продаж для last_mes + 1 по всем mdlp
        с выгрузкой в csv

    """
    sale_ds = init_dataset()
    model = init_model('models/lgb_model_regress_2.txt')

    ls_result = []
    for mdlp_id in tqdm(sale_ds.ls_mdlp_id):
        sale = predict(sale_ds, model, mdlp_id)
        ls_result.append({'mdlp_id': mdlp_id, 'sale': sale})
    df = pd.DataFrame(ls_result)
    df.to_csv(file_result, index=False)


def eval(file_sale: str, file_predict: str):
    """  Оценка модели регрессии на основе файла продаж и файла прогнозов.

    """
    # Продажи(факт)
    df_sale_true = pd.read_csv(file_sale)
    df_sale_true = df_sale_true[(df_sale_true['exit_type'] == 'Продажа') & (
        df_sale_true['Territory'].str.contains('moscow', case=False, na=False))]

    # Продажи(прогноз)
    df_sale_predict = pd.read_csv(file_predict)

    # Продажи + Прогноз
    df_merge = pd.merge(
        df_sale_true,
        df_sale_predict,
        left_on='location_mdlp_id',
        right_on='mdlp_id',
        how='inner'  # inner join (только совпадающие записи)
    )
    de_merge = df_merge[['mdlp_id', '2026-01-01', 'sale']]
    de_merge = de_merge.rename(columns={'2026-01-01': 'fact', 'sale': 'predict'})
    de_merge = de_merge.fillna(0)

    # Оценка прогноза
    fact = de_merge['fact'].to_list()
    predict = de_merge['predict'].to_list()
    evaluate_model(fact, predict, 'LightGBM')


if __name__ == '__main__':
    export_result_predict(file_result='data/result_predict.csv')
    eval(file_sale='etl/sale_01_2026.csv', file_predict='data/result_predict.csv')
