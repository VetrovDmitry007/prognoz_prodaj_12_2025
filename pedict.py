import numpy as np
import lightgbm as lgb
import pandas as pd
from tqdm import tqdm

from sales_dataset import SalesDataset
from utils import pred_proc
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
    result = round(y_pred[0])
    return result


def export_result_predict():
    """ Прогноз продаж для last_mes + 1 по всем mdlp
        с выгрузкой в csv

    """
    sale_ds = init_dataset()
    model = init_model('lgb_model_regress_2.txt')

    ls_result = []
    for mdlp_id in tqdm(sale_ds.ls_mdlp_id):
        sale = predict(sale_ds, model, mdlp_id)
        ls_result.append({'mdlp_id': mdlp_id, 'sale': sale})
    df = pd.DataFrame(ls_result)
    df.to_csv('result_predict.csv', index=False)


if __name__ == '__main__':
    export_result_predict()




