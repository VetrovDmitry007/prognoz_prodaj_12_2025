import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from window_features import FeatureExtractor


class PredictSale:
    def __init__(self, config: Dict) -> None:
        model_file = config['model_file']
        file_source = config['file_source']
        self.date_source_beg = config['date_source_beg']
        self.date_source_end = config['date_source_end']

        # номер месяца для предсказания
        self.num_month = self._get_num_month(date_end=self.date_source_end)

        self.model = lgb.Booster(model_file=model_file)
        self.ft = FeatureExtractor()
        df_xls = pd.read_excel(file_source)
        self.ls_rec = self._prepare_data(df_xls)

    def _prepare_data(self, df_xls: pd.DataFrame):
        """

        """
        # 1. Список полей с данными продаж
        fields = self._get_fields(date_beg=self.date_source_beg, date_end=self.date_source_end)
        df_filter = df_xls[fields]
        ls_rec_dict = df_filter.to_dict('records')

        # 2. Интерполяция данных

        # 3. Значения продаж для последних 12 месяцев
        ls_rec = [ [val for key, val in rec.items()] for rec in ls_rec_dict]
        return ls_rec

    def _get_num_month(self, date_end: str) -> int:
        """ Номер месяца для предсказания -- последняя дата окна + 1 мес.

        :param date_end: Дата конца окна
        :return: Номер месяца для предсказания
        """
        num_month = date_end.split('-')[1]
        return int(num_month) + 1

    def _get_fields(self, date_beg: str, date_end: str) -> List[str]:
        """ Генерация списка дат для выборки окна значений (последние 12 мес.)

        :param date_beg: Дата начала окна
        :param date_end: Дата конца окна
        :return: Список дат
        """
        # 1 Проверка кол-ва месяцев между датами
        d0 = datetime.strptime(date_beg, "%Y-%m-%d").date()
        d1 = datetime.strptime(date_end, "%Y-%m-%d").date()
        # Кол-во месяцев между (год, месяц) + 1 (включительно)
        months = (d1.year - d0.year) * 12 + (d1.month - d0.month) + 1

        if months != 12:
            raise ValueError(
                f"Период должен составлять ровно 12 календарных месяцев, "
                f"получено: {months} "
                f"(date_beg={date_beg}, date_end={date_end})"
            )

        # 2. Генерация списка дат
        d2 = d0
        ls_dates = []
        ls_dates.append(d0.strftime("%Y-%m-%d"))
        while d1 != d2:
            d2 += relativedelta(months=1)
            ls_dates.append(d2.strftime("%Y-%m-%d"))

        return ls_dates

    def predict(self, num_rec: int):
        """
        !!! исправить
        """
        data_window = self.ls_rec[num_rec]
        features = self.ft.compute_window(data_window, num_month=self.num_month)
        X = np.ndarray(features)
        y_pred = self.model.predict(X)
        return y_pred


if __name__ == '__main__':
    config = {
            'model_file': 'ipynb/lgb_model_regress.txt',
            'date_source_beg': '2024-08-01',
            'date_source_end': '2025-07-01',
            'file_source': 'ipynb/temp_8_9_10_итог_с_июлем.xlsx'
            }
    predict_sale_obj = PredictSale(config)
    predict_sale_obj.predict(num_rec=1)
