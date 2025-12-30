import datetime

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from dateutil.relativedelta import relativedelta

from window_features import FeatureExtractor

class SalesDataset:
    """
    Класс для создания датасета динамики продаж.
    Преобразует временные ряды продаж в формат признаков (X) и целевой переменной (y).
    """

    def __init__(self, sales_df: pd.DataFrame, start_date: str, cn_mes: int):
        self.cn_mes = cn_mes
        self.start_date = start_date
        self.sales_df = sales_df.copy()
        # self._validate_data()
        self.ft = FeatureExtractor()
        self.all_windows = self._prepare_data()
        self._build_features(self.all_windows)

    def _prepare_data(self):
        date_0 = datetime.datetime.strptime(self.start_date, '%Y-%m-%d')
        ls_date = [(date_0 + relativedelta(months=num_mes)).strftime('%Y-%m-%d') for num_mes in range(self.cn_mes)]

        df_stable = self.sales_df[self.sales_df['class'] == 'Стабильная'][ls_date]
        filtered_df = df_stable.dropna(subset=ls_date)
        ls_rec_dict = filtered_df.to_dict('records')
        ls_rec_tuples = [list(rec.items()) for rec in ls_rec_dict]

        all_windows = []
        for prod_aptek in ls_rec_tuples:
            ls_windows = self._sliding_windows(prod_aptek, window=12, step=1)
            all_windows.extend(ls_windows)

        return all_windows

    def _sliding_windows(self, data: List, window: int, step: int = 1) -> List[List]:
        """ Деление списка [('месяц_продаж', 'объём_продаж'), ..] на список окон.
            Каждая запись исходного списка -- продажи одной аптеки за N месяцев

        :param data: Исходный список
        :param window: Размер окна
        :param step: Шаг сдвига окна
        :return: Список окон.
        """
        # значение последнего месяца
        last_val = data.pop()

        ls_window = []
        n = len(data)
        for i in range(0, n, step):
            w = data[i:i + window]
            if len(w) < window:
                break
            ls_window.append(w)

        # Заполнение "прогнозированное значение"
        result = []
        while len(ls_window):
            one_window = ls_window.pop()
            dc = {'window': one_window, 'last_val': last_val}
            last_val = one_window[-1]
            result.append(dc)

        return result

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Возвращает данные в формате (X, y).

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Кортеж с данными (data, target)
        """
        return self.data, self.target

    def _build_features(self, all_windows):
        # Подготовка данных в формате sklearn
        X_data = []
        y_data = []

        for dc_window in all_windows:
            data_windows = [w[1] for w in dc_window['window']]
            predict_date = dc_window['last_val'][0]
            num_month = int(predict_date.split('-')[1])
            ft_features = self.ft.compute_window(data_windows, num_month=num_month)
            features = ft_features.to_list()

            target =  dc_window['last_val'][1]
            X_data.append(features)
            y_data.append(target)

        self.data = np.array(X_data)
        self.target = np.array(y_data)


if __name__ == '__main__':
    df_xls = pd.read_excel('ipynb/temp_8_9_10_итог_с_июлем.xlsx')
    ds = SalesDataset(df_xls, start_date='2024-01-01', cn_mes=19)
    X, y = ds.load_data()


