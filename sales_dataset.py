import random
import re
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from dateutil.relativedelta import relativedelta

from window_features import FeatureExtractor


@dataclass(slots=True)
class LastWindow:
    """
    window_sale -- Продажи за последние 12 мес
                   [16.0, 23.0, 17.0, 27.0, 11.0, 16.0, 19.0, 21.0, ...]
    month_predict -- Номер месяца для которого делается прогноз
    """
    window_sale: list[int]
    month_predict: int

@dataclass(slots=True)
class SplitDataSet:
    """ Класс DataSet продаж аптек

    """
    X_train: np.ndarray = None
    X_test: np.ndarray = None
    y_train: np.ndarray = None
    y_test: np.ndarray = None


class SalesDataset:
    """
    Класс создания датасета для обучения модели прогноза динамики продаж.
    Преобразует временные ряды продаж в формат признаков (X) и целевой переменной (y).

    Варианты выходных данных:
    --------------------
    1. self.all_windows_to_predict -- Продажи в разрезе аптек с разбивкой на окна по 12 месяцев
        [{'mdlp_id': 127357, 'num_window': 25,
         'window': [('2023-01-01', 7.0),
           ('2023-02-01', 2.0), ('2023-03-01', 3.0), ('2023-04-01', 8.0),
           ('2023-05-01', 3.0), ('2023-06-01', 15.0), ('2023-07-01', 5.0),
           ('2023-08-01', 2.0), ('2023-09-01', 11.0), ('2023-10-01', 22.0),
           ('2023-11-01', 7.0), ('2023-12-01', 9.0)]}, ...]

    2. self.get_last_window(mdlp_id=127357) -- Продажи конкретной аптеки с разбивкой на окна по 12 месяцев
       Структура аналогична self.all_windows_to_predict

    3. self.load_data_regress() -- Возвращает данные в формате (X, y) для задачи регрессии
    4. self.load_data_class() -- Возвращает данные в формате (X, y) для задачи бинарной классификации
    """

    def __init__(self, sales_df: pd.DataFrame, start_date: str = None):
        self.sale_dataset = SplitDataSet()

        cn_mes = self.get_date_column(sales_df)
        if not start_date:
            self.start_date = self.get_start_date(sales_df)
        else:
            self.start_date = start_date

        self.sales_df = sales_df.copy()
        self.ls_mdlp_id = self.sales_df['location_mdlp_id'].to_list()
        self.ft = FeatureExtractor()
        # Окна для обучения
        print("Start windows dataset")
        self.all_windows_to_train = self._prepare_data_train(sales_df, self.start_date, cn_mes)
        self.split_windows_dataset(self.all_windows_to_train)
        # Окна для прогноза
        self.all_windows_to_predict = self._prepare_data_predict(sales_df, self.start_date, cn_mes)


    def get_start_date(self, df: pd.DataFrame):
        """ Наименование столбца с минимальной датой

        """
        pat = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        date_cols = [c for c in df.columns if pat.fullmatch(c)]

        if date_cols:
            dts = pd.to_datetime(date_cols, format='%Y-%m-%d', errors='coerce')
            min_col = dts.min().strftime('%Y-%m-%d')
        else:
            min_col = None

        return min_col

    def get_date_column(self, df: pd.DataFrame):
        """ Кол-во столбцов в формате 2023-01-20

        """
        pat = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        date_cols = [c for c in df.columns if pat.fullmatch(c)]
        count_date_cols = len(date_cols)
        return count_date_cols

    def _prepare_data_predict(self, sales_df: pd.DataFrame, start_date: str, cn_mes: int) -> list[dict]:
        """ Преобразует DataFrame в список окон. Для одной аптеки формируется несколько окон.
            !!! Подготовка данных для прогноза

        :param sales_df: Исходные данные
        :param start_date: Дата начала выборки
        :param cn_mes: Размер выборки в месяцах

        :result: Список окон
                [{'mdlp_id': 166, 'num_window': 1, 'window': [('месяц_продаж', 'объём_продаж'), ..]}, ...]
                mdlp_id -- id аптеки
                num_window -- номер окна
                window -- окно с данными о 12-ти месяцев продаж
        """
        date_0 = pd.to_datetime(start_date).date()
        ls_date = [(date_0 + relativedelta(months=num_mes)).strftime('%Y-%m-%d') for num_mes in range(cn_mes)]
        ls_date.append('location_mdlp_id')

        # DataFrame --> [[('2024-01-01', 18.0), ('2024-02-01', 7.0), ('2024-03-01', 14.0), ('2024-04-01', 15.0) ...]]
        df_stable = sales_df.copy()[ls_date]
        filtered_df = df_stable.fillna(0)
        ls_rec_dict = filtered_df.to_dict('records')
        ls_rec_tuples = [list(rec.items()) for rec in ls_rec_dict]

        # [('месяц_продаж', 'объём_продаж'), ..] --> {'target_val': ('2025-01-01', 21.0), 'window': [('месяц_продаж', 'объём_продаж'), ..]}
        all_windows = []
        for prod_aptek in ls_rec_tuples:
            ls_windows = self._sliding_windows_prd(prod_aptek, window=12, step=1)
            all_windows.extend(ls_windows)

        return all_windows

    def _prepare_data_train(self, sales_df: pd.DataFrame, start_date: str, cn_mes: int) -> list[dict]:
        """ Преобразует DataFrame в список окон
            [{'num_window': 1, 'target_val': ('2025-01-01', 21.0), 'window': [('месяц_продаж', 'объём_продаж'), ..]},.]
            !!! Подготовка данных для обучения

        :param sales_df: Исходные данные
        :param start_date: Дата начала выборки
        :param cn_mes: Размер выборки в месяцах

        :result: Список окон
                [{'num_window': 1, 'target_val': ('2025-01-01', 21.0), 'window': [('месяц_продаж', 'объём_продаж'), ..]}, ...]
                num_window -- номер окна
                target_val -- значение целевой переменной
                window -- окно с данными о 12-ти месяцев продаж
        """
        date_0 = pd.to_datetime(start_date).date()
        ls_date = [(date_0 + relativedelta(months=num_mes)).strftime('%Y-%m-%d') for num_mes in range(cn_mes)]

        # DataFrame --> [[('2024-01-01', 18.0), ('2024-02-01', 7.0), ('2024-03-01', 14.0), ('2024-04-01', 15.0) ...]]
        df_stable = sales_df.copy()[ls_date]
        filtered_df = df_stable.fillna(0)
        ls_rec_dict = filtered_df.to_dict('records')
        ls_rec_tuples = [list(rec.items()) for rec in ls_rec_dict]

        # [('месяц_продаж', 'объём_продаж'), ..] --> {'target_val': ('2025-01-01', 21.0), 'window(12)': [('месяц_продаж', 'объём_продаж'), ..]}
        all_windows = []
        for prod_aptek in ls_rec_tuples:
            ls_windows = self._sliding_windows(prod_aptek, window=12, step=1)
            all_windows.extend(ls_windows)

        return all_windows

    def _sliding_windows(self, data: List, window: int, step: int = 1) -> List[Dict]:
        """ Деление списка [('месяц_продаж', 'объём_продаж'), ..] на список окон.
            Каждая запись исходного списка -- продажи одной аптеки за N месяцев
            !!! Подготовка данных для обучения

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
        num_window = 1
        while len(ls_window):
            one_window = ls_window.pop()
            dc = {'window': one_window, 'target_val': last_val, 'num_window': num_window}
            last_val = one_window[-1]
            num_window += 1
            result.append(dc)

        return result

    def _sliding_windows_prd(self, data: List, window: int, step: int = 1) -> List[Dict]:
        """ Деление списка [('месяц_продаж', 'объём_продаж'), ..] на список окон.
            Каждая запись исходного списка -- продажи одной аптеки за N месяцев
            !!! Подготовка данных для прогноза

        :param data: Исходный список
        :param window: Размер окна
        :param step: Шаг сдвига окна
        :param location_mdlp_id: id аптеки

        :return: Список окон.
        """
        mdlp_id = int(data.pop()[1])

        ls_window = []
        n = len(data)
        for i in range(0, n, step):
            w = data[i:i + window]
            if len(w) < window:
                break
            ls_window.append(w)

        # Заполнение "кол-ва продаж"
        result = []
        num_window = 1
        while len(ls_window):
            one_window = ls_window.pop()
            dc = {'window': one_window, 'mdlp_id': mdlp_id, 'num_window': num_window}
            num_window += 1
            result.append(dc)

        return result

    @property
    def data_class(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Возвращает данные в формате (X, y) для задачи бинарной классификации

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Кортеж с данными (data, target)
        """
        return self.data, self.target_class

    @property
    def data_regress(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Возвращает данные в формате (X, y) для задачи регрессии

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Кортеж с данными (data, target)
        """
        return self.data, self.target_regress

    def split_windows_dataset(self, all_windows_to_train: list[dict], test_size=0.2, random_state=42):
        """ 1. Разбивка окон данных на train и test
            2. Формирование: X_train, y_train, X_test, y_test

        :param all_windows_to_train:
                 [{'num_window': 1, 'target_val': ('2025-01-01', 21.0),
                  'window': [('месяц_продаж', 'объём_продаж'), ..]},.]
        :return: X_train, y_train, X_test, y_test
        """
        random.seed(random_state)
        cn_all = len(all_windows_to_train)
        cn_test = round(cn_all * test_size)

        data_sorted = sorted(all_windows_to_train, key=lambda x: x["num_window"])
        windows_test = data_sorted[:cn_test]
        windows_train = data_sorted[cn_test:]
        random.shuffle(windows_test)
        random.shuffle(windows_train)

        X_test, y_test = self._build_features(windows_test)
        X_train, y_train = self._build_features(windows_train)

        self.split_dataset = SplitDataSet(X_train=X_train,
                                         X_test=X_test,
                                         y_train=y_train,
                                         y_test=y_test)

    def get_split_dataset(self)-> SplitDataSet:
        """

        """
        print("Split dataset")
        return self.split_dataset

    def _build_features(self, windows_split: list[dict]):
        """ Построитель данных (X, y) для задач регрессии и классификации

        target_class <--
            1 -- продажи упали
            2 -- продажи выросли
            0 -- продажи не изменились

        :param all_windows: {'num_window': 1, 'target_val': ('2025-07-01', 20.0),
                                    'window': [('2024-07-01', 23.0), ('2024-08-01', 25.0), ..]}
        """
        X_data = []
        y_data_class = []
        y_data_regress = []

        for dc_window in windows_split:
            data_windows = [w[1] for w in dc_window['window']]
            predict_date = dc_window['target_val'][0]
            num_month = int(predict_date.split('-')[1])
            ft_features = self.ft.compute_window(data_windows, num_month=num_month)
            features = ft_features.to_list()

            last_val = dc_window['window'][-1][1]
            target_val = dc_window['target_val'][1]

            if target_val < last_val:
                target_class = 1  # продажи упали
            elif target_val > last_val:
                target_class = 2  # продажи выросли
            else:
                target_class = 0  # продажи не изменились

            X_data.append(features)
            y_data_class.append(target_class)
            y_data_regress.append(target_val)

        self.data = np.array(X_data)
        # self.target_class = np.array(y_data_class)
        self.target_regress = np.array(y_data_regress)
        return  self.data, self.target_regress

    def get_last_window(self, mdlp_id: int) -> LastWindow:
        """ Возвращает окно продаж последних 12-и месяцев для заданной mdlp

        :param mdlp_id: id аптеки
        :return: (window_sale=, month_predict=)
                window_sale -- Продажи за последние 12 мес
                               [16.0, 23.0, 17.0, 27.0, 11.0, 16.0, 19.0, 21.0, ...]
                month_predict -- Номер месяца для которого делается прогноз
        """
        df = pd.DataFrame(self.all_windows_to_predict)
        df = df[df['mdlp_id'] == mdlp_id]
        rec = df.sort_values("num_window", ascending=True).iloc[0].to_dict()

        window_sale = [i[1] for i in rec['window']]
        date_sale = [i[0] for i in rec['window']]
        date_predict = (pd.to_datetime(date_sale).max() + relativedelta(months=1)).strftime('%Y-%m-%d')
        month_predict = int(date_predict.split('-')[1])
        return LastWindow(window_sale=window_sale, month_predict=month_predict)


if __name__ == '__main__':
    from utils import pred_proc

    df_conv = pred_proc('etl/final_report_2026-01-27.csv')
    ds = SalesDataset(df_conv)
    # res = ds.get_split_dataset()
    res = ds.get_last_window(166)
    print(res)