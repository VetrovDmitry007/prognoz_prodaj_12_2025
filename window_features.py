"""
Создание фичей на основе данных продаж окна 12 месяцев.

Исходные данные:
1. Список (окно) из продаж 12 месяцев
2. Номер месяца для которого делается прогноз

Выходные данные:
1. Набор фичей для модели регрессии
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Sequence, List, Dict, Optional

import numpy as np
from scipy.stats import linregress


@dataclass(frozen=True)
class Features:
    # 15 фичей
    MeanW: float
    StdW: float
    CV: float
    RelLast: float
    ZLast: float

    Delta1: float
    LogRet1: float
    Mom3: float
    Mom6: float

    Slope10: float
    R2_Trend: float
    ZResLast: float

    UpShare: float
    SignChanges: float
    MaxDrawdown: float

    MonthCos: float
    MonthSin: float
    Lag12Target: float

    @classmethod
    def names(cls) -> List[str]:
        # Важно: стабильный порядок для ML
        return [
            "MeanW", "StdW", "CV", "RelLast", "ZLast",
            "Delta1", "LogRet1", "Mom3", "Mom6",
            "Slope10", "R2_Trend", "ZResLast",
            "UpShare", "SignChanges", "MaxDrawdown",
            "MonthCos", "MonthSin", "Lag12Target"
        ]

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def to_list(self) -> List[float]:
        d = self.to_dict()
        return [d[name] for name in self.names()]

class FeatureExtractor:
    """
    Извлекает из документа 15 функций временных рядов на основе окон:
    MeanW, StdW, CV, RelLast, ZLast, Delta1, LogRet1, Mom3, Mom6,
    Slope10, R2_Trend, ZResLast, UpShare, SignChanges, MaxDrawdown
    и
    MonthSin и MonthCos -- месяца целевой точки
    lag12_target -- сколько обычно продаётся в прогнозируемый месяц
    """
    def __init__(self, window_size: int = 12, epsilon: float = 1e-6):
        self.window_size = window_size
        self.epsilon = epsilon
        self._reset()

    def _reset(self) -> None:
        # Все поля в None, чтобы легко ловить забытые вычисления
        self.meanw = None
        self.stdw = None
        self.cv = None
        self.rel_last = None
        self.z_last = None
        self.delta1 = None
        self.logret1 = None
        self.mom3 = None
        self.mom6 = None
        self.slope10 = None
        self.r2_trend = None
        self.z_res_last = None
        self.up_share = None
        self.sign_changes = None
        self.max_drawdown = None
        self.month_sin = None
        self.month_cos = None

    def compute_window(self, w: List[float], num_month: int) -> Features:
        """ Вычисление фичей для окна значений.

        :param w: Окно прошлых значений, для которого вычисляются фичи.
                  [16.0, 23.0, 17.0, 27.0, 11.0, 16.0, 19.0, 21.0, ...]
        :param num_month: Номер месяца для которого делается прогноз. -- 11
        :return: Экземпляр класса Features
        """

        if len(w) != self.window_size:
            raise ValueError(f"Expected {self.window_size}, got {len(w)}")

        self._reset()
        self._w = w
        self._calc_mean_std_cv()
        self._calc_last_stats()
        self._calc_returns_momentum()
        self._calc_trend()
        self._calc_structure()
        self._calc_drawdown()
        self.month_sin_cos(num_month)
        self.seasona_lag12_target(w)

        return self.to_features()

    def _calc_mean_std_cv(self) -> None:
        """ Вычисление:
            1. meanw - средний уровень окна
            2. stdw - стандартное отклонение окна
            3. cv - относительная волатильность
        """
        x = self._w
        eps = self.epsilon
        self.meanw = float(np.mean(x))
        self.stdw = float(np.std(x, ddof=0))  # population std
        self.cv = float(self.stdw / (abs(self.meanw) + eps))

    def _calc_last_stats(self) -> None:
        """ Вычисление:
            1. rel_last - относительный уровень последнего значения к среднему
            2. z-score - последнего значения внутри окна
        """
        x = self._w
        eps = self.epsilon
        last = float(x[-1])
        # RelLast: относительный уровень последнего значения к среднему
        self.rel_last = float(last / (self.meanw + eps))
        # ZLast: z-score последнего значения внутри окна
        self.z_last = float((last - self.meanw) / (self.stdw + eps))

    def _calc_returns_momentum(self) -> None:
        """ Вычисление:
            1. logret1 - Лог-доходность (стабилизация eps)
            2. mom3 - импульс за 3 месяца (текущее - значение 3 месяца назад)
            3. mom6 - импульс за 6 месяцев => индекс -1 и -7
        """
        x = self._w
        eps = self.epsilon

        last = float(x[-1])
        prev = float(x[-2])
        self.delta1 = float(last - prev)
        # Лог-доходность (стабилизация eps)
        self.logret1 = float(np.log(last + eps) - np.log(prev + eps))
        # Mom3: импульс за 3 месяца (текущее - значение 3 месяца назад)
        # при месячных данных: t - (t-3) => индекс -1 и -4
        self.mom3 = float(last - float(x[-4]))
        # Mom6: импульс за 6 месяцев => индекс -1 и -7
        self.mom6 = float(last - float(x[-7]))

    def _calc_structure(self) -> None:
        """ Вычисление:
            1. up_share - доля положительных изменений (ростов)
            2. sign_changes - сколько раз меняется знак приращений (игнорируем нули)
        """
        x = self._w
        diffs = np.diff(x)
        if diffs.size == 0:
            self.up_share = 0.0
            self.sign_changes = 0.0
            return
        # UpShare: доля положительных изменений (ростов)
        self.up_share = float(np.mean(diffs > 0))
        # SignChanges: сколько раз меняется знак приращений (игнорируем нули)
        s = np.sign(diffs)
        s = s[s != 0]  # выкидываем нулевые приращения, чтобы не “ломали” смены знака
        if s.size <= 1:
            self.sign_changes = 0.0
        else:
            self.sign_changes = float(np.sum(s[1:] != s[:-1]))

    def _calc_drawdown(self) -> None:
        """ Вычисление:
            1. max_drawdown -максимальная просадка от исторического пика (в долях)
        """
        x = self._w
        eps = self.epsilon
        # MaxDrawdown: максимальная просадка от исторического пика (в долях)
        peaks = np.maximum.accumulate(x)
        drawdowns = (peaks - x) / (peaks + eps)  # 0..+
        self.max_drawdown = float(np.max(drawdowns))

    def month_sin_cos(self, num_month: int):
        """
        Кодирует номер месяца, **для которого делается прогноз** (месяц целевой точки),
        в два циклических признака: MonthSin и MonthCos.
        Позволяет учитывать годовую сезонность (полный цикл)

        :param num_month: Номер месяца для которого делается прогноз
        """
        ang = 2 * np.pi * (num_month / 12.0)
        self.month_sin = float(np.sin(ang))
        self.month_cos= float(np.cos(ang))

    def seasona_lag12_target(self,  w: Sequence[float]):
        """ Вычисляет, сколько обычно продаётся в прогнозируемый месяц.
        !!! Главное условие -- длина окна (w) 12 месяцев.

        :param w: Окно прошлых значений, для которого вычисляются фичи.
        :return: Сколько обычно продаётся в прогнозируемый месяц.
        """
        self.lag12_target = w[0]

    def _calc_trend(self) -> None:
        """ Вычисление:
            1. slope10 - наклон тренда, slope > 0 → тренд вверх (растёт)
            2. r2_trend - сила тренда
            3. z_res_last - стандартизированный остаток в последней точке


        :return:
        """
        x = self._w
        eps = self.epsilon
        n = len(x)

        if n < 2:
            self.slope10 = 0.0
            self.r2_trend = 0.0
            self.z_res_last = 0.0
            return

        # Если ряд константный — тренда нет, остатки нулевые
        if np.allclose(x, x[0]):
            self.slope10 = 0.0
            self.r2_trend = 0.0
            self.z_res_last = 0.0
            return

        # Ось времени: 1..W (как и раньше)
        t = np.arange(1, n + 1, dtype=float)

        slope, intercept, r_value, p_value, std_err = linregress(t, x)  # slope, intercept, rvalue, pvalue, stderr, intercept_stderr
        slope = float(slope)
        self.slope10 = slope
        intercept = float(intercept)

        # Построение линейного тренда и вычисление ошибок аппроксимации (residuals)
        y_hat = intercept + slope * t
        residuals = x - y_hat

        # R2_Trend = r^2 сила тренда
        rvalue = 0.0 if np.isnan(r_value) else float(r_value)
        self.r2_trend = rvalue * rvalue

        # ZResLast: стандартизированный остаток в последней точке
        res_std = float(np.std(residuals, ddof=0))
        self.z_res_last = float(residuals[-1] / (res_std + eps))

    def to_features(self) -> Features:
        """ Проверка, что всё посчитано.

        :return:
        """
        required = [
            self.meanw, self.stdw, self.cv, self.rel_last, self.z_last,
            self.delta1, self.logret1, self.mom3, self.mom6,
            self.slope10, self.r2_trend, self.z_res_last,
            self.up_share, self.sign_changes, self.max_drawdown,
            self.month_sin, self.month_cos, self.lag12_target
        ]
        if any(v is None for v in required):
            raise RuntimeError("Not all features computed")

        return Features(
            MeanW=self.meanw,
            StdW=self.stdw,
            CV=self.cv,
            RelLast=self.rel_last,
            ZLast=self.z_last,
            Delta1=self.delta1,
            LogRet1=self.logret1,
            Mom3=self.mom3,
            Mom6=self.mom6,
            Slope10=self.slope10,
            R2_Trend=self.r2_trend,
            ZResLast=self.z_res_last,
            UpShare=self.up_share,
            SignChanges=self.sign_changes,
            MaxDrawdown=self.max_drawdown,
            MonthSin = self.month_sin,
            MonthCos = self.month_cos,
            Lag12Target= self.lag12_target
        )


if __name__ == '__main__':
    data_windows = [16.0, 23.0, 17.0, 27.0, 11.0, 16.0, 19.0, 21.0, 21.0, 16.0, 24.0, 10.0]
    ft = FeatureExtractor()
    res= ft.compute_window(data_windows, num_month=11)
    print(res)

