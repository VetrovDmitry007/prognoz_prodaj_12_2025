import pandas as pd
import numpy as np
import lightgbm as lgb

from sales_dataset import SalesDataset
from utils import pred_proc, create_dataset, evaluate_model


def train(train_data, valid_data):
    # Параметры модели
    params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.7,
        "metric": ["tweedie", "rmse", "mae"],
        'learning_rate': 0.01,
        'max_depth': 5,
        'num_leaves': 31,
        'verbosity': -1
    }

    # Обучение с ранней остановкой
    print("Start train model")
    model_early_stop = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=300),  # Если в течение 300 итераций подряд метрика не стала лучше
            lgb.log_evaluation(50)  # Выводим логи каждые 50 итераций
        ]
    )

    model_early_stop.save_model("lgb_model_regress_2.txt")
    print("Save model")
    return model_early_stop


def gain(model):
    """ Определение значимости признаков

    """
    gain = model.feature_importance(importance_type="gain")
    names = model.feature_name()

    fi = (
        pd.DataFrame({"feature": names, "gain": gain})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    print(fi)


def pipeline():
    df_conv = pred_proc('etl/final_report_2026-01-27.csv')
    sale_ds = SalesDataset(df_conv)

    split_dataset = sale_ds.get_split_dataset()
    train_data, valid_data = create_dataset(split_dataset)
    model = train(train_data, valid_data)

    y_pred = model.predict(split_dataset.X_test)
    evaluate_model(split_dataset.y_test, y_pred, "LightGBM с ранней остановкой")

    gain(model)


if __name__ == '__main__':
    pipeline()
