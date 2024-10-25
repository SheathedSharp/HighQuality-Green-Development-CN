'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-25 15:43:56
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-25 15:55:40
FilePath: /2024_tjjm/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
from src.data_processing.data_preparation import data_prepare, add_future_three_years_data, split_data_by_province
from src.data_processing.missing_value_imputation import missing_values_Imputation_poly_regression
from src.models.lstm_models import dynamic_residuals_stack_LSTM_model
from src.models.model_evaluation import evaluate_model_performance, save_model, prediction_future_three_years_HQ_score
from src.utils.visualization import scatter_plot, check_effect_of_poly_regression_Imputation
import config

def main():
    # 数据准备
    X_train, y_train, X_test, y_test = data_prepare(config.PROCESSED_DATA_PATH)

    # 模型训练
    model = dynamic_residuals_stack_LSTM_model(X_train, y_train, X_test, y_test)

    # 模型评估
    evaluate_model_performance(X_test, y_test, model=model)

    # 保存模型
    save_model(model, config.BEST_MODEL_PATH)

    # 预测未来三年
    add_future_three_years_data(config.PROCESSED_DATA_PATH, config.OUTPUT_DATA_PATH)
    prediction_future_three_years_HQ_score(config.OUTPUT_DATA_PATH, config.OUTPUT_DATA_PATH, pre_model_url=config.BEST_MODEL_PATH)

if __name__ == "__main__":
    main()