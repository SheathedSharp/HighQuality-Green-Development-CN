'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-25 15:43:56
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-25 23:18:14
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
    # 第一部分：训练模型
    ## 数据准备
    X_train, y_train, X_test, y_test = data_prepare(config.PROCESSED_DATA_PATH)

    ## 模型训练
    model = dynamic_residuals_stack_LSTM_model(X_train, y_train, X_test, y_test)

    ## 模型评估
    evaluate_model_performance(X_test, y_test, model=model)

    ## 保存模型
    save_model(model, config.BEST_MODEL_PATH)

    # 第二部分：进行预测
    ## 往表中新增的2024到2026年的数据，并且初始化为0，保存为future_three_years_data_0.xlsx
    add_future_three_years_data(config.PROCESSED_DATA_PATH, config.OUTPUT_DATA0_PATH)
    
    ## 使用ARIMA进行2024到2026年的各个属性列的填充，填充后的文件保存为future_three_years_data.xlsx
    
    ## 利用填充好的后三年数据新表（future_three_years_data.xlsx），加载相应的模型后进行后三年的绿色高质量发展指标的预测
    prediction_future_three_years_HQ_score(config.OUTPUT_DATA1_PATH, config.OUTPUT_DATA2_PATH, pre_model_url=config.BEST_MODEL_PATH)

if __name__ == "__main__":
    main()