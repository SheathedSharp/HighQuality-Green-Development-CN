'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-25 15:43:56
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-26 11:21:22
'''

import pandas as pd
from src.data_processing.data_preparation import data_prepare, add_future_three_years_data, split_data_by_province
from src.data_processing.missing_value_imputation import missing_values_Imputation_poly_regression
from src.models.lstm_models import dynamic_residuals_stack_LSTM_model
from src.models.model_evaluation import evaluate_model_performance, save_model, prediction_future_three_years_HQ_score
from src.utils.visualization import scatter_plot, check_effect_of_poly_regression_Imputation
import config

def main():
    # First part: train model
    ## Data preparation
    X_train, y_train, X_test, y_test = data_prepare(config.PROCESSED_DATA_PATH)

    ## Model training
    model = dynamic_residuals_stack_LSTM_model(X_train, y_train, X_test, y_test)

    ## Model evaluation
    evaluate_model_performance(X_test, y_test, model=model)

    ## Save model
    save_model(model, config.BEST_MODEL_PATH)

    # Second part: prediction
    ## Add 2024 to 2026 data to the table and initialize it to 0, save as future_three_years_data_0.xlsx
    add_future_three_years_data(config.PROCESSED_DATA_PATH, config.OUTPUT_DATA0_PATH)
    
    ## Use ARIMA to fill the 2024 to 2026 data of each attribute column, and save the filled file as future_three_years_data.xlsx
    
    ## Use the newly filled table (future_three_years_data.xlsx) after adding the corresponding model, and then predict the green high-quality development indicators for the next three years
    prediction_future_three_years_HQ_score(config.OUTPUT_DATA1_PATH, config.OUTPUT_DATA2_PATH, pre_model_url=config.BEST_MODEL_PATH)

if __name__ == "__main__":
    main()