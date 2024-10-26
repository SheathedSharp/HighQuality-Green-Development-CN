'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-26 13:27:45
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-26 14:02:00
'''
import os
import pandas as pd
from src.data_processing.data_preparation import data_prepare, add_future_three_years_data, split_data_by_province
from src.data_processing.missing_value_imputation import missing_values_Imputation_poly_regression
from src.models.lstm_models import dynamic_residuals_stack_LSTM_model, stack_LSTM_model, traditional_LSTM_model
from src.models.model_evaluation import evaluate_model_performance, save_model, prediction_future_three_years_HQ_score
from src.utils.visualization import scatter_plot, check_effect_of_poly_regression_Imputation
import config

def predict_future_score():
    # Second part: prediction
    ## Add 2024 to 2026 data to the table and initialize it to 0, save as future_three_years_data_0.xlsx
    add_future_three_years_data(config.PROCESSED_DATA_PATH, config.OUTPUT_DATA0_PATH)
    
    ## Use ARIMA to fill the 2024 to 2026 data of each attribute column, and save the filled file as future_three_years_data.xlsx
    # OUTPUT_DATA0 -> OUTPUT_DATA1


    ## Use the newly filled table (future_three_years_data.xlsx) after adding the corresponding model, and then predict the green high-quality development indicators for the next three years
    ## choose the existing model in the model folder
    model_list = os.listdir(config.MODEL_PATH)
    print("The existing models are: ", model_list)
    pre_model_url = input("Please choose the model: ")
    
    if pre_model_url in model_list:
        prediction_future_three_years_HQ_score(config.OUTPUT_DATA1_PATH, config.OUTPUT_DATA2_PATH, pre_model_url=config.MODEL_PATH + "/" + pre_model_url)
    else:
        print("The model you chose does not exist, please try again.")

if __name__ == "__main__":
    predict_future_score()